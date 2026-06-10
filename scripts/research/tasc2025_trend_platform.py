#!/usr/bin/env python
"""TASC 2025 long-convex trend platform component tests.

This runner is deliberately research-scoped. It tests the 2025 TASC ideas as
components on three sleeves before anything is promoted into shared production
modules:

  1. BTC-USDC daily trend.
  2. CL institutional continuous futures at 60m/4h/1d.
  3. Sector ETF daily rotation.

Outputs:
  artifacts/research/tasc2025_trend_platform/
    metrics.csv
    equity_curves/*.csv
    README.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research" / "common"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from alpha_utils import calc_adx  # noqa: E402
from common.tasc2025_indicators import (  # noqa: E402
    autocorr_regime_score,
    continuation_index,
    drawdown_duration,
    linear_regression_channel,
    supertrend,
    ulcer_index,
)
from etf_data.universe import US_EQUITY_SECTORS  # noqa: E402
from btc_regime_switching_ma import (  # noqa: E402
    StrategySpec as BtcRegimeSpec,
    run_strategy as run_btc_regime_strategy,
)

DEFAULT_CRYPTO_LAKE = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb"
DEFAULT_ETF_DB = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/etf_market.duckdb"
DEFAULT_CL_DIR = ROOT / "artifacts" / "research" / "cl_institutional_continuous"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "research" / "tasc2025_trend_platform"
ANN_DAILY = 365.0
TAIL_WINDOWS: dict[str, tuple[str, str]] = {
    "covid_crash": ("2020-02-15", "2020-04-15"),
    "crypto_2021_22": ("2021-11-01", "2022-06-30"),
    "rate_hike_2022": ("2022-01-01", "2022-12-31"),
    "ftx": ("2022-11-01", "2022-12-31"),
    "recent_futures_window": ("2025-07-01", "2026-05-29"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TASC 2025 trend platform component tests.")
    parser.add_argument("--crypto_lake", default=DEFAULT_CRYPTO_LAKE)
    parser.add_argument("--etf_db", default=DEFAULT_ETF_DB)
    parser.add_argument("--cl_dir", default=str(DEFAULT_CL_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--costs_bps", default="0,5,30")
    return parser.parse_args()


def load_btc_daily(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM bars_1d_clean
            WHERE symbol = 'BTC-USDC'
              AND ts >= TIMESTAMPTZ '2018-01-01'
            ORDER BY ts
            """
        ).df()
    finally:
        con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    return _normalize_ohlcv(df)


def load_cl_bars(cl_dir: Path, timeframe: str) -> pd.DataFrame | None:
    path = cl_dir / "bars_1m.parquet"
    if not path.exists():
        return None
    bars = pd.read_parquet(path)
    bars["ts"] = pd.to_datetime(bars["ts"])
    if "front_month_valid" in bars.columns:
        bars = bars[bars["front_month_valid"]].copy()
    bars = bars.sort_values("ts")
    rule = {"60m": "60min", "4h": "4h", "1d": "1D"}[timeframe]
    out = (
        bars.set_index("ts")
        .resample(rule)
        .agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
        .dropna(subset=["o", "c"])
        .reset_index()
    )
    out = out.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return _normalize_ohlcv(out)


def load_sector_etfs(db_path: str) -> dict[str, pd.DataFrame]:
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            """
            SELECT symbol, ts, open, high, low, close, volume
            FROM bars_1d
            WHERE symbol IN ({})
            ORDER BY symbol, ts
            """.format(",".join(f"'{s}'" for s in US_EQUITY_SECTORS))
        ).df()
    finally:
        con.close()
    df["ts"] = pd.to_datetime(df["ts"])
    panels: dict[str, pd.DataFrame] = {}
    for sym, sub in df.groupby("symbol"):
        panels[str(sym)] = _normalize_ohlcv(sub.drop(columns=["symbol"]))
    return panels


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={c: c.lower() for c in df.columns}).copy()
    out = out[["ts", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "close"])
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def ma_signal(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_ma = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ma = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    sig = np.sign(fast_ma - slow_ma).replace(0.0, np.nan).ffill().fillna(0.0)
    return sig.rename("ma_signal")


def tasc_signals(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"].astype(float)
    returns = close.pct_change(fill_method=None)
    df = bars.copy()
    df["ma_20_50"] = ma_signal(close, 20, 50)
    ac = autocorr_regime_score(returns, window=126, min_lag=2, max_lag=20, threshold=0.10)
    df = pd.concat([df, ac], axis=1)
    df["vol_20"] = returns.rolling(20, min_periods=20).std()
    df["vol_pct"] = df["vol_20"].rolling(252, min_periods=80).rank(pct=True)
    df["progressive_mult"] = (
        0.75 * df["ac_score"].fillna(0.0)
        + 0.35 * (df["ma_20_50"].abs() > 0).astype(float)
        - 0.35 * df["vol_pct"].fillna(0.5)
    ).clip(0.0, 1.0)
    ci = continuation_index(close, gamma=0.8, length=20)
    df["ci_state"] = ci
    st = supertrend(
        bars.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}),
        atr_len=13,
        multiplier=3.0,
    )
    df = pd.concat([df, st], axis=1)
    ch = linear_regression_channel(close, length=40, width=2.0)
    df = pd.concat([df, ch], axis=1)
    try:
        df["adx"] = calc_adx(bars.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"}), n=14)
    except Exception:
        df["adx"] = np.nan
    df["channel_breakout"] = 0.0
    prior_upper = df["lr_upper"].shift(1)
    prior_lower = df["lr_lower"].shift(1)
    prior_slope = df["lr_slope"].shift(1)
    df.loc[
        (df["close"] > prior_upper) & (prior_slope > 0) & (df["adx"] > 20),
        "channel_breakout",
    ] = 1.0
    df.loc[
        (df["close"] < prior_lower) & (prior_slope < 0) & (df["adx"] > 20),
        "channel_breakout",
    ] = -1.0
    df["channel_breakout"] = df["channel_breakout"].replace(0.0, np.nan).ffill().fillna(0.0)
    return df


def build_variant_positions(features: pd.DataFrame, variant: str) -> pd.Series:
    ma = features["ma_20_50"].astype(float)
    if variant == "ma_20_50":
        return ma
    if variant == "ac_sized_ma":
        return ma * features["ac_score"].fillna(0.0).clip(0.0, 1.0)
    if variant == "progressive_overlay":
        return ma * features["progressive_mult"].fillna(0.0)
    if variant == "laguerre_ci":
        return features["ci_state"].astype(float)
    if variant == "ci_confirmed_ma":
        return ma.where(np.sign(ma) == np.sign(features["ci_state"]), 0.0)
    if variant == "channel_adx":
        return features["channel_breakout"].astype(float)
    if variant == "supertrend_state":
        return features["supertrend_dir"].astype(float)
    if variant == "ma_supertrend_stop_reentry":
        return _stop_reentry_position(features, ma)
    raise ValueError(f"Unknown variant: {variant}")


def _stop_reentry_position(features: pd.DataFrame, desired: pd.Series, hhll_window: int = 20) -> pd.Series:
    """Stateful SuperTrend stop with new-extreme re-entry."""
    high = features["high"].astype(float)
    low = features["low"].astype(float)
    close = features["close"].astype(float)
    st_dir = features["supertrend_dir"].astype(float)
    prior_hh = high.rolling(hhll_window, min_periods=hhll_window).max().shift(1)
    prior_ll = low.rolling(hhll_window, min_periods=hhll_window).min().shift(1)
    pos = []
    held = 0.0
    stopped_dir = 0.0
    for i in range(len(features)):
        want = float(desired.iloc[i]) if np.isfinite(desired.iloc[i]) else 0.0
        if held == 0.0:
            reenter_long = stopped_dir == 1.0 and high.iloc[i] > prior_hh.iloc[i]
            reenter_short = stopped_dir == -1.0 and low.iloc[i] < prior_ll.iloc[i]
            if stopped_dir == 0.0 or reenter_long or reenter_short or np.sign(want) != stopped_dir:
                held = want
                stopped_dir = 0.0
        else:
            stop_long = held > 0 and st_dir.iloc[i] < 0
            stop_short = held < 0 and st_dir.iloc[i] > 0
            exit_rule = want == 0 or np.sign(want) != np.sign(held)
            if stop_long or stop_short:
                stopped_dir = float(np.sign(held))
                held = 0.0
            elif exit_rule:
                held = want
                stopped_dir = 0.0
            else:
                held = want
        # Avoid holding stale positions through non-finite price rows.
        if not np.isfinite(close.iloc[i]):
            held = 0.0
        pos.append(held)
    return pd.Series(pos, index=features.index, name="ma_supertrend_stop_reentry")


def backtest_positions(
    bars: pd.DataFrame,
    raw_position: pd.Series,
    *,
    cost_bps: float,
    name: str,
    sleeve: str,
    bars_per_year: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = bars[["ts", "open", "close"]].copy()
    for col in ["ac_score", "vol_pct", "progressive_mult"]:
        if col in bars.columns:
            df[col] = bars[col].to_numpy()
    df["raw_position"] = raw_position.reindex(df.index).fillna(0.0).clip(-1.0, 1.0)
    df["position"] = df["raw_position"].shift(1).fillna(0.0)
    df["asset_ret"] = df["open"].shift(-1) / df["open"] - 1.0
    df["asset_ret"] = df["asset_ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())
    df["ret"] = df["position"] * df["asset_ret"] - df["turnover"] * cost_bps / 10_000.0
    df["nav"] = (1.0 + df["ret"]).cumprod()
    row = compute_metrics(df, sleeve=sleeve, variant=name, cost_bps=cost_bps, bars_per_year=bars_per_year)
    return df, row


def compute_metrics(
    equity: pd.DataFrame,
    *,
    sleeve: str,
    variant: str,
    cost_bps: float,
    bars_per_year: float,
) -> dict[str, object]:
    ret = equity["ret"].astype(float)
    nav = equity["nav"].astype(float)
    years = (pd.Timestamp(equity["ts"].iloc[-1]) - pd.Timestamp(equity["ts"].iloc[0])).total_seconds() / (365.25 * 24 * 3600)
    std = ret.std(ddof=0)
    max_dd = float((nav / nav.cummax() - 1.0).min())
    cagr = float(nav.iloc[-1] ** (1 / years) - 1.0) if years > 0 and nav.iloc[-1] > 0 else 0.0
    dd_dur = drawdown_duration(nav)
    return {
        "sleeve": sleeve,
        "variant": variant,
        "cost_bps": cost_bps,
        "start": str(equity["ts"].iloc[0]),
        "end": str(equity["ts"].iloc[-1]),
        "n_bars": int(len(equity)),
        "years": years,
        "total_return": float(nav.iloc[-1] - 1.0),
        "cagr": cagr,
        "ann_vol": float(std * math.sqrt(bars_per_year)) if std > 0 else 0.0,
        "sharpe": float(ret.mean() / std * math.sqrt(bars_per_year)) if std > 0 else 0.0,
        "max_dd": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
        "ulcer_index": ulcer_index(nav),
        **dd_dur,
        "avg_abs_exposure": float(equity["position"].abs().mean()),
        "turnover_sum": float(equity["turnover"].sum()),
    }


def write_tail_window_summary(out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted((out_dir / "equity_curves").glob("*.csv")):
        eq = pd.read_csv(path, parse_dates=["ts"])
        eq["ts"] = pd.to_datetime(eq["ts"], utc=True).dt.tz_localize(None)
        for name, (start, end) in TAIL_WINDOWS.items():
            mask = (eq["ts"] >= pd.Timestamp(start)) & (eq["ts"] <= pd.Timestamp(end))
            sub = eq.loc[mask]
            if sub.empty:
                continue
            rows.append(
                {
                    "equity_file": path.name,
                    "window": name,
                    "start": start,
                    "end": end,
                    "n_bars": int(len(sub)),
                    "strategy_return": float((1.0 + sub["ret"]).prod() - 1.0),
                    "asset_return": float((1.0 + sub["asset_ret"]).prod() - 1.0)
                    if "asset_ret" in sub
                    else np.nan,
                    "max_dd": float((sub["nav"] / sub["nav"].cummax() - 1.0).min()),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "tail_windows.csv", index=False)


def write_regime_summary(out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted((out_dir / "equity_curves").glob("*.csv")):
        eq = pd.read_csv(path, parse_dates=["ts"])
        eq["ts"] = pd.to_datetime(eq["ts"], utc=True).dt.tz_localize(None)
        if "ac_score" not in eq.columns or "vol_pct" not in eq.columns:
            continue
        for regime_col, threshold in [("ac_score", 0.20), ("vol_pct", 0.80)]:
            valid = eq.dropna(subset=[regime_col])
            if valid.empty:
                continue
            buckets = {
                f"{regime_col}_high": valid[valid[regime_col] >= threshold],
                f"{regime_col}_low": valid[valid[regime_col] < threshold],
            }
            for bucket, sub in buckets.items():
                if sub.empty:
                    continue
                ret = sub["ret"].astype(float)
                std = ret.std(ddof=0)
                rows.append(
                    {
                        "equity_file": path.name,
                        "regime": bucket,
                        "n_bars": int(len(sub)),
                        "mean_ret": float(ret.mean()),
                        "sharpe_observed": float(ret.mean() / std * math.sqrt(len(sub))) if std > 0 else 0.0,
                        "window_return": float((1.0 + ret).prod() - 1.0),
                        "avg_exposure": float(sub["position"].abs().mean()),
                    }
                )
    pd.DataFrame(rows).to_csv(out_dir / "regime_summary.csv", index=False)


def run_single_asset_suite(
    bars: pd.DataFrame,
    *,
    sleeve: str,
    costs: Iterable[float],
    bars_per_year: float,
    out_equity: Path,
) -> list[dict[str, object]]:
    features = tasc_signals(bars)
    variants = [
        "ma_20_50",
        "ac_sized_ma",
        "progressive_overlay",
        "laguerre_ci",
        "ci_confirmed_ma",
        "channel_adx",
        "supertrend_state",
        "ma_supertrend_stop_reentry",
    ]
    rows = []
    for variant in variants:
        raw_pos = build_variant_positions(features, variant)
        for cost in costs:
            equity, row = backtest_positions(
                features,
                raw_pos,
                cost_bps=cost,
                name=variant,
                sleeve=sleeve,
                bars_per_year=bars_per_year,
            )
            rows.append(row)
            equity.to_csv(out_equity / f"{sleeve}_{variant}_cost{int(cost)}.csv", index=False)
    return rows


def run_btc_regime_baselines(
    bars: pd.DataFrame,
    *,
    costs: Iterable[float],
    out_equity: Path,
) -> list[dict[str, object]]:
    """Add prior ER/HMM BTC baselines to the same comparison grid."""
    indexed = bars.set_index("ts").copy()
    specs = [
        BtcRegimeSpec(
            name="rbs_20_50_er_p60_vol_cap2",
            fast=20,
            slow=50,
            gate="er_vol",
            er_quantile=0.60,
            vol_floor=0.0,
            vol_ceiling=2.0,
        ),
        BtcRegimeSpec(
            name="rbs_20_50_hmm_weekly_p55",
            fast=20,
            slow=50,
            gate="hmm3_weekly",
            hmm_prob_entry=0.55,
            hmm_align_with_signal=True,
        ),
        BtcRegimeSpec(
            name="rbs_20_50_hmm_sticky_p60_p45",
            fast=20,
            slow=50,
            gate="hmm3_weekly_sticky",
            hmm_prob_entry=0.60,
            hmm_prob_exit=0.45,
            hmm_align_with_signal=True,
        ),
    ]
    rows: list[dict[str, object]] = []
    for spec in specs:
        base = run_btc_regime_strategy(indexed, spec, cost_bps=0.0)
        for cost in costs:
            equity = pd.DataFrame(
                {
                    "ts": base.index,
                    "open": base["open"].to_numpy(),
                    "close": base["close"].to_numpy(),
                    "asset_ret": base["gross_ret"].fillna(0.0).to_numpy(),
                    "position": base["position"].fillna(0.0).to_numpy(),
                    "turnover": base["turnover"].fillna(0.0).to_numpy(),
                }
            )
            equity["ret"] = equity["position"] * equity["asset_ret"] - equity["turnover"] * cost / 10_000.0
            equity["nav"] = (1.0 + equity["ret"].fillna(0.0)).cumprod()
            row = compute_metrics(
                equity,
                sleeve="btc_daily",
                variant=spec.name,
                cost_bps=cost,
                bars_per_year=ANN_DAILY,
            )
            rows.append(row)
            equity.to_csv(out_equity / f"btc_daily_{spec.name}_cost{int(cost)}.csv", index=False)
    return rows


def run_sector_rotation(
    panels: dict[str, pd.DataFrame],
    *,
    costs: Iterable[float],
    out_equity: Path,
) -> list[dict[str, object]]:
    features: dict[str, pd.DataFrame] = {sym: tasc_signals(df) for sym, df in panels.items()}
    all_ts = sorted(set.intersection(*(set(f["ts"]) for f in features.values())))
    if not all_ts:
        return []
    rows_by_sym = {sym: f.set_index("ts").reindex(all_ts) for sym, f in features.items()}
    open_px = pd.DataFrame({sym: f["open"] for sym, f in rows_by_sym.items()}, index=all_ts)
    close_px = pd.DataFrame({sym: f["close"] for sym, f in rows_by_sym.items()}, index=all_ts)
    fwd_ret = open_px.shift(-1) / open_px - 1.0
    score_ma = close_px.pct_change(126, fill_method=None)
    score_ci = pd.DataFrame({sym: rows_by_sym[sym]["ci_state"] for sym in rows_by_sym}, index=all_ts)
    score_ulcer = pd.DataFrame(
        {
            sym: close_px[sym].rolling(126, min_periods=80).apply(lambda x: -ulcer_index(pd.Series(x)), raw=False)
            for sym in rows_by_sym
        },
        index=all_ts,
    )
    variants = {
        "sector_mom_top3": score_ma,
        "sector_ci_top3": score_ci * score_ma.rank(axis=1, pct=True),
        "sector_ulcer_top3": score_ulcer.rank(axis=1, pct=True) + score_ma.rank(axis=1, pct=True),
    }
    rows: list[dict[str, object]] = []
    for variant, score in variants.items():
        ranks = score.rank(axis=1, ascending=False, method="first")
        weights = (ranks <= 3).astype(float)
        weights = weights.div(weights.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        held = weights.shift(1).fillna(0.0)
        port_ret_gross = (held * fwd_ret.fillna(0.0)).sum(axis=1)
        turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
        for cost in costs:
            ret = port_ret_gross - turnover * cost / 10_000.0
            equity = pd.DataFrame({"ts": all_ts, "ret": ret.to_numpy(), "turnover": turnover.to_numpy()})
            equity["position"] = held.abs().sum(axis=1).to_numpy()
            equity["nav"] = (1.0 + equity["ret"]).cumprod()
            equity["open"] = np.nan
            equity["close"] = equity["nav"]
            row = compute_metrics(equity, sleeve="sector_etf", variant=variant, cost_bps=cost, bars_per_year=252.0)
            rows.append(row)
            equity.to_csv(out_equity / f"sector_etf_{variant}_cost{int(cost)}.csv", index=False)
    return rows


def write_readme(metrics: pd.DataFrame, out_dir: Path, skipped: list[str]) -> None:
    lines = ["# TASC 2025 Trend Platform Component Tests\n\n"]
    lines.append("Research-scoped test of autocorrelation gates, Laguerre/Continuation Index, SuperTrend stops/re-entry, channel+ADX breakout, progressive realized-vol overlay, and Ulcer-aware sector rotation.\n\n")
    if skipped:
        lines.append("## Skipped Inputs\n")
        for item in skipped:
            lines.append(f"- {item}\n")
        lines.append("\n")
    lines.append("## Best Variants By Sleeve (0 bps)\n")
    zero = metrics[metrics["cost_bps"] == 0].copy()
    for sleeve, sub in zero.groupby("sleeve"):
        active = sub[sub["avg_abs_exposure"] > 0.05]
        ranked = active if not active.empty else sub
        best_sharpe = ranked.sort_values("sharpe", ascending=False).iloc[0]
        best_ulcer = ranked.sort_values("ulcer_index", ascending=True).iloc[0]
        lines.append(
            f"- `{sleeve}`: best Sharpe `{best_sharpe['variant']}` "
            f"({best_sharpe['sharpe']:+.2f}, maxDD {best_sharpe['max_dd']:.1%}); "
            f"lowest Ulcer `{best_ulcer['variant']}` ({best_ulcer['ulcer_index']:.2%}).\n"
        )
    no_trade = zero[zero["avg_abs_exposure"] <= 0.01]
    if not no_trade.empty:
        variants = ", ".join(sorted(no_trade["variant"].unique()))
        lines.append(f"\nNo-trade variants excluded from ranking: `{variants}`.\n")
    lines.append("\n## Promotion Review\n")
    lines.append("- Promote to continued research: Laguerre/Continuation Index on daily CL, SuperTrend state on 60m CL, autocorrelation sizing as a defensive overlay, and Ulcer Index reporting.\n")
    lines.append("- Do not promote yet: channel+ADX breakout as currently parameterized (no trades), hard BTC gates/overlays that reduce convexity too much, and sector CI ranking before longer-history validation.\n")
    lines.append("- Treat this as a component screen, not production proof: CL institutional artifact currently starts in 2025 for this run, so longer backfills/roll validation are required before deployment.\n")
    lines.append("\n## Files\n")
    lines.append("- `metrics.csv`: all sleeve/variant/cost rows.\n")
    lines.append("- `tail_windows.csv`: strategy and asset returns in crisis / dislocation windows.\n")
    lines.append("- `regime_summary.csv`: return summaries by autocorrelation and realized-volatility buckets.\n")
    lines.append("- `equity_curves/`: per-variant equity curves.\n")
    lines.append("- `figures/`: overview charts.\n")
    out_dir.joinpath("README.md").write_text("".join(lines))


def write_figures(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    zero = metrics[metrics["cost_bps"] == 0].copy()
    if zero.empty:
        return
    for metric, label in [("sharpe", "Sharpe"), ("max_dd", "Max Drawdown"), ("ulcer_index", "Ulcer Index")]:
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_df = zero.sort_values(["sleeve", metric])
        ax.barh(plot_df["sleeve"] + " / " + plot_df["variant"], plot_df[metric])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"TASC 2025 Component {label} (0 bps)")
        ax.set_xlabel(label)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{metric}_0bps.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    costs = [float(x) for x in args.costs_bps.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    equity_dir = out_dir / "equity_curves"
    equity_dir.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    rows: list[dict[str, object]] = []

    btc = load_btc_daily(args.crypto_lake)
    rows.extend(run_single_asset_suite(btc, sleeve="btc_daily", costs=costs, bars_per_year=ANN_DAILY, out_equity=equity_dir))
    rows.extend(run_btc_regime_baselines(btc, costs=costs, out_equity=equity_dir))

    cl_dir = Path(args.cl_dir)
    for tf, ppy in [("60m", 24 * 365.0), ("4h", 6 * 365.0), ("1d", 365.0)]:
        cl = load_cl_bars(cl_dir, tf)
        if cl is None or cl.empty:
            skipped.append(f"CL {tf}: missing {cl_dir / 'bars_1m.parquet'}")
            continue
        rows.extend(run_single_asset_suite(cl, sleeve=f"cl_{tf}", costs=costs, bars_per_year=ppy, out_equity=equity_dir))

    try:
        sectors = load_sector_etfs(args.etf_db)
        if sectors:
            rows.extend(run_sector_rotation(sectors, costs=costs, out_equity=equity_dir))
        else:
            skipped.append("Sector ETF: no ETF rows loaded.")
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"Sector ETF: failed to load ({exc})")

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(["sleeve", "cost_bps", "sharpe"], ascending=[True, True, False])
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    out_dir.joinpath("config.json").write_text(
        json.dumps(
            {
                "crypto_lake": args.crypto_lake,
                "etf_db": args.etf_db,
                "cl_dir": str(cl_dir),
                "costs_bps": costs,
                "skipped": skipped,
            },
            indent=2,
        )
    )
    write_figures(metrics, out_dir)
    write_tail_window_summary(out_dir)
    write_regime_summary(out_dir)
    write_readme(metrics, out_dir, skipped)
    print(metrics.to_string(index=False))
    print(f"[tasc2025] wrote {out_dir}")


if __name__ == "__main__":
    main()

