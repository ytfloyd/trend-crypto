#!/usr/bin/env python
"""Full-USDC Quantitative Momentum research backtest.

Crypto adaptation of Gray/Vogel Quantitative Momentum:
  1. Rank the universe by generic momentum over a formation window that skips
     the most recent short-term reversal window.
  2. Keep the strongest momentum names, then prefer smoother "frog-in-the-pan"
     paths measured by the fraction of positive daily returns in the formation
     window.
  3. Rebalance monthly, using close(t) signals and open-to-close returns from
     t+1 after a one-bar execution lag.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT.parent / "data" / "coinbase_crypto_ohlcv_lake.duckdb"
DEFAULT_OUT = ROOT / "artifacts" / "research" / "quant_momentum_usdc"

ANN_FACTOR = 365.0
INITIAL_EQUITY = 100_000.0

STABLE_BASES = {
    "USDT",
    "USDC",
    "DAI",
    "BUSD",
    "TUSD",
    "USDP",
    "GUSD",
    "FRAX",
    "PYUSD",
    "FDUSD",
    "EURC",
    "EURT",
    "GBPT",
    "GYEN",
    "USDS",
    "UST",
    "MIM",
    "LUSD",
    "SUSD",
    "CRVUSD",
    "GHO",
    "MKUSD",
    "CBETH",
    "MSOL",
    "LSETH",
    "OETH",
    "WSTETH",
    "WBTC",
    "CBBTC",
    "WETH",
}


@dataclass(frozen=True)
class StrategySpec:
    label: str
    lookback_days: int
    skip_days: int


@dataclass(frozen=True)
class ResearchConfig:
    min_history_days: int = 365
    min_coverage: float = 0.90
    liquidity_window: int = 90
    min_dollar_volume: float = 500_000.0
    top_momentum: int = 100
    final_positions: int = 50
    vol_lookback: int = 42
    vol_floor: float = 0.10
    cost_bps_per_side: float = 30.0
    execution_lag: int = 1
    cash_yield: float = 0.0


SPECS = [
    StrategySpec("qm_365_30", 365, 30),
    StrategySpec("qm_180_30", 180, 30),
    StrategySpec("qm_90_14", 90, 14),
]
WEIGHT_METHODS = ("equal", "inv_vol")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-USDC Quantitative Momentum research")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--min-dollar-volume", type=float, default=500_000.0)
    p.add_argument("--top-momentum", type=int, default=100)
    p.add_argument("--final-positions", type=int, default=50)
    p.add_argument("--cost-bps-per-side", type=float, default=30.0)
    return p.parse_args()


def load_usdc_panel(
    db_path: str | Path,
    start: str,
    end: str,
    cfg: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clean daily bars for structurally eligible Coinbase USDC pairs."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        universe = con.execute(
            """
            SELECT symbol, MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n_days
            FROM bars_1d_clean
            WHERE symbol LIKE '%-USDC'
            GROUP BY symbol
            """
        ).fetch_df()
        universe["first_ts"] = pd.to_datetime(universe["first_ts"], utc=True)
        universe["last_ts"] = pd.to_datetime(universe["last_ts"], utc=True)
        universe["span_days"] = (universe["last_ts"] - universe["first_ts"]).dt.days
        universe["coverage"] = universe["n_days"] / universe["span_days"].replace(0, np.nan)
        universe["base"] = universe["symbol"].str.split("-").str[0]
        universe["structural_eligible"] = (
            ~universe["base"].isin(STABLE_BASES)
            & (universe["span_days"] >= cfg.min_history_days)
            & (universe["coverage"] >= cfg.min_coverage)
        )
        symbols = sorted(universe.loc[universe["structural_eligible"], "symbol"].tolist())
        if not symbols:
            raise ValueError("No structurally eligible USDC symbols found")

        placeholders = ",".join(["?"] * len(symbols))
        panel = con.execute(
            f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM bars_1d_clean
            WHERE symbol IN ({placeholders})
              AND ts >= ?
              AND ts <= ?
              AND open > 0
              AND close > 0
              AND high >= low
            ORDER BY ts, symbol
            """,
            [*symbols, start, end],
        ).fetch_df()
    finally:
        con.close()

    required = {"symbol", "ts", "open", "high", "low", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    panel["ts"] = pd.to_datetime(panel["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    panel = panel.dropna(subset=["open", "close"]).sort_values(["ts", "symbol"])
    return panel, universe.sort_values("symbol").reset_index(drop=True)


def to_wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    return panel.pivot(index="ts", columns="symbol", values=field).sort_index()


def compute_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    spec: StrategySpec,
    cfg: ResearchConfig,
) -> dict[str, pd.DataFrame]:
    """Compute generic momentum, FIP path quality, and point-in-time eligibility."""
    ret_cc = close.pct_change(fill_method=None)
    momentum = close.shift(spec.skip_days) / close.shift(spec.skip_days + spec.lookback_days) - 1.0

    positive_days = (ret_cc > 0).astype(float).where(ret_cc.notna())
    path_quality = positive_days.shift(spec.skip_days).rolling(
        spec.lookback_days,
        min_periods=spec.lookback_days,
    ).mean()

    dollar_volume = close * volume
    med_dollar_volume = dollar_volume.rolling(
        cfg.liquidity_window,
        min_periods=cfg.liquidity_window,
    ).median()
    eligible = close.notna() & (med_dollar_volume >= cfg.min_dollar_volume)

    return {
        "momentum": momentum,
        "path_quality": path_quality,
        "eligible": eligible,
        "ret_cc": ret_cc,
        "med_dollar_volume": med_dollar_volume,
    }


def monthly_signal_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available observation in each month, used as close-of-month signal dates."""
    dates = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(dates.groupby(index.to_period("M")).max().to_list())


def build_selection_mask(
    momentum: pd.DataFrame,
    path_quality: pd.DataFrame,
    eligible: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    selected = pd.DataFrame(False, index=momentum.index, columns=momentum.columns)
    for dt in monthly_signal_dates(momentum.index):
        candidates = pd.DataFrame(
            {
                "momentum": momentum.loc[dt],
                "path_quality": path_quality.loc[dt],
                "eligible": eligible.loc[dt],
            }
        ).dropna(subset=["momentum", "path_quality"])
        candidates = candidates[candidates["eligible"]]
        if len(candidates) < cfg.final_positions:
            continue
        mom_pool = candidates.sort_values("momentum", ascending=False).head(cfg.top_momentum)
        final = mom_pool.sort_values(
            ["path_quality", "momentum"],
            ascending=[False, False],
        ).head(cfg.final_positions)
        selected.loc[dt, final.index] = True
    return selected


def build_target_weights(
    selected: pd.DataFrame,
    ret_cc: pd.DataFrame,
    method: str,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    if method not in WEIGHT_METHODS:
        raise ValueError(f"Unknown weight method {method!r}; expected one of {WEIGHT_METHODS}")

    weights = pd.DataFrame(np.nan, index=selected.index, columns=selected.columns, dtype=float)
    vol = ret_cc.rolling(cfg.vol_lookback, min_periods=max(10, cfg.vol_lookback // 2)).std()
    vol = (vol * np.sqrt(ANN_FACTOR)).clip(lower=cfg.vol_floor)

    for dt in monthly_signal_dates(selected.index):
        names = selected.columns[selected.loc[dt]].tolist()
        weights.loc[dt] = 0.0
        if not names:
            continue
        if method == "equal":
            w = pd.Series(1.0 / len(names), index=names)
        else:
            inv = (1.0 / vol.loc[dt, names]).replace([np.inf, -np.inf], np.nan)
            inv = inv.fillna(inv.median()).fillna(1.0)
            w = inv / inv.sum()
        weights.loc[dt, names] = w

    return weights.ffill().fillna(0.0)


def simulate_portfolio(
    weights_signal: pd.DataFrame,
    open_: pd.DataFrame,
    close: pd.DataFrame,
    cfg: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_index = weights_signal.index.intersection(open_.index).intersection(close.index).sort_values()
    common_cols = weights_signal.columns.intersection(open_.columns).intersection(close.columns)

    w = weights_signal.reindex(index=common_index, columns=common_cols).fillna(0.0)
    ret_oc = (close.reindex(index=common_index, columns=common_cols) /
              open_.reindex(index=common_index, columns=common_cols) - 1.0).fillna(0.0)

    w_held = w.shift(cfg.execution_lag).fillna(0.0)
    gross = w_held.abs().sum(axis=1)
    cash_weight = (1.0 - gross).clip(lower=0.0)
    turnover = (w_held - w_held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_ret = turnover * (cfg.cost_bps_per_side / 10_000.0)
    cash_ret = cash_weight * (cfg.cash_yield / ANN_FACTOR)
    portfolio_ret = (w_held * ret_oc).sum(axis=1) + cash_ret - cost_ret
    equity = INITIAL_EQUITY * (1.0 + portfolio_ret).cumprod()

    equity_df = pd.DataFrame(
        {
            "ts": common_index,
            "portfolio_ret": portfolio_ret.values,
            "portfolio_equity": equity.values,
            "gross_exposure": gross.values,
            "cash_weight": cash_weight.values,
            "turnover": turnover.values,
            "cost_ret": cost_ret.values,
            "cash_ret": cash_ret.values,
        }
    )
    weights_held = w_held.stack().rename("weight").reset_index()
    weights_held = weights_held[weights_held["weight"].abs() > 1e-12]
    return equity_df, weights_held


def compute_metrics(equity_df: pd.DataFrame, label: str, cfg: ResearchConfig) -> dict[str, float | str]:
    if "gross_exposure" in equity_df.columns and (equity_df["gross_exposure"] > 0).any():
        first_active = equity_df.index[equity_df["gross_exposure"] > 0][0]
        equity_df = equity_df.loc[first_active:].copy()
    eq = pd.Series(equity_df["portfolio_equity"].values, index=pd.to_datetime(equity_df["ts"]))
    ret = eq.pct_change(fill_method=None).dropna()
    if len(eq) < 2:
        return {"label": label}
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0
    vol = ret.std() * np.sqrt(ANN_FACTOR)
    sharpe = (ret.mean() * ANN_FACTOR) / vol if vol > 0 else np.nan
    downside = ret[ret < 0]
    sortino = (
        (ret.mean() * ANN_FACTOR) / (downside.std() * np.sqrt(ANN_FACTOR))
        if len(downside) > 1 and downside.std() > 0
        else np.nan
    )
    drawdown = eq / eq.cummax() - 1.0
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "label": label,
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "years": float(years),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "avg_gross": float(equity_df["gross_exposure"].mean()),
        "avg_turnover": float(equity_df["turnover"].mean()),
        "annual_cost_drag": float(equity_df["cost_ret"].mean() * ANN_FACTOR),
        "cost_bps_per_side": float(cfg.cost_bps_per_side),
    }


def btc_buy_hold(close: pd.DataFrame, index: Iterable[pd.Timestamp]) -> pd.Series:
    if "BTC-USDC" not in close.columns:
        return pd.Series(dtype=float, name="btc_usdc_bh")
    btc = close["BTC-USDC"].reindex(pd.DatetimeIndex(index)).dropna()
    if btc.empty:
        return pd.Series(dtype=float, name="btc_usdc_bh")
    nav = INITIAL_EQUITY * btc / btc.iloc[0]
    nav.name = "btc_usdc_bh"
    return nav


def selection_long(selected: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for dt in monthly_signal_dates(selected.index):
        for symbol in selected.columns[selected.loc[dt]]:
            rows.append({"ts": dt, "strategy": label, "symbol": symbol})
    return pd.DataFrame(rows)


def write_figures(
    navs: pd.DataFrame,
    metrics: pd.DataFrame,
    holdings: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    for col in navs.columns:
        if col == "btc_usdc_bh":
            ax.plot(navs.index, navs[col] / 1_000, lw=1.6, ls="--", label="BTC-USDC B&H")
        else:
            ax.plot(navs.index, navs[col] / 1_000, lw=1.3, label=col)
    ax.set_yscale("log")
    ax.set_ylabel("NAV ($k, log)")
    ax.set_title("Full-USDC Quantitative Momentum variants")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    for col in navs.columns:
        dd = navs[col] / navs[col].cummax() - 1.0
        ax.plot(navs.index, dd * 100.0, lw=1.0, label=col)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_equity_drawdown.png", dpi=160)
    plt.close(fig)

    plot_df = metrics[metrics["label"] != "btc_usdc_bh"].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["sharpe"], color="#1f77b4")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=30, ha="right")
    ax.set_ylabel("Sharpe")
    ax.set_title("Variant Sharpe comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_sharpe_comparison.png", dpi=160)
    plt.close(fig)

    if not holdings.empty:
        top = holdings["symbol"].value_counts().head(25).sort_values()
        fig, ax = plt.subplots(figsize=(10, 7))
        top.plot(kind="barh", ax=ax, color="#2ca02c")
        ax.set_xlabel("Monthly selections across all variants")
        ax.set_title("Most frequently selected symbols")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "03_top_selected_symbols.png", dpi=160)
        plt.close(fig)


def run_study(panel: pd.DataFrame, cfg: ResearchConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    open_ = to_wide(panel, "open")
    close = to_wide(panel, "close")
    volume = to_wide(panel, "volume")

    metrics_rows = []
    navs = pd.DataFrame(index=close.index)
    holdings_parts = []
    selection_diag = []

    for spec in SPECS:
        features = compute_features(close, volume, spec, cfg)
        selected = build_selection_mask(
            features["momentum"],
            features["path_quality"],
            features["eligible"],
            cfg,
        )
        eligible_counts = features["eligible"].sum(axis=1)
        selected_counts = selected.sum(axis=1)
        for method in WEIGHT_METHODS:
            label = f"{spec.label}_{method}"
            weights = build_target_weights(selected, features["ret_cc"], method, cfg)
            equity_df, weights_held = simulate_portfolio(weights, open_, close, cfg)
            equity_df.to_csv(out_dir / f"equity_{label}.csv", index=False)
            weights_held.to_csv(out_dir / f"weights_held_{label}.csv", index=False)
            metrics_rows.append(compute_metrics(equity_df, label, cfg))
            nav = pd.Series(
                equity_df["portfolio_equity"].values,
                index=pd.to_datetime(equity_df["ts"]),
            )
            if (equity_df["gross_exposure"] > 0).any():
                active_start = equity_df.loc[equity_df["gross_exposure"] > 0, "ts"].iloc[0]
                nav = nav.loc[pd.Timestamp(active_start):]
            navs[label] = nav

        holdings = selection_long(selected, spec.label)
        holdings_parts.append(holdings)
        selection_diag.append(
            {
                "spec": spec.label,
                "lookback_days": spec.lookback_days,
                "skip_days": spec.skip_days,
                "first_selection": (
                    str(selected_counts[selected_counts > 0].index[0].date())
                    if (selected_counts > 0).any()
                    else None
                ),
                "avg_eligible": float(eligible_counts.mean()),
                "max_eligible": int(eligible_counts.max()),
                "avg_selected_on_signal_dates": float(
                    selected_counts.loc[monthly_signal_dates(selected.index)].mean()
                ),
            }
        )

    first_nav = navs.dropna(how="all").index[0]
    btc = btc_buy_hold(close, navs.loc[first_nav:].index)
    if not btc.empty:
        navs.loc[btc.index, "btc_usdc_bh"] = btc
        metrics_rows.append(
            compute_metrics(
                pd.DataFrame({"ts": btc.index, "portfolio_equity": btc.values,
                              "gross_exposure": 1.0, "turnover": 0.0, "cost_ret": 0.0}),
                "btc_usdc_bh",
                cfg,
            )
        )

    holdings_all = pd.concat(holdings_parts, ignore_index=True) if holdings_parts else pd.DataFrame()
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_rows, f, indent=2)
    navs.dropna(how="all").to_csv(out_dir / "navs.csv", index_label="ts")
    holdings_all.to_csv(out_dir / "holdings.csv", index=False)
    pd.DataFrame(selection_diag).to_csv(out_dir / "selection_diagnostics.csv", index=False)

    config_blob = {
        "research_config": asdict(cfg),
        "specs": [asdict(s) for s in SPECS],
        "weight_methods": list(WEIGHT_METHODS),
        "initial_equity": INITIAL_EQUITY,
        "ann_factor": ANN_FACTOR,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_blob, f, indent=2)

    write_figures(navs.dropna(how="all"), metrics, holdings_all, out_dir)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cfg = ResearchConfig(
        min_dollar_volume=args.min_dollar_volume,
        top_momentum=args.top_momentum,
        final_positions=args.final_positions,
        cost_bps_per_side=args.cost_bps_per_side,
    )
    panel, universe = load_usdc_panel(args.db, args.start, args.end, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    universe.to_csv(out_dir / "universe.csv", index=False)
    run_study(panel, cfg, out_dir)
    print(f"[quant_momentum_usdc] wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
