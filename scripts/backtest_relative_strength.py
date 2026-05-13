#!/usr/bin/env python3
"""
Faber Relative Strength — Crypto Adaptation
============================================
Backtest of the strategy in Faber (2010) "Relative Strength Strategies for
Investing" applied to the Coinbase USD spot universe.

Original rules (sector ETF version):
  - Each month, rank assets by trailing total return over a fixed lookback
    (1m / 3m / 6m / 9m / 12m, or the average of all five = "combo")
  - Equal-weight the Top N (typically 1, 2, or 3)
  - Rebalance monthly at the month-end close
  - Optional dynamic hedge: go 100% cash whenever the market proxy (S&P 500
    in the paper, BTC here) is below its 10-month SMA

Crypto adaptations:
  - "Month" = 21 trading-day window (crypto trades 24/7; calendar months
    work fine but 21d gives consistent compounding) -- we use calendar
    month-ends with daily bars for the rebalance schedule
  - "Total return" = price return only (no dividends in crypto)
  - Universe filter at each rebalance: USD pairs ex-stables/wrapped, with
    >= 12 months of history and 24h $vol >= $100k
  - Hedge proxy: BTC vs its 10-month (= 210 day) SMA
  - Equity is compounded (in contrast to Faber's monthly returns table,
    which is also compounded -- this matches the paper)

Sweeps (defaults):
  Lookbacks: 1m / 3m / 6m / 12m / combo(1,3,6,9,12)
  Top N:     1 / 3 / 5 / 10
  Hedge:     none / btc-sma10m
  Costs:     0 / 30 / 60 / 100 bps round-trip

Run:

    python scripts/backtest_relative_strength.py \
        --db /Users/russellfloyd/Dropbox/NRT/nrt_dev/data/market.duckdb \
        --start 2018-01-01 \
        --out artifacts/backtests/relative_strength

Outputs (under --out):

    sweep.csv          per-variant CAGR / Sharpe / MaxDD / turnover
    daily_log.csv      daily equity for every variant + benchmarks
    monthly_picks.csv  which assets the "combo / Top 3 / hedged" book held
    summary.json       headline stats for the recommended variant
    equity.png         best variant + benchmarks
    heatmap.png        sweep grid (Sharpe, net of 60bps round-trip costs)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rel_strength")


# Reuse the Turtle script's exclusion list — same universe rationale.
EXCLUDED_BASES: set[str] = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "GUSD", "FRAX", "PYUSD",
    "FDUSD", "EURC", "EURT", "GBPT", "GYEN", "USDS", "UST", "MIM", "LUSD",
    "SUSD", "CRVUSD", "GHO", "MKUSD", "WBTC", "CBBTC", "CBETH", "WETH",
    "STETH", "RETH", "MSOL", "PAX", "HUSD", "TRIBE", "FEI", "ALUSD", "RAI",
}


@dataclass(frozen=True)
class Config:
    starting_capital: float = 20_000.0
    min_history_months: int = 12
    min_dollar_vol: float = 100_000.0   # 24h USD volume floor at rebalance
    sma_proxy: str = "BTC-USD"
    sma_months: int = 10                 # = ~210 trading-day SMA, monthly resampled
    cost_bps_default: float = 60.0       # one-way (60 bps round-trip ≈ 30 bps each side)


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_base(symbol: str) -> str:
    return symbol.split("-", 1)[0].upper()


def load_panel(db_path: str, start: str | None, end: str | None) -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    bounds = []
    if start:
        bounds.append(f"ts >= TIMESTAMP '{start}'")
    if end:
        bounds.append(f"ts <= TIMESTAMP '{end}'")
    where = (" AND " + " AND ".join(bounds)) if bounds else ""
    sql = f"""
        SELECT symbol, CAST(ts AS TIMESTAMP) AS ts,
               close, volume, close * volume AS dollar_volume
        FROM bars_1d_clean
        WHERE symbol LIKE '%-USD'
        {where}
        ORDER BY symbol, ts
    """
    df = conn.execute(sql).fetchdf()
    conn.close()
    if df.empty:
        raise RuntimeError(f"No rows from {db_path}")

    df["base"] = df["symbol"].map(_strip_base)
    df = df[~df["base"].isin(EXCLUDED_BASES)].drop(columns=["base"])
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None).dt.normalize()
    df = (df.sort_values(["symbol", "ts"])
            .drop_duplicates(["symbol", "ts"], keep="last")
            .reset_index(drop=True))
    return df


def to_panel(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="ts", columns="symbol", values=col).sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
# Signal construction
# ═══════════════════════════════════════════════════════════════════════════════

LOOKBACKS_DAYS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "9m": 189,
    "12m": 252,
}
LOOKBACK_LABELS = ["1m", "3m", "6m", "9m", "12m", "combo"]


def compute_score(closes_monthly: pd.DataFrame, lookback: str) -> pd.DataFrame:
    """Trailing total return at each month-end. 'combo' averages 1/3/6/9/12."""
    if lookback == "combo":
        parts = []
        for k in (1, 3, 6, 9, 12):
            parts.append(closes_monthly / closes_monthly.shift(k) - 1.0)
        return sum(parts) / len(parts)
    months = {"1m": 1, "3m": 3, "6m": 6, "9m": 9, "12m": 12}[lookback]
    return closes_monthly / closes_monthly.shift(months) - 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Single-variant backtest
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VariantResult:
    name: str
    lookback: str
    top_n: int
    hedge: str
    cost_bps: float
    daily_equity: pd.Series
    monthly_picks: pd.DataFrame   # date x rank → symbol
    monthly_weights: pd.DataFrame # date x symbol weights
    metrics: dict[str, float]


def annualised_metrics(equity: pd.Series) -> dict[str, float]:
    if equity.empty or equity.iloc[0] <= 0:
        return {"cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "max_dd": 0.0, "calmar": 0.0, "total_return": 0.0}
    rets = equity.pct_change().fillna(0.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    vol = rets.std(ddof=0) * math.sqrt(365.25)
    sharpe = (rets.mean() * 365.25) / vol if vol > 0 else 0.0
    neg = rets[rets < 0]
    nvol = neg.std(ddof=0) * math.sqrt(365.25)
    sortino = (rets.mean() * 365.25) / nvol if nvol > 0 else 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    return {
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": max_dd,
        "calmar": float(calmar) if calmar == calmar else 0.0,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
    }


def run_variant(
    closes: pd.DataFrame,
    closes_monthly: pd.DataFrame,
    monthly_dates: pd.DatetimeIndex,
    eligible_monthly: pd.DataFrame,
    btc_above_sma: pd.Series | None,
    *,
    lookback: str,
    top_n: int,
    hedge: bool,
    cost_bps: float,
    cfg: Config,
) -> VariantResult:
    """Run one (lookback, top_n, hedge, cost) variant, daily equity curve."""

    score = compute_score(closes_monthly, lookback)

    # At each rebalance month-end, pick top_n eligible, equal-weight.
    weights_monthly = pd.DataFrame(
        0.0, index=monthly_dates, columns=closes.columns
    )
    picks_rows: list[dict[str, Any]] = []

    for ts in monthly_dates:
        if ts not in score.index or ts not in eligible_monthly.index:
            continue
        s = score.loc[ts]
        elig = eligible_monthly.loc[ts]
        candidates = s.where(elig).dropna()
        if candidates.empty:
            continue
        top = candidates.sort_values(ascending=False).head(top_n)

        # Hedge: if BTC is below its 10-month SMA at this month-end, go to cash.
        if hedge and btc_above_sma is not None and ts in btc_above_sma.index:
            if not bool(btc_above_sma.loc[ts]):
                # All cash this month; record empty pick.
                picks_rows.append({"ts": ts, "n_picks": 0, "hedged": True})
                continue

        w = 1.0 / len(top)
        weights_monthly.loc[ts, top.index] = w
        picks_rows.append({
            "ts": ts, "n_picks": len(top), "hedged": False,
            **{f"rank_{i+1}": sym for i, sym in enumerate(top.index)},
        })

    # Forward-fill weights to daily, then shift by 1 day to avoid using the
    # rebalance bar's close (we know weights at the close of the month-end
    # and apply them on the next trading day).
    weights_daily = weights_monthly.reindex(closes.index).ffill().shift(1).fillna(0.0)

    # Daily returns (close-to-close). Missing (delisted) → 0 return that day.
    rets = closes.pct_change(fill_method=None).fillna(0.0).clip(lower=-0.99, upper=10.0)

    # Strategy gross return = sum(w_i * r_i)
    gross = (weights_daily * rets).sum(axis=1)

    # Transaction costs: charged on rebalance days only, proportional to
    # one-way turnover.  cost_bps is *round-trip* (so half is paid each side
    # of a trade — net of buys/sells, total turnover is the L1 weight delta).
    turnover_per_rebalance = (
        weights_monthly.diff().abs().sum(axis=1).fillna(weights_monthly.iloc[0].abs().sum())
    )
    # Re-attach turnover on the day weights actually change in the daily panel
    # (which is the day after each month-end after the shift(1) above).
    turnover_daily = pd.Series(0.0, index=closes.index)
    rebalance_days = weights_monthly.index + pd.Timedelta(days=1)
    rebalance_days = rebalance_days[rebalance_days.isin(closes.index)]
    for d, t in zip(rebalance_days, turnover_per_rebalance.values):
        turnover_daily.loc[d] = t
    cost_daily = turnover_daily * (cost_bps / 10_000.0)

    net = gross - cost_daily
    equity = (1.0 + net).cumprod() * cfg.starting_capital
    # Mark first day with starting equity (cumprod starts at (1+r0)).
    equity.iloc[0] = cfg.starting_capital * (1.0 + net.iloc[0])

    metrics = annualised_metrics(equity)
    metrics["avg_turnover_per_rebalance"] = float(turnover_per_rebalance.mean())
    metrics["annual_turnover"] = float(turnover_per_rebalance.sum() / max(
        (equity.index[-1] - equity.index[0]).days / 365.25, 1e-9
    ))
    metrics["pct_months_hedged_to_cash"] = float(
        weights_monthly.sum(axis=1).eq(0.0).mean()
    )
    metrics["cost_bps_round_trip"] = float(cost_bps)

    name = f"{lookback}_top{top_n}{'_hedge' if hedge else ''}"
    picks_df = pd.DataFrame(picks_rows).set_index("ts") if picks_rows else pd.DataFrame()
    return VariantResult(
        name=name,
        lookback=lookback,
        top_n=top_n,
        hedge=hedge,
        cost_bps=cost_bps,
        daily_equity=equity,
        monthly_picks=picks_df,
        monthly_weights=weights_monthly,
        metrics=metrics,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Eligibility / regime
# ═══════════════════════════════════════════════════════════════════════════════

def build_eligibility(
    closes: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    monthly_dates: pd.DatetimeIndex,
    cfg: Config,
) -> pd.DataFrame:
    """Boolean panel: at each month-end, is each symbol eligible?

    Rules: at least min_history_months (252 days) of non-NaN closes AND
    rolling 30-day median dollar volume >= cfg.min_dollar_vol.
    """
    has_history = closes.notna().rolling(cfg.min_history_months * 21,
                                         min_periods=cfg.min_history_months * 21).count() \
                       >= cfg.min_history_months * 21
    median_dvol = dollar_vol.rolling(30, min_periods=15).median()
    elig_daily = has_history & (median_dvol >= cfg.min_dollar_vol)
    return elig_daily.reindex(monthly_dates, method="pad").fillna(False)


def build_btc_regime(closes: pd.DataFrame, cfg: Config) -> pd.Series:
    """1 if BTC monthly close > BTC 10-month SMA, else 0."""
    if cfg.sma_proxy not in closes.columns:
        logger.warning("BTC proxy %s missing — hedge disabled", cfg.sma_proxy)
        return pd.Series(True, index=closes.resample("ME").last().index, dtype=bool)
    btc_m = closes[cfg.sma_proxy].resample("ME").last()
    sma = btc_m.rolling(cfg.sma_months, min_periods=cfg.sma_months).mean()
    return (btc_m > sma).fillna(False)


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep + reporting
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(
    closes: pd.DataFrame, dollar_vol: pd.DataFrame, cfg: Config,
    *, top_ns: list[int], lookbacks: list[str], hedges: list[bool],
    cost_bps: float,
) -> tuple[list[VariantResult], pd.DatetimeIndex]:
    closes_monthly = closes.resample("ME").last()
    monthly_dates = closes_monthly.index
    eligible_monthly = build_eligibility(closes, dollar_vol, monthly_dates, cfg)
    btc_above = build_btc_regime(closes, cfg)

    n_eligible = eligible_monthly.sum(axis=1)
    logger.info(
        "Eligibility: avg %.1f symbols/month (min %d, max %d)",
        n_eligible.mean(), n_eligible.min(), n_eligible.max(),
    )

    results: list[VariantResult] = []
    for lb in lookbacks:
        for n in top_ns:
            for hedge in hedges:
                r = run_variant(
                    closes=closes, closes_monthly=closes_monthly,
                    monthly_dates=monthly_dates,
                    eligible_monthly=eligible_monthly,
                    btc_above_sma=btc_above,
                    lookback=lb, top_n=n, hedge=hedge,
                    cost_bps=cost_bps, cfg=cfg,
                )
                results.append(r)
    return results, monthly_dates


def benchmarks(closes: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    """BTC HODL and equal-weight buy-and-hold (over BTC+ETH)."""
    out: dict[str, pd.Series] = {}
    if "BTC-USD" in closes.columns:
        btc = closes["BTC-USD"].dropna()
        if not btc.empty:
            eq = cfg.starting_capital * btc / btc.iloc[0]
            out["BTC HODL"] = eq.reindex(closes.index).ffill()
    # Equal weight rebalanced monthly across the top 5 majors that exist
    majors = [s for s in ("BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "LINK-USD")
              if s in closes.columns]
    if majors:
        sub = closes[majors].pct_change(fill_method=None).fillna(0.0).clip(lower=-0.99, upper=10.0)
        weights = pd.DataFrame(0.0, index=closes.index, columns=majors)
        for ts in closes.resample("ME").last().index:
            avail = closes[majors].loc[:ts].dropna(axis=1, how="all").columns
            if len(avail) > 0:
                weights.loc[ts:, avail] = 1.0 / len(avail)
                weights.loc[ts:, [c for c in majors if c not in avail]] = 0.0
        weights = weights.shift(1).fillna(0.0)
        port_ret = (weights * sub).sum(axis=1)
        out["EW Majors"] = (1.0 + port_ret).cumprod() * cfg.starting_capital
    return out


def save_outputs(
    out_dir: Path,
    results: list[VariantResult],
    benches: dict[str, pd.Series],
    cfg: Config,
    cost_bps: float,
) -> tuple[VariantResult, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- sweep.csv -----------------------------------------------------------
    sweep_rows: list[dict[str, Any]] = []
    for r in results:
        row = {
            "variant": r.name, "lookback": r.lookback, "top_n": r.top_n,
            "hedge": r.hedge, **r.metrics,
        }
        sweep_rows.append(row)
    sweep = pd.DataFrame(sweep_rows).sort_values("sharpe", ascending=False)
    sweep.to_csv(out_dir / "sweep.csv", index=False)

    # --- daily_log.csv -------------------------------------------------------
    daily = pd.DataFrame({r.name: r.daily_equity for r in results})
    for k, v in benches.items():
        daily[k] = v.reindex(daily.index).ffill()
    daily.to_csv(out_dir / "daily_log.csv")

    # --- pick the recommended variant ---------------------------------------
    # Prefer Sharpe > 1.0 with reasonable max_dd; then highest Calmar.
    candidates = sweep[(sweep.sharpe >= 0.7) & (sweep.max_dd > -0.7)]
    if not candidates.empty:
        best_row = candidates.sort_values("calmar", ascending=False).iloc[0]
    else:
        best_row = sweep.iloc[0]
    best = next(r for r in results if r.name == best_row["variant"])
    best.monthly_picks.to_csv(out_dir / "monthly_picks.csv")

    summary = {
        "recommended": best.name,
        "lookback": best.lookback,
        "top_n": best.top_n,
        "hedge": best.hedge,
        "cost_bps_round_trip": cost_bps,
        "metrics": best.metrics,
        "benchmarks": {
            name: annualised_metrics(eq.dropna()) for name, eq in benches.items()
        },
        "n_variants": len(results),
        "sample_size": {
            "start": str(daily.index[0].date()),
            "end": str(daily.index[-1].date()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # --- equity.png ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(best.daily_equity.index, best.daily_equity.values,
                label=f"Faber RS — {best.name}", color="#1f77b4", lw=1.6)
        for color, (name, eq) in zip(["#ff7f0e", "#2ca02c"], benches.items()):
            ax.plot(eq.index, eq.values, label=name, color=color, lw=1.0, alpha=0.85)
        ax.set_yscale("log")
        ax.set_ylabel("Equity ($)")
        ax.set_title(
            f"Faber Relative Strength (Crypto)  —  {summary['sample_size']['start']} → "
            f"{summary['sample_size']['end']}\n"
            f"recommended: {best.name}  CAGR {best.metrics['cagr']*100:.1f}%  "
            f"Sharpe {best.metrics['sharpe']:.2f}  MaxDD {best.metrics['max_dd']*100:.1f}%  "
            f"Turnover {best.metrics['annual_turnover']:.1f}x/yr"
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "equity.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        logger.warning("equity.png skipped (%s)", exc)

    # --- heatmap.png: Sharpe across lookback × top_n × hedge -----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for hedge, label in [(False, "no_hedge"), (True, "btc_sma10m_hedge")]:
            sub = sweep[sweep.hedge == hedge]
            if sub.empty:
                continue
            grid = sub.pivot(index="lookback", columns="top_n", values="sharpe")
            grid = grid.reindex(LOOKBACK_LABELS)
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(grid.values, aspect="auto", cmap="RdYlGn",
                           vmin=-0.5, vmax=2.0)
            ax.set_xticks(range(len(grid.columns)))
            ax.set_xticklabels(grid.columns)
            ax.set_yticks(range(len(grid.index)))
            ax.set_yticklabels(grid.index)
            ax.set_xlabel("Top N")
            ax.set_ylabel("Lookback")
            ax.set_title(f"Sharpe — {label} — {cost_bps:.0f}bps round-trip")
            for i in range(len(grid.index)):
                for j in range(len(grid.columns)):
                    v = grid.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                color="black" if abs(v - 0.75) < 0.75 else "white",
                                fontsize=9)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(out_dir / f"heatmap_{label}.png", dpi=140)
            plt.close(fig)
    except Exception as exc:  # pragma: no cover
        logger.warning("heatmap.png skipped (%s)", exc)

    return best, summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Faber Relative Strength — Crypto Adaptation")
    p.add_argument("--db", default=str(PROJECT_ROOT.parent / "data" / "market.duckdb"))
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "artifacts" / "backtests" / "relative_strength"),
    )
    p.add_argument("--cost-bps", type=float, default=60.0,
                   help="Round-trip cost in bps (default: 60).")
    p.add_argument("--min-dollar-vol", type=float, default=100_000.0,
                   help="Minimum 30-day median 24h $-volume at rebalance "
                        "(default: 100,000). Use 5_000_000 for majors-only.")
    p.add_argument("--cost-sweep", action="store_true",
                   help="Also run a cost-sensitivity sweep (0/30/60/100 bps).")
    p.add_argument("--top-ns", default="1,3,5,10",
                   help="Comma-separated Top N values.")
    p.add_argument("--lookbacks", default="1m,3m,6m,12m,combo",
                   help="Comma-separated lookbacks: 1m,3m,6m,9m,12m,combo")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = Config(min_dollar_vol=args.min_dollar_vol)

    logger.info("Loading panel from %s …", args.db)
    df = load_panel(args.db, args.start, args.end)
    closes = to_panel(df, "close")
    dvol = to_panel(df, "dollar_volume")
    logger.info("Panel: %d dates × %d symbols", *closes.shape)

    top_ns = [int(x) for x in args.top_ns.split(",")]
    lookbacks = args.lookbacks.split(",")
    hedges = [False, True]

    results, _ = run_sweep(
        closes, dvol, cfg,
        top_ns=top_ns, lookbacks=lookbacks, hedges=hedges,
        cost_bps=args.cost_bps,
    )
    benches = benchmarks(closes, cfg)

    out_dir = Path(args.out)
    best, summary = save_outputs(out_dir, results, benches, cfg, args.cost_bps)

    # ─── Print top-line table ──────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f" FABER RELATIVE STRENGTH (CRYPTO) — {summary['sample_size']['start']} → "
          f"{summary['sample_size']['end']}")
    print(f" {len(results)} variants @ {args.cost_bps:.0f}bps round-trip cost")
    print("=" * 88)

    sweep_df = pd.DataFrame([
        {"variant": r.name, **{k: r.metrics[k] for k in
            ("cagr", "vol", "sharpe", "max_dd", "calmar", "annual_turnover")}}
        for r in results
    ]).sort_values("sharpe", ascending=False)

    def _fmt(row: pd.Series) -> str:
        return (
            f"  {row['variant']:<22s}  "
            f"CAGR {row['cagr']*100:>6.1f}%  "
            f"Vol {row['vol']*100:>5.1f}%  "
            f"Sharpe {row['sharpe']:>4.2f}  "
            f"MaxDD {row['max_dd']*100:>6.1f}%  "
            f"Calmar {row['calmar']:>5.2f}  "
            f"TO {row['annual_turnover']:>4.1f}x/yr"
        )

    print("\n  TOP 10 BY SHARPE")
    for _, row in sweep_df.head(10).iterrows():
        print(_fmt(row))

    print("\n  BENCHMARKS")
    for name, eq in benches.items():
        m = annualised_metrics(eq.dropna())
        row = pd.Series({"variant": name, **m, "annual_turnover": 0.0})
        print(_fmt(row))

    print("\n  RECOMMENDED")
    rec_row = sweep_df.loc[sweep_df["variant"] == best.name].iloc[0]
    print(_fmt(rec_row))
    print("=" * 88)
    print(f"  artefacts → {out_dir}")

    # Optional cost sensitivity sweep on the recommended variant.
    if args.cost_sweep:
        print("\n  COST SENSITIVITY (recommended variant)")
        for cb in (0.0, 30.0, 60.0, 100.0, 150.0):
            r = run_variant(
                closes=closes,
                closes_monthly=closes.resample("ME").last(),
                monthly_dates=closes.resample("ME").last().index,
                eligible_monthly=build_eligibility(
                    closes, dvol, closes.resample("ME").last().index, cfg
                ),
                btc_above_sma=build_btc_regime(closes, cfg),
                lookback=best.lookback, top_n=best.top_n, hedge=best.hedge,
                cost_bps=cb, cfg=cfg,
            )
            m = r.metrics
            print(f"  {cb:>5.0f} bps :  CAGR {m['cagr']*100:>6.1f}%  "
                  f"Sharpe {m['sharpe']:>4.2f}  MaxDD {m['max_dd']*100:>6.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
