#!/usr/bin/env python
"""Phase A: Alpha decay analysis.

Based on: "Not All Factors Crowd Equally" (Chorok Lee, 2025)

Computes factor returns for six reference cross-sectional factors on crypto,
measures rolling Sharpe decay, estimates half-lives, and produces a
priority-ranked research queue for Phases B and C.

Data source: DuckDB ``bars_1d`` view via ``scripts/research/common/data``.
Outputs:
    artifacts/research/alpha_decay/decay_report.md
    artifacts/research/alpha_decay/research_queue.json
    artifacts/research/alpha_decay/plots/<factor>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.common.data import (
    ANN_FACTOR,
    load_daily_bars,
    filter_universe,
)
from scripts.research.common.metrics import compute_metrics

SEED = 42
np.random.seed(SEED)

MIN_HISTORY_DAYS = 365
MIN_ASSETS_FOR_QUINTILE = 5
WINDOW_DAYS = 90
QUINTILE_PCT = 0.20

OUT_DIR = Path("artifacts/research/alpha_decay")
LOG_PATH = Path("artifacts/research/run_log.txt")

# ---------------------------------------------------------------------------
# Factor definitions
# ---------------------------------------------------------------------------
FACTOR_DEFS: dict[str, dict] = {
    "MOM_1M": {
        "description": "1-month cross-sectional momentum (21d return, ranked)",
        "lookback": 21,
        "field": "return",
        "flip": False,
    },
    "MOM_3M": {
        "description": "3-month cross-sectional momentum (63d return, ranked)",
        "lookback": 63,
        "field": "return",
        "flip": False,
    },
    "MOM_12M": {
        "description": "12-month momentum ex last month (252-21 day return, ranked)",
        "lookback": 252,
        "skip_recent": 21,
        "field": "return",
        "flip": False,
    },
    "REV_1W": {
        "description": "1-week short-term reversal (5d return, ranked, sign-flipped)",
        "lookback": 5,
        "field": "return",
        "flip": True,
    },
    "VOL_LT": {
        "description": "Low-volatility factor (20d realized vol, ranked, sign-flipped)",
        "lookback": 20,
        "field": "volatility",
        "flip": True,
    },
    "VOL_RL": {
        "description": "Volume-relative factor (5d avg vol / 60d avg vol, ranked)",
        "lookback_fast": 5,
        "lookback_slow": 60,
        "field": "volume_ratio",
        "flip": False,
    },
}


# ---------------------------------------------------------------------------
# A1. Load and prepare data
# ---------------------------------------------------------------------------
def _load_from_table(db_path: str, table: str, start: str, end: str) -> pd.DataFrame:
    """Load data directly from a named table (fast path for pre-materialized data)."""
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM {table}
            WHERE ts >= ? AND ts <= ?
              AND open > 0 AND close > 0
            ORDER BY ts, symbol
            """,
            [start, end],
        ).fetch_df()
    finally:
        con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    return df


def load_and_prepare(
    db_path: str,
    start: str,
    end: str,
    table: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load daily bars, apply universe filter, return close and returns pivots."""
    log(f"A1. Loading daily bars from {db_path} ({start} to {end})")

    if table:
        log(f"A1. Using pre-materialized table: {table}")
        panel = _load_from_table(db_path, table, start, end)
        panel = filter_universe(panel, min_adv_usd=500_000, min_history_days=MIN_HISTORY_DAYS)
    else:
        panel = load_daily_bars(db_path=db_path, start=start, end=end)
        panel = filter_universe(panel, min_adv_usd=1_000_000, min_history_days=MIN_HISTORY_DAYS)

    # Keep only in-universe rows
    panel = panel[panel["in_universe"]].copy()

    n_assets = panel["symbol"].nunique()
    date_range = f"{panel['ts'].min().date()} to {panel['ts'].max().date()}"
    log(f"A1. Universe: {n_assets} assets, {date_range}")

    if n_assets < MIN_ASSETS_FOR_QUINTILE:
        raise RuntimeError(
            f"Need >= {MIN_ASSETS_FOR_QUINTILE} assets for quintile sort, got {n_assets}"
        )

    close_wide = panel.pivot(index="ts", columns="symbol", values="close")
    volume_wide = panel.pivot(index="ts", columns="symbol", values="volume")
    returns_wide = close_wide.pct_change()

    return close_wide, returns_wide, volume_wide


# ---------------------------------------------------------------------------
# A2. Compute factor signals and long-short returns
# ---------------------------------------------------------------------------
def compute_factor_signal(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    """Compute cross-sectional factor signal (ranked scores) for a given factor.

    Returns a DataFrame of ranks (0=worst, 1=best) with same shape as close.
    """
    fdef = FACTOR_DEFS[factor_name]
    field = fdef["field"]

    if field == "return":
        lookback = fdef["lookback"]
        skip = fdef.get("skip_recent", 0)
        if skip > 0:
            raw = close.pct_change(lookback) - close.pct_change(skip)
        else:
            raw = close.pct_change(lookback)
    elif field == "volatility":
        lookback = fdef["lookback"]
        raw = returns.rolling(lookback, min_periods=lookback).std()
    elif field == "volume_ratio":
        fast = fdef["lookback_fast"]
        slow = fdef["lookback_slow"]
        vol_fast = volume.rolling(fast, min_periods=fast).mean()
        vol_slow = volume.rolling(slow, min_periods=slow).mean()
        raw = vol_fast / vol_slow.replace(0, np.nan)
    else:
        raise ValueError(f"Unknown field: {field}")

    if fdef.get("flip", False):
        raw = -raw

    # Cross-sectional rank each day (0 = worst, 1 = best)
    ranked = raw.rank(axis=1, pct=True)
    return ranked


def compute_long_short_returns(
    signal_ranks: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Construct daily long-short return from quintile-sorted signal ranks.

    Long top quintile, short bottom quintile, equal-weighted within each leg.
    """
    # Shift signal by 1 day (signal at close t, earn return at t+1)
    sig = signal_ranks.shift(1)
    ret = returns

    common = sig.index.intersection(ret.index).sort_values()
    sig = sig.reindex(common)
    ret = ret.reindex(common)

    ls_returns = []
    for ts in common:
        ranks = sig.loc[ts].dropna()
        rets = ret.loc[ts].dropna()
        common_syms = ranks.index.intersection(rets.index)
        if len(common_syms) < MIN_ASSETS_FOR_QUINTILE:
            ls_returns.append(np.nan)
            continue
        ranks = ranks[common_syms]
        rets = rets[common_syms]

        top = ranks >= (1.0 - QUINTILE_PCT)
        bottom = ranks <= QUINTILE_PCT

        n_long = top.sum()
        n_short = bottom.sum()
        if n_long == 0 or n_short == 0:
            ls_returns.append(np.nan)
            continue

        long_ret = rets[top].mean()
        short_ret = rets[bottom].mean()
        ls_returns.append(long_ret - short_ret)

    return pd.Series(ls_returns, index=common, name="ls_return")


# ---------------------------------------------------------------------------
# A3. Measure alpha decay
# ---------------------------------------------------------------------------
def rolling_window_sharpe(
    ls_returns: pd.Series,
    window: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Compute Sharpe ratio in non-overlapping windows."""
    ls = ls_returns.dropna()
    n = len(ls)
    windows = []
    for start_idx in range(0, n - window + 1, window):
        chunk = ls.iloc[start_idx:start_idx + window]
        mu = chunk.mean()
        sigma = chunk.std()
        sr = (mu / sigma) * np.sqrt(ANN_FACTOR) if sigma > 1e-12 else np.nan
        mid_date = chunk.index[len(chunk) // 2]
        windows.append({
            "window_idx": len(windows),
            "mid_date": mid_date,
            "sharpe": sr,
            "mean_ret": mu,
            "vol": sigma,
        })
    return pd.DataFrame(windows)


def fit_decay(rolling_sharpes: pd.DataFrame) -> dict:
    """Fit linear and exponential decay models to rolling Sharpe series."""
    df = rolling_sharpes.dropna(subset=["sharpe"])
    if len(df) < 3:
        return {
            "linear_slope": np.nan,
            "decay_label": "INSUFFICIENT_DATA",
            "half_life_days": np.nan,
        }

    x = df["window_idx"].values.astype(float)
    y = df["sharpe"].values

    # Linear fit: Sharpe ~ a + b * window_idx
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]

    if slope < -0.02:
        decay_label = "DECAYING"
    elif slope > 0.02:
        decay_label = "STRENGTHENING"
    else:
        decay_label = "STABLE"

    # Exponential decay half-life
    half_life = np.nan
    if decay_label == "DECAYING" and y[0] > 0:
        try:
            def exp_decay(t, a, lam):
                return a * np.exp(-lam * t)
            popt, _ = curve_fit(
                exp_decay, x, np.maximum(y, 0.01),
                p0=[y[0], 0.05], maxfev=2000,
            )
            lam = popt[1]
            if lam > 0:
                half_life_windows = np.log(2) / lam
                half_life = half_life_windows * WINDOW_DAYS
        except (RuntimeError, ValueError):
            pass
    elif decay_label in ("STABLE", "STRENGTHENING"):
        half_life = np.inf

    return {
        "linear_slope": float(slope),
        "decay_label": decay_label,
        "half_life_days": float(half_life) if np.isfinite(half_life) else None,
    }


def compute_crowding_proxy(signal_ranks: pd.DataFrame, lookback_12m: int = 252) -> dict:
    """Measure crowding as trend in cross-asset rank correlation."""
    # Average pairwise correlation of signal ranks, rolling
    window = 63  # quarterly
    n = len(signal_ranks)
    if n < lookback_12m:
        return {"crowding_trend": "INSUFFICIENT_DATA", "corr_change_12m": np.nan}

    recent = signal_ranks.iloc[-lookback_12m:]

    def _avg_corr(chunk: pd.DataFrame) -> float:
        corr = chunk.dropna(axis=1, how="all").corr()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        return float(corr.values[mask].mean()) if mask.sum() > 0 else np.nan

    first_q = recent.iloc[:window]
    last_q = recent.iloc[-window:]
    corr_start = _avg_corr(first_q)
    corr_end = _avg_corr(last_q)
    delta = corr_end - corr_start

    if delta > 0.05:
        trend = "INCREASING"
    elif delta < -0.05:
        trend = "DECREASING"
    else:
        trend = "STABLE"

    return {"crowding_trend": trend, "corr_change_12m": float(delta)}


# ---------------------------------------------------------------------------
# A4. Reports and outputs
# ---------------------------------------------------------------------------
def assign_priority(
    full_sharpe: float,
    recent_sharpe: float,
    decay_label: str,
    half_life: float | None,
    crowding_trend: str,
) -> str:
    """Assign research priority based on decay analysis."""
    hl = half_life if half_life is not None else np.inf

    if full_sharpe > 0.5 and decay_label in ("STABLE", "STRENGTHENING") \
            and crowding_trend in ("STABLE", "DECREASING"):
        return "HIGH"
    if full_sharpe > 0.3 and hl > 180:
        return "MEDIUM"
    if full_sharpe < 0.1 or (0 < hl < 60) or crowding_trend == "INCREASING":
        return "AVOID"
    if 0.1 <= full_sharpe <= 0.3 or 60 <= hl <= 180:
        return "LOW"
    return "MEDIUM"


def plot_factor(
    factor_name: str,
    ls_equity: pd.Series,
    rolling_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Save equity curve and rolling Sharpe plot for a factor."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Equity curve
    eq = (1 + ls_equity.dropna()).cumprod()
    ax1.plot(eq.index, eq.values, linewidth=1.2)
    ax1.set_title(f"{factor_name} — Long-Short Equity Curve")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, alpha=0.3)

    # Rolling Sharpe
    if len(rolling_df) > 0:
        ax2.bar(rolling_df["mid_date"], rolling_df["sharpe"], width=60, alpha=0.7)
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.set_title(f"{factor_name} — {WINDOW_DAYS}d Rolling Sharpe")
        ax2.set_ylabel("Sharpe Ratio (annualized)")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "plots" / f"{factor_name}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_decay_report(results: list[dict], out_dir: Path) -> Path:
    """Write decay_report.md."""
    lines = [
        "# Alpha Decay Report",
        f"Generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC",
        "",
    ]

    for r in results:
        hl_str = f"{r['half_life_days']:.0f} days" if r["half_life_days"] else "STABLE"
        lines.extend([
            f"## FACTOR: {r['factor']}",
            f"_{r['description']}_",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Sharpe (full period) | {r['sharpe_full']:.3f} |",
            f"| Sharpe (last 90d) | {r['sharpe_recent']:.3f} |",
            f"| Decay slope | {r['linear_slope']:.4f} ({r['decay_label']}) |",
            f"| Estimated half-life | {hl_str} |",
            f"| Crowding trend (12m) | {r['crowding_trend']} |",
            f"| **Research priority** | **{r['priority']}** |",
            "",
        ])

    report_path = out_dir / "decay_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def write_research_queue(results: list[dict], out_dir: Path) -> Path:
    """Write research_queue.json ranked by priority."""
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "AVOID": 3}
    ranked = sorted(results, key=lambda r: (priority_order.get(r["priority"], 9), -r["sharpe_full"]))

    queue = []
    for r in ranked:
        queue.append({
            "factor": r["factor"],
            "priority": r["priority"],
            "sharpe_full": round(r["sharpe_full"], 4),
            "sharpe_recent": round(r["sharpe_recent"], 4),
            "decay_label": r["decay_label"],
            "half_life_days": round(r["half_life_days"], 1) if r["half_life_days"] else None,
            "crowding_trend": r["crowding_trend"],
        })

    queue_path = out_dir / "research_queue.json"
    queue_path.write_text(json.dumps(queue, indent=2))
    return queue_path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    """Append to run log and print."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A: Alpha Decay Analysis")
    parser.add_argument("--db", type=str, default=None, help="DuckDB path (default: auto-detect)")
    parser.add_argument("--table", type=str, default=None,
                        help="Pre-materialized table name (e.g. bars_1d_usd_universe_clean)")
    parser.add_argument("--start", type=str, default="2017-01-01")
    parser.add_argument("--end", type=str, default="2026-12-31")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.db
    if db_path is None:
        candidates = [
            PROJECT_ROOT / ".." / "data" / "market.duckdb",
            PROJECT_ROOT / ".." / "data" / "coinbase_daily_121025.duckdb",
        ]
        for c in candidates:
            if c.resolve().exists():
                db_path = str(c.resolve())
                break
        if db_path is None:
            print("ERROR: No DuckDB file found. Pass --db explicitly.")
            sys.exit(1)

    n_factors = len(FACTOR_DEFS)
    log(f"PHASE A STARTING — {n_factors} factors, data {args.start} to {args.end}")

    # A1: Load data
    try:
        close, returns, volume = load_and_prepare(db_path, args.start, args.end, table=args.table)
    except Exception as e:
        fail_path = out_dir / "FAILED.md"
        fail_path.write_text(f"# Phase A Failed\n\nData load error: {e}\n")
        log(f"PHASE A FAILED — {e}")
        return

    n_assets = close.shape[1]
    date_range = f"{close.index.min().date()} to {close.index.max().date()}"

    # A2 + A3: Factor loop
    results: list[dict] = []
    for factor_name, fdef in FACTOR_DEFS.items():
        log(f"A2. Computing factor: {factor_name}")

        signal_ranks = compute_factor_signal(close, returns, volume, factor_name)
        ls_returns = compute_long_short_returns(signal_ranks, returns)

        full_metrics = compute_metrics((1 + ls_returns.dropna()).cumprod())
        sharpe_full = full_metrics.get("sharpe", np.nan)

        # Recent 90d Sharpe
        recent = ls_returns.dropna().iloc[-WINDOW_DAYS:]
        if len(recent) >= 30:
            recent_eq = (1 + recent).cumprod()
            recent_metrics = compute_metrics(recent_eq)
            sharpe_recent = recent_metrics.get("sharpe", np.nan)
        else:
            sharpe_recent = np.nan

        # Decay analysis
        log(f"A3. Decay analysis: {factor_name}")
        rolling_df = rolling_window_sharpe(ls_returns)
        decay = fit_decay(rolling_df)

        # Crowding proxy
        crowding = compute_crowding_proxy(signal_ranks)

        priority = assign_priority(
            sharpe_full, sharpe_recent,
            decay["decay_label"],
            decay["half_life_days"],
            crowding["crowding_trend"],
        )

        results.append({
            "factor": factor_name,
            "description": fdef["description"],
            "sharpe_full": sharpe_full if np.isfinite(sharpe_full) else 0.0,
            "sharpe_recent": sharpe_recent if np.isfinite(sharpe_recent) else 0.0,
            "linear_slope": decay["linear_slope"],
            "decay_label": decay["decay_label"],
            "half_life_days": decay["half_life_days"],
            "crowding_trend": crowding["crowding_trend"],
            "corr_change_12m": crowding["corr_change_12m"],
            "priority": priority,
            "full_metrics": full_metrics,
        })

        # Plot
        plot_factor(factor_name, ls_returns, rolling_df, out_dir)

        log(
            f"  {factor_name}: Sharpe={sharpe_full:.3f}, "
            f"decay={decay['decay_label']}, priority={priority}"
        )

    # A4: Reports
    log("A4. Writing reports")
    report_path = write_decay_report(results, out_dir)
    queue_path = write_research_queue(results, out_dir)

    paths = f"{report_path}, {queue_path}"
    log(f"PHASE A COMPLETE — {n_assets} assets, {date_range}, "
        f"{n_factors} factors — outputs written to {paths}")


if __name__ == "__main__":
    main()
