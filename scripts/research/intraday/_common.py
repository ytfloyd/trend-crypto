"""
Shared utilities for intraday strategy research.

Provides data loading, universe filtering, signal-to-weight conversion,
risk overlays, and evaluation metrics — all calibrated for intraday
(1h) crypto trading.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import numpy as np
import pandas as pd

from common.data import load_bars, filter_universe, BARS_PER_DAY, ANN_FACTOR
from common.backtest import simple_backtest
from common.metrics import compute_metrics, format_metrics_table
from common.risk_overlays import (
    apply_vol_targeting,
    apply_dd_control,
    apply_position_limit_wide,
)

RESULTS_DIR = Path(__file__).resolve().parent / "_results"
RESULTS_DIR.mkdir(exist_ok=True)

COST_BPS = 15.0  # tighter than daily due to limit orders on 1h timeframe

# Annualisation factor for 1h bars: 24 bars/day × 365 days
ANN_FACTOR_1H = 24 * 365


def load_hourly_universe(
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    min_adv_usd: float = 5_000_000,
    min_history_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 1h bars, filter universe, return (panel, returns_wide, close_wide).

    Returns
    -------
    panel : pd.DataFrame with in_universe column
    returns_wide : pd.DataFrame (index=ts, columns=symbols)
    close_wide : pd.DataFrame (index=ts, columns=symbols)
    """
    panel = load_bars("1h", start=start, end=end)
    panel = filter_universe(
        panel, min_adv_usd=min_adv_usd, min_history_days=min_history_days, adv_window=24 * 20
    )

    # Pivot to wide format
    close_wide = panel.pivot_table(index="ts", columns="symbol", values="close")
    returns_wide = np.log(close_wide / close_wide.shift(1))
    returns_wide = returns_wide.replace([np.inf, -np.inf], np.nan)

    return panel, returns_wide, close_wide


def convexity_metrics(equity: pd.Series) -> dict:
    """Extended metrics focused on convexity and tail behaviour."""
    m = compute_metrics(equity)
    ret = equity.pct_change().dropna()

    # Gain-to-pain ratio
    gains = ret[ret > 0].sum()
    pains = ret[ret < 0].abs().sum()
    m["gain_to_pain"] = float(gains / pains) if pains > 0 else np.inf

    # Tail ratio: 95th percentile gain / 5th percentile loss
    p95 = ret.quantile(0.95)
    p05 = abs(ret.quantile(0.05))
    m["tail_ratio"] = float(p95 / p05) if p05 > 0 else np.inf

    # Omega ratio at 0
    m["omega"] = float(gains / pains) if pains > 0 else np.inf

    # Percent of time in drawdown
    dd = equity / equity.cummax() - 1
    m["pct_time_in_dd"] = float((dd < -0.01).mean())

    # Average drawdown recovery time (bars)
    dd_periods = (dd < -0.01).astype(int)
    changes = dd_periods.diff().fillna(0)
    dd_starts = changes[changes == 1].index
    dd_ends = changes[changes == -1].index
    if len(dd_starts) > 0 and len(dd_ends) > 0:
        durations = []
        for s in dd_starts:
            end_candidates = dd_ends[dd_ends > s]
            if len(end_candidates) > 0:
                durations.append((end_candidates[0] - s).total_seconds() / 3600)
        m["avg_dd_hours"] = float(np.mean(durations)) if durations else np.nan
    else:
        m["avg_dd_hours"] = 0.0

    return m


def apply_full_risk_stack(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_target: float = 0.25,
    dd_threshold: float = 0.15,
    max_position: float = 0.20,
) -> pd.DataFrame:
    """Apply the full risk overlay stack for convexity."""
    w = weights.copy()
    w = apply_position_limit_wide(w, max_wt=max_position)
    w = apply_vol_targeting(w, returns_wide, vol_target=vol_target, lookback=24*14, max_leverage=2.0)
    w = apply_dd_control(w, returns_wide, dd_threshold=dd_threshold, cost_bps=COST_BPS)
    return w


def run_and_report(
    strategy_name: str,
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
) -> dict:
    """Run backtest with and without risk overlays, report results."""
    common_cols = weights.columns.intersection(returns_wide.columns)
    w_raw = weights[common_cols]
    r = returns_wide[common_cols]

    # Raw backtest
    bt_raw = simple_backtest(w_raw, r, cost_bps=COST_BPS)
    eq_raw = bt_raw.set_index("ts")["portfolio_equity"]
    m_raw = convexity_metrics(eq_raw)
    m_raw["label"] = f"{strategy_name} (raw)"

    # With risk overlays
    w_risk = apply_full_risk_stack(w_raw, r)
    bt_risk = simple_backtest(w_risk, r, cost_bps=COST_BPS)
    eq_risk = bt_risk.set_index("ts")["portfolio_equity"]
    m_risk = convexity_metrics(eq_risk)
    m_risk["label"] = f"{strategy_name} (risk-managed)"

    print(f"\n{'='*70}")
    print(f"  {strategy_name}")
    print(f"{'='*70}")
    print(format_metrics_table([m_raw, m_risk]))

    print(f"\n  Convexity metrics ({strategy_name}, risk-managed):")
    print(f"    Skewness:        {m_risk['skewness']:>8.3f}")
    print(f"    Tail ratio:      {m_risk['tail_ratio']:>8.3f}")
    print(f"    Gain/pain:       {m_risk['gain_to_pain']:>8.3f}")
    print(f"    Calmar:          {m_risk['calmar']:>8.3f}")
    print(f"    Sortino:         {m_risk['sortino']:>8.3f}")
    print(f"    Max DD:          {m_risk['max_dd']:>8.1%}")
    print(f"    % time in DD:    {m_risk['pct_time_in_dd']:>8.1%}")

    # Save
    bt_risk.to_parquet(RESULTS_DIR / f"{strategy_name.lower().replace(' ', '_')}.parquet", index=False)

    return m_risk
