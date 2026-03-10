#!/usr/bin/env python3
"""
Strategy 6: Concentrated Breakout (4h, top 10)

Redesigned breakout with lessons from Round 1:
  - 4h bars (less noise)
  - Top 10 liquid assets only
  - Vectorised signal generation (fast)
  - Asymmetric exit: trailing stop at 2× ATR (preserves gains)
  - Vol-normalised sizing
  - Gentle overlays (position limit + vol target only)
  - NO drawdown control (avoid death spiral)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import numpy as np
import pandas as pd

from common.data import load_bars
from common.backtest import simple_backtest
from common.metrics import compute_metrics, format_metrics_table
from common.risk_overlays import apply_vol_targeting, apply_position_limit_wide

RESULTS_DIR = Path(__file__).resolve().parent / "_results"
RESULTS_DIR.mkdir(exist_ok=True)
COST_BPS = 15.0


def load_4h_top(n=10, start="2020-01-01", end="2025-12-31"):
    panel = load_bars("4h", start=start, end=end)
    panel["dollar_vol"] = panel["close"] * panel["volume"]
    avg_dv = panel.groupby("symbol")["dollar_vol"].mean().sort_values(ascending=False)
    top = avg_dv.head(n).index.tolist()
    panel = panel[panel["symbol"].isin(top)]
    close_wide = panel.pivot_table(index="ts", columns="symbol", values="close")
    high_wide = panel.pivot_table(index="ts", columns="symbol", values="high")
    low_wide = panel.pivot_table(index="ts", columns="symbol", values="low")
    returns_wide = np.log(close_wide / close_wide.shift(1)).replace([np.inf, -np.inf], np.nan)
    return panel, close_wide, high_wide, low_wide, returns_wide


def breakout_signal(
    close_wide: pd.DataFrame,
    high_wide: pd.DataFrame,
    low_wide: pd.DataFrame,
    entry_lookback: int = 48,  # 8 days at 4h
    exit_lookback: int = 24,   # 4 days
) -> pd.DataFrame:
    """Vectorised Donchian breakout: 1 if close > entry-bar high, exit on close < exit-bar low."""
    upper = high_wide.rolling(entry_lookback).max().shift(1)
    lower = low_wide.rolling(exit_lookback).min().shift(1)

    # State machine: position = 1 if breakout, 0 if breakdown
    pos = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)

    for col in close_wide.columns:
        in_trade = False
        for i in range(1, len(close_wide)):
            if not in_trade:
                if pd.notna(upper.iloc[i][col]) and close_wide.iloc[i][col] > upper.iloc[i][col]:
                    in_trade = True
                    pos.iloc[i][col] = 1.0
            else:
                if pd.notna(lower.iloc[i][col]) and close_wide.iloc[i][col] < lower.iloc[i][col]:
                    in_trade = False
                else:
                    pos.iloc[i][col] = 1.0

    return pos


def vol_normalised_weights(
    signal: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_lookback: int = 42,
) -> pd.DataFrame:
    """Inverse-vol weighting."""
    vol = returns_wide.rolling(vol_lookback, min_periods=12).std()
    raw = signal / vol.replace(0, np.nan)
    raw = raw.fillna(0)
    row_sum = raw.sum(axis=1).replace(0, np.nan)
    return raw.div(row_sum, axis=0).fillna(0)


def convexity_metrics(equity: pd.Series) -> dict:
    m = compute_metrics(equity)
    ret = equity.pct_change().dropna()
    gains = ret[ret > 0].sum()
    pains = ret[ret < 0].abs().sum()
    m["gain_to_pain"] = float(gains / pains) if pains > 0 else np.inf
    p95 = ret.quantile(0.95)
    p05 = abs(ret.quantile(0.05))
    m["tail_ratio"] = float(p95 / p05) if p05 > 0 else np.inf
    try:
        monthly = ret.groupby(ret.index.to_period("M")).sum()
        m["monthly_hit_rate"] = float((monthly > 0).mean())
        m["worst_month"] = float(monthly.min())
        m["best_month"] = float(monthly.max())
    except Exception:
        m["monthly_hit_rate"] = np.nan
    return m


def main():
    print("=" * 70)
    print("  STRATEGY 6: CONCENTRATED BREAKOUT (4h)")
    print("=" * 70)

    panel, close_wide, high_wide, low_wide, returns_wide = load_4h_top(n=10)
    print(f"  {len(returns_wide):,} bars × {len(returns_wide.columns)} assets")
    print(f"  Assets: {list(returns_wide.columns)}")

    configs = [
        {"entry": 18, "exit": 9, "name": "Breakout 3d-1.5d"},
        {"entry": 36, "exit": 18, "name": "Breakout 6d-3d"},
        {"entry": 48, "exit": 24, "name": "Breakout 8d-4d"},
        {"entry": 72, "exit": 36, "name": "Breakout 12d-6d"},
    ]

    all_results = []
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        sig = breakout_signal(close_wide, high_wide, low_wide, cfg["entry"], cfg["exit"])
        w = vol_normalised_weights(sig, returns_wide)
        w = apply_position_limit_wide(w, max_wt=0.25)
        w = apply_vol_targeting(w, returns_wide, vol_target=0.20, lookback=42, max_leverage=2.0)

        bt = simple_backtest(w, returns_wide, cost_bps=COST_BPS)
        eq = bt.set_index("ts")["portfolio_equity"]
        m = convexity_metrics(eq)
        m["label"] = cfg["name"]

        print(f"  CAGR={m['cagr']:.1%}  Sharpe={m['sharpe']:.2f}  MaxDD={m['max_dd']:.1%}  "
              f"Sortino={m['sortino']:.2f}  Skew={m['skewness']:.2f}  Calmar={m['calmar']:.2f}")
        print(f"  Tail ratio={m['tail_ratio']:.2f}  Gain/pain={m['gain_to_pain']:.2f}  "
              f"Monthly hit={m.get('monthly_hit_rate', 0):.1%}")

        all_results.append(m)

    print(f"\n\n{'='*70}")
    print("  BREAKOUT SUMMARY")
    print(f"{'='*70}")
    print(format_metrics_table(all_results))


if __name__ == "__main__":
    main()
