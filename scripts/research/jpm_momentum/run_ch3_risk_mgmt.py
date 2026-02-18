#!/usr/bin/env python
"""
Chapter 3: Risk Management Overlays.

Tests each risk overlay independently on the best-performing baseline
from Chapter 2 (default: RET_L21 absolute, inv_vol weighted).

Overlays tested:
1. Volatility targeting (20% target)
2. Trailing stop-loss (15%)
3. Trailing stop + time-based re-entry (5-bar wait)
4. Mean reversion overlay (5d window, 2σ)
5. Vol filter (reduce at 2× median vol)
6. Combined: vol target + trailing stop + mean reversion

Usage:
    python -m scripts.research.jpm_momentum.run_ch3_risk_mgmt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import load_daily_bars, compute_returns_wide, compute_close_wide
from .signals import compute_signal
from .universe import filter_universe
from .weights import inverse_volatility
from .backtest import simple_backtest
from .risk import (
    apply_vol_targeting,
    apply_trailing_stop,
    apply_mean_reversion_overlay,
    apply_vol_filter,
)
from .metrics import compute_metrics, format_metrics_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ch3: Risk management overlays")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--out-dir", default="output/jpm_momentum/ch3_risk_mgmt")
    p.add_argument("--cost-bps", type=float, default=20.0)
    p.add_argument("--signal-type", default="RET")
    p.add_argument("--lookback", type=int, default=21)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Ch3: Risk Management Overlays ===")
    panel = load_daily_bars(start=args.start, end=args.end)
    panel_u = filter_universe(panel)
    returns_wide = compute_returns_wide(panel)
    close_wide = compute_close_wide(panel)

    # Base signal
    sig_panel = compute_signal(panel_u, signal_type=args.signal_type, lookback=args.lookback)
    selected = sig_panel[sig_panel["in_universe"]].copy()
    selected["selected"] = selected["signal"] > 0
    mask = selected.pivot(index="ts", columns="symbol", values="selected").fillna(False)
    w_base = inverse_volatility(mask, returns_wide)

    overlays = {
        "baseline": w_base,
        "vol_target_20pct": apply_vol_targeting(w_base, returns_wide, vol_target=0.20),
        "trailing_stop_15pct": apply_trailing_stop(w_base, close_wide, stop_pct=0.15),
        "stop_15pct_reentry_5": apply_trailing_stop(w_base, close_wide, stop_pct=0.15, reentry_bars=5),
        "mean_revert_5d_2sig": apply_mean_reversion_overlay(w_base, returns_wide, window=5, threshold_sigma=2.0),
        "vol_filter_2x": apply_vol_filter(w_base, returns_wide, multiplier=2.0),
        "combined": apply_vol_targeting(
            apply_mean_reversion_overlay(
                apply_trailing_stop(w_base, close_wide, stop_pct=0.15),
                returns_wide,
            ),
            returns_wide,
        ),
    }

    results = []
    for label, w in overlays.items():
        bt = simple_backtest(w, returns_wide, cost_bps=args.cost_bps)
        eq = pd.Series(bt["portfolio_equity"].values, index=pd.to_datetime(bt["ts"]))
        m = compute_metrics(eq)
        m["label"] = label
        m["avg_turnover"] = float(bt["turnover"].mean())
        results.append(m)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "risk_overlay_results.csv", index=False)
    print(f"\n{format_metrics_table(results)}")


if __name__ == "__main__":
    main()
