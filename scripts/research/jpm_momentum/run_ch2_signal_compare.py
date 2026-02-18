#!/usr/bin/env python
"""
Chapter 2: Signal Type Comparison.

For a fixed lookback (default 21d), compares all six signal types
in both absolute and relative mode.  Produces side-by-side metrics
and correlation analysis between signals.

Usage:
    python -m scripts.research.jpm_momentum.run_ch2_signal_compare [--lookback 21]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SIGNAL_TYPES
from .data import load_daily_bars, compute_returns_wide
from .signals import compute_signal
from .universe import filter_universe
from .weights import equal_weight
from .backtest import simple_backtest
from .metrics import compute_metrics, format_metrics_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ch2: Signal type comparison")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--out-dir", default="output/jpm_momentum/ch2_signal_compare")
    p.add_argument("--lookback", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=20.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Ch2: Signal Comparison (L={args.lookback}) ===")
    panel = load_daily_bars(start=args.start, end=args.end)
    panel_u = filter_universe(panel)
    returns_wide = compute_returns_wide(panel)

    results = []
    equity_curves = {}

    for sig_type in SIGNAL_TYPES:
        label = f"{sig_type}_L{args.lookback}"
        sig_panel = compute_signal(panel_u, signal_type=sig_type, lookback=args.lookback)

        # Absolute mode
        selected = sig_panel[sig_panel["in_universe"]].copy()
        selected["selected"] = selected["signal"] > 0
        mask = selected.pivot(index="ts", columns="symbol", values="selected").fillna(False)
        w = equal_weight(mask)
        bt = simple_backtest(w, returns_wide, cost_bps=args.cost_bps)

        eq = pd.Series(bt["portfolio_equity"].values, index=pd.to_datetime(bt["ts"]))
        m = compute_metrics(eq)
        m["label"] = f"{label}_abs"
        m["signal_type"] = sig_type
        m["mode"] = "absolute"
        results.append(m)
        equity_curves[f"{label}_abs"] = eq

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "signal_comparison.csv", index=False)
    print(f"\n{format_metrics_table(results)}")

    # Signal correlation matrix
    sig_rets = pd.DataFrame({k: v.pct_change() for k, v in equity_curves.items()})
    corr = sig_rets.corr()
    corr.to_csv(out_dir / "signal_correlation.csv")
    print(f"\nSignal return correlations:\n{corr.round(2)}")


if __name__ == "__main__":
    main()
