#!/usr/bin/env python
"""
Chapter 2: Relative / Cross-Sectional Momentum (XSMOM).

Ranks all eligible assets by signal strength, goes long the top-K
(or top quintile).  No short leg — long-only.

Usage:
    python -m scripts.research.jpm_momentum.run_ch2_relative [--start 2017-01-01] [--end 2026-12-31]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_LOOKBACKS, SIGNAL_TYPES
from .data import load_daily_bars
from .grid import run_grid
from .metrics import format_metrics_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ch2: Relative momentum factor scan")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--out-dir", default="output/jpm_momentum/ch2_relative")
    p.add_argument("--cost-bps", type=float, default=20.0)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Ch2: Relative Momentum Factor Scan ===")
    panel = load_daily_bars(start=args.start, end=args.end)
    print(f"Loaded {len(panel):,} rows, {panel['symbol'].nunique()} symbols")

    results = run_grid(
        panel,
        signal_types=SIGNAL_TYPES,
        lookbacks=DEFAULT_LOOKBACKS,
        weight_methods=("equal", "inv_vol"),
        cost_bps=args.cost_bps,
        mode="relative",
        top_k=args.top_k,
    )

    results.to_csv(out_dir / "grid_results.csv", index=False)
    print(f"\nResults saved to {out_dir / 'grid_results.csv'}")
    print(f"\n{format_metrics_table(results.to_dict('records'))}")


if __name__ == "__main__":
    main()
