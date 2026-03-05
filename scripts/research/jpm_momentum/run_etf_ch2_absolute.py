#!/usr/bin/env python
"""
ETF — Chapter 2: Absolute Momentum (TSMOM) Factor Scan.

Scans all signal types × lookbacks in absolute-momentum mode
across the curated ETF universe (Tiingo data).

Usage:
    python -m scripts.research.jpm_momentum.run_etf_ch2_absolute [--start 2006-01-01]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_LOOKBACKS, SIGNAL_TYPES
from .data import load_etf_daily_bars, ANN_FACTOR_ETF, compute_benchmark, compute_returns_wide
from .grid import run_grid
from .metrics import compute_metrics, format_metrics_table
from scripts.research.etf_data.universe import get_core_universe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF Ch2: Absolute momentum factor scan")
    p.add_argument("--start", default="2006-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--out-dir", default="output/jpm_momentum_etf/ch2_absolute")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--universe", choices=["full", "core"], default="core")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.universe == "core":
        from scripts.research.etf_data.universe import get_core_universe
        tickers = get_core_universe()
    else:
        from scripts.research.etf_data.universe import get_full_universe
        tickers = get_full_universe()

    print("=== ETF Ch2: Absolute Momentum Factor Scan ===")
    panel = load_etf_daily_bars(start=args.start, end=args.end, tickers=tickers)
    print(f"Loaded {len(panel):,} rows, {panel['symbol'].nunique()} ETFs")

    results = run_grid(
        panel,
        signal_types=SIGNAL_TYPES,
        lookbacks=DEFAULT_LOOKBACKS,
        weight_methods=("equal", "inv_vol"),
        cost_bps=args.cost_bps,
        min_adv_usd=5_000_000,
        min_history_days=252,
        mode="absolute",
        ann_factor=ANN_FACTOR_ETF,
        return_method="close_to_close",
    )

    results.to_csv(out_dir / "grid_results.csv", index=False)
    print(f"\nResults saved to {out_dir / 'grid_results.csv'}")

    # SPY benchmark
    try:
        spy_eq = compute_benchmark(panel, ticker="SPY")
        spy_m = compute_metrics(spy_eq, ann_factor=ANN_FACTOR_ETF)
        spy_m["label"] = "SPY_buy_hold"
        results_with_bench = pd.concat([results, pd.DataFrame([spy_m])], ignore_index=True)
    except ValueError:
        results_with_bench = results

    print(f"\n{format_metrics_table(results_with_bench.to_dict('records'))}")


if __name__ == "__main__":
    main()
