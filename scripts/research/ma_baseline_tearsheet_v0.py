#!/usr/bin/env python
"""MA(5/40) baseline tear sheet - thin wrapper over canonical builder."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tearsheet_common_v0 import (
    build_standard_tearsheet,
    compute_stats,
    get_default_benchmark_equity,
    load_equity_csv,
    load_strategy_stats_from_metrics,
)

# Re-export for testing
__all__ = [
    'build_standard_tearsheet',
    'compute_stats',
    'get_default_benchmark_equity',
    'load_equity_csv',
    'load_strategy_stats_from_metrics',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PDF tear sheet for MA(5/40) baseline.")
    p.add_argument("--research_dir", type=str, required=True, help="Directory containing equity/metrics artifacts.")
    p.add_argument("--equity_csv", type=str, default=None, help="Explicit equity CSV path.")
    p.add_argument("--metrics_csv", type=str, default=None, help="Optional metrics CSV path.")
    p.add_argument("--benchmark_equity_csv", type=str, default=None, help="Optional benchmark equity CSV.")
    p.add_argument("--benchmark_label", type=str, default="BTC Buy & Hold", help="Benchmark label.")
    p.add_argument("--no-benchmark", action="store_true", help="Disable benchmark overlay.")
    p.add_argument("--out_pdf", type=str, required=True, help="Output PDF path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    research_dir = Path(args.research_dir)
    
    # Resolve inputs
    equity_path = Path(args.equity_csv) if args.equity_csv else research_dir / "equity.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"Equity CSV not found: {equity_path}")
    
    metrics_path = Path(args.metrics_csv) if args.metrics_csv else research_dir / "metrics_ma_5_40_baseline_v0.csv"
    if not metrics_path.exists():
        # Try to load strategy_stats from equity if metrics missing
        print(f"[ma_baseline_tearsheet_v0] Metrics CSV not found: {metrics_path}; computing stats from equity.")
        strat_eq = load_equity_csv(str(equity_path))
        strategy_stats = compute_stats(strat_eq)
    else:
        strat_eq = load_equity_csv(str(equity_path))
        strategy_stats = load_strategy_stats_from_metrics(str(metrics_path))
    
    # Load benchmark
    benchmark_eq = None
    benchmark_label = None
    if not args.no_benchmark:
        benchmark_eq, benchmark_label = get_default_benchmark_equity(
            strategy_index=strat_eq.index,
            research_dir=str(research_dir),
            benchmark_equity_csv=args.benchmark_equity_csv,
            benchmark_label=args.benchmark_label,
            default_symbol="BTC-USD",
        )
    
    # Resolve manifest
    manifest_path = research_dir / "manifest.json"
    manifest_path_str = str(manifest_path) if manifest_path.exists() else None
    
    # Build tear sheet using canonical builder
    build_standard_tearsheet(
        out_pdf=args.out_pdf,
        strategy_label="MA(5/40) Baseline",
        strategy_equity=strat_eq,
        strategy_stats=strategy_stats,
        benchmark_equity=benchmark_eq,
        benchmark_label=benchmark_label,
        equity_csv_path=str(equity_path),
        metrics_csv_path=str(metrics_path),
        manifest_path=manifest_path_str,
        subtitle="Daily MA(5/40) crossover long-only baseline",
    )
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
