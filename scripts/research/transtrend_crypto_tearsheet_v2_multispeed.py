#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tearsheet_common_v0 import (
    build_standard_tearsheet,
    compute_stats,
    get_default_benchmark_equity,
    load_equity_csv,
    load_strategy_stats_from_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transtrend Crypto v2 multispeed tearsheet")
    p.add_argument("--research_dir", required=True, help="Research directory")
    p.add_argument("--equity_csv", default=None, help="Equity CSV path")
    p.add_argument("--metrics_csv", default=None, help="Metrics CSV path")
    p.add_argument("--benchmark_equity_csv", default=None, help="Benchmark equity CSV")
    p.add_argument("--benchmark_label", default="BTC Buy & Hold", help="Benchmark label")
    p.add_argument("--no-benchmark", action="store_true", help="Disable benchmark overlay")
    p.add_argument("--out_pdf", required=True, help="Output PDF path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    research_dir = Path(args.research_dir)

    equity_path = Path(args.equity_csv) if args.equity_csv else research_dir / "combined" / "equity.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"Equity CSV not found: {equity_path}")

    metrics_path = Path(args.metrics_csv) if args.metrics_csv else research_dir / "combined" / "metrics_transtrend_crypto_v2.csv"
    if not metrics_path.exists():
        print(f"[transtrend_crypto_tearsheet_v2] Metrics CSV not found: {metrics_path}; computing stats from equity.")
        strat_eq = load_equity_csv(str(equity_path))
        strategy_stats = compute_stats(strat_eq)
    else:
        strat_eq = load_equity_csv(str(equity_path))
        strategy_stats = load_strategy_stats_from_metrics(str(metrics_path))

    # Normalize to daily dates if index is daily at a fixed hour (e.g., 16:00:00)
    if isinstance(strat_eq.index, pd.DatetimeIndex):
        idx = strat_eq.index
        hours = set(idx.hour.tolist())
        if (idx.minute == 0).all() and (idx.second == 0).all():
            if len(hours) <= 2 and hours.issubset({0, 16, 17}) and hours != {0}:
                strat_eq.index = strat_eq.index.normalize()

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

    manifest_path = research_dir / "combined" / "run_manifest.json"
    manifest_path_str = str(manifest_path) if manifest_path.exists() else None

    out_path = Path(args.out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_standard_tearsheet(
        out_pdf=args.out_pdf,
        strategy_label="Transtrend Crypto v2 (Multi-Speed, Spot Long-Only)",
        strategy_equity=strat_eq,
        strategy_stats=strategy_stats,
        benchmark_equity=benchmark_eq,
        benchmark_label=benchmark_label,
        equity_csv_path=str(equity_path),
        metrics_csv_path=str(metrics_path),
        manifest_path=manifest_path_str,
        subtitle=(
            "Fast sleeve (10/20) + Slow sleeve (50/200), 30/70 mix; vol target=20%; danger gross=0.25; cost=20bps"
        ),
    )


if __name__ == "__main__":
    main()
