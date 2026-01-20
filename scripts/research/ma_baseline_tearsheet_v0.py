#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tearsheet_common_v0 import (
    build_benchmark_comparison_table,
    build_provenance_text,
    compute_drawdown,
    compute_stats,
    get_default_benchmark_equity,
    load_equity_csv,
    scale_equity_to_start,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PDF tear sheet for MA(5/40) baseline.")
    p.add_argument(
        "--research_dir",
        type=str,
        required=True,
        help="Directory containing equity/metrics artifacts for the baseline.",
    )
    p.add_argument("--equity_csv", type=str, default=None, help="Explicit equity CSV path.")
    p.add_argument("--metrics_csv", type=str, default=None, help="Optional metrics CSV path.")
    p.add_argument("--benchmark_equity_csv", type=str, default=None, help="Optional benchmark equity CSV.")
    p.add_argument("--benchmark_label", type=str, default="BTC Buy & Hold", help="Benchmark label.")
    p.add_argument("--no-benchmark", action="store_true", help="Disable benchmark overlay.")
    p.add_argument("--out_pdf", type=str, required=True, help="Output PDF path.")
    return p.parse_args()


def _resolve_path(research_dir: Path, explicit: Optional[str], filename: str) -> Path:
    if explicit:
        return Path(explicit)
    return research_dir / filename


def make_tearsheet(
    research_dir: Path,
    out_pdf: Path,
    equity_csv: Optional[str] = None,
    metrics_csv: Optional[str] = None,
    benchmark_equity_csv: Optional[str] = None,
    benchmark_label: str = "BTC Buy & Hold",
    no_benchmark: bool = False,
) -> None:
    equity_path = _resolve_path(research_dir, equity_csv, "equity.csv")
    if not equity_path.exists():
        raise FileNotFoundError(f"Equity CSV not found: {equity_path}")

    metrics_path = _resolve_path(research_dir, metrics_csv, "metrics_ma_5_40_baseline_v0.csv")
    benchmark_path = Path(benchmark_equity_csv) if benchmark_equity_csv else None

    strat_eq = load_equity_csv(str(equity_path))
    strat_stats = compute_stats(strat_eq)

    benchmark_eq = None
    resolved_label: Optional[str] = None
    if not no_benchmark:
        benchmark_eq, resolved_label = get_default_benchmark_equity(
            strategy_index=strat_eq.index,
            research_dir=str(research_dir),
            benchmark_equity_csv=str(benchmark_path) if benchmark_path else None,
            benchmark_label=benchmark_label,
            default_symbol="BTC-USD",
        )

    comparison = build_benchmark_comparison_table(
        strategy_label="MA(5/40) baseline",
        strategy_stats=strat_stats,
        benchmark_label=resolved_label if benchmark_eq is not None else None,
        benchmark_eq=benchmark_eq,
    )

    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        axes[0].plot(strat_eq.index, strat_eq.values, label="MA(5/40) baseline")
        if benchmark_eq is not None:
            bench_plot = scale_equity_to_start(benchmark_eq, float(strat_eq.iloc[0]))
            axes[0].plot(bench_plot.index, bench_plot.values, label=resolved_label, linestyle="--")
        axes[0].set_title("Equity Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        dd = compute_drawdown(strat_eq)
        axes[1].plot(dd.index, dd.values, label="Drawdown")
        axes[1].axhline(0.0, color="black", linewidth=0.5)
        axes[1].set_title("Drawdown")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if not comparison.empty:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            table = ax.table(
                cellText=comparison.values.tolist(),
                colLabels=comparison.columns.tolist(),
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            pdf.savefig(fig)
            plt.close(fig)

        provenance = build_provenance_text(str(equity_path), str(metrics_path))
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.02, 0.98, provenance, va="top", ha="left", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    make_tearsheet(
        research_dir=Path(args.research_dir),
        out_pdf=Path(args.out_pdf),
        equity_csv=args.equity_csv,
        metrics_csv=args.metrics_csv,
        benchmark_equity_csv=args.benchmark_equity_csv,
        benchmark_label=args.benchmark_label,
        no_benchmark=args.no_benchmark,
    )
    print(f"Wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
