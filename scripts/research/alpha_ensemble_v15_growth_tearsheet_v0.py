#!/usr/bin/env python
from __future__ import annotations

"""
Tear sheet for Alpha Ensemble V1.5 Growth Sleeve (daily-only v0).

Uses shared helpers in tearsheet_common_v0.py for:
- Loading equity/benchmark
- Stats from metrics CSV
- Overlay plots and comparison table
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from tearsheet_common_v0 import (  # noqa: E402
    add_benchmark_summary_table,
    build_benchmark_comparison_table,
    compute_stats,
    load_benchmark_equity,
    load_equity_csv,
    load_strategy_stats_from_metrics,
    plot_drawdown_with_benchmark,
    plot_equity_with_benchmark,
    resolve_tearsheet_inputs,
    build_provenance_text,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha Ensemble V1.5 Growth Sleeve tear sheet (daily-only v0).")
    p.add_argument("--research_dir", required=False, help="Directory containing growth_equity_v0.csv, etc.")
    p.add_argument("--equity_csv", required=False, help="Explicit equity CSV path (overrides research_dir discovery).")
    p.add_argument("--metrics_csv", required=False, help="Metrics CSV (with period=full row).")
    p.add_argument("--benchmark_equity_csv", required=True, help="Benchmark equity CSV (ts,equity), e.g., ETH buy&hold.")
    p.add_argument("--benchmark_label", default="ETH-USD Buy & Hold", help="Label for benchmark table/plots.")
    p.add_argument("--strategy_note_md", required=True, help="Markdown note to render on final pages.")
    p.add_argument("--out_pdf", required=True, help="Output PDF path.")
    return p.parse_args()


def load_strategy_equity(research_dir: Path) -> pd.Series:
    eq_path = research_dir / "growth_equity_v0.csv"
    if not eq_path.exists():
        raise FileNotFoundError(f"Strategy equity not found at {eq_path}")
    return load_equity_csv(str(eq_path))


def plot_rolling(ax: plt.Axes, ret: pd.Series, bench_ret: pd.Series | None, window: int = 90) -> None:
    roll_vol = ret.rolling(window).std() * (365 ** 0.5)
    roll_sharpe = ret.rolling(window).mean() / ret.rolling(window).std() * (365 ** 0.5)
    ax.plot(roll_vol.index, roll_vol.values, label="Rolling Vol (ann)")
    ax.plot(roll_sharpe.index, roll_sharpe.values, label="Rolling Sharpe")
    if bench_ret is not None and not bench_ret.empty:
        b_roll_vol = bench_ret.rolling(window).std() * (365 ** 0.5)
        b_roll_sharpe = bench_ret.rolling(window).mean() / bench_ret.rolling(window).std() * (365 ** 0.5)
        ax.plot(b_roll_vol.index, b_roll_vol.values, linestyle="--", label="Bench Rolling Vol")
        ax.plot(b_roll_sharpe.index, b_roll_sharpe.values, linestyle="--", label="Bench Rolling Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Rolling {window}D Vol & Sharpe")


def render_note_page(pdf: PdfPages, note_path: Path, title: str) -> None:
    text = note_path.read_text()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    ax.text(0.02, 0.95, text, va="top", ha="left", wrap=True, fontsize=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    research_dir = Path(args.research_dir) if args.research_dir else None
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    eq_resolved, metrics_resolved, manifest_path = resolve_tearsheet_inputs(
        research_dir=str(research_dir) if research_dir else None,
        equity_csv=args.equity_csv,
        metrics_csv=args.metrics_csv,
        equity_patterns=["growth_equity_*.csv", "growth_equity_v0.csv"],
        metrics_patterns=["metrics_growth_v15_*.csv", "metrics_growth_v15_v0.csv", "*metrics*.csv"],
    )
    strategy_stats = load_strategy_stats_from_metrics(metrics_resolved)
    strat_eq = load_equity_csv(eq_resolved)
    bench_eq = load_benchmark_equity(args.benchmark_equity_csv, strat_eq.index)

    comparison_df = build_benchmark_comparison_table(
        strategy_label="Growth V1.5",
        strategy_stats=strategy_stats,
        benchmark_label=args.benchmark_label,
        benchmark_eq=bench_eq,
    )

    bench_ret = bench_eq.pct_change().dropna() if bench_eq is not None else None
    strat_ret = strat_eq.pct_change().dropna()

    with PdfPages(out_pdf) as pdf:
        # Page 1 summary + table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        summary_lines = [
            "Alpha Ensemble – V1.5 Growth Sleeve (Top50 ADV>10M)",
            f"Sample: {strategy_stats.get('start').strftime('%Y-%m-%d')} – {strategy_stats.get('end').strftime('%Y-%m-%d')}",
            f"CAGR: {strategy_stats['cagr']:.2%} | Vol: {strategy_stats['vol']:.2%} | Sharpe: {strategy_stats['sharpe']:.2f}",
            f"Sortino: {strategy_stats['sortino']:.2f} | Calmar: {strategy_stats['calmar']:.2f}",
            f"MaxDD: {strategy_stats['max_dd']:.2%} | Avg DD: {strategy_stats['avg_dd']:.2%}",
            f"Hit %: {strategy_stats['hit_ratio']*100:.1f}% | Exp %: {strategy_stats['expectancy']*100:.2f}%",
        ]
        ax.text(0.02, 0.95, "\n".join(summary_lines), va="top", ha="left", fontsize=11)
        add_benchmark_summary_table(fig, comparison_df, anchor=(0.62, 0.05))
        fig.suptitle("Performance Summary", fontsize=14, fontweight="bold", y=0.98)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Equity overlay
        fig, ax = plt.subplots(figsize=(11, 8.5))
        plot_equity_with_benchmark(ax, strat_eq, bench_eq, "Growth V1.5", args.benchmark_label)
        ax.set_title("Equity Curve (with Benchmark Overlay)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Drawdown overlay
        fig, ax = plt.subplots(figsize=(11, 8.5))
        plot_drawdown_with_benchmark(ax, strat_eq, bench_eq, "Growth V1.5", args.benchmark_label)
        ax.set_title("Drawdowns (with Benchmark Overlay)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Rolling vol/sharpe
        fig, ax = plt.subplots(figsize=(11, 8.5))
        plot_rolling(ax, strat_ret, bench_ret)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Strategy note
        render_note_page(pdf, Path(args.strategy_note_md), "Strategy Note")

        # Provenance page
        prov_text = build_provenance_text(eq_resolved, metrics_resolved, manifest_path)
        fig_p, ax_p = plt.subplots(figsize=(11, 8.5))
        ax_p.axis("off")
        ax_p.text(0.02, 0.98, prov_text, ha="left", va="top", fontsize=10)
        pdf.savefig(fig_p, bbox_inches="tight")
        plt.close(fig_p)

    print(f"[growth_tearsheet] Wrote tear sheet to {out_pdf}")


if __name__ == "__main__":
    main()
