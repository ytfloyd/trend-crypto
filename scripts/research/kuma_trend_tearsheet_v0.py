#!/usr/bin/env python
from __future__ import annotations

"""
NOTE: This tear sheet uses tearsheet_common_v0.py for benchmark overlays and summary tables.
"""

import argparse
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# Allow running as a script without installing as a package
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tearsheet_common_v0 import (
    load_equity_csv,
    load_benchmark_equity,
    compute_stats,
    compute_drawdown,
    plot_equity_with_benchmark,
    plot_drawdown_with_benchmark,
    add_benchmark_summary_table,
    build_benchmark_comparison_table,
    load_strategy_stats_from_metrics,
    resolve_tearsheet_inputs,
    build_provenance_text,
)
from run_manifest_v0 import update_run_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PDF tear sheet for kuma_trend v0.")
    p.add_argument(
        "--research_dir",
        type=str,
        default="artifacts/research/kuma_trend",
        help="Root directory containing kuma_trend artifacts.",
    )
    p.add_argument(
        "--equity_csv",
        type=str,
        default=None,
        help="Explicit path to equity CSV (overrides research_dir discovery).",
    )
    p.add_argument(
        "--metrics_csv",
        type=str,
        default=None,
        help="Explicit path to metrics CSV (overrides research_dir discovery).",
    )
    p.add_argument(
        "--out_pdf",
        type=str,
        default="artifacts/research/kuma_trend/kuma_trend_tearsheet_v0.pdf",
        help="Output PDF path.",
    )
    p.add_argument(
        "--strategy_note_md",
        type=str,
        default="docs/research/kuma_trend_overview_v0.md",
        help="Optional Markdown file with IC-style strategy description to append as final page(s).",
    )
    p.add_argument(
        "--benchmark_equity_csv",
        type=str,
        default=None,
        help="Optional CSV with benchmark equity (ts, benchmark_equity) to overlay.",
    )
    p.add_argument(
        "--benchmark_label",
        type=str,
        default="BTC-USD buy-and-hold",
        help="Label for the benchmark line and stats.",
    )
    p.add_argument(
        "--metrics_csv",
        type=str,
        default="artifacts/research/kuma_trend/metrics_kuma_trend_v0.csv",
        help="Metrics CSV for kuma_trend (strategy stats).",
    )
    return p.parse_args()


def _load_csv(path: Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def _load_csv_optional(path: Path, parse_dates: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[kuma_trend_tearsheet_v0] Optional file missing, skipping: {path}")
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def add_strategy_note_pages(pdf: PdfPages, md_path: str, title: str = "Strategy Description") -> None:
    """
    Append one or more pages with a markdown-sourced strategy description.
    Markdown is flattened into wrapped plain text for simple PDF rendering.
    """
    path = Path(md_path)
    if not path.exists():
        print(f"[kuma_trend_tearsheet_v0] strategy_note_md not found, skipping: {md_path}")
        return

    text = path.read_text(encoding="utf-8")
    lines = []
    for raw in text.splitlines():
        line = raw.lstrip("#").strip()
        if not line:
            lines.append("")
        else:
            lines.append(line)

    wrapped: list[str] = []
    for line in lines:
        if line == "":
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, width=100))

    if not wrapped:
        return

    max_lines_per_page = 38
    for idx in range(0, len(wrapped), max_lines_per_page):
        chunk = wrapped[idx : idx + max_lines_per_page]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        ax.text(
            0.5,
            0.95,
            title,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

        y = 0.9
        for ln in chunk:
            if ln == "":
                y -= 0.02
                continue
            ax.text(0.05, y, ln, ha="left", va="top", fontsize=9)
            y -= 0.022

        pdf.savefig(fig)
        plt.close(fig)


def make_tearsheet(
    research_dir: Path,
    out_pdf: Path,
    equity_csv: Optional[str] = None,
    metrics_csv: Optional[str] = None,
    strategy_note_md: Optional[str] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
    benchmark_label: str = "BTC-USD buy-and-hold",
    benchmark_path: Optional[str] = None,
) -> None:
    eq_resolved, metrics_resolved, manifest_path = resolve_tearsheet_inputs(
        research_dir=str(research_dir) if (equity_csv is None and metrics_csv is None) else None,
        equity_csv=equity_csv,
        metrics_csv=metrics_csv,
        equity_patterns=["kuma_trend_equity_*.csv", "kuma_trend_equity_v0.csv"],
        metrics_patterns=["metrics_kuma_trend_*.csv", "metrics_kuma_trend_v0.csv"],
    )
    equity_path = Path(eq_resolved)
    metrics_path = Path(metrics_resolved)
    turnover_path = research_dir / "kuma_trend_turnover_v0.csv"
    weights_path = research_dir / "kuma_trend_weights_v0.parquet"
    positions_path = research_dir / "kuma_trend_positions_v0.parquet"

    equity = _load_csv(equity_path, parse_dates=["ts"]).sort_values("ts")
    strat_eq = load_equity_csv(str(equity_path))
    strategy_stats = load_strategy_stats_from_metrics(metrics_csv or str(metrics_path))
    metrics = _load_csv(metrics_path)
    turnover = _load_csv_optional(turnover_path, parse_dates=["ts"])
    weights = None
    if weights_path.exists():
        weights = pd.read_parquet(weights_path)
    positions = None
    if positions_path.exists():
        positions = pd.read_parquet(positions_path)

    equity["drawdown"] = equity["portfolio_equity"] / equity["portfolio_equity"].cummax() - 1.0
    equity["rolling_vol_63"] = (
        equity["portfolio_ret"]
        .rolling(63, min_periods=63)
        .std()
        * np.sqrt(252.0)
    )
    roll_std = equity["portfolio_ret"].rolling(63, min_periods=63).std()
    roll_mean = equity["portfolio_ret"].rolling(63, min_periods=63).mean()
    equity["rolling_sharpe_63"] = (roll_mean / roll_std) * np.sqrt(252.0)

    full_row = metrics.loc[metrics["period"] == "full"].iloc[0] if not metrics.empty else None

    turnover_stats = {}
    if turnover is not None:
        t = turnover["turnover"].dropna()
        turnover_stats = {
            "mean": t.mean(),
            "median": t.median(),
            "pct25": t.quantile(0.25),
            "pct75": t.quantile(0.75),
            "max": t.max(),
        }

    # Prepare weight stats if available
    weight_stats = None
    weight_pivot = None
    if weights is not None and {"ts", "symbol", "weight"}.issubset(weights.columns):
        weights["ts"] = pd.to_datetime(weights["ts"])
        weight_pivot = (
            weights.pivot(index="ts", columns="symbol", values="weight")
            .sort_index()
            .fillna(0.0)
        )
        avg_abs = weight_pivot.abs().mean()
        holding_ratio = (weight_pivot.abs() > 0).mean()
        weight_stats = pd.DataFrame(
            {
                "symbol": avg_abs.index,
                "avg_abs_weight": avg_abs.values,
                "holding_ratio": holding_ratio.values,
            }
        ).sort_values("avg_abs_weight", ascending=False)

    # Benchmark handling
    bench_eq = None
    bench_rets = None
    benchmark_stats = None
    if benchmark_path:
        try:
            bench_eq = load_benchmark_equity(benchmark_path, strat_eq.index)
        except Exception as e:
            print(f"[kuma_trend_tearsheet_v0] Benchmark load failed ({e}); falling back to provided frame if any.")
    if bench_eq is None and benchmark_df is not None and not benchmark_df.empty:
        df_b = benchmark_df.copy()
        if "equity" not in df_b.columns:
            value_cols = [c for c in df_b.columns if c != "ts"]
            if value_cols:
                df_b = df_b.rename(columns={value_cols[0]: "equity"})
        if {"ts", "equity"}.issubset(df_b.columns):
            bench_series = df_b.set_index("ts")["equity"].astype(float).sort_index()
            bench_series = bench_series.reindex(strat_eq.index).ffill().dropna()
            if not bench_series.empty:
                bench_eq = bench_series / bench_series.iloc[0]
    if bench_eq is not None and not bench_eq.empty:
        benchmark_stats = compute_stats(bench_eq)
        bench_rets = bench_eq.pct_change().dropna()
    else:
        print("[kuma_trend_tearsheet_v0] Benchmark missing or no overlap; skipping benchmark.")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # Page 1: Summary & equity with BTC vs Strategy table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        plot_equity_with_benchmark(ax, strat_eq, bench_eq, "kuma_trend", benchmark_label)
        ax.set_yscale("log")
        ax.set_title("kuma_trend v0 – 20D Breakout + MA(5/40), ATR(20) Trailing Stop")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Equity (log)")

        text_lines = []
        if full_row is not None:
            text_lines.append(f"Sample: {strategy_stats.get('start')} – {strategy_stats.get('end')}")
            text_lines.append(f"CAGR: {strategy_stats.get('cagr'):.2%}")
            text_lines.append(f"Vol: {strategy_stats.get('vol'):.2%}")
            text_lines.append(f"Sharpe: {strategy_stats.get('sharpe'):.2f}")
            text_lines.append(f"Sortino: {strategy_stats.get('sortino', float('nan')):.2f}")
            text_lines.append(f"Calmar: {strategy_stats.get('calmar', float('nan')):.2f}")
            text_lines.append(f"MaxDD: {strategy_stats.get('max_dd'):.2%}")
            text_lines.append(f"Avg DD: {strategy_stats.get('avg_dd', float('nan')):.2%}")
            text_lines.append(f"Hit%: {strategy_stats.get('hit_ratio', float('nan'))*100:.1f}%")
            text_lines.append(f"Exp%: {strategy_stats.get('expectancy', float('nan'))*100:.2f}%")
            text_lines.append(f"n_days: {int(strategy_stats.get('n_days', 0))}")
        if turnover_stats:
            text_lines.append(f"Mean turnover: {turnover_stats['mean']:.3f}")
        text_lines.append("Cash yield: 4% annual (research assumption)")
        anchored_text = "\n".join(text_lines)
        ax.text(
            0.02,
            0.98,
            anchored_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        if benchmark_stats is not None:
            comparison_df = build_benchmark_comparison_table(
                strategy_label="kuma_trend",
                strategy_stats=strategy_stats,
                benchmark_label=benchmark_label,
                benchmark_eq=bench_eq,
            )
            add_benchmark_summary_table(
                fig,
                comparison_df,
                anchor=(0.68, 0.02),
            )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Drawdowns
        fig, ax = plt.subplots(figsize=(11, 8.5))
        plot_drawdown_with_benchmark(ax, strat_eq, bench_eq, "kuma_trend", benchmark_label)
        for lvl in [-0.1, -0.2, -0.3]:
            ax.axhline(lvl, color="gray", linestyle="--", linewidth=0.7)
        ax.set_title("Drawdowns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Rolling risk & turnover
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
        axes[0].plot(equity["ts"], equity["rolling_vol_63"], label="Strategy")
        if bench_rets is not None:
            bench_roll_vol = bench_rets.rolling(63).std() * np.sqrt(252.0)
            axes[0].plot(bench_roll_vol.index, bench_roll_vol.values, label=benchmark_label, linestyle="--")
            axes[0].legend()
        axes[0].set_ylabel("Vol (ann.)")
        axes[0].set_title("Rolling 63-day Volatility")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(equity["ts"], equity["rolling_sharpe_63"], label="Strategy")
        if bench_rets is not None:
            bench_roll_sharpe = (bench_rets.rolling(63).mean() / (bench_rets.rolling(63).std() + 1e-12)) * np.sqrt(252.0)
            axes[1].plot(bench_roll_sharpe.index, bench_roll_sharpe.values, label=benchmark_label, linestyle="--")
            axes[1].legend()
        axes[1].set_ylabel("Sharpe (63d)")
        axes[1].set_title("Rolling 63-day Sharpe")
        axes[1].grid(True, alpha=0.3)

        if turnover is not None:
            axes[2].plot(turnover["ts"], turnover["turnover"])
            if turnover_stats:
                axes[2].axhline(turnover_stats["mean"], color="gray", linestyle="--", linewidth=0.8, label="Mean")
                axes[2].legend()
        axes[2].set_ylabel("Turnover")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Daily Turnover (two-sided)")
        axes[2].grid(True, alpha=0.3)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Return distribution
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ret = equity["portfolio_ret"].dropna()
        ax.hist(ret, bins=50, alpha=0.8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title("Daily Return Distribution")
        ax.set_xlabel("Daily return")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        mu = ret.mean()
        sd = ret.std()
        ax.text(
            0.98,
            0.95,
            f"mean={mu:.4f}\nstd={sd:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Symbol exposure (if weights available)
        if weight_stats is not None and not weight_stats.empty:
            top = weight_stats.head(10)
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
            axes[0].bar(top["symbol"], top["avg_abs_weight"])
            axes[0].set_ylabel("Avg |weight|")
            axes[0].set_title("Top Symbols by Avg |Weight|")
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].grid(True, axis="y", alpha=0.3)

            axes[1].bar(top["symbol"], top["holding_ratio"])
            axes[1].set_ylabel("Holding ratio")
            axes[1].set_title("Holding Ratio (fraction of days in position)")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].grid(True, axis="y", alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print("[kuma_trend_tearsheet_v0] Weights not found or empty; skipping symbol exposure page.")

        # Optional Page 6: Position diagnostics (skip if missing)
        if positions is not None and {"ts", "symbol"}.issubset(positions.columns):
            # placeholder simple count by symbol
            fig, ax = plt.subplots(figsize=(11, 8.5))
            counts = positions["symbol"].value_counts()
            counts.head(20).plot(kind="bar", ax=ax)
            ax.set_title("Positions count by symbol (top 20)")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print("[kuma_trend_tearsheet_v0] Positions not found or incomplete; skipping position page.")

        # Page 7+: Strategy description (optional)
        if strategy_note_md:
            add_strategy_note_pages(pdf, strategy_note_md, title="Strategy Description – kuma_trend")

        # Provenance page
        prov_text = build_provenance_text(str(equity_path), str(metrics_path), manifest_path)
        fig_p, ax_p = plt.subplots(figsize=(11, 8.5))
        ax_p.axis("off")
        ax_p.text(0.02, 0.98, prov_text, ha="left", va="top", fontsize=10)
        pdf.savefig(fig_p, bbox_inches="tight")
        plt.close(fig_p)

    print(f"[kuma_trend_tearsheet_v0] Wrote tear sheet to {out_pdf}")
    manifest_path = Path(args.research_dir) / "run_manifest.json"
    update_run_manifest(manifest_path, {"artifacts_written": {"tearsheet_pdf": str(out_pdf)}})


def main() -> None:
    args = parse_args()
    research_dir = Path(args.research_dir)
    out_pdf = Path(args.out_pdf)
    benchmark_df = None
    if args.benchmark_equity_csv and Path(args.benchmark_equity_csv).exists():
        benchmark_df = pd.read_csv(args.benchmark_equity_csv, parse_dates=["ts"])
    make_tearsheet(
        research_dir,
        out_pdf,
        strategy_note_md=args.strategy_note_md,
        benchmark_df=benchmark_df,
        benchmark_label=args.benchmark_label,
        benchmark_path=args.benchmark_equity_csv,
        metrics_csv=args.metrics_csv,
    )


if __name__ == "__main__":
    main()

