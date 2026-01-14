#!/usr/bin/env python

"""
NOTE: This tear sheet uses tearsheet_common_v0.py for benchmark overlays and summary tables.

Alphas 101 Ensemble Tearsheet v0

Reads the standard 101_alphas research artifacts and produces a
multi-page PDF tearsheet with:

- Performance summary and net-of-cost metrics
- Equity curve and drawdown
- Rolling Sharpe and turnover
- Regime breakdown (trend/mean_rev/danger)
- Alpha selection / IC panel summary
- Cross-sectional concentration

Assumes you have already run the Phase 4 pipeline to generate:
- ensemble_equity_v0.csv
- ensemble_turnover_v0.csv
- metrics_101_alphas_ensemble_v0.csv
- metrics_101_alphas_ensemble_v0_costs_bps*.csv (optional but recommended)
- alphas101_beta_vs_btc_v0.csv
- alphas101_ic_panel_v0_h1.csv
- alphas101_selected_v0.csv
- alphas101_concentration_summary_v0_sel.csv
- alphas101_regimes_v0.csv
"""

import argparse
import glob
import os
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


def _load_csv(path: str, *, parse_dates: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)


def _load_csv_optional(path: str, *, parse_dates: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)


def _rolling_sharpe(returns: pd.Series, window: int = 90, ann_factor: float = 365.0) -> pd.Series:
    roll = returns.rolling(window=window)
    mean = roll.mean()
    std = roll.std()
    sharpe = (mean * ann_factor) / (std * np.sqrt(ann_factor))
    return sharpe


def _infer_metrics_csv(args: argparse.Namespace) -> Optional[str]:
    if args.metrics_csv:
        return args.metrics_csv

    v1 = sorted(glob.glob(os.path.join(args.research_dir, "metrics_101_ensemble_filtered_v1.csv")))
    if v1:
        return v1[0]
    v0 = sorted(glob.glob(os.path.join(args.research_dir, "metrics_101_alphas_ensemble_v0.csv")))
    return v0[0] if v0 else None


def _infer_tca_files(metrics_csv: str, research_dir: str) -> List[str]:
    base = os.path.basename(metrics_csv)
    if "filtered_v1" in base:
        pattern = os.path.join(research_dir, "metrics_101_ensemble_filtered_v1_costs_bps*.csv")
    elif "ensemble_v0" in base:
        pattern = os.path.join(research_dir, "metrics_101_alphas_ensemble_v0_costs_bps*.csv")
    else:
        return []
    return sorted(glob.glob(pattern))


def _load_tca(files: List[str]) -> pd.DataFrame:
    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if df.empty:
                continue
            row = df.iloc[0].copy()
            row["cost_bps"] = int(os.path.basename(fp).split("bps", 1)[1].split(".csv", 1)[0])
            rows.append(row)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("cost_bps").reset_index(drop=True)


def _infer_capacity_csv(args: argparse.Namespace, metrics_csv: str) -> Optional[str]:
    if args.capacity_csv:
        return args.capacity_csv
    base = os.path.basename(metrics_csv)
    if "filtered_v1" in base:
        path = os.path.join(args.research_dir, "capacity_sensitivity_v1.csv")
        return path if os.path.exists(path) else None
    return None


def add_strategy_note_pages(
    pdf: PdfPages,
    md_path: str,
    title: str = "Strategy Description",
    metrics_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Append one or more pages with a markdown-sourced strategy description.
    The first page includes a compact metrics table when metrics_df is provided.
    Headings and bullets are rendered with simple formatting for readability.
    """
    path = Path(md_path)
    if not path.exists():
        print(f"[alphas101_tearsheet_v0] strategy_note_md not found, skipping: {md_path}")
        return

    raw_lines = path.read_text(encoding="utf-8").splitlines()

    blocks = []
    current: dict = {"type": "text", "lines": []}

    def flush_current():
        nonlocal current
        if current["lines"]:
            blocks.append(current)
        current = {"type": "text", "lines": []}

    for raw in raw_lines:
        if raw.strip().startswith("#"):
            flush_current()
            level = len(raw) - len(raw.lstrip("#"))
            text = raw.lstrip("#").strip()
            blocks.append({"type": "heading", "level": level, "text": text})
        elif raw.strip().startswith(("-", "*")):
            if current["type"] != "bullet":
                flush_current()
                current = {"type": "bullet", "lines": []}
            current["lines"].append(raw.strip().lstrip("-* ").strip())
        elif raw.strip() == "":
            flush_current()
        else:
            if current["type"] != "text":
                flush_current()
                current = {"type": "text", "lines": []}
            current["lines"].append(raw.strip())
    flush_current()

    # Convert blocks to styled lines
    styled_lines: List[tuple[str, str]] = []
    for block in blocks:
        if block["type"] == "heading":
            lvl = block.get("level", 1)
            style = "h1" if lvl == 1 else "h2" if lvl == 2 else "h3"
            styled_lines.append((style, block["text"]))
            styled_lines.append(("space", ""))
        elif block["type"] == "bullet":
            for item in block["lines"]:
                for seg in textwrap.wrap(item, width=90):
                    styled_lines.append(("bullet", "• " + seg))
            styled_lines.append(("space", ""))
        else:
            paragraph = " ".join(block["lines"])
            for seg in textwrap.wrap(paragraph, width=95):
                styled_lines.append(("text", seg))
            styled_lines.append(("space", ""))

    # Metrics table (first page)
    table_lines: list[tuple[str, str]] = []
    if metrics_df is not None and not metrics_df.empty:
        row = metrics_df.loc[metrics_df["period"] == "full"].iloc[0]
        table_lines = [
            ("Metric", "Value"),
            ("CAGR", f"{row['cagr']:.2%}"),
            ("Vol (ann.)", f"{row['vol']:.2%}"),
            ("Sharpe", f"{row['sharpe']:.2f}"),
            ("Max DD", f"{row['max_dd']:.2%}"),
            ("Sample", f"{int(row['n_days'])} days"),
        ]

    if not styled_lines and not table_lines:
        return

    max_lines_per_page = 34
    idx = 0
    first_page = True
    while idx < len(styled_lines) or (first_page and table_lines):
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=15, fontweight="bold")
        y = 0.9

        if first_page and table_lines:
            col_x = [0.08, 0.32]
            for i, (k, v) in enumerate(table_lines):
                weight = "bold" if i == 0 else "normal"
                ax.text(col_x[0], y, k, ha="left", va="top", fontsize=10, fontweight=weight)
                ax.text(col_x[1], y, v, ha="left", va="top", fontsize=10, fontweight=weight)
                y -= 0.035
            y -= 0.02

        lines_used = 0
        while idx < len(styled_lines) and lines_used < max_lines_per_page:
            kind, text = styled_lines[idx]
            idx += 1
            if kind == "space":
                y -= 0.015
                lines_used += 1
                continue
            if kind == "h1":
                ax.text(0.05, y, text, ha="left", va="top", fontsize=12.5, fontweight="bold")
                y -= 0.03
            elif kind == "h2":
                ax.text(0.05, y, text, ha="left", va="top", fontsize=11.5, fontweight="bold")
                y -= 0.028
            elif kind == "h3":
                ax.text(0.05, y, text, ha="left", va="top", fontsize=10.5, fontweight="bold")
                y -= 0.026
            elif kind == "bullet":
                ax.text(0.07, y, text, ha="left", va="top", fontsize=10)
                y -= 0.022
            else:
                ax.text(0.05, y, text, ha="left", va="top", fontsize=10)
                y -= 0.021
            lines_used += 1
            if lines_used >= max_lines_per_page and idx < len(styled_lines) and styled_lines[idx][0] != "space":
                break

        pdf.savefig(fig)
        plt.close(fig)
        first_page = False


def make_tearsheet(
    research_dir: str,
    out_pdf: str,
    equity_path: Optional[str] = None,
    turnover_path: Optional[str] = None,
    base_metrics_path: Optional[str] = None,
    base_metrics_df: Optional[pd.DataFrame] = None,
    beta_path: Optional[str] = None,
    ic_panel_path: Optional[str] = None,
    selection_path: Optional[str] = None,
    concentration_path: Optional[str] = None,
    regimes_path: Optional[str] = None,
    capacity_path: Optional[str] = None,
    capacity_df: Optional[pd.DataFrame] = None,
    tca_df: Optional[pd.DataFrame] = None,
    symbol_stats_top_path: Optional[str] = None,
    strategy_note_md: Optional[str] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
    benchmark_label: str = "BTC-USD buy-and-hold",
    benchmark_path: Optional[str] = None,
) -> None:
    # Resolve equity/metrics strictly
    eq_resolved, metrics_resolved, manifest_path = resolve_tearsheet_inputs(
        research_dir=research_dir if (equity_path is None and base_metrics_path is None) else None,
        equity_csv=equity_path,
        metrics_csv=base_metrics_path,
        equity_patterns=["ensemble_equity_*.csv", "ensemble_equity_v0.csv"],
        metrics_patterns=["metrics_101_ensemble_filtered_v1.csv", "metrics_101_alphas_ensemble_v0.csv", "*metrics*.csv"],
    )
    equity_path = eq_resolved
    base_metrics_path = metrics_resolved

    # Resolve remaining defaults
    rd = research_dir
    turnover_path = turnover_path or os.path.join(rd, "ensemble_turnover_v0.csv")

    if base_metrics_path is None:
        raise ValueError("base_metrics_path must be provided")
    metrics_name = os.path.basename(base_metrics_path)
    is_v1 = "filtered_v1" in metrics_name

    # Core series
    equity = _load_csv(equity_path, parse_dates=["ts"])
    strat_eq = load_equity_csv(equity_path)
    strategy_stats = load_strategy_stats_from_metrics(base_metrics_path)
    turnover = _load_csv(turnover_path, parse_dates=["ts"])
    base_metrics = base_metrics_df if base_metrics_df is not None else _load_csv(base_metrics_path)

    beta_path = beta_path or (
        os.path.join(rd, "alphas101_beta_vs_btc_v1_adv10m.csv") if is_v1 else os.path.join(rd, "alphas101_beta_vs_btc_v0.csv")
    )
    beta_df = _load_csv_optional(beta_path)

    if ic_panel_path is None:
        ic_panel_path = (
            os.path.join(rd, "alphas101_ic_panel_v1_adv10m_filtered.csv") if is_v1 else os.path.join(rd, "alphas101_ic_panel_v0_h1.csv")
        )
    ic_panel = _load_csv(ic_panel_path)

    if selection_path is None:
        selection_path = (
            os.path.join(rd, "alphas101_selected_v1_adv10m.csv") if is_v1 else os.path.join(rd, "alphas101_selected_v0.csv")
        )
    selection = _load_csv(selection_path)

    if concentration_path is None:
        if is_v1:
            concentration_path = os.path.join(rd, "alphas101_concentration_summary_v1_adv10m.csv")
        else:
            fallback = os.path.join(rd, "alphas101_concentration_summary_v0_sel.csv")
            concentration_path = fallback if os.path.exists(fallback) else os.path.join(rd, "alphas101_concentration_summary_v0.csv")
    concentration = _load_csv(concentration_path)

    if regimes_path is None:
        regimes_path = os.path.join(rd, "alphas101_regimes_v1_adv10m.csv") if is_v1 else os.path.join(rd, "alphas101_regimes_v0.csv")
    regimes = _load_csv(regimes_path, parse_dates=["ts"])

    capacity = capacity_df if capacity_df is not None else (_load_csv_optional(capacity_path) if capacity_path else None)

    if symbol_stats_top_path is None and is_v1:
        sym_default = os.path.join(rd, "alphas101_symbol_stats_top20_v1_adv10m.csv")
        if os.path.exists(sym_default):
            symbol_stats_top_path = sym_default
    symbol_stats_top = _load_csv_optional(symbol_stats_top_path) if symbol_stats_top_path else None

    tca = tca_df if tca_df is not None else pd.DataFrame()

    # Join regimes into equity
    equity = equity.sort_values("ts").reset_index(drop=True)
    regimes = regimes.sort_values("ts").reset_index(drop=True)
    equity = equity.merge(regimes[["ts", "regime"]], on="ts", how="left")

    strat_eq = equity.set_index("ts")["portfolio_equity"]
    strat_stats = strategy_stats

    # Derived series
    equity["drawdown"] = compute_drawdown(equity["portfolio_equity"])
    equity["rolling_sharpe_90d"] = _rolling_sharpe(equity["portfolio_ret"], window=90)
    equity["rolling_vol_90d"] = equity["portfolio_ret"].rolling(window=90).std() * np.sqrt(365.0)
    # Merge turnover
    turnover = turnover.sort_values("ts")
    equity = equity.merge(turnover[["ts", "turnover"]], on="ts", how="left")

    # Regime stats
    regime_stats = (
        equity.groupby("regime")["portfolio_ret"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n_days"})
        .reset_index()
    )
    regime_stats["ann_return"] = (1.0 + regime_stats["mean"]) ** 365 - 1.0
    regime_stats["ann_vol"] = regime_stats["std"] * np.sqrt(365.0)
    regime_stats["sharpe"] = regime_stats["ann_return"] / regime_stats["ann_vol"]

    # Selection / IC
    # selection: ['alpha','sign','n_days','mean_ic','std_ic','tstat_ic','mean_ic_oriented']
    selected_sorted = selection.sort_values("mean_ic_oriented", ascending=False).reset_index(drop=True)

    # Merge IC panel + selection to show where selection sits in the universe
    ic_full = ic_panel.merge(
        selection[["alpha", "sign", "mean_ic_oriented"]],
        on="alpha",
        how="left",
        suffixes=("", "_sel"),
    )
    ic_full["selected"] = ~ic_full["sign"].isna()

    # Benchmark handling
    benchmark_stats = None
    bench_eq = None
    bench_rets = None
    if benchmark_path:
        try:
            bench_eq = load_benchmark_equity(benchmark_path, strat_eq.index)
        except Exception as e:
            print(f"[alphas101_tearsheet_v0] Benchmark load failed ({e}); falling back to provided frame if any.")
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
        print("[alphas101_tearsheet_v0] Benchmark missing or no overlap; skipping benchmark.")

    # Start PDF
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # Page 1: Title + summary tables + BTC vs Strategy table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        title = (
            "Alpha Ensemble – V1 ADV>10M (Long+Cash)"
            if is_v1
            else "Alpha Ensemble – Legacy USD Universe (V0)"
        )
        subtitle = (
            "Daily long+cash ensemble over ADV>10M Coinbase USD pairs | selection + regime gating"
            if is_v1
            else "Daily long+cash ensemble over Coinbase USD pairs (ex-stablecoins) | selection + regime gating"
        )

        # Performance summary from metrics CSV (strategy_stats)
        summary_lines = [
            f"Sample: {strategy_stats.get('start')} – {strategy_stats.get('end')}",
            f"CAGR: {strategy_stats.get('cagr'):.2%}",
            f"Vol (ann.): {strategy_stats.get('vol'):.2%}",
            f"Sharpe: {strategy_stats.get('sharpe'):.2f}",
            f"Sortino: {strategy_stats.get('sortino', float('nan')):.2f}",
            f"Calmar: {strategy_stats.get('calmar', float('nan')):.2f}",
            f"Max drawdown: {strategy_stats.get('max_dd'):.2%}",
            f"Avg drawdown: {strategy_stats.get('avg_dd', float('nan')):.2%}",
            f"Hit ratio: {strategy_stats.get('hit_ratio', float('nan'))*100:.1f}%",
            f"Expectancy: {strategy_stats.get('expectancy', float('nan'))*100:.2f}%",
            f"Sample length: {int(strategy_stats.get('n_days', 0))} days",
        ]

        text_y = 0.9
        ax.text(0.02, text_y, title, fontsize=18, fontweight="bold", transform=ax.transAxes)
        text_y -= 0.05
        ax.text(0.02, text_y, subtitle, fontsize=11, transform=ax.transAxes)

        text_y -= 0.08
        ax.text(
            0.02,
            text_y,
            "Performance summary (before transaction costs):",
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )
        text_y -= 0.04
        for line in summary_lines:
            ax.text(0.04, text_y, f"• {line}", fontsize=11, transform=ax.transAxes)
            text_y -= 0.035

        # Net-of-cost metrics if present
        if not tca.empty:
            text_y -= 0.03
            ax.text(
                0.02,
                text_y,
                "Net-of-cost performance (per-side cost assumption):",
                fontsize=12,
                fontweight="bold",
                transform=ax.transAxes,
            )
            text_y -= 0.04
            for _, row in tca.iterrows():
                line = (
                    f"{int(row['cost_bps'])} bps: "
                    f"CAGR {row['cagr']:.2%}, "
                    f"Sharpe {row['sharpe']:.2f}, "
                    f"MaxDD {row['max_dd']:.2%}"
                )
                ax.text(0.04, text_y, f"• {line}", fontsize=11, transform=ax.transAxes)
                text_y -= 0.032

        # Beta vs BTC
        if beta_df is not None and not beta_df.empty:
            row = beta_df.iloc[0]
            text_y -= 0.03
            ax.text(0.02, text_y, "Beta / correlation vs BTC-USD:", fontsize=12, fontweight="bold", transform=ax.transAxes)
            text_y -= 0.04
            beta_line = (
                f"Corr={row['corr']:.2f}, Beta={row['beta']:.2f}, "
                f"R²={row['r2']:.3f}, t_beta={row['t_beta']:.2f}"
            )
            ax.text(0.04, text_y, f"• {beta_line}", fontsize=11, transform=ax.transAxes)

        # Regime stats overview
        text_y -= 0.06
        ax.text(0.02, text_y, "Regime breakdown (daily):", fontsize=12, fontweight="bold", transform=ax.transAxes)
        text_y -= 0.04
        for _, row in regime_stats.iterrows():
            line = (
                f"{row['regime']}: "
                f"{int(row['n_days'])} days, "
                f"ann. return {row['ann_return']:.2%}, "
                f"ann. vol {row['ann_vol']:.2%}, "
                f"Sharpe {row['sharpe']:.2f}"
            )
            ax.text(0.04, text_y, f"• {line}", fontsize=11, transform=ax.transAxes)
            text_y -= 0.032

        # Capacity sensitivity (optional)
        if capacity is not None and not capacity.empty:
            text_y -= 0.03
            ax.text(0.02, text_y, "Capacity sensitivity (per-side cost sweep):", fontsize=12, fontweight="bold", transform=ax.transAxes)
            text_y -= 0.04
            for _, row in capacity.iterrows():
                line = (
                    f"{int(row['cost_bps'])} bps: Sharpe {row['sharpe']:.2f}, "
                    f"CAGR {row['cagr']:.2%}, MaxDD {row['max_dd']:.2%}"
                )
                ax.text(0.04, text_y, f"• {line}", fontsize=11, transform=ax.transAxes)
                text_y -= 0.032

        if benchmark_stats is not None:
            comparison_df = build_benchmark_comparison_table(
                strategy_label="Alpha Ensemble",
                strategy_stats=strategy_stats,
                benchmark_label=benchmark_label,
                benchmark_eq=bench_eq,
            )
            add_benchmark_summary_table(
                fig,
                comparison_df,
                anchor=(0.7, 0.02),
            )

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Equity & drawdown
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        plot_equity_with_benchmark(
            axes[0],
            strat_eq,
            bench_eq,
            strat_label="Alpha Ensemble",
            bench_label=benchmark_label,
        )
        axes[0].set_ylabel("Equity (nav)")
        axes[0].set_title("Equity curve")

        plot_drawdown_with_benchmark(
            axes[1],
            strat_eq,
            bench_eq,
            strat_label="Alpha Ensemble",
            bench_label=benchmark_label,
        )
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Date")
        axes[1].set_title("Drawdown (from peak)")

        for ax_ in axes:
            ax_.grid(True, alpha=0.3)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Rolling risk & turnover
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

        axes[0].plot(equity["ts"], equity["rolling_sharpe_90d"], label="Strategy")
        if bench_rets is not None:
            bench_roll_sharpe = _rolling_sharpe(bench_rets, window=90)
            axes[0].plot(bench_roll_sharpe.index, bench_roll_sharpe.values, label=benchmark_label, linestyle="--")
            axes[0].legend()
        axes[0].set_ylabel("Sharpe (90d)")
        axes[0].set_title("Rolling 90-day Sharpe")

        axes[1].plot(equity["ts"], equity["rolling_vol_90d"], label="Strategy")
        if bench_rets is not None:
            bench_roll_vol = bench_rets.rolling(window=90).std() * np.sqrt(365.0)
            axes[1].plot(bench_roll_vol.index, bench_roll_vol.values, label=benchmark_label, linestyle="--")
            axes[1].legend()
        axes[1].set_ylabel("Ann. vol (90d)")
        axes[1].set_title("Rolling 90-day annualized volatility")

        axes[2].plot(equity["ts"], equity["turnover"])
        axes[2].set_ylabel("Turnover")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Daily turnover")

        for ax_ in axes:
            ax_.grid(True, alpha=0.3)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Regimes & exposures
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

        # Regime-colored returns
        colors = {"trend": "tab:green", "mean_rev": "tab:blue", "danger": "tab:red"}
        for regime_name, group in equity.groupby("regime"):
            axes[0].scatter(
                group["ts"],
                group["portfolio_ret"],
                s=8,
                label=regime_name,
                alpha=0.7,
                color=colors.get(regime_name, "gray"),
            )
        axes[0].axhline(0.0, color="black", linewidth=0.5)
        axes[0].set_ylabel("Daily return")
        axes[0].set_title("Daily returns by regime")
        axes[0].legend()

        # Gross long & cash weight
        axes[1].plot(equity["ts"], equity["gross_long"], label="Gross long")
        axes[1].plot(equity["ts"], equity["cash_weight"], label="Cash weight")
        axes[1].set_ylabel("Weight")
        axes[1].set_title("Gross exposure vs cash")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Regime occupancy time series (stacked area, 30d rolling)
        regime_dummies = pd.get_dummies(equity["regime"])
        regime_share = regime_dummies.rolling(window=30, min_periods=1).mean()
        axes[2].stackplot(
            equity["ts"],
            *[regime_share[col] for col in regime_share.columns],
            labels=list(regime_share.columns),
        )
        axes[2].set_ylabel("Share (30d rolling)")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Regime occupancy (30-day rolling share)")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Alpha IC / selection
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))

        # Top oriented IC alphas (selected)
        top_sel = selected_sorted.head(20)
        axes[0].bar(top_sel["alpha"], top_sel["mean_ic_oriented"])
        axes[0].set_ylabel("Mean IC (oriented)")
        axes[0].set_title("Top selected alphas by oriented mean IC (horizon 1)")
        axes[0].tick_params(axis="x", rotation=90)
        axes[0].grid(True, axis="y", alpha=0.3)

        # IC distribution (all alphas)
        axes[1].hist(ic_full["mean_ic"], bins=30, alpha=0.7, label="All alphas")
        axes[1].hist(
            ic_full.loc[ic_full["selected"], "mean_ic"],
            bins=30,
            alpha=0.7,
            label="Selected",
        )
        axes[1].set_xlabel("Mean IC (horizon 1)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of mean IC (all vs selected alphas)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: Concentration (top names)
        fig, ax = plt.subplots(figsize=(11, 8.5))

        top_names = concentration.head(20).copy()
        ax.bar(top_names["symbol"], top_names["avg_abs_weight"])
        ax.set_ylabel("Avg |weight|")
        ax.set_title("Top symbols by average absolute weight")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(True, axis="y", alpha=0.3)

        # BTC/ETH share (single number from concentration table)
        if "btc_eth_share_total" in concentration.columns and not concentration["btc_eth_share_total"].isna().all():
            btc_eth_share = concentration["btc_eth_share_total"].iloc[0]
            ax.text(
                0.02,
                0.95,
                f"BTC+ETH share of total |weight|: {btc_eth_share:.1%}",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 7: Symbol exposure & turnover (top 20) if available
        if symbol_stats_top is not None and not symbol_stats_top.empty:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")

            cols = ["symbol", "avg_abs_weight", "holding_ratio", "turnover_share_pct"]
            tbl = symbol_stats_top[cols].copy()
            tbl["avg_abs_weight"] = tbl["avg_abs_weight"].map(lambda x: f"{x:.4f}")
            tbl["holding_ratio"] = tbl["holding_ratio"].map(lambda x: f"{x:.2%}")
            tbl["turnover_share_pct"] = tbl["turnover_share_pct"].map(lambda x: f"{x*100:.2f}%")

            table = ax.table(
                cellText=tbl.values,
                colLabels=tbl.columns,
                loc="center",
                cellLoc="center",
            )
            table.scale(1, 1.5)
            ax.set_title("Symbol Exposure & Turnover (Top 20)", fontsize=14, pad=10)
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print("[alphas101_tearsheet_v0] Symbol stats not found; skipping symbol exposure page.")

        # Page 8: Return distribution
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.hist(equity["portfolio_ret"], bins=50, alpha=0.8)
        ax.set_xlabel("Daily return")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of daily returns")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 9+: Strategy description (optional)
        if strategy_note_md:
            # Pick a note header based on metrics source
            note_header = "Strategy Description – Alpha Ensemble"
            metrics_name = Path(base_metrics_path).name
            if metrics_name == "metrics_101_alphas_ensemble_v0.csv":
                note_header = "Alpha Ensemble – Legacy Full USD Universe (IC Summary)"
            elif metrics_name == "metrics_101_ensemble_filtered_v1.csv":
                note_header = "Alpha Ensemble – ADV>10M V1 (IC Summary)"

            add_strategy_note_pages(pdf, strategy_note_md, title=note_header, metrics_df=base_metrics)

        # Provenance page
        prov_text = build_provenance_text(equity_path, base_metrics_path, manifest_path)
        fig_p, ax_p = plt.subplots(figsize=(11, 8.5))
        ax_p.axis("off")
        ax_p.text(0.02, 0.98, prov_text, ha="left", va="top", fontsize=10)
        pdf.savefig(fig_p, bbox_inches="tight")
        plt.close(fig_p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PDF tearsheet for 101_alphas ensemble.")
    parser.add_argument(
        "--research_dir",
        type=str,
        default="artifacts/research/101_alphas",
        help="Root directory containing 101_alphas artifacts.",
    )
    parser.add_argument(
        "--out_pdf",
        type=str,
        default=None,
        help="Output PDF path (default: <research_dir>/alphas101_tearsheet_v0.pdf)",
    )
    parser.add_argument(
        "--equity",
        type=str,
        default=None,
        help="Optional override path for ensemble_equity_v0.csv",
    )
    parser.add_argument(
        "--turnover",
        type=str,
        default=None,
        help="Optional override path for ensemble_turnover_v0.csv",
    )
    parser.add_argument(
        "--metrics_csv",
        dest="metrics_csv",
        type=str,
        default=None,
        help="Explicit metrics CSV for this configuration.",
    )
    parser.add_argument(
        "--capacity_csv",
        dest="capacity_csv",
        type=str,
        default=None,
        help="Optional capacity CSV for this configuration; if not provided, capacity is skipped.",
    )
    parser.add_argument(
        "--beta",
        type=str,
        default=None,
        help="Optional override path for alphas101_beta_vs_btc_v0.csv",
    )
    parser.add_argument(
        "--ic_panel",
        type=str,
        default=None,
        help="Optional override path for alphas101_ic_panel_v0_h1.csv",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default=None,
        help="Optional override path for alphas101_selected_v0.csv",
    )
    parser.add_argument(
        "--concentration",
        type=str,
        default=None,
        help="Optional override path for alphas101_concentration_summary_v0_sel.csv",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default=None,
        help="Optional override path for alphas101_regimes_v0.csv",
    )
    parser.add_argument(
        "--symbol_stats_top",
        type=str,
        default=None,
        help="Optional override path for alphas101_symbol_stats_top20_v1_adv10m.csv (or variant).",
    )
    parser.add_argument(
        "--strategy_note_md",
        type=str,
        default="docs/research/alphas101_ic_note_v0.md",
        help="Optional Markdown file with IC-style strategy description to append as final page(s).",
    )
    parser.add_argument(
        "--benchmark_equity_csv",
        type=str,
        default=None,
        help="Optional CSV with benchmark equity (ts, benchmark_equity) to overlay.",
    )
    parser.add_argument(
        "--benchmark_label",
        type=str,
        default="BTC-USD buy-and-hold",
        help="Label for the benchmark line and stats.",
    )
    args = parser.parse_args()

    out_pdf = args.out_pdf or os.path.join(args.research_dir, "alphas101_tearsheet_v0.pdf")

    metrics_csv = _infer_metrics_csv(args)
    if not metrics_csv:
        raise SystemExit("No metrics CSV found (neither V1 filtered nor legacy V0).")
    base_metrics = pd.read_csv(metrics_csv)
    tca_files = _infer_tca_files(metrics_csv, args.research_dir)
    tca_df = _load_tca(tca_files)
    capacity_csv = _infer_capacity_csv(args, metrics_csv)
    capacity_df = pd.read_csv(capacity_csv) if capacity_csv else None

    benchmark_df = None
    if args.benchmark_equity_csv and os.path.exists(args.benchmark_equity_csv):
        benchmark_df = pd.read_csv(args.benchmark_equity_csv, parse_dates=["ts"])
    elif args.benchmark_equity_csv:
        print(f"[alphas101_tearsheet_v0] Benchmark CSV {args.benchmark_equity_csv} not found; skipping benchmark overlay.")

    make_tearsheet(
        research_dir=args.research_dir,
        out_pdf=out_pdf,
        equity_path=args.equity,
        turnover_path=args.turnover,
        base_metrics_path=metrics_csv,
        base_metrics_df=base_metrics,
        beta_path=args.beta,
        ic_panel_path=args.ic_panel,
        selection_path=args.selection,
        concentration_path=args.concentration,
        regimes_path=args.regimes,
        capacity_path=capacity_csv,
        capacity_df=capacity_df,
        tca_df=tca_df,
        symbol_stats_top_path=args.symbol_stats_top,
        strategy_note_md=args.strategy_note_md,
        benchmark_df=benchmark_df,
        benchmark_label=args.benchmark_label,
        benchmark_path=args.benchmark_equity_csv,
    )


if __name__ == "__main__":
    main()

