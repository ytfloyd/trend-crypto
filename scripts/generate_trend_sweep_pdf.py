#!/usr/bin/env python3
"""Generate a PDF report for a single-asset trend sweep.

Produces 11 exhibits matching the ETH Trend Following Report format:

  1. Sharpe distribution across all configs
  2. CAGR vs max drawdown scatter
  3. Performance by frequency (box plot)
  4. Performance by signal family (daily only, horizontal bar)
  5. Sharpe vs time-in-market
  6. Sharpe vs skewness
  7. Aggregate stop comparison (grouped bar)
  8. Matched ATR vs fixed stop comparison
  9. Multiple testing (Bonferroni correction)
 10. CAGR vs max DD with Calmar iso-lines
 11. Drawdown distribution by stop type

Usage:
    python scripts/generate_trend_sweep_pdf.py --symbol BTC-USD
    python scripts/generate_trend_sweep_pdf.py --symbol ETH-USD --results artifacts/research/tsmom/ethusd_trend_sweep/results_v2.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DARK_BG = "#1a1a2e"
ACCENT = "#00d2ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd93d"
GRID_COLOR = "#333355"
TEXT_COLOR = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 10,
})


def _title_page(pdf: PdfPages, symbol: str, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.65, f"{symbol}", transform=ax.transAxes, ha="center",
            fontsize=48, fontweight="bold", color=ACCENT)
    ax.text(0.5, 0.52, "Trend Following Signal Sweep", transform=ax.transAxes,
            ha="center", fontsize=24, color=TEXT_COLOR)

    n = len(df)
    n_signals = df["label"].nunique()
    n_fam = df["signal_family"].nunique()
    freqs = sorted(df["freq"].unique())
    top_s = df["sharpe"].max()
    med_s = df["sharpe"].median()
    pct_pos = (df["sharpe"] > 0).mean() * 100

    summary = (
        f"{n:,} configurations  ·  {n_signals} signals  ·  {n_fam} families\n"
        f"Frequencies: {', '.join(freqs)}\n"
        f"Top Sharpe: {top_s:.2f}  ·  Median Sharpe: {med_s:.2f}  ·  "
        f"{pct_pos:.0f}% positive Sharpe"
    )
    ax.text(0.5, 0.35, summary, transform=ax.transAxes, ha="center",
            fontsize=13, color=TEXT_COLOR, linespacing=1.8)

    ax.text(0.5, 0.08, "NRT Research  ·  Binary Long/Cash  ·  20 bps costs  ·  1-bar lag",
            transform=ax.transAxes, ha="center", fontsize=10, color="#888888")
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_1_sharpe_dist(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    data = df["sharpe"].dropna()
    data_clipped = data.clip(-5, 5)
    ax.hist(data_clipped, bins=80, color=ACCENT, alpha=0.7, edgecolor="none")
    ax.axvline(0, color=ACCENT2, linestyle="--", linewidth=1.5, label="Zero")
    ax.axvline(data.median(), color=ACCENT3, linestyle="-", linewidth=1.5,
               label=f"Median: {data.median():.2f}")
    ax.set_xlabel("Annualized Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"Exhibit 1: Sharpe Distribution — {symbol} ({len(df):,} configs)", fontsize=14)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_2_cagr_vs_dd(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, ax = plt.subplots(figsize=(11, 7))
    sub = df[(df["sharpe"].between(-3, 3)) & (df["max_dd"] < 0)].copy()
    scatter = ax.scatter(sub["max_dd"] * 100, sub["cagr"] * 100,
                         c=sub["sharpe"], cmap="RdYlGn", s=8, alpha=0.5,
                         vmin=-1, vmax=2)
    plt.colorbar(scatter, ax=ax, label="Sharpe", shrink=0.8)
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title(f"Exhibit 2: CAGR vs Max Drawdown — {symbol}", fontsize=14)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_3_by_frequency(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    freq_order = ["1h", "4h", "1d"]
    groups = [df[df["freq"] == f]["sharpe"].dropna().values for f in freq_order if f in df["freq"].values]
    labels = [f for f in freq_order if f in df["freq"].values]

    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color=ACCENT3, linewidth=2))
    colors = ["#4ecdc4", "#45b7d1", "#96ceb4"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (g, l) in enumerate(zip(groups, labels), 1):
        med = np.median(g)
        ax.text(i, med + 0.05, f"med={med:.2f}", ha="center", fontsize=9, color=ACCENT3)

    ax.axhline(0, color=ACCENT2, linestyle="--", alpha=0.5)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"Exhibit 3: Sharpe by Frequency — {symbol}", fontsize=14)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_4_by_family(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    daily = df[df["freq"] == "1d"] if "1d" in df["freq"].values else df
    fam_med = daily.groupby("signal_family")["sharpe"].median().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(8, len(fam_med) * 0.25)))
    colors = [ACCENT if v > 0 else ACCENT2 for v in fam_med.values]
    ax.barh(range(len(fam_med)), fam_med.values, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(range(len(fam_med)))
    ax.set_yticklabels(fam_med.index, fontsize=8)
    ax.axvline(0, color=TEXT_COLOR, linestyle="-", alpha=0.3)
    ax.set_xlabel("Median Sharpe")
    freq_label = "Daily" if "1d" in df["freq"].values else "All"
    ax.set_title(f"Exhibit 4: Median Sharpe by Signal Family ({freq_label}) — {symbol}", fontsize=14)
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_5_sharpe_vs_tim(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, ax = plt.subplots(figsize=(11, 7))
    sub = df[df["sharpe"].between(-3, 3)].copy()
    scatter = ax.scatter(sub["time_in_market"] * 100, sub["sharpe"],
                         c=sub["cagr"] * 100, cmap="viridis", s=6, alpha=0.4,
                         vmin=-50, vmax=100)
    plt.colorbar(scatter, ax=ax, label="CAGR (%)", shrink=0.8)
    ax.axhline(0, color=ACCENT2, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time in Market (%)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"Exhibit 5: Sharpe vs Time-in-Market — {symbol}", fontsize=14)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_6_sharpe_vs_skew(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, ax = plt.subplots(figsize=(11, 7))
    sub = df[df["sharpe"].between(-3, 3) & df["skewness"].between(-5, 5)].copy()
    scatter = ax.scatter(sub["skewness"], sub["sharpe"],
                         c=sub["time_in_market"], cmap="plasma", s=6, alpha=0.4)
    plt.colorbar(scatter, ax=ax, label="Time in Market", shrink=0.8)
    ax.axhline(0, color=ACCENT2, linestyle="--", alpha=0.5)
    ax.axvline(0, color=TEXT_COLOR, linestyle="--", alpha=0.3)
    ax.set_xlabel("Return Skewness")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"Exhibit 6: Sharpe vs Skewness — {symbol}", fontsize=14)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_7_stop_comparison(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharey=True)
    metrics = ["sharpe", "cagr", "max_dd"]
    titles = ["Sharpe", "CAGR", "Max Drawdown"]
    stop_order = ["none", "pct5", "pct10", "pct20", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "atr4.0"]
    colors_map = {
        "none": "#888888",
        "pct5": "#ff6b6b", "pct10": "#ff9f43", "pct20": "#ffd93d",
        "atr1.5": "#4ecdc4", "atr2.0": "#45b7d1", "atr2.5": "#96ceb4",
        "atr3.0": "#a29bfe", "atr4.0": "#6c5ce7"
    }
    for ax, metric, title in zip(axes, metrics, titles):
        med = df.groupby("stop")[metric].median().reindex(stop_order)
        bars = ax.bar(range(len(med)), med.values, color=[colors_map.get(s, ACCENT) for s in stop_order],
                      alpha=0.8)
        ax.set_xticks(range(len(med)))
        ax.set_xticklabels(stop_order, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"Exhibit 7: Stop Variant Comparison — {symbol}", fontsize=14, y=1.02)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_8_matched_stops(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ATR stops vs no stop
    none_df = df[df["stop"] == "none"].set_index("label")
    for stop_name, color, ax_idx in [("atr2.0", ACCENT, 0), ("pct10", ACCENT2, 1)]:
        ax = axes[ax_idx]
        stop_df = df[df["stop"] == stop_name].set_index("label")
        common = none_df.index.intersection(stop_df.index)
        if len(common) == 0:
            continue
        x = none_df.loc[common, "sharpe"].values
        y = stop_df.loc[common, "sharpe"].values

        ax.scatter(x, y, c=ACCENT3, s=6, alpha=0.3)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "--", color=TEXT_COLOR, alpha=0.4)
        ax.set_xlabel("Sharpe (no stop)")
        ax.set_ylabel(f"Sharpe ({stop_name})")
        ax.set_title(f"{stop_name} vs None")
        ax.grid(True, alpha=0.2)

        pct_better = (y > x).mean() * 100
        ax.text(0.05, 0.95, f"{pct_better:.0f}% improved", transform=ax.transAxes,
                va="top", fontsize=10, color=ACCENT3)

    fig.suptitle(f"Exhibit 8: Matched Stop Comparison — {symbol}", fontsize=14, y=1.02)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_9_multiple_testing(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, ax = plt.subplots(figsize=(11, 6))

    n_tests = len(df)
    daily = df[df["freq"] == "1d"] if "1d" in df["freq"].values else df

    sharpes = daily["sharpe"].dropna().sort_values(ascending=False)
    ranks = np.arange(1, len(sharpes) + 1)

    bonferroni_threshold = stats.norm.ppf(1 - 0.05 / (2 * n_tests))
    holm_thresholds = stats.norm.ppf(1 - 0.05 / (2 * (n_tests - ranks + 1)))

    ax.plot(ranks, sharpes.values, color=ACCENT, linewidth=1.5, label="Observed Sharpe")
    ax.axhline(bonferroni_threshold, color=ACCENT2, linestyle="--", linewidth=1.5,
               label=f"Bonferroni ({bonferroni_threshold:.2f})")
    ax.plot(ranks, holm_thresholds, color=ACCENT3, linestyle=":", linewidth=1.5,
            label="Holm step-down")
    ax.axhline(0, color=TEXT_COLOR, linestyle="-", alpha=0.2)

    n_bonf = (sharpes.values > bonferroni_threshold).sum()
    n_holm = (sharpes.values > holm_thresholds[:len(sharpes)]).sum()
    ax.text(0.95, 0.95, f"Bonferroni survivors: {n_bonf}\nHolm survivors: {n_holm}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            color=ACCENT3, bbox=dict(boxstyle="round", facecolor=DARK_BG, edgecolor=GRID_COLOR))

    ax.set_xlabel("Rank")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"Exhibit 9: Multiple Testing Correction — {symbol} (N={n_tests:,})", fontsize=14)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, min(200, len(sharpes)))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_10_calmar_isolines(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, ax = plt.subplots(figsize=(11, 7))
    sub = df[(df["max_dd"] < 0) & (df["cagr"].between(-1, 3))].copy()

    dd_pct = sub["max_dd"].abs() * 100
    cagr_pct = sub["cagr"] * 100

    scatter = ax.scatter(dd_pct, cagr_pct, c=sub["sharpe"], cmap="RdYlGn",
                         s=8, alpha=0.5, vmin=-1, vmax=2)
    plt.colorbar(scatter, ax=ax, label="Sharpe", shrink=0.8)

    dd_range = np.linspace(1, dd_pct.max() if len(dd_pct) > 0 else 80, 200)
    for calmar, ls in [(0.5, ":"), (1.0, "--"), (2.0, "-")]:
        ax.plot(dd_range, calmar * dd_range, ls, color=TEXT_COLOR, alpha=0.4,
                label=f"Calmar={calmar:.1f}")

    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title(f"Exhibit 10: CAGR vs Max DD with Calmar Iso-Lines — {symbol}", fontsize=14)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _exhibit_11_dd_by_stop(pdf: PdfPages, df: pd.DataFrame, symbol: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    stop_order = ["none", "pct5", "pct10", "pct20", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "atr4.0"]
    groups = [df[df["stop"] == s]["max_dd"].dropna().values * 100 for s in stop_order if s in df["stop"].values]
    labels = [s for s in stop_order if s in df["stop"].values]

    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color=ACCENT3, linewidth=2))
    colors = ["#888888", "#ff6b6b", "#ff9f43", "#ffd93d",
              "#4ecdc4", "#45b7d1", "#96ceb4", "#a29bfe", "#6c5ce7"]
    for patch, color in zip(bp["boxes"], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title(f"Exhibit 11: Drawdown Distribution by Stop Type — {symbol}", fontsize=14)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def generate_report(symbol: str, results_csv: Path, output_pdf: Path):
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df):,} rows for {symbol}")

    with PdfPages(str(output_pdf)) as pdf:
        _title_page(pdf, symbol, df)
        _exhibit_1_sharpe_dist(pdf, df, symbol)
        _exhibit_2_cagr_vs_dd(pdf, df, symbol)
        _exhibit_3_by_frequency(pdf, df, symbol)
        _exhibit_4_by_family(pdf, df, symbol)
        _exhibit_5_sharpe_vs_tim(pdf, df, symbol)
        _exhibit_6_sharpe_vs_skew(pdf, df, symbol)
        _exhibit_7_stop_comparison(pdf, df, symbol)
        _exhibit_8_matched_stops(pdf, df, symbol)
        _exhibit_9_multiple_testing(pdf, df, symbol)
        _exhibit_10_calmar_isolines(pdf, df, symbol)
        _exhibit_11_dd_by_stop(pdf, df, symbol)

    print(f"Report saved: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Generate trend sweep PDF report")
    parser.add_argument("--symbol", required=True, help="e.g. BTC-USD")
    parser.add_argument("--results", default=None, help="Path to results_v2.csv")
    parser.add_argument("--output", default=None, help="Output PDF path")
    args = parser.parse_args()

    slug = args.symbol.replace("-", "").lower()
    if args.results:
        results_csv = Path(args.results)
    else:
        results_csv = PROJECT_ROOT / "artifacts" / "research" / "tsmom" / f"{slug}_trend_sweep" / "results_v2.csv"

    if args.output:
        output_pdf = Path(args.output)
    else:
        output_pdf = PROJECT_ROOT / "artifacts" / "research" / "tsmom" / f"{slug}_trend_sweep" / f"{slug}_trend_sweep_report.pdf"

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    generate_report(args.symbol, results_csv, output_pdf)


if __name__ == "__main__":
    main()
