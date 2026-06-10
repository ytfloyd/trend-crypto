#!/usr/bin/env python3
"""Generate cross-asset trend sweep summary PDF.

Compares findings across 21 assets to answer:
  1. Does frequency dominance hold? (daily > 4h > 1h)
  2. Do the same signal families cluster at the top?
  3. Does the ~40% TIM optimum persist?
  4. Do ATR stops consistently outperform fixed stops?

Usage:
    python scripts/generate_cross_asset_summary.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

TARGET_ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
    "ADA-USD", "LINK-USD", "DOT-USD", "AVAX-USD", "ICP-USD",
    "LTC-USD", "XLM-USD", "BCH-USD", "ATOM-USD", "ALGO-USD",
    "UNI-USD", "AAVE-USD", "COMP-USD", "FIL-USD",
    "GRT-USD", "ETC-USD",
]


def load_all() -> dict[str, pd.DataFrame]:
    root = PROJECT_ROOT / "artifacts" / "research" / "tsmom"
    data = {}
    for sym in TARGET_ASSETS:
        slug = sym.replace("-", "").lower() + "_trend_sweep"
        csv = root / slug / "results_v2.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            if len(df) > 10000:
                df["symbol"] = sym
                data[sym] = df
    return data


def _title_page(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.70, "Cross-Asset Trend Sweep", transform=ax.transAxes, ha="center",
            fontsize=40, fontweight="bold", color=ACCENT)
    ax.text(0.5, 0.58, "Comparative Analysis", transform=ax.transAxes, ha="center",
            fontsize=24, color=TEXT_COLOR)

    total = sum(len(df) for df in data.values())
    ax.text(0.5, 0.40,
            f"{len(data)} assets  ·  {total:,} total configurations\n"
            f"492 signals  ·  42 families  ·  9 stop variants  ·  3 frequencies\n"
            f"Binary long/cash  ·  20 bps costs  ·  1-bar lag",
            transform=ax.transAxes, ha="center", fontsize=13, color=TEXT_COLOR, linespacing=1.8)

    assets_str = "  ".join(s.replace("-USD", "") for s in data.keys())
    ax.text(0.5, 0.18, assets_str, transform=ax.transAxes, ha="center",
            fontsize=9, color="#888888", wrap=True)

    ax.text(0.5, 0.05, "NRT Research", transform=ax.transAxes, ha="center",
            fontsize=10, color="#888888")
    pdf.savefig(fig)
    plt.close(fig)


def _page_1_summary_table(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    """Master summary heatmap: asset vs key metrics."""
    rows = []
    for sym, df in data.items():
        rows.append({
            "Asset": sym.replace("-USD", ""),
            "Configs": len(df),
            "Med Sharpe": df["sharpe"].median(),
            "Top Sharpe": df["sharpe"].max(),
            "% Pos Sharpe": (df["sharpe"] > 0).mean() * 100,
            "Med CAGR": df["cagr"].median() * 100,
            "Med MaxDD": df["max_dd"].median() * 100,
            "Best Family (1d)": df[df["freq"] == "1d"].groupby("signal_family")["sharpe"].median().idxmax() if "1d" in df["freq"].values else "N/A",
        })
    summary = pd.DataFrame(rows).sort_values("Top Sharpe", ascending=False)

    fig, ax = plt.subplots(figsize=(11, max(7, len(summary) * 0.35)))
    ax.axis("off")
    ax.set_title("Exhibit A: Cross-Asset Summary Table", fontsize=16, pad=20)

    cols = ["Asset", "Top Sharpe", "Med Sharpe", "% Pos Sharpe", "Med CAGR", "Med MaxDD", "Best Family (1d)"]
    col_widths = [0.08, 0.10, 0.10, 0.12, 0.10, 0.10, 0.15]

    y_start = 0.92
    y_step = 0.038
    x_starts = [0.02]
    for w in col_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    for j, col in enumerate(cols):
        ax.text(x_starts[j], y_start, col, fontsize=9, fontweight="bold",
                color=ACCENT, transform=ax.transAxes)

    for i, (_, row) in enumerate(summary.iterrows()):
        y = y_start - (i + 1) * y_step
        for j, col in enumerate(cols):
            val = row[col]
            if isinstance(val, float):
                txt = f"{val:.2f}"
            else:
                txt = str(val)
            color = TEXT_COLOR
            if col == "Top Sharpe" and isinstance(val, float):
                color = ACCENT if val > 1.0 else (ACCENT3 if val > 0.5 else ACCENT2)
            elif col == "Med Sharpe" and isinstance(val, float):
                color = ACCENT if val > 0.3 else (ACCENT2 if val < 0 else TEXT_COLOR)
            ax.text(x_starts[j], y, txt, fontsize=8, color=color, transform=ax.transAxes)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_2_freq_dominance(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    """Does daily dominate across all assets?"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    # Panel A: median sharpe by freq per asset
    ax = axes[0]
    freq_colors = {"1h": "#4ecdc4", "4h": "#45b7d1", "1d": "#96ceb4"}
    assets = list(data.keys())
    x = np.arange(len(assets))
    width = 0.25

    for i, freq in enumerate(["1h", "4h", "1d"]):
        vals = []
        for sym in assets:
            sub = data[sym][data[sym]["freq"] == freq]
            vals.append(sub["sharpe"].median() if len(sub) > 0 else 0)
        ax.barh(x + i * width, vals, width, label=freq, color=freq_colors[freq], alpha=0.8)

    ax.set_yticks(x + width)
    ax.set_yticklabels([s.replace("-USD", "") for s in assets], fontsize=7)
    ax.axvline(0, color=TEXT_COLOR, alpha=0.3)
    ax.set_xlabel("Median Sharpe")
    ax.set_title("A: Median Sharpe by Frequency", fontsize=11)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    # Panel B: which freq wins per asset
    ax = axes[1]
    freq_wins = {"1h": 0, "4h": 0, "1d": 0}
    for sym, df in data.items():
        best_freq = df.groupby("freq")["sharpe"].median().idxmax()
        freq_wins[best_freq] += 1

    freqs = ["1h", "4h", "1d"]
    vals = [freq_wins[f] for f in freqs]
    bars = ax.bar(freqs, vals, color=[freq_colors[f] for f in freqs], alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(v), ha="center", fontsize=12, color=ACCENT3)
    ax.set_ylabel("# Assets Where This Freq Wins")
    ax.set_title("B: Frequency Dominance", fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Exhibit B: Frequency Dominance Across Assets", fontsize=14, y=1.02)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_3_family_heatmap(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    """Heatmap of signal family median Sharpe across assets (daily only)."""
    daily_data = {}
    for sym, df in data.items():
        daily = df[df["freq"] == "1d"]
        if len(daily) > 0:
            daily_data[sym.replace("-USD", "")] = daily.groupby("signal_family")["sharpe"].median()

    matrix = pd.DataFrame(daily_data)
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(11, max(8, len(matrix) * 0.25)))
    im = ax.imshow(matrix.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1.5)
    plt.colorbar(im, ax=ax, label="Median Sharpe", shrink=0.6)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)

    ax.set_title("Exhibit C: Signal Family × Asset Heatmap (Daily, Median Sharpe)", fontsize=14)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_4_tim_optimum(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    """Does ~40% TIM optimum persist?"""
    fig, axes = plt.subplots(2, 3, figsize=(11, 8))
    axes = axes.flatten()

    showcased = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD", "DOGE-USD"]

    for idx, sym in enumerate(showcased):
        if sym not in data:
            continue
        ax = axes[idx]
        df = data[sym]
        daily = df[df["freq"] == "1d"]
        if len(daily) == 0:
            daily = df

        tim_bins = pd.cut(daily["time_in_market"], bins=20)
        grouped = daily.groupby(tim_bins, observed=False)["sharpe"].median()

        x = [interval.mid for interval in grouped.index]
        ax.plot([v * 100 for v in x], grouped.values, color=ACCENT, linewidth=2)
        ax.fill_between([v * 100 for v in x], grouped.values, alpha=0.2, color=ACCENT)
        ax.axhline(0, color=ACCENT2, linestyle="--", alpha=0.5)

        peak_idx = grouped.values.argmax() if len(grouped) > 0 else 0
        if peak_idx < len(x):
            ax.axvline(x[peak_idx] * 100, color=ACCENT3, linestyle=":", alpha=0.7)
            ax.text(x[peak_idx] * 100, grouped.values[peak_idx],
                    f" {x[peak_idx]*100:.0f}%", fontsize=8, color=ACCENT3)

        ax.set_title(sym.replace("-USD", ""), fontsize=10)
        ax.set_xlabel("TIM (%)", fontsize=8)
        ax.set_ylabel("Med Sharpe", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Exhibit D: Sharpe vs Time-in-Market by Asset (Daily)", fontsize=14, y=1.02)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_5_stop_analysis(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    """ATR vs fixed vs no stop across all assets."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    # Panel A: avg improvement from ATR 2.0 vs none
    ax = axes[0]
    improvements = []
    asset_labels = []
    for sym, df in data.items():
        none_med = df[df["stop"] == "none"]["sharpe"].median()
        atr2_med = df[df["stop"] == "atr2.0"]["sharpe"].median()
        improvements.append(atr2_med - none_med)
        asset_labels.append(sym.replace("-USD", ""))

    order = np.argsort(improvements)[::-1]
    colors = [ACCENT if improvements[i] > 0 else ACCENT2 for i in order]
    ax.barh(range(len(order)), [improvements[i] for i in order],
            color=colors, alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([asset_labels[i] for i in order], fontsize=7)
    ax.axvline(0, color=TEXT_COLOR, alpha=0.3)
    ax.set_xlabel("Sharpe Improvement (ATR 2.0 - None)")
    ax.set_title("A: ATR 2.0 Stop Impact", fontsize=11)
    ax.grid(True, alpha=0.2, axis="x")

    # Panel B: stop type summary across all assets
    ax = axes[1]
    all_df = pd.concat(data.values())
    stop_order = ["none", "pct5", "pct10", "pct20", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "atr4.0"]
    stop_meds = all_df.groupby("stop")["sharpe"].median().reindex(stop_order)
    stop_colors = ["#888888", "#ff6b6b", "#ff9f43", "#ffd93d",
                   "#4ecdc4", "#45b7d1", "#96ceb4", "#a29bfe", "#6c5ce7"]
    ax.bar(range(len(stop_meds)), stop_meds.values,
           color=stop_colors[:len(stop_meds)], alpha=0.8)
    ax.set_xticks(range(len(stop_meds)))
    ax.set_xticklabels(stop_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median Sharpe (all assets)")
    ax.set_title("B: Global Stop Comparison", fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Exhibit E: Stop Variant Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_6_top_sharpe_scatter(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    """Top Sharpe per asset scatter."""
    fig, ax = plt.subplots(figsize=(11, 7))

    for sym, df in data.items():
        top = df.nlargest(1, "sharpe").iloc[0]
        color = {"1d": "#96ceb4", "4h": "#45b7d1", "1h": "#4ecdc4"}.get(top["freq"], ACCENT)
        ax.scatter(top["max_dd"] * 100, top["sharpe"], s=80, color=color,
                   edgecolors="white", linewidth=0.5, zorder=5)
        ax.annotate(sym.replace("-USD", ""), (top["max_dd"] * 100, top["sharpe"]),
                    fontsize=7, color=TEXT_COLOR, xytext=(5, 5),
                    textcoords="offset points")

    ax.axhline(1.0, color=ACCENT3, linestyle="--", alpha=0.4, label="Sharpe=1.0")
    ax.set_xlabel("Max Drawdown of Best Config (%)")
    ax.set_ylabel("Top Sharpe")
    ax.set_title("Exhibit F: Best Configuration per Asset", fontsize=14)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_7_pct_positive(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    """% of configs with positive Sharpe by freq."""
    fig, ax = plt.subplots(figsize=(11, 6))

    freq_colors = {"1h": "#4ecdc4", "4h": "#45b7d1", "1d": "#96ceb4"}
    assets = list(data.keys())
    x = np.arange(len(assets))
    width = 0.25

    for i, freq in enumerate(["1h", "4h", "1d"]):
        vals = []
        for sym in assets:
            sub = data[sym][data[sym]["freq"] == freq]
            vals.append((sub["sharpe"] > 0).mean() * 100 if len(sub) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=freq, color=freq_colors[freq], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace("-USD", "") for s in assets], rotation=45, ha="right", fontsize=7)
    ax.axhline(50, color=ACCENT2, linestyle="--", alpha=0.5, label="50%")
    ax.set_ylabel("% Configs with Sharpe > 0")
    ax.set_title("Exhibit G: Fraction of Profitable Configs by Frequency", fontsize=14)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COLOR, fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_8_bonferroni_survivors(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    """How many configs survive Bonferroni per asset?"""
    fig, ax = plt.subplots(figsize=(11, 6))

    assets = list(data.keys())
    survivors = []
    for sym in assets:
        df = data[sym]
        n = len(df)
        threshold = stats.norm.ppf(1 - 0.05 / (2 * n))
        n_surv = (df["sharpe"] > threshold).sum()
        survivors.append(n_surv)

    colors = [ACCENT if s > 0 else ACCENT2 for s in survivors]
    ax.bar(range(len(assets)), survivors, color=colors, alpha=0.8)
    ax.set_xticks(range(len(assets)))
    ax.set_xticklabels([s.replace("-USD", "") for s in assets], rotation=45, ha="right", fontsize=8)

    for i, v in enumerate(survivors):
        if v > 0:
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=8, color=ACCENT3)

    ax.set_ylabel("# Configs Surviving Bonferroni")
    ax.set_title("Exhibit H: Bonferroni Survivors per Asset (α=0.05)", fontsize=14)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_9_key_findings(pdf: PdfPages, data: dict[str, pd.DataFrame]):
    """Summary findings page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    all_df = pd.concat(data.values())

    # Q1: Frequency dominance
    freq_med = all_df.groupby("freq")["sharpe"].median()
    best_freq = freq_med.idxmax()
    freq_str = ", ".join(f"{f}: {v:.3f}" for f, v in freq_med.sort_values(ascending=False).items())

    # Q2: Top families
    daily = all_df[all_df["freq"] == "1d"]
    fam_med = daily.groupby("signal_family")["sharpe"].median().sort_values(ascending=False)
    top5_fam = ", ".join(f"{f} ({v:.2f})" for f, v in fam_med.head(5).items())

    # Q3: TIM optimum
    daily_pos = daily[daily["sharpe"] > 0]
    if len(daily_pos) > 0:
        tim_med = daily_pos["time_in_market"].median() * 100
    else:
        tim_med = 0

    # Q4: Stop analysis
    none_med = all_df[all_df["stop"] == "none"]["sharpe"].median()
    atr_med = all_df[all_df["stop"].str.startswith("atr")]["sharpe"].median()
    pct_med = all_df[all_df["stop"].str.startswith("pct")]["sharpe"].median()

    # Q5: Bonferroni
    bonf_total = 0
    for sym, df in data.items():
        n = len(df)
        threshold = stats.norm.ppf(1 - 0.05 / (2 * n))
        bonf_total += (df["sharpe"] > threshold).sum()

    n_assets_pos_med = sum(1 for df in data.values() if df["sharpe"].median() > 0)

    findings = [
        ("Q1: Frequency Dominance", f"Best median freq: {best_freq} ({freq_str})"),
        ("Q2: Top Signal Families (Daily)", top5_fam),
        ("Q3: TIM Optimum", f"Median TIM of profitable configs: {tim_med:.0f}%"),
        ("Q4: Stop Impact", f"None: {none_med:.3f}  |  ATR avg: {atr_med:.3f}  |  Fixed avg: {pct_med:.3f}"),
        ("Q5: Bonferroni Survivors", f"{bonf_total} total configs across all assets"),
        ("Q6: Asset Breadth", f"{n_assets_pos_med}/{len(data)} assets have positive median Sharpe"),
    ]

    ax.text(0.5, 0.92, "Key Findings", transform=ax.transAxes, ha="center",
            fontsize=24, fontweight="bold", color=ACCENT)

    for i, (title, body) in enumerate(findings):
        y = 0.80 - i * 0.12
        ax.text(0.08, y, title, transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=ACCENT3)
        ax.text(0.08, y - 0.04, body, transform=ax.transAxes, fontsize=11,
                color=TEXT_COLOR)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    data = load_all()
    print(f"Loaded {len(data)} assets")

    output_pdf = PROJECT_ROOT / "artifacts" / "research" / "tsmom" / "cross_asset_trend_sweep_summary.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_pdf)) as pdf:
        _title_page(pdf, data)
        _page_1_summary_table(pdf, data)
        _page_2_freq_dominance(pdf, data)
        _page_3_family_heatmap(pdf, data)
        _page_4_tim_optimum(pdf, data)
        _page_5_stop_analysis(pdf, data)
        _page_6_top_sharpe_scatter(pdf, data)
        _page_7_pct_positive(pdf, data)
        _page_8_bonferroni_survivors(pdf, data)
        _page_9_key_findings(pdf, data)

    print(f"Report saved: {output_pdf}")


if __name__ == "__main__":
    main()
