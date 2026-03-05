"""
Generate PDF report for the El Farol Crowding Overlay v0 research.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "research"
    / "crowding"
)
OUTPUT_PDF = ARTIFACT_DIR / "crowding_overlay_v0_report.pdf"

# Consistent styling
COLOR_BG = "#FAFAFA"
COLOR_TITLE = "#1a1a2e"
COLOR_ACCENT = "#3b82f6"
COLOR_GREEN = "#22c55e"
COLOR_RED = "#ef4444"
COLOR_ORANGE = "#FFA726"
COLOR_PINK = "#EC407A"
COLOR_GRAY = "#666666"


def title_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top accent bar
    ax.axhspan(0.82, 0.84, color=COLOR_ACCENT, alpha=0.9)

    ax.text(0.5, 0.72, "El Farol Crowding Overlay", fontsize=32,
            fontweight="bold", ha="center", va="center", color=COLOR_TITLE)
    ax.text(0.5, 0.63, "Can signal unanimity predict trend-following drawdowns in crypto?",
            fontsize=14, ha="center", va="center", color=COLOR_GRAY, style="italic")

    ax.axhspan(0.56, 0.565, xmin=0.3, xmax=0.7, color=COLOR_ACCENT, alpha=0.4)

    body = (
        "Hypothesis:  When trend signals are unanimous across the universe,\n"
        "the trade is crowded and subsequent returns degrade (Arthur / El Farol).\n\n"
        "Method:  EMAC(5/40) trend strategy on daily crypto bars, long-only,\n"
        "inverse-volatility weighted, BTC regime filter, 20 bps costs.\n\n"
        "Overlays tested:  Signal breadth, return breadth, signal dispersion,\n"
        "and combined variants at multiple thresholds and fade strengths."
    )
    ax.text(0.5, 0.40, body, fontsize=12, ha="center", va="center",
            color=COLOR_TITLE, family="monospace", linespacing=1.6)

    ax.text(0.5, 0.12, "Research Script:  scripts/research/crowding/run_crowding_overlay_v0.py",
            fontsize=9, ha="center", color=COLOR_GRAY, family="monospace")
    ax.text(0.5, 0.08, "NRT Research  |  v0  |  2026-02",
            fontsize=9, ha="center", color=COLOR_GRAY)

    pdf.savefig(fig)
    plt.close(fig)


def page_theory(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Theoretical Background: The El Farol Problem",
            fontsize=18, fontweight="bold", ha="center", va="top", color=COLOR_TITLE)

    ax.axhspan(0.92, 0.925, xmin=0.15, xmax=0.85, color=COLOR_ACCENT, alpha=0.3,
               transform=ax.transAxes)

    blocks = [
        ("The El Farol Problem (W. Brian Arthur, 1994)", 0.88,
         "100 people decide independently whether to go to a bar.  If >60 go, it's\n"
         "overcrowded and nobody enjoys it.  If nearly everyone expects it to be\n"
         "empty, they all go — negating their own forecast.\n\n"
         "Key insight:  When agents form identical expectations, those expectations\n"
         "self-negate.  No single forecasting model can dominate — an ecology of\n"
         "heterogeneous models is the stable outcome."),

        ("Application to Trend Following", 0.60,
         "When all trend signals agree (high breadth), the position is crowded.\n"
         "All participants hold the same direction.  A reversal triggers\n"
         "simultaneous exits → amplified drawdowns.\n\n"
         "Counter-hypothesis:  In crypto, broad-based trends reflect genuine\n"
         "regime shifts (halving cycles, institutional adoption).  Breadth\n"
         "may be an alpha signal, not a crowding signal."),

        ("What We Test", 0.33,
         "Three crowding indicators applied as multiplicative weight overlays:\n\n"
         "  1. Signal breadth  — fraction of universe with positive EMAC signal\n"
         "  2. Return breadth  — fraction with positive 20-day trailing return\n"
         "  3. Signal dispersion — cross-sectional std of signals (low = crowded)\n\n"
         "Each tested at 3 thresholds × 3 fade strengths = 9 variants per type,\n"
         "plus 4 combined variants.  Total: 31 overlay configurations."),
    ]

    for title, y, body in blocks:
        ax.text(0.02, y, title, fontsize=12, fontweight="bold",
                va="top", color=COLOR_ACCENT)
        ax.text(0.04, y - 0.04, body, fontsize=9.5, va="top",
                color=COLOR_TITLE, family="monospace", linespacing=1.5)

    pdf.savefig(fig)
    plt.close(fig)


def page_results_table(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.set_facecolor("white")

    ax_title = fig.add_axes([0, 0.92, 1, 0.08])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, "Results: All Overlay Variants (sorted by Sharpe)",
                  fontsize=16, fontweight="bold", ha="center", va="center",
                  color=COLOR_TITLE)

    ax = fig.add_axes([0.04, 0.04, 0.92, 0.86])
    ax.axis("off")

    cols = ["label", "sharpe", "sharpe_delta", "cagr", "max_dd",
            "dd_improvement", "vol", "avg_turnover"]
    headers = ["Strategy", "Sharpe", "dSh", "CAGR", "MaxDD", "dDD", "Vol", "TO"]

    n_rows = len(df)
    n_cols = len(headers)

    table_data = []
    cell_colors = []
    for _, row in df.iterrows():
        is_baseline = "Baseline" in str(row["label"])
        lbl = str(row["label"])
        if len(lbl) > 32:
            lbl = lbl[:32]

        r = [
            lbl,
            f"{row['sharpe']:.2f}",
            f"{row['sharpe_delta']:+.2f}" if not is_baseline else "--",
            f"{row['cagr']:.1%}",
            f"{row['max_dd']:.1%}",
            f"{row['dd_improvement']:+.1%}" if not is_baseline else "--",
            f"{row['vol']:.1%}",
            f"{row['avg_turnover']:.3f}",
        ]
        table_data.append(r)

        if is_baseline:
            cell_colors.append(["#E3F2FD"] * n_cols)
        elif row["sharpe_delta"] > 0.02:
            cell_colors.append(["#E8F5E9"] * n_cols)
        elif row["sharpe_delta"] < -0.05:
            cell_colors.append(["#FFF3E0"] * n_cols)
        else:
            cell_colors.append(["white"] * n_cols)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=[COLOR_ACCENT] * n_cols,
        loc="upper center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white", fontsize=8)
            cell.set_edgecolor("white")
        else:
            cell.set_edgecolor("#E0E0E0")
        if col == 0:
            cell.set_text_props(ha="left", fontsize=6.5)
            cell._loc = "left"

    # Legend at bottom
    ax.text(0.02, -0.02, "Green = Sharpe improved > +0.02  |  Blue = Baseline  |  Orange = Sharpe degraded > -0.05",
            fontsize=7, color=COLOR_GRAY, transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)


def page_equity_curves(pdf: PdfPages) -> None:
    img_path = ARTIFACT_DIR / "crowding_overlay_v0.png"
    if not img_path.exists():
        return

    fig = plt.figure(figsize=(11, 8.5))
    fig.set_facecolor("white")

    ax_title = fig.add_axes([0, 0.92, 1, 0.08])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, "Equity Curves, Indicators, and Overlay Impact",
                  fontsize=16, fontweight="bold", ha="center", va="center",
                  color=COLOR_TITLE)

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
    ax.axis("off")
    img = plt.imread(str(img_path))
    ax.imshow(img, aspect="auto")

    pdf.savefig(fig)
    plt.close(fig)


def page_conditional(pdf: PdfPages) -> None:
    img_path = ARTIFACT_DIR / "crowding_conditional_v0.png"
    if not img_path.exists():
        return

    fig = plt.figure(figsize=(11, 8.5))
    fig.set_facecolor("white")

    ax_title = fig.add_axes([0, 0.90, 1, 0.10])
    ax_title.axis("off")
    ax_title.text(0.5, 0.7, "Conditional Performance: Crowded vs Not Crowded",
                  fontsize=16, fontweight="bold", ha="center", va="center",
                  color=COLOR_TITLE)
    ax_title.text(0.5, 0.2,
                  "Signal breadth = fraction of universe with positive EMAC signal.  "
                  "Bars show annualized Sharpe ratio of baseline strategy\n"
                  "returns on days when breadth exceeds the threshold (Crowded) vs days when it does not.",
                  fontsize=9, ha="center", va="center", color=COLOR_GRAY)

    ax = fig.add_axes([0.05, 0.08, 0.90, 0.78])
    ax.axis("off")
    img = plt.imread(str(img_path))
    ax.imshow(img, aspect="auto")

    pdf.savefig(fig)
    plt.close(fig)


def page_key_findings(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Key Findings & Interpretation",
            fontsize=18, fontweight="bold", ha="center", va="top", color=COLOR_TITLE)

    ax.axhspan(0.92, 0.925, xmin=0.15, xmax=0.85, color=COLOR_ACCENT, alpha=0.3,
               transform=ax.transAxes)

    baseline = df.loc[df["label"].str.contains("Baseline")].iloc[0]
    best = df.iloc[0]

    blocks = [
        ("1. The El Farol Hypothesis Is Partially Inverted in Crypto", 0.88,
         "Conditional analysis shows that when signal breadth > 90% (unanimous\n"
         "trends), the baseline strategy Sharpe is 0.92.  When breadth < 90%,\n"
         "Sharpe drops to -0.51.  Broad trend agreement is the PROFIT CENTER,\n"
         "not a danger signal.  This differs from equity markets where crowded\n"
         "momentum trades attract mean-reversion capital."),

        ("2. Gentle Fading at Extremes Does Help", 0.66,
         f"Best overlay: Return Breadth > 90%, 70% fade\n"
         f"    Sharpe:  {baseline['sharpe']:.2f} -> {best['sharpe']:.2f}  ({best['sharpe_delta']:+.2f})\n"
         f"    MaxDD:   {baseline['max_dd']:.1%} -> {best['max_dd']:.1%}  ({best['dd_improvement']:+.1%})\n"
         f"    Vol:     {baseline['vol']:.1%} -> {best['vol']:.1%}\n\n"
         "A moderate fade at the most extreme unanimity captures euphoric tops\n"
         "without cutting exposure during normal bull phases."),

        ("3. Signal Dispersion Is a Cleaner Crowding Measure", 0.42,
         "Low cross-sectional signal dispersion (everyone has similar signal\n"
         "magnitude, not just direction) improves Sharpe by +0.02 to +0.03\n"
         "with 3-6% max DD improvement.  This is more robust than breadth\n"
         "because it measures conviction uniformity, not just direction."),

        ("4. Aggressive Fading Destroys Alpha", 0.22,
         "Signal breadth overlays with strong fading (70%) reduce Sharpe by\n"
         "up to -0.24.  The strategy NEEDS broad-based trends to work.\n"
         "Cutting exposure when breadth is high kills the strategy's edge.\n\n"
         "Takeaway: Use crowding overlays as a GENTLE risk reducer, not a\n"
         "binary on/off switch.  Optimal fade strength is 50-70% at 90%+."),
    ]

    for title, y, body in blocks:
        ax.text(0.02, y, title, fontsize=11, fontweight="bold",
                va="top", color=COLOR_ACCENT)
        ax.text(0.04, y - 0.035, body, fontsize=9, va="top",
                color=COLOR_TITLE, family="monospace", linespacing=1.5)

    pdf.savefig(fig)
    plt.close(fig)


def page_next_steps(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Next Steps & Recommendations",
            fontsize=18, fontweight="bold", ha="center", va="top", color=COLOR_TITLE)

    ax.axhspan(0.92, 0.925, xmin=0.15, xmax=0.85, color=COLOR_ACCENT, alpha=0.3,
               transform=ax.transAxes)

    blocks = [
        ("Promote to Production (Low Risk)", 0.86,
         "  - Add signal dispersion overlay to existing risk management stack\n"
         "    (src/risk/).  Conservative: SigDisp < 0.15, 50% fade.\n"
         "  - Integrate return breadth at 90% threshold as a position-sizing\n"
         "    modifier in the portfolio constructor."),

        ("Further Research (Medium Priority)", 0.68,
         "  - Test on Sornette LPPL and Kuma Trend baselines — different\n"
         "    signal types may have different crowding dynamics.\n"
         "  - Add perpetual funding rate as a direct positioning measure\n"
         "    (requires new data pipeline).\n"
         "  - Test asymmetric overlays: only fade above 95th percentile,\n"
         "    not linearly above threshold.\n"
         "  - Multi-frequency version: crowding at 4h vs 1d granularity."),

        ("Regime-Conditional Kelly (Quick Win)", 0.44,
         "  - Wire Kelly fraction to regime state: full Kelly in NORMAL,\n"
         "    half-Kelly in CRISIS, quarter-Kelly in high-uncertainty.\n"
         "  - This is ~20 lines of code connecting existing Kelly and\n"
         "    regime detection infrastructure."),

        ("What NOT To Do", 0.28,
         "  - Don't use signal breadth as a binary on/off switch.\n"
         "  - Don't implement adaptive signal weighting (overfit magnet)\n"
         "    without a year of walk-forward validation.\n"
         "  - Don't conflate crypto breadth with equity crowding — the\n"
         "    dynamics are structurally different (fewer participants,\n"
         "    higher correlation, reflexive bubble dynamics)."),
    ]

    for title, y, body in blocks:
        ax.text(0.02, y, title, fontsize=11, fontweight="bold",
                va="top", color=COLOR_ACCENT)
        ax.text(0.04, y - 0.035, body, fontsize=9, va="top",
                color=COLOR_TITLE, family="monospace", linespacing=1.5)

    pdf.savefig(fig)
    plt.close(fig)


def page_summary_table(pdf: PdfPages, df: pd.DataFrame) -> None:
    """Top-10 overlays vs baseline — focused comparison."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.set_facecolor("white")

    ax_title = fig.add_axes([0, 0.90, 1, 0.10])
    ax_title.axis("off")
    ax_title.text(0.5, 0.6, "Summary: Best Overlays vs Baseline",
                  fontsize=16, fontweight="bold", ha="center", va="center",
                  color=COLOR_TITLE)
    ax_title.text(0.5, 0.15,
                  "Top overlays ranked by Sharpe ratio.  Only variants that improve on baseline shown.",
                  fontsize=10, ha="center", va="center", color=COLOR_GRAY)

    ax = fig.add_axes([0.06, 0.08, 0.88, 0.78])
    ax.axis("off")

    # Filter to improved variants + baseline
    baseline_row = df.loc[df["label"].str.contains("Baseline")].iloc[0]
    improved = df.loc[df["sharpe_delta"] > 0].head(10)
    show_df = pd.concat([improved, df.loc[df["label"].str.contains("Baseline")]]).reset_index(drop=True)

    cols = ["Strategy", "Sharpe", "\u0394Sh", "CAGR", "MaxDD", "\u0394DD", "Vol"]
    table_data = []
    cell_colors = []

    for _, row in show_df.iterrows():
        is_base = "Baseline" in str(row["label"])
        lbl = str(row["label"])
        if len(lbl) > 35:
            lbl = lbl[:35]

        r = [
            lbl,
            f"{row['sharpe']:.2f}",
            f"{row['sharpe_delta']:+.2f}" if not is_base else "--",
            f"{row['cagr']:.1%}",
            f"{row['max_dd']:.1%}",
            f"{row['dd_improvement']:+.1%}" if not is_base else "--",
            f"{row['vol']:.1%}",
        ]
        table_data.append(r)

        if is_base:
            cell_colors.append(["#E3F2FD"] * len(cols))
        elif row["sharpe_delta"] >= 0.03:
            cell_colors.append(["#C8E6C9"] * len(cols))
        else:
            cell_colors.append(["#E8F5E9"] * len(cols))

    table = ax.table(
        cellText=table_data,
        colLabels=cols,
        cellColours=cell_colors,
        colColours=[COLOR_ACCENT] * len(cols),
        loc="upper center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white", fontsize=10)
            cell.set_edgecolor("white")
        else:
            cell.set_edgecolor("#E0E0E0")
        if col == 0:
            cell.set_text_props(ha="left", fontsize=8.5)
            cell._loc = "left"

    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(ARTIFACT_DIR / "crowding_overlay_results_v0.csv")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(OUTPUT_PDF)) as pdf:
        title_page(pdf)
        page_theory(pdf)
        page_summary_table(pdf, df)
        page_equity_curves(pdf)
        page_conditional(pdf)
        page_results_table(pdf, df)
        page_key_findings(pdf, df)
        page_next_steps(pdf)

    print(f"PDF report written: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
