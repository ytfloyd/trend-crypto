"""
PDF report generator for logistic regression filter experiment.

Produces a multi-page ReportLab PDF with:
  1. Executive summary + ablation table
  2. Equity curves + drawdown comparison
  3. Model diagnostics (AUC, calibration, coefficients)
  4. Threshold sweep analysis
  5. Exposure and turnover comparison
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .model import ModelOutput

# ── Colour palette ────────────────────────────────────────────────────
BLUE = colors.Color(0.06, 0.18, 0.37)
BLUE_LIGHT = colors.Color(0.20, 0.40, 0.65)
GOLD = colors.Color(0.76, 0.63, 0.33)
GRAY = colors.Color(0.55, 0.55, 0.55)
GREEN = colors.Color(0.14, 0.55, 0.14)
RED = colors.Color(0.70, 0.15, 0.15)

MPL_BLUE = "#0F2E5E"
MPL_GOLD = "#C2A054"
MPL_GREEN = "#238B23"
MPL_RED = "#B32626"
VARIANT_COLORS = {
    "baseline": "#888888",
    "filter": MPL_BLUE,
    "filter_sizing": MPL_GOLD,
    "filter_sizing_regime": MPL_GREEN,
}


def _styles() -> dict:
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=ss["Title"], fontSize=22,
                                textColor=BLUE, spaceAfter=6),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14,
                             textColor=BLUE, spaceAfter=8, spaceBefore=14),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11,
                             textColor=BLUE_LIGHT, spaceAfter=6, spaceBefore=10),
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontSize=9,
                               leading=13, alignment=TA_JUSTIFY, spaceAfter=6),
        "caption": ParagraphStyle("caption", parent=ss["BodyText"], fontSize=8,
                                  textColor=GRAY, alignment=TA_CENTER,
                                  spaceAfter=10, spaceBefore=2),
        "body_bold": ParagraphStyle("body_bold", parent=ss["BodyText"],
                                    fontSize=9, leading=13, spaceAfter=6),
    }


def _fig_to_image(fig, ratio: float = 0.5) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w = 6.5 * inch
    return Image(buf, width=w, height=w * ratio)


def _make_table(headers: list[str], rows: list[list], col_widths=None,
                highlight_row: int | None = None) -> Table:
    data = [headers] + rows
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.8, 0.8, 0.8)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if highlight_row is not None:
        style_cmds.append(
            ("BACKGROUND", (0, highlight_row + 1), (-1, highlight_row + 1),
             colors.Color(0.85, 0.93, 0.85))
        )
    return Table(data, colWidths=col_widths, style=TableStyle(style_cmds))


# ── Chart functions ───────────────────────────────────────────────────

def _chart_equity(equity_curves: dict[str, pd.Series]) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})
    for name, eq in equity_curves.items():
        c = VARIANT_COLORS.get(name, "#333333")
        ax1.plot(eq.index, eq.values, color=c, linewidth=1.2, label=name)

    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)

    for name, eq in equity_curves.items():
        dd = eq / eq.cummax() - 1.0
        c = VARIANT_COLORS.get(name, "#333333")
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color=c)
        ax2.plot(dd.index, dd.values, color=c, linewidth=0.7)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Equity Curves & Drawdown by Variant", fontsize=11, y=0.98)
    fig.tight_layout()
    return fig


def _chart_exposure(backtest_dfs: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True,
                                    gridspec_kw={"hspace": 0.08})
    for name, bt in backtest_dfs.items():
        c = VARIANT_COLORS.get(name, "#333333")
        ts = pd.to_datetime(bt["ts"])
        ax1.plot(ts, bt["gross_exposure"], color=c, linewidth=0.8, label=name)
        ax2.plot(ts, bt["turnover"].rolling(21).mean(), color=c, linewidth=0.8)
    ax1.set_ylabel("Gross Exposure")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("Turnover (21d MA)")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Exposure & Turnover", fontsize=11, y=0.98)
    fig.tight_layout()
    return fig


def _chart_fold_metrics(model_output: ModelOutput) -> plt.Figure:
    df = model_output.fold_metrics_df()
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))

    axes[0].bar(range(len(df)), df["auc"], color=MPL_BLUE, alpha=0.8)
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_title("AUC by Fold", fontsize=9)
    axes[0].set_xlabel("Fold")

    axes[1].bar(range(len(df)), df["log_loss"], color=MPL_GOLD, alpha=0.8)
    axes[1].set_title("Log Loss by Fold", fontsize=9)
    axes[1].set_xlabel("Fold")

    axes[2].bar(range(len(df)), df["brier"], color=MPL_GREEN, alpha=0.8)
    axes[2].set_title("Brier Score by Fold", fontsize=9)
    axes[2].set_xlabel("Fold")

    fig.tight_layout()
    return fig


def _chart_coefficients(model_output: ModelOutput) -> plt.Figure:
    coef_df = model_output.coefficient_summary()
    coef_df = coef_df.head(15)  # top 15 by importance

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y_pos = range(len(coef_df))
    colors_list = [MPL_GREEN if v > 0 else MPL_RED for v in coef_df["coef_mean"]]
    ax.barh(y_pos, coef_df["coef_mean"], xerr=coef_df["coef_std"],
            color=colors_list, alpha=0.8, capsize=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df["feature"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("Coefficient Means ± Std (Top 15 by |coef|)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def _chart_threshold_sweep(sweep_results: list[dict]) -> plt.Figure:
    df = pd.DataFrame(sweep_results)
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))

    axes[0].plot(df["p_enter"], df["sharpe"], "o-", color=MPL_BLUE, markersize=5)
    axes[0].set_xlabel("p_enter")
    axes[0].set_ylabel("Sharpe")
    axes[0].set_title("Sharpe vs Threshold", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["p_enter"], df["cagr"], "s-", color=MPL_GREEN, markersize=5)
    axes[1].set_xlabel("p_enter")
    axes[1].set_ylabel("CAGR")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[1].set_title("CAGR vs Threshold", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["p_enter"], df["max_dd"], "^-", color=MPL_RED, markersize=5)
    axes[2].set_xlabel("p_enter")
    axes[2].set_ylabel("Max DD")
    axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[2].set_title("Max DD vs Threshold", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── Main report builder ───────────────────────────────────────────────

def generate_report(
    results: dict[str, dict],
    sweep_results: list[dict],
    equity_curves: dict[str, pd.Series],
    backtest_dfs: dict[str, pd.DataFrame],
    model_output: ModelOutput,
    cfg: dict,
    out_path: str | Path = "logreg_filter_report.pdf",
) -> None:
    """Generate the full experiment PDF report."""
    out_path = Path(out_path)
    sty = _styles()

    frame = Frame(0.75 * inch, 0.75 * inch,
                  letter[0] - 1.5 * inch, letter[1] - 1.5 * inch)
    doc = BaseDocTemplate(str(out_path), pagesize=letter)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    story: list = []

    # ── Page 1: Title + Executive Summary ──
    story.append(Spacer(1, 40))
    story.append(Paragraph("Logistic Regression Probability Filter", sty["title"]))
    story.append(Paragraph("Experiment Report", sty["h2"]))
    story.append(Spacer(1, 20))

    n_folds = len(model_output.fold_results)
    mean_auc = np.nanmean([fr.auc for fr in model_output.fold_results])
    story.append(Paragraph(
        f"Walk-forward evaluation across {n_folds} folds with mean OOS AUC = "
        f"{mean_auc:.3f}. The model is tested as an entry filter, conviction "
        "sizer, and regime throttle on a top-N momentum base strategy.",
        sty["body"],
    ))

    story.append(Paragraph("Ablation Results", sty["h2"]))

    headers = ["Variant", "CAGR", "Vol", "Sharpe", "Sortino", "Max DD",
               "Turnover", "Exposure"]
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            f"{m.get('cagr', 0):.1%}",
            f"{m.get('vol', 0):.1%}",
            f"{m.get('sharpe', 0):.2f}",
            f"{m.get('sortino', 0):.2f}",
            f"{m.get('max_dd', 0):.1%}",
            f"{m.get('avg_turnover', 0):.4f}",
            f"{m.get('avg_exposure', 0):.2f}",
        ])
    best_idx = None
    if rows:
        sharpes = [results[name].get("sharpe", -999) for name in results]
        best_idx = int(np.argmax(sharpes))
    story.append(_make_table(headers, rows, highlight_row=best_idx))
    story.append(Paragraph(
        "Exhibit 1: Ablation table — baseline vs overlay variants. "
        "Green row = highest Sharpe.", sty["caption"],
    ))

    # ── Page 2: Equity curves ──
    story.append(PageBreak())
    story.append(Paragraph("1. Equity Curves & Drawdown", sty["h1"]))
    story.append(_fig_to_image(_chart_equity(equity_curves), ratio=0.55))
    story.append(Paragraph(
        "Exhibit 2: Log-scale equity curves and drawdown for each variant.",
        sty["caption"],
    ))

    story.append(_fig_to_image(_chart_exposure(backtest_dfs), ratio=0.45))
    story.append(Paragraph(
        "Exhibit 3: Gross exposure and rolling 21-day turnover.",
        sty["caption"],
    ))

    # ── Page 3: Model diagnostics ──
    story.append(PageBreak())
    story.append(Paragraph("2. Model Diagnostics", sty["h1"]))

    story.append(_fig_to_image(_chart_fold_metrics(model_output), ratio=0.35))
    story.append(Paragraph(
        "Exhibit 4: Per-fold classification metrics (AUC, log loss, Brier score).",
        sty["caption"],
    ))

    fold_df = model_output.fold_metrics_df()
    fold_headers = ["Fold", "Train Period", "Test Period", "N Train", "AUC",
                    "Brier", "Base Rate"]
    fold_rows = []
    for _, row in fold_df.iterrows():
        fold_rows.append([
            str(int(row["fold"])),
            f"{pd.Timestamp(row['train_start']):%Y-%m-%d} — "
            f"{pd.Timestamp(row['train_end']):%Y-%m-%d}",
            f"{pd.Timestamp(row['test_start']):%Y-%m-%d} — "
            f"{pd.Timestamp(row['test_end']):%Y-%m-%d}",
            f"{int(row['n_train']):,}",
            f"{row['auc']:.3f}" if not np.isnan(row["auc"]) else "—",
            f"{row['brier']:.3f}" if not np.isnan(row["brier"]) else "—",
            f"{row['base_rate']:.1%}",
        ])
    cw = [0.5 * inch, 2.0 * inch, 1.8 * inch, 0.8 * inch,
          0.6 * inch, 0.6 * inch, 0.8 * inch]
    story.append(_make_table(fold_headers, fold_rows, col_widths=cw))
    story.append(Paragraph("Exhibit 5: Fold-level details.", sty["caption"]))

    # ── Page 4: Coefficient stability ──
    story.append(PageBreak())
    story.append(Paragraph("3. Feature Importance & Stability", sty["h1"]))

    story.append(_fig_to_image(_chart_coefficients(model_output), ratio=0.42))
    story.append(Paragraph(
        "Exhibit 6: Coefficient means with cross-fold standard deviation bars. "
        "Green = positive mean, red = negative.", sty["caption"],
    ))

    coef_df = model_output.coefficient_summary()
    coef_headers = ["Feature", "Coef Mean", "Coef Std", "|Coef| Mean",
                    "Sign Stability"]
    coef_rows = []
    for _, row in coef_df.iterrows():
        coef_rows.append([
            row["feature"],
            f"{row['coef_mean']:.4f}",
            f"{row['coef_std']:.4f}",
            f"{row['coef_abs_mean']:.4f}",
            f"{row['sign_stability']:+.2f}",
        ])
    story.append(_make_table(coef_headers, coef_rows))
    story.append(Paragraph(
        "Exhibit 7: Coefficient summary across folds. Sign stability = ±1 means "
        "the feature has the same sign in every fold.", sty["caption"],
    ))

    # ── Page 5: Threshold sweep ──
    story.append(PageBreak())
    story.append(Paragraph("4. Threshold Sweep", sty["h1"]))

    story.append(Paragraph(
        "The entry probability threshold p_enter controls the trade-off between "
        "selectivity and participation. Higher thresholds reduce turnover and "
        "drawdown at the cost of lower exposure and potentially lower CAGR.",
        sty["body"],
    ))

    story.append(_fig_to_image(_chart_threshold_sweep(sweep_results), ratio=0.35))
    story.append(Paragraph(
        "Exhibit 8: Sharpe, CAGR, and Max DD as functions of p_enter threshold.",
        sty["caption"],
    ))

    sweep_headers = ["p_enter", "CAGR", "Vol", "Sharpe", "Max DD", "Turnover"]
    sweep_rows = []
    for sr in sweep_results:
        sweep_rows.append([
            f"{sr['p_enter']:.2f}",
            f"{sr.get('cagr', 0):.1%}",
            f"{sr.get('vol', 0):.1%}",
            f"{sr.get('sharpe', 0):.2f}",
            f"{sr.get('max_dd', 0):.1%}",
            f"{sr.get('avg_turnover', 0):.4f}",
        ])
    story.append(_make_table(sweep_headers, sweep_rows))
    story.append(Paragraph(
        "Exhibit 9: Threshold sweep results.", sty["caption"],
    ))

    # ── Build PDF ──
    doc.build(story)
    print(f"[report] PDF saved to {out_path}")
