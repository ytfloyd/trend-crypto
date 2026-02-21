#!/usr/bin/env python
"""
Generate Investment-Committee-Ready Research PDF
=================================================

Combines all 8 steps of JPM Big Data & AI Strategies crypto research
into a single professional PDF document, styled after JPMorgan Research
publications. Mirrors the format of the momentum study report.

Includes: cover page, executive summary, 8 research chapters with
embedded charts and tables, JPM comparison section, and appendices.
"""
from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = BASE / "artifacts" / "research" / "jpm_bigdata_ai"
OUT_PATH = ARTIFACT_DIR / "jpm_bigdata_ai_crypto_research.pdf"

# ---------------------------------------------------------------------------
# Colour palette (JPM-inspired — matches momentum report)
# ---------------------------------------------------------------------------
JPM_BLUE = colors.Color(0.06, 0.18, 0.37)
JPM_BLUE_LIGHT = colors.Color(0.20, 0.40, 0.65)
JPM_GOLD = colors.Color(0.76, 0.63, 0.33)
JPM_GRAY = colors.Color(0.55, 0.55, 0.55)
JPM_GRAY_LIGHT = colors.Color(0.93, 0.93, 0.93)
WHITE = colors.white
BLACK = colors.black

CHART_BLUE = "#0F2E5F"
CHART_GOLD = "#C2A154"
CHART_RED = "#B22222"
CHART_GREEN = "#2E7D32"
CHART_GRAY = "#888888"
CHART_LIGHT_BLUE = "#3366A6"

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch


# ---------------------------------------------------------------------------
# Styles (identical to momentum report)
# ---------------------------------------------------------------------------
def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "Title", parent=ss["Title"],
        fontName="Times-Bold", fontSize=28, leading=34,
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=12,
    )
    s["subtitle"] = ParagraphStyle(
        "Subtitle", parent=ss["Normal"],
        fontName="Times-Roman", fontSize=16, leading=20,
        textColor=colors.Color(0.85, 0.85, 0.85), alignment=TA_CENTER, spaceAfter=6,
    )
    s["cover_date"] = ParagraphStyle(
        "CoverDate", parent=ss["Normal"],
        fontName="Helvetica", fontSize=11, leading=14,
        textColor=JPM_GOLD, alignment=TA_CENTER, spaceAfter=4,
    )
    s["h1"] = ParagraphStyle(
        "H1", parent=ss["Heading1"],
        fontName="Helvetica-Bold", fontSize=18, leading=22,
        textColor=JPM_BLUE, spaceBefore=24, spaceAfter=10,
    )
    s["h2"] = ParagraphStyle(
        "H2", parent=ss["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=16,
        textColor=JPM_BLUE_LIGHT, spaceBefore=16, spaceAfter=6,
    )
    s["h3"] = ParagraphStyle(
        "H3", parent=ss["Heading3"],
        fontName="Helvetica-Bold", fontSize=11, leading=14,
        textColor=JPM_BLUE, spaceBefore=10, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "Body", parent=ss["Normal"],
        fontName="Times-Roman", fontSize=10, leading=13.5,
        textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6,
    )
    s["body_bold"] = ParagraphStyle(
        "BodyBold", parent=s["body"], fontName="Times-Bold",
    )
    s["body_italic"] = ParagraphStyle(
        "BodyItalic", parent=s["body"],
        fontName="Times-Italic", textColor=JPM_GRAY,
    )
    s["bullet"] = ParagraphStyle(
        "Bullet", parent=s["body"],
        leftIndent=18, bulletIndent=6,
        spaceBefore=1, spaceAfter=2,
    )
    s["caption"] = ParagraphStyle(
        "Caption", parent=ss["Normal"],
        fontName="Helvetica", fontSize=8.5, leading=11,
        textColor=JPM_GRAY, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=10,
    )
    s["disclaimer"] = ParagraphStyle(
        "Disclaimer", parent=ss["Normal"],
        fontName="Helvetica", fontSize=7, leading=9,
        textColor=JPM_GRAY, alignment=TA_JUSTIFY,
    )
    return s


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------
def _header_footer(canvas, doc, is_cover=False):
    canvas.saveState()
    if not is_cover:
        canvas.setStrokeColor(JPM_BLUE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - MARGIN + 6, PAGE_W - MARGIN, PAGE_H - MARGIN + 6)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(JPM_GRAY)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 10,
                          "Machine Learning Strategies for Digital Assets")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 10, "NRT Research")
        canvas.setStrokeColor(JPM_BLUE)
        canvas.line(MARGIN, MARGIN - 14, PAGE_W - MARGIN, MARGIN - 14)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, MARGIN - 24, "CONFIDENTIAL")
        canvas.drawCentredString(PAGE_W / 2, MARGIN - 24, f"Page {doc.page}")
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 24, "February 2026")
    canvas.restoreState()


def on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(JPM_BLUE)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(JPM_GOLD)
    canvas.rect(0, PAGE_H * 0.42, PAGE_W, 3, fill=1, stroke=0)
    canvas.restoreState()


def on_body(canvas, doc):
    _header_footer(canvas, doc, is_cover=False)


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------
def make_table(headers, rows, col_widths=None, highlight_row=None):
    data = [headers] + rows
    cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 0), (-1, 0), JPM_BLUE),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.8, 0.8, 0.8)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, JPM_GRAY_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if highlight_row is not None:
        r = highlight_row + 1
        cmds.append(("BACKGROUND", (0, r), (-1, r), colors.Color(0.85, 0.92, 1.0)))
        cmds.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(cmds))
    return t


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------
def embed_image(path, width=6.0 * inch, ratio=0.55):
    if Path(path).exists():
        return Image(str(path), width=width, height=width * ratio)
    return Spacer(1, 12)


def fig_to_image(fig, width=6.5 * inch):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width, height=width * 0.55)


# ---------------------------------------------------------------------------
# Custom charts
# ---------------------------------------------------------------------------
def set_chart_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
        "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.fontsize": 8, "legend.framealpha": 0.9,
    })


def chart_journey_summary():
    set_chart_style()
    steps = list(range(1, 9))
    labels = [
        "Feature\nEngineering",
        "Linear\nModels",
        "Tree\nModels",
        "SVM &\nClassification",
        "Unsupervised\nLearning",
        "Deep\nLearning",
        "Algorithm\nShootout",
        "Portfolio\nConstruction",
    ]
    best_ics = [0.046, 0.032, 0.007, 0.032, 0.009, 0.005, 0.032, 0.032]
    best_sharpes = [None, None, None, None, None, None, None, 0.15]
    bar_colors = [CHART_LIGHT_BLUE, CHART_BLUE, CHART_GOLD, CHART_BLUE,
                  CHART_GRAY, CHART_RED, CHART_BLUE, CHART_GREEN]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(steps))
    bars = ax.bar(x, best_ics, 0.55, color=bar_colors, alpha=0.85, zorder=3)
    ax.set_ylabel("Best OOS Spearman IC (5d)", color=CHART_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.axhline(0, color="gray", linewidth=0.5)

    for i, (bar, ic) in enumerate(zip(bars, best_ics)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{ic:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=CHART_BLUE)

    ax.set_title("Research Journey: Best OOS IC by Step", fontsize=12, pad=12)
    fig.tight_layout()
    return fig


def chart_method_comparison():
    set_chart_style()
    categories = ["Linear\n(Ridge)", "Classification\n(XGB_Clf)", "Tree\n(XGBoost)",
                   "Deep Learning\n(MLP)", "Ensemble\n(top 3)"]
    ics = [0.007, 0.032, -0.0001, 0.0003, 0.028]
    bar_colors = [CHART_LIGHT_BLUE, CHART_BLUE, CHART_GOLD, CHART_RED, CHART_GREEN]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(categories, ics, color=bar_colors, alpha=0.85)
    ax.set_ylabel("OOS Spearman IC (5d)")
    ax.axhline(0, color="gray", linewidth=0.5)
    for bar, ic in zip(bars, ics):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{ic:+.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("ML Method Comparison — Unified Shootout (54 Features, 45 Folds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def chart_portfolio_comparison():
    set_chart_style()
    strategies = ["Top-Q EW\n(naive)", "DynCash\n+PosLim", "Kitchen\nSink", "BTC\nBuy & Hold"]
    sharpes = [-0.10, 0.15, 0.06, 0.85]
    max_dds = [98.6, 50.0, 31.4, 76.7]
    bar_c = [CHART_RED, CHART_GREEN, CHART_BLUE, CHART_GOLD]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    bars = ax.bar(strategies, sharpes, color=bar_c, alpha=0.85)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Risk-Adjusted Return")
    ax.axhline(0, color="gray", linewidth=0.5)
    for bar, v in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if v >= 0 else -0.06),
                f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax = axes[1]
    bars = ax.bar(strategies, max_dds, color=bar_c, alpha=0.85)
    ax.set_ylabel("|Max Drawdown| (%)")
    ax.set_title("Tail Risk")
    for bar, v in zip(bars, max_dds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Portfolio Strategies: ML Signal to Tradeable Portfolio",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------
def build_pdf():
    styles = build_styles()
    story = []

    # ==================================================================
    # COVER PAGE
    # ==================================================================
    story.append(Spacer(1, 2.2 * inch))
    story.append(Paragraph(
        "Machine Learning Strategies<br/>for Digital Assets", styles["title"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "A Systematic Recreation of Kolanovic &amp; Krishnamachari (2017)<br/>"
        "for Cryptocurrency Markets",
        styles["subtitle"]
    ))
    story.append(Spacer(1, 0.6 * inch))
    story.append(Paragraph("NRT Research", styles["cover_date"]))
    story.append(Paragraph("February 2026", styles["cover_date"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "CONFIDENTIAL \u2014 For Investment Committee Review Only",
        styles["cover_date"]))
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())

    # ==================================================================
    # TABLE OF CONTENTS
    # ==================================================================
    story.append(Paragraph("Table of Contents", styles["h1"]))
    story.append(Spacer(1, 0.15 * inch))
    toc = [
        ("Executive Summary", "3"),
        ("1. Introduction and Methodology", "5"),
        ("2. Feature Engineering", "7"),
        ("3. Linear Models", "9"),
        ("4. Tree-Based Models", "11"),
        ("5. SVM and Classification", "13"),
        ("6. Unsupervised Learning", "15"),
        ("7. Deep Learning", "17"),
        ("8. Algorithm Shootout", "19"),
        ("9. Portfolio Construction", "21"),
        ("10. Comparison with Kolanovic &amp; Krishnamachari (2017)", "23"),
        ("11. Conclusions and Recommendations", "25"),
        ("Appendix: Feature Definitions and Methodology", "27"),
    ]
    for title, page in toc:
        dots = "." * max(5, 60 - len(title.replace("&amp;", "&")))
        story.append(Paragraph(
            f'<font face="Times-Roman" size="11">{title}</font>'
            f'<font face="Times-Roman" size="11" color="#888888"> {dots} {page}</font>',
            styles["body"]
        ))
    story.append(PageBreak())

    # ==================================================================
    # EXECUTIVE SUMMARY
    # ==================================================================
    story.append(Paragraph("Executive Summary", styles["h1"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        'This report presents a systematic recreation of the machine learning framework from '
        'Kolanovic &amp; Krishnamachari\'s 2017 JPMorgan research paper, '
        '<i>"Big Data and AI Strategies: Machine Learning and Alternative Data Approach to Investing,"</i> '
        'adapted for digital asset markets. The study spans eight chapters of progressive investigation, '
        'testing linear models, tree-based ensembles, SVMs, unsupervised learning, and deep neural '
        'networks on a universe of 287 cryptocurrencies with 54 technical features.',
        styles["body"]
    ))

    story.append(Paragraph("Key Findings", styles["h2"]))
    highlights = [
        "<b>Classification framing outperforms regression.</b> "
        "Predicting the probability of a positive 5-day return (binary classification) and using "
        "that probability as a continuous signal yields IC +0.032 \u2014 the best of any ML method "
        "tested. Regression models on the same features and splits achieve only IC +0.007.",

        "<b>Simple models dominate complex ones.</b> XGBoost Classifier and Logistic Regression "
        "(IC +0.023) outperform all tree regressors (IC ~0), MLP (IC ~0), and LSTM (IC -0.053). "
        "Deep learning actively destroys value in crypto\u2019s high-noise environment.",

        "<b>The signal is statistically significant but economically weak.</b> "
        "XGB Classifier achieves a t-statistic of 2.63 across 45 walk-forward folds with "
        "IC > 0 in 71% of folds. However, IC = +0.032 is insufficient for a standalone strategy.",

        "<b>Risk overlays rescue the ML signal from ruin.</b> "
        "A naive top-quintile portfolio loses 38% annually. Dynamic cash allocation + position "
        "limits transforms this into +0.6% CAGR with Sharpe +0.15 and MaxDD -50%. "
        "The Kitchen Sink (all overlays) achieves the lowest drawdown at -31%.",

        "<b>Unsupervised learning provides insight, not alpha.</b> "
        "PCA reveals crypto\u2019s one-factor structure (PC1 = 72% of variance). "
        "K-Means identifies 4 regimes. Neither improves prediction quality.",

        "<b>The momentum study dominates.</b> "
        "The companion momentum study (Sharpe 0.73, MaxDD -30%) substantially outperforms "
        "the best ML portfolio (Sharpe 0.15, MaxDD -50%), confirming that in crypto, "
        "trend-following beats cross-sectional ML prediction.",
    ]
    for h in highlights:
        story.append(Paragraph(f"\u2022  {h}", styles["bullet"]))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Final Results", styles["h2"]))

    final_table = make_table(
        ["Metric", "DynCash + PosLim", "Kitchen Sink", "BTC Buy & Hold", "Momentum Study"],
        [
            ["Sharpe Ratio", "0.15", "0.06", "0.85", "0.73"],
            ["CAGR (net)", "+0.6%", "-0.3%", "+39.5%", "+14.2%"],
            ["Annualized Vol", "24.9%", "16.1%", "63.9%", "21.5%"],
            ["Max Drawdown", "-50.0%", "-31.4%", "-76.7%", "-30.2%"],
            ["Avg Exposure", "~21%", "~14%", "100%", "~20%"],
            ["Avg Holdings", "~16", "~16", "1", "~15"],
            ["ML Signal IC", "+0.032", "+0.032", "N/A", "N/A"],
            ["Cost Assumption", "20 bps RT", "20 bps RT", "N/A", "20 bps RT"],
        ],
        col_widths=[1.5 * inch, 1.4 * inch, 1.3 * inch, 1.3 * inch, 1.5 * inch],
        highlight_row=0,
    )
    story.append(final_table)
    story.append(Paragraph(
        "Table ES.1: Final portfolio characteristics. All metrics over Jan 2018 \u2013 Dec 2025.",
        styles["caption"]
    ))

    fig = chart_journey_summary()
    story.append(fig_to_image(fig, width=6.5 * inch))
    story.append(Paragraph(
        "Figure ES.1: Best OOS information coefficient by research step.",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 1: INTRODUCTION
    # ==================================================================
    story.append(Paragraph("1. Introduction and Methodology", styles["h1"]))

    story.append(Paragraph("1.1 Background", styles["h2"]))
    story.append(Paragraph(
        "Kolanovic &amp; Krishnamachari (2017) present a comprehensive survey of machine learning "
        "methods applied to investment management, covering linear models, decision trees, support "
        "vector machines, unsupervised learning, and deep neural networks. Their 280-page report "
        "argues that ML can extract predictive signals from large cross-sectional feature sets that "
        "traditional linear models miss.",
        styles["body"]
    ))
    story.append(Paragraph(
        "This study applies the paper\u2019s full ML pipeline to digital assets \u2014 an asset "
        "class characterised by extreme volatility, high cross-asset correlation, and limited "
        "fundamental data. We test whether the paper\u2019s key finding \u2014 that non-linear "
        "models and alternative data sources can improve prediction \u2014 holds in crypto\u2019s "
        "unique market structure.",
        styles["body"]
    ))

    story.append(Paragraph("1.2 Data and Universe", styles["h2"]))
    data_table = make_table(
        ["Parameter", "Value"],
        [
            ["Data source", "Coinbase (via market.duckdb)"],
            ["Granularity", "Daily OHLCV bars (resampled from 1-min)"],
            ["Symbols", "362 USD pairs (287 in filtered universe)"],
            ["Backtest period", "2018-01-01 to 2025-12-31"],
            ["Universe filter", "ADV > $1M USD, min 90 days history"],
            ["Features", "54 TA-Lib technical indicators"],
            ["Target", "5-day forward return (regression and classification)"],
        ],
        col_widths=[2.0 * inch, 4.5 * inch],
    )
    story.append(data_table)
    story.append(Paragraph("Table 1.1: Data and methodology summary.", styles["caption"]))

    story.append(Paragraph("1.3 Walk-Forward Validation", styles["h2"]))
    story.append(Paragraph(
        "All models are evaluated using strict walk-forward validation: 2-year rolling training "
        "window, 63-day (quarterly) test window, stepped forward by 63 days. This produces 45 "
        "non-overlapping out-of-sample folds. Features are standardised within each fold to prevent "
        "lookahead. The primary evaluation metric is the Spearman rank correlation (Information "
        "Coefficient, IC) between predicted and realised 5-day returns.",
        styles["body"]
    ))

    story.append(Paragraph("1.4 ML Pipeline", styles["h2"]))
    story.append(Paragraph(
        "The research follows the paper\u2019s structure: (1) feature engineering with TA-Lib, "
        "(2) linear regression baselines, (3) tree-based ensembles, (4) SVM / classification "
        "reframing, (5) unsupervised dimensionality reduction and regime clustering, (6) deep "
        "learning (MLP and LSTM), (7) a unified algorithm shootout with statistical significance "
        "testing, and (8) portfolio construction with risk overlays from the companion momentum study.",
        styles["body"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 2: FEATURE ENGINEERING
    # ==================================================================
    story.append(Paragraph("2. Feature Engineering", styles["h1"]))
    story.append(Paragraph(
        "Following the paper\u2019s emphasis on comprehensive feature construction (pp. 30-45), we "
        "build 54 technical features from daily OHLCV data using TA-Lib. All features are lagged "
        "by one day to prevent lookahead bias.",
        styles["body"]
    ))

    story.append(Paragraph("2.1 Feature Groups", styles["h2"]))
    feat_groups = [
        "<b>Returns &amp; Volatility (15 features):</b> Trailing returns, realised volatility, "
        "and volume ratios at 5, 10, 21, 42, and 63-day lookbacks.",
        "<b>Trend Indicators (8):</b> ADX (14d, 28d), MACD (line, signal, histogram), "
        "Aroon oscillator, linear regression slope (14d, 42d).",
        "<b>Momentum Oscillators (11):</b> RSI (14d, 28d), Stochastic K/D, CCI (14d, 28d), "
        "Williams %R, MFI, ROC (10d, 21d), Ultimate Oscillator.",
        "<b>Volatility Indicators (5):</b> ATR, NATR, Bollinger bandwidth, Bollinger %B, "
        "high-low range ratio.",
        "<b>Volume Indicators (2):</b> OBV slope, Accumulation/Distribution slope.",
        "<b>Price Structure (2):</b> Channel position (14d, 42d).",
        "<b>Candlestick Features (11):</b> Overnight gap, body ratio, upper/lower shadows, "
        "and 7 TA-Lib candlestick patterns (Doji, Hammer, Engulfing, etc.).",
    ]
    for f in feat_groups:
        story.append(Paragraph(f"\u2022  {f}", styles["bullet"]))

    story.append(Paragraph("2.2 Univariate Information Coefficients", styles["h2"]))
    story.append(Paragraph(
        "Before building models, we measure the univariate Spearman IC of each feature against "
        "forward returns at 1d, 5d, and 21d horizons. The strongest univariate predictors are "
        "volatility-related features (ATR, NATR, Bollinger bandwidth) with IC ~0.04-0.05 at the "
        "21d horizon \u2014 negative correlation, meaning high recent volatility predicts lower "
        "forward returns. Momentum features (RSI, ROC) show weaker but positive IC at short horizons.",
        styles["body"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_01" / "ic_bar_chart_5d.png"))
    story.append(Paragraph(
        "Figure 2.1: Univariate Spearman IC for all 54 features (5d horizon).",
        styles["caption"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_01" / "feature_correlation_heatmap.png"))
    story.append(Paragraph(
        "Figure 2.2: Feature correlation matrix. Strong clusters within feature groups.",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 3: LINEAR MODELS
    # ==================================================================
    story.append(Paragraph("3. Linear Models", styles["h1"]))
    story.append(Paragraph(
        "The paper (pp. 52-65) begins with linear regression as the ML baseline: OLS, Ridge, "
        "LASSO, and Elastic Net. We train each model on 108 features (54 raw + 54 cross-sectional "
        "ranks) using walk-forward validation on the 5-day forward return target.",
        styles["body"]
    ))

    lin_table = make_table(
        ["Model", "IC", "IC (fold mean)", "IC>0 %", "Hit Rate"],
        [
            ["Ridge", "+0.032", "+0.050", "64%", "50.9%"],
            ["OLS", "+0.031", "+0.046", "62%", "50.9%"],
            ["Elastic Net", "+0.027", "+0.043", "62%", "50.7%"],
            ["LASSO", "+0.013", "+0.032", "56%", "50.2%"],
        ],
        col_widths=[1.5 * inch, 1.0 * inch, 1.3 * inch, 1.0 * inch, 1.0 * inch],
        highlight_row=0,
    )
    story.append(lin_table)
    story.append(Paragraph(
        "Table 3.1: Linear model evaluation (5d horizon, 108 features with ranks).",
        styles["caption"]
    ))

    story.append(Paragraph(
        "Ridge regression leads with IC +0.032 and the highest fold-level IC consistency. "
        "LASSO\u2019s aggressive feature selection retains only ~20 of 108 features, sacrificing "
        "predictive power. The cross-sectional ranks contribute meaningful signal: Ridge on 54 "
        "raw features achieves only IC +0.007. Key LASSO-selected features: vol_42d, natr_14, "
        "rsi_14, channel_pos_14, bb_width.",
        styles["body"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_02" / "model_ic_comparison.png"))
    story.append(Paragraph(
        "Figure 3.1: Linear model IC comparison across horizons.",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 4: TREE-BASED MODELS
    # ==================================================================
    story.append(Paragraph("4. Tree-Based Models", styles["h1"]))
    story.append(Paragraph(
        "The paper (pp. 66-82) argues that tree-based ensembles can capture non-linear interactions "
        "that linear models miss. We test Random Forest, XGBoost, and LightGBM regressors on the "
        "54 raw features (cross-sectional ranks removed, as trees handle non-linearity natively).",
        styles["body"]
    ))

    tree_table = make_table(
        ["Model", "IC (5d)", "IC (1d)", "IC (21d)", "IC>0 %"],
        [
            ["XGBoost", "+0.007", "+0.013", "-0.012", "56%"],
            ["LightGBM", "+0.007", "+0.013", "-0.014", "53%"],
            ["RandomForest", "+0.000", "+0.005", "-0.046", "51%"],
        ],
        col_widths=[1.5 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
    )
    story.append(tree_table)
    story.append(Paragraph(
        "Table 4.1: Tree model evaluation across horizons (54 raw features).",
        styles["caption"]
    ))

    story.append(Paragraph(
        "<b>Trees fail to improve over linear models.</b> XGBoost and LightGBM achieve IC ~+0.007 "
        "on the 5d horizon \u2014 less than one quarter of Ridge\u2019s IC (+0.032). Random Forest is "
        "near zero. At the 21d horizon, all trees produce <i>negative</i> IC, indicating overfitting. "
        "Feature importance analysis shows trees concentrate on short-term volatility features "
        "(natr_14, atr_14, vol_5d), failing to exploit the broader feature set that linear models capture.",
        styles["body_bold"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_03" / "tree_vs_linear_ic.png"))
    story.append(Paragraph(
        "Figure 4.1: Tree vs linear model IC comparison across horizons.",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 5: SVM & CLASSIFICATION
    # ==================================================================
    story.append(Paragraph("5. SVM and Classification", styles["h1"]))
    story.append(Paragraph(
        "The paper (pp. 83-95) discusses Support Vector Machines for both regression and "
        "classification. We reframe the problem as binary classification: predict whether "
        "the 5-day forward return is positive. The predicted probability P(ret &gt; 0) is "
        "then used as a continuous signal for cross-sectional ranking.",
        styles["body"]
    ))

    clf_table = make_table(
        ["Model", "IC (5d)", "AUC", "Accuracy", "F1"],
        [
            ["XGB Classifier", "+0.030", "0.514", "52.4%", "0.41"],
            ["Logistic Regression", "+0.023", "0.511", "52.0%", "0.39"],
            ["RBF SVM", "+0.024", "0.510", "52.3%", "0.40"],
            ["Linear SVM", "+0.023", "0.512", "52.4%", "0.35"],
        ],
        col_widths=[1.5 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
        highlight_row=0,
    )
    story.append(clf_table)
    story.append(Paragraph(
        "Table 5.1: Classification model evaluation (5d horizon).",
        styles["caption"]
    ))

    story.append(Paragraph(
        "<b>The classification framing is a breakthrough.</b> By predicting sign rather than "
        "magnitude, classification models focus on what matters for trading: direction. "
        "XGB Classifier achieves IC +0.030, competitive with Ridge on raw features (+0.032), "
        "and Logistic Regression at IC +0.023 outperforms all tree regressors. The binary "
        "framing effectively reduces target noise, allowing models to extract cleaner signal.",
        styles["body_bold"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_04" / "all_methods_ic_comparison.png"))
    story.append(Paragraph(
        "Figure 5.1: IC comparison across all methods tested through Step 4.",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 6: UNSUPERVISED LEARNING
    # ==================================================================
    story.append(Paragraph("6. Unsupervised Learning", styles["h1"]))
    story.append(Paragraph(
        "The paper (pp. 96-101) explores unsupervised methods for dimensionality reduction and "
        "regime identification. We apply PCA on both the return matrix and the feature matrix, "
        "and K-Means clustering on daily market-level features to identify regimes.",
        styles["body"]
    ))

    story.append(Paragraph("6.1 PCA on Returns", styles["h2"]))
    story.append(Paragraph(
        "The first principal component explains 72% of cross-sectional return variance \u2014 "
        "confirming crypto\u2019s extreme one-factor structure. In equities, PC1 typically explains "
        "30-40%. This means the vast majority of altcoin moves are driven by BTC/ETH beta, "
        "leaving very little idiosyncratic signal for ML models to exploit.",
        styles["body"]
    ))

    story.append(Paragraph("6.2 K-Means Regime Clustering", styles["h2"]))
    story.append(Paragraph(
        "Four regimes emerge from K-Means on daily feature medians: (1) low-vol bull, "
        "(2) high-vol bull, (3) low-vol bear, (4) high-vol bear. Regime-conditioned Ridge "
        "models yield a marginal improvement (IC +0.009 vs +0.007 for unconditioned), "
        "but the difference is not statistically significant.",
        styles["body"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_05" / "pca_scree_plot.png"))
    story.append(Paragraph(
        "Figure 6.1: Return PCA scree plot. PC1 dominates with 72% variance explained.",
        styles["caption"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_05" / "regime_timeseries.png"))
    story.append(Paragraph(
        "Figure 6.2: K-Means regime labels over time (4 regimes).",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 7: DEEP LEARNING
    # ==================================================================
    story.append(Paragraph("7. Deep Learning", styles["h1"]))
    story.append(Paragraph(
        "The paper (pp. 102-116) argues that neural networks can capture complex non-linear "
        "temporal patterns. We test a 2-layer feedforward MLP (64\u219232, LayerNorm, dropout) "
        "and an LSTM (64 hidden, 21-day lookback) with Adam optimizer and early stopping.",
        styles["body"]
    ))

    dl_table = make_table(
        ["Model", "IC", "IC (fold mean)", "IC>0 %", "RMSE"],
        [
            ["Ridge (baseline)", "+0.007", "+0.017", "51%", "0.148"],
            ["MLP", "+0.005", "+0.005", "53%", "0.149"],
            ["LSTM", "-0.053", "-0.032", "44%", "0.168"],
        ],
        col_widths=[1.5 * inch, 1.0 * inch, 1.3 * inch, 1.0 * inch, 1.0 * inch],
    )
    story.append(dl_table)
    story.append(Paragraph(
        "Table 7.1: Deep learning evaluation (5d horizon, 54 raw features).",
        styles["caption"]
    ))

    story.append(Paragraph(
        "<b>Deep learning adds nothing \u2014 LSTM is actively destructive.</b> "
        "The MLP marginally trails Ridge (IC +0.005 vs +0.007), while the LSTM "
        "produces IC -0.053 despite early stopping and dropout. The LSTM\u2019s "
        "per-symbol sequential approach finds spurious temporal patterns in "
        "crypto\u2019s high-noise environment. With 287 symbols and extreme "
        "idiosyncrasy, there is insufficient stable temporal structure for "
        "recurrent nets to exploit.",
        styles["body_bold"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_06" / "model_comparison.png"))
    story.append(Paragraph(
        "Figure 7.1: Deep learning model IC comparison.",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 8: ALGORITHM SHOOTOUT
    # ==================================================================
    story.append(Paragraph("8. Algorithm Shootout", styles["h1"]))
    story.append(Paragraph(
        "To ensure a fair comparison, we re-run all 9 models on the exact same data pipeline: "
        "54 raw features, identical walk-forward splits, identical StandardScaler preprocessing. "
        "Statistical significance is assessed with paired t-tests on fold-level ICs.",
        styles["body"]
    ))

    shoot_table = make_table(
        ["Rank", "Model", "IC", "IC (mean)", "t-stat", "IC>0 %"],
        [
            ["1", "XGB Classifier", "+0.032", "+0.041", "2.63", "71%"],
            ["2", "Logistic Regression", "+0.023", "+0.050", "2.97", "69%"],
            ["3", "Ridge", "+0.007", "+0.017", "0.98", "51%"],
            ["4", "Elastic Net", "+0.006", "+0.012", "0.67", "51%"],
            ["5", "LightGBM", "+0.003", "+0.009", "0.60", "53%"],
            ["6", "MLP", "+0.000", "-0.003", "-0.20", "47%"],
            ["7", "XGBoost (reg)", "-0.000", "+0.005", "0.29", "51%"],
            ["8", "Random Forest", "-0.001", "+0.000", "0.01", "53%"],
            ["9", "LASSO", "-0.005", "+0.002", "0.13", "42%"],
        ],
        col_widths=[0.5 * inch, 1.7 * inch, 0.9 * inch, 1.0 * inch, 0.8 * inch, 0.9 * inch],
        highlight_row=0,
    )
    story.append(shoot_table)
    story.append(Paragraph(
        "Table 8.1: Unified algorithm shootout (54 features, 45 folds, 5d horizon).",
        styles["caption"]
    ))

    story.append(Paragraph("8.1 Statistical Significance", styles["h2"]))
    story.append(Paragraph(
        "Only two models achieve t &gt; 2: XGB Classifier (t=2.63) and Logistic Regression "
        "(t=2.97). The ensemble of the top 3 (XGB_Clf + LogReg + Ridge) achieves IC +0.028 "
        "with the highest t-statistic (t=3.17), but trails the best single model on raw IC. "
        "XGB_Clf significantly beats all tree and neural models (p &lt; 0.05) but does "
        "<i>not</i> significantly beat LogisticReg (p=0.62) or Ridge (p=0.20).",
        styles["body"]
    ))

    fig = chart_method_comparison()
    story.append(fig_to_image(fig, width=6.0 * inch))
    story.append(Paragraph(
        "Figure 8.1: ML method comparison from the unified shootout.",
        styles["caption"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_07" / "significance_heatmap.png"))
    story.append(Paragraph(
        "Figure 8.2: Pairwise statistical significance heatmap (paired t-test p-values).",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 9: PORTFOLIO CONSTRUCTION
    # ==================================================================
    story.append(Paragraph("9. Portfolio Construction", styles["h1"]))
    story.append(Paragraph(
        "Taking the winning XGB Classifier signal (IC +0.032), we construct tradeable portfolios "
        "using the risk overlay toolkit developed in the companion momentum study (Chapters 6-8 "
        "of Kolanovic &amp; Wei recreation). We test 8 portfolio variants with 20 bps transaction "
        "costs and 5-day rebalancing.",
        styles["body"]
    ))

    port_table = make_table(
        ["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "Exposure"],
        [
            ["A: Top-Q Equal Weight", "-38.0%", "87.5%", "-0.10", "-98.6%", "100%"],
            ["B: Signal-Proportional IVW", "-41.6%", "84.6%", "-0.20", "-99.0%", "100%"],
            ["C: SigProp + Position Limit", "-38.7%", "86.7%", "-0.12", "-99.0%", "100%"],
            ["D: DynCash + PosLim", "+0.6%", "24.9%", "+0.15", "-50.0%", "21%"],
            ["E: VolTarget + PosLim", "-4.2%", "23.8%", "-0.06", "-59.0%", "27%"],
            ["G: Kitchen Sink (all)", "-0.3%", "16.1%", "+0.06", "-31.4%", "14%"],
            ["H: Long/Short", "-26.6%", "29.8%", "-0.89", "-92.3%", "95%"],
            ["BTC Buy & Hold", "+39.5%", "63.9%", "+0.85", "-76.7%", "100%"],
        ],
        col_widths=[2.0 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch],
        highlight_row=3,
    )
    story.append(port_table)
    story.append(Paragraph(
        "Table 9.1: Full portfolio strategy comparison. DynCash + PosLim is the best ML strategy.",
        styles["caption"]
    ))

    story.append(Paragraph(
        "<b>Risk overlays are essential.</b> Without them, the ML signal produces -38% CAGR and "
        "-99% MaxDD. Dynamic cash allocation (scaling exposure to 0.5 \u00d7 fraction of positive "
        "signals) reduces average exposure from 100% to 21%, transforming the result to +0.6% "
        "CAGR with -50% MaxDD. The Kitchen Sink variant (position limits + dynamic cash + vol "
        "targeting + drawdown control) achieves the best risk profile: -31% MaxDD and only "
        "-0.3% CAGR loss.",
        styles["body_bold"]
    ))

    fig = chart_portfolio_comparison()
    story.append(fig_to_image(fig, width=6.5 * inch))
    story.append(Paragraph(
        "Figure 9.1: Portfolio strategy comparison \u2014 Sharpe ratio and maximum drawdown.",
        styles["caption"]
    ))

    story.append(embed_image(ARTIFACT_DIR / "step_08" / "best_strategy_deep_dive.png",
                             width=6.0 * inch, ratio=0.75))
    story.append(Paragraph(
        "Figure 9.2: Best ML strategy deep dive \u2014 equity curve, drawdown, and rolling Sharpe.",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 10: COMPARISON WITH JPM PAPER
    # ==================================================================
    story.append(Paragraph(
        "10. Comparison with Kolanovic &amp; Krishnamachari (2017)", styles["h1"]))

    story.append(Paragraph("10.1 Findings That Transfer", styles["h2"]))
    transfers = [
        "<b>Feature engineering matters.</b> Both studies find that a comprehensive "
        "feature set (50+ technical indicators) outperforms any single indicator. "
        "TA-Lib indicators provide a solid feature foundation across asset classes.",

        "<b>Walk-forward validation is essential.</b> Both studies use expanding-window "
        "or rolling walk-forward validation. In-sample performance drastically overstates "
        "out-of-sample results for all model types, especially trees and neural nets.",

        "<b>Regularisation outperforms unconstrained models.</b> Ridge beats OLS in both "
        "studies. L2 regularisation prevents overfitting to noise in the feature matrix.",

        "<b>Classification framing can outperform regression.</b> The paper notes that "
        "binary classification often performs well for trading signals. Our crypto results "
        "confirm this strongly \u2014 classifiers achieve 4x the IC of tree regressors.",

        "<b>Risk management is non-negotiable.</b> Both studies conclude that the ML signal "
        "alone is insufficient; proper position sizing and risk overlays are essential for "
        "converting weak prediction into viable portfolios.",
    ]
    for t in transfers:
        story.append(Paragraph(f"\u2022  {t}", styles["bullet"]))
        story.append(Spacer(1, 2))

    story.append(Paragraph("10.2 Findings That Diverge", styles["h2"]))
    divergences = [
        "<b>Non-linear models fail in crypto.</b> The paper finds trees and neural nets "
        "can capture interactions that improve prediction in equities. In crypto, the "
        "signal-to-noise ratio is too low: trees overfit, and deep learning finds "
        "spurious patterns. Simple linear models and classifiers dominate.",

        "<b>Unsupervised methods provide insight but not alpha.</b> The paper reports "
        "that PCA features and regime conditioning improve predictions. In crypto, "
        "the one-factor structure (PC1 = 72%) leaves too little residual signal, "
        "and regime conditioning provides negligible improvement.",

        "<b>Crypto\u2019s IC ceiling is much lower.</b> The paper reports ICs of "
        "0.05-0.10 for equity cross-sectional models. Our best crypto IC is +0.032, "
        "reflecting higher noise and fewer idiosyncratic opportunities.",

        "<b>LSTM\u2019s sequential advantage disappears.</b> The paper argues LSTM "
        "captures temporal dependencies. In crypto, where 287 symbols exhibit "
        "extreme co-movement, there is insufficient stable temporal structure "
        "for recurrent models \u2014 LSTM actually destroys value (IC = -0.053).",
    ]
    for d in divergences:
        story.append(Paragraph(f"\u2022  {d}", styles["bullet"]))
        story.append(Spacer(1, 2))

    story.append(Paragraph("10.3 Quantitative Comparison", styles["h2"]))
    comp_table = make_table(
        ["Dimension", "JPM Paper (Equities)", "This Study (Crypto)"],
        [
            ["Best model type", "Gradient-boosted trees", "XGB Classifier"],
            ["Best IC", "~0.05-0.10", "+0.032"],
            ["Tree vs Linear", "Trees improve over linear", "Trees trail linear"],
            ["Deep learning", "Marginal improvement", "Harmful (LSTM IC -0.05)"],
            ["Unsupervised value", "PCA features help", "No prediction improvement"],
            ["Classification framing", "Often beneficial", "Critical (4x improvement)"],
            ["Feature set size", "100-200+ features", "54 features"],
            ["Signal-to-noise", "Moderate", "Very low"],
            ["Portfolio Sharpe", "~0.5-1.0 (est.)", "0.15 (best ML portfolio)"],
            ["Risk overlays needed", "Important", "Essential / transformative"],
        ],
        col_widths=[1.8 * inch, 2.2 * inch, 2.5 * inch],
    )
    story.append(comp_table)
    story.append(Paragraph(
        "Table 10.1: Side-by-side comparison of key findings.",
        styles["caption"]
    ))
    story.append(PageBreak())

    # ==================================================================
    # CH 11: CONCLUSIONS
    # ==================================================================
    story.append(Paragraph("11. Conclusions and Recommendations", styles["h1"]))

    story.append(Paragraph("11.1 Summary of Findings", styles["h2"]))
    story.append(Paragraph(
        "This study demonstrates that the ML framework of Kolanovic &amp; Krishnamachari "
        "(2017) can extract a weak but statistically significant signal from crypto cross-sections. "
        "However, the signal\u2019s economic value is limited compared to the companion momentum "
        "study, and aggressive risk management is required to avoid catastrophic losses.",
        styles["body"]
    ))

    conclusions = [
        "<b>Best model: XGB Classifier</b> (IC +0.032, t=2.63, IC>0 in 71% of folds). "
        "Classification framing is the single most important methodological choice.",
        "<b>Best portfolio: Dynamic Cash + Position Limits</b> (Sharpe 0.15, MaxDD -50%). "
        "Risk overlays transform a losing strategy into a marginally positive one.",
        "<b>Model hierarchy: Classification > Linear > Tree > Deep Learning.</b> "
        "This ranking is the reverse of complexity, consistent with the low signal-to-noise ratio.",
        "<b>Momentum dominates ML for crypto.</b> The companion study (Sharpe 0.73) "
        "substantially outperforms the best ML portfolio (Sharpe 0.15). Trend-following "
        "captures a more robust signal in crypto than cross-sectional feature-based prediction.",
        "<b>ML adds value as a complement, not a substitute.</b> The ML signal\u2019s low "
        "correlation with momentum (~0.15) suggests it could improve a blended portfolio.",
    ]
    for c in conclusions:
        story.append(Paragraph(f"\u2022  {c}", styles["bullet"]))
        story.append(Spacer(1, 2))

    story.append(Paragraph("11.2 Recommended Configuration", styles["h2"]))
    impl_table = make_table(
        ["Parameter", "Setting", "Rationale"],
        [
            ["Model", "XGB Classifier", "Best IC, statistically significant"],
            ["Features", "54 TA-Lib indicators", "Comprehensive, lag-adjusted"],
            ["Target", "P(5d return > 0)", "Classification framing"],
            ["Walk-forward", "2yr train, 63d test, 63d step", "45 OOS folds"],
            ["Position limit", "15% max per asset", "Concentration control"],
            ["Cash allocation", "Sensitivity 0.5", "Reduce exposure in weak regimes"],
            ["Rebalance", "Every 5 days", "Balance signal decay vs costs"],
            ["Cost budget", "20 bps round-trip", "Conservative assumption"],
        ],
        col_widths=[1.3 * inch, 1.8 * inch, 3.0 * inch],
    )
    story.append(impl_table)
    story.append(Paragraph(
        "Table 11.1: Recommended configuration for ML-based crypto portfolio.",
        styles["caption"]
    ))

    story.append(Paragraph("11.3 Next Steps", styles["h2"]))
    next_steps = [
        "Blend ML signal with momentum signal (Sharpe-weighted) for a combined portfolio",
        "Test alternative data features (on-chain metrics, social sentiment, funding rates)",
        "Explore higher-frequency features (4h, 1h bars) to increase signal density",
        "Implement online learning / incremental model updates for production deployment",
        "Add crypto-native ML targets (liquidation-adjusted returns, funding rate prediction)",
    ]
    for i, ns in enumerate(next_steps, 1):
        story.append(Paragraph(f"{i}.  {ns}", styles["bullet"]))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("11.4 Risk Considerations", styles["h2"]))
    risks = [
        "<b>Walk-forward is not fully out-of-sample.</b> Feature selection and model hyperparameters "
        "were chosen with knowledge of the full dataset. True OOS requires held-out test periods.",
        "<b>Survivorship bias.</b> Only tokens listed on Coinbase are included. Delisted tokens "
        "with 100% loss are not captured, potentially overstating model quality.",
        "<b>Regime dependence.</b> The model performs well in 2019-2021 and 2023-2024 but fails "
        "in 2018 and 2022. A new structural regime (e.g., regulatory ban) could produce "
        "unprecedented losses.",
        "<b>Capacity constraints.</b> With ~21% average exposure and $1M ADV filter, realistic "
        "capacity is $20-50M before market impact becomes material.",
    ]
    for r in risks:
        story.append(Paragraph(f"\u2022  {r}", styles["bullet"]))
        story.append(Spacer(1, 2))
    story.append(PageBreak())

    # ==================================================================
    # APPENDIX
    # ==================================================================
    story.append(Paragraph("Appendix: Feature Definitions and Methodology", styles["h1"]))

    story.append(Paragraph("A.1 Feature List (54 Features)", styles["h2"]))
    feat_table = make_table(
        ["Group", "Features", "Count"],
        [
            ["Returns", "ret_5d, ret_10d, ret_21d, ret_42d, ret_63d", "5"],
            ["Volatility", "vol_5d, vol_10d, vol_21d, vol_42d, vol_63d", "5"],
            ["Volume Ratio", "vol_ratio_5d, ..., vol_ratio_63d", "5"],
            ["Trend", "adx_14, adx_28, macd, macd_signal, macd_hist, aroon_osc, lreg_slope_14, lreg_slope_42", "8"],
            ["Momentum", "rsi_14, rsi_28, stoch_k, stoch_d, cci_14, cci_28, willr, mfi, roc_10, roc_21, ultosc", "11"],
            ["Volatility Ind.", "atr_14, natr_14, bb_width, bb_pctb, hl_range", "5"],
            ["Volume Ind.", "obv_slope_14, ad_slope_14", "2"],
            ["Price Structure", "channel_pos_14, channel_pos_42", "2"],
            ["Candlestick", "overnight_gap, body_ratio, upper/lower_shadow, cdl_doji, cdl_hammer, cdl_engulfing, +4 more", "11"],
        ],
        col_widths=[1.2 * inch, 3.8 * inch, 0.8 * inch],
    )
    story.append(feat_table)
    story.append(Paragraph("Table A.1: Feature inventory.", styles["caption"]))

    story.append(Paragraph("A.2 Model Hyperparameters", styles["h2"]))
    hp_table = make_table(
        ["Model", "Key Hyperparameters"],
        [
            ["Ridge", "alpha=1.0"],
            ["LASSO", "alpha=0.001, max_iter=2000"],
            ["Elastic Net", "alpha=0.001, l1_ratio=0.5"],
            ["Random Forest", "n_estimators=200, max_depth=6, min_samples_leaf=20"],
            ["XGBoost (reg)", "n_estimators=200, max_depth=4, lr=0.05, subsample=0.8"],
            ["LightGBM", "n_estimators=200, max_depth=4, lr=0.05, subsample=0.8"],
            ["XGB Classifier", "Same as XGBoost reg, eval_metric=logloss"],
            ["Logistic Reg", "C=1.0, solver=lbfgs, max_iter=500"],
            ["MLP", "64\u219232, LayerNorm, dropout=0.3, Adam lr=1e-3, patience=8"],
            ["LSTM", "hidden=64, seq_len=21, dropout=0.2, Adam lr=1e-3"],
        ],
        col_widths=[1.5 * inch, 5.0 * inch],
    )
    story.append(hp_table)
    story.append(Paragraph("Table A.2: Model hyperparameters.", styles["caption"]))

    story.append(Paragraph("A.3 Source Code", styles["h2"]))
    story.append(Paragraph(
        'All research scripts are located in <font face="Courier" size="9">'
        "scripts/research/jpm_bigdata_ai/</font>. Each step corresponds to a self-contained "
        "Python script (step_01 through step_08) that loads data from the shared helpers module, "
        'runs the analysis, and saves artifacts to <font face="Courier" size="9">'
        "artifacts/research/jpm_bigdata_ai/</font>.",
        styles["body"]
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "This document is prepared for internal investment committee review only. "
        "Past performance does not guarantee future results. All backtested results "
        "are hypothetical and subject to the limitations described in Section 11.4. "
        "This is not investment advice.",
        styles["disclaimer"]
    ))

    # ==================================================================
    # BUILD PDF
    # ==================================================================
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUT_PATH), pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Machine Learning Strategies for Digital Assets",
        author="NRT Research",
    )
    cover_frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN, id="cover")
    body_frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN, id="body")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=on_cover),
        PageTemplate(id="body", frames=[body_frame], onPage=on_body),
    ])
    doc.build(story)
    print(f"\n[PDF] Generated: {OUT_PATH}")
    print(f"[PDF] Size: {OUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build_pdf()
