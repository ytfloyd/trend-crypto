#!/usr/bin/env python
"""Generate formatted PDF for the Paper Pipeline Factor Portfolio research note."""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether, HRFlowable, Image,
)
from reportlab.lib import colors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "artifacts" / "research"
PLOTS_B = OUT_DIR / "phase_b_spo"
PLOTS_C = OUT_DIR / "phase_c_correlation" / "plots"

NAVY = HexColor("#1a2744")
DARK_BLUE = HexColor("#2c3e6b")
ACCENT = HexColor("#3b7dd8")
LIGHT_GREY = HexColor("#f4f6f8")
MED_GREY = HexColor("#e0e4ea")
TEXT = HexColor("#222222")
MUTED = HexColor("#666666")


def build_styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        "DocTitle", parent=ss["Title"],
        fontName="Helvetica-Bold", fontSize=20, leading=26,
        textColor=NAVY, spaceAfter=4, alignment=TA_LEFT,
    ))
    ss.add(ParagraphStyle(
        "DocSubtitle", parent=ss["Normal"],
        fontName="Helvetica-Oblique", fontSize=10, leading=14,
        textColor=MUTED, spaceAfter=14,
    ))
    ss.add(ParagraphStyle(
        "H1", parent=ss["Heading1"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=NAVY, spaceBefore=20, spaceAfter=8,
        borderWidth=0, borderPadding=0,
    ))
    ss.add(ParagraphStyle(
        "H2", parent=ss["Heading2"],
        fontName="Helvetica-Bold", fontSize=11, leading=15,
        textColor=DARK_BLUE, spaceBefore=14, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "Body", parent=ss["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=13.5,
        textColor=TEXT, spaceAfter=6, alignment=TA_JUSTIFY,
    ))
    ss.add(ParagraphStyle(
        "BodyBold", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=9.5, leading=13.5,
        textColor=TEXT, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "Caption", parent=ss["Normal"],
        fontName="Helvetica-BoldOblique", fontSize=8.5, leading=12,
        textColor=DARK_BLUE, spaceBefore=4, spaceAfter=8,
    ))
    ss.add(ParagraphStyle(
        "TableCell", fontName="Helvetica", fontSize=8.5, leading=11,
        textColor=TEXT,
    ))
    ss.add(ParagraphStyle(
        "TableHeader", fontName="Helvetica-Bold", fontSize=8.5, leading=11,
        textColor=colors.white,
    ))
    ss.add(ParagraphStyle(
        "BulletItem", parent=ss["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=13.5,
        textColor=TEXT, leftIndent=18, bulletIndent=6, spaceAfter=3,
    ))
    ss.add(ParagraphStyle(
        "Footer", fontName="Helvetica", fontSize=7.5, leading=10,
        textColor=MUTED, alignment=TA_CENTER,
    ))
    return ss


def make_table(headers, rows, col_widths=None):
    """Build a styled Table with navy header row."""
    ss = build_styles()
    data = [[Paragraph(h, ss["TableHeader"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), ss["TableCell"]) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, MED_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


def add_image(elements, path, width=5.5 * inch, caption=None, ss=None):
    if Path(path).exists():
        img = Image(str(path), width=width, height=width * 0.42)
        elements.append(img)
        if caption and ss:
            elements.append(Paragraph(caption, ss["Caption"]))
    else:
        if ss:
            elements.append(Paragraph(f"<i>[Plot not found: {path}]</i>", ss["Body"]))


def build_pdf(output_path: str):
    ss = build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    W = doc.width
    elements = []

    # ── TITLE PAGE ──
    elements.append(Spacer(1, 1.2 * inch))
    elements.append(HRFlowable(width="100%", thickness=2, color=NAVY))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "Academic Paper Pipeline:<br/>Cross-Sectional Factor Discovery<br/>and Portfolio Construction",
        ss["DocTitle"],
    ))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "From Literature Review to Deployable Two-Factor Portfolio in Digital Assets",
        ss["DocSubtitle"],
    ))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MED_GREY))
    elements.append(Spacer(1, 14))
    elements.append(Paragraph(
        "Research Note &mdash; Pipeline: Discovery &rarr; Decay Analysis &rarr; ML Overlay &rarr; "
        "Correlation-Aware Construction", ss["Body"],
    ))
    elements.append(Paragraph(
        "Data: Coinbase Advanced spot, 232 assets, 2017&ndash;2025 &nbsp;|&nbsp; "
        "Run date: 2026-02-21", ss["Body"],
    ))
    elements.append(Spacer(1, 24))

    # Executive summary box
    exec_text = (
        "The pipeline discovered two deployable cross-sectional factors in crypto: "
        "<b>VOL_LT</b> (low-volatility, Sharpe 0.59) and <b>VOL_RL</b> (volume-relative, Sharpe 0.55). "
        "Both factors are strengthening and negatively correlated (&rho; = &minus;0.196), providing "
        "regime-complementary coverage &mdash; VOL_LT dominates in bear markets (Sharpe 2.55) while "
        "VOL_RL dominates in bulls (Sharpe 1.56). ML overlays added no value over the raw factor signals. "
        "Volatility-parity portfolio construction achieves a combined net Sharpe of <b>2.77</b>, "
        "CAGR of <b>61.7%</b>, and max drawdown of <b>&minus;6.1%</b> on the test period."
    )
    exec_table = Table(
        [[Paragraph(exec_text, ss["Body"])]],
        colWidths=[W - 0.3 * inch],
    )
    exec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.8, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    elements.append(exec_table)
    elements.append(PageBreak())

    # ── 1. INTRODUCTION ──
    elements.append(Paragraph("1. Introduction", ss["H1"]))
    elements.append(Paragraph(
        "This research note documents an end-to-end pipeline that begins with automated academic paper "
        "discovery and ends with a deployable two-factor portfolio for cryptocurrency spot markets. "
        "The pipeline processes 111 papers from arXiv and SSRN, applies a five-stage quality filter "
        "plus a methodology audit, and translates the surviving strategies into the crypto universe "
        "using the same Coinbase data infrastructure as our prior JPM momentum research.", ss["Body"],
    ))
    elements.append(Paragraph(
        "The work is organized in four phases: <b>Paper Pipeline</b> (discovery, filtering, methodology "
        "audit), <b>Phase A</b> (alpha decay analysis across six cross-sectional factors), <b>Phase B</b> "
        "(Smart Predict-then-Optimize ML overlay comparison), and <b>Phase C</b> (correlation-aware "
        "portfolio construction).", ss["Body"],
    ))

    # ── 2. PAPER PIPELINE ──
    elements.append(Paragraph("2. Paper Pipeline: Discovery and Filtration", ss["H1"]))
    elements.append(Paragraph(
        "We query arXiv (q-fin.TR, q-fin.PM, q-fin.ST) and SSRN for papers published within the last "
        "five years. Each paper passes through five sequential filters; failure at any stage is terminal.",
        ss["Body"],
    ))
    elements.append(Paragraph("Table 0: Filter Stack", ss["Caption"]))
    elements.append(make_table(
        ["Filter", "Criterion", "Rejection Rate"],
        [
            ["Alpha", "Claims excess return, not pure risk decomposition", "~15%"],
            ["Econ. Rationale", "Plausible causal mechanism (behavioral, structural)", "~20%"],
            ["Stat. Robustness", "OOS testing, multi-market, TCA, \u22655 yrs data", "~30%"],
            ["Implementability", "Public data, no HFT, no proprietary feeds", "~10%"],
            ["Staleness", "Known anomaly >10 yrs w/o adaptation \u2192 flagged", "~5% flagged"],
        ],
        col_widths=[1.1 * inch, 3.3 * inch, 0.9 * inch],
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "A post-filter <b>Methodology Audit</b> gate rejects papers with implausible Sharpe ratios "
        "(>5.0 auto-reject, >3.0 suspicious), circular OOS testing, or selected evaluation windows. "
        "Result: 111 papers discovered, 5 passed all filters (4.5% survival rate), 4 translatable "
        "to crypto (1 microstructure paper untranslatable without L2 order book data).",
        ss["Body"],
    ))

    # ── 3. PHASE A ──
    elements.append(Paragraph("3. Phase A: Alpha Decay Analysis", ss["H1"]))
    elements.append(Paragraph(
        "Based on <i>\"Not All Factors Crowd Equally\"</i> (Lee, 2025), we compute daily returns for "
        "six reference cross-sectional factors on 232 Coinbase assets over 2017&ndash;2025. Each factor "
        "is a quintile-sorted long-short portfolio (long top 20%, short bottom 20%, equal-weighted) "
        "with a one-day execution lag.", ss["Body"],
    ))

    elements.append(Paragraph("Table 1: Factor Performance and Decay (232 assets, 2017\u20132025)", ss["Caption"]))
    elements.append(make_table(
        ["Factor", "Full Sharpe", "Last 90d", "Decay", "Half-Life", "Priority"],
        [
            ["VOL_LT", "0.59", "3.68", "STRENGTHENING", "\u221e", "HIGH"],
            ["VOL_RL", "0.55", "2.15", "STRENGTHENING", "\u221e", "HIGH"],
            ["REV_1W", "0.24", "\u22121.05", "STRENGTHENING", "\u221e", "LOW"],
            ["MOM_1M", "\u22120.46", "0.53", "DECAYING", "1,938d", "AVOID"],
            ["MOM_12M", "\u22120.52", "\u22121.66", "DECAYING", "\u2014", "AVOID"],
            ["MOM_3M", "\u22121.02", "\u22120.17", "DECAYING", "1,089d", "AVOID"],
        ],
        col_widths=[0.8 * inch, 0.85 * inch, 0.75 * inch, 1.15 * inch, 0.75 * inch, 0.7 * inch],
    ))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Cross-sectional momentum is dead in crypto.</b> All three lookbacks produce negative "
        "Sharpe ratios. MOM_3M at &minus;1.02 exhibits a strong anti-momentum effect. This is "
        "consistent with our Chapter 1 JPM baseline finding: crypto trend cycles are too short "
        "and altcoin distributions too fat-tailed for cross-sectional momentum to work.",
        ss["Body"],
    ))
    elements.append(Paragraph(
        "<b>Low-volatility and volume-relative factors are alive.</b> VOL_LT and VOL_RL are the only "
        "factors with positive full-period Sharpe, strengthening decay slopes, and stable crowding. "
        "The \"boring coins outperform\" anomaly persists because the retail-dominated market "
        "structurally overprices volatile, lottery-like assets.",
        ss["Body"],
    ))

    # ── 4. PHASE B ──
    elements.append(PageBreak())
    elements.append(Paragraph("4. Phase B: ML Overlay \u2014 Smart Predict-then-Optimize", ss["H1"]))

    elements.append(Paragraph("4.1 Sanity Checks", ss["H2"]))
    elements.append(Paragraph(
        "<b>Regime Sensitivity (most important finding):</b> VOL_LT is a bear-market factor "
        "(BULL = &minus;0.75, BEAR = 2.55); VOL_RL is a bull-market factor (BULL = 1.56, BEAR = &minus;0.03). "
        "These are regime complements, not redundant signals.",
        ss["Body"],
    ))
    elements.append(Paragraph("Table 2: Factor Sharpe by Market Regime", ss["Caption"]))
    elements.append(make_table(
        ["Factor", "BULL", "BEAR", "CHOP", "Flag"],
        [
            ["VOL_LT", "\u22120.75", "2.55", "0.35", "REGIME_DEPENDENT"],
            ["VOL_RL", "1.56", "\u22120.03", "\u22120.17", "\u2014"],
        ],
        col_widths=[0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 1.6 * inch],
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Both factors are well-diversified (top 5 assets = 10\u201315% of PnL) and the "
        "recent 90-day Sharpe spike has historical precedent (19 prior episodes for VOL_LT, 18 for VOL_RL).",
        ss["Body"],
    ))

    elements.append(Paragraph("4.2 Model Results", ss["H2"]))
    elements.append(Paragraph(
        "Two LightGBM models per factor (MSE loss and SPO decision-focused loss) trained on 18 features "
        "with strict walk-forward: train to 2022-12, validation to 2024-03, test from 2024-04.",
        ss["Body"],
    ))
    elements.append(Paragraph("Table 3: ML Overlay vs. Raw Factor \u2014 Test Period", ss["Caption"]))
    elements.append(make_table(
        ["Model", "VOL_LT Sharpe", "VOL_RL Sharpe"],
        [
            ["Raw Factor", "2.03", "1.79"],
            ["MSE Model", "1.41", "0.39"],
            ["SPO Model", "1.18", "1.10"],
        ],
        col_widths=[1.5 * inch, 1.5 * inch, 1.5 * inch],
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("Table 4: Overfit and Differentiation Diagnostics", ss["Caption"]))
    elements.append(make_table(
        ["Diagnostic", "VOL_LT", "VOL_RL"],
        [
            ["MSE IS/OOS Sharpe ratio", "2.3\u00d7 (OVERFIT)", "23.7\u00d7 (OVERFIT)"],
            ["SPO pred. corr. with MSE", "0.869 (DIFFERENTIATED)", "0.479 (DIFFERENTIATED)"],
            ["SPO improvement over MSE", "\u221216%", "+184%"],
        ],
        col_widths=[1.8 * inch, 1.6 * inch, 1.6 * inch],
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "<b>Raw factors win decisively.</b> The MSE model\u2019s VOL_RL result is instructive: "
        "IS Sharpe 9.21 collapses to 0.39 OOS \u2014 a 24\u00d7 overfit. SPO produces genuinely different "
        "predictions (correlation 0.479) and nearly triples MSE\u2019s Sharpe, but still cannot beat "
        "the raw factor. The well-specified rank-and-sort procedure is already close to optimal.",
        ss["Body"],
    ))

    # Phase B plots
    add_image(elements, PLOTS_B / "plots" / "equity_curves_VOL_LT.png",
              caption="Figure 1: VOL_LT equity curves \u2014 Raw vs MSE vs SPO (test period)", ss=ss)
    add_image(elements, PLOTS_B / "plots" / "equity_curves_VOL_RL.png",
              caption="Figure 2: VOL_RL equity curves \u2014 Raw vs MSE vs SPO (test period)", ss=ss)

    # ── 5. PHASE C ──
    elements.append(PageBreak())
    elements.append(Paragraph("5. Phase C: Correlation-Aware Portfolio Construction", ss["H1"]))

    elements.append(Paragraph("5.1 Regime-Conditional Correlation Analysis", ss["H2"]))
    elements.append(Paragraph(
        "The true unconditional correlation between VOL_LT and VOL_RL over 2017\u20132025 is "
        "<b>&minus;0.196</b> \u2014 the factors actively hedge each other. The rolling 60-day "
        "correlation dropped below &minus;0.3 on 540 days in the full history.",
        ss["Body"],
    ))
    elements.append(Paragraph("Table 5: Factor Correlation by Market Regime", ss["Caption"]))
    elements.append(make_table(
        ["Regime", "Correlation", "Interpretation"],
        [
            ["BULL", "\u22120.221", "Active hedge during bull markets"],
            ["BEAR", "\u22120.116", "Mild hedge during bear markets"],
            ["CHOP", "\u22120.248", "Strongest hedge in range-bound markets"],
            ["Unconditional", "\u22120.196", "Structural negative correlation"],
        ],
        col_widths=[1.2 * inch, 1.0 * inch, 2.8 * inch],
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Flags: <b>NEGATIVE_HEDGE</b> (corr &lt; &minus;0.2 in BULL and CHOP) and "
        "<b>REGIME_INVARIANT</b> (all regimes within &plusmn;0.15 of unconditional). "
        "The negative correlation is structural, not regime-dependent. "
        "Lag-5 autocorrelation = 0.933: <b>CORRELATION_FORECASTABLE</b>.",
        ss["Body"],
    ))

    add_image(elements, PLOTS_C / "rolling_correlation.png",
              caption="Figure 3: Rolling 60-day correlation, VOL_LT vs VOL_RL (full history)", ss=ss)

    elements.append(Paragraph("5.2 Construction Method Comparison", ss["H2"]))
    elements.append(Paragraph(
        "Four methods evaluated on the test set (2024-04 to 2025-12), all with 15% vol target "
        "and 20 bps costs:", ss["Body"],
    ))
    elements.append(Paragraph("Table 6: Portfolio Construction Results \u2014 Test Period", ss["Caption"]))
    elements.append(make_table(
        ["Metric", "M1-EqWt", "M2-VolPar", "M3-Regime", "M4-Shrink"],
        [
            ["Net Sharpe", "2.587", "2.774", "2.078", "2.487"],
            ["CAGR", "57.8%", "61.7%", "46.2%", "52.9%"],
            ["Max Drawdown", "\u22125.8%", "\u22126.1%", "\u22125.0%", "\u22126.0%"],
            ["Calmar", "9.90", "10.17", "9.22", "8.83"],
            ["Sortino", "4.01", "4.32", "3.45", "3.83"],
        ],
        col_widths=[1.1 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch],
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("Table 7: Regime Breakdown \u2014 Net Sharpe", ss["Caption"]))
    elements.append(make_table(
        ["Regime", "M1-EqWt", "M2-VolPar", "M3-Regime", "M4-Shrink"],
        [
            ["BULL", "3.32", "4.68", "4.16", "5.02"],
            ["BEAR", "4.75", "3.95", "3.82", "3.02"],
            ["CHOP", "\u22120.09", "\u22120.14", "\u22121.61", "\u22120.41"],
        ],
        col_widths=[1.1 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch],
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "<b>Volatility parity wins.</b> M2 achieves the highest net Sharpe (2.774), CAGR (61.7%), "
        "and Calmar (10.17), beating equal-weight by +0.186 Sharpe. The mechanism is simple: when a "
        "factor\u2019s realized volatility spikes (typically during its strong regime), vol parity "
        "automatically underweights it, preserving capital for when the factor mean-reverts. "
        "No regime classification or correlation forecasting required.",
        ss["Body"],
    ))
    elements.append(Paragraph(
        "<b>Regime switching (M3) is the worst method</b> despite being directionally correct. "
        "Fixed 80/20 allocations create concentration at regime transitions, and CHOP Sharpe of "
        "&minus;1.61 is a serious problem. <b>Shrinkage (M4)</b> adds marginal value over its "
        "complexity: the MAE barely beats the raw sample estimator and is worse than the naive "
        "unconditional mean.",
        ss["Body"],
    ))

    add_image(elements, PLOTS_C / "equity_curves_comparison.png",
              caption="Figure 4: All four construction methods \u2014 test period equity curves (net)", ss=ss)
    add_image(elements, PLOTS_C / "rolling_sharpe_comparison.png",
              caption="Figure 5: Rolling 90-day Sharpe \u2014 all construction methods", ss=ss)
    add_image(elements, PLOTS_C / "regime_correlation_bars.png",
              caption="Figure 6: Cross-factor correlation by regime", ss=ss)

    # ── 6. JOURNEY SUMMARY ──
    elements.append(PageBreak())
    elements.append(Paragraph("6. Journey Summary", ss["H1"]))
    elements.append(Paragraph(
        "Table 8: From 111 Papers to a Deployable Strategy", ss["Caption"],
    ))
    elements.append(make_table(
        ["Phase", "Action", "Key Finding"],
        [
            ["Pipeline", "111 papers \u2192 5-stage filter", "4.5% survival; momentum dead, low-vol alive"],
            ["Phase A", "Six factors, decay analysis", "VOL_LT (0.59) + VOL_RL (0.55) strengthening"],
            ["Phase B", "MSE + SPO LightGBM overlays", "Raw factors beat ML; 24\u00d7 overfit on MSE"],
            ["Phase B (sanity)", "Regime sensitivity test", "VOL_LT = bear, VOL_RL = bull; \u03c1 = 0.042"],
            ["Phase C (pre)", "Regime-conditional corr.", "True \u03c1 = \u22120.196; NEGATIVE_HEDGE"],
            ["Phase C", "Four construction methods", "Vol parity: Sharpe 2.77, CAGR 61.7%, DD \u22126.1%"],
        ],
        col_widths=[1.1 * inch, 1.8 * inch, 2.4 * inch],
    ))

    # ── 7. DEPLOYMENT SPEC ──
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=NAVY))
    elements.append(Paragraph("7. Deployment Specification", ss["H1"]))

    spec_items = [
        ("<b>Strategy:</b> Combined VOL_LT + VOL_RL Raw Factor Portfolio",),
        ("<b>VOL_LT signal:</b> Long bottom-quintile 20d-realized-vol, short top-quintile, "
         "equal-weighted. Daily rebalance.",),
        ("<b>VOL_RL signal:</b> Long top-quintile (5d/60d avg volume ratio), short bottom-quintile, "
         "equal-weighted. Daily rebalance.",),
        ("<b>Construction:</b> Volatility parity (inverse-vol weighted), weekly rebalance.",),
        ("<b>Vol target:</b> 15% annualized | Lookback: 20d | Scale: [0.25, 2.0]",),
        ("<b>Position limit:</b> 20% max per asset in either leg.",),
        ("<b>Transaction costs:</b> 20 bps per side.",),
    ]
    for item in spec_items:
        elements.append(Paragraph(f"\u2022 &nbsp;{item[0]}", ss["BulletItem"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Test Period Performance (2024-04 to 2025-12)", ss["Caption"]))
    elements.append(make_table(
        ["Metric", "Value"],
        [
            ["Net Sharpe", "2.774"],
            ["CAGR", "61.7%"],
            ["Max Drawdown", "\u22126.1%"],
            ["Calmar", "10.17"],
            ["Sortino", "4.32"],
        ],
        col_widths=[2.0 * inch, 2.0 * inch],
    ))

    elements.append(Paragraph("7.1 Kill Switch Conditions", ss["H2"]))
    hard = [
        "Portfolio drawdown exceeds 10% from forward-test peak",
        "Either factor rolling 10d Sharpe &lt; &minus;1.0 for 5 consecutive days",
        "Any single asset moves &gt; 4\u03c3 in one day \u2192 reduce position 50%",
    ]
    for h in hard:
        elements.append(Paragraph(f"\u25a0 &nbsp;<b>Hard stop:</b> {h}", ss["BulletItem"]))
    soft = [
        "Factor correlation (10d rolling) exceeds 0.6",
        "Universe drops below 50 assets on either factor",
        "Vol target scale factor hits 0.25 floor for 10 consecutive days",
    ]
    for s in soft:
        elements.append(Paragraph(f"\u25cb &nbsp;<b>Soft review:</b> {s}", ss["BulletItem"]))

    # ── 8. CAVEATS ──
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("8. Caveats and Robustness Considerations", ss["H1"]))
    caveats = [
        "<b>Test-period Sharpe of 2.77 is high.</b> Full-period individual factor Sharpes are 0.59 "
        "and 0.55. The 2.77 reflects a favorable 20-month window. Forward performance should be "
        "benchmarked against full-cycle expectations (combined Sharpe likely 0.8\u20131.5).",
        "<b>CHOP regime is negative across all methods.</b> A prolonged range-bound market will "
        "underperform. The vol-targeting overlay will naturally reduce exposure.",
        "<b>Low-risk anomaly may eventually be arbitraged.</b> In equities, low-vol has partially "
        "eroded since publication. As institutional participation grows, VOL_LT alpha may decay. "
        "Phase A re-runs every 90 days are designed to detect this.",
        "<b>Short-leg implementation risk.</b> Shorting is expensive or unavailable for many altcoins. "
        "A long-only adaptation (underweight instead of short) was not tested.",
        "<b>ML null result is target-specific.</b> Daily portfolio-level factor return prediction is "
        "inherently noisy. Asset-level prediction or signal-strength conditioning may differ.",
    ]
    for i, c in enumerate(caveats, 1):
        elements.append(Paragraph(f"{i}. {c}", ss["Body"]))

    # ── 9. NEXT STEPS ──
    elements.append(Paragraph("9. Open Research Directions", ss["H1"]))
    nexts = [
        "<b>Time-series momentum (TSMOM):</b> Cross-sectional momentum is dead, but TSMOM may "
        "behave differently. A combined TSMOM + cross-sectional factor portfolio is the natural next step.",
        "<b>Microstructure signals:</b> Requires L2 order book data (Coinbase WebSocket). "
        "Highest-signal alpha vertical in crypto; 2\u20133 day infrastructure build.",
        "<b>Funding rate arbitrage:</b> Cross-exchange perp funding rates are a documented crypto-native edge. "
        "Requires adding a perpetuals data source.",
        "<b>Factor + TSMOM blend:</b> The cross-sectional factor portfolio (Sharpe ~2.77 test) and "
        "Chapter 8 TSMOM Sharpe Blend (0.73 full-period) are driven by orthogonal return sources. "
        "A risk-parity blend is likely to improve both Sharpe and drawdown.",
        "<b>Alpha decay re-run cadence:</b> Factor half-lives in crypto are short. Automate 90-day "
        "Phase A re-runs to detect crowding or structural changes.",
    ]
    for i, n in enumerate(nexts, 1):
        elements.append(Paragraph(f"{i}. {n}", ss["Body"]))

    # ── FOOTER ──
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MED_GREY))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Data: Coinbase daily bars via coinbase_daily_121025.duckdb, 232 USD pairs, 2017\u20132025 &nbsp;|&nbsp; "
        "Artifacts: artifacts/research/{alpha_decay, phase_b_spo, phase_c_correlation}/",
        ss["Footer"],
    ))

    doc.build(elements)
    return output_path


if __name__ == "__main__":
    out = str(OUT_DIR / "paper_pipeline_factor_portfolio.pdf")
    build_pdf(out)
    print(f"PDF written to {out}")
