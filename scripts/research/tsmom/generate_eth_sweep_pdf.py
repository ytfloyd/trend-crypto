#!/usr/bin/env python3
"""
Generate AQR-style PDF for ETH-USD Trend Sweep report.

Reads:  artifacts/research/tsmom/eth_trend_sweep/exhibit_*.png
        artifacts/research/tsmom/eth_trend_sweep/results_v2.csv
Writes: artifacts/research/tsmom/eth_trend_sweep_report.pdf

Usage:
    python -m scripts.research.tsmom.generate_eth_sweep_pdf
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, NextPageTemplate, PageBreak,
    Paragraph, Spacer, Image, Table, TableStyle, KeepTogether,
    HRFlowable,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_sweep"
PDF_PATH = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_sweep_report.pdf"

NAVY      = HexColor("#003366")
TEAL      = HexColor("#006B6B")
DARK_GRAY = HexColor("#333333")
MED_GRAY  = HexColor("#666666")
LIGHT_GRAY = HexColor("#CCCCCC")
RULE_GRAY = HexColor("#AAAAAA")
BG_LIGHT  = HexColor("#F5F5F5")

SAMPLE_YEARS = 9
PAGE_W, PAGE_H = letter
L_MARGIN = 0.9 * inch
R_MARGIN = 0.9 * inch
T_MARGIN = 0.75 * inch
B_MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - L_MARGIN - R_MARGIN

FONT_BODY = "Times-Roman"
FONT_BOLD = "Times-Bold"
FONT_ITAL = "Times-Italic"
FONT_BOLDITAL = "Times-BoldItalic"


def _styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle("AQR_Title", fontName=FONT_BOLD, fontSize=24, leading=28,
                           textColor=NAVY, spaceAfter=4, alignment=TA_LEFT))
    ss.add(ParagraphStyle("AQR_Subtitle", fontName=FONT_ITAL, fontSize=12, leading=15,
                           textColor=MED_GRAY, spaceAfter=6))
    ss.add(ParagraphStyle("AQR_Author", fontName=FONT_ROMAN if hasattr(ss, 'x') else FONT_BODY,
                           fontSize=10, leading=13, textColor=MED_GRAY, spaceAfter=3))
    ss.add(ParagraphStyle("AQR_Section", fontName=FONT_BOLD, fontSize=14, leading=17,
                           textColor=NAVY, spaceBefore=18, spaceAfter=8))
    ss.add(ParagraphStyle("AQR_Subsection", fontName=FONT_BOLD, fontSize=11, leading=14,
                           textColor=NAVY, spaceBefore=12, spaceAfter=6))
    ss.add(ParagraphStyle("AQR_Body", fontName=FONT_BODY, fontSize=9.5, leading=13.5,
                           textColor=DARK_GRAY, spaceAfter=7, alignment=TA_JUSTIFY))
    ss.add(ParagraphStyle("AQR_Body_Bold", fontName=FONT_BOLD, fontSize=9.5, leading=13.5,
                           textColor=DARK_GRAY, spaceAfter=7, alignment=TA_JUSTIFY))
    ss.add(ParagraphStyle("AQR_Bullet", fontName=FONT_BODY, fontSize=9.5, leading=13.5,
                           textColor=DARK_GRAY, spaceAfter=4, leftIndent=18,
                           bulletIndent=6, alignment=TA_JUSTIFY))
    ss.add(ParagraphStyle("AQR_Caption", fontName=FONT_ITAL, fontSize=7.5, leading=10,
                           textColor=MED_GRAY, spaceBefore=3, spaceAfter=10,
                           alignment=TA_LEFT))
    ss.add(ParagraphStyle("AQR_Footnote", fontName=FONT_BODY, fontSize=7.5, leading=10,
                           textColor=MED_GRAY, spaceBefore=2, spaceAfter=2))
    ss.add(ParagraphStyle("AQR_ExhibitTitle", fontName=FONT_BOLD, fontSize=9.5, leading=12,
                           textColor=DARK_GRAY, spaceBefore=8, spaceAfter=4))
    ss.add(ParagraphStyle("AQR_TOC_Entry", fontName=FONT_BODY, fontSize=9.5, leading=14,
                           textColor=DARK_GRAY, spaceBefore=2, spaceAfter=2))
    ss.add(ParagraphStyle("AQR_Header", fontName=FONT_ITAL, fontSize=7, leading=9,
                           textColor=MED_GRAY))
    ss.add(ParagraphStyle("AQR_PageNum", fontName=FONT_BODY, fontSize=8, leading=10,
                           textColor=MED_GRAY, alignment=TA_CENTER))
    ss.add(ParagraphStyle("AQR_Abstract", fontName=FONT_ITAL, fontSize=9.5, leading=13.5,
                           textColor=DARK_GRAY, spaceAfter=8, leftIndent=18,
                           rightIndent=18, alignment=TA_JUSTIFY))
    return ss


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_ITAL, 7)
    canvas.setFillColor(MED_GRAY)
    canvas.drawString(L_MARGIN, PAGE_H - 0.5 * inch,
                      "NRT Alternative Thinking 2026 Issue 1: Follow the Trend?")
    canvas.drawRightString(PAGE_W - R_MARGIN, PAGE_H - 0.5 * inch,
                           f"{doc.page}")
    canvas.setStrokeColor(LIGHT_GRAY)
    canvas.setLineWidth(0.5)
    canvas.line(L_MARGIN, PAGE_H - 0.55 * inch, PAGE_W - R_MARGIN, PAGE_H - 0.55 * inch)
    canvas.line(L_MARGIN, B_MARGIN - 0.1 * inch, PAGE_W - R_MARGIN, B_MARGIN - 0.1 * inch)
    canvas.restoreState()


def _title_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LIGHT_GRAY)
    canvas.setLineWidth(0.5)
    canvas.line(L_MARGIN, B_MARGIN - 0.1 * inch, PAGE_W - R_MARGIN, B_MARGIN - 0.1 * inch)
    canvas.setFont(FONT_BODY, 7)
    canvas.setFillColor(MED_GRAY)
    canvas.drawRightString(PAGE_W - R_MARGIN, PAGE_H - 0.5 * inch,
                           f"{doc.page}")
    canvas.restoreState()


def img(name, width=None):
    p = OUT_DIR / name
    if not p.exists():
        return Spacer(1, 12)
    w = width or CONTENT_W
    from reportlab.lib.utils import ImageReader
    ir = ImageReader(str(p))
    iw, ih = ir.getSize()
    aspect = ih / iw
    h = w * aspect
    max_h = 4.2 * inch
    if h > max_h:
        h = max_h
        w = h / aspect
    return Image(str(p), width=w, height=h)


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=LIGHT_GRAY,
                       spaceBefore=6, spaceAfter=6)


def load_data():
    df = pd.read_csv(OUT_DIR / "results_v2.csv")
    strats = df[df["label"] != "BUY_AND_HOLD"].copy()
    bh = df[df["label"] == "BUY_AND_HOLD"].iloc[0]
    return df, strats, bh


def build_pdf():
    ss = _styles()
    df, strats, bh = load_data()

    n = len(strats)
    n_beat_sr = (strats["sharpe"] > bh["sharpe"]).sum()
    n_beat_dd = (strats["max_dd"] > bh["max_dd"]).sum()
    n_beat_cagr = (strats["cagr"] > bh["cagr"]).sum()
    med_sr = strats["sharpe"].median()
    med_cagr = strats["cagr"].median()
    med_dd = strats["max_dd"].median()

    daily = strats[strats["freq"] == "1d"]
    four_h = strats[strats["freq"] == "4h"]
    one_h = strats[strats["freq"] == "1h"]

    n_eff = 493 * 3
    bonf_z = stats.norm.ppf(1 - 0.05 / n_eff / 2)
    sharpe_thresh = bonf_z / np.sqrt(SAMPLE_YEARS)
    n_survivors = (strats["sharpe"] > sharpe_thresh).sum()
    top10 = strats[strats["sharpe"] > sharpe_thresh].nlargest(10, "sharpe")

    # ATR matched comparison
    matched = []
    for (label, freq), group in strats.groupby(["label", "freq"]):
        atr = group[group["stop_type"] == "atr"]
        pct = group[group["stop_type"] == "pct"]
        if atr.empty or pct.empty:
            continue
        matched.append({
            "atr_wins": atr["sharpe"].max() > pct["sharpe"].max(),
        })
    atr_win_pct = pd.DataFrame(matched)["atr_wins"].mean() if matched else 0.5

    # -- Build document --
    doc = BaseDocTemplate(
        str(PDF_PATH), pagesize=letter,
        leftMargin=L_MARGIN, rightMargin=R_MARGIN,
        topMargin=T_MARGIN, bottomMargin=B_MARGIN,
        title="NRT Alternative Thinking 2026 Issue 1",
        author="NRT Portfolio Research Group",
    )

    frame_title = Frame(L_MARGIN, B_MARGIN, CONTENT_W, PAGE_H - T_MARGIN - B_MARGIN,
                        id="title_frame")
    frame_body = Frame(L_MARGIN, B_MARGIN, CONTENT_W, PAGE_H - T_MARGIN - B_MARGIN,
                       id="body_frame")

    doc.addPageTemplates([
        PageTemplate(id="title_page", frames=[frame_title], onPage=_title_page),
        PageTemplate(id="body_page", frames=[frame_body], onPage=_header_footer),
    ])

    E = []

    # ================================================================
    # TITLE PAGE
    # ================================================================
    E.append(Spacer(1, 1.8 * inch))
    E.append(Paragraph("NRT Alternative Thinking", ss["AQR_Subtitle"]))
    E.append(Paragraph("2026 Issue 1", ss["AQR_Subtitle"]))
    E.append(Spacer(1, 0.3 * inch))
    E.append(Paragraph("Follow the Trend?", ss["AQR_Title"]))
    E.append(Paragraph("What 13,000 Crypto Strategies<br/>Actually Tell Us", ss["AQR_Title"]))
    E.append(Spacer(1, 0.35 * inch))
    E.append(Paragraph("Portfolio Research Group", ss["AQR_Author"]))
    E.append(Spacer(1, 0.5 * inch))
    E.append(hr())

    abstract = (
        f'"Follow the Trend" has become the working hypothesis for systematic crypto allocation '
        f"at this desk. In this article, we test that hypothesis exhaustively — building "
        f"<b>{n:,}</b> trend-following configurations on ETH-USD and asking a simple question: "
        f"does trend beat buy-and-hold? The short answer is: mostly no. <b>{n - n_beat_sr:,} of "
        f"{n:,} ({(n - n_beat_sr)/n:.0%})</b> trend strategies produce worse risk-adjusted returns "
        f"than passive buy-and-hold. But {n_beat_dd:,} of {n:,} ({n_beat_dd/n:.0%}) have shallower "
        f"max drawdowns. The question is not whether trend works — it is whether the protection "
        f"it buys is worth the return it costs."
    )
    E.append(Paragraph("<b>Executive Summary</b>", ss["AQR_Body_Bold"]))
    E.append(Paragraph(abstract, ss["AQR_Abstract"]))
    E.append(Spacer(1, 0.15 * inch))
    E.append(Paragraph(
        "<super>1</super> A disclaimer is necessary: we are testing 13,293 strategies on a "
        "single asset over a nine-year period. No matter how significant a result appears, it "
        "reflects in-sample data mining until validated out-of-sample and across assets. We "
        "would take very little risk on any single configuration, preferring to diversify "
        "across many — and even then, no directional strategy is anywhere near perfect.",
        ss["AQR_Footnote"],
    ))

    E.append(NextPageTemplate("body_page"))
    E.append(PageBreak())

    # ================================================================
    # CONTENTS
    # ================================================================
    E.append(Paragraph("Contents", ss["AQR_Section"]))
    toc_items = [
        ("Introduction", 3),
        ("Part 1: The Baseline — Buy-and-Hold Is Extremely Hard to Beat", 4),
        ("Part 2: What Trend Buys and What It Costs", 5),
        ("Part 3: What Actually Drives Performance? Frequency Dominates Everything", 7),
        ("Part 4: Do Vol-Adaptive Stops Beat Fixed Stops?", 10),
        ("Part 5: The Multiple Testing Problem", 12),
        ("Part 6: The Convexity Profile", 14),
        ("Concluding Thoughts", 15),
        ("Appendix: Parameter Grid and Data Notes", 17),
        ("References and Further Reading", 18),
    ]
    for title, pg in toc_items:
        E.append(Paragraph(f"{title}", ss["AQR_TOC_Entry"]))
    E.append(Spacer(1, 0.3 * inch))
    E.append(Paragraph(
        "The authors thank the NRT Quantitative Research team for helpful comments.",
        ss["AQR_Footnote"],
    ))
    E.append(PageBreak())

    # ================================================================
    # INTRODUCTION
    # ================================================================
    E.append(Paragraph("Introduction", ss["AQR_Section"]))
    E.append(Paragraph(
        f"Crypto allocators face a unique problem. The asset class has delivered extraordinary "
        f"long-term returns — ETH-USD compounded at {bh['cagr']:.0%} annualized from 2017 to "
        f"2026 — but the path was brutal: a {bh['max_dd']:.0%} peak-to-trough drawdown, with "
        f"multiple drawdowns exceeding 70%. No investor, institutional or otherwise, can "
        f"plausibly hold through a {bh['max_dd']:.0%} drawdown.<super>2</super> The standard "
        f"response is to apply trend-following logic: be long when the asset is trending up, "
        f"move to cash when it is not.", ss["AQR_Body"]))
    E.append(Paragraph(
        "The premise has theoretical support. Moskowitz, Ooi, and Pedersen (2012) documented "
        "time-series momentum across dozens of futures markets. Hurst, Ooi, and Pedersen (2017) "
        "extended the evidence to a century of data. In our own prior work at this desk, we "
        "attempted a portfolio-level TSMOM framework for crypto, applying vol-scaled signals "
        "with portfolio-level vol targeting across multiple assets. The results were poor: "
        "the framework produced a Sharpe of 0.77 with 87% time-in-market — essentially "
        "dampened buy-and-hold at a fraction of the CAGR.<super>3</super>", ss["AQR_Body"]))
    E.append(Paragraph(
        "This led to a natural question: if portfolio-level momentum fails, do simpler "
        "per-asset trend signals do better? And if so, what matters more — the entry signal, "
        "the data frequency, or the exit mechanism?", ss["AQR_Body"]))
    E.append(Paragraph(
        f"To answer these questions, we built <b>{n:,} configurations</b> from the cross-product "
        f"of 493 base trend signals from 30+ families, 3 frequencies (daily, 4-hour, 1-hour), "
        f"and 9 stop-loss variants (no stop; fixed trailing stops at 5%, 10%, 20%; vol-adaptive "
        f"trailing stops at 1.5×, 2.0×, 2.5×, 3.0×, 4.0× entry-date ATR). All configurations "
        f"use identical backtest rules: binary long or cash, signal applied with one-bar lag, "
        f"20 bps round-trip transaction costs, no leverage, no position sizing.", ss["AQR_Body"]))

    E.append(Spacer(1, 6))
    E.append(Paragraph(
        "<super>2</super> Luna Foundation Guard, Three Arrows Capital, and Alameda Research "
        "all failed to maintain positions through drawdowns of comparable magnitude.",
        ss["AQR_Footnote"]))
    E.append(Paragraph(
        "<super>3</super> See internal memo, \"TSMOM Framework Results,\" January 2026.",
        ss["AQR_Footnote"]))
    E.append(PageBreak())

    # ================================================================
    # PART 1
    # ================================================================
    E.append(Paragraph("Part 1: The Baseline — Buy-and-Hold Is Extremely Hard to Beat",
                        ss["AQR_Section"]))
    E.append(Paragraph(
        f"Before evaluating trend strategies, it is worth establishing how strong the baseline "
        f"is. ETH-USD buy-and-hold from January 2017 to February 2026: Sharpe {bh['sharpe']:.2f}, "
        f"CAGR {bh['cagr']:.0%}, max drawdown {bh['max_dd']:.0%}, Calmar {bh['calmar']:.2f}, "
        f"Sortino {bh['sortino']:.2f}, skewness {bh['skewness']:.2f}.", ss["AQR_Body"]))
    E.append(Paragraph(
        f"A Sharpe ratio of {bh['sharpe']:.2f} is exceptional by any standard. Any strategy "
        f"that sits in cash for part of the period faces a substantial headwind from missing "
        f"this strong secular drift.<super>4</super>", ss["AQR_Body"]))
    E.append(Paragraph(
        f"Exhibit 1 shows the distribution of Sharpe ratios across all {n:,} trend "
        f"configurations. The median strategy ({med_sr:.2f}) falls well below buy-and-hold "
        f"({bh['sharpe']:.2f}). Only {n_beat_sr:,} ({n_beat_sr/n:.0%}) of configurations "
        f"outperform on this metric.", ss["AQR_Body"]))

    E.append(Spacer(1, 4))
    E.append(Paragraph(
        "Exhibit 1: Most Trend Strategies Underperform Buy-and-Hold on a Risk-Adjusted Basis",
        ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_1_sharpe_dist.png"))
    E.append(Paragraph(
        "Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February "
        "2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps "
        "round-trip costs, no leverage. Past performance is not a reliable indicator of future "
        "results.", ss["AQR_Caption"]))

    E.append(Spacer(1, 6))
    E.append(Paragraph(
        "<super>4</super> Throughout this piece, \"cash\" means zero return. We do not model "
        "stablecoin yield.", ss["AQR_Footnote"]))
    E.append(PageBreak())

    # ================================================================
    # PART 2
    # ================================================================
    E.append(Paragraph("Part 2: What Trend Buys and What It Costs", ss["AQR_Section"]))
    E.append(Paragraph(
        f"If trend-following in crypto mostly underperforms buy-and-hold on a risk-adjusted "
        f"basis, why consider it? Because risk-adjusted returns are not the only thing that "
        f"matters. An investor who cannot hold through a {bh['max_dd']:.0%} drawdown does not "
        f"earn the {bh['cagr']:.0%} CAGR. The relevant question is: what does trend cost, "
        f"and what does it buy?", ss["AQR_Body"]))
    E.append(Paragraph(
        "Exhibit 2 shows the tradeoff directly: the left panel shows the CAGR distribution "
        "(what you give up), and the right panel shows the drawdown distribution (what you get).",
        ss["AQR_Body"]))

    E.append(Paragraph(
        "Exhibit 2: The Trend Tradeoff — CAGR for Drawdown Protection",
        ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_2_tradeoff.png"))
    E.append(Paragraph(
        "Source: NRT Research. Same data and methodology as Exhibit 1.",
        ss["AQR_Caption"]))

    # Tradeoff table
    tdata = [
        ["Metric", "Buy & Hold", "Median Strategy", "Difference"],
        ["CAGR", f"{bh['cagr']:.1%}", f"{med_cagr:.1%}",
         f"{med_cagr - bh['cagr']:+.1%}"],
        ["Max Drawdown", f"{bh['max_dd']:.1%}", f"{med_dd:.1%}",
         f"{med_dd - bh['max_dd']:+.1%}"],
        ["Skewness", f"{bh['skewness']:.2f}",
         f"{strats['skewness'].median():.2f}",
         f"{strats['skewness'].median() - bh['skewness']:+.2f}"],
    ]
    t = Table(tdata, colWidths=[1.4 * inch, 1.3 * inch, 1.5 * inch, 1.3 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    E.append(Spacer(1, 6))
    E.append(t)

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        f"The median strategy gives up roughly {abs(med_cagr - bh['cagr']):.0%} of annual "
        f"return to compress the max drawdown by {abs(med_dd - bh['max_dd']):.0%}. Is that a "
        f"good trade? It depends on the investor's utility function. For an allocator who cannot "
        f"hold through {bh['max_dd']:.0%} but can hold through {med_dd:.0%}, trend converts an "
        f"undeployable return stream into a deployable one — even if the headline CAGR is lower."
        f"<super>5</super>", ss["AQR_Body"]))
    n_pos_skew = (strats["skewness"] > 0).sum()
    E.append(Paragraph(
        f"The skewness improvement is real but partly mechanical. A binary long/cash strategy on "
        f"a strongly trending asset will exhibit positive skewness by construction: it participates "
        f"in the large up-moves while exiting before or during some of the large down-moves. "
        f"This does not require signal skill — {n_pos_skew:,} of {n:,} ({n_pos_skew/n:.0%}) "
        f"strategies exhibit positive skewness regardless of signal choice. The skewness is a "
        f"property of the trade structure, not the signal.<super>6</super>", ss["AQR_Body"]))

    E.append(Spacer(1, 6))
    E.append(Paragraph(
        "<super>5</super> This framing is consistent with the crypto-specific finding that the "
        "binding constraint is not expected return but the ability to stay allocated through "
        "drawdowns.", ss["AQR_Footnote"]))
    E.append(Paragraph(
        "<super>6</super> A randomly-timed long/cash strategy on ETH-USD would also exhibit "
        "positive skewness over this period, simply because the right tail of ETH daily returns "
        "is fatter than the left tail.", ss["AQR_Footnote"]))
    E.append(PageBreak())

    # ================================================================
    # PART 3
    # ================================================================
    E.append(Paragraph(
        "Part 3: What Actually Drives Performance? Frequency Dominates Everything",
        ss["AQR_Section"]))
    E.append(Paragraph(
        f"Across {n:,} configurations, we vary three dimensions: signal choice (493 base "
        f"signals), frequency (daily, 4-hour, 1-hour), and stop type (9 variants). Which "
        f"dimension matters most? The answer is unambiguous: <b>frequency</b>.", ss["AQR_Body"]))
    E.append(Paragraph(
        f"Exhibit 3 shows the Sharpe ratio distribution at each frequency. Daily signals have "
        f"a median Sharpe of {daily['sharpe'].median():.2f} and "
        f"{(daily['sharpe'] > bh['sharpe']).sum()/len(daily):.0%} beat buy-and-hold. Four-hour "
        f"signals drop to {four_h['sharpe'].median():.2f} "
        f"({(four_h['sharpe'] > bh['sharpe']).sum()/len(four_h):.0%} beat B&H). One-hour "
        f"signals collapse to {one_h['sharpe'].median():.2f} "
        f"({(one_h['sharpe'] > bh['sharpe']).sum()/len(one_h):.0%} beat B&H).",
        ss["AQR_Body"]))

    E.append(Paragraph(
        "Exhibit 3: Frequency Is the Dominant Variable — Daily Signals Are Overwhelmingly Better",
        ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_3_frequency.png"))
    E.append(Paragraph(
        "Source: NRT Research. Same data and methodology as Exhibit 1. Red line = buy-and-hold "
        "Sharpe.", ss["AQR_Caption"]))

    # Frequency table
    freq_data = [["Frequency", "N", "Median Sharpe", "Median CAGR", "Median MaxDD", "% Beat B&H"]]
    for freq, flabel, sub in [("1d", "Daily", daily), ("4h", "4-Hour", four_h), ("1h", "1-Hour", one_h)]:
        freq_data.append([
            flabel, f"{len(sub):,}", f"{sub['sharpe'].median():.3f}",
            f"{sub['cagr'].median():.1%}", f"{sub['max_dd'].median():.1%}",
            f"{(sub['sharpe'] > bh['sharpe']).sum()/len(sub):.0%}",
        ])
    ft = Table(freq_data, colWidths=[0.9*inch, 0.7*inch, 1.1*inch, 1.0*inch, 1.1*inch, 1.0*inch])
    ft.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    E.append(Spacer(1, 6))
    E.append(Paragraph("Exhibit 3a: Performance by Frequency", ss["AQR_ExhibitTitle"]))
    E.append(ft)

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "The mechanism is straightforward. Higher-frequency signals generate more trades, and "
        "each trade incurs transaction costs. In a strong secular uptrend, frequent trading also "
        "increases the probability of being whipsawed out of a trend during intraday noise, then "
        "missing the subsequent continuation.<super>7</super>", ss["AQR_Body"]))
    E.append(Spacer(1, 6))
    E.append(Paragraph(
        "<super>7</super> This finding is consistent with the broader trend-following literature. "
        "Moskowitz et al (2012) found that time-series momentum is strongest at monthly "
        "frequencies and degrades at shorter horizons.", ss["AQR_Footnote"]))
    E.append(PageBreak())

    E.append(Paragraph(
        "Within the daily-frequency universe, Exhibit 4 shows median Sharpe by signal family. "
        "The spread is modest: the variation within families (across parameter choices) is often "
        "as large as the variation across families. This suggests that signal choice, while not "
        "irrelevant, is secondary to frequency and time-in-market.", ss["AQR_Body"]))

    E.append(Paragraph(
        "Exhibit 4: Median Sharpe by Signal Family (daily, no stop)",
        ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_4_family.png"))
    E.append(Paragraph(
        "Source: NRT Research. Shows daily-frequency, no-stop configurations only. Families with "
        "fewer than 3 configurations excluded.", ss["AQR_Caption"]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "Exhibit 5 confirms that time-in-market is the core dial. More time invested means "
        "higher CAGR but deeper drawdowns — there is no free lunch.",
        ss["AQR_Body"]))

    E.append(Paragraph(
        "Exhibit 5: Time in Market Controls the Return/Risk Dial",
        ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_5_tim.png"))
    E.append(Paragraph(
        "Source: NRT Research. Same data and methodology as Exhibit 1. Red star = buy-and-hold.",
        ss["AQR_Caption"]))

    # TIM bucket table
    tim_data = [["TIM Bucket", "N", "Median Sharpe", "Median MaxDD", "Median Skew"]]
    for lo, hi in [(0,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,1.0)]:
        sub = strats[(strats["time_in_market"] >= lo) & (strats["time_in_market"] < hi)]
        if sub.empty:
            continue
        tim_data.append([
            f"{lo:.0%}–{hi:.0%}", f"{len(sub):,}",
            f"{sub['sharpe'].median():.3f}", f"{sub['max_dd'].median():.1%}",
            f"{sub['skewness'].median():.2f}",
        ])
    tt = Table(tim_data, colWidths=[0.9*inch, 0.8*inch, 1.1*inch, 1.1*inch, 1.0*inch])
    tt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    E.append(Spacer(1, 6))
    E.append(Paragraph("Exhibit 5a: Performance by Time-in-Market Bucket",
                        ss["AQR_ExhibitTitle"]))
    E.append(tt)
    E.append(PageBreak())

    # ================================================================
    # PART 4
    # ================================================================
    E.append(Paragraph("Part 4: Do Vol-Adaptive Stops Beat Fixed Stops?", ss["AQR_Section"]))
    E.append(Paragraph(
        "A common hypothesis in systematic trading is that vol-adaptive exits (e.g., ATR-based "
        "trailing stops) should dominate fixed-percentage exits because they adapt to the "
        "prevailing volatility regime. We test this by comparing nine stop variants across all "
        "base signals.", ss["AQR_Body"]))

    E.append(Paragraph(
        "Exhibit 7: Aggregate Performance by Stop Type",
        ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_7_stops_agg.png"))
    E.append(Paragraph(
        "Source: NRT Research. Gray = no stop, blue = fixed %, teal = ATR. Each bar represents "
        "the median across ~1,477 configurations.", ss["AQR_Caption"]))

    # Stop table
    stop_data = [["Stop", "Type", "Med Sharpe", "Med MaxDD", "Med Skew", "Med TIM"]]
    order = ["none", "pct5", "pct10", "pct20", "atr1.5", "atr2.0", "atr2.5", "atr3.0", "atr4.0"]
    display_s = ["None", "5%", "10%", "20%", "1.5× ATR", "2.0× ATR", "2.5× ATR", "3.0× ATR", "4.0× ATR"]
    for sl, dl in zip(order, display_s):
        sub = strats[strats["stop"] == sl]
        if sub.empty:
            continue
        stype = "—" if sl == "none" else ("Fixed %" if sl.startswith("pct") else "ATR")
        stop_data.append([
            dl, stype, f"{sub['sharpe'].median():.3f}",
            f"{sub['max_dd'].median():.1%}", f"{sub['skewness'].median():.2f}",
            f"{sub['time_in_market'].median():.0%}",
        ])
    st = Table(stop_data, colWidths=[0.85*inch, 0.65*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.7*inch])
    st.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    E.append(Spacer(1, 6))
    E.append(Paragraph("Exhibit 7a: Aggregate Performance by Stop Type",
                        ss["AQR_ExhibitTitle"]))
    E.append(st)
    E.append(PageBreak())

    E.append(Paragraph(
        "The more interesting test is the <b>matched comparison</b>: for the same base signal "
        "and frequency, does the best ATR stop outperform the best fixed stop? Exhibit 8 shows "
        "the result.", ss["AQR_Body"]))

    E.append(Paragraph(
        "Exhibit 8: Matched Comparison — ATR vs Fixed Stops on Same Base Signal",
        ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_8_matched.png"))
    E.append(Paragraph(
        "Source: NRT Research. Each point = one base signal × frequency pair. Above the "
        "diagonal, ATR outperforms fixed %.", ss["AQR_Caption"]))

    E.append(Paragraph(
        f"The answer is not what we expected. <b>ATR stops win only {atr_win_pct:.0%} of the "
        f"time on Sharpe</b> across {len(matched):,} matched pairs — essentially a coin flip. "
        f"The aggregate medians in Exhibit 7 were misleading because ATR and fixed stops have "
        f"different time-in-market profiles, which confounds the comparison. Once we match on "
        f"the same base signal, the stop type is a wash.<super>8</super>", ss["AQR_Body"]))
    E.append(Paragraph(
        "This is an honest but uncomfortable finding. The economic intuition for vol-adaptive "
        "stops is compelling, but the data do not support a strong claim of dominance over "
        "this sample. Both stop types achieve roughly the same thing: they compress drawdowns "
        "at the cost of CAGR, with the compression proportional to how tight the stop is."
        "<super>9</super>", ss["AQR_Body"]))
    E.append(Spacer(1, 6))
    E.append(Paragraph(
        "<super>8</super> The matched comparison uses the best ATR and best fixed stop for "
        "each signal. This is generous to both.", ss["AQR_Footnote"]))
    E.append(Paragraph(
        "<super>9</super> One interpretation is that in a single-asset context with binary "
        "positioning, the stop distance matters more than how it is calibrated. The theoretical "
        "advantage of ATR may require a more diverse asset universe to manifest.",
        ss["AQR_Footnote"]))
    E.append(PageBreak())

    # ================================================================
    # PART 5
    # ================================================================
    E.append(Paragraph("Part 5: The Multiple Testing Problem", ss["AQR_Section"]))
    E.append(Paragraph(
        f"We tested {n:,} configurations. Even if every strategy were generated by a coin flip "
        f"with zero true Sharpe, we would expect some to look impressive by chance alone. Any "
        f"honest assessment of these results must address the multiple testing problem."
        f"<super>10</super>", ss["AQR_Body"]))
    E.append(Paragraph(
        f"The strategies are not independent — stop variants of the same base signal are highly "
        f"correlated (average pairwise Sharpe correlation: 0.94). We estimate the effective "
        f"number of independent tests at approximately {n_eff:,} (493 base signals × 3 "
        f"frequencies), treating stop variants as dependent.", ss["AQR_Body"]))
    E.append(Paragraph(
        f"At this test count, a Bonferroni-corrected significance threshold requires a "
        f"z-statistic of {bonf_z:.2f}. Over our {SAMPLE_YEARS}-year sample, this translates to "
        f"a Sharpe ratio of <b>{sharpe_thresh:.2f}</b>. Only <b>{n_survivors:,} of {n:,} "
        f"({n_survivors/n:.1%})</b> strategies survive this threshold.", ss["AQR_Body"]))

    E.append(Paragraph(
        f"Exhibit 9: After Bonferroni Correction, Only {n_survivors:,} of {n:,} Strategies "
        f"({n_survivors/n:.1%}) Survive",
        ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_9_multiple_testing.png"))
    E.append(Paragraph(
        f"Source: NRT Research. Bonferroni correction assumes {n_eff:,} effective independent "
        f"tests. The {SAMPLE_YEARS}-year sample converts z-thresholds to Sharpe thresholds "
        f"via SR = z / √T.", ss["AQR_Caption"]))

    # Top 10 survivors table
    surv_data = [["#", "Signal", "Stop", "Freq", "Sharpe", "CAGR", "MaxDD", "Calmar", "Skew", "TIM"]]
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        surv_data.append([
            str(rank), row["label"], row["stop"], row["freq"],
            f"{row['sharpe']:.2f}", f"{row['cagr']:.1%}", f"{row['max_dd']:.1%}",
            f"{row['calmar']:.2f}", f"{row['skewness']:.2f}", f"{row['time_in_market']:.0%}",
        ])
    surv_t = Table(surv_data, colWidths=[
        0.25*inch, 1.4*inch, 0.5*inch, 0.4*inch, 0.55*inch, 0.55*inch,
        0.55*inch, 0.55*inch, 0.5*inch, 0.4*inch,
    ])
    surv_t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    E.append(Spacer(1, 8))
    E.append(Paragraph("Exhibit 9a: Top 10 Strategies Surviving Bonferroni Correction",
                        ss["AQR_ExhibitTitle"]))
    E.append(surv_t)

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "The survivors cluster in daily-frequency, medium-lookback MA crossovers and a few "
        "channel-based signals. This is reassuring in one sense — the surviving families are "
        "well-established in the trend-following literature — but concerning in another: the "
        "specific parameterizations are almost certainly influenced by the particular path "
        "of ETH over this sample.<super>11</super>", ss["AQR_Body"]))
    E.append(Spacer(1, 6))
    E.append(Paragraph(
        f"<super>10</super> AQR's analysis of 196 \"Buy the Dip\" strategies (Cao, Chong, "
        f"and Villalon, 2025) faces a similar challenge with far fewer tests. With {n:,} "
        f"strategies, the concern is proportionally more severe.", ss["AQR_Footnote"]))
    E.append(Paragraph(
        "<super>11</super> If we were to run the same sweep on BTC-USD or SOL-USD, the "
        "specific winning parameterizations would likely differ, even if the winning signal "
        "families remain similar.", ss["AQR_Footnote"]))
    E.append(PageBreak())

    # ================================================================
    # PART 6
    # ================================================================
    E.append(Paragraph("Part 6: The Convexity Profile", ss["AQR_Section"]))
    E.append(Paragraph(
        "For allocators whose mandate is long convexity — bounded downside with exposure to "
        "unbounded upside — the joint distribution of Sharpe and skewness matters more than "
        "Sharpe alone.", ss["AQR_Body"]))

    E.append(Paragraph(
        "Exhibit 6: Positive Skewness Is Nearly Universal in Binary Long/Cash Crypto Trend",
        ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_6_sharpe_skew.png"))
    E.append(Paragraph(
        "Source: NRT Research. Same data and methodology as Exhibit 1.",
        ss["AQR_Caption"]))

    E.append(Paragraph(
        "Exhibit 10: The Risk/Return Map", ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_10_cagr_dd.png"))
    E.append(Paragraph(
        "Source: NRT Research. Dotted lines show Calmar ratio contours (CAGR / |MaxDD|).",
        ss["AQR_Caption"]))
    E.append(PageBreak())

    E.append(Paragraph(
        "Exhibit 11: Drawdown Distribution by Stop Type", ss["AQR_ExhibitTitle"]))
    E.append(img("exhibit_11_dd_box.png"))
    E.append(Paragraph(
        "Source: NRT Research. Gray = no stop, blue = fixed %, teal = ATR. Red dashed = "
        "buy-and-hold.", ss["AQR_Caption"]))
    E.append(PageBreak())

    # ================================================================
    # CONCLUDING THOUGHTS
    # ================================================================
    E.append(Paragraph("Concluding Thoughts", ss["AQR_Section"]))
    E.append(Paragraph(
        "Four findings emerge from this study:", ss["AQR_Body"]))
    E.append(Paragraph(
        f"<b>First, trend-following in crypto mostly underperforms buy-and-hold on risk-adjusted "
        f"returns.</b> {(n-n_beat_sr)/n:.0%} of the {n:,} strategies we tested produced lower "
        f"Sharpe ratios than passive buy-and-hold. This is not an indictment of trend-following "
        f"— it is a statement about how strong the secular uptrend in crypto has been. In a "
        f"{bh['cagr']:.0%} CAGR environment, any time spent in cash is expensive.",
        ss["AQR_Body"]))
    E.append(Paragraph(
        f"<b>Second, the value of trend is in drawdown compression, not return enhancement.</b> "
        f"{n_beat_dd/n:.0%} of strategies have shallower drawdowns than buy-and-hold's "
        f"{bh['max_dd']:.0%}. For allocators constrained by drawdown tolerance, this converts "
        f"an undeployable return stream into a deployable one.", ss["AQR_Body"]))
    E.append(Paragraph(
        "<b>Third, data frequency dominates signal choice and exit mechanism.</b> Daily signals "
        "vastly outperform intraday signals. Within the daily universe, the choice of signal "
        "family and stop type matters far less than the choice of frequency. Vol-adaptive "
        "(ATR-based) stops do not reliably dominate fixed-percentage stops in a matched "
        "comparison.", ss["AQR_Body"]))
    E.append(Paragraph(
        f"<b>Fourth, after multiple-testing correction, only {n_survivors/n:.1%} of strategies "
        f"survive.</b> The {n_survivors:,} surviving configurations cluster in well-known trend "
        f"signals at daily frequency. We hypothesize that the signal family result may be robust "
        f"across assets, but the specific parameterizations are almost certainly sample-dependent. "
        f"Cross-asset validation is required before deployment.", ss["AQR_Body"]))
    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "A final note on interpretation. The fact that most trend strategies underperform "
        "buy-and-hold in crypto does not mean trend-following is useless. It means the secular "
        "uptrend is so strong that the opportunity cost of being in cash — even temporarily — "
        "is enormous. In a lower-drift environment, the calculus shifts in trend's favor. "
        "The historical crypto drift is an anomaly, not a steady state, and strategies should "
        "be evaluated against a range of possible futures, not just the most favorable past."
        "<super>12</super>", ss["AQR_Body"]))
    E.append(Spacer(1, 6))
    E.append(Paragraph(
        f"<super>12</super> This is analogous to AQR's observation that \"Buy the Dip\" "
        f"strategies appear to work in recent data primarily because equities have gone up a "
        f"lot — not because the timing adds value. Similarly, many crypto trend strategies "
        f"\"work\" primarily because ETH has gone up {bh['total_return']:.0f}× over this "
        f"period.", ss["AQR_Footnote"]))
    E.append(PageBreak())

    # ================================================================
    # APPENDIX
    # ================================================================
    E.append(Paragraph("Appendix: Parameter Grid and Data Notes", ss["AQR_Section"]))
    E.append(Paragraph(
        "<b>Data</b>: Coinbase Advanced spot OHLCV, ETH-USD. January 1, 2017 – February 22, "
        "2026. Daily, 4-hour, and 1-hour bars.", ss["AQR_Body"]))
    E.append(Paragraph(
        "<b>Base signals (493)</b>: SMA crossover, EMA crossover, DEMA crossover, Hull MA "
        "crossover, price vs SMA/EMA, Donchian channel, Bollinger Bands, Keltner Channel, "
        "Supertrend, raw momentum, vol-scaled momentum, linear regression t-stat, MACD, RSI, "
        "ADX, CCI, Aroon, Stochastic, Parabolic SAR, Williams %R, MFI, TRIX, PPO, APO, MOM, "
        "ROC, CMO, Ichimoku, OBV, Heikin-Ashi, Kaufman Efficiency Ratio, VWAP, dual momentum, "
        "triple MA, Turtle breakout, regime-filter SMA, ATR breakout, close-above-high, "
        "mean-reversion band.", ss["AQR_Body"]))
    E.append(Paragraph(
        "<b>Stop variants (9)</b>: None; fixed trailing at 5%, 10%, 20%; ATR-based trailing "
        "at 1.5×, 2.0×, 2.5×, 3.0×, 4.0× (14-period ATR at entry date).", ss["AQR_Body"]))
    E.append(Paragraph(
        "<b>Backtest rules</b>: Binary long/cash. One-bar lag. 20 bps round-trip transaction "
        "costs. No leverage. No position sizing. Intraday signals resampled to daily for P&L.",
        ss["AQR_Body"]))
    E.append(Paragraph(
        "<b>Multiple testing</b>: Effective independent tests estimated at 1,479 (493 signals "
        "× 3 frequencies). Stop variants treated as dependent (avg pairwise Sharpe correlation "
        "= 0.94). Bonferroni correction at 5% FWER.", ss["AQR_Body"]))

    # Parameter grid table
    grid_data = [
        ["Variable", "Definition", "What We Test"],
        ["Signals", "Which trend rule to apply", "493 base signals from 30+ families"],
        ["Frequency", "Bar interval for signal computation", "Daily, 4-hour, 1-hour"],
        ["Stops", "Trailing stop exit overlay", "None; 5%, 10%, 20% fixed; 1.5×–4.0× ATR"],
    ]
    gt = Table(grid_data, colWidths=[0.9*inch, 2.0*inch, 2.8*inch])
    gt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    E.append(Spacer(1, 12))
    E.append(Paragraph("Exhibit A1: Building 13,293 Trend Strategies",
                        ss["AQR_ExhibitTitle"]))
    E.append(gt)
    E.append(PageBreak())

    # ================================================================
    # REFERENCES
    # ================================================================
    E.append(Paragraph("References and Further Reading", ss["AQR_Section"]))
    refs = [
        "Cao, Jeffrey, Nathan Chong, and Dan Villalon. \"Hold the Dip.\" <i>AQR Alternative "
        "Thinking</i> 2025, Issue 4.",
        "Hurst, Brian, Yao Hua Ooi, Lasse Heje Pedersen. \"A Century of Evidence on Trend-"
        "Following Investing.\" <i>The Journal of Portfolio Management</i> 44, no. 1 (2017).",
        "Moskowitz, Tobias J., Yao Hua Ooi, Lasse Heje Pedersen. \"Time series momentum.\" "
        "<i>Journal of Financial Economics</i> 104, Issue 2 (2012): 228-50.",
        "Babu, Abilash, Brendan Hoffman, Ari Levine, et al. \"You Can't Always Trend When You "
        "Want.\" <i>The Journal of Portfolio Management</i> 46, no. 4 (2020).",
        "AQR. \"Trend-Following: Why Now? A Macro Perspective.\" AQR whitepaper, November 16, "
        "2022.",
    ]
    for ref in refs:
        E.append(Paragraph(ref, ss["AQR_Body"]))

    E.append(Spacer(1, 0.5 * inch))
    E.append(hr())
    E.append(Paragraph(
        "HYPOTHETICAL PERFORMANCE RESULTS HAVE MANY INHERENT LIMITATIONS. NO REPRESENTATION "
        "IS BEING MADE THAT ANY ACCOUNT WILL OR IS LIKELY TO ACHIEVE PROFITS OR LOSSES SIMILAR "
        "TO THOSE SHOWN. IN FACT, THERE ARE FREQUENTLY SHARP DIFFERENCES BETWEEN HYPOTHETICAL "
        "PERFORMANCE RESULTS AND THE ACTUAL RESULTS SUBSEQUENTLY ACHIEVED BY ANY PARTICULAR "
        "TRADING PROGRAM.",
        ParagraphStyle("Disclaimer", fontName=FONT_BODY, fontSize=6.5, leading=8.5,
                        textColor=MED_GRAY, alignment=TA_JUSTIFY),
    ))

    # ================================================================
    # BUILD
    # ================================================================
    doc.build(E)
    print(f"[pdf] Written: {PDF_PATH}")
    print(f"[pdf] Size: {PDF_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build_pdf()
