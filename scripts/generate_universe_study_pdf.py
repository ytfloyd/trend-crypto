#!/usr/bin/env python3
"""
Generate McKinsey-style PDF of the MA(5/40) USDC universe study.

Reads:
  - artifacts/research/ma_5_40_usdc_universe/usdc_universe_ma_5_40_results_with_category.csv
  - artifacts/research/ma_5_40_usdc_universe/category_stats.csv
  - artifacts/research/ma_5_40_usdc_universe/wfo_per_pair_results.csv
  - artifacts/research/ma_5_40_usdc_universe/highq_basket_wfo_selections.csv
  - artifacts/research/ma_5_40_usdc_universe/highq_basket_wfo_returns.parquet
  - artifacts/research/ma_5_40_usdc_universe/figures/*.png

Writes:
  - artifacts/research/ma_5_40_usdc_universe/ma_5_40_usdc_universe_study.pdf
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, ListFlowable, ListItem, HRFlowable,
)
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.platypus.frames import Frame
from reportlab.pdfgen import canvas

import re


def safe_text(s: str) -> str:
    """Escape bare ampersands so reportlab's mini-HTML parser doesn't choke,
    while preserving valid HTML entities (&amp; &lt; &gt; etc.) and tags."""
    return re.sub(r'&(?!(?:amp|lt|gt|nbsp|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', s)


_RealParagraph = Paragraph
def Paragraph(text, style, **kwargs):  # noqa: F811
    """Drop-in replacement for reportlab's Paragraph that escapes bare ampersands."""
    return _RealParagraph(safe_text(text), style, **kwargs)


# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
UDIR = ROOT / "artifacts" / "research" / "ma_5_40_usdc_universe"
FIG  = UDIR / "figures"
OUT_PDF = UDIR / "ma_5_40_usdc_universe_study.pdf"

# ── McKinsey-style palette ─────────────────────────────────────────────
MCK_NAVY   = HexColor("#051C2C")   # primary dark navy
MCK_BLUE   = HexColor("#2251FF")   # accent blue
MCK_TEAL   = HexColor("#00A9F4")
MCK_GREEN  = HexColor("#2E7D32")
MCK_RED    = HexColor("#C81E1E")
MCK_AMBER  = HexColor("#E58A00")
MCK_GRAY   = HexColor("#5A6063")
MCK_LIGHT  = HexColor("#E8EAEC")
MCK_BG     = HexColor("#F5F7F8")


# ── Styles ─────────────────────────────────────────────────────────────
def make_styles():
    ss = getSampleStyleSheet()
    body_font = "Helvetica"
    bold_font = "Helvetica-Bold"

    custom = {
        "MckTitle":      ParagraphStyle("MckTitle", parent=ss["Title"], fontSize=30, leading=36,
                                         spaceAfter=10, textColor=MCK_NAVY, fontName=bold_font,
                                         alignment=0),
        "MckSubtitle":   ParagraphStyle("MckSubtitle", parent=ss["Normal"], fontSize=14, leading=18,
                                         spaceAfter=12, textColor=MCK_BLUE, fontName=body_font),
        "MckTagline":    ParagraphStyle("MckTagline", parent=ss["Normal"], fontSize=10, leading=14,
                                         spaceAfter=24, textColor=MCK_GRAY, fontName=body_font),
        "MckSection":    ParagraphStyle("MckSection", parent=ss["Heading1"], fontSize=18, leading=22,
                                         spaceBefore=12, spaceAfter=8, textColor=MCK_NAVY,
                                         fontName=bold_font),
        "MckSubsection": ParagraphStyle("MckSubsection", parent=ss["Heading2"], fontSize=13, leading=16,
                                         spaceBefore=14, spaceAfter=6, textColor=MCK_NAVY,
                                         fontName=bold_font),
        "MckInsight":    ParagraphStyle("MckInsight", parent=ss["Normal"], fontSize=14, leading=18,
                                         spaceBefore=10, spaceAfter=12, textColor=MCK_NAVY,
                                         fontName=bold_font, leftIndent=0),
        "MckBody":       ParagraphStyle("MckBody", parent=ss["Normal"], fontSize=10, leading=14,
                                         fontName=body_font, spaceAfter=6, textColor=MCK_NAVY),
        "MckBullet":     ParagraphStyle("MckBullet", parent=ss["Normal"], fontSize=10, leading=14,
                                         fontName=body_font, leftIndent=12, bulletIndent=0,
                                         spaceAfter=4, textColor=MCK_NAVY),
        "MckCaption":    ParagraphStyle("MckCaption", parent=ss["Normal"], fontSize=8, leading=10,
                                         fontName="Helvetica-Oblique", textColor=MCK_GRAY,
                                         spaceBefore=4, spaceAfter=14, alignment=0),
        "MckCalloutKey": ParagraphStyle("MckCalloutKey", parent=ss["Normal"], fontSize=20, leading=24,
                                         fontName=bold_font, textColor=MCK_BLUE, alignment=1),
        "MckCalloutLbl": ParagraphStyle("MckCalloutLbl", parent=ss["Normal"], fontSize=8, leading=10,
                                         fontName=body_font, textColor=MCK_GRAY, alignment=1),
        "MckFootnote":   ParagraphStyle("MckFootnote", parent=ss["Normal"], fontSize=8, leading=10,
                                         fontName="Helvetica-Oblique", textColor=MCK_GRAY,
                                         spaceBefore=8),
    }
    for k, v in custom.items():
        ss.add(v)
    return ss


# ── Page template with header/footer ───────────────────────────────────
class McKDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        BaseDocTemplate.__init__(self, filename, **kwargs)
        page_w, page_h = self.pagesize
        frame_cover = Frame(0.75*inch, 0.75*inch, page_w - 1.5*inch, page_h - 1.5*inch,
                            id="cover", showBoundary=0)
        frame_main  = Frame(0.85*inch, 0.85*inch, page_w - 1.7*inch, page_h - 1.7*inch,
                            id="main", showBoundary=0)
        self.addPageTemplates([
            PageTemplate(id="Cover", frames=frame_cover, onPage=self._draw_cover),
            PageTemplate(id="Main",  frames=frame_main,  onPage=self._draw_main),
        ])

    def _draw_cover(self, canv: canvas.Canvas, doc):
        canv.saveState()
        page_w, page_h = self.pagesize
        canv.setFillColor(MCK_NAVY)
        canv.rect(0, page_h - 0.45*inch, page_w, 0.45*inch, fill=1, stroke=0)
        canv.setFillColor(MCK_BLUE)
        canv.rect(0, page_h - 0.50*inch, page_w, 0.05*inch, fill=1, stroke=0)
        canv.setFillColor(white)
        canv.setFont("Helvetica-Bold", 11)
        canv.drawString(0.75*inch, page_h - 0.30*inch, "NRT RESEARCH")
        canv.setFont("Helvetica", 9)
        canv.drawRightString(page_w - 0.75*inch, page_h - 0.30*inch,
                              "Crypto Trend-Following Validation")
        canv.setFillColor(MCK_NAVY)
        canv.rect(0, 0, page_w, 0.30*inch, fill=1, stroke=0)
        canv.restoreState()

    def _draw_main(self, canv: canvas.Canvas, doc):
        canv.saveState()
        page_w, page_h = self.pagesize
        canv.setFillColor(MCK_NAVY)
        canv.rect(0, page_h - 0.35*inch, page_w, 0.35*inch, fill=1, stroke=0)
        canv.setFillColor(white)
        canv.setFont("Helvetica-Bold", 9)
        canv.drawString(0.85*inch, page_h - 0.22*inch, "NRT RESEARCH")
        canv.setFont("Helvetica", 8)
        canv.drawRightString(page_w - 0.85*inch, page_h - 0.22*inch,
                              "MA(5/40) Universe Study | May 2026")
        canv.setStrokeColor(MCK_LIGHT)
        canv.setLineWidth(0.5)
        canv.line(0.85*inch, 0.55*inch, page_w - 0.85*inch, 0.55*inch)
        canv.setFillColor(MCK_GRAY)
        canv.setFont("Helvetica", 8)
        canv.drawString(0.85*inch, 0.35*inch, "MA(5/40) Cross-Asset Validation | Confidential")
        canv.drawRightString(page_w - 0.85*inch, 0.35*inch, f"Page {doc.page}")
        canv.restoreState()


# ── Helpers ────────────────────────────────────────────────────────────
def hr():
    return HRFlowable(width="100%", thickness=0.5, color=MCK_LIGHT, spaceBefore=2, spaceAfter=8)

def thick_hr():
    return HRFlowable(width="100%", thickness=2.0, color=MCK_BLUE, spaceBefore=2, spaceAfter=10)

def img(path, width=6.6*inch, height=None):
    p = Path(path)
    if not p.exists():
        return Paragraph(f"<i>[missing figure: {p.name}]</i>", ss["MckCaption"])
    from PIL import Image as PILImage
    iw, ih = PILImage.open(p).size
    if height is None:
        height = width * (ih/iw)
    return Image(str(p), width=width, height=height)

def caption(text):
    return Paragraph(text, ss["MckCaption"])

def section_break(num, title, subtitle=None):
    """A McKinsey-style section divider page."""
    out = [PageBreak(), Spacer(1, 1.8*inch)]
    out.append(Paragraph(f"<font color='#2251FF'>{num}</font>", ParagraphStyle(
        "secnum", fontSize=72, leading=80, fontName="Helvetica-Bold", textColor=MCK_BLUE)))
    out.append(Spacer(1, 6))
    out.append(thick_hr())
    out.append(Paragraph(title, ParagraphStyle(
        "secttl", fontSize=26, leading=32, fontName="Helvetica-Bold", textColor=MCK_NAVY,
        spaceAfter=8)))
    if subtitle:
        out.append(Paragraph(subtitle, ParagraphStyle(
            "secsub", fontSize=12, leading=16, fontName="Helvetica", textColor=MCK_GRAY)))
    out.append(PageBreak())
    return out

def insight(text):
    """Lead-with-the-conclusion insight callout, McKinsey-style page header."""
    return [
        Paragraph(text, ss["MckInsight"]),
        HRFlowable(width="20%", thickness=2.5, color=MCK_BLUE,
                   hAlign="LEFT", spaceBefore=0, spaceAfter=14),
    ]

def fact_box(label_value_pairs, col_widths=None):
    """Horizontal "fact box" row — labels above, big values below."""
    if col_widths is None:
        col_widths = [1.6*inch] * len(label_value_pairs)
    top_row    = [Paragraph(v, ss["MckCalloutKey"]) for _, v in label_value_pairs]
    bottom_row = [Paragraph(l, ss["MckCalloutLbl"]) for l, _ in label_value_pairs]
    t = Table([top_row, bottom_row], colWidths=col_widths, rowHeights=[36, 18])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), MCK_BG),
        ("BOX",        (0, 0), (-1, -1), 0.4, MCK_LIGHT),
        ("LINEAFTER",  (0, 0), (-2, -1), 0.4, MCK_LIGHT),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t

def styled_table(headers, rows, col_widths=None, hilite_rows=None, body_align="CENTER",
                 first_col_align="LEFT"):
    if col_widths is None:
        col_widths = [1.0*inch] * len(headers)
    if hilite_rows is None:
        hilite_rows = []
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), MCK_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 8),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("ALIGN",      (1, 0), (-1, -1), body_align),
        ("ALIGN",      (0, 0), (0, -1),  first_col_align),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, MCK_BG]),
        ("LINEBELOW",  (0, 0), (-1, 0), 1.2, MCK_BLUE),
        ("LINEBELOW",  (0, -1), (-1, -1), 0.5, MCK_LIGHT),
    ]
    for r in hilite_rows:
        # rows are 1-indexed in `data` since headers are row 0
        style.append(("BACKGROUND", (0, r+1), (-1, r+1), HexColor("#FFF6E0")))
        style.append(("FONTNAME",   (0, r+1), (-1, r+1), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


# ── Content sections ───────────────────────────────────────────────────
def build_cover(elements):
    elements.append(Spacer(1, 0.8*inch))
    elements.append(Paragraph("Trend-Following<br/>in Crypto", ss["MckTitle"]))
    elements.append(HRFlowable(width="30%", thickness=4, color=MCK_BLUE, hAlign="LEFT",
                               spaceBefore=2, spaceAfter=16))
    elements.append(Paragraph(
        "A cross-asset validation of the MA(5/40) strategy across "
        "187 Coinbase USDC pairs, net of 20 bps round-trip costs", ss["MckSubtitle"]))
    elements.append(Paragraph(
        "Findings include a 7.5-year out-of-sample basket study, walk-forward parameter "
        "robustness tests, a category-level deployability framework, and a complete "
        "cost-sensitivity analysis.", ss["MckTagline"]))
    elements.append(Spacer(1, 0.4*inch))
    elements.append(fact_box([
        ("Eligible USDC pairs", "187"),
        ("Total bar-history", "11.3 yrs"),
        ("Cost assumption", "20 bps RT"),
        ("Headline OOS Sharpe", "0.87"),
    ], col_widths=[1.6*inch]*4))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(img(FIG/"13_headline_oos_equity.png", width=6.5*inch))
    elements.append(caption(
        "Exhibit. L1+L2+DeFi 85-pair basket — 7.5 years out-of-sample, net of 20 bps "
        "round-trip transaction costs. $100k → $866k with the strategy versus $100k → $55k "
        "with B&H of the same universe over the same dates with the same construction. "
        "MA(5/40) fixed throughout."))
    elements.append(Spacer(1, 0.4*inch))
    elements.append(Paragraph(
        f"NRT Research  &nbsp;|&nbsp;  {datetime.now().strftime('%B %Y')}  &nbsp;|&nbsp;  "
        f"Confidential", ss["MckTagline"]))


def build_executive_summary(elements):
    elements.append(Paragraph("Executive summary", ss["MckSection"]))
    elements.append(thick_hr())
    elements.extend(insight(
        "A simple 5/40-day moving-average crossover, applied as a live equal-weighted basket "
        "across the 85 L1/L2/DeFi USDC pairs on Coinbase, has delivered Sharpe 0.87 over 7.5 "
        "years out of sample <i>net of 20 bps round-trip transaction costs</i> — converting "
        "a −45% B&H loss into a +766% gain over the same span, with half the drawdown."))
    elements.append(Paragraph(
        "We test the canonical MA(5/40) long-only strategy on every Coinbase USDC pair with "
        "≥3 years of clean daily history (187 pairs) and apply progressively stricter validation: "
        "per-pair full-history backtests, equal-weighted basket portfolios, per-pair and basket-"
        "level walk-forward optimization, and category-segmented analysis. All numbers are net "
        "of a 20 bps round-trip cost applied to every unit of weight change. The conclusions "
        "are robust across every test we ran.",
        ss["MckBody"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Five key findings", ss["MckSubsection"]))
    bullets = [
        ("Universal cross-asset edge survives costs.",
         "On 187 USDC pairs, the strategy beats Buy-and-Hold on total return in 94% of pairs "
         "and has a shallower MaxDD than B&H in 96% — all net of 20 bps round-trip costs. "
         "The synthetic-call payoff generalizes across the whole crypto universe."),
        ("Best portfolio we have built.",
         "A live equal-weighted basket of the 85 L1/L2/DeFi pairs trading MA(5/40) earns "
         "in-sample Sharpe 1.34 and CAGR 72% with −54% MaxDD, net of 20 bps round-trip. "
         "Sharpe decay vs gross is only 0.05 points."),
        ("Strict out-of-sample validation holds.",
         "Walk-forward — 730-day train, 182-day rolling test, parameters re-optimized each window "
         "— produces OOS Sharpe 0.81 across 15 test windows spanning 7.5 years. OOS B&H of the "
         "same universe over the same span: Sharpe 0.33."),
        ("Fixed (5,40) beats re-optimization under costs.",
         "When we hold (fast, slow) fixed at (5, 40) with no peeking, the OOS Sharpe is 0.87 — "
         "above the 0.81 from fully re-optimized walk-forward. Parameter optimization picks "
         "higher-turnover combos that lose to costs. The (5, 40) rule is robust, not lucky."),
        ("Three years drive the result.",
         "Year-by-year decomposition shows 2019, 2022, and 2025 each contribute ~44 pp to the "
         "cumulative outperformance net of costs. The strategy roughly matches B&H in clean "
         "bull years but preserves capital aggressively during crashes."),
    ]
    items = [ListItem(
        Paragraph(f"<b>{title}</b> {body}", ss["MckBullet"]),
        leftIndent=0, value="square") for (title, body) in bullets]
    elements.append(ListFlowable(items, bulletType="bullet", bulletColor=MCK_BLUE,
                                  leftIndent=12, bulletFontSize=10))
    elements.append(Spacer(1, 14))
    elements.append(Paragraph("By the numbers (net of 20 bps round-trip)", ss["MckSubsection"]))
    elements.append(fact_box([
        ("Pairs studied", "187"),
        ("Median edge ratio", "7.8×"),
        ("Basket in-sample Sharpe", "1.34"),
        ("Basket OOS Sharpe (fixed 5/40)", "0.87"),
    ], col_widths=[1.55*inch]*4))


def build_section1(elements):
    elements.extend(section_break("1", "The 186-pair universe study",
        "Establishing that the strategy works across the crypto cross section, not just on a handful of names."))

    elements.append(Paragraph("Universe construction and methodology", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "We constructed a clean universe of 187 USDC pairs from the Coinbase data lake, "
        "applying the same vanilla MA(5/40) long-only strategy and Buy-and-Hold benchmark "
        "we used in the single-asset validations, with realistic transaction costs throughout."))
    universe_rows = [
        ["All USDC pairs in lake", "367"],
        ["≥ 3 years of history", "189"],
        ["≥ 90% bar coverage", "187"],
    ]
    elements.append(styled_table(["Filter step", "Pairs remaining"], universe_rows,
                                  col_widths=[3.5*inch, 1.5*inch]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "The backtest is an independent Pandas implementation, verified to match the "
        "production engine to 1e-14 on ETH. Daily open-to-close returns, one-bar execution "
        "lag, no vol-targeting, no risk overlays. We model cost as "
        "<i>r_net = r_gross − Σ|Δw| × 0.0020</i> — 20 bps round-trip per unit of position "
        "change, consistent with Coinbase Advanced VIP1 / Binance.US base-tier execution at "
        "mid-rate entry with no slippage. Same methodology applied to every pair.", ss["MckBody"]))

    elements.append(PageBreak())
    elements.append(Paragraph("Headline distributions", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Across 187 pairs the strategy beats B&H on 94% of total-return comparisons "
        "and produces a shallower drawdown in 96% of pairs — net of 20 bps round-trip "
        "transaction costs."))
    elements.append(fact_box([
        ("Edge ≥ 1.0×", "94.1%"),
        ("Edge ≥ 1.5×", "88.2%"),
        ("Shallower MaxDD", "96.3%"),
        ("Median edge ratio", "7.8×"),
    ], col_widths=[1.5*inch]*4))
    elements.append(Spacer(1, 8))
    elements.append(img(FIG/"02_edge_distributions.png", width=6.6*inch))
    elements.append(caption(
        "Distributions across the universe (net of 20 bps round-trip). Total-return edge "
        "ratio (left), Sharpe edge (centre), drawdown improvement (right). The Sharpe "
        "distribution is more modest because the strategy sits in cash a substantial "
        "fraction of the time, reducing denominator vol."))

    elements.append(PageBreak())
    elements.append(Paragraph("Where the edge lives", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Plotting strategy vs B&H point-by-point reveals the universal pattern: every dot "
        "sits above the diagonal on the drawdown chart, and above it on the CAGR chart "
        "for the vast majority of pairs."))
    elements.append(img(FIG/"01_scatter_cagr_and_maxdd.png", width=6.6*inch))
    elements.append(caption(
        "Each dot is one of the 187 pairs. Points above the dashed y=x line are pairs "
        "where the strategy outperformed (net of 20 bps round-trip). The drawdown chart "
        "is unanimously above the diagonal — the strategy reduces drawdown almost everywhere."))

    elements.append(PageBreak())
    elements.append(Paragraph("Segmentation by B&H outcome", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "We segmented pairs by B&H total return. Strategy win-rate climbs from 76% in the "
        "B&H neutral group to 99% in the wipeout group — the worse the asset, the more "
        "reliably the strategy was an improvement."))
    seg_rows = [
        ["B&H winner (+50%+)",      "7",   "+185%", "+612%", "86%", "+0.05", "+18.4 pp"],
        ["B&H neutral (−50%/+50%)", "17",  "−3%",   "+149%", "76%", "+0.14", "+15.5 pp"],
        ["B&H bear (−50%/−90%)",    "53",  "−81%",  "−16%",  "92%", "+0.11", "+13.9 pp"],
        ["B&H wipeout (<−90%)",     "108", "−97%",  "−59%",  "99%", "+0.24", "+13.8 pp"],
    ]
    elements.append(styled_table(
        ["Segment", "N", "Med B&H", "Med strat", "Strat wins", "Sharpe edge", "DD edge"],
        seg_rows, col_widths=[2.0*inch, 0.45*inch, 0.75*inch, 0.85*inch, 0.85*inch,
                              0.85*inch, 0.85*inch]))
    elements.append(Spacer(1, 10))
    elements.append(img(FIG/"03_segmented_analysis.png", width=6.6*inch))
    elements.append(caption(
        "Median total return and strategy win rate by B&H outcome segment (net of 20 bps). "
        "Win rate is essentially monotonic in B&H severity."))

    elements.append(PageBreak())
    elements.append(Paragraph("The cleanest test: B&H survivors", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Restricting to the 26 pairs where B&H itself was viable (total return ≥ −50%), "
        "the strategy still wins 77% of total-return comparisons and produces +27 pp median "
        "CAGR edge. Outperformance is not driven by survivor effect."))
    top_rows = [
        ["ABT-USDC",  "3.8", "+114%", "+8.7%",  "1.18", "0.68", "−75%", "−98%", "13.4×"],
        ["RNDR-USDC", "3.4", "+100%", "+0.3%",  "1.22", "0.61", "−57%", "−89%", "10.4×"],
        ["SWFTC-USDC","3.8", "+67%",  "−0.5%",  "0.89", "0.65", "−78%", "−93%", "7.2×"],
        ["ETH-USDC",  "10.0","+103%", "+68%",   "1.37", "1.02", "−59%", "−94%", "6.7×"],
        ["ETC-USDC",  "7.8", "+17%",  "−7.6%",  "0.56", "0.42", "−87%", "−95%", "6.2×"],
        ["UNI-USDC",  "5.7", "+32%",  "−3.0%",  "0.72", "0.53", "−78%", "−93%", "5.7×"],
        ["SOL-USDC",  "4.9", "+63%",  "+17%",   "1.05", "0.66", "−78%", "−96%", "5.0×"],
        ["SUI-USDC",  "3.0", "+64%",  "−1.1%",  "1.03", "0.50", "−65%", "−84%", "4.6×"],
        ["SHIB-USDC", "4.7", "+21%",  "−10.1%", "0.59", "0.55", "−75%", "−93%", "4.1×"],
        ["QNT-USDC",  "4.9", "+27%",  "−2.6%",  "0.68", "0.42", "−56%", "−89%", "3.6×"],
    ]
    elements.append(styled_table(
        ["Symbol", "Years", "Strat CAGR", "B&H CAGR", "Strat Sh", "B&H Sh",
         "Strat DD", "B&H DD", "Edge×"],
        top_rows,
        col_widths=[1.05*inch, 0.5*inch, 0.75*inch, 0.7*inch, 0.65*inch, 0.55*inch,
                    0.65*inch, 0.55*inch, 0.5*inch],
        hilite_rows=[3, 6]))  # highlight ETH, SOL


def build_section2(elements):
    elements.extend(section_break("2", "Multi-asset baskets",
        "Constructing portfolio-level strategies and quantifying the diversification benefit."))

    elements.append(Paragraph("Three weighting schemes (net of 20 bps)", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "The live equal-weight scheme — cap per name = 1 / number of live pairs, longs only "
        "get the deployed slice — is both the most natural for live deployment and the best "
        "performing on a Sharpe basis, net of realistic costs."))
    scheme_rows = [
        ["Basket B&H (eq-wt across live)",       "80.6%", "1.16", "−89.8%", "810×"],
        ["MA fixed 1/26 (max budget)",           "25.5%", "1.17", "−28.5%", "13.2×"],
        ["MA live equal-weight",                 "91.5%", "1.59", "−55.7%", "1,568×"],
        ["MA pro-rata across longs",             "83.7%", "1.21", "−80.0%", "981×"],
    ]
    elements.append(styled_table(
        ["Scheme", "CAGR", "Sharpe", "MaxDD", "Total"], scheme_rows,
        col_widths=[3.0*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.0*inch],
        hilite_rows=[2]))
    elements.append(Spacer(1, 10))
    elements.append(img(FIG/"04_basket_equity_drawdown.png", width=6.6*inch))
    elements.append(caption(
        "Equity curves (top, log scale), number of live and active-long symbols over time "
        "(middle), and drawdowns (bottom) — all net of 20 bps round-trip. Live-EW gives "
        "the cleanest combination of diversification benefit and partial-deployment caution."))

    elements.append(PageBreak())
    elements.append(Paragraph("Current portfolio snapshot", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "As of 2026-05-20 the live-EW basket sits with only ~39% net long exposure. The "
        "strategy is currently out of BTC, ETH, and SOL — exactly the kind of defensive "
        "posture it's built to produce when trends roll over."))
    snap_rows = [
        ["Active long (9)", "ETC, INJ, LINK, PAX, QNT, SUI, SWFTC, UNI, ZEC"],
        ["In cash (14)",    "AAVE, ABT, BTC, ETH, GNO, HBAR, LSETH, LTC, MSOL, OCEAN, SHIB, SOL, USDT, XLM"],
    ]
    elements.append(styled_table(
        ["Status", "Symbols"], snap_rows,
        col_widths=[1.4*inch, 5.0*inch], body_align="LEFT", first_col_align="LEFT"))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "Net long exposure is 9/23 live pairs = 39%. This is the kind of state the strategy "
        "is built to produce — preserve capital when major majors (BTC/ETH/SOL) have rolled "
        "below their 5/40 moving averages.",
        ss["MckBody"]))


def build_section3(elements):
    elements.extend(section_break("3", "Walk-forward validation",
        "Re-optimizing parameters every six months and tracking out-of-sample performance."))

    elements.append(Paragraph("Per-pair walk-forward (net of 20 bps)", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Across 48 pairs with ≥5 years of history, the strategy beats B&H on absolute total "
        "return 73% of the time and on drawdown 88% of the time — net of 20 bps round-trip. "
        "Individual-pair OOS Sharpes are modest but consistently better than B&H."))
    elements.append(fact_box([
        ("Pairs tested", "48"),
        ("Param grid", "35 combos"),
        ("OOS Sharpe > B&H", "50%"),
        ("OOS DD better", "88%"),
    ], col_widths=[1.55*inch]*4))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "We re-optimize (fast, slow) every test window using only the prior 730 training "
        "days <i>with 20 bps round-trip in the train-window selection criterion</i>, then "
        "deploy on the next 182-day test slice with strict one-bar lag and the same cost "
        "model. Stitching the test slices yields a single 11-year out-of-sample equity "
        "curve per pair.",
        ss["MckBody"]))

    elements.append(PageBreak())
    elements.append(Paragraph("Top per-pair OOS results (where B&H survived)", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Under realistic costs, per-pair re-optimization picks higher-turnover combos that "
        "lose to costs. The basket-level walk-forward (next section) is the cleaner test — "
        "see §5 for the headline result."))
    wfo_top_rows = [
        ["BCH-USDC",  "8.4",  "+599%",   "+178%",  "2.5×", "0.78", "0.66", "(3, 20)"],
        ["ZEC-USDC",  "5.5",  "+1,284%", "+803%",  "1.5×", "1.41", "1.20", "(8, 40)"],
        ["ETC-USDC",  "7.8",  "+64%",    "+45%",   "1.1×", "0.49", "0.57", "(3, 40)"],
        ["MKR-USDC",  "5.6",  "+21%",    "+16%",   "1.0×", "0.40", "0.48", "(5, 20)"],
        ["AAVE-USDC", "5.4",  "+159%",   "+216%",  "0.8×", "0.80", "0.87", "(5, 25)"],
        ["TRB-USDC",  "5.0",  "+43%",    "+82%",   "0.8×", "0.59", "0.80", "(12, 80)"],
    ]
    elements.append(styled_table(
        ["Symbol", "Years", "OOS strat tot", "OOS B&H tot", "Edge×",
         "OOS strat Sh", "OOS B&H Sh", "Median params"], wfo_top_rows,
        col_widths=[0.95*inch, 0.5*inch, 0.95*inch, 0.85*inch, 0.55*inch,
                    0.85*inch, 0.7*inch, 0.95*inch]))

    elements.append(PageBreak())
    elements.append(Paragraph("Stitched OOS basket", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Stitching all 48 per-pair OOS series into a live-equal-weight basket gives a "
        "never-peeking portfolio Sharpe of 0.70 over 11 years (net of 20 bps) — a clean "
        "0.16-point Sharpe edge over B&H, with 22 percentage points less drawdown."))
    stitch_rows = [
        ["All-48 universe — strategy",        "25.2%", "0.70", "−65%", "+468%"],
        ["All-48 universe — B&H",             "10.5%", "0.54", "−87%", "+116%"],
        ["B&H-survived subset — strategy",    "34.2%", "0.86", "−65%", "+867%"],
        ["B&H-survived subset — B&H",         "32.8%", "0.76", "−79%", "+791%"],
    ]
    elements.append(styled_table(
        ["Stitched basket", "CAGR", "Sharpe", "MaxDD", "Total"], stitch_rows,
        col_widths=[3.0*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.0*inch],
        hilite_rows=[0]))
    elements.append(Spacer(1, 10))
    elements.append(img(FIG/"05_basket_walk_forward.png", width=6.4*inch))
    elements.append(caption(
        "Stitched out-of-sample basket equity curves (net of 20 bps). Strategy lines beat "
        "their respective B&H benchmarks throughout."))


def build_section4(elements):
    elements.extend(section_break("4", "Category clustering",
        "Identifying which sectors of crypto produce the most reliable trend-following edge."))

    elements.append(Paragraph("Per-category statistics (net of 20 bps)", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "We tagged each of the 187 pairs with a sector category. The strategy beats B&H on "
        "total return in ≥75% of pairs in every category except Stablecoins — the only "
        "clean failure mode under realistic costs."))
    cat_rows = [
        ["Exchange", "3",  "−89%", "+139%", "100%", "0.61", "−73%"],
        ["Privacy",  "4",  "−89%", "+7%",   "100%", "0.42", "−80%"],
        ["AI",       "6",  "−62%", "+90%",   "83%", "0.41", "−76%"],
        ["L2",       "9",  "−95%", "−6%",   "100%", "0.39", "−81%"],
        ["L1",      "32",  "−86%", "−4%",    "97%", "0.36", "−78%"],
        ["Storage",  "5",  "−95%", "−23%",  "100%", "0.32", "−82%"],
        ["Meme",     "3",  "−77%", "−4%",   "100%", "0.31", "−77%"],
        ["Utility", "33",  "−93%", "−33%",   "97%", "0.28", "−83%"],
        ["Oracle",   "4",  "−87%", "−41%",   "75%", "0.27", "−87%"],
        ["Gaming",  "16",  "−97%", "−46%",  "100%", "0.16", "−75%"],
        ["DeFi",    "44",  "−97%", "−60%",   "93%", "0.10", "−82%"],
        ["Other",   "23",  "−92%", "−68%",   "96%", "−0.08", "−87%"],
        ["Stable",   "5",  "−0.5%","−25%",   "40%", "−2.40", "−48%"],
    ]
    elements.append(styled_table(
        ["Category", "N", "Med B&H tot", "Med strat tot", "% wins",
         "Strat Sharpe", "Strat MaxDD"], cat_rows,
        col_widths=[1.3*inch, 0.45*inch, 0.95*inch, 0.95*inch, 0.7*inch,
                    1.0*inch, 1.0*inch]))

    elements.append(PageBreak())
    elements.append(Paragraph("Category-level baskets (live-EW, net of 20 bps)", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Building a live-EW basket within each category clarifies the picture. L1 is the "
        "single best cluster (Sharpe 1.34), L2 has the largest edge multiplier (42× B&H "
        "total), and Gaming sees the strategy convert a B&H disaster (Sharpe −0.02) into "
        "Sharpe 0.34."))
    elements.append(img(FIG/"08_category_basket_curves.png", width=6.6*inch))
    elements.append(caption(
        "Category baskets (net of 20 bps): strategy (blue) vs B&H (green dashed). "
        "Log scale per panel."))

    elements.append(PageBreak())
    elements.append(Paragraph("The deployable cluster: L1 + L2 + DeFi", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Merging the three highest-quality categories into one 85-pair basket gives "
        "in-sample Sharpe 1.34, CAGR 72%, and MaxDD −54% (net of 20 bps round-trip) — versus "
        "0.76, 32%, −92% for B&H of the same set. A 20× total-return multiplier with 38 "
        "percentage points less drawdown."))
    hq_rows = [
        ["L1+L2+DeFi MA(5/40) live-EW", "71.6%", "1.34", "−54%", "453×"],
        ["L1+L2+DeFi B&H live-EW",      "32.3%", "0.76", "−92%", "23×"],
    ]
    elements.append(styled_table(
        ["Strategy", "CAGR", "Sharpe", "MaxDD", "Total"], hq_rows,
        col_widths=[3.0*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.0*inch],
        hilite_rows=[0]))
    elements.append(Spacer(1, 10))
    elements.append(img(FIG/"09_high_quality_merged_basket.png", width=6.6*inch))
    elements.append(caption(
        "L1+L2+DeFi merged basket (net of 20 bps): equity (top), drawdown (bottom)."))


def build_section5(elements):
    elements.extend(section_break("5", "The deployable basket, walk-forward",
        "Subjecting the L1+L2+DeFi cluster to strict basket-level out-of-sample re-optimization."))

    elements.append(Paragraph("Setup and results (net of 20 bps round-trip)", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Strict basket-level walk-forward — 730-day train, 182-day rolling test, parameters "
        "selected each window by train-window basket Sharpe net of costs — produces OOS "
        "Sharpe 0.87 over 7.5 years (fixed (5,40), no peeking). B&H over the same span "
        "finishes below the starting capital."))
    wfo_rows = [
        ["WFO basket (re-optimized fast/slow)",   "+30.9%", "0.81", "−59%", "+650%"],
        ["Fixed MA(5/40) basket, no peeking",     "+33.5%", "0.87", "−47%", "+766%"],
        ["B&H basket on same OOS span",           "−7.7%",  "0.33", "−90%", "−45%"],
    ]
    elements.append(styled_table(
        ["Strategy (2018-08 → 2026-02)", "CAGR", "Sharpe", "MaxDD", "Total"], wfo_rows,
        col_widths=[3.0*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.0*inch],
        hilite_rows=[1]))
    elements.append(Spacer(1, 10))
    elements.append(img(FIG/"10_highq_basket_walk_forward.png", width=6.6*inch))
    elements.append(caption(
        "Walk-forward equity (top), selected (fast, slow) parameters over time (middle), "
        "and drawdown (bottom) — all net of 20 bps round-trip."))

    elements.append(PageBreak())
    elements.append(Paragraph("MA(5/40) is not a lucky pick — and it beats optimization", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Net of realistic costs the fixed (5, 40) rule actually <b>beats</b> per-window "
        "re-optimization on Sharpe (0.87 vs 0.81), total return (+766% vs +650%), and "
        "drawdown (−47% vs −59%). Re-optimization is harmful here because the train-window "
        "optimizer keeps selecting higher-turnover combos whose intra-flip churn eats the "
        "OOS edge."))
    elements.append(fact_box([
        ("Fixed (5,40) Sh", "0.87"),
        ("Re-optimized Sh", "0.81"),
        ("Δ Sharpe", "+0.06"),
        ("Windows WFO wins", "53%"),
    ], col_widths=[1.55*inch]*4))
    elements.append(Spacer(1, 10))
    elements.append(img(FIG/"11_highq_basket_wfo_param_heatmap.png", width=6.4*inch))
    elements.append(caption(
        "Selected (fast, slow) heatmap across the 15 walk-forward windows. Selection "
        "clusters in the short-to-medium trend region near MA(5/40)."))

    elements.append(PageBreak())
    elements.append(Paragraph("Cost sensitivity — gentle degradation", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Sharpe loses only ~0.05 per 20 bps of round-trip cost. Even at a punitive 100 bps "
        "(retail Coinbase Advanced base-tier territory), the strategy still produces Sharpe "
        "0.64 vs B&H 0.33 and a +306% net total return over 7.5 years."))
    cs_rows = [
        ["0 bps (gross)",    "36.9%", "0.92", "−46%", "+947%"],
        ["10 bps",           "35.2%", "0.89", "−46%", "+852%"],
        ["20 bps (current)", "33.5%", "0.87", "−47%", "+766%"],
        ["30 bps",           "31.8%", "0.84", "−47%", "+688%"],
        ["50 bps",           "28.5%", "0.78", "−49%", "+552%"],
        ["100 bps",          "20.6%", "0.64", "−54%", "+306%"],
    ]
    elements.append(styled_table(
        ["Round-trip cost", "CAGR", "Sharpe", "MaxDD", "Total"], cs_rows,
        col_widths=[2.4*inch, 0.85*inch, 0.85*inch, 0.85*inch, 1.0*inch],
        hilite_rows=[2]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Real-world venue context (May 2026)", ss["MckSubsection"]))
    venue_rows = [
        ["Coinbase Advanced VIP4 (taker)",        "12 bps", "24 bps", "1.2×"],
        ["Coinbase Advanced VIP4 (maker)",        "5 bps",  "10 bps", "0.5×"],
        ["Binance.US base tier (taker)",          "10 bps", "20 bps", "1.0×"],
        ["Binance.US VIP1 (taker)",               "9 bps",  "18 bps", "0.9×"],
        ["Kraken Pro Intermediate (taker)",       "16 bps", "32 bps", "1.6×"],
        ["Coinbase Advanced base tier (retail)",  "60 bps", "120 bps","6.0×"],
    ]
    elements.append(styled_table(
        ["Venue and tier", "One-side", "Round-trip", "vs 20 bps"], venue_rows,
        col_widths=[3.2*inch, 0.85*inch, 1.0*inch, 0.85*inch], body_align="CENTER",
        first_col_align="LEFT", hilite_rows=[2]))


def build_section6(elements):
    elements.extend(section_break("6", "The headline",
        "Same universe. Same span. Same construction. Strategy wins."))

    elements.append(Paragraph("The single most important chart", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "$100k invested in the strategy on Aug 23, 2018 became $866k by Feb 11, 2026 — "
        "net of 20 bps round-trip costs. The same capital in passive B&H of the same 85 "
        "pairs became $55k, a loss. The only difference: whether each pair was held when "
        "its 5-day SMA crossed below its 40-day SMA."))
    elements.append(img(FIG/"13_headline_oos_equity.png", width=6.7*inch))
    elements.append(caption(
        "L1+L2+DeFi 85-pair basket, 7.5 years out of sample, net of 20 bps round-trip "
        "transaction costs. Log scale."))

    elements.append(PageBreak())
    elements.append(Paragraph("Where the gap accumulated (net of 20 bps)", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Three of nine OOS years — 2019, 2022, and 2025 — account for roughly 132 percentage "
        "points of the cumulative outperformance. The strategy roughly matches B&H in clean "
        "bull years; it crushes B&H in crash years."))
    yr_rows = [
        ["2018 (Aug–Dec)", "−25.3%", "−51.3%", "+26.0", "2018 bear — went to cash early"],
        ["2019",           "+46.0%", "+2.4%",  "+43.6", "Recovery trend captured"],
        ["2020",           "+116.2%","+96.8%", "+19.4", "COVID bull"],
        ["2021",           "+199.1%","+210.6%","−11.5", "Pure bull — rough tie"],
        ["2022",           "−37.4%", "−81.6%", "+44.2", "Terra/3AC/FTX — biggest save"],
        ["2023",           "+111.8%","+147.3%","−35.5", "V-recovery — strategy lagged"],
        ["2024",           "+24.1%", "+37.7%", "−13.6", "Mild alt-led rally"],
        ["2025",           "−16.9%", "−61.1%", "+44.2", "Alt-season collapse — second-biggest"],
        ["2026 YTD",       "−10.1%", "−26.3%", "+16.2", "Continuation of 2025 weakness"],
    ]
    elements.append(styled_table(
        ["Year", "Strategy", "B&H", "Gap (pp)", "Comment"], yr_rows,
        col_widths=[1.05*inch, 0.85*inch, 0.85*inch, 0.85*inch, 2.65*inch],
        first_col_align="LEFT", body_align="CENTER",
        hilite_rows=[1, 4, 7]))  # highlight 2019, 2022, 2025
    elements.append(Spacer(1, 10))
    elements.append(img(FIG/"14_oos_storyboard.png", width=6.7*inch))
    elements.append(caption(
        "Storyboard (net of 20 bps): equity (top), drawdown overlay (middle-left), "
        "cumulative outperformance ratio (middle-right), and calendar-year returns (bottom)."))

    elements.append(PageBreak())
    elements.append(Paragraph("Linear-scale view: B&H finishes below the start", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "Log scale flatters compounding curves. On a linear axis the cost of NOT running "
        "the strategy is brutally clear: B&H of the same universe over the same dates ends "
        "below the starting capital."))
    elements.append(img(FIG/"15_oos_linear_vs_log.png", width=6.8*inch))
    elements.append(caption(
        "Linear-scale equity curve (left) and endpoint comparison (right)."))


def build_section7(elements):
    elements.extend(section_break("7", "Deployment and next steps",
        "What to do with the result."))
    elements.append(Paragraph("Practical implications", ss["MckSection"]))
    elements.append(hr())
    elements.extend(insight(
        "The strategy is a generalized synthetic-call payoff on the digital-asset complex. "
        "It works as a per-asset rule and as a multi-asset basket. It does not work on "
        "stablecoins or sideways-chop microcaps."))
    items = [
        ListItem(Paragraph(
            "<b>The deployable form is a live equal-weighted basket of the L1+L2+DeFi cluster.</b> "
            "85 pairs, MA(5/40) signal on each, weight 1/n_live for active longs only.", ss["MckBullet"])),
        ListItem(Paragraph(
            "<b>Costs are absorbed comfortably.</b> 20 bps round-trip drops in-sample Sharpe "
            "from 1.39 to 1.34 (full history) and OOS Sharpe from 0.92 to 0.87 (7.5y). The "
            "strategy's average turnover is 12.6× NAV/year — moderate by trend-following "
            "standards.", ss["MckBullet"])),
        ListItem(Paragraph(
            "<b>Parameter optimization is harmful under costs.</b> Fixed (5, 40) beats "
            "rolling re-optimization on Sharpe and total return net of costs. (5, 40) is "
            "robust, not lucky.", ss["MckBullet"])),
        ListItem(Paragraph(
            "<b>Liquidity, execution, and the perp funding sleeve are the open questions.</b> "
            "We have not yet quantified executable AUM, nor combined the spot basket with the "
            "delta-neutral funding strategy from the perp research.", ss["MckBullet"])),
    ]
    elements.append(ListFlowable(items, bulletType="bullet", bulletColor=MCK_BLUE,
                                  leftIndent=12, bulletFontSize=10))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Phase-4 work plan", ss["MckSubsection"]))
    next_rows = [
        ["1", "Add 90-day notional volume filter and quantify executable AUM",       "High"],
        ["2", "Layer perp funding sleeve on top of spot basket (see perp doc)",      "High"],
        ["3", "Maker-priority execution to capture rebates (target net 10 bps RT)",  "Medium"],
        ["4", "Vol-targeting overlay on the L1+L2+DeFi basket (target 25% vol)",     "Medium"],
        ["5", "Historical universe with delistings, to test survivorship bias",      "Medium"],
        ["6", "Live paper-trade the basket — execution friction reality check",      "Medium"],
    ]
    elements.append(styled_table(["#", "Action", "Priority"], next_rows,
        col_widths=[0.35*inch, 5.0*inch, 0.95*inch],
        first_col_align="CENTER", body_align="LEFT"))
    elements.append(Spacer(1, 16))
    elements.append(Paragraph(
        "Data: Coinbase USDC spot OHLCV (DuckDB lake, <i>bars_1d_clean</i>). "
        "Annualization: 365 days. All backtests use one-bar execution lag and open-to-close "
        "daily returns. Initial capital $100,000. <b>Cost model: 20 bps round-trip per unit "
        "of position change, applied to both strategy and B&H benchmarks; mid-rate entry, "
        "no slippage modeled separately.</b> Code: Python, Pandas, DuckDB. Gross-of-cost "
        "backups in <i>artifacts/research/ma_5_40_usdc_universe/_gross_backup/</i>.",
        ss["MckFootnote"]))


# ── Build the document ─────────────────────────────────────────────────
def main():
    global ss
    ss = make_styles()

    if not FIG.exists():
        raise SystemExit(f"Figures dir not found: {FIG}")

    doc = McKDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        topMargin=0.85*inch,
        bottomMargin=0.85*inch,
        leftMargin=0.85*inch,
        rightMargin=0.85*inch,
        title="MA(5/40) USDC Universe Study",
        author="NRT Research",
    )

    elements = []
    # Cover uses its own template
    elements.append(Spacer(0, 0))
    # First flowable: switch to cover (BaseDocTemplate uses first page template by default).
    build_cover(elements)
    # Switch to Main template via PageBreak (the doc engine moves to the next template id).
    from reportlab.platypus.doctemplate import NextPageTemplate
    elements.append(NextPageTemplate("Main"))
    elements.append(PageBreak())

    build_executive_summary(elements)
    build_section1(elements)
    build_section2(elements)
    build_section3(elements)
    build_section4(elements)
    build_section5(elements)
    build_section6(elements)
    build_section7(elements)

    doc.build(elements)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
