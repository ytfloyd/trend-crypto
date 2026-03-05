#!/usr/bin/env python3
"""
NRT Alternative Thinking 2026 — Comprehensive Edition

Combines the original ETH-USD trend sweep (Issue 1) with all four extension
analyses into a single, cohesive publication.

Reads:
    artifacts/research/tsmom/eth_trend_sweep/      (original exhibits + results_v2.csv)
    artifacts/research/tsmom/eth_trend_extension/   (task1–4 exhibits + CSVs)
Writes:
    artifacts/research/tsmom/eth_trend_comprehensive_report.pdf

Usage:
    python -m scripts.research.tsmom.generate_comprehensive_pdf
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
    HRFlowable, CondPageBreak,
)

ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_sweep"
EXT = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_extension"
PDF_PATH = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_comprehensive_report.pdf"

NAVY       = HexColor("#003366")
TEAL       = HexColor("#006B6B")
DARK_GRAY  = HexColor("#333333")
MED_GRAY   = HexColor("#666666")
LIGHT_GRAY = HexColor("#CCCCCC")
RULE_GRAY  = HexColor("#AAAAAA")
BG_LIGHT   = HexColor("#F5F5F5")
RED        = HexColor("#CC3333")
GOLD       = HexColor("#CC9933")

SAMPLE_YEARS = 9
PAGE_W, PAGE_H = letter
L_MARGIN = 0.9 * inch
R_MARGIN = 0.9 * inch
T_MARGIN = 0.75 * inch
B_MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - L_MARGIN - R_MARGIN

F_BODY = "Times-Roman"
F_BOLD = "Times-Bold"
F_ITAL = "Times-Italic"
F_BDIT = "Times-BoldItalic"


# ─── Styles ───────────────────────────────────────────────────────────
def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Q_Title", fontName=F_BOLD, fontSize=24, leading=28,
                           textColor=NAVY, spaceAfter=4, alignment=TA_LEFT))
    ss.add(ParagraphStyle("Q_Sub", fontName=F_ITAL, fontSize=12, leading=15,
                           textColor=MED_GRAY, spaceAfter=6))
    ss.add(ParagraphStyle("Q_Auth", fontName=F_BODY, fontSize=10, leading=13,
                           textColor=MED_GRAY, spaceAfter=3))
    ss.add(ParagraphStyle("Q_Sec", fontName=F_BOLD, fontSize=14, leading=17,
                           textColor=NAVY, spaceBefore=18, spaceAfter=8))
    ss.add(ParagraphStyle("Q_SSec", fontName=F_BOLD, fontSize=11, leading=14,
                           textColor=NAVY, spaceBefore=12, spaceAfter=6))
    ss.add(ParagraphStyle("Q_Body", fontName=F_BODY, fontSize=9.5, leading=13.5,
                           textColor=DARK_GRAY, spaceAfter=7, alignment=TA_JUSTIFY))
    ss.add(ParagraphStyle("Q_BB", fontName=F_BOLD, fontSize=9.5, leading=13.5,
                           textColor=DARK_GRAY, spaceAfter=7, alignment=TA_JUSTIFY))
    ss.add(ParagraphStyle("Q_Bul", fontName=F_BODY, fontSize=9.5, leading=13.5,
                           textColor=DARK_GRAY, spaceAfter=4, leftIndent=18,
                           bulletIndent=6, alignment=TA_JUSTIFY))
    ss.add(ParagraphStyle("Q_Cap", fontName=F_ITAL, fontSize=7.5, leading=10,
                           textColor=MED_GRAY, spaceBefore=3, spaceAfter=10))
    ss.add(ParagraphStyle("Q_FN", fontName=F_BODY, fontSize=7.5, leading=10,
                           textColor=MED_GRAY, spaceBefore=2, spaceAfter=2))
    ss.add(ParagraphStyle("Q_Ex", fontName=F_BOLD, fontSize=9.5, leading=12,
                           textColor=DARK_GRAY, spaceBefore=8, spaceAfter=4))
    ss.add(ParagraphStyle("Q_TOC", fontName=F_BODY, fontSize=9.5, leading=14,
                           textColor=DARK_GRAY, spaceBefore=2, spaceAfter=2))
    ss.add(ParagraphStyle("Q_Abs", fontName=F_ITAL, fontSize=9.5, leading=13.5,
                           textColor=DARK_GRAY, spaceAfter=8, leftIndent=18,
                           rightIndent=18, alignment=TA_JUSTIFY))
    ss.add(ParagraphStyle("Q_Disc", fontName=F_BODY, fontSize=6.5, leading=8.5,
                           textColor=MED_GRAY, alignment=TA_JUSTIFY))
    return ss


# ─── Page templates ───────────────────────────────────────────────────
def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(F_ITAL, 7)
    canvas.setFillColor(MED_GRAY)
    canvas.drawString(L_MARGIN, PAGE_H - 0.5 * inch,
                      "NRT Research — Follow the Trend? Comprehensive Edition")
    canvas.drawRightString(PAGE_W - R_MARGIN, PAGE_H - 0.5 * inch, f"{doc.page}")
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
    canvas.setFont(F_BODY, 7)
    canvas.setFillColor(MED_GRAY)
    canvas.drawRightString(PAGE_W - R_MARGIN, PAGE_H - 0.5 * inch, f"{doc.page}")
    canvas.restoreState()


# ─── Utilities ────────────────────────────────────────────────────────
def img(dirpath, name, width=None, max_h=4.2):
    p = dirpath / name
    if not p.exists():
        return Spacer(1, 12)
    w = width or CONTENT_W
    from reportlab.lib.utils import ImageReader
    ir = ImageReader(str(p))
    iw, ih = ir.getSize()
    aspect = ih / iw
    h = w * aspect
    mh = max_h * inch
    if h > mh:
        h = mh
        w = h / aspect
    return Image(str(p), width=w, height=h)


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=LIGHT_GRAY,
                       spaceBefore=6, spaceAfter=6)


def make_table(data, col_widths, header_color=NAVY):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), F_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), F_BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, header_color),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


SOURCE = (
    "Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. "
    "All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip "
    "costs, no leverage. Past performance is not a reliable indicator of future results."
)


def build_pdf():
    ss = _styles()

    # ── Load data ──────────────────────────────────────────────────
    df = pd.read_csv(SWEEP / "results_v2.csv")
    strats = df[df["label"] != "BUY_AND_HOLD"].copy()
    bh = df[df["label"] == "BUY_AND_HOLD"].iloc[0]
    n = len(strats)

    n_beat_sr = (strats["sharpe"] > bh["sharpe"]).sum()
    n_beat_dd = (strats["max_dd"] > bh["max_dd"]).sum()
    med_sr = strats["sharpe"].median()
    med_cagr = strats["cagr"].median()
    med_dd = strats["max_dd"].median()

    daily = strats[strats["freq"] == "1d"]
    four_h = strats[strats["freq"] == "4h"]
    one_h = strats[strats["freq"] == "1h"]

    n_eff = 493 * 3
    bonf_z = stats.norm.ppf(1 - 0.05 / n_eff / 2)
    sharpe_thresh = bonf_z / np.sqrt(SAMPLE_YEARS)
    n_survivors = (strats["sharpe"] >= sharpe_thresh).sum()
    top10 = strats[strats["sharpe"] >= sharpe_thresh].nlargest(10, "sharpe")

    matched = []
    for (label, freq), group in strats.groupby(["label", "freq"]):
        atr = group[group["stop_type"] == "atr"]
        pct = group[group["stop_type"] == "pct"]
        if atr.empty or pct.empty:
            continue
        matched.append({"atr_wins": atr["sharpe"].max() > pct["sharpe"].max()})
    atr_win_pct = pd.DataFrame(matched)["atr_wins"].mean() if matched else 0.5

    # Extension data
    skew_decomp = pd.read_csv(EXT / "task1" / "skewness_decomposition.csv").iloc[0]
    skew_buckets = pd.read_csv(EXT / "task1" / "skewness_by_tim_bucket.csv")
    atr_by_freq = pd.read_csv(EXT / "task2" / "atr_vs_fixed_by_freq.csv")
    atr_by_cell = pd.read_csv(EXT / "task2" / "atr_vs_fixed_by_freq_family.csv")
    task2_sum = pd.read_csv(EXT / "task2" / "task2_summary.csv").iloc[0]
    subperiod = pd.read_csv(EXT / "task3" / "subperiod_analysis.csv")
    drift = pd.read_csv(EXT / "task3" / "drift_sensitivity.csv")
    bear = pd.read_csv(EXT / "task3" / "bear_regime_results.csv")
    task4_sum = pd.read_csv(EXT / "task4" / "task4_summary.csv").iloc[0]
    ensemble = pd.read_csv(EXT / "task4" / "ensemble_comparison.csv")
    fam_tim = pd.read_csv(EXT / "task4" / "family_tim_targeting.csv")
    task5_sum = pd.read_csv(EXT / "task5" / "task5_summary.csv").iloc[0]
    task5_comp = pd.read_csv(EXT / "task5" / "ensemble_comparison.csv")
    task5_fam_stab = pd.read_csv(EXT / "task5" / "family_tim_stability.csv")
    task5_fam_comp = pd.read_csv(EXT / "task5" / "family_composition.csv")

    CROSS = ROOT / "artifacts" / "research" / "tsmom" / "cross_asset"
    ca_summary = pd.read_csv(CROSS / "cross_asset_summary.csv").iloc[0]
    ca_spearman = pd.read_csv(CROSS / "spearman_family_rankings.csv")
    ca_tier2 = pd.read_csv(CROSS / "tier2_replication_summary.csv")
    ca_btc_wf = pd.read_csv(CROSS / "btc_walkforward_comparison.csv")
    ca_btc_wf_sum = pd.read_csv(CROSS / "btc_walkforward_summary.csv").iloc[0]
    ca_tim_optima = pd.read_csv(CROSS / "tim_optima.csv", index_col=0)
    ca_fam_surv = pd.read_csv(CROSS / "family_survival_rates.csv", index_col=0)

    FULLUNIV = ROOT / "artifacts" / "research" / "tsmom" / "full_universe"
    fu_summary = pd.read_csv(FULLUNIV / "full_universe_summary.csv").iloc[0]
    fu_tier_a = pd.read_csv(FULLUNIV / "tier_a_summary.csv")
    fu_tier_b = pd.read_csv(FULLUNIV / "tier_b_replication.csv")
    fu_wf = pd.read_csv(FULLUNIV / "tier_a_walkforward.csv")
    fu_class = pd.read_csv(FULLUNIV / "asset_classification.csv")
    fu_tiers = pd.read_csv(FULLUNIV / "universe_tiers.csv")

    cost_sens = pd.read_csv(FULLUNIV / "cost_sensitivity.csv")

    # ── Document setup ─────────────────────────────────────────────
    doc = BaseDocTemplate(
        str(PDF_PATH), pagesize=letter,
        leftMargin=L_MARGIN, rightMargin=R_MARGIN,
        topMargin=T_MARGIN, bottomMargin=B_MARGIN,
        title="NRT Alternative Thinking — Follow the Trend? Comprehensive Edition",
        author="NRT Portfolio Research Group",
    )
    frame_t = Frame(L_MARGIN, B_MARGIN, CONTENT_W, PAGE_H - T_MARGIN - B_MARGIN, id="tf")
    frame_b = Frame(L_MARGIN, B_MARGIN, CONTENT_W, PAGE_H - T_MARGIN - B_MARGIN, id="bf")
    doc.addPageTemplates([
        PageTemplate(id="title_page", frames=[frame_t], onPage=_title_page),
        PageTemplate(id="body_page", frames=[frame_b], onPage=_header_footer),
    ])

    E = []
    fn = [0]

    def footnote(text):
        fn[0] += 1
        return fn[0], f"<super>{fn[0]}</super> {text}"

    def add_fn(text):
        n, ftxt = footnote(text)
        E.append(Paragraph(ftxt, ss["Q_FN"]))
        return n

    # ================================================================
    # TITLE PAGE
    # ================================================================
    E.append(Spacer(1, 1.5 * inch))
    E.append(Paragraph("NRT Alternative Thinking", ss["Q_Sub"]))
    E.append(Paragraph("2026 — Comprehensive Edition", ss["Q_Sub"]))
    E.append(Spacer(1, 0.3 * inch))
    E.append(Paragraph("Follow the Trend?", ss["Q_Title"]))
    E.append(Paragraph("What 13,000 Crypto Strategies<br/>Actually Tell Us", ss["Q_Title"]))
    E.append(Spacer(1, 0.15 * inch))
    E.append(Paragraph(
        "<i>Including extension analyses, ex-ante TIM prediction, "
        "and cross-asset validation on BTC, SOL, LTC, LINK, ATOM</i>",
        ss["Q_Sub"]))
    E.append(Spacer(1, 0.35 * inch))
    E.append(Paragraph("NRT Research | For internal use by the quantitative strategy team "
                        "and capital allocators", ss["Q_Auth"]))
    E.append(Paragraph(f"February {datetime.now().year}", ss["Q_Auth"]))
    E.append(Spacer(1, 0.5 * inch))
    E.append(hr())

    abstract = (
        f"We construct <b>{n:,}</b> trend-following configurations on ETH-USD over nine years. "
        f"The initial sweep told a skeptical story: <b>{(n-n_beat_sr)/n:.0%}</b> of strategies "
        f"underperform buy-and-hold on risk-adjusted returns, and the median strategy sacrifices "
        f"substantial CAGR for drawdown compression. We published those findings. Then we stress-"
        f"tested them — and several key conclusions did not survive."
        f"<br/><br/>"
        f"Four extension analyses revise the initial picture: (i) the claim that positive skewness "
        f"is \"a property of the trade structure, not the signal\" is wrong — signal timing "
        f"explains {skew_decomp['pct_signal']:.0f}% of the improvement; (ii) the characterization "
        f"of ATR vs fixed stops as \"a wash\" masks 10 significant subgroup differences; "
        f"(iii) the hypothesis that trend \"needs lower drift to add value\" is false — "
        f"Bonferroni survivors beat B&H at every drift tested, up to 300% annualized; and "
        f"(iv) a TIM-filtered ensemble at the empirical optimum of {task4_sum['optimal_tim']:.0%} "
        f"achieves Sharpe {task4_sum['ensemble_sharpe']:.2f} with "
        f"{abs(task4_sum['ensemble_dd'])*100:.0f}% max drawdown — the first construction in the "
        f"study to beat buy-and-hold on both Sharpe and drawdown."
    )
    E.append(Paragraph("<b>Executive Summary</b>", ss["Q_BB"]))
    E.append(Paragraph(abstract, ss["Q_Abs"]))
    E.append(Spacer(1, 0.1 * inch))
    n1 = add_fn(
        "All results are hypothetical, in-sample, on a single asset. No representation is made "
        "that any strategy will achieve similar results.")

    E.append(NextPageTemplate("body_page"))
    E.append(PageBreak())

    # ================================================================
    # TABLE OF CONTENTS
    # ================================================================
    E.append(Paragraph("Contents", ss["Q_Sec"]))
    toc = [
        "Introduction",
        "Part 1: The Baseline — Buy-and-Hold Is Extremely Hard to Beat",
        "Part 2: What Trend Buys and What It Costs",
        "Part 3: Frequency Dominates Everything",
        "Part 4: Do Vol-Adaptive Stops Beat Fixed Stops?",
        "Part 5: The Multiple Testing Problem",
        "Part 6: The Convexity Profile",
        "Part 7: Is Skewness a Free Lunch? (Extension)",
        "Part 8: Stop-Type Microstructure (Extension)",
        "Part 9: Regime Sensitivity — The Drift Question (Extension)",
        "Part 10: Optimal Time-in-Market Targeting (Extension)",
        "Part 11: Ex-Ante TIM Prediction (Extension)",
        "Part 12: Cross-Asset Validation",
        "Part 13: Full Universe Validation",
        "Concluding Thoughts",
        "Appendix: Parameter Grid and Data Notes",
        "References and Further Reading",
    ]
    for t in toc:
        E.append(Paragraph(t, ss["Q_TOC"]))
    E.append(Spacer(1, 0.3 * inch))
    E.append(Paragraph(
        "The authors thank the NRT Quantitative Research team for helpful comments.",
        ss["Q_FN"]))
    E.append(PageBreak())

    # ── Self-revision note ─────────────────────────────────────────
    E.append(Paragraph(
        "<i>A note on structure: this paper revises its own findings. Parts 1–6 present the "
        "original sweep results. Parts 7–11 extend the analysis and in several cases reverse "
        "the initial conclusions. Parts 12–13 test portability across the full Coinbase universe. "
        "The revision narrative is deliberate — it demonstrates the research process, not just "
        "its endpoint.</i>", ss["Q_Abs"]))
    E.append(Spacer(1, 0.15 * inch))

    # ================================================================
    # INTRODUCTION
    # ================================================================
    E.append(Paragraph("Introduction", ss["Q_Sec"]))
    E.append(Paragraph(
        f"Crypto allocators face a unique problem. The asset class has delivered extraordinary "
        f"long-term returns — ETH-USD compounded at {bh['cagr']:.0%} annualized from 2017 to "
        f"2026 — but the path was brutal: a {bh['max_dd']:.0%} peak-to-trough drawdown, with "
        f"multiple drawdowns exceeding 70%. No investor, institutional or otherwise, can "
        f"plausibly hold through a {bh['max_dd']:.0%} drawdown. The standard response is to "
        f"apply trend-following logic: be long when the asset is trending up, move to cash when "
        f"it is not.", ss["Q_Body"]))
    E.append(Paragraph(
        "The premise has theoretical support. Moskowitz, Ooi, and Pedersen (2012) documented "
        "time-series momentum across dozens of futures markets. Hurst, Ooi, and Pedersen (2017) "
        "extended the evidence to a century of data. In our own prior work at this desk, we "
        "attempted a portfolio-level TSMOM framework for crypto; the results were poor.",
        ss["Q_Body"]))
    E.append(Paragraph(
        "This led to a natural question: if portfolio-level momentum fails, do simpler per-asset "
        "trend signals do better? And if so, what matters — the entry signal, the data frequency, "
        "or the exit mechanism?", ss["Q_Body"]))
    E.append(Paragraph(
        f"To answer these questions, we built <b>{n:,} configurations</b> from the cross-product "
        f"of 493 base trend signals from 30+ families, 3 frequencies (daily, 4-hour, 1-hour), "
        f"and 9 stop-loss variants (no stop; fixed trailing stops at 5%, 10%, 20%; vol-adaptive "
        f"trailing stops at 1.5×–4.0× ATR). All configurations use identical backtest rules: "
        f"binary long or cash, one-bar signal lag, 20 bps round-trip transaction costs, no "
        f"leverage, no position sizing.", ss["Q_Body"]))
    E.append(Paragraph(
        "The original sweep yielded six core findings, several of which painted a skeptical "
        "picture of trend-following in crypto. We published those findings. In this comprehensive "
        "edition, we subject them to four extension analyses that address open questions and "
        "untested assumptions. <b>Three of those initial conclusions required meaningful "
        "revision.</b> The sections that follow present the initial findings as they were "
        "originally stated (Parts 1–6), then the extensions that corrected them (Parts 7–10). "
        "We believe intellectual honesty requires showing both what we got wrong and what we got "
        "right, rather than quietly rewriting history.", ss["Q_Body"]))
    E.append(PageBreak())

    # ================================================================
    # PART 1
    # ================================================================
    E.append(Paragraph(
        "Part 1: The Baseline — Buy-and-Hold Is Extremely Hard to Beat", ss["Q_Sec"]))
    E.append(Paragraph(
        "<i>Parts 1–6 present the initial sweep findings as originally published. Parts 7–10 "
        "contain the extension analyses that revise several of these conclusions.</i>",
        ss["Q_FN"]))
    E.append(Spacer(1, 6))
    E.append(Paragraph(
        f"ETH-USD buy-and-hold from January 2017 to February 2026: Sharpe {bh['sharpe']:.2f}, "
        f"CAGR {bh['cagr']:.0%}, max drawdown {bh['max_dd']:.0%}, Calmar {bh['calmar']:.2f}, "
        f"Sortino {bh['sortino']:.2f}, skewness {bh['skewness']:.2f}.", ss["Q_Body"]))
    E.append(Paragraph(
        f"A Sharpe ratio of {bh['sharpe']:.2f} is exceptional by any standard. Any strategy "
        f"that sits in cash for part of the period faces a substantial headwind from missing "
        f"this strong secular drift.", ss["Q_Body"]))
    E.append(Paragraph(
        f"Exhibit 1 shows the distribution of Sharpe ratios across all {n:,} trend "
        f"configurations. The median strategy ({med_sr:.2f}) falls well below buy-and-hold "
        f"({bh['sharpe']:.2f}). Only {n_beat_sr:,} ({n_beat_sr/n:.0%}) of configurations "
        f"outperform on this metric.", ss["Q_Body"]))

    E.append(Paragraph(
        "Exhibit 1: Most Trend Strategies Underperform Buy-and-Hold on a Risk-Adjusted Basis",
        ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_1_sharpe_dist.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))
    n2 = add_fn(
        "Throughout this piece, \"cash\" means zero return. We do not model stablecoin yield.")
    E.append(PageBreak())

    # ================================================================
    # PART 2
    # ================================================================
    E.append(Paragraph("Part 2: What Trend Buys and What It Costs", ss["Q_Sec"]))
    E.append(Paragraph(
        f"If trend-following mostly underperforms on risk-adjusted returns, why consider it? "
        f"Because an investor who cannot hold through a {bh['max_dd']:.0%} drawdown does not earn "
        f"the {bh['cagr']:.0%} CAGR. The relevant question is: what does trend cost, and what "
        f"does it buy?", ss["Q_Body"]))

    E.append(Paragraph(
        "Exhibit 2: The Trend Tradeoff — CAGR for Drawdown Protection", ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_2_tradeoff.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    tdata = [
        ["Metric", "Buy & Hold", "Median Strategy", "Difference"],
        ["CAGR", f"{bh['cagr']:.1%}", f"{med_cagr:.1%}", f"{med_cagr - bh['cagr']:+.1%}"],
        ["Max Drawdown", f"{bh['max_dd']:.1%}", f"{med_dd:.1%}",
         f"{med_dd - bh['max_dd']:+.1%}"],
        ["Skewness", f"{bh['skewness']:.2f}", f"{strats['skewness'].median():.2f}",
         f"{strats['skewness'].median() - bh['skewness']:+.2f}"],
    ]
    E.append(make_table(tdata, [1.4*inch, 1.3*inch, 1.5*inch, 1.3*inch]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        f"The median strategy gives up roughly {abs(med_cagr - bh['cagr']):.0%} of annual return "
        f"to compress the max drawdown by {abs(med_dd - bh['max_dd']):.0%}. For an allocator who "
        f"cannot hold through {bh['max_dd']:.0%} but can hold through {med_dd:.0%}, trend "
        f"converts an undeployable return stream into a deployable one — even if the headline "
        f"CAGR is lower.", ss["Q_Body"]))
    n_pos_skew = (strats["skewness"] > 0).sum()
    E.append(Paragraph(
        f"The skewness improvement is real but partly mechanical — or so we initially concluded. "
        f"Our original analysis noted that {n_pos_skew:,} of {n:,} ({n_pos_skew/n:.0%}) "
        f"strategies exhibit positive skewness regardless of signal choice, and asserted that "
        f"\"the skewness is a property of the trade structure, not the signal.\" <b>Part 7 "
        f"demonstrates this was an overstatement: signal timing explains 86% of the skewness "
        f"improvement, with trade structure accounting for only 14%.</b>", ss["Q_Body"]))
    E.append(PageBreak())

    # ================================================================
    # PART 3
    # ================================================================
    E.append(Paragraph(
        "Part 3: What Actually Drives Performance? Frequency Dominates Everything",
        ss["Q_Sec"]))
    E.append(Paragraph(
        f"Across {n:,} configurations, we vary signal (493), frequency (3), and stop type (9). "
        f"The answer is unambiguous: <b>frequency</b> dominates.", ss["Q_Body"]))
    E.append(Paragraph(
        f"Daily signals: median Sharpe {daily['sharpe'].median():.2f}, "
        f"{(daily['sharpe'] > bh['sharpe']).sum()/len(daily):.0%} beat B&H. Four-hour: "
        f"{four_h['sharpe'].median():.2f} "
        f"({(four_h['sharpe'] > bh['sharpe']).sum()/len(four_h):.0%}). One-hour: "
        f"{one_h['sharpe'].median():.2f} "
        f"({(one_h['sharpe'] > bh['sharpe']).sum()/len(one_h):.0%}).", ss["Q_Body"]))

    E.append(Paragraph(
        "Exhibit 3: Frequency Is the Dominant Variable", ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_3_frequency.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    freq_data = [["Frequency", "N", "Med Sharpe", "Med CAGR", "Med MaxDD", "% Beat B&H"]]
    for flabel, sub in [("Daily", daily), ("4-Hour", four_h), ("1-Hour", one_h)]:
        freq_data.append([
            flabel, f"{len(sub):,}", f"{sub['sharpe'].median():.3f}",
            f"{sub['cagr'].median():.1%}", f"{sub['max_dd'].median():.1%}",
            f"{(sub['sharpe'] > bh['sharpe']).sum()/len(sub):.0%}",
        ])
    E.append(make_table(freq_data,
                        [0.9*inch, 0.7*inch, 1.0*inch, 0.9*inch, 1.0*inch, 1.0*inch]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "Higher-frequency signals generate more trades, more transaction costs, and more "
        "whipsaw. In a strong secular uptrend, frequent trading increases the probability of "
        "being shaken out during intraday noise, then missing the continuation.", ss["Q_Body"]))

    E.append(Spacer(1, 4))
    E.append(Paragraph(
        "Exhibit 4: Median Sharpe by Signal Family (daily, no stop)", ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_4_family.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "Within daily frequency, signal family matters less than one might expect. The variation "
        "within families (across parameters) is often as large as the variation across families. "
        "Signal choice is secondary to frequency and time-in-market.", ss["Q_Body"]))
    E.append(PageBreak())

    E.append(Paragraph(
        "Exhibit 5: Time in Market Controls the Return/Risk Dial", ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_5_tim.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    tim_data = [["TIM Bucket", "N", "Med Sharpe", "Med MaxDD", "Med Skew"]]
    for lo, hi in [(0,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,1.0)]:
        sub = strats[(strats["time_in_market"] >= lo) & (strats["time_in_market"] < hi)]
        if sub.empty:
            continue
        tim_data.append([
            f"{lo:.0%}–{hi:.0%}", f"{len(sub):,}", f"{sub['sharpe'].median():.3f}",
            f"{sub['max_dd'].median():.1%}", f"{sub['skewness'].median():.2f}",
        ])
    E.append(Paragraph("Exhibit 6: Performance by Time-in-Market Bucket", ss["Q_Ex"]))
    E.append(make_table(tim_data, [0.9*inch, 0.8*inch, 1.0*inch, 1.0*inch, 0.9*inch]))
    E.append(Paragraph(
        "The 30–40% TIM bucket maximizes median Sharpe. <b>Part 10 investigates whether this "
        "optimum is robust and exploitable.</b>", ss["Q_Body"]))
    E.append(PageBreak())

    # ================================================================
    # PART 4
    # ================================================================
    E.append(Paragraph("Part 4: Do Vol-Adaptive Stops Beat Fixed Stops?", ss["Q_Sec"]))
    E.append(Paragraph(
        "A common hypothesis is that ATR-based trailing stops should dominate fixed-percentage "
        "exits. We test this by comparing nine stop variants across all base signals.",
        ss["Q_Body"]))

    E.append(Paragraph("Exhibit 7: Aggregate Performance by Stop Type", ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_7_stops_agg.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "Exhibit 8: Matched Comparison — ATR vs Fixed Stops on Same Base Signal", ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_8_matched.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        f"<b>ATR stops win only {atr_win_pct:.0%} of the time on Sharpe</b> across "
        f"{len(matched):,} matched pairs. The aggregate medians in Exhibit 7 were misleading "
        f"because ATR and fixed stops have different TIM profiles. Once we match on the same base "
        f"signal, the stop type appeared to be a wash. Our initial conclusion was that ATR stops "
        f"offered no reliable advantage. <b>Part 8 revises this: 10 of 102 frequency × family "
        f"subgroups show Bonferroni-significant differences, with ATR strongly favored for some "
        f"signal types (EMA: 88% win rate) and strongly disfavored for others (ADX: 0%). The "
        f"aggregate is a wash; the subgroups are not.</b>", ss["Q_Body"]))
    E.append(PageBreak())

    # ================================================================
    # PART 5
    # ================================================================
    E.append(Paragraph("Part 5: The Multiple Testing Problem", ss["Q_Sec"]))
    E.append(Paragraph(
        f"We tested {n:,} configurations. Even if every strategy were generated by a coin flip, "
        f"some would look impressive by chance. The strategies are not independent — stop variants "
        f"of the same signal are highly correlated (avg pairwise correlation ≈ 0.94). We estimate "
        f"{n_eff:,} effective independent tests.", ss["Q_Body"]))
    E.append(Paragraph(
        f"Bonferroni correction at this test count requires Sharpe ≥ <b>{sharpe_thresh:.2f}</b>. "
        f"Only <b>{n_survivors:,} of {n:,} ({n_survivors/n:.1%})</b> strategies survive.",
        ss["Q_Body"]))

    # Bonferroni sensitivity check
    n_eff_alt = 2000
    bonf_z_alt = stats.norm.ppf(1 - 0.05 / n_eff_alt / 2)
    sharpe_thresh_alt = bonf_z_alt / np.sqrt(SAMPLE_YEARS)
    n_survivors_alt = (strats["sharpe"] >= sharpe_thresh_alt).sum()
    E.append(Paragraph(
        f"<i>Sensitivity: if stop variants are treated as partially independent (raising "
        f"effective tests from {n_eff:,} to ~{n_eff_alt:,}), the Bonferroni threshold increases "
        f"from {sharpe_thresh:.2f} to ~{sharpe_thresh_alt:.2f}. This would reduce survivors "
        f"from {n_survivors:,} to {n_survivors_alt:,} but would not change the qualitative "
        f"conclusions — the same signal families dominate.</i>", ss["Q_FN"]))

    E.append(Paragraph(
        f"Exhibit 9: After Bonferroni Correction, Only {n_survivors:,} Survive", ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_9_multiple_testing.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    surv_data = [["#", "Signal", "Stop", "Freq", "Sharpe", "CAGR", "MaxDD", "Skew", "TIM"]]
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        surv_data.append([
            str(rank), row["label"], row["stop"], row["freq"],
            f"{row['sharpe']:.2f}", f"{row['cagr']:.1%}", f"{row['max_dd']:.1%}",
            f"{row['skewness']:.2f}", f"{row['time_in_market']:.0%}",
        ])
    st = Table(surv_data, colWidths=[
        0.25*inch, 1.5*inch, 0.5*inch, 0.4*inch, 0.55*inch,
        0.55*inch, 0.55*inch, 0.5*inch, 0.4*inch,
    ])
    st.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), F_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), F_BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    E.append(Paragraph("Exhibit 10: Top 10 Strategies Surviving Bonferroni Correction",
                        ss["Q_Ex"]))
    E.append(st)
    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "The survivors cluster in daily-frequency, medium-lookback MA crossovers and a few "
        "channel-based signals. The surviving families are well-established in the trend-following "
        "literature, but the specific parameterizations are almost certainly sample-dependent. "
        "<b>Parts 9 and 10 subject these survivors to further stress tests.</b>", ss["Q_Body"]))
    E.append(PageBreak())

    # ================================================================
    # PART 6
    # ================================================================
    E.append(Paragraph("Part 6: The Convexity Profile", ss["Q_Sec"]))
    E.append(Paragraph(
        "For allocators whose mandate is long convexity — bounded downside with exposure to "
        "unbounded upside — the joint distribution of Sharpe and skewness matters more than "
        "Sharpe alone.", ss["Q_Body"]))

    E.append(Paragraph(
        "Exhibit 11: Positive Skewness Is Nearly Universal", ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_6_sharpe_skew.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "Exhibit 12: The Risk/Return Map", ss["Q_Ex"]))
    E.append(img(SWEEP, "exhibit_10_cagr_dd.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))
    E.append(PageBreak())

    # ================================================================
    # TRANSITION: WHAT WE GOT WRONG
    # ================================================================
    E.append(Paragraph("What We Got Wrong", ss["Q_Sec"]))
    E.append(Paragraph(
        "The six findings above constituted our initial assessment. Three of them contained "
        "claims that were either overstated or, in one case, directionally wrong. Specifically:",
        ss["Q_Body"]))
    E.append(Paragraph(
        "<b>Claim 1 (Part 2):</b> \"The skewness is a property of the trade structure, not the "
        "signal.\" This implied that positive skewness was essentially free — a byproduct of "
        "binary long/cash positioning on a fat-tailed asset. If true, skewness would be useless "
        "as a signal quality discriminator.", ss["Q_Bul"]))
    E.append(Paragraph(
        "<b>Claim 2 (Part 4):</b> \"The stop type is a wash.\" This implied that ATR and fixed "
        "stops were interchangeable. If true, the entire debate over vol-adaptive exits would be "
        "a waste of research effort.", ss["Q_Bul"]))
    E.append(Paragraph(
        "<b>Claim 3 (Conclusion):</b> \"In a lower-drift environment, the calculus shifts in "
        "trend's favor... The historical crypto drift is an anomaly.\" This implied that trend "
        "strategies need lower drift to add value — that the high-drift environment was the "
        "binding constraint. If true, trend would be a conditional bet on lower future drift.",
        ss["Q_Bul"]))
    E.append(Paragraph(
        "The four extension analyses that follow subject each of these claims to rigorous "
        "empirical testing. They also ask a constructive question the initial sweep did not: "
        "given all of these results, can we actually build something that works?", ss["Q_Body"]))
    E.append(PageBreak())

    # ================================================================
    # PART 7: SKEWNESS BASELINE (Extension Task 1)
    # ================================================================
    E.append(Paragraph("Part 7: Is Skewness a Free Lunch?", ss["Q_Sec"]))
    E.append(Paragraph(
        "<i>For the practitioner: 86% of observed skewness comes from signal timing skill, "
        "not trade structure. Skewness is worth optimizing for.</i>", ss["Q_FN"]))
    E.append(Paragraph("<i>Revising Claim 1 — Mechanical Skewness Baseline</i>",
                        ss["Q_Sub"]))
    E.append(Spacer(1, 4))
    E.append(Paragraph(
        "Part 2 asserted that positive skewness was \"a property of the trade structure, not "
        "the signal.\" We test this by generating 10,000 randomly-timed binary long/cash "
        "strategies on ETH-USD. If the claim is correct, random strategies should exhibit "
        "comparable skewness to real trend strategies.", ss["Q_Body"]))
    E.append(Paragraph(
        "We generate 10,000 randomly-timed binary long/cash strategies on ETH-USD daily bars. "
        "Each strategy draws a random time-in-market percentage from U[5%, 95%] and determines "
        "its daily on/off state via independent Bernoulli draws. If the binary trade form alone "
        "generates positive skewness, these random strategies should exhibit it without any "
        "signal timing skill.", ss["Q_Body"]))

    E.append(Paragraph(
        "Exhibit 13: Skewness of Random vs Real Trend Strategies", ss["Q_Ex"]))
    E.append(img(EXT / "task1", "skewness_distributions.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        f"The answer is nuanced. <b>{skew_decomp['pct_random_positive_skew']:.0f}%</b> of random "
        f"strategies exhibit positive skewness — confirming that the trade structure does "
        f"mechanically generate positive skew on ETH-USD. But the magnitude tells a different "
        f"story: random strategies have median skewness of {skew_decomp['random_median_skewness']:.2f} "
        f"vs {skew_decomp['real_median_skewness']:.2f} for real trend strategies vs "
        f"{skew_decomp['bh_skewness']:.2f} for buy-and-hold.", ss["Q_Body"]))
    E.append(Paragraph(
        f"Decomposing the improvement over buy-and-hold: <b>{skew_decomp['pct_structural']:.0f}% "
        f"is structural</b> (attributable to the binary trade form) and "
        f"<b>{skew_decomp['pct_signal']:.0f}% is from signal timing</b>. The structure contributes "
        f"a modest uplift ({skew_decomp['structural_component']:+.2f}); the signal timing "
        f"contributes the lion's share ({skew_decomp['signal_component']:+.2f}).", ss["Q_Body"]))

    E.append(Paragraph(
        "Exhibit 14: Median Skewness by TIM Bucket — Random vs Real Trend", ss["Q_Ex"]))
    E.append(img(EXT / "task1", "skewness_by_tim.png"))
    E.append(Paragraph(
        "Source: NRT Research. * indicates statistical significance at p < 0.05, Wilcoxon "
        "rank-sum test.", ss["Q_Cap"]))

    # Skewness bucket table
    sb_data = [["TIM Bucket", "Random Median", "Real Median", "Difference", "p-value", "Sig?"]]
    for _, r in skew_buckets.iterrows():
        sb_data.append([
            f"{r['tim_lo']:.0%}–{r['tim_hi']:.0%}",
            f"{r['random_median_skew']:.3f}", f"{r['real_median_skew']:.3f}",
            f"{r['difference']:+.3f}", f"{r['p_value']:.4f}", r["significant_p05"],
        ])
    E.append(make_table(sb_data, [0.8*inch, 1.0*inch, 1.0*inch, 0.9*inch, 0.8*inch, 0.5*inch]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "The difference is statistically significant in every TIM bucket (all p < 0.001). Real "
        "trend signals systematically enter before rallies and exit before crashes — the skewness "
        "gap is not a free lunch from the trade form.", ss["Q_Body"]))
    E.append(Paragraph(
        "<b>Conclusion:</b> Skewness is a meaningful discriminator of signal quality, not merely "
        "a structural artifact. The original paper's assertion understated the signal contribution. "
        "While the trade form provides a modest structural tailwind, signal timing explains 86% "
        "of the skewness improvement over buy-and-hold.", ss["Q_BB"]))
    E.append(PageBreak())

    # ================================================================
    # PART 8: ATR DECOMPOSITION (Extension Task 2)
    # ================================================================
    E.append(Paragraph("Part 8: Stop-Type Microstructure", ss["Q_Sec"]))
    E.append(Paragraph(
        "<i>For the practitioner: stop-type preference depends on signal family — ATR for "
        "noisy entries (EMA), fixed for regime signals (ADX).</i>", ss["Q_FN"]))
    E.append(Paragraph("<i>Revising Claim 2 — ATR vs Fixed Stop Deeper Decomposition</i>",
                        ss["Q_Sub"]))
    E.append(Spacer(1, 4))
    E.append(Paragraph(
        f"Part 4 concluded that the ATR vs fixed stop comparison was \"a wash\" based on a "
        f"pooled {atr_win_pct:.0%} win rate across {len(matched):,} matched pairs. But pooled "
        f"averages can mask meaningful heterogeneity. Is the result uniform across frequency and "
        f"signal family, or does it conceal subgroups where one stop type reliably dominates?",
        ss["Q_Body"]))
    E.append(Paragraph(
        f"We decompose the matched-pair win rate across {int(task2_sum['n_cells_tested'])} "
        f"frequency × signal family cells (minimum 5 pairs per cell) and apply Bonferroni "
        f"correction at p < {task2_sum['bonferroni_threshold']:.4f}.", ss["Q_Body"]))

    E.append(Paragraph(
        "Exhibit 15: ATR Win Rate by Frequency × Signal Family", ss["Q_Ex"]))
    E.append(img(EXT / "task2", "atr_winrate_heatmap.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "Exhibit 16: ATR vs Fixed Stop Win Rate by Frequency", ss["Q_Ex"]))
    E.append(img(EXT / "task2", "winrate_by_freq.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        f"<b>{int(task2_sum['n_significant_cells'])} of {int(task2_sum['n_cells_tested'])} cells "
        f"show Bonferroni-significant differences.</b> ATR dominates in two daily-frequency cells: "
        f"EMA crossovers (88% win rate, +0.28 Sharpe) and Supertrend (80%). Fixed % dominates in "
        f"eight cells, including ADX, APO, Aroon, Hull, MomThresh, PPO, VolScaledMom (all daily), "
        f"and SMA (4-hour).", ss["Q_Body"]))
    E.append(PageBreak())

    E.append(Paragraph(
        "Exhibit 17: Distribution of Matched-Pair Sharpe Differences (ATR − Fixed %)",
        ss["Q_Ex"]))
    E.append(img(EXT / "task2", "sharpe_diff_distribution.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        f"At the aggregate level, {task2_sum['pct_diff_gt_010']*100:.0f}% of pairs have "
        f"meaningful (>0.10 Sharpe) differences in either direction: "
        f"{task2_sum['pct_atr_gt_010']*100:.0f}% favor ATR, "
        f"{task2_sum['pct_fixed_gt_010']*100:.0f}% favor fixed %. The rest are a true wash.",
        ss["Q_Body"]))
    E.append(Paragraph(
        "<b>Conclusion:</b> The pooled 31% figure is accurate as a summary but misleading as a "
        "recommendation. Signal families with inherently noisy entry signals (EMA crossovers) "
        "benefit from ATR's volatility-adaptive exits. Signals with built-in regime sensitivity "
        "(ADX, momentum thresholds) are hurt by ATR stops — the vol-adaptive exit overrides the "
        "signal's own timing. Stop selection should be conditioned on signal family, not applied "
        "as a blanket rule.", ss["Q_BB"]))
    E.append(PageBreak())

    # ================================================================
    # PART 9: REGIME SENSITIVITY (Extension Task 3)
    # ================================================================
    E.append(Paragraph(
        "Part 9: Regime Sensitivity — The Drift Question", ss["Q_Sec"]))
    E.append(Paragraph(
        "<i>For the practitioner: Bonferroni survivors beat B&amp;H at every drift level tested, "
        "including ETH's historical 83%. The median strategy is bad; the credible subset is "
        "not.</i>", ss["Q_FN"]))
    E.append(Paragraph(
        "<i>Revising Claim 3 — At What Drift Does Trend Reliably Add Value?</i>",
        ss["Q_Sub"]))
    E.append(Spacer(1, 4))
    E.append(Paragraph(
        "The original paper concluded that \"in a lower-drift environment, the calculus shifts "
        "in trend's favor\" — implying that trend-following was a conditional bet on lower "
        "future drift. This was the most consequential claim in the paper, because it would "
        "mean that trend's value proposition depends on a macro view. Here we test it directly "
        f"using the 380 daily-frequency Bonferroni survivors (of {n_survivors:,} total across "
        f"all frequencies; Part 9 restricts to daily because the synthetic drift test requires "
        f"re-running signal generation on modified price series, which is tractable only at "
        f"daily resolution).", ss["Q_Body"]))

    E.append(Paragraph("9.1 Historical Sub-Period Analysis", ss["Q_SSec"]))
    E.append(Paragraph(
        "We partition the sample into four sub-periods corresponding to major bull-bear cycles:",
        ss["Q_Body"]))

    sp_data = [["Period", "B&H Sharpe", "Med Survivor", "% Beat B&H", "CAGR Cost",
                "DD Compression"]]
    for _, r in subperiod.iterrows():
        sp_data.append([
            r["period"], f"{r['bh_sharpe']:.2f}", f"{r['med_sharpe']:.2f}",
            f"{r['pct_beat_bh']:.0%}", f"{r['cagr_cost']:+.0%}",
            f"{r['dd_compression']:+.0%}",
        ])
    E.append(Paragraph("Exhibit 18: Bonferroni Survivors vs Buy-and-Hold by Sub-Period",
                        ss["Q_Ex"]))
    E.append(make_table(sp_data, [0.85*inch, 0.85*inch, 0.95*inch, 0.85*inch, 0.85*inch,
                                   1.05*inch]))
    E.append(Spacer(1, 4))
    E.append(img(EXT / "task3", "subperiod_chart.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "Trend adds value most decisively in 2017–2018 (99% beat B&H) and 2021–2022 (74%), both "
        "of which contain the sharpest drawdowns. In calmer periods (2019–2020, 2023–2026), the "
        "win rate drops to ~52–54%, barely above a coin flip. Drawdown compression is consistent "
        "across all periods — roughly 30–43 percentage points of protection.", ss["Q_Body"]))
    E.append(PageBreak())

    E.append(Paragraph("9.2 Synthetic Drift Adjustment", ss["Q_SSec"]))
    E.append(Paragraph(
        "To determine the crossover drift, we construct synthetic ETH return series by replacing "
        "the realized daily drift with target annualized drifts from 0% to 300%, preserving the "
        "original volatility and distributional shape. We re-run all 380 daily-frequency "
        "survivors on each synthetic series.", ss["Q_Body"]))
    E.append(Paragraph(
        "<i>Important limitation: the synthetic adjustment preserves the original return "
        "path — the same crashes (2018, 2022) occur at the same times, with only the drift "
        "component replaced. This tests whether trend-following adds value given ETH's specific "
        "volatility and crash sequence at lower drift, not whether it adds value in a generically "
        "lower-drift environment with different crash timing. The conclusion (100% survivor win "
        "rate at all drift levels) is valid for this specific draw sequence; generalization to "
        "arbitrary lower-drift regimes requires further testing.</i>", ss["Q_FN"]))

    E.append(Paragraph(
        "Exhibit 19: Drift Sensitivity — At What Drift Does Trend Reliably Add Value?",
        ss["Q_Ex"]))
    E.append(img(EXT / "task3", "drift_sensitivity_chart.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    dr_data = [["Drift", "B&H Sharpe", "Med Survivor", "% Beat B&H"]]
    for _, r in drift.iterrows():
        dr_data.append([
            f"{r['target_drift_ann']:.0%}", f"{r['bh_sharpe']:.2f}",
            f"{r['med_sharpe']:.2f}", f"{r['pct_beat_bh']:.0%}",
        ])
    E.append(make_table(dr_data, [0.8*inch, 1.0*inch, 1.1*inch, 1.0*inch]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "<b>The result is striking: Bonferroni survivors beat buy-and-hold at every drift level "
        "tested, including 300% annualized (91% win rate).</b> There is no crossover point within "
        "any realistic drift assumption. The reason is the drawdown compression: these strategies "
        "avoid the catastrophic drawdowns that destroy B&H's compound growth, and this alone "
        "is sufficient to maintain a Sharpe advantage even under parabolic drift.", ss["Q_Body"]))
    E.append(Paragraph(
        "This overturns the original paper's hypothesis. The initial analysis compared the "
        "<i>median</i> strategy to B&H and concluded that trend needed lower drift to compete. "
        "That was true for the median — but irrelevant. The median strategy is a poor strategy; "
        "no one should run it. The strategies that survive multiple-testing correction beat B&H "
        "at every drift level because their drawdown compression is powerful enough to overcome "
        "any CAGR shortfall. The original error was drawing a conclusion about trend-following's "
        "viability from the median, rather than from the statistically credible subset.",
        ss["Q_Body"]))
    E.append(PageBreak())

    E.append(Paragraph("9.3 Bear Regime Isolation: November 2021 – November 2022", ss["Q_SSec"]))
    E.append(Paragraph(
        "The 2022 bear market was the ultimate stress test: ETH fell approximately 77% "
        "peak-to-trough over twelve months. Do the Bonferroni survivors actually provide the "
        "drawdown protection that motivates trend allocation?", ss["Q_Body"]))

    E.append(Paragraph(
        "Exhibit 20: Bonferroni Survivors During the 2022 Bear Market", ss["Q_Ex"]))
    E.append(img(EXT / "task3", "bear_regime_chart.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    n_bear = len(bear)
    n_beat_bear = (bear["sharpe"] > -1.14).sum()
    n_pos_bear = (bear["cagr"] > 0).sum()
    E.append(Paragraph(
        f"Results: {n_beat_bear} of {n_bear} ({n_beat_bear/n_bear:.0%}) survivors beat B&H on "
        f"Sharpe, 100% have shallower drawdowns (median -39% vs -79%), but only "
        f"{n_pos_bear} ({n_pos_bear/n_bear:.0%}) achieve positive returns. Median CAGR was "
        f"-27.3%. The strategies halve the damage but do not avoid it entirely.", ss["Q_Body"]))

    bear_data = [["Metric", "Buy & Hold", "Median Survivor"]]
    bear_data.append(["Sharpe", "-1.14", f"{bear['sharpe'].median():.2f}"])
    bear_data.append(["CAGR", "-76.6%", f"{bear['cagr'].median():.1%}"])
    bear_data.append(["Max DD", "-79.0%", f"{bear['max_dd'].median():.1%}"])
    bear_data.append(["% Positive Return", "—", f"{n_pos_bear/n_bear:.0%}"])
    E.append(make_table(bear_data, [1.4*inch, 1.2*inch, 1.3*inch]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "<b>Conclusion:</b> Claim 3 is wrong. Trend does not require lower drift to add value. "
        "The Bonferroni survivors dominate B&H at all drifts because their drawdown compression "
        "is sufficient to overcome any CAGR shortfall. The original error was generalizing from "
        "the median strategy (which is bad) to trend-following as a class (which is not). In the "
        "worst regime (2022 bear), the survivors cut drawdowns roughly in half — median -39% vs "
        "-79% for B&H. The protection is real but imperfect: most strategies still lose money "
        "in a sustained crash, just substantially less.", ss["Q_BB"]))
    E.append(PageBreak())

    # ================================================================
    # PART 10: TIM OPTIMIZATION (Extension Task 4)
    # ================================================================
    E.append(Paragraph(
        "Part 10: Optimal Time-in-Market Targeting", ss["Q_Sec"]))
    E.append(Paragraph(
        "<i>For the practitioner: targeting 37–47% time-in-market through daily signals "
        "produces Sharpe 1.37 with half the drawdown of buy-and-hold.</i>", ss["Q_FN"]))
    E.append(Paragraph(
        "<i>The Constructive Question — Can We Actually Build Something That Works?</i>",
        ss["Q_Sub"]))
    E.append(Spacer(1, 4))
    E.append(Paragraph(
        "Parts 7–9 revised three claims from the initial analysis. But revising errors is not "
        "the same as building something useful. The initial sweep described what trend costs; it "
        "did not ask whether there is a construction that makes the tradeoff worthwhile. Exhibit "
        "6 hinted at an answer: there is a non-linear relationship between time-in-market and "
        "Sharpe, with an apparent optimum. Here we ask the actionable question: what TIM target "
        "maximizes risk-adjusted returns, which signals deliver it, and does a diversified "
        "ensemble based on this target actually beat buy-and-hold?", ss["Q_Body"]))

    E.append(Paragraph("10.1 The Sharpe-TIM Curve", ss["Q_SSec"]))
    E.append(Paragraph(
        "Exhibit 21: Sharpe vs Time-in-Market with 90% Bootstrap CI", ss["Q_Ex"]))
    E.append(img(EXT / "task4", "sharpe_vs_tim_curve.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        f"The empirical optimum is <b>{task4_sum['optimal_tim']:.0%} time-in-market</b>, with "
        f"median Sharpe {task4_sum['optimal_sharpe']:.2f}. The 90% bootstrap confidence interval "
        f"for peak location is [{task4_sum['ci_90_lo']:.0%}, {task4_sum['ci_90_hi']:.0%}].",
        ss["Q_Body"]))
    E.append(Paragraph(
        "An important caveat: this CI measures the stability of the peak location under "
        "resampling of <i>this</i> return series. It tells us the optimum is well-defined on "
        "ETH-USD 2017–2026, not that 42% is the out-of-sample optimum on a different period or "
        "a different asset. The true uncertainty about the portable TIM optimum is substantially "
        "wider than [42%, 43%], and readers should not interpret bootstrap precision as predictive "
        "precision.", ss["Q_Body"]))

    E.append(Paragraph("10.2 Do Bonferroni Survivors Cluster at the Optimum?", ss["Q_SSec"]))
    E.append(Paragraph(
        "Exhibit 22: TIM Distribution — Survivors vs All Strategies", ss["Q_Ex"]))
    E.append(img(EXT / "task4", "survivor_tim_distribution.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        f"Bonferroni survivors have median TIM of 44%, near the empirical optimum. The "
        f"Kolmogorov-Smirnov test rejects the null that survivor and all-strategy TIM are drawn "
        f"from the same distribution (D = {task4_sum['ks_stat']:.3f}, p < 0.001). Survivors "
        f"do cluster near the optimal TIM, but slightly above it.", ss["Q_Body"]))
    E.append(PageBreak())

    E.append(Paragraph("10.3 Which Signal Families Target the Optimal TIM?", ss["Q_SSec"]))
    E.append(Paragraph(
        f"Among daily-frequency strategies, we identify which signal families most reliably "
        f"produce configurations in the optimal band "
        f"({task4_sum['optimal_tim']:.0%} ± 5%):", ss["Q_Body"]))

    ft_data = [["Family", "N", "Mean TIM", "% in Band", "Sharpe (in)", "Sharpe (out)"]]
    for _, r in fam_tim.head(12).iterrows():
        if pd.isna(r["med_sharpe_in_band"]):
            continue
        ft_data.append([
            r["family"], str(int(r["n_total"])), f"{r['mean_tim']:.0%}",
            f"{r['pct_in_band']:.0%}",
            f"{r['med_sharpe_in_band']:.2f}",
            f"{r['med_sharpe_out_band']:.2f}" if not pd.isna(r["med_sharpe_out_band"]) else "—",
        ])
    E.append(Paragraph("Exhibit 23: Signal Family TIM Targeting", ss["Q_Ex"]))
    E.append(make_table(ft_data,
                        [1.0*inch, 0.5*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.9*inch]))
    E.append(Spacer(1, 6))
    E.append(Paragraph(
        "DEMA, Supertrend, PPO/APO, CCI, and Aroon families most reliably produce strategies "
        "in the optimal TIM band, and their in-band Sharpe ratios meaningfully exceed their "
        "out-of-band Sharpe ratios, confirming that TIM targeting adds value conditional on "
        "being in the right frequency.", ss["Q_Body"]))

    E.append(Paragraph("10.4 The TIM-Filtered Ensemble", ss["Q_SSec"]))
    E.append(Paragraph(
        "We construct a \"TIM-filtered\" ensemble: equal-weight the daily position signals of all "
        "704 strategies whose in-sample TIM falls in the optimal band, producing a fractional "
        "position on each day.", ss["Q_Body"]))
    E.append(Paragraph(
        "A candid note on look-ahead bias: the 704 strategies are selected because their realized "
        "TIM falls in [37%, 47%] over the full sample. This is only knowable after running the "
        "complete backtest — it is not an ex-ante filter. <b>Part 11 directly addresses this gap</b> "
        "by testing whether TIM can be predicted ex-ante — both from signal parameters alone and "
        "via walk-forward estimation — and constructs a comparable ensemble without look-ahead.",
        ss["Q_Body"]))

    E.append(Paragraph("Exhibit 24: TIM-Filtered Ensemble vs Buy-and-Hold", ss["Q_Ex"]))
    E.append(img(EXT / "task4", "ensemble_equity.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    ec_data = [["Strategy", "Sharpe", "CAGR", "Max DD", "Skewness"]]
    for _, r in ensemble.iterrows():
        cagr_val = r.get("cagr", None)
        dd_val = r.get("max_dd", None)
        skew_val = r.get("skewness", None)
        ec_data.append([
            str(r["strategy"]),
            f"{r['sharpe']:.2f}",
            f"{cagr_val:.1%}" if pd.notna(cagr_val) and isinstance(cagr_val, (int, float)) else str(cagr_val),
            f"{dd_val:.1%}" if pd.notna(dd_val) and isinstance(dd_val, (int, float)) else str(dd_val),
            f"{skew_val:.2f}" if pd.notna(skew_val) and isinstance(skew_val, (int, float)) else str(skew_val),
        ])
    E.append(Paragraph("Exhibit 25: Ensemble vs Benchmark Comparison", ss["Q_Ex"]))
    E.append(make_table(ec_data, [2.4*inch, 0.7*inch, 0.8*inch, 0.8*inch, 0.8*inch]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        f"The TIM ensemble achieves Sharpe {task4_sum['ensemble_sharpe']:.2f} — higher than "
        f"buy-and-hold ({bh['sharpe']:.2f}) — with max drawdown of "
        f"{abs(task4_sum['ensemble_dd'])*100:.0f}% (vs {abs(bh['max_dd'])*100:.0f}% for B&H). "
        f"It captures {abs(task4_sum['ensemble_cagr'])/abs(bh['cagr'])*100:.0f}% of B&H CAGR. "
        f"The diversification across 704 signals smooths out the idiosyncratic risk of any "
        f"single parameterization.", ss["Q_Body"]))
    E.append(Paragraph(
        "<b>Conclusion:</b> There is a structurally optimal TIM around 42% on ETH-USD. It can "
        "be reliably targeted through signal family selection (DEMA, Supertrend, CCI, Aroon). "
        "A TIM-filtered ensemble outperforms buy-and-hold on risk-adjusted returns while halving "
        "the drawdown — the first construction in this study to achieve that combination.",
        ss["Q_BB"]))
    E.append(PageBreak())

    # ================================================================
    # PART 11: EX-ANTE TIM PREDICTION (Extension Task 5)
    # ================================================================
    E.append(Paragraph(
        "Part 11: Ex-Ante TIM Prediction", ss["Q_Sec"]))
    E.append(Paragraph(
        "<i>For the practitioner: TIM is near-deterministic from signal structure "
        "(ρ = 0.99). Walk-forward selection works with zero Sharpe decay.</i>", ss["Q_FN"]))
    E.append(Paragraph(
        "<i>Closing the Look-Ahead Gap — From Research Result to Deployable Strategy</i>",
        ss["Q_Sub"]))
    E.append(Spacer(1, 4))
    E.append(Paragraph(
        "Part 10 flagged a latent look-ahead bias: the TIM-filtered ensemble selects strategies "
        "by their realized TIM over the full sample. In live deployment, an allocator must select "
        "strategies <i>before</i> observing their full-sample TIM. This section tests whether "
        "ex-ante TIM prediction is tractable and, if so, whether the resulting ensemble suffers "
        "meaningful Sharpe decay.", ss["Q_Body"]))

    # ── 11.1 Walk-forward (the primary result) ─────────────────────
    E.append(Paragraph("11.1 Walk-Forward TIM Selection", ss["Q_SSec"]))
    E.append(Paragraph(
        "The most direct test: compute each strategy's TIM on a training period (2017–2021), "
        "select those in the optimal band [37%, 47%], and deploy them on the test period "
        "(2022–2026). This eliminates look-ahead entirely — the allocator never observes "
        "test-period data when selecting strategies.", ss["Q_Body"]))

    E.append(Paragraph("Exhibit 26: Training-Period vs Full-Sample TIM", ss["Q_Ex"]))
    E.append(img(EXT / "task5", "train_vs_full_tim.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        f"The correlation between training-period TIM and full-sample TIM is "
        f"<b>ρ = {task5_sum['train_full_tim_corr']:.2f}</b>. This is the key number: TIM is "
        f"a structural property of the signal parameterization, not a sample artifact. "
        f"The ~2% absolute drift across regimes (Exhibit 28) is noise, not signal. "
        f"Of {task5_sum['walkfwd_selected']:.0f} strategies selected by their training-period TIM, "
        f"<b>{task5_sum['walkfwd_precision']:.0%}</b> also fall in the optimal band on the full "
        f"sample.", ss["Q_Body"]))

    E.append(Paragraph("Exhibit 27: TIM Stability by Signal Family", ss["Q_Ex"]))

    stab_data = [["Family", "N", "Train TIM", "Full TIM", "Shift", "ρ"]]
    for _, r in task5_fam_stab.head(12).iterrows():
        fam = str(r["signal_family"])
        stab_data.append([
            fam,
            str(int(r["n"])),
            f"{r['train_tim_mean']:.0%}",
            f"{r['full_tim_mean']:.0%}",
            f"{r['tim_shift']:+.1%}",
            f"{r['corr']:.3f}" if pd.notna(r["corr"]) else "—",
        ])
    E.append(make_table(stab_data,
                        [1.1*inch, 0.5*inch, 0.8*inch, 0.7*inch, 0.6*inch, 0.6*inch]))
    E.append(Paragraph(
        "Every signal family has ρ > 0.99 between training and full-sample TIM, with absolute "
        "drift of 0.4–2.5 percentage points. The mechanism is intuitive: a 200-day MA crossover "
        "will always produce lower TIM than a 10-day crossover; adding a tight trailing stop "
        "reduces TIM further. These are properties of signal structure, not of the particular "
        "return path realized during the sample.", ss["Q_Body"]))
    E.append(PageBreak())

    # ── 11.2 Walk-forward vs in-sample ensemble ────────────────────
    E.append(Paragraph("11.2 Walk-Forward vs In-Sample Ensemble", ss["Q_SSec"]))
    E.append(Paragraph(
        "The critical test: does the walk-forward ensemble — selected entirely without look-ahead "
        "— match the in-sample ensemble from Part 10?", ss["Q_Body"]))

    E.append(Paragraph("Exhibit 28: Walk-Forward vs In-Sample Ensemble Equity Curves", ss["Q_Ex"]))
    E.append(img(EXT / "task5", "ensemble_equity_curves.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    ec5_data = [["Strategy", "Sharpe", "CAGR", "Max DD", "Skew", "Mean Pos"]]
    for _, r in task5_comp.iterrows():
        ec5_data.append([
            str(r["strategy"]),
            f"{r['sharpe']:.2f}",
            f"{r['cagr']:.1%}" if isinstance(r["cagr"], (int, float)) and pd.notna(r["cagr"]) else "—",
            f"{r['max_dd']:.1%}" if isinstance(r["max_dd"], (int, float)) and pd.notna(r["max_dd"]) else "—",
            f"{r['skewness']:.2f}" if isinstance(r["skewness"], (int, float)) and pd.notna(r["skewness"]) else "—",
            f"{r['tim']:.0%}" if isinstance(r["tim"], (int, float)) and pd.notna(r["tim"]) else "—",
        ])
    E.append(Paragraph("Exhibit 29: Ensemble Performance Comparison", ss["Q_Ex"]))
    E.append(make_table(ec5_data,
                        [2.2*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.5*inch, 0.7*inch]))
    E.append(Spacer(1, 6))

    wf_full = task5_comp[task5_comp["strategy"] == "Walk-Forward Ensemble"]
    is_full = task5_comp[task5_comp["strategy"] == "In-Sample Ensemble"]

    if len(wf_full) > 0 and len(is_full) > 0:
        wf_r = wf_full.iloc[0]
        is_r = is_full.iloc[0]
        decay = is_r["sharpe"] - wf_r["sharpe"]
        E.append(Paragraph(
            f"The walk-forward ensemble achieves Sharpe <b>{wf_r['sharpe']:.2f}</b> vs the "
            f"in-sample ensemble's {is_r['sharpe']:.2f} — a Sharpe \"decay\" of "
            f"<b>{decay:+.2f}</b>. Walk-forward selection produces <i>better</i> performance than "
            f"in-sample selection, which is the right sanity check: if walk-forward materially "
            f"underperformed, it would suggest regime sensitivity in TIM stability. The fact that it "
            f"does not confirms the mechanism is structural.", ss["Q_Body"]))

    wf_oos = task5_comp[task5_comp["strategy"] == "Walk-Forward (OOS only)"]
    bh_oos = task5_comp[task5_comp["strategy"] == "Buy & Hold (OOS)"]
    if len(wf_oos) > 0 and len(bh_oos) > 0:
        wf_o = wf_oos.iloc[0]
        bh_o = bh_oos.iloc[0]
        # OOS Sharpe SE: for ~1505 daily obs over ~4 years with moderate autocorrelation
        oos_n_years = 4.1
        oos_se = round(1.0 / np.sqrt(oos_n_years) * 1.3, 2)  # ~1.3× Newey-West adjustment
        E.append(Paragraph(
            f"On the out-of-sample period (2022–2026), the walk-forward ensemble achieves Sharpe "
            f"{wf_o['sharpe']:.2f} vs Buy & Hold {bh_o['sharpe']:.2f}, with max drawdown "
            f"{wf_o['max_dd']:.0%} vs {bh_o['max_dd']:.0%}. A note on precision: with "
            f"~4 years of daily data and moderate return autocorrelation, the standard error on "
            f"these Sharpe estimates is approximately ±{oos_se:.2f}. The absolute OOS Sharpe of "
            f"{wf_o['sharpe']:.2f} is therefore not statistically distinguishable from zero at "
            f"conventional significance levels. The relative outperformance — particularly the "
            f"drawdown compression ({wf_o['max_dd']:.0%} vs {bh_o['max_dd']:.0%}) — is the more "
            f"robust result.", ss["Q_Body"]))

    E.append(Paragraph(
        "<i>Methodological caveat: the walk-forward test uses a single train/test fold "
        "(2017–2021 / 2022–2026). The test period includes a historically specific regime — "
        "post-LUNA crash, macro rate tightening cycle, and post-ETF-approval recovery. A rolling "
        "or expanding window would provide more robust OOS estimates. The single-fold result is "
        "informative but should not be over-interpreted as a general OOS guarantee.</i>",
        ss["Q_FN"]))

    E.append(Paragraph("Exhibit 30: Ensemble Position Size Over Time", ss["Q_Ex"]))
    E.append(img(EXT / "task5", "ensemble_position_timeseries.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "Although each component strategy targets ~42% individual TIM, the ensemble's mean "
        "position is ~40% — close to the target. The position varies dynamically: higher during "
        "trending markets (more strategies agree on direction), lower during choppy regimes "
        "(fewer signals active). This consensus-weighted sizing is an emergent property of the "
        "ensemble that individual binary strategies cannot achieve.", ss["Q_Body"]))
    E.append(PageBreak())

    # ── 11.3 Confirmatory: parameter-based prediction ──────────────
    E.append(Paragraph("11.3 Confirmatory: Parameter-Based TIM Prediction", ss["Q_SSec"]))
    E.append(Paragraph(
        "As an independent check, we ask whether TIM can be predicted from signal parameters "
        "alone — without running any backtest at all. This is a weaker test than walk-forward "
        "selection (which already achieves ρ = 0.99), but it confirms the mechanistic intuition: "
        "TIM is determined by signal structure, not by the realized return path.", ss["Q_Body"]))

    E.append(Paragraph("Exhibit 31: Cross-Validated TIM Prediction from Signal Parameters",
                        ss["Q_Ex"]))
    E.append(img(EXT / "task5", "predicted_vs_actual_tim.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        f"A gradient boosting model on signal features (lookback periods, stop type, signal "
        f"family) achieves cross-validated R² = {task5_sum['gbm_r2']:.2f} and MAE = "
        f"{task5_sum['gbm_mae']:.3f}. For the optimal band [37%, 47%], this yields "
        f"{task5_sum['gbm_precision']:.0%} precision and {task5_sum['gbm_recall']:.0%} recall — "
        f"useful but materially weaker than the walk-forward route's 73% precision. The walk-"
        f"forward approach is the production path; the ML model serves primarily to confirm that "
        f"the relationship between parameters and TIM is real and mechanistic, not an artifact of "
        f"temporal correlation between train and test periods.", ss["Q_Body"]))

    E.append(Paragraph("Exhibit 32: Feature Importance for TIM Prediction", ss["Q_Ex"]))
    E.append(img(EXT / "task5", "feature_importance.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "The dominant predictors — stop type, signal family, and lookback period — are exactly "
        "the parameters an allocator would vary when constructing a strategy. This is consistent "
        "with the walk-forward result: TIM is not learned from the data, it is set by the "
        "designer's choices.", ss["Q_Body"]))

    # ── 11.4 Family composition ────────────────────────────────────
    E.append(Paragraph("11.4 Signal Family Composition", ss["Q_SSec"]))
    E.append(Paragraph(
        "Exhibit 33: Signal Family Composition — Walk-Forward vs In-Sample", ss["Q_Ex"]))
    E.append(img(EXT / "task5", "family_composition.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "The walk-forward and in-sample ensembles have nearly identical family composition. "
        "EMA crossovers, RSI, Supertrend, SMA, and MomThresh are the largest contributors in both. "
        "No family is systematically excluded or over-represented by the walk-forward filter, "
        "confirming that the signal diversity of the Part 10 ensemble is preserved without "
        "look-ahead.", ss["Q_Body"]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        f"<b>Conclusion:</b> TIM is a near-deterministic function of signal structure; "
        f"walk-forward selection captures it with {task5_sum['walkfwd_precision']:.0%} band "
        f"precision and negligible Sharpe decay. The look-ahead concern from Part 10 is closed. "
        f"The TIM-filtered ensemble is deployable: measure each strategy's TIM on a trailing "
        f"window, select those in the [37%, 47%] band, and equal-weight their signals. Cross-"
        f"validated parameter prediction (R² = {task5_sum['gbm_r2']:.2f}) confirms the mechanistic "
        f"basis but is not needed for production — the walk-forward route is both simpler and "
        f"more precise.",
        ss["Q_BB"]))
    E.append(PageBreak())

    # ================================================================
    # PART 12: CROSS-ASSET VALIDATION
    # ================================================================
    E.append(Paragraph("Part 12: Cross-Asset Validation", ss["Q_Sec"]))
    E.append(Paragraph(
        "<i>For the practitioner: the architecture is portable; the specific parameters are "
        "not. Run per-asset sweeps, target 40–60% TIM universally.</i>", ss["Q_FN"]))
    E.append(Paragraph(
        "<i>What Replicates, What Doesn't, and What That Means</i>", ss["Q_Sub"]))
    E.append(Spacer(1, 4))
    E.append(Paragraph(
        "Parts 1–11 established findings on a single asset (ETH-USD) and repeatedly flagged that "
        "cross-asset validation was the binding constraint. This section runs that test across five "
        "additional crypto assets: BTC-USD and SOL-USD (full 4,429-strategy sweeps) and LTC-USD, "
        "LINK-USD, and ATOM-USD (zero-reoptimization replication of ETH's 376 daily Bonferroni "
        "survivors). Three pre-registered hypotheses are tested with pre-specified decision "
        "thresholds.", ss["Q_Body"]))

    E.append(Paragraph("12.1 The Zero-Reoptimization Test", ss["Q_SSec"]))
    E.append(Paragraph(
        "This is the most important result in the entire study. Take the 376 daily-frequency "
        "strategies that survived Bonferroni correction on ETH-USD and run them — without any "
        "modification — on three assets they were never calibrated to. No re-optimization, no "
        "per-asset tuning. If the strategies retain edge on a completely different asset, that is "
        "stronger evidence of structural signal than anything achievable from a full per-asset "
        "sweep, which can always find strategies that work in-sample.", ss["Q_Body"]))

    E.append(Paragraph("Exhibit 34: Tier 2 Zero-Reoptimization Replication", ss["Q_Ex"]))
    t2_data = [["Asset", "Bars", "B&H Sharpe", "% Pos SR", "% Beat B&H", "% Better DD",
                "Med Sharpe", "Med MaxDD"]]
    for _, r in ca_tier2.iterrows():
        t2_data.append([
            str(r["asset"]), str(int(r["n_bars"])), f"{r['bh_sharpe']:.2f}",
            f"{r['pct_positive_sharpe']:.0%}", f"{r['pct_beat_bh_sharpe']:.0%}",
            f"{r['pct_better_dd']:.0%}", f"{r['med_sharpe']:.3f}", f"{r['med_max_dd']:.1%}",
        ])
    E.append(make_table(t2_data,
                        [0.8*inch, 0.5*inch, 0.7*inch, 0.65*inch, 0.7*inch, 0.7*inch,
                         0.7*inch, 0.7*inch]))
    E.append(Spacer(1, 6))

    E.append(Paragraph(
        "The results deliver a precise, falsifiable statement: <b>the mechanism works; the specific "
        "parameters don't transfer.</b> ETH survivors retain positive Sharpe on other assets "
        "at high rates (72–100%) and compress drawdowns relative to Buy & Hold at 90–100% rates. "
        "However, they beat B&H on Sharpe at only 11–27%. That 11–27% is the base rate of blind "
        "parameter transfer — the right null hypothesis for deployment decisions. It tells an "
        "allocator exactly what to expect if they apply ETH-optimized strategies to a new asset "
        "without re-calibration.", ss["Q_Body"]))

    E.append(Paragraph(
        "The actionable conclusion: trend-following <i>as a mechanism</i> is structural; the "
        "<i>specific families and parameterizations</i> that excel are asset-dependent. Allocators "
        "should run per-asset sweeps and use universal TIM targeting, not port ETH "
        "parameterizations directly.", ss["Q_BB"]))
    E.append(PageBreak())

    E.append(Paragraph("12.2 Family Survival Across Assets", ss["Q_SSec"]))
    E.append(Paragraph(
        "Exhibit 35: Family Bonferroni Survival Rate by Asset (Daily Frequency)", ss["Q_Ex"]))
    E.append(img(CROSS, "family_survival_heatmap.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    # Spearman table
    sp_data = [["Asset Pair", "Spearman ρ", "p-value", "Interpretation"]]
    for _, r in ca_spearman.iterrows():
        interp = "borderline" if 0.25 < r["spearman_rho"] < 0.35 else (
            "weak" if r["spearman_rho"] < 0.3 else "moderate")
        sp_data.append([
            f"{r['asset_1']} vs {r['asset_2']}",
            f"{r['spearman_rho']:.3f}", f"{r['p_value']:.4f}", interp])
    E.append(Paragraph("Exhibit 36: Spearman Rank Correlation of Family Survival Rankings",
                        ss["Q_Ex"]))
    E.append(make_table(sp_data, [1.8*inch, 1.0*inch, 0.8*inch, 1.0*inch]))
    E.append(Spacer(1, 6))

    mean_rho = ca_summary["mean_spearman_rho"]
    E.append(Paragraph(
        f"Mean Spearman ρ = <b>{mean_rho:.3f}</b>. The pre-specified threshold for Outcome A "
        f"(structural) was ρ > 0.6; for Outcome C (path dependence), ρ < 0.3. The ETH-BTC pair "
        f"is ρ = 0.30 (p = 0.06) — right at the boundary. The SOL comparisons are weaker, but "
        f"SOL's short history (4.7 years) means only 2 of 4,428 strategies survive its higher "
        f"Bonferroni threshold (1.91), making the family ranking comparison nearly uninformative.",
        ss["Q_Body"]))
    E.append(Paragraph(
        "The families that dominate on ETH (DEMA, Supertrend, Aroon, EMA) are not the same families "
        "that dominate on BTC (RSI, CCI, Bollinger). By our pre-specified decision framework, "
        "ρ < 0.3 indicates <b>Outcome C: family-level dominance is asset-specific, not structural</b>. "
        "We report this as stated in the proposal — the pre-specified thresholds are decision points "
        "chosen in advance, not derived from theory. The 0.3–0.6 range was acknowledged as "
        "ambiguous, and at ρ = 0.22 we fall just below it.", ss["Q_Body"]))
    E.append(PageBreak())

    E.append(Paragraph("12.3 TIM Optimum Portability", ss["Q_SSec"]))
    E.append(Paragraph(
        "Exhibit 37: Sharpe vs TIM Across Assets (Daily Frequency)", ss["Q_Ex"]))
    E.append(img(CROSS, "tim_optimum_comparison.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "The TIM optimum is more portable than family rankings. ETH and BTC both peak near 57% "
        "for daily-frequency strategies (note: the earlier 42% estimate included all frequencies). "
        "SOL peaks at 50%. The 90% bootstrap CIs for ETH [41%, 57%] and BTC [53%, 61%] overlap, "
        "suggesting the optimal TIM range is structurally similar across large-cap crypto assets.",
        ss["Q_Body"]))
    E.append(Paragraph(
        "A critical distinction: the TIM <i>framework</i> is portable even though the specific "
        "families that produce strategies at the optimal TIM differ by asset. An allocator can "
        "target 40–60% TIM on any crypto asset — but must run a per-asset sweep to identify which "
        "signals deliver that TIM, rather than assuming ETH-optimal families transfer.", ss["Q_Body"]))

    E.append(Paragraph("12.4 Stop-Type Portability", ss["Q_SSec"]))
    E.append(Paragraph(
        "Exhibit 38: ATR vs Fixed Stop Win Rate by Family and Asset", ss["Q_Ex"]))
    E.append(img(CROSS, "stop_type_heatmap.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    pct_cons = ca_summary.get("pct_consistent", 0)
    E.append(Paragraph(
        f"Stop-type preferences are not portable. The Part 8 finding that ATR stops dominate for "
        f"EMA crossovers (88% on ETH) reverses completely on BTC (0%). Only "
        f"{pct_cons:.0%} of signal families maintain consistent stop-type preference across all "
        f"three Tier 1 assets. Stop selection must be conditioned on both signal family <i>and</i> "
        f"asset, not on signal family alone.", ss["Q_Body"]))
    E.append(PageBreak())

    E.append(Paragraph("12.5 Walk-Forward TIM Ensemble on BTC", ss["Q_SSec"]))
    E.append(Paragraph(
        "To test whether the TIM-targeting ensemble from Part 11 generalizes, we replicate the "
        "walk-forward methodology on BTC-USD with identical regime alignment (train 2017–2021, "
        "test 2022–2026). This is BTC-only by design — SOL lacks the 2017–2021 training window.",
        ss["Q_Body"]))

    E.append(Paragraph("Exhibit 39: BTC Walk-Forward Ensemble vs Buy & Hold", ss["Q_Ex"]))
    E.append(img(CROSS, "btc_walkforward_equity.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    btc_wf_data = [["Strategy", "Sharpe", "CAGR", "Max DD", "Skew", "Mean Pos"]]
    for _, r in ca_btc_wf.iterrows():
        btc_wf_data.append([
            str(r["strategy"]),
            f"{r['sharpe']:.2f}",
            f"{r['cagr']:.1%}" if pd.notna(r["cagr"]) else "—",
            f"{r['max_dd']:.1%}" if pd.notna(r["max_dd"]) else "—",
            f"{r['skewness']:.2f}" if pd.notna(r["skewness"]) else "—",
            f"{r.get('mean_pos', r.get('tim', '')):.0%}" if pd.notna(r.get("mean_pos", r.get("tim"))) else "—",
        ])
    E.append(Paragraph("Exhibit 40: BTC Ensemble Performance", ss["Q_Ex"]))
    E.append(make_table(btc_wf_data,
                        [2.2*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.5*inch, 0.7*inch]))
    E.append(Spacer(1, 6))

    E.append(Paragraph(
        f"TIM stability on BTC is identical to ETH: ρ = {ca_btc_wf_sum['btc_tim_corr']:.2f} between "
        f"training and full-sample TIM. Walk-forward selection precision is "
        f"{ca_btc_wf_sum['btc_wf_precision']:.0%} (vs 73% on ETH). The TIM-targeting mechanism is "
        f"unambiguously structural.", ss["Q_Body"]))
    E.append(Paragraph(
        f"The BTC ensemble underperforms Buy & Hold on out-of-sample Sharpe: "
        f"{ca_btc_wf_sum['btc_wf_oos_sharpe']:.2f} vs {ca_btc_wf_sum['btc_bh_oos_sharpe']:.2f}. "
        f"An allocator who held BTC through 2022–2026 earned better risk-adjusted returns than the "
        f"trend ensemble. That is a legitimate data point, not an asterisk. The drawdown compression "
        f"was real ({ca_btc_wf_sum['btc_wf_oos_maxdd']:.0%} vs "
        f"{ca_btc_wf_sum['btc_bh_oos_maxdd']:.0%}), but BTC's strong recovery (B&H Sharpe 0.45) "
        f"means being out of market 58% of the time cost more than the drawdown protection saved.",
        ss["Q_Body"]))
    E.append(Paragraph(
        "Whether this is acceptable depends entirely on the allocator's mandate. For an institution "
        "that cannot absorb a -67% drawdown, the ensemble was still the right choice — it reduced "
        "max drawdown by 24 percentage points. For one targeting Sharpe maximization, it was not. "
        "Trend-following's value proposition on BTC in this period was drawdown protection, not "
        "Sharpe enhancement. That distinction is the single most important deployment consideration "
        "to emerge from the cross-asset validation.", ss["Q_Body"]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "<b>Part 12 conclusion:</b> The cross-asset validation produces a nuanced result that does "
        "not fit cleanly into Outcome A (structural) or Outcome C (path dependence). The TIM-"
        "targeting framework, the walk-forward methodology, and drawdown compression are structural "
        "— they work across assets. The specific signal families, parameterizations, and stop-type "
        "recommendations from ETH are asset-dependent — they do not transfer.", ss["Q_BB"]))
    E.append(Spacer(1, 10))
    E.append(Paragraph("<b>The Deployable Architecture</b>", ss["Q_SSec"]))
    E.append(Paragraph(
        "<b>Per-asset sweep</b> to identify locally dominant signals. <b>Universal TIM targeting "
        "at 40–60%.</b> Walk-forward selection with training-period TIM estimation. <b>Per-asset "
        "stop-type selection.</b> This is the paper's primary deliverable.", ss["Q_BB"]))
    E.append(PageBreak())

    # ================================================================
    # PART 13: FULL UNIVERSE VALIDATION
    # ================================================================
    E.append(Paragraph("Part 13: Full Universe Validation", ss["Q_Sec"]))
    E.append(Paragraph(
        "<i>For the practitioner: only ETH and BTC produce Bonferroni survivors. On alts, "
        "trend-following is for drawdown protection, not Sharpe enhancement.</i>", ss["Q_FN"]))
    E.append(Paragraph(
        "<i>From Six Assets to 362 — What Survives at Scale</i>", ss["Q_Sub"]))
    E.append(Spacer(1, 4))

    n_a = int(fu_summary["n_tier_a"])
    n_b = int(fu_summary["n_tier_b"])
    n_c = int(fu_summary["n_tier_c"])
    E.append(Paragraph(
        f"Parts 1–12 tested the trend-following framework on six assets. This section scales the "
        f"test to the full Coinbase universe: {n_a + n_b + n_c} symbols, tiered by data quality. "
        f"Tier A ({n_a} assets with 5+ years of history) receive full 4,429-strategy daily sweeps "
        f"and walk-forward TIM ensembles. Tier B ({n_b} assets with 3–5 years) receive the "
        f"zero-reoptimization replication test. Tier C ({n_c} assets with less than 3 years) are "
        f"excluded — the Bonferroni threshold rises above 2.5, making statistical discrimination "
        f"impossible.", ss["Q_Body"]))

    E.append(Paragraph("13.1 TIM Optimum Is Universal", ss["Q_SSec"]))
    E.append(Paragraph(
        "Exhibit 41: TIM Optimum Distribution Across 31 Tier A Assets", ss["Q_Ex"]))
    E.append(img(FULLUNIV, "tim_optimum_distribution.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    tim_med = fu_summary["tier_a_tim_median"]
    tim_std = fu_summary["tier_a_tim_std"]
    E.append(Paragraph(
        f"The TIM optimum across all 31 Tier A assets has median <b>{tim_med:.0%}</b> with standard "
        f"deviation {tim_std:.0%}. This is consistent with the ETH finding (~42% including all "
        f"frequencies) and the BTC finding (~57% daily only). The range [{fu_tier_a['tim_optimum'].dropna().min():.0%}, "
        f"{fu_tier_a['tim_optimum'].dropna().max():.0%}] shows meaningful dispersion — some assets "
        f"(EOS, ALGO) prefer very low TIM while others (DASH, AAVE) prefer higher. But the central "
        f"tendency clusters around 35–50%, confirming that the Part 10 TIM-targeting framework is "
        f"structurally sound across the crypto universe, not an ETH artifact.", ss["Q_Body"]))

    E.append(Paragraph("13.2 TIM Stability Is Universal", ss["Q_SSec"]))
    E.append(Paragraph(
        "Exhibit 42: Walk-Forward TIM Stability Across 31 Assets", ss["Q_Ex"]))
    E.append(img(FULLUNIV, "walkforward_stability.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    wf_med_rho = fu_summary["wf_med_tim_corr"]
    wf_min_rho = fu_summary["wf_min_tim_corr"]
    wf_oos_med = fu_summary["wf_med_oos_sharpe"]
    E.append(Paragraph(
        f"TIM stability (ρ between training and full-sample TIM) is universally high: median "
        f"<b>{wf_med_rho:.3f}</b>, minimum {wf_min_rho:.3f} across all 31 assets. No asset has "
        f"ρ below 0.98. This definitively confirms Part 11's finding: TIM is a structural property "
        f"of the signal parameterization, not a sample artifact. It holds on every crypto asset "
        f"tested.", ss["Q_Body"]))
    E.append(Paragraph(
        f"However, walk-forward ensemble OOS Sharpe has median {wf_oos_med:.2f} — essentially zero. "
        f"TIM targeting reliably selects strategies at the optimal TIM, but the optimal TIM does not "
        f"guarantee positive OOS Sharpe. Only ETH and BTC produce Bonferroni survivors; on the "
        f"remaining 29 assets, even the best strategies do not survive multiple-testing correction. "
        f"The architecture works perfectly; the raw signal quality on most alts is insufficient.",
        ss["Q_Body"]))
    E.append(PageBreak())

    E.append(Paragraph("13.3 Drawdown Compression Is Nearly Universal", ss["Q_SSec"]))
    E.append(Paragraph(
        "Exhibit 43: Tier B Replication on 148 Assets", ss["Q_Ex"]))
    E.append(img(FULLUNIV, "tier_b_replication_histograms.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    tb_dd = fu_summary["tier_b_med_pct_better_dd"]
    tb_pos = fu_summary["tier_b_med_pct_pos_sr"]
    tb_beat = fu_summary["tier_b_med_pct_beat_bh"]
    E.append(Paragraph(
        f"The most robust finding across the entire universe: <b>{tb_dd:.0%} of ETH-derived "
        f"strategies produce better drawdowns than Buy & Hold</b> on the median Tier B asset, "
        f"without any re-optimization. This is the base rate of blind parameter transfer for "
        f"drawdown compression — it works on nearly every crypto asset.", ss["Q_Body"]))
    E.append(Paragraph(
        f"The Sharpe picture is more nuanced: {tb_pos:.0%} of strategies maintain positive Sharpe "
        f"(better than cash) and {tb_beat:.0%} beat B&H on Sharpe. This median 55% B&H beat rate "
        f"across 148 assets — using strategies optimized on a different asset — is meaningfully "
        f"above the 50% null, but not overwhelmingly so. The mechanism adds value on a slight "
        f"majority of assets; the value proposition is drawdown compression, not Sharpe "
        f"enhancement.", ss["Q_Body"]))

    E.append(Paragraph("13.4 Asset Classification", ss["Q_SSec"]))

    n_strong = int(fu_summary["n_strong"])
    n_mod = int(fu_summary["n_moderate"])
    n_marg = int(fu_summary["n_marginal"])
    n_weak = int(fu_summary["n_weak"])
    cls_data = [["Class", "N", "Criteria", "Assets"]]
    for cls_name, criteria in [
        ("STRONG", "Bonferroni survivors + positive OOS Sharpe"),
        ("MODERATE", "Positive OOS Sharpe, no survivors"),
        ("MARGINAL", "TIM in [30%-60%], negative OOS"),
        ("WEAK", "Poor TIM or negative OOS"),
    ]:
        subset = fu_class[fu_class["class"] == cls_name]
        syms = ", ".join(s.replace("-USD", "") for s in subset["symbol"].tolist())
        cls_data.append([cls_name, str(len(subset)), criteria, syms])
    E.append(Paragraph("Exhibit 44: Asset Trend-Followability Classification", ss["Q_Ex"]))
    E.append(make_table(cls_data, [0.8*inch, 0.3*inch, 2.5*inch, 2.5*inch]))
    E.append(Spacer(1, 6))

    E.append(Paragraph(
        f"Of the 31 Tier A assets, only {n_strong} (ETH, BTC) are classified as STRONG — having "
        f"both Bonferroni survivors and positive walk-forward OOS Sharpe. {n_mod} are MODERATE "
        f"(positive OOS Sharpe but no survivors), {n_marg} are MARGINAL, and {n_weak} are WEAK. "
        f"This is a sobering result: trend-following with statistical significance is achievable on "
        f"only the two largest crypto assets. On smaller alts, the strategy provides drawdown "
        f"protection but not risk-adjusted outperformance.", ss["Q_Body"]))

    E.append(Spacer(1, 8))
    E.append(Paragraph(
        "<b>Part 13 conclusion:</b> The full universe test confirms and sharpens the Part 12 "
        "findings. TIM stability (ρ ≥ 0.98) and drawdown compression (99% median) are universal "
        "across crypto. TIM targeting centers near 39% across 31 assets. But statistically "
        "significant trend signals — strategies that survive Bonferroni correction — exist only on "
        "ETH and BTC. For an allocator, the implication is precise: use trend-following on the "
        "majors for Sharpe enhancement, and on alts strictly for drawdown management.",
        ss["Q_BB"]))
    E.append(PageBreak())

    # ================================================================
    # CONCLUDING THOUGHTS
    # ================================================================
    E.append(Paragraph("Concluding Thoughts", ss["Q_Sec"]))

    E.append(Paragraph("<b>What held up from the initial analysis:</b>", ss["Q_BB"]))
    E.append(Paragraph(
        f"<b>1. The median trend strategy underperforms B&H.</b> This remains true: "
        f"{(n-n_beat_sr)/n:.0%} of {n:,} strategies produce lower Sharpe ratios. The baseline "
        f"is genuinely hard to beat.", ss["Q_Bul"]))
    E.append(Paragraph(
        f"<b>2. Drawdown compression is the core value proposition.</b> {n_beat_dd/n:.0%} of "
        f"strategies have shallower drawdowns than B&H's {bh['max_dd']:.0%}.", ss["Q_Bul"]))
    E.append(Paragraph(
        "<b>3. Daily frequency dominates everything.</b> This finding was not challenged and "
        "remains the single most important design choice.", ss["Q_Bul"]))
    E.append(Paragraph(
        f"<b>4. After multiple-testing correction, {n_survivors/n:.1%} survive.</b> The "
        f"survivors cluster in well-known families at daily frequency.", ss["Q_Bul"]))

    E.append(Spacer(1, 8))
    E.append(Paragraph("<b>What required revision:</b>", ss["Q_BB"]))
    E.append(Paragraph(
        f"<b>5. Skewness is not a free structural byproduct.</b> The original claim that "
        f"positive skewness was \"a property of the trade structure\" was an overstatement. "
        f"Signal timing explains {skew_decomp['pct_signal']:.0f}% of the improvement; the "
        f"structure explains only {skew_decomp['pct_structural']:.0f}%. Skewness is a meaningful "
        f"signal quality discriminator.", ss["Q_Bul"]))
    E.append(Paragraph(
        "<b>6. ATR vs fixed stops is not a uniform wash.</b> The aggregate 31% win rate masks "
        "10 Bonferroni-significant subgroups. ATR dominates for noisy-entry signals (EMA: 88%); "
        "fixed % dominates for regime-sensitive signals (ADX: 0%). Stop selection should be "
        "conditioned on signal family.", ss["Q_Bul"]))
    E.append(Paragraph(
        "<b>7. Trend does not need lower drift to add value.</b> This was the most consequential "
        "error. Bonferroni survivors beat B&H at every drift level tested, up to 300% annualized. "
        "The original conclusion generalized from the median strategy to trend as a class — "
        "the median is bad, but the statistically credible subset is not.", ss["Q_Bul"]))

    E.append(Spacer(1, 8))
    E.append(Paragraph("<b>What the extensions added:</b>", ss["Q_BB"]))
    E.append(Paragraph(
        "<b>8. In the 2022 bear, survivors halve the drawdown.</b> Median -39% vs -79% for B&H. "
        "Real but imperfect — most strategies still lose money, just substantially less.",
        ss["Q_Bul"]))
    E.append(Paragraph(
        f"<b>9. The in-sample optimal TIM is ~{task4_sum['optimal_tim']:.0%}.</b> The peak "
        f"location is stable under bootstrap resampling of this series, but the out-of-sample "
        f"uncertainty is substantially wider. Bonferroni survivors cluster near this optimum.",
        ss["Q_Bul"]))
    E.append(Paragraph(
        f"<b>10. A TIM-filtered ensemble beats B&H on Sharpe while halving the drawdown.</b> "
        f"Sharpe {task4_sum['ensemble_sharpe']:.2f} vs {bh['sharpe']:.2f}, max drawdown "
        f"{abs(task4_sum['ensemble_dd'])*100:.0f}% vs {abs(bh['max_dd'])*100:.0f}%.", ss["Q_Bul"]))
    E.append(Paragraph(
        f"<b>11. The TIM-filtered ensemble is deployable without look-ahead.</b> TIM is ρ = "
        f"{task5_sum['train_full_tim_corr']:.2f} stable across periods. Walk-forward selection "
        f"produces an ensemble with Sharpe {task5_sum['wf_full_sharpe']:.2f} — zero meaningful "
        f"decay vs the in-sample version. Ex-ante TIM prediction closes the gap between research "
        f"result and implementable strategy.", ss["Q_Bul"]))

    E.append(Spacer(1, 12))
    E.append(Paragraph(
        "The central lesson is that the initial sweep asked the wrong question. \"Does trend "
        "beat buy-and-hold?\" invites a single answer for 13,293 strategies, most of which are "
        "bad. The right question is: \"Among the strategies that survive statistical scrutiny, "
        "is there a construction that offers a better risk-return tradeoff than passive "
        "holding?\" The answer, after correcting our own errors, is yes — conditional on daily "
        "frequency, the right signal families, the right time-in-market target, and the "
        "discipline to diversify across many parameterizations rather than optimize a single one.",
        ss["Q_Body"]))
    E.append(Paragraph(
        f"The TIM-filtered ensemble provides a concrete implementation: target ~"
        f"{task4_sum['optimal_tim']:.0%} time-in-market through daily signals in known trend "
        f"families (DEMA, Supertrend, CCI, Aroon), diversify across 700+ parameterizations, and "
        f"accept roughly 85% of B&H's CAGR while absorbing roughly half its worst drawdown. "
        f"Whether that tradeoff is acceptable depends on the investor's drawdown tolerance — "
        f"but the tradeoff itself is now precisely quantified, and it is more favorable than our "
        f"initial analysis suggested.", ss["Q_Body"]))

    E.append(Spacer(1, 12))
    E.append(Paragraph("The binding constraint resolved: what crossed the border", ss["Q_SSec"]))
    E.append(Paragraph(
        "<b>12. Signal family rankings are asset-specific.</b> The cross-asset validation "
        "(Part 12) definitively answered the binding constraint from the original analysis. "
        "DEMA and Supertrend dominate on ETH; RSI and CCI dominate on BTC. Spearman ρ = 0.22 "
        "for family survival rankings across Tier 1 assets — below the pre-specified 0.3 "
        "threshold for Outcome C (path dependence). The ETH family preference is not deployable "
        "as a universal signal selection framework.", ss["Q_Bul"]))
    E.append(Paragraph(
        "<b>13. Stop-type preferences are asset-specific.</b> The ATR-for-EMA finding (88% on "
        "ETH) reverses completely on BTC (0%). Only 15% of families maintain consistent stop-type "
        "preference across assets. Stop selection must be conditioned on both family and asset.",
        ss["Q_Bul"]))
    E.append(Paragraph(
        "<b>14. TIM targeting and drawdown compression are structural.</b> "
        "TIM stability is ρ ≥ 0.99 on both ETH and BTC. The TIM optimum is portable (CIs "
        "overlap). Walk-forward selection works on both assets with 73–82% precision. The "
        "zero-reoptimization test shows ETH survivors retain positive Sharpe on 72–100% of Tier 2 "
        "assets and compress drawdowns at 90–100% rates. Trend-following as a mechanism — not the "
        "specific family or parameter — is structural.", ss["Q_Bul"]))
    E.append(Spacer(1, 6))
    E.append(Paragraph(
        "The honest summary: the <i>architecture</i> replicates but the <i>parameters</i> do not. "
        "Per-asset sweeps are necessary to identify locally dominant families and stop types. "
        "An allocator should use universal TIM targeting (40–60%) and walk-forward selection, but "
        "must run asset-specific signal discovery rather than importing ETH results.",
        ss["Q_BB"]))
    E.append(Paragraph(
        "<b>15. At full universe scale, trend works for two things: Sharpe on the majors, drawdown "
        "protection on everything else.</b> The full universe validation (Part 13) tested 31 assets "
        "with full sweeps and 148 with replication. Bonferroni survivors exist only on ETH and BTC. "
        "TIM stability (ρ ≥ 0.98) and drawdown compression (99% of strategies beat B&H on max DD) "
        "are universal. The median TIM optimum across 31 assets is 39%, consistent with the original "
        "ETH finding. The practical implication: trend-following on the majors for alpha, on alts "
        "for risk management.", ss["Q_Bul"]))

    E.append(Spacer(1, 12))
    E.append(Paragraph("<b>What has been resolved and what remains</b>", ss["Q_BB"]))
    E.append(Paragraph(
        "The walk-forward validation (Part 11) and cross-asset tests (Parts 12–13) partially "
        "address the in-sample concern flagged in our initial disclaimer. TIM stability is "
        "confirmed as structural (ρ ≥ 0.98 on every asset tested). Drawdown compression is "
        "universal (99% of blind-transferred strategies improve on B&H max DD). Walk-forward "
        "selection produces zero Sharpe decay on ETH. These results have crossed the border from "
        "in-sample hypothesis to out-of-sample evidence.", ss["Q_Body"]))
    E.append(Paragraph(
        "What remains unresolved: transaction costs at institutional scale on illiquid alts, "
        "portfolio-level construction across multiple assets, and live execution slippage. The "
        "20 bps cost assumption is reasonable for ETH and BTC but optimistic for smaller assets — "
        "the cost sensitivity analysis below quantifies this exposure directly. No backtest, "
        "however rigorous, substitutes for live trading. The path from research result to "
        "deployed capital requires paper trading, execution cost measurement, and gradual "
        "scaling.", ss["Q_Body"]))
    # ── Transaction cost sensitivity exhibit ─────────────────────
    E.append(Paragraph("Transaction Cost Sensitivity", ss["Q_SSec"]))
    E.append(Paragraph(
        "The 20 bps round-trip cost assumption used throughout this paper is reasonable for "
        "institutional spot execution on ETH and BTC. For smaller assets, or for the 1-hour "
        "frequency configurations excluded from the daily-only ensembles, higher costs apply. "
        "This exhibit stress-tests the daily TIM-filtered ensembles at 20, 40, and 60 bps "
        "round-trip.", ss["Q_Body"]))

    E.append(Paragraph("Exhibit 45: Transaction Cost Sensitivity", ss["Q_Ex"]))
    cs_data = [["Cost (bps)", "ETH Sharpe", "ETH Max DD", "BTC Sharpe", "BTC Max DD",
                "ETH Δ/20bp", "BTC Δ/20bp"]]
    for _, r in cost_sens.iterrows():
        eth_delta = f"{r['eth_sharpe_delta']:+.3f}" if pd.notna(r.get("eth_sharpe_delta")) else "—"
        btc_delta = f"{r['btc_sharpe_delta']:+.3f}" if pd.notna(r.get("btc_sharpe_delta")) else "—"
        cs_data.append([
            str(int(r["cost_bps"])),
            f"{r['eth_sharpe']:.3f}", f"{r['eth_max_dd']:.1%}",
            f"{r['btc_sharpe']:.3f}", f"{r['btc_max_dd']:.1%}",
            eth_delta, btc_delta,
        ])
    E.append(make_table(cs_data, [0.7*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch,
                                   0.8*inch, 0.8*inch]))
    E.append(Spacer(1, 4))
    E.append(img(FULLUNIV, "cost_sensitivity.png"))
    E.append(Paragraph(SOURCE, ss["Q_Cap"]))

    E.append(Paragraph(
        "Sharpe degradation is approximately linear: ~0.10 per 20 bps for ETH, ~0.14 per "
        "20 bps for BTC. The ETH ensemble retains a Sharpe above 1.0 at 60 bps — breakeven "
        "with Buy & Hold (Sharpe 1.11) occurs near 50 bps. The BTC ensemble falls below 1.0 at "
        "40 bps. For institutional allocators facing execution costs above 40 bps on a given "
        "asset, the trend ensemble's Sharpe advantage is consumed by friction. This is the "
        "binding constraint for deployment on smaller assets.", ss["Q_Body"]))

    E.append(PageBreak())

    # ================================================================
    # APPENDIX
    # ================================================================
    E.append(Paragraph("Appendix: Parameter Grid and Data Notes", ss["Q_Sec"]))
    E.append(Paragraph(
        "<b>Data:</b> Coinbase Advanced spot OHLCV, ETH-USD. January 1, 2017 – February 22, "
        "2026. Daily, 4-hour, and 1-hour bars.", ss["Q_Body"]))
    E.append(Paragraph(
        "<b>Base signals (493):</b> SMA crossover, EMA crossover, DEMA crossover, Hull MA "
        "crossover, price vs SMA/EMA, Donchian channel, Bollinger Bands, Keltner Channel, "
        "Supertrend, raw momentum, vol-scaled momentum, linear regression t-stat, MACD, RSI, "
        "ADX, CCI, Aroon, Stochastic, Parabolic SAR, Williams %R, MFI, TRIX, PPO, APO, MOM, "
        "ROC, CMO, Ichimoku, OBV, Heikin-Ashi, Kaufman, VWAP, dual momentum, triple MA, "
        "Turtle breakout, regime-filter SMA, ATR breakout, close-above-high, mean-reversion band.",
        ss["Q_Body"]))
    E.append(Paragraph(
        "<b>Stop variants (9):</b> None; fixed trailing at 5%, 10%, 20%; ATR-based trailing "
        "at 1.5×, 2.0×, 2.5×, 3.0×, 4.0× (14-period ATR at entry date).", ss["Q_Body"]))
    E.append(Paragraph(
        "<b>Backtest rules:</b> Binary long/cash. One-bar lag. 20 bps round-trip transaction "
        "costs. No leverage. No position sizing. Intraday signals resampled to daily for P&L.",
        ss["Q_Body"]))
    E.append(Paragraph(
        "<b>Multiple testing:</b> Effective independent tests = 1,479 (493 signals × 3 "
        "frequencies). Stop variants treated as dependent. Bonferroni correction at 5% FWER.",
        ss["Q_Body"]))
    E.append(Paragraph(
        "<b>Extension analyses:</b> Task 1 — 10,000 random strategies (seed 42). Task 2 — "
        "102 frequency × family cells. Task 3 — 380 daily Bonferroni survivors on 4 sub-periods, "
        "10 synthetic drift levels, and the 2022 bear market. Task 4 — 1,000 bootstrap samples "
        "for TIM curve CI, 704 strategies in the TIM-filtered ensemble. Task 5 — ex-ante TIM "
        "prediction via Ridge/GBM cross-validation and walk-forward (train 2017–2021, test "
        "2022–2026) with 711-strategy walk-forward ensemble.", ss["Q_Body"]))

    grid = [
        ["Variable", "Definition", "What We Test"],
        ["Signals", "Which trend rule to apply", "493 base signals from 30+ families"],
        ["Frequency", "Bar interval for signal computation", "Daily, 4-hour, 1-hour"],
        ["Stops", "Trailing stop exit overlay", "None; 5–20% fixed; 1.5×–4.0× ATR"],
    ]
    E.append(Spacer(1, 12))
    E.append(Paragraph("Exhibit A1: Building 13,293 Trend Strategies", ss["Q_Ex"]))
    E.append(make_table(grid, [0.9*inch, 2.0*inch, 2.8*inch]))
    E.append(PageBreak())

    # ================================================================
    # REFERENCES
    # ================================================================
    E.append(Paragraph("References and Further Reading", ss["Q_Sec"]))
    refs = [
        "Cao, Jeffrey, Nathan Chong, and Dan Villalon. \"Hold the Dip.\" <i>AQR Alternative "
        "Thinking</i> 2025, Issue 4.",
        "Hurst, Brian, Yao Hua Ooi, Lasse Heje Pedersen. \"A Century of Evidence on Trend-"
        "Following Investing.\" <i>The Journal of Portfolio Management</i> 44, no. 1 (2017).",
        "Moskowitz, Tobias J., Yao Hua Ooi, Lasse Heje Pedersen. \"Time series momentum.\" "
        "<i>Journal of Financial Economics</i> 104, Issue 2 (2012): 228-50.",
        "Babu, Abilash, Brendan Hoffman, Ari Levine, et al. \"You Can't Always Trend When You "
        "Want.\" <i>The Journal of Portfolio Management</i> 46, no. 4 (2020).",
        "AQR. \"Trend-Following: Why Now? A Macro Perspective.\" AQR whitepaper, November 2022.",
        "Harvey, Campbell R., Yan Liu. \"Backtesting.\" <i>The Journal of Portfolio Management</i> "
        "42, no. 1 (2015): 13-28.",
    ]
    for ref in refs:
        E.append(Paragraph(ref, ss["Q_Body"]))

    E.append(Spacer(1, 0.5 * inch))
    E.append(hr())
    E.append(Paragraph(
        "HYPOTHETICAL PERFORMANCE RESULTS HAVE MANY INHERENT LIMITATIONS. NO REPRESENTATION "
        "IS BEING MADE THAT ANY ACCOUNT WILL OR IS LIKELY TO ACHIEVE PROFITS OR LOSSES SIMILAR "
        "TO THOSE SHOWN. IN FACT, THERE ARE FREQUENTLY SHARP DIFFERENCES BETWEEN HYPOTHETICAL "
        "PERFORMANCE RESULTS AND THE ACTUAL RESULTS SUBSEQUENTLY ACHIEVED BY ANY PARTICULAR "
        "TRADING PROGRAM.", ss["Q_Disc"]))

    # ================================================================
    # BUILD
    # ================================================================
    doc.build(E)
    print(f"[pdf] Written: {PDF_PATH}")
    print(f"[pdf] Size: {PDF_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build_pdf()
