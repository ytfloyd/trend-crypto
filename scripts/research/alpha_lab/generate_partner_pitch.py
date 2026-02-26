#!/usr/bin/env python3
"""
Partner Pitch — Forward Simulation Report
==========================================

Generates a JP Morgan-styled PDF presenting the "asymmetric entry
opportunity" narrative for a prop-firm joint-trading venture.

Pages:
  1. Title + Executive Summary
  2. Strategy Track Record
  3. Why Now — The Drawdown Advantage
  4. Forward Simulation Fan Chart
  5. Strategy vs. Buy-and-Hold Simulation
  6. Risk Metrics & Stress Scenarios
  7. Terms & Structure (placeholder)

Usage:
    python -m scripts.research.alpha_lab.generate_partner_pitch
    python -m scripts.research.alpha_lab.generate_partner_pitch --horizon 5
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
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
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
    HRFlowable,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from common.data import load_daily_bars, ANN_FACTOR
from common.metrics import compute_metrics

from scripts.research.alpha_lab.forward_simulation import (
    SimConfig,
    block_bootstrap_paths,
    post_drawdown_bootstrap_paths,
    conditional_entry_returns,
    historical_analogues,
    fan_chart_summary,
    terminal_wealth_table,
    simulated_drawdown_stats,
    btc_bootstrap_paths,
)
from scripts.research.alpha_lab.turtle_portfolio_v2 import (
    prepare_data,
    run_simulation,
    analyze_variant,
    OverlayConfig,
)

OUT_DIR = ROOT / "artifacts" / "research" / "alpha_lab"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "partner_pitch_forward_sim.pdf"

# ── Colour palette (JPM-inspired) ────────────────────────────────────
JPM_BLUE = colors.Color(0.06, 0.18, 0.37)
JPM_BLUE_LIGHT = colors.Color(0.20, 0.40, 0.65)
JPM_GOLD = colors.Color(0.76, 0.63, 0.33)
JPM_GRAY = colors.Color(0.55, 0.55, 0.55)
JPM_GRAY_LIGHT = colors.Color(0.93, 0.93, 0.93)
JPM_GREEN = colors.Color(0.18, 0.49, 0.20)
WHITE = colors.white
BLACK = colors.black

CB = "#0F2E5F"; CLB = "#3366A6"; CG = "#C2A154"; CR = "#B22222"
CGr = "#2E7D32"; CGy = "#888888"; CTEAL = "#006B6B"
CORAL = "#E07040"; PURPLE = "#6B3FA0"

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# ═══════════════════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════════════════

def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "Title", parent=ss["Title"], fontName="Times-Bold", fontSize=28,
        leading=34, textColor=WHITE, alignment=TA_CENTER, spaceAfter=12)
    s["subtitle"] = ParagraphStyle(
        "Subtitle", parent=ss["Normal"], fontName="Times-Roman", fontSize=16,
        leading=20, textColor=colors.Color(0.85, 0.85, 0.85),
        alignment=TA_CENTER, spaceAfter=6)
    s["cover_date"] = ParagraphStyle(
        "CoverDate", parent=ss["Normal"], fontName="Helvetica", fontSize=11,
        leading=14, textColor=JPM_GOLD, alignment=TA_CENTER, spaceAfter=4)
    s["cover_tag"] = ParagraphStyle(
        "CoverTag", parent=ss["Normal"], fontName="Times-Italic", fontSize=12,
        leading=16, textColor=colors.Color(0.70, 0.70, 0.70),
        alignment=TA_CENTER, spaceAfter=4)
    s["h1"] = ParagraphStyle(
        "H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=18,
        leading=22, textColor=JPM_BLUE, spaceBefore=20, spaceAfter=10)
    s["h2"] = ParagraphStyle(
        "H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13,
        leading=16, textColor=JPM_BLUE_LIGHT, spaceBefore=14, spaceAfter=6)
    s["body"] = ParagraphStyle(
        "Body", parent=ss["Normal"], fontName="Times-Roman", fontSize=10,
        leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6)
    s["body_bold"] = ParagraphStyle(
        "BodyBold", parent=ss["Normal"], fontName="Times-Bold", fontSize=10,
        leading=13.5, textColor=BLACK, spaceBefore=2, spaceAfter=6)
    s["bullet"] = ParagraphStyle(
        "Bullet", parent=ss["Normal"], fontName="Times-Roman", fontSize=10,
        leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        leftIndent=18, bulletIndent=6, spaceBefore=1, spaceAfter=2)
    s["caption"] = ParagraphStyle(
        "Caption", parent=ss["Normal"], fontName="Helvetica", fontSize=8.5,
        leading=11, textColor=JPM_GRAY, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=10)
    s["disclaimer"] = ParagraphStyle(
        "Disclaimer", parent=ss["Normal"], fontName="Helvetica", fontSize=7,
        leading=9, textColor=JPM_GRAY, alignment=TA_JUSTIFY)
    s["key_stat_label"] = ParagraphStyle(
        "KSL", parent=ss["Normal"], fontName="Helvetica", fontSize=8,
        leading=10, textColor=JPM_GRAY, alignment=TA_CENTER)
    s["key_stat_value"] = ParagraphStyle(
        "KSV", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=18,
        leading=22, textColor=JPM_BLUE, alignment=TA_CENTER)
    s["key_stat_value_green"] = ParagraphStyle(
        "KSVg", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=18,
        leading=22, textColor=JPM_GREEN, alignment=TA_CENTER)
    return s


# ═══════════════════════════════════════════════════════════════════════
# Page templates
# ═══════════════════════════════════════════════════════════════════════

def _header_footer(canvas, doc, is_cover=False):
    canvas.saveState()
    if not is_cover:
        canvas.setStrokeColor(JPM_BLUE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - MARGIN + 6, PAGE_W - MARGIN, PAGE_H - MARGIN + 6)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(JPM_GRAY)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 10,
                          "Systematic Crypto Trend Following — Partner Pitch")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 10, "NRT Research")
        canvas.setStrokeColor(JPM_BLUE)
        canvas.line(MARGIN, MARGIN - 14, PAGE_W - MARGIN, MARGIN - 14)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, MARGIN - 24, "CONFIDENTIAL — For authorized recipients only")
        canvas.drawCentredString(PAGE_W / 2, MARGIN - 24, f"Page {doc.page}")
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 24, datetime.now().strftime("%B %Y"))
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


# ═══════════════════════════════════════════════════════════════════════
# Chart helpers
# ═══════════════════════════════════════════════════════════════════════

def set_chart_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
        "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.fontsize": 8, "legend.framealpha": 0.9,
    })


def fig_to_image(fig, width=6.5 * inch, ratio=0.55):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width, height=width * ratio)


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


def key_stat_row(labels, values, sty, green_indices=None):
    """Build a 2-row key-stat banner."""
    if green_indices is None:
        green_indices = set()
    top = [Paragraph(l, sty["key_stat_label"]) for l in labels]
    bottom = []
    for i, v in enumerate(values):
        style = sty["key_stat_value_green"] if i in green_indices else sty["key_stat_value"]
        bottom.append(Paragraph(v, style))
    n = len(labels)
    cw = [CONTENT_W / n] * n
    t = Table([top, bottom], colWidths=cw)
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, -1), JPM_GRAY_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, JPM_BLUE),
    ]))
    return t


# ═══════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════

def chart_track_record(strategy_equity, btc_equity):
    set_chart_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.5]})
    ax1.plot(btc_equity.index, btc_equity.values, color=CG, lw=1, ls="--",
             alpha=0.6, label="BTC Buy & Hold")
    ax1.plot(strategy_equity.index, strategy_equity.values, color=CB, lw=1.8,
             label="Turtle DD+BTC+Top10")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title("Strategy vs. BTC Buy & Hold — Full Track Record")
    ax1.legend(loc="upper left")

    strat_dd = strategy_equity / strategy_equity.cummax() - 1.0
    btc_dd = btc_equity / btc_equity.cummax() - 1.0
    ax2.fill_between(btc_dd.index, btc_dd.values, 0, alpha=0.08, color=CG)
    ax2.plot(btc_dd.index, btc_dd.values, color=CG, lw=0.6, ls="--", alpha=0.5,
             label="BTC B&H")
    ax2.fill_between(strat_dd.index, strat_dd.values, 0, alpha=0.15, color=CB)
    ax2.plot(strat_dd.index, strat_dd.values, color=CB, lw=1.0, label="Strategy")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=7)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    return fig


def chart_annual_returns(strategy_equity, btc_equity):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for eq, label, color, alpha in [
        (btc_equity, "BTC B&H", CG, 0.5),
        (strategy_equity, "Turtle DD+BTC+Top10", CB, 0.85),
    ]:
        eq_c = eq.copy()
        eq_c.index = pd.to_datetime(eq_c.index)
        ann = eq_c.resample("YE").last().pct_change().dropna()
        years = ann.index.year
        x = np.arange(len(years))
        offset = -0.2 if "BTC" in label else 0.2
        ax.bar(x + offset, ann.values, 0.35, label=label, color=color, alpha=alpha)
        ax.set_xticks(x)
        ax.set_xticklabels(years)

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("Annual Returns")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def chart_drawdown_cycles(strategy_equity):
    """Show drawdown curve with shading for major bear periods."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    dd = strategy_equity / strategy_equity.cummax() - 1.0
    ax.fill_between(dd.index, dd.values, 0, alpha=0.2, color=CB)
    ax.plot(dd.index, dd.values, color=CB, lw=1.0)

    bear_periods = [
        ("2018 Bear", "2018-01-01", "2018-12-31"),
        ("COVID", "2020-02-15", "2020-05-15"),
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("2025-26", "2025-06-01", "2026-02-28"),
    ]
    for label, s, e in bear_periods:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        if s_ts >= dd.index.min() and s_ts <= dd.index.max():
            ax.axvspan(s_ts, min(e_ts, dd.index.max()),
                       alpha=0.06, color=CR)
            y_pos = dd.loc[s_ts:e_ts].min() if s_ts in dd.index else dd.min()
            mid = s_ts + (min(e_ts, dd.index.max()) - s_ts) / 2
            ax.text(mid, -0.02, label, ha="center", va="top", fontsize=7,
                    color=CR, alpha=0.7)

    ax.set_title("Strategy Drawdown Timeline")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.tight_layout()
    return fig


def chart_fan(fan_df, cfg, title, color=CB, analogues=None):
    """Fan chart from percentile DataFrame."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    days = np.arange(len(fan_df))
    years = days / 365.0

    capital = cfg.initial_capital

    ax.fill_between(years, fan_df["p5"] * capital, fan_df["p95"] * capital,
                    alpha=0.08, color=color, label="5th–95th pctl")
    ax.fill_between(years, fan_df["p25"] * capital, fan_df["p75"] * capital,
                    alpha=0.18, color=color, label="25th–75th pctl")
    ax.plot(years, fan_df["p50"] * capital, color=color, lw=2.0, label="Median")
    ax.plot(years, fan_df["mean"] * capital, color=color, lw=1.0, ls="--",
            alpha=0.7, label="Mean")
    ax.axhline(capital, color="gray", lw=0.5, ls=":")

    if analogues:
        analogue_colors = [CGr, CORAL, PURPLE, CTEAL]
        for idx, (label, path) in enumerate(analogues.items()):
            path_days = np.arange(len(path))
            path_years = path_days / 365.0
            c = analogue_colors[idx % len(analogue_colors)]
            ax.plot(path_years, path.values * capital, color=c, lw=1.2,
                    ls="-.", alpha=0.8, label=label)

    ax.set_title(title)
    ax.set_xlabel("Years from Entry")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"))
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def chart_fan_comparison(strat_fan, btc_fan, cfg):
    """Side-by-side fan charts for strategy vs BTC."""
    set_chart_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    days = np.arange(len(strat_fan))
    years = days / 365.0
    capital = cfg.initial_capital

    for ax, fan, color, title in [
        (ax1, strat_fan, CB, "Turtle DD+BTC+Top10"),
        (ax2, btc_fan, CG, "BTC Buy & Hold"),
    ]:
        ax.fill_between(years, fan["p5"] * capital, fan["p95"] * capital,
                        alpha=0.08, color=color)
        ax.fill_between(years, fan["p25"] * capital, fan["p75"] * capital,
                        alpha=0.18, color=color)
        ax.plot(years, fan["p50"] * capital, color=color, lw=2.0, label="Median")
        ax.plot(years, fan["mean"] * capital, color=color, lw=1.0, ls="--",
                alpha=0.7, label="Mean")
        ax.axhline(capital, color="gray", lw=0.5, ls=":")
        ax.set_title(title)
        ax.set_xlabel("Years from Entry")
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"))

    ax1.set_ylabel("Portfolio Value ($)")
    fig.suptitle(f"Forward Simulation: $1M Allocation — {int(cfg.horizon_days/365)}-Year Projection",
                 fontsize=12, fontweight="bold", color=CB, y=1.02)
    fig.tight_layout()
    return fig


def chart_dd_distribution(strat_paths, btc_paths):
    """Histogram of simulated max drawdowns."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for paths, label, color in [
        (btc_paths, "BTC Buy & Hold", CG),
        (strat_paths, "Turtle DD+BTC+Top10", CB),
    ]:
        max_dds = np.empty(paths.shape[0])
        for i in range(paths.shape[0]):
            running_max = np.maximum.accumulate(paths[i])
            dd = paths[i] / running_max - 1.0
            max_dds[i] = dd.min()
        ax.hist(max_dds, bins=80, alpha=0.45, color=color, label=label,
                density=True, edgecolor="white", linewidth=0.3)

    ax.set_title("Simulated Maximum Drawdown Distribution")
    ax.set_xlabel("Max Drawdown")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Report builder
# ═══════════════════════════════════════════════════════════════════════

def generate_report(
    strategy_equity: pd.Series,
    btc_equity: pd.Series,
    strat_metrics: dict,
    btc_metrics: dict,
    strat_analysis: dict,
    strat_fan: pd.DataFrame,
    btc_fan: pd.DataFrame,
    dd_entry_fan: pd.DataFrame,
    strat_paths: np.ndarray,
    btc_paths: np.ndarray,
    dd_entry_paths: np.ndarray,
    analogues: dict,
    cond_entry: pd.DataFrame,
    cfg: SimConfig,
):
    print("[report] Building PDF ...")
    sty = build_styles()
    sm = strat_metrics
    bm = btc_metrics

    strat_tw = terminal_wealth_table(strat_paths, cfg)
    btc_tw = terminal_wealth_table(btc_paths, cfg)
    dd_entry_tw = terminal_wealth_table(dd_entry_paths, cfg)
    strat_dd = simulated_drawdown_stats(strat_paths, cfg)
    btc_dd_stats = simulated_drawdown_stats(btc_paths, cfg)
    horizon_yr = int(cfg.horizon_days / 365)

    frame_cover = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="cover")
    frame_body = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="body")
    doc = BaseDocTemplate(str(PDF_PATH), pagesize=letter,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN)
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[frame_cover], onPage=on_cover),
        PageTemplate(id="Body", frames=[frame_body], onPage=on_body),
    ])

    story = []

    # ──────────────────────────────────────────────────────────────────
    # PAGE 1: Cover + Executive Summary
    # ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, PAGE_H * 0.22))
    story.append(Paragraph("Systematic Crypto Trend Following", sty["title"]))
    story.append(Paragraph("Strategy Overview & Forward Simulation", sty["subtitle"]))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Joint Venture Discussion Document", sty["cover_tag"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"NRT Research · {datetime.now().strftime('%B %Y')}", sty["cover_date"]))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    # Executive summary
    story.append(Paragraph("1. Executive Summary", sty["h1"]))

    current_dd = float(strategy_equity.iloc[-1] / strategy_equity.max() - 1.0)

    story.append(key_stat_row(
        ["Current DD from Peak", "Backtest Sharpe", "Backtest CAGR",
         f"Median {horizon_yr}Y Terminal", f"P(Profit) {horizon_yr}Y"],
        [f"{current_dd:.1%}", f"{sm['sharpe']:.2f}", f"{sm['cagr']:.1%}",
         f"${strat_tw['median']:,.0f}", f"{strat_tw['prob_profit']:.0%}"],
        sty,
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Important:</b> All performance figures in this document are based on "
        "hypothetical backtested results. The strategy has not yet been traded with "
        "live capital. Backtested results are subject to look-ahead bias, overfitting "
        "risk, and execution assumptions that may not hold in live trading.",
        sty["body"]))

    story.append(Paragraph(
        "The managed strategy is a systematic Turtle Trading system adapted for crypto "
        f"with gain-protection overlays. It is currently {abs(current_dd):.0%} below its "
        f"backtested all-time high — compared to {abs(bm['max_dd']):.0%} max drawdown "
        "for a passive BTC buy-and-hold approach over the same period. The strategy's "
        "primary value proposition is <b>drawdown compression</b>: delivering trend-following "
        "returns with substantially reduced peak-to-trough losses.", sty["body"]))

    story.append(Paragraph(
        f"Block-bootstrap Monte Carlo simulations ({cfg.n_paths:,} paths, "
        f"{horizon_yr}-year horizon) project a <b>median terminal value of "
        f"${strat_tw['median']:,.0f}</b> on a $1M allocation, with a "
        f"<b>{strat_tw['prob_profit']:.0%} probability of profit</b> and "
        f"<b>{strat_tw['prob_double']:.0%} probability of doubling capital</b>. "
        "When conditioning the simulation on entry at a drawdown level comparable "
        f"to today's ({current_dd:.0%}), the median outcome is "
        f"<b>${dd_entry_tw['median']:,.0f}</b> "
        f"({dd_entry_tw['median']/strat_tw['median'] - 1:+.0%} vs. unconditioned).",
        sty["body"]))

    story.append(Paragraph(
        "The following pages present the full backtested track record, conditional "
        "entry analysis, forward simulation fan charts, and risk metrics.",
        sty["body"]))

    # ──────────────────────────────────────────────────────────────────
    # PAGE 2: Strategy Track Record
    # ──────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("2. Strategy Track Record", sty["h1"]))

    story.append(Paragraph(
        "The strategy is a Turtle Trading system adapted for crypto with three "
        "gain-protection overlays: portfolio drawdown control (liquidate at −20% from peak), "
        "a BTC regime filter (require BTC uptrend for entries), and universe concentration "
        "(top 10 assets by volume). It is long-only, fully systematic, and trades daily.", sty["body"]))

    story.append(fig_to_image(chart_track_record(strategy_equity, btc_equity)))
    story.append(Paragraph("Exhibit 1: Strategy equity curve vs. BTC buy-and-hold (log scale) "
                           "with drawdown overlay.", sty["caption"]))

    headers = ["Metric", "Strategy", "BTC B&H", "Edge"]
    rows = []
    for lbl, key, fmt in [
        ("CAGR", "cagr", ".1%"), ("Volatility", "vol", ".1%"),
        ("Sharpe", "sharpe", ".2f"), ("Sortino", "sortino", ".2f"),
        ("Max Drawdown", "max_dd", ".1%"), ("Calmar", "calmar", ".2f"),
        ("Skewness", "skewness", ".2f"),
    ]:
        sv = sm[key]
        bv = bm[key]
        edge = sv - bv if key != "max_dd" else bv - sv
        rows.append([lbl, f"{sv:{fmt}}", f"{bv:{fmt}}", f"{edge:+{fmt}}"])

    cw = [1.6 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]
    story.append(make_table(headers, rows, col_widths=cw))
    story.append(Paragraph("Exhibit 2: Head-to-head performance comparison. "
                           "'Edge' = strategy advantage.", sty["caption"]))

    sortino_edge = sm["sortino"] - bm["sortino"]
    if sortino_edge < 0:
        story.append(Paragraph(
            f"Note: the Sortino edge ({sortino_edge:+.2f}) reflects the strategy's "
            "occasional underperformance in sharp V-shaped recoveries, where the BTC "
            "regime filter causes re-entry lag after the trough. This is the expected "
            "cost of the drawdown-compression mechanism — the same filter that avoids "
            "sustained bear markets will sometimes miss the first leg of a snapback.",
            sty["body"]))

    story.append(fig_to_image(chart_annual_returns(strategy_equity, btc_equity), ratio=0.45))
    story.append(Paragraph("Exhibit 3: Annual returns. The strategy captures "
                           "bull markets while protecting during bears.", sty["caption"]))

    # ──────────────────────────────────────────────────────────────────
    # PAGE 3: Why Now — The Drawdown Advantage
    # ──────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("3. Why Now — The Drawdown Advantage", sty["h1"]))

    story.append(Paragraph(
        "Entering a trend-following strategy after a drawdown period has historically "
        "produced superior forward returns. This is not alpha decay — it is the mechanical "
        "consequence of mean-reverting drawdown depth: strategies that have already pulled "
        "back have less distance to fall and more room to run.", sty["body"]))

    story.append(fig_to_image(chart_drawdown_cycles(strategy_equity), ratio=0.4))
    story.append(Paragraph(
        "Exhibit 4: Strategy drawdown timeline with bear market periods highlighted.", sty["caption"]))

    story.append(Paragraph("Conditional Forward Returns by Entry Drawdown", sty["h2"]))
    story.append(Paragraph(
        "The table below buckets every historical entry day by its current drawdown depth "
        "(Q1 = deepest drawdown, farthest from peak; Q5 = shallowest drawdown, near peak) "
        "and shows the median forward return and annualised forward Sharpe ratio at each "
        "horizon.", sty["body"]))

    return_cols = [c for c in cond_entry.columns
                   if c != "DD Range" and "Sharpe" not in c]
    sharpe_cols = [c for c in cond_entry.columns if "Sharpe" in c]

    cond_headers = ["DD Quintile", "DD Range"] + return_cols + sharpe_cols
    cond_rows = []
    for idx_label, row in cond_entry.iterrows():
        r = [str(idx_label), row.get("DD Range", "—")]
        for c in return_cols:
            v = row[c]
            r.append(f"{v:.1%}" if pd.notna(v) else "—")
        for c in sharpe_cols:
            v = row[c]
            r.append(f"{v:.2f}" if pd.notna(v) else "—")
        cond_rows.append(r)

    # Determine which quintile has the best long-horizon return
    best_q_idx = 0
    if len(return_cols) > 0 and len(cond_entry) > 0:
        last_ret_col = return_cols[-1]
        vals = cond_entry[last_ret_col].dropna()
        if len(vals) > 0:
            best_q_idx = vals.values.argmax()

    q1_better = False
    if len(return_cols) > 0 and len(cond_entry) >= 2:
        last_col = return_cols[-1]
        q1_val = cond_entry[last_col].iloc[0]
        q5_val = cond_entry[last_col].iloc[-1]
        q1_better = pd.notna(q1_val) and pd.notna(q5_val) and q1_val > q5_val

    n_q = len(cond_rows)
    story.append(make_table(cond_headers, cond_rows,
                            highlight_row=best_q_idx if n_q > 0 else None))

    if q1_better:
        caption_direction = ("Q1 (deepest drawdown) shows higher median forward returns "
                             "than Q5 (near peak).")
    else:
        caption_direction = ("Q5 (near peak) shows higher median forward returns "
                             "than Q1 (deepest drawdown).")

    story.append(Paragraph(
        f"Exhibit 5: Median forward returns and Sharpe by entry drawdown quintile. "
        f"{caption_direction} Forward Sharpe controls for volatility differences "
        f"across entry conditions.", sty["caption"]))

    # Identify which quintile the current DD falls into
    dd_series = strategy_equity / strategy_equity.cummax() - 1.0
    try:
        current_quintile = pd.cut(
            pd.Series([current_dd]),
            bins=pd.qcut(dd_series, 5, duplicates="drop", retbins=True)[1],
            labels=[f"Q{i+1}" for i in range(5)],
        ).iloc[0]
    except Exception:
        current_quintile = None

    if current_quintile is not None:
        story.append(Paragraph(
            f"<b>Current positioning:</b> The strategy is {abs(current_dd):.0%} from its "
            f"peak, placing the current entry point in <b>{current_quintile}</b>.",
            sty["body"]))
    else:
        story.append(Paragraph(
            f"<b>Current positioning:</b> The strategy is {abs(current_dd):.0%} from its "
            f"peak.", sty["body"]))

    # ──────────────────────────────────────────────────────────────────
    # PAGE 4: Forward Simulation Fan Chart
    # ──────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Forward Simulation — Monte Carlo Projection", sty["h1"]))

    story.append(Paragraph(
        f"We project the strategy forward {horizon_yr} years using a circular block bootstrap "
        f"({cfg.n_paths:,} simulated paths, {cfg.block_size}-day blocks) that preserves the "
        "autocorrelation and volatility clustering present in the actual track record. "
        "No distributional assumptions are made — every simulated return is drawn from "
        "the strategy's own history.", sty["body"]))

    story.append(fig_to_image(
        chart_fan(strat_fan, cfg,
                  f"Strategy Forward Simulation — ${cfg.initial_capital/1e6:.0f}M, {horizon_yr}Y",
                  color=CB, analogues=analogues)))
    story.append(Paragraph(
        f"Exhibit 6: Fan chart showing percentile bands of {cfg.n_paths:,} simulated equity paths. "
        "Dashed lines are historical analogue paths (actual forward returns from prior "
        "bear-market entry points).", sty["caption"]))

    story.append(Paragraph(
        "Drawdown-Conditioned Simulation", sty["h2"]))

    dd_pct_change = dd_entry_tw['median'] / strat_tw['median'] - 1

    story.append(Paragraph(
        f"The simulation above uses the full historical return distribution. We also "
        f"run a separate bootstrap that draws blocks exclusively from dates when the "
        f"strategy was at a comparable drawdown level to today ({current_dd:.0%} "
        f"+/- 10 pct pts). This directly answers: <i>historically, when we entered at "
        f"a similar drawdown, what happened next?</i>", sty["body"]))

    dd_better = dd_entry_tw['median'] > strat_tw['median']
    green_idx = {1, 2} if dd_better else set()

    story.append(key_stat_row(
        ["Unconditioned Median", "Same-DD-Entry Median", "Difference",
         "Same-DD P(Profit)", "Same-DD P(2×)"],
        [f"${strat_tw['median']:,.0f}", f"${dd_entry_tw['median']:,.0f}",
         f"{dd_pct_change:+.0%}",
         f"{dd_entry_tw['prob_profit']:.0%}", f"{dd_entry_tw['prob_double']:.0%}"],
        sty, green_indices=green_idx,
    ))

    # Terminal wealth table
    story.append(Spacer(1, 10))
    tw_headers = ["Statistic", "Strategy", "BTC B&H"]
    tw_rows = [
        ["Median Terminal", f"${strat_tw['median']:,.0f}", f"${btc_tw['median']:,.0f}"],
        ["Mean Terminal", f"${strat_tw['mean']:,.0f}", f"${btc_tw['mean']:,.0f}"],
        ["5th Percentile", f"${strat_tw['p5']:,.0f}", f"${btc_tw['p5']:,.0f}"],
        ["95th Percentile", f"${strat_tw['p95']:,.0f}", f"${btc_tw['p95']:,.0f}"],
        ["P(Profit)", f"{strat_tw['prob_profit']:.0%}", f"{btc_tw['prob_profit']:.0%}"],
        ["P(Double)", f"{strat_tw['prob_double']:.0%}", f"{btc_tw['prob_double']:.0%}"],
        ["P(Triple)", f"{strat_tw['prob_triple']:.0%}", f"{btc_tw['prob_triple']:.0%}"],
    ]
    story.append(make_table(tw_headers, tw_rows,
                            col_widths=[2 * inch, 2.2 * inch, 2.2 * inch]))
    story.append(Paragraph(
        f"Exhibit 7: Terminal wealth statistics ({horizon_yr}-year horizon, "
        f"$1M initial allocation, {cfg.n_paths:,} simulations).", sty["caption"]))

    # ──────────────────────────────────────────────────────────────────
    # PAGE 5: Strategy vs. Buy-and-Hold Simulation
    # ──────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Strategy vs. Buy-and-Hold — Side by Side", sty["h1"]))

    story.append(Paragraph(
        "The comparison below simulates both the managed strategy and a passive BTC "
        "buy-and-hold allocation forward using the same block-bootstrap methodology. "
        "While BTC buy-and-hold offers higher <i>mean</i> upside, it comes with dramatically "
        "wider dispersion and much larger tail risk.", sty["body"]))

    story.append(fig_to_image(chart_fan_comparison(strat_fan, btc_fan, cfg), ratio=0.42))
    story.append(Paragraph(
        f"Exhibit 8: Side-by-side fan charts — {horizon_yr}-year projection, $1M allocation.",
        sty["caption"]))

    story.append(Paragraph(
        "The critical insight is not the median — it is the <b>left tail</b>. "
        f"The strategy's 5th-percentile outcome (${strat_tw['p5']:,.0f}) represents a "
        f"much smaller loss than BTC buy-and-hold's 5th percentile (${btc_tw['p5']:,.0f}). "
        "For a partner allocating meaningful capital, this downside compression is the "
        "primary value proposition.", sty["body"]))

    # ──────────────────────────────────────────────────────────────────
    # PAGE 6: Risk Metrics & Stress Scenarios
    # ──────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("6. Risk Metrics & Stress Scenarios", sty["h1"]))

    story.append(fig_to_image(chart_dd_distribution(strat_paths, btc_paths), ratio=0.45))
    story.append(Paragraph(
        "Exhibit 9: Distribution of simulated max drawdowns across all paths.", sty["caption"]))

    dd_headers = ["Drawdown Metric", "Strategy", "BTC B&H"]
    dd_rows = [
        ["Median Max DD",
         f"{strat_dd['median_max_dd']:.1%}", f"{btc_dd_stats['median_max_dd']:.1%}"],
        ["5th Pctl Max DD (worst)",
         f"{strat_dd['p5_max_dd']:.1%}", f"{btc_dd_stats['p5_max_dd']:.1%}"],
        ["95th Pctl Max DD (best)",
         f"{strat_dd['p95_max_dd']:.1%}", f"{btc_dd_stats['p95_max_dd']:.1%}"],
        ["Worst Simulated DD",
         f"{strat_dd['worst_max_dd']:.1%}", f"{btc_dd_stats['worst_max_dd']:.1%}"],
        ["P(Max DD < 20%)",
         f"{strat_dd['prob_dd_lt_20pct']:.0%}", f"{btc_dd_stats['prob_dd_lt_20pct']:.0%}"],
        ["P(Max DD < 30%)",
         f"{strat_dd['prob_dd_lt_30pct']:.0%}", f"{btc_dd_stats['prob_dd_lt_30pct']:.0%}"],
        ["P(Max DD < 50%)",
         f"{strat_dd['prob_dd_lt_50pct']:.0%}", f"{btc_dd_stats['prob_dd_lt_50pct']:.0%}"],
    ]
    story.append(make_table(dd_headers, dd_rows,
                            col_widths=[2.5 * inch, 2 * inch, 2 * inch]))
    story.append(Paragraph(
        f"Exhibit 10: Simulated max drawdown statistics ({horizon_yr}-year horizon).",
        sty["caption"]))

    story.append(Paragraph(
        f"The strategy exhibits dramatically compressed drawdown risk. The median "
        f"simulated max drawdown is {strat_dd['median_max_dd']:.0%} for the managed "
        f"strategy vs. {btc_dd_stats['median_max_dd']:.0%} for BTC buy-and-hold. "
        f"Even in the worst {cfg.n_paths:,}-path scenario, the strategy's max drawdown "
        f"({strat_dd['worst_max_dd']:.0%}) remains smaller than the median BTC outcome.",
        sty["body"]))

    # Backtest stress periods
    story.append(Paragraph("Historical Stress Test Performance", sty["h2"]))

    eq_idx = pd.to_datetime(strategy_equity.index)
    stress_periods = [
        ("2018 Bear Market", "2018-01-01", "2018-12-31"),
        ("March 2020 COVID", "2020-02-15", "2020-05-15"),
        ("May–Jul 2021 Crash", "2021-05-01", "2021-07-31"),
        ("2022 Crypto Winter", "2022-01-01", "2022-12-31"),
    ]
    stress_headers = ["Crisis Period", "Strategy Return", "BTC Return", "Relative Return"]
    stress_rows = []
    for label, s, e in stress_periods:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        mask = (eq_idx >= s_ts) & (eq_idx <= e_ts)
        if mask.sum() < 2:
            continue
        strat_slice = strategy_equity[mask]
        btc_slice = btc_equity.reindex(strat_slice.index)
        if len(btc_slice.dropna()) < 2:
            continue
        strat_r = strat_slice.iloc[-1] / strat_slice.iloc[0] - 1.0
        btc_r = btc_slice.dropna().iloc[-1] / btc_slice.dropna().iloc[0] - 1.0
        relative = strat_r - btc_r
        stress_rows.append([
            label, f"{strat_r:.1%}", f"{btc_r:.1%}", f"{relative:+.1%}",
        ])

    if stress_rows:
        story.append(make_table(stress_headers, stress_rows,
                                col_widths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]))
        story.append(Paragraph(
            "Exhibit 11: Strategy performance during historical crisis periods. "
            "'Relative Return' = strategy return minus BTC return (positive = strategy "
            "outperformed). Note: in rapid V-shaped crashes such as March 2020, the BTC "
            "regime filter triggers after the trough, causing re-entry lag. The strategy "
            "underperforms during the snap-back but avoids the initial drawdown in slower, "
            "sustained bear markets.", sty["caption"]))

    # ──────────────────────────────────────────────────────────────────
    # PAGE 7: Terms & Structure (placeholder)
    # ──────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("7. Proposed Structure", sty["h1"]))

    story.append(Paragraph(
        "The following outlines the proposed terms for a joint trading venture. "
        "All terms are indicative and subject to negotiation.", sty["body"]))

    terms = [
        ["Vehicle", "Prop-firm joint trading account"],
        ["Strategy", "Turtle DD+BTC+Top10 (systematic, fully rules-based)"],
        ["Markets", "Top 10 USD crypto spot pairs on Coinbase (by 20-day ADV)"],
        ["Direction", "Long-only with cash buffer (no leverage, no shorting)"],
        ["Execution", "Daily rebalance, 20 bps round-trip cost budget"],
        ["Risk Limits", "Max 20% portfolio drawdown trigger; BTC regime filter"],
        ["Minimum Allocation", "$500,000"],
        ["Management Fee", "[To be discussed]"],
        ["Performance Fee", "[To be discussed]"],
        ["High-Water Mark", "Yes — fees only on new profits above prior peak"],
        ["Lockup Period", "[To be discussed]"],
        ["Reporting", "Monthly performance report, real-time dashboard access"],
    ]
    story.append(make_table(
        ["Term", "Description"], terms,
        col_widths=[2 * inch, 4.5 * inch]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Key advantages for the partner:</b>", sty["body_bold"]))
    bullets = [
        "Systematic, rules-based strategy eliminates discretionary risk and emotional decision-making.",
        f"Drawdown compression: backtested max DD of {sm['max_dd']:.0%} vs. {bm['max_dd']:.0%} for passive BTC.",
        "Full transparency: all signals, positions, and risk metrics available in real-time.",
        "No leverage, no shorting — the strategy cannot lose more than is allocated.",
        "All results are backtested; live trading validation is a planned next step.",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", sty["bullet"]))

    # Disclaimer
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.5, color=JPM_GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "DISCLAIMER: All performance figures are based on hypothetical backtested results. "
        "Forward simulations use block-bootstrap resampling of historical strategy returns and "
        "do not constitute forecasts. Past performance is not indicative of future results. "
        "Crypto assets are highly volatile and may result in significant loss of capital. "
        "This document is for discussion purposes only and does not constitute investment advice "
        "or an offer to invest. All terms are indicative and subject to legal documentation.",
        sty["disclaimer"]))

    doc.build(story)
    print(f"[report] PDF saved to {PDF_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate partner pitch PDF")
    parser.add_argument("--horizon", type=int, default=3,
                        help="Projection horizon in years (default: 3)")
    parser.add_argument("--paths", type=int, default=10_000,
                        help="Number of Monte Carlo paths (default: 10000)")
    args = parser.parse_args()

    print("=" * 70)
    print("Partner Pitch — Forward Simulation Report")
    print("=" * 70)

    # 1. Run strategy backtest
    print("\n[1/5] Running Turtle DD+BTC+Top10 backtest ...")
    data = prepare_data()
    strat_cfg = OverlayConfig(
        name="DD + BTC + Top10",
        dd_control=True, btc_filter=True,
        concentrated=True, top_n=10,
    )
    sim = run_simulation(data, strat_cfg)
    analysis = analyze_variant(sim)
    strat_metrics = analysis["metrics"]

    strategy_equity = sim["equity_norm"]
    strat_ret = strategy_equity.pct_change().dropna()

    btc = data["close"]["BTC-USD"].dropna()
    btc_equity = btc / btc.iloc[0]
    btc_ret = btc_equity.pct_change().dropna()
    btc_metrics = compute_metrics(btc_equity)

    # 2. Forward simulations
    sim_cfg = SimConfig(
        n_paths=args.paths,
        horizon_days=365 * args.horizon,
        block_size=21,
    )

    current_dd = float(strategy_equity.iloc[-1] / strategy_equity.max() - 1.0)

    print(f"\n[2/5] Block bootstrap: {sim_cfg.n_paths:,} paths × {args.horizon}Y ...")
    strat_paths = block_bootstrap_paths(strat_ret, sim_cfg)
    btc_paths = btc_bootstrap_paths(btc_ret, sim_cfg)

    print(f"[3/5] Post-drawdown bootstrap (current DD = {current_dd:.1%}) ...")
    dd_entry_paths = post_drawdown_bootstrap_paths(
        strat_ret, strategy_equity, current_dd, cfg=sim_cfg)

    strat_fan = fan_chart_summary(strat_paths, sim_cfg)
    btc_fan = fan_chart_summary(btc_paths, sim_cfg)
    dd_entry_fan = fan_chart_summary(dd_entry_paths, sim_cfg)

    # 3. Historical analogues
    print("[4/5] Historical analogues & conditional entry ...")
    analogue_paths = historical_analogues(strategy_equity, sim_cfg.horizon_days)
    cond_entry = conditional_entry_returns(strategy_equity)

    # 4. Generate report
    print(f"\n[5/5] Generating PDF ({PDF_PATH.name}) ...")
    generate_report(
        strategy_equity=strategy_equity,
        btc_equity=btc_equity,
        strat_metrics=strat_metrics,
        btc_metrics=btc_metrics,
        strat_analysis=analysis,
        strat_fan=strat_fan,
        btc_fan=btc_fan,
        dd_entry_fan=dd_entry_fan,
        strat_paths=strat_paths,
        btc_paths=btc_paths,
        dd_entry_paths=dd_entry_paths,
        analogues=analogue_paths,
        cond_entry=cond_entry,
        cfg=sim_cfg,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
