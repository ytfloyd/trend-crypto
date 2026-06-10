#!/usr/bin/env python3
"""
Generate a CTO-level project overview PDF for the trend_crypto platform.

Usage:
    python scripts/generate_cto_report.py
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from reportlab.lib import colors as rl_colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, HRFlowable,
)
from reportlab.lib.colors import HexColor

OUT_DIR = Path("artifacts")

# ── Chart colors ─────────────────────────────────────────────────────
C_BG      = "#0f172a"
C_GRID    = "#334155"
C_TEXT    = "#f1f5f9"
C_BLUE    = "#3b82f6"
C_PURPLE  = "#8b5cf6"
C_GREEN   = "#22c55e"
C_AMBER   = "#f59e0b"
C_RED     = "#ef4444"
C_CYAN    = "#06b6d4"
C_SLATE   = "#94a3b8"
C_TEAL    = "#14b8a6"


def _build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=ss["Title"], fontSize=24, leading=28,
        textColor=HexColor("#1e293b"), spaceAfter=2,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=ss["Normal"], fontSize=11, leading=14,
        textColor=HexColor("#64748b"), spaceAfter=18,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=ss["Heading1"], fontSize=16, leading=20,
        textColor=HexColor("#1e293b"), spaceBefore=18, spaceAfter=8,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontSize=13, leading=16,
        textColor=HexColor("#334155"), spaceBefore=12, spaceAfter=6,
    )
    s["h3"] = ParagraphStyle(
        "h3", parent=ss["Heading3"], fontSize=11, leading=14,
        textColor=HexColor("#475569"), spaceBefore=8, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body", parent=ss["Normal"], fontSize=9.5, leading=13,
        textColor=HexColor("#334155"), alignment=TA_JUSTIFY, spaceAfter=6,
    )
    s["body_sm"] = ParagraphStyle(
        "body_sm", parent=ss["Normal"], fontSize=8.5, leading=11,
        textColor=HexColor("#475569"), spaceAfter=4,
    )
    s["code"] = ParagraphStyle(
        "code", parent=ss["Normal"], fontName="Courier", fontSize=8,
        leading=10, textColor=HexColor("#1e293b"),
        backColor=HexColor("#f1f5f9"), spaceAfter=6,
        leftIndent=12, rightIndent=12,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=ss["Normal"], fontSize=9.5, leading=13,
        textColor=HexColor("#334155"), leftIndent=20,
        bulletIndent=8, spaceAfter=3,
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=ss["Normal"], fontSize=8, leading=10,
        textColor=HexColor("#64748b"), alignment=TA_CENTER, spaceAfter=12,
    )
    s["metric_value"] = ParagraphStyle(
        "metric_value", parent=ss["Normal"], fontSize=20, leading=24,
        textColor=HexColor("#1e293b"), alignment=TA_CENTER,
    )
    s["metric_label"] = ParagraphStyle(
        "metric_label", parent=ss["Normal"], fontSize=8, leading=10,
        textColor=HexColor("#64748b"), alignment=TA_CENTER,
    )
    return s


def _p(sty, key, text):
    return Paragraph(text, sty[key])


def _hr():
    return HRFlowable(
        width="100%", thickness=0.5,
        color=HexColor("#cbd5e1"), spaceAfter=8, spaceBefore=4,
    )


def _make_table(data, col_widths, sty, header=True):
    tbl_data = []
    for row in data:
        tbl_data.append([_p(sty, "body_sm", str(c)) for c in row])
    t = Table(tbl_data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ]
    t.setStyle(TableStyle(style_cmds))
    return t


def _fig_to_image(fig, width=6.5 * inch, ratio=0.55):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=width * ratio)


def _dark_ax(fig, ax, title=""):
    ax.set_facecolor(C_BG)
    fig.set_facecolor(C_BG)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)


def _draw_box(ax, x, y, w, h, label, color, fontsize=8, sublabel=None):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.92,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2 + (0.02 if sublabel else 0),
        label, ha="center", va="center",
        color="white", fontsize=fontsize, fontweight="bold",
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h / 2 - 0.035, sublabel,
            ha="center", va="center", color="#cbd5e1", fontsize=6.5,
        )


def _draw_arrow(ax, x1, y1, x2, y2, color="white"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                        connectionstyle="arc3,rad=0"),
    )


# ═══════════════════════════════════════════════════════════════════════
# DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════

def _chart_platform_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    _dark_ax(fig, ax, "Platform Architecture Overview")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Data Layer (top) ──
    ax.text(0.50, 0.97, "DATA LAYER", ha="center", va="center",
            color=C_SLATE, fontsize=9, fontweight="bold")
    _draw_box(ax, 0.02, 0.84, 0.17, 0.09, "Coinbase API", C_SLATE, 8, "REST + WS")
    _draw_box(ax, 0.23, 0.84, 0.20, 0.09, "Collector", C_CYAN, 8, "1m candles, cron 5m")
    _draw_box(ax, 0.47, 0.84, 0.18, 0.09, "DuckDB", C_BLUE, 9, "bars_1h / bars_1d")
    _draw_box(ax, 0.69, 0.84, 0.16, 0.09, "DataPortal", C_BLUE, 8, "resample + validate")
    _draw_arrow(ax, 0.19, 0.885, 0.23, 0.885)
    _draw_arrow(ax, 0.43, 0.885, 0.47, 0.885)
    _draw_arrow(ax, 0.65, 0.885, 0.69, 0.885)

    # ── Research Layer (middle-top) ──
    ax.text(0.50, 0.78, "RESEARCH LAYER", ha="center", va="center",
            color=C_SLATE, fontsize=9, fontweight="bold")
    _draw_box(ax, 0.02, 0.63, 0.22, 0.12, "Strategy\nDevelopment", C_PURPLE, 8,
              "notebooks + scripts")
    _draw_box(ax, 0.28, 0.63, 0.22, 0.12, "Factor Models", C_PURPLE, 8,
              "101 alphas, ML, AFML")
    _draw_box(ax, 0.54, 0.63, 0.22, 0.12, "Hardened\nBacktester", C_GREEN, 8,
              "Model B timing")
    _draw_box(ax, 0.80, 0.63, 0.18, 0.12, "Tearsheet\nEngine", C_TEAL, 7.5,
              "PDF + HTML")
    _draw_arrow(ax, 0.24, 0.69, 0.28, 0.69, C_PURPLE)
    _draw_arrow(ax, 0.50, 0.69, 0.54, 0.69, C_GREEN)
    _draw_arrow(ax, 0.76, 0.69, 0.80, 0.69, C_TEAL)
    _draw_arrow(ax, 0.60, 0.84, 0.65, 0.75, C_BLUE)

    # ── Core Engine (middle) ──
    ax.text(0.50, 0.57, "CORE ENGINE", ha="center", va="center",
            color=C_SLATE, fontsize=9, fontweight="bold")
    _draw_box(ax, 0.02, 0.42, 0.20, 0.12, "BacktestEngine", C_GREEN, 8,
              "single-asset")
    _draw_box(ax, 0.26, 0.42, 0.22, 0.12, "PortfolioEngine", C_GREEN, 8,
              "multi-asset")
    _draw_box(ax, 0.52, 0.42, 0.22, 0.12, "RiskManager", C_RED, 8,
              "vol target + DD throttle")
    _draw_box(ax, 0.78, 0.42, 0.20, 0.12, "Metrics", C_TEAL, 8,
              "Sharpe, CAGR, DD")
    _draw_arrow(ax, 0.22, 0.48, 0.26, 0.48)
    _draw_arrow(ax, 0.48, 0.48, 0.52, 0.48, C_RED)
    _draw_arrow(ax, 0.74, 0.48, 0.78, 0.48, C_TEAL)

    # ── Execution Layer (lower) ──
    ax.text(0.50, 0.36, "EXECUTION LAYER", ha="center", va="center",
            color=C_SLATE, fontsize=9, fontweight="bold")
    _draw_box(ax, 0.02, 0.21, 0.22, 0.12, "Signal Service", C_AMBER, 8,
              "MedallionSignalService")
    _draw_box(ax, 0.28, 0.21, 0.22, 0.12, "OMS", C_AMBER, 8,
              "weights → orders")
    _draw_box(ax, 0.54, 0.21, 0.22, 0.12, "Broker\nInterface", C_RED, 8,
              "Coinbase / Paper")
    _draw_box(ax, 0.80, 0.21, 0.18, 0.12, "Reconciliation", C_TEAL, 7.5,
              "drift + audit")
    _draw_arrow(ax, 0.24, 0.27, 0.28, 0.27, C_AMBER)
    _draw_arrow(ax, 0.50, 0.27, 0.54, 0.27, C_RED)
    _draw_arrow(ax, 0.76, 0.27, 0.80, 0.27, C_TEAL)

    # ── Monitoring (bottom) ──
    ax.text(0.50, 0.15, "MONITORING & CI", ha="center", va="center",
            color=C_SLATE, fontsize=9, fontweight="bold")
    _draw_box(ax, 0.08, 0.02, 0.20, 0.09, "Alerts", C_RED, 8, "rule-based")
    _draw_box(ax, 0.32, 0.02, 0.20, 0.09, "Metrics", C_TEAL, 8, "JSONL gauges")
    _draw_box(ax, 0.56, 0.02, 0.20, 0.09, "GitHub CI", C_GREEN, 8,
              "mypy + ruff + pytest")
    _draw_box(ax, 0.80, 0.02, 0.18, 0.09, "Dashboard", C_BLUE, 7.5, "HTML tearsheet")

    return fig


def _chart_codebase_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.set_facecolor(C_BG)

    packages = [
        "afml", "data", "backtest", "strategy", "live",
        "monitoring", "common", "execution", "risk",
    ]
    lines = [2561, 2556, 1068, 1048, 730, 602, 598, 479, 120]
    colors = [C_PURPLE, C_BLUE, C_GREEN, C_AMBER, C_RED,
              C_TEAL, C_SLATE, C_CYAN, "#f472b6"]

    ax1.set_facecolor(C_BG)
    bars = ax1.barh(packages, lines, color=colors, edgecolor="none", height=0.65)
    ax1.set_xlabel("Lines of Code", color=C_TEXT, fontsize=8)
    ax1.set_title("src/ Package Size (LOC)", color=C_TEXT, fontsize=10, fontweight="bold")
    ax1.tick_params(colors=C_TEXT, labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color(C_GRID)
    for bar, val in zip(bars, lines):
        ax1.text(bar.get_width() + 40, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", color=C_TEXT, fontsize=7)
    ax1.set_xlim(0, max(lines) * 1.2)
    ax1.invert_yaxis()

    categories = ["src/ core", "tests/", "scripts/\nresearch", "scripts/\noperational",
                   "notebooks/", "configs/"]
    counts = [53, 57, 85, 28, 20, 29]
    cat_colors = [C_GREEN, C_BLUE, C_PURPLE, C_AMBER, C_TEAL, C_SLATE]

    ax2.set_facecolor(C_BG)
    wedges, texts, autotexts = ax2.pie(
        counts, labels=categories, colors=cat_colors, autopct="%1.0f%%",
        textprops={"color": C_TEXT, "fontsize": 7},
        pctdistance=0.78, startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("white")
    ax2.set_title("File Distribution by Category", color=C_TEXT,
                   fontsize=10, fontweight="bold")

    fig.tight_layout(pad=2)
    return fig


def _chart_data_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    _dark_ax(fig, ax, "Data Pipeline — Coinbase to DuckDB")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_box(ax, 0.01, 0.45, 0.16, 0.15, "Coinbase\nAdvanced", C_SLATE, 8, "REST API")
    _draw_box(ax, 0.22, 0.45, 0.16, 0.15, "Collector", C_CYAN, 8, "1m OHLCV")
    _draw_box(ax, 0.43, 0.45, 0.16, 0.15, "DuckDB", C_BLUE, 9, "bars_1m")
    _draw_box(ax, 0.64, 0.45, 0.16, 0.15, "Resample", C_PURPLE, 8, "1h / 4h / 1d")
    _draw_box(ax, 0.85, 0.45, 0.14, 0.15, "Views", C_GREEN, 8, "bars_1h\nbars_1d")

    _draw_arrow(ax, 0.17, 0.525, 0.22, 0.525)
    _draw_arrow(ax, 0.38, 0.525, 0.43, 0.525)
    _draw_arrow(ax, 0.59, 0.525, 0.64, 0.525)
    _draw_arrow(ax, 0.80, 0.525, 0.85, 0.525)

    ax.text(0.09, 0.30, "All Coinbase USD\nproducts collected", ha="center",
            color=C_SLATE, fontsize=7)
    ax.text(0.30, 0.30, "Backfill +\nincremental", ha="center",
            color=C_SLATE, fontsize=7)
    ax.text(0.51, 0.30, "Parquet-backed\ncolumnar store", ha="center",
            color=C_SLATE, fontsize=7)
    ax.text(0.72, 0.30, "OHLCV aggregation\ncoverage filter", ha="center",
            color=C_SLATE, fontsize=7)
    ax.text(0.92, 0.30, "Research +\nLive feeds", ha="center",
            color=C_SLATE, fontsize=7)

    return fig


# ═══════════════════════════════════════════════════════════════════════
# PDF SECTIONS
# ═══════════════════════════════════════════════════════════════════════

def _section_cover(sty):
    ts = datetime.now(timezone.utc).strftime("%B %d, %Y")
    return [
        Spacer(1, 1.5 * inch),
        _p(sty, "title", "Trend Crypto"),
        _p(sty, "subtitle",
           "Systematic Crypto Trading Platform — Technical Overview"),
        Spacer(1, 0.3 * inch),
        _hr(),
        _p(sty, "body",
           f"<b>Prepared for:</b> CTO / Engineering Leadership<br/>"
           f"<b>Date:</b> {ts}<br/>"
           f"<b>Classification:</b> Internal — Confidential"),
        Spacer(1, 0.5 * inch),
        _p(sty, "h2", "Document Purpose"),
        _p(sty, "body",
           "This document provides a comprehensive technical overview of the "
           "trend_crypto platform — a systematic crypto trading research and "
           "execution system built to institutional standards. It covers "
           "architecture, technology stack, backtesting infrastructure, strategy "
           "inventory, quality assurance, and development roadmap."),
        PageBreak(),
    ]


def _section_executive_summary(sty):
    elems = [
        _p(sty, "h1", "1. Executive Summary"),
        _hr(),
        _p(sty, "body",
           "trend_crypto is a Python-based systematic trading platform focused "
           "on cryptocurrency markets. The system spans the full lifecycle from "
           "data collection through signal generation, backtesting, and live "
           "execution. It is designed with the rigor expected of an institutional "
           "quantitative trading operation."),
        Spacer(1, 8),
    ]

    metrics_data = [
        ["<b>9,762</b>", "<b>53</b>", "<b>57</b>", "<b>391</b>", "<b>68</b>"],
        ["Lines of Core Code", "Source Modules", "Test Files", "Test Functions", "Git Commits"],
    ]
    t = Table(metrics_data, colWidths=[1.3 * inch] * 5)
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, 0), 16),
        ("FONTSIZE", (0, 1), (-1, 1), 7),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 1), (-1, 1), HexColor("#64748b")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 1, HexColor("#e2e8f0")),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 12))

    elems.append(_p(sty, "h2", "Key Capabilities"))
    bullets = [
        "<b>Data pipeline</b> — Automated Coinbase data collection (all USD "
        "products), 1-minute granularity, DuckDB storage with resample views.",
        "<b>Hardened backtesting</b> — Model B timing (signal@close, fill@open+1), "
        "transaction costs, funding, slippage, drawdown throttle, deadband filtering.",
        "<b>Multi-asset portfolio engine</b> — Cross-sectional factor models, "
        "inverse-vol sizing, regime gating, per-bar risk management.",
        "<b>Live execution</b> — Signal service, OMS, broker abstraction "
        "(Coinbase live + paper), position reconciliation, safety rails.",
        "<b>Monitoring</b> — JSONL metrics, rule-based alerts, HTML dashboards, "
        "position drift reconciliation.",
        "<b>CI/CD</b> — GitHub Actions: mypy type checking, ruff linting, "
        "strategy registry validation, pytest with coverage.",
    ]
    for b in bullets:
        elems.append(Paragraph(f"• {b}", sty["bullet"]))
    elems.append(PageBreak())
    return elems


def _section_architecture(sty):
    elems = [
        _p(sty, "h1", "2. Platform Architecture"),
        _hr(),
        _p(sty, "body",
           "The platform is organized into clearly separated layers: data "
           "ingestion, research, core engine, execution, and monitoring. Each "
           "layer communicates through well-defined interfaces (Pydantic configs, "
           "Python protocols, JSON contracts)."),
        Spacer(1, 6),
        _fig_to_image(_chart_platform_architecture(), ratio=0.6),
        _p(sty, "caption", "Figure 1: Platform architecture — layered design from data to monitoring"),
    ]
    return elems


def _section_tech_stack(sty):
    elems = [
        _p(sty, "h1", "3. Technology Stack"),
        _hr(),
    ]
    data = [
        ["Layer", "Technology", "Purpose"],
        ["Language", "Python 3.12", "Type hints, pattern matching, performance"],
        ["Data Engine", "Polars + DuckDB", "Columnar analytics, zero-copy, SQL views"],
        ["Configuration", "Pydantic v2", "Typed config validation, YAML loading"],
        ["Visualization", "Matplotlib + Plotly", "Research charts and interactive tearsheets"],
        ["PDF Reports", "ReportLab", "Institutional-quality PDF generation"],
        ["Type Checking", "mypy (strict)", "Static type analysis on src/"],
        ["Linting", "ruff", "Fast Python linter and formatter"],
        ["Testing", "pytest + coverage", "391 tests across 57 files"],
        ["CI/CD", "GitHub Actions", "Push/PR to main: mypy → ruff → registry → pytest"],
        ["Broker", "Coinbase Advanced", "REST API for data + order execution"],
        ["Options Data", "IB TWS/Gateway", "Vol surface snapshots, option chains (optional)"],
        ["ML Libraries", "scikit-learn, statsmodels", "Factor models, vol prediction"],
    ]
    elems.append(_make_table(data, [1.1 * inch, 1.5 * inch, 3.9 * inch], sty))
    elems.append(Spacer(1, 12))
    elems.append(_p(sty, "body",
                     "The stack prioritizes <b>Polars over Pandas</b> in the "
                     "engine layer for performance, while pandas is permitted "
                     "in research notebooks. <b>DuckDB</b> serves as both the "
                     "data warehouse and the live feed source, eliminating the "
                     "need for a separate database server."))
    elems.append(PageBreak())
    return elems


def _section_codebase(sty):
    elems = [
        _p(sty, "h1", "4. Codebase Structure"),
        _hr(),
        _fig_to_image(_chart_codebase_breakdown(), ratio=0.4),
        _p(sty, "caption",
           "Figure 2: Core library size by package (left) and file distribution (right)"),
        Spacer(1, 6),
    ]

    data = [
        ["Package", "Files", "LOC", "Responsibility"],
        ["src/afml/", "11", "2,561",
         "AFML research toolkit (Lopez de Prado): bars, labeling, fracdiff, "
         "microstructure, HRP, CV, feature importance"],
        ["src/data/", "8", "2,556",
         "Coinbase collector, DuckDB portal, universe filters, live feed, "
         "options data (IB)"],
        ["src/backtest/", "8", "1,068",
         "BacktestEngine (single-asset), PortfolioEngine (multi-asset), "
         "impact model, metrics, validators"],
        ["src/strategy/", "8", "1,048",
         "Strategy protocols (ABC + Protocol), MA crossover, Medallion Lite V3, "
         "Carver forecast, buy-and-hold"],
        ["src/live/", "2", "730",
         "LiveRunner (data→signal→risk→orders), MedallionSignalService"],
        ["src/monitoring/", "4", "602",
         "Alerts, JSONL metrics, HTML dashboard, position reconciliation"],
        ["src/common/", "5", "598",
         "Pydantic configs, metric helpers, logging, timeframe utils, hashing"],
        ["src/execution/", "5", "479",
         "OMS (weights→orders), broker interface, paper broker, execution sim"],
        ["src/risk/", "2", "120",
         "Vol targeting, portfolio risk caps"],
    ]
    elems.append(_make_table(
        data, [1.0 * inch, 0.4 * inch, 0.5 * inch, 4.6 * inch], sty))
    elems.append(Spacer(1, 8))
    elems.append(_p(sty, "body",
                     "<b>Total core library: 53 modules, 9,762 lines.</b> "
                     "The codebase maintains a clean separation between "
                     "library code (src/) and research scripts (scripts/research/). "
                     "All strategy backtests run through the hardened engine — "
                     "ad hoc backtesting code has been systematically removed."))
    elems.append(PageBreak())
    return elems


def _section_backtest_engine(sty):
    elems = [
        _p(sty, "h1", "5. Backtesting Infrastructure"),
        _hr(),
        _p(sty, "body",
           "The backtesting engine is the foundation of all research. It enforces "
           "strict execution timing to prevent lookahead bias and ensure "
           "reproducibility. Two engines exist for different use cases:"),
        Spacer(1, 4),
        _p(sty, "h2", "5.1 Execution Model — Model B"),
        _p(sty, "body",
           "All backtests use <b>Model B timing</b>: signals are computed at "
           "bar close using only data available through that bar, and trades "
           "execute at the next bar's open with configurable lag. PnL attribution "
           "uses open-to-close returns, correctly reflecting that execution "
           "happens at open, not at the prior close."),
    ]

    data = [
        ["Property", "BacktestEngine", "PortfolioEngine"],
        ["Scope", "Single asset (e.g. BTC-USD)", "Multi-asset cross-section"],
        ["Strategy Interface", "TargetWeightStrategy ABC", "PortfolioStrategy Protocol"],
        ["Risk Management", "RiskManager (vol target)", "PortfolioRiskManager (vol + caps)"],
        ["Position Sizing", "Per-asset vol targeting", "Inverse-vol + regime scaling"],
        ["Cost Model", "Fee bps + slippage bps + funding", "Same, per asset"],
        ["Safety Rails", "Deadband filter, DD throttle", "Same + max weight cap"],
        ["Output", "Equity DF, summary dict", "PortfolioResult (equity, weights, trades)"],
    ]
    elems.append(_make_table(data, [1.3 * inch, 2.6 * inch, 2.6 * inch], sty))
    elems.append(Spacer(1, 8))

    elems.append(_p(sty, "h2", "5.2 Shared Execution Logic"))
    elems.append(_p(sty, "body",
                     "Common execution primitives are centralized in a shared "
                     "module (<font face='Courier' size='8'>src/backtest/_execution.py</font>) "
                     "to eliminate duplication between engines:"))
    funcs = [
        "<b>apply_deadband()</b> — Suppresses small rebalances below a "
        "configurable threshold to reduce turnover.",
        "<b>apply_dd_throttle()</b> — Scales exposure when drawdown exceeds "
        "a configurable limit, with a floor to prevent full de-risking.",
        "<b>compute_summary_stats()</b> — Standardized performance statistics "
        "(Sharpe, CAGR, max drawdown, etc.) from equity curves.",
    ]
    for f in funcs:
        elems.append(Paragraph(f"• {f}", sty["bullet"]))

    elems.append(Spacer(1, 8))
    elems.append(_p(sty, "h2", "5.3 Data Validation"))
    elems.append(_p(sty, "body",
                     "The engine validates all input data before execution: "
                     "OHLCV column presence, sorted timestamps, non-negative "
                     "prices/volume, and temporal integrity. Strict validation "
                     "mode (configurable) adds additional checks for bar "
                     "completeness and gap detection."))
    elems.append(PageBreak())
    return elems


def _section_data_pipeline(sty):
    elems = [
        _p(sty, "h1", "6. Data Pipeline"),
        _hr(),
        _fig_to_image(_chart_data_pipeline(), ratio=0.35),
        _p(sty, "caption", "Figure 3: Data flow from exchange to research/live feeds"),
        Spacer(1, 6),
    ]

    elems.append(_p(sty, "h2", "6.1 Collection"))
    elems.append(_p(sty, "body",
                     "The <b>CoinbaseCollector</b> (1,021 LOC) fetches 1-minute "
                     "OHLCV candles for all Coinbase USD-quoted products. It "
                     "supports both historical backfill and incremental updates, "
                     "designed to run via cron every 5 minutes. Data is stored "
                     "in DuckDB with Parquet-backed columnar storage."))

    elems.append(_p(sty, "h2", "6.2 Universe Management"))
    elems.append(_p(sty, "body",
                     "Universe construction filters Coinbase products by: "
                     "USD quote currency, stablecoin base exclusion, and average "
                     "daily volume (ADV) thresholds. Multiple pre-built views "
                     "serve different research needs (full universe, ADV>$10M, "
                     "Top-50 by volume)."))

    elems.append(_p(sty, "h2", "6.3 Resampling"))
    elems.append(_p(sty, "body",
                     "The DataPortal resamples native 1-minute bars to any "
                     "integer-multiple timeframe (1h, 4h, 1d). Aggregation "
                     "rules: open=first, high=max, low=min, close=last, "
                     "volume=sum. Incomplete bucket coverage below 80% is "
                     "dropped to prevent stale data artifacts."))
    elems.append(PageBreak())
    return elems


def _section_strategies(sty):
    elems = [
        _p(sty, "h1", "7. Strategy Inventory"),
        _hr(),
        _p(sty, "body",
           "All strategies implement either the <b>TargetWeightStrategy</b> "
           "ABC (single-asset) or the <b>PortfolioStrategy</b> Protocol "
           "(multi-asset) and run exclusively through the hardened engines."),
        Spacer(1, 6),
    ]

    data = [
        ["Strategy", "Type", "Status", "Key Features"],
        ["Medallion Lite V3", "PortfolioStrategy", "Live-ready",
         "15 cross-sectional factors, tiered core/satellite, "
         "ensemble regime gating, inverse-vol sizing"],
        ["Sornette LPPLS", "Research", "Research",
         "Log-periodic power law bubble detection, hourly bars, "
         "validated against simplicity benchmark"],
        ["101 Alphas Ensemble", "Cross-sectional", "Research",
         "Formulaic alpha factory, IC-based selection, "
         "danger-regime gating, capacity analysis"],
        ["MA Crossover", "TargetWeightStrategy", "Production",
         "Long-only, optional ADX filter, vol targeting, "
         "configurable fast/slow windows"],
        ["Growth Sleeve v1.5", "Cross-sectional", "Research",
         "Top-50 ADV>$10M universe, multi-indicator ensemble, "
         "trend + momentum factors"],
    ]
    elems.append(_make_table(
        data, [1.2 * inch, 1.1 * inch, 0.8 * inch, 3.4 * inch], sty))
    elems.append(Spacer(1, 12))

    elems.append(_p(sty, "h2", "7.1 Flagship: Medallion Lite V3"))
    elems.append(_p(sty, "body",
                     "The flagship strategy is a cross-sectional momentum system "
                     "with ensemble regime gating. It trades the full Coinbase USD "
                     "universe (~90 tokens) with event-driven entry/exit logic."))
    features = [
        "<b>Regime Model</b> — 4-component continuous [0,1] score: BTC trend "
        "(SMA), cross-sectional breadth, BTC volatility compression, BTC momentum.",
        "<b>Factor Model (V3)</b> — 15 adaptive factors including idiosyncratic "
        "momentum, liquidity improvement, relative strength breadth, volatility "
        "contraction, and multi-timeframe agreement. Factors are ranked within "
        "liquidity tiers.",
        "<b>Portfolio Construction</b> — Tiered core/satellite: core positions "
        "(top quintile) get larger allocations; satellite positions (2nd quintile) "
        "are smaller. Inverse-volatility weighted, regime-scaled, 10% per-name cap.",
        "<b>Risk Controls</b> — Trailing stop (15%), max hold period (336h), "
        "factor degradation exit, regime collapse exit, drawdown throttle.",
    ]
    for f in features:
        elems.append(Paragraph(f"• {f}", sty["bullet"]))
    elems.append(PageBreak())
    return elems


def _section_execution(sty):
    elems = [
        _p(sty, "h1", "8. Live Execution System"),
        _hr(),
        _p(sty, "body",
           "The execution layer translates research signals into live orders "
           "through a modular, safety-first architecture."),
        Spacer(1, 6),
        _p(sty, "h2", "8.1 Signal Service"),
        _p(sty, "body",
           "The <b>MedallionSignalService</b> (550 LOC) runs independently of "
           "the execution engine. It loads the latest OHLCV from DuckDB, "
           "computes the ensemble regime score and factor model, evaluates "
           "entries/exits, and publishes target weights via a JSON contract. "
           "State is persisted in <font face='Courier' size='8'>medallion_state.json</font> "
           "for crash recovery."),
        _p(sty, "h2", "8.2 Order Management System"),
        _p(sty, "body",
           "The OMS converts target weights into orders by diffing against "
           "current positions. It supports deadband filtering (suppress small "
           "rebalances), idempotency checks, and post-fill reconciliation. "
           "Orders route through the broker abstraction layer."),
        _p(sty, "h2", "8.3 Broker Interface"),
    ]
    data = [
        ["Mode", "Broker", "Fills", "Use Case"],
        ["Live", "Coinbase Advanced", "Real exchange fills", "Production trading"],
        ["Paper", "PaperBroker", "Instant at given price + slippage",
         "Strategy validation"],
        ["Backtest", "ExecutionSim", "Simulated with fee/slippage bps",
         "Historical research"],
    ]
    elems.append(_make_table(data, [0.8 * inch, 1.5 * inch, 2.0 * inch, 2.2 * inch], sty))
    elems.append(Spacer(1, 8))

    elems.append(_p(sty, "h2", "8.4 Deployment Modes"))
    modes = [
        "<b>Cron (recommended)</b> — Single-cycle execution via "
        "<font face='Courier' size='8'>python scripts/run_medallion_live.py --mode once</font>. "
        "Stateless, idempotent, easy to monitor.",
        "<b>Daemon</b> — Continuous loop with configurable interval. "
        "Suitable for low-latency requirements.",
        "<b>Paper</b> — Full integration test using PaperBroker. "
        "Validates signal → OMS → reconciliation pipeline.",
    ]
    for m in modes:
        elems.append(Paragraph(f"• {m}", sty["bullet"]))
    elems.append(PageBreak())
    return elems


def _section_research(sty):
    elems = [
        _p(sty, "h1", "9. Research Framework"),
        _hr(),
        _p(sty, "body",
           "Research is organized into dedicated subdirectories under "
           "<font face='Courier' size='8'>scripts/research/</font>, with "
           "Jupyter notebooks for exploratory work and Python scripts for "
           "reproducible pipelines."),
        Spacer(1, 6),
    ]

    data = [
        ["Area", "Location", "Description"],
        ["Medallion Lite", "medallion_lite/",
         "Flagship strategy: factors V1-V3, regime ensemble, tiered portfolio, "
         "hardened runner"],
        ["Sornette LPPLS", "sornette_lppl/",
         "Bubble detection via log-periodic power law fitting, hourly + daily"],
        ["101 Alphas", "Top-level research scripts",
         "Formulaic alpha factory, IC testing, selection, ensemble, tearsheets"],
        ["JPM Momentum", "jpm_momentum/",
         "Momentum signals (RET, MAC, EMAC, BRK, LREG, RADJ), "
         "crypto + ETF universe"],
        ["AFML Toolkit", "src/afml/ + notebooks/afml/",
         "Lopez de Prado: alternative bars, triple-barrier, fracdiff, "
         "microstructure, HRP, CPCV"],
        ["Alpha Lab", "alpha_lab/",
         "Parameterized signal factory + on-chain data integration"],
        ["Paper Pipeline", "paper_pipeline/",
         "arXiv/SSRN paper discovery → methodology audit → strategy extraction"],
        ["TA-Lib Scanner", "talib_scanner/",
         "95 TA-Lib features, IC-based scan, conditional-return edge analysis"],
        ["ETF Data", "etf_data/",
         "Tiingo REST API, ETF universe definitions, DuckDB ingestion"],
    ]
    elems.append(_make_table(
        data, [1.1 * inch, 1.3 * inch, 4.1 * inch], sty))
    elems.append(Spacer(1, 10))

    elems.append(_p(sty, "h2", "9.1 Notebooks"))
    elems.append(_p(sty, "body",
                     "20 Jupyter notebooks cover: data exploration, turtle trading, "
                     "logistic regression filters, Bayesian strategy evaluation, "
                     "vol estimators, AHL pure momentum, cross-rate mean reversion, "
                     "realized vol prediction, and the full AFML curriculum "
                     "(Ch. 1-10: bars, labeling, fracdiff, CV, feature importance, "
                     "bet sizing, structural breaks, microstructure, backtest dangers, "
                     "portfolio construction)."))

    elems.append(_p(sty, "h2", "9.2 Strategy Registry"))
    elems.append(_p(sty, "body",
                     "A JSON-based strategy registry tracks canonical metrics, "
                     "equity curves, tearsheets, and run recipes for all strategies. "
                     "The registry is validated in CI to ensure consistency."))
    elems.append(PageBreak())
    return elems


def _section_quality(sty):
    elems = [
        _p(sty, "h1", "10. Quality Assurance"),
        _hr(),
        _p(sty, "h2", "10.1 Continuous Integration"),
        _p(sty, "body",
           "Every push and pull request to <b>main</b> triggers the full "
           "CI pipeline on GitHub Actions:"),
    ]
    steps = [
        "<b>mypy</b> — Static type checking on all src/ modules "
        "(strict mode, ignore-missing-imports).",
        "<b>ruff</b> — Fast Python linting on src/ and tests/. "
        "Line length 100, standard rule set.",
        "<b>Strategy Registry Validation</b> — Ensures all registered "
        "strategies have valid configs and referenced artifacts.",
        "<b>pytest</b> — 391 tests with coverage reporting (XML output).",
    ]
    for s in steps:
        elems.append(Paragraph(f"• {s}", sty["bullet"]))

    elems.append(Spacer(1, 8))
    elems.append(_p(sty, "h2", "10.2 Test Coverage"))

    data = [
        ["Category", "Files", "Coverage"],
        ["AFML (Lopez de Prado)", "11", "Bars, labeling, fracdiff, microstructure, HRP, CV"],
        ["Engine / Backtest", "4", "Cost model, portfolio engine, regressions"],
        ["Registry / Tearsheet", "6", "Validation, runs, input resolution, policy"],
        ["Alpha Factory", "3", "Table resolution, tearsheet generation"],
        ["Formulaic Alphas", "3", "Parser, warmup, two-stage pipeline"],
        ["Benchmarks", "3", "Performance, alignment, timezone handling"],
        ["Monitoring / Live", "2", "Alert rules, live runner integration"],
        ["Metrics / Stats", "4", "Period compat, summary stats, consistency"],
        ["Other", "21", "Signals, timing, resampling, impact, purged CV"],
    ]
    elems.append(_make_table(
        data, [1.5 * inch, 0.5 * inch, 4.5 * inch], sty))

    elems.append(Spacer(1, 10))
    elems.append(_p(sty, "h2", "10.3 Recent Architectural Cleanup"))
    elems.append(_p(sty, "body",
                     "A comprehensive audit identified and removed all ad hoc "
                     "backtesting code throughout the project. This cleanup:"))
    cleanup = [
        "Deleted ~105 files containing hand-rolled backtesting logic that "
        "bypassed the hardened engine.",
        "Consolidated shared execution primitives (deadband, DD throttle, "
        "summary stats) into a single module.",
        "Removed 9 unused risk modules, 12 orphan tearsheet scripts, "
        "and 1 dead strategy file.",
        "Reduced net codebase by ~60 files and 66,800 lines.",
        "Verified: all tests pass, ruff clean, mypy clean post-cleanup.",
    ]
    for c in cleanup:
        elems.append(Paragraph(f"• {c}", sty["bullet"]))
    elems.append(PageBreak())
    return elems


def _section_config(sty):
    elems = [
        _p(sty, "h1", "11. Configuration System"),
        _hr(),
        _p(sty, "body",
           "All backtests and live runs are driven by YAML configuration files "
           "validated through Pydantic models. This ensures reproducibility "
           "and prevents configuration drift."),
        Spacer(1, 6),
    ]

    data = [
        ["Config Section", "Key Fields", "Purpose"],
        ["data", "db_path, table, symbol, start, end, timeframe",
         "Data source and date range"],
        ["engine", "strict_validation, lookback, initial_cash",
         "Engine behavior and capital"],
        ["strategy", "fast_window, slow_window, vol_window, mode",
         "Strategy parameters"],
        ["risk", "vol_window, target_vol_annual, max_weight",
         "Risk overlay parameters"],
        ["execution", "fee_bps, slippage_bps, deadband, cooldown, lag_bars",
         "Transaction cost model"],
    ]
    elems.append(_make_table(
        data, [1.1 * inch, 2.5 * inch, 2.9 * inch], sty))
    elems.append(Spacer(1, 8))
    elems.append(_p(sty, "body",
                     "<b>29 YAML configs</b> exist across runs/ (26 backtest "
                     "configs for BTC/ETH variants) and research/ (3 research "
                     "configs). Each run produces a deterministic artifact "
                     "directory containing equity curves, positions, trades, "
                     "and summary statistics."))
    elems.append(PageBreak())
    return elems


def _section_roadmap(sty):
    elems = [
        _p(sty, "h1", "12. Roadmap & Open Items"),
        _hr(),
        _p(sty, "h2", "12.1 Near-Term (0–3 months)"),
    ]
    near = [
        "<b>Live deployment of Medallion Lite V3</b> — Paper trading validation, "
        "then production with cron-based signal generation.",
        "<b>Expand universe coverage</b> — Integrate additional exchanges "
        "(Binance, Bybit) for broader coverage and cross-venue arbitrage data.",
        "<b>Funding rate integration</b> — Perpetual futures funding rates "
        "as a live signal component and carry cost in backtests.",
        "<b>Factor model iteration</b> — Continue V4 factor development with "
        "on-chain data integration and alternative data sources.",
    ]
    for n in near:
        elems.append(Paragraph(f"• {n}", sty["bullet"]))

    elems.append(_p(sty, "h2", "12.2 Medium-Term (3–6 months)"))
    mid = [
        "<b>Multi-strategy portfolio</b> — Combine Medallion Lite with "
        "mean-reversion, LPPLS, and carry strategies in a risk-parity wrapper.",
        "<b>Advanced execution</b> — TWAP/VWAP algorithms, smart order "
        "routing, market impact minimization.",
        "<b>Real-time monitoring</b> — Grafana/Prometheus integration for "
        "live strategy performance, fill quality, and system health.",
        "<b>Walk-forward optimization</b> — Automated parameter re-fitting "
        "with purged cross-validation (AFML CPCV).",
    ]
    for m in mid:
        elems.append(Paragraph(f"• {m}", sty["bullet"]))

    elems.append(_p(sty, "h2", "12.3 Long-Term (6–12 months)"))
    long_term = [
        "<b>Multi-asset class expansion</b> — Equities, futures, options "
        "using existing IB integration infrastructure.",
        "<b>ML pipeline</b> — Feature store, model registry, automated "
        "retraining with AFML-compliant validation.",
        "<b>High-frequency layer</b> — Sub-minute signals using websocket "
        "feeds and in-memory order book analysis.",
        "<b>Team scaling</b> — API documentation, contributor guidelines, "
        "strategy development SDK for new researchers.",
    ]
    for lt in long_term:
        elems.append(Paragraph(f"• {lt}", sty["bullet"]))

    elems.append(Spacer(1, 16))
    elems.append(_hr())
    elems.append(_p(sty, "body_sm",
                     "This document was auto-generated from the trend_crypto codebase. "
                     "For questions, contact the platform engineering team."))
    return elems


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "trend_crypto_cto_report.pdf"

    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )

    sty = _build_styles()
    story = []

    story += _section_cover(sty)
    story += _section_executive_summary(sty)
    story += _section_architecture(sty)
    story += _section_tech_stack(sty)
    story += _section_codebase(sty)
    story += _section_backtest_engine(sty)
    story += _section_data_pipeline(sty)
    story += _section_strategies(sty)
    story += _section_execution(sty)
    story += _section_research(sty)
    story += _section_quality(sty)
    story += _section_config(sty)
    story += _section_roadmap(sty)

    doc.build(story)
    print(f"CTO report written to {out_path}")


if __name__ == "__main__":
    main()
