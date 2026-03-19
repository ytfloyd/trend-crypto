#!/usr/bin/env python3
"""
Generate execution architecture PDF for Medallion Lite strategy.

Covers system architecture, signal contract, execution flow,
timing, safety rails, and deployment options.

Usage:
    python -m scripts.research.medallion_lite.generate_execution_pdf
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

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

OUT_DIR = Path(__file__).resolve().parent / "output"

# ── Colors ───────────────────────────────────────────────────────────
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

# ── Reportlab Styles ────────────────────────────────────────────────

def _build_styles():
    ss = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "title", parent=ss["Title"], fontSize=22, leading=26,
        textColor=HexColor("#1e293b"), spaceAfter=4,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", parent=ss["Normal"], fontSize=11, leading=14,
        textColor=HexColor("#64748b"), spaceAfter=18,
    )
    styles["h1"] = ParagraphStyle(
        "h1", parent=ss["Heading1"], fontSize=16, leading=20,
        textColor=HexColor("#1e293b"), spaceBefore=16, spaceAfter=8,
    )
    styles["h2"] = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontSize=13, leading=16,
        textColor=HexColor("#334155"), spaceBefore=12, spaceAfter=6,
    )
    styles["body"] = ParagraphStyle(
        "body", parent=ss["Normal"], fontSize=9.5, leading=13,
        textColor=HexColor("#334155"), alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    styles["body_sm"] = ParagraphStyle(
        "body_sm", parent=ss["Normal"], fontSize=8.5, leading=11,
        textColor=HexColor("#475569"), spaceAfter=4,
    )
    styles["code"] = ParagraphStyle(
        "code", parent=ss["Normal"], fontName="Courier", fontSize=8,
        leading=10, textColor=HexColor("#1e293b"),
        backColor=HexColor("#f1f5f9"), spaceAfter=6,
        leftIndent=12, rightIndent=12,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", parent=ss["Normal"], fontSize=9.5, leading=13,
        textColor=HexColor("#334155"), leftIndent=20,
        bulletIndent=8, spaceAfter=3,
    )
    styles["caption"] = ParagraphStyle(
        "caption", parent=ss["Normal"], fontSize=8, leading=10,
        textColor=HexColor("#64748b"), alignment=TA_CENTER,
        spaceAfter=12,
    )
    return styles


def _p(styles, key, text):
    return Paragraph(text, styles[key])


def _hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=HexColor("#cbd5e1"), spaceAfter=8, spaceBefore=4)


def _desc(styles, txt):
    return Paragraph(txt, styles["body_sm"])


def _make_table(data, col_widths, styles):
    """Build a styled table with header row."""
    tbl_data = []
    for row in data:
        tbl_data.append([_desc(styles, str(c)) for c in row])
    t = Table(tbl_data, colWidths=col_widths, repeatRows=1)
    header_bg = HexColor("#1e293b")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _fig_to_image(fig, width=6.5 * inch):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=width, height=width * 0.55)
    return img


# ── Diagram Generators ──────────────────────────────────────────────

def _dark_ax(fig, ax, title=""):
    ax.set_facecolor(C_BG)
    fig.set_facecolor(C_BG)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)


def _draw_box(ax, x, y, w, h, label, color, fontsize=8, sublabel=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor="white",
                         linewidth=1.2, alpha=0.92)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2 + (0.02 if sublabel else 0),
            label, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.035, sublabel,
                ha="center", va="center",
                color="#cbd5e1", fontsize=6.5)


def _draw_arrow(ax, x1, y1, x2, y2, color="white"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, connectionstyle="arc3,rad=0"))


def _chart_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _dark_ax(fig, ax, "Medallion Lite — System Architecture")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Data layer (top)
    _draw_box(ax, 0.02, 0.82, 0.18, 0.10, "Coinbase API", C_SLATE, 8)
    _draw_box(ax, 0.28, 0.82, 0.20, 0.10, "Collector (cron)", C_CYAN, 8, "every 5 min")
    _draw_box(ax, 0.56, 0.82, 0.20, 0.10, "DuckDB", C_BLUE, 9, "bars_1h")
    _draw_arrow(ax, 0.20, 0.87, 0.28, 0.87)
    _draw_arrow(ax, 0.48, 0.87, 0.56, 0.87)
    ax.text(0.50, 0.95, "DATA LAYER", ha="center", va="center",
            color=C_SLATE, fontsize=9, fontweight="bold")

    # Signal service (middle)
    _draw_box(ax, 0.10, 0.45, 0.50, 0.25, "", C_PURPLE, 9)
    ax.text(0.35, 0.67, "SIGNAL SERVICE", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")
    steps = [
        "1. Load 5000h OHLCV from DuckDB",
        "2. Compute ensemble regime score",
        "3. Cross-sectional factor ranking",
        "4. Evaluate exits (regime/stop/factor/maxhold)",
        "5. Evaluate entries (score > 0.65, regime > 0.45)",
        "6. Inverse-vol sizing, regime scaling",
        "7. Publish target weights",
    ]
    for i, s in enumerate(steps):
        ax.text(0.15, 0.625 - i * 0.028, s, ha="left", va="center",
                color="#e2e8f0", fontsize=6.5, fontfamily="monospace")

    # State files
    _draw_box(ax, 0.66, 0.55, 0.18, 0.08, "State JSON", C_AMBER, 7.5, "medallion_state.json")
    _draw_box(ax, 0.66, 0.45, 0.18, 0.08, "DuckDB Log", C_BLUE, 7.5, "live_signals table")
    _draw_arrow(ax, 0.60, 0.59, 0.66, 0.59, C_AMBER)
    _draw_arrow(ax, 0.60, 0.49, 0.66, 0.49, C_BLUE)

    # Arrow from DuckDB down to signal service
    _draw_arrow(ax, 0.66, 0.82, 0.35, 0.70)

    # Signal contract (middle output)
    _draw_box(ax, 0.25, 0.25, 0.22, 0.10, "signal_output.json", C_GREEN, 8, "THE CONTRACT")
    _draw_arrow(ax, 0.35, 0.45, 0.36, 0.35, C_GREEN)

    # Execution engine (bottom)
    _draw_box(ax, 0.10, 0.03, 0.50, 0.15, "", C_RED, 9)
    ax.text(0.35, 0.155, "EXECUTION ENGINE", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")
    exec_steps = [
        "1. Read signal_output.json",
        "2. Safety checks (stale, idempotency)",
        "3. Diff target vs current weights",
        "4. Generate orders (deadband filter)",
        "5. TWAP/VWAP on Coinbase",
        "6. Post-fill reconciliation",
    ]
    for i, s in enumerate(exec_steps):
        ax.text(0.15, 0.13 - i * 0.017, s, ha="left", va="center",
                color="#fecaca", fontsize=6.5, fontfamily="monospace")

    _draw_arrow(ax, 0.36, 0.25, 0.35, 0.18, C_GREEN)

    # Exchange
    _draw_box(ax, 0.68, 0.06, 0.18, 0.09, "Coinbase", C_SLATE, 8, "Order Router")
    _draw_arrow(ax, 0.60, 0.10, 0.68, 0.10)

    return fig


def _chart_timing():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    _dark_ax(fig, ax, "Hourly Execution Timeline")
    ax.set_xlim(-2, 62)
    ax.set_ylim(-0.5, 4)
    ax.axis("off")

    segments = [
        (0, 5, "Collector\nsyncs", C_CYAN, 3.0),
        (5, 5, "Signal\ncompute", C_PURPLE, 3.0),
        (10, 45, "TWAP execution window", C_RED, 3.0),
        (55, 5, "Reconcile", C_AMBER, 3.0),
    ]

    for start, dur, label, color, y in segments:
        rect = FancyBboxPatch((start, y - 0.35), dur, 0.7,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor="white",
                              linewidth=1, alpha=0.85)
        ax.add_patch(rect)
        ax.text(start + dur / 2, y, label, ha="center", va="center",
                color="white", fontsize=7.5, fontweight="bold")

    for m in [0, 5, 10, 55, 60]:
        ax.plot([m, m], [1.8, 2.4], color=C_SLATE, lw=0.8, ls="--")
        ax.text(m, 1.6, f":{m:02d}", ha="center", va="center",
                color=C_TEXT, fontsize=8, fontfamily="monospace")

    ax.text(30, 1.0, "← hour boundary                                                   hour boundary →",
            ha="center", va="center", color=C_SLATE, fontsize=7)

    # Signal file write marker
    ax.annotate("signal_output.json\nwritten", xy=(10, 2.65), xytext=(10, 0.5),
                ha="center", color=C_GREEN, fontsize=7,
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.2))

    return fig


def _chart_signal_cycle():
    fig, ax = plt.subplots(figsize=(10, 5))
    _dark_ax(fig, ax, "Signal Service — Decision Flow")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Start
    _draw_box(ax, 0.40, 0.90, 0.20, 0.07, "run_cycle()", C_BLUE, 9)

    # Load data
    _draw_box(ax, 0.40, 0.78, 0.20, 0.07, "Load OHLCV", C_CYAN, 8, "5000h from DuckDB")
    _draw_arrow(ax, 0.50, 0.90, 0.50, 0.85)

    # Freshness check
    _draw_box(ax, 0.40, 0.66, 0.20, 0.07, "Data fresh?", C_AMBER, 8, "< 3h staleness")
    _draw_arrow(ax, 0.50, 0.78, 0.50, 0.73)

    # Stale branch
    _draw_box(ax, 0.72, 0.66, 0.18, 0.07, "Publish stale=true", C_RED, 7.5, "DO NOT TRADE")
    ax.text(0.65, 0.695, "NO", ha="center", va="center", color=C_RED, fontsize=7, fontweight="bold")
    _draw_arrow(ax, 0.60, 0.695, 0.72, 0.695, C_RED)

    # Regime
    _draw_box(ax, 0.40, 0.54, 0.20, 0.07, "Ensemble Regime", C_PURPLE, 8, "4-component score")
    ax.text(0.38, 0.635, "YES", ha="center", va="center", color=C_GREEN, fontsize=7, fontweight="bold")
    _draw_arrow(ax, 0.50, 0.66, 0.50, 0.61, C_GREEN)

    # Emergency flat branch
    _draw_box(ax, 0.72, 0.54, 0.18, 0.07, "Emergency flat", C_RED, 7.5, "all weights → 0")
    ax.text(0.65, 0.575, "< 0.15", ha="center", va="center", color=C_RED, fontsize=7, fontweight="bold")
    _draw_arrow(ax, 0.60, 0.575, 0.72, 0.575, C_RED)

    # Exits
    _draw_box(ax, 0.40, 0.42, 0.20, 0.07, "Evaluate Exits", C_RED, 8, "stop/factor/maxhold")
    _draw_arrow(ax, 0.50, 0.54, 0.50, 0.49)

    # Exit types on the side
    exits = ["Trailing stop (−15%)", "Factor < 0.40", "Max hold (336h)", "Regime collapse"]
    for i, e in enumerate(exits):
        ax.text(0.08, 0.47 - i * 0.025, f"• {e}", ha="left", va="center",
                color=C_SLATE, fontsize=6.5)

    # Entries
    _draw_box(ax, 0.40, 0.30, 0.20, 0.07, "Evaluate Entries", C_GREEN, 8, "every 24h rebalance")
    _draw_arrow(ax, 0.50, 0.42, 0.50, 0.37)

    entry_rules = ["Score > 0.65", "Regime > 0.45", "Max 25 positions", "10% per-name cap"]
    for i, e in enumerate(entry_rules):
        ax.text(0.08, 0.35 - i * 0.025, f"• {e}", ha="left", va="center",
                color=C_SLATE, fontsize=6.5)

    # Sizing
    _draw_box(ax, 0.40, 0.18, 0.20, 0.07, "Position Sizing", C_BLUE, 8, "inv-vol × regime")
    _draw_arrow(ax, 0.50, 0.30, 0.50, 0.25)

    # Publish
    _draw_box(ax, 0.40, 0.05, 0.20, 0.07, "Publish Weights", C_GREEN, 8, "→ signal_output.json")
    _draw_arrow(ax, 0.50, 0.18, 0.50, 0.12)

    return fig


def _chart_contract_schema():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _dark_ax(fig, ax, "Signal Output Contract — JSON Schema")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    json_lines = [
        ('{', C_TEXT),
        ('  "ts": "2026-03-17T14:00:00+00:00",', C_SLATE),
        ('  "cycle_id": "a1b2c3d4-...",', C_SLATE),
        ('  "target_weights": {', C_GREEN),
        ('    "BTC-USD": 0.082,', C_GREEN),
        ('    "ETH-USD": 0.095,', C_GREEN),
        ('    "SOL-USD": 0.078,', C_GREEN),
        ('    "AVAX-USD": 0.065,', C_GREEN),
        ('    "DOGE-USD": 0.0', C_GREEN),
        ('  },', C_GREEN),
        ('  "regime_score": 0.72,', C_PURPLE),
        ('  "actions": [ ... ],', C_AMBER),
        ('  "diagnostics": { ... },', C_CYAN),
        ('  "stale": false', C_RED),
        ('}', C_TEXT),
    ]

    x_json = 0.05
    y_start = 0.90
    for i, (line, color) in enumerate(json_lines):
        ax.text(x_json, y_start - i * 0.048, line, ha="left", va="center",
                color=color, fontsize=9, fontfamily="monospace", fontweight="bold")

    annotations = [
        (0.55, 0.85, "UTC timestamp of computation", C_SLATE),
        (0.55, 0.80, "Idempotency key — never execute same cycle twice", C_SLATE),
        (0.55, 0.68, "THE KEY OUTPUT — target portfolio weight per symbol\nSum ≤ 1.0. Zero = close position.", C_GREEN),
        (0.55, 0.48, "Market regime probability [0, 1]\n< 0.15 = emergency flat", C_PURPLE),
        (0.55, 0.42, "What changed (entries/exits) — for audit trail", C_AMBER),
        (0.55, 0.36, "Health metrics: n_holdings, gross_exposure, etc.", C_CYAN),
        (0.55, 0.30, "TRUE if data > 3h old — DO NOT TRADE", C_RED),
    ]

    for x, y, text, color in annotations:
        ax.text(x, y, text, ha="left", va="center",
                color=color, fontsize=7.5)
        ax.plot([0.50, x - 0.02], [y, y], color=color, lw=0.6, alpha=0.5)

    # Bottom summary
    ax.text(0.50, 0.12, "Execution engine only needs:", ha="center", va="center",
            color=C_TEXT, fontsize=10, fontweight="bold")
    ax.text(0.50, 0.06, "target_weights  +  stale  +  cycle_id", ha="center", va="center",
            color=C_GREEN, fontsize=12, fontfamily="monospace", fontweight="bold")

    return fig


def _chart_safety():
    fig, ax = plt.subplots(figsize=(10, 4))
    _dark_ax(fig, ax, "Safety Rails & Monitoring Alerts")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rails = [
        ("stale: true", "Skip execution — data pipeline behind", C_RED, "CRITICAL"),
        ("regime_score < 0.15", "Emergency flat — all weights → 0", C_RED, "CRITICAL"),
        ("Same cycle_id as last run", "Skip — idempotency guard", C_AMBER, "WARNING"),
        ("gross_exposure > 1.0", "Bug in sizing — do not trade", C_RED, "CRITICAL"),
        ("Signal file >2h old", "Signal service crashed — alert ops", C_AMBER, "WARNING"),
        ("n_holdings == 0 for >48h", "Possible entry logic / data issue", C_AMBER, "WARNING"),
        ("Execution drift >5%", "Reconciliation issue — investigate", C_AMBER, "WARNING"),
    ]

    y = 0.88
    for condition, response, color, severity in rails:
        sev_color = C_RED if severity == "CRITICAL" else C_AMBER
        badge = FancyBboxPatch((0.02, y - 0.035), 0.10, 0.06,
                               boxstyle="round,pad=0.01",
                               facecolor=sev_color, edgecolor="none", alpha=0.85)
        ax.add_patch(badge)
        ax.text(0.07, y, severity, ha="center", va="center",
                color="white", fontsize=6.5, fontweight="bold")
        ax.text(0.14, y, condition, ha="left", va="center",
                color=C_TEXT, fontsize=8, fontfamily="monospace")
        ax.text(0.55, y, "→  " + response, ha="left", va="center",
                color=C_SLATE, fontsize=8)
        y -= 0.11

    return fig


# ── PDF Assembly ────────────────────────────────────────────────────

def build_pdf():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "medallion_lite_execution_architecture.pdf"

    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )
    styles = _build_styles()
    story = []

    W = 6.5 * inch

    # ── Page 1: Title + Architecture ─────────────────────────────
    story.append(_p(styles, "title", "Medallion Lite — Execution Architecture"))
    story.append(_p(styles, "subtitle",
                     f"Internal Technical Reference  •  "
                     f"{datetime.now(timezone.utc).strftime('%B %Y')}"))
    story.append(_hr())

    story.append(_p(styles, "h1", "1. System Architecture"))
    story.append(_p(styles, "body",
        "The system is split into <b>three independent processes</b> that communicate "
        "through files and a database — no shared memory, no direct API calls. "
        "Each can crash and restart independently without corrupting the others."))

    story.append(_p(styles, "body",
        "<b>Process 1 — Data Collector</b> pulls 1-minute candles from the Coinbase Advanced "
        "Trade API every 5 minutes and writes aggregated hourly bars into DuckDB (bars_1h). "
        "Neither the signal service nor execution engine touches the exchange API for data."))

    story.append(_p(styles, "body",
        "<b>Process 2 — Signal Service</b> (MedallionSignalService) runs once per hour. "
        "It loads ~5,000 hours of OHLCV, computes the ensemble regime score and "
        "cross-sectional factor rankings, manages portfolio holdings (entries, exits, "
        "trailing stops), and publishes target weights to signal_output.json."))

    story.append(_p(styles, "body",
        "<b>Process 3 — Execution Engine</b> reads the signal file, diffs target weights "
        "against current portfolio weights, generates orders for deltas above a deadband, "
        "and executes via TWAP/VWAP on Coinbase."))

    story.append(_fig_to_image(_chart_architecture(), width=W))
    story.append(_p(styles, "caption", "Figure 1: Three-process architecture with file-based signal contract"))

    story.append(PageBreak())

    # ── Page 2: Signal Cycle Flow ────────────────────────────────
    story.append(_p(styles, "h1", "2. Signal Service — Decision Flow"))
    story.append(_p(styles, "body",
        "Each hourly cycle follows a deterministic pipeline. The signal service is "
        "<b>stateful</b> — it tracks holdings, entry timestamps, cumulative returns, "
        "and peak prices for trailing stops — but produces a <b>stateless output</b> "
        "(target weights). If the execution engine crashes, it naturally converges "
        "back to the correct portfolio on the next cycle without recovery logic."))

    story.append(_fig_to_image(_chart_signal_cycle(), width=W))
    story.append(_p(styles, "caption", "Figure 2: Signal service decision flow per cycle"))

    story.append(_p(styles, "h2", "Exit Hierarchy (evaluated every hour)"))
    exit_data = [
        ["Exit Type", "Condition", "Rationale"],
        ["Regime collapse", "Ensemble score < 0.15",
         "Emergency risk-off when market structure deteriorates across all 4 components"],
        ["Trailing stop", "−15% from peak cumulative return",
         "Protects gains and caps per-position loss without prediction"],
        ["Factor degradation", "Composite score drops below 0.40",
         "Original thesis no longer holds — cross-sectional rank has deteriorated"],
        ["Max hold", "Position open > 336 hours (14 days)",
         "Prevents stale positions; crypto blow-off tops rarely sustain > 1 week"],
    ]
    story.append(_make_table(exit_data, [1.0 * inch, 1.8 * inch, 3.7 * inch], styles))

    story.append(Spacer(1, 10))
    story.append(_p(styles, "h2", "Entry Rules (evaluated every 24 hours)"))
    entry_data = [
        ["Parameter", "Value", "Description"],
        ["Entry threshold", "Composite score > 0.65", "Top-tercile factor rank required"],
        ["Regime minimum", "Ensemble score > 0.45", "Market must be at least neutral"],
        ["Max positions", "25", "Diversification limit"],
        ["Per-name cap", "10%", "Concentration guard"],
        ["Sizing", "Inverse-volatility weighted", "Lower-vol tokens get larger weights"],
        ["Regime scaling", "weight × max(regime, 0.20)", "Gross exposure adapts to environment"],
    ]
    story.append(_make_table(entry_data, [1.2 * inch, 1.8 * inch, 3.5 * inch], styles))

    story.append(PageBreak())

    # ── Page 3: Signal Contract ──────────────────────────────────
    story.append(_p(styles, "h1", "3. Signal Output Contract"))
    story.append(_p(styles, "body",
        "The signal service writes <b>signal_output.json</b> every cycle. This is the "
        "<b>only file the execution engine needs to read</b>. The contract is intentionally "
        "minimal — the execution engine only needs three fields: target_weights, stale, "
        "and cycle_id. Everything else is informational for logging and monitoring."))

    story.append(_fig_to_image(_chart_contract_schema(), width=W))
    story.append(_p(styles, "caption", "Figure 3: Signal output JSON schema with field annotations"))

    story.append(_p(styles, "h2", "Field Reference"))
    field_data = [
        ["Field", "Type", "Description"],
        ["ts", "ISO 8601", "UTC timestamp of signal computation"],
        ["cycle_id", "string", "Unique ID for idempotency — never execute same cycle twice"],
        ["target_weights", "{symbol: float}", "Target portfolio weight per symbol. Sum ≤ 1. Zero = close."],
        ["regime_score", "float [0, 1]", "Market regime probability. Below 0.15 = emergency flat."],
        ["actions", "list[dict]", "Entries and exits this cycle — informational for audit trail"],
        ["diagnostics", "dict", "System health: n_holdings, gross_exposure, data_freshness_hours"],
        ["stale", "bool", "True if data > 3h old. Execution MUST NOT trade on stale signals."],
    ]
    story.append(_make_table(field_data, [1.1 * inch, 1.1 * inch, 4.3 * inch], styles))

    story.append(PageBreak())

    # ── Page 4: Execution Pseudocode ─────────────────────────────
    story.append(_p(styles, "h1", "4. Execution Engine — Reference Implementation"))
    story.append(_p(styles, "body",
        "The execution engine is deliberately simple. It is a stateless consumer that "
        "reads the signal file, computes deltas, and submits orders. No strategy logic, "
        "no factor computation, no regime awareness — all intelligence lives in the "
        "signal service."))

    code_text = """<pre>
import json

signal = json.load(open("signal_output.json"))

# ── Safety gates ──────────────────────────────────────────
if signal["stale"]:
    log.warning("Stale signal — skipping execution")
    return

if signal["cycle_id"] == last_executed_cycle_id:
    log.info("Already executed this cycle — skipping")
    return

# ── Compute deltas ────────────────────────────────────────
target_weights  = signal["target_weights"]
current_weights = get_current_portfolio_weights()   # from broker

for symbol in set(target_weights) | set(current_weights):
    target  = target_weights.get(symbol, 0.0)
    current = current_weights.get(symbol, 0.0)
    delta   = target - current

    if abs(delta) &lt; DEADBAND:              # e.g. 0.01 (1%)
        continue
    if abs(delta) * NAV &lt; MIN_NOTIONAL:    # e.g. $50
        continue

    # ── Submit order ──────────────────────────────────────
    side     = "BUY" if delta &gt; 0 else "SELL"
    notional = abs(delta) * NAV
    submit_twap_order(symbol, side, notional)

last_executed_cycle_id = signal["cycle_id"]
</pre>"""
    story.append(Paragraph(code_text, styles["code"]))
    story.append(Spacer(1, 8))

    story.append(_p(styles, "h2", "Key Design Decisions"))
    decisions = [
        "<b>Deadband filter</b> — Ignores weight changes smaller than 1% to avoid dust trades "
        "and unnecessary transaction costs. Tunable per deployment.",
        "<b>Minimum notional</b> — Coinbase has minimum order sizes. Skip any trade below $50 "
        "notional to avoid rejected orders.",
        "<b>TWAP execution</b> — Spread orders over the hour (minutes 10–55) to minimize "
        "market impact. For tokens with low liquidity, consider extending to 2-hour TWAP.",
        "<b>Idempotency</b> — The cycle_id prevents double-execution if the engine is "
        "triggered multiple times in the same hour.",
    ]
    for d in decisions:
        story.append(Paragraph(f"• {d}", styles["bullet"]))

    story.append(PageBreak())

    # ── Page 5: Timing + Safety ──────────────────────────────────
    story.append(_p(styles, "h1", "5. Timing"))
    story.append(_p(styles, "body",
        "All three processes are staggered within each hour to ensure data freshness. "
        "The collector finishes before the signal service starts, and the signal file "
        "is written before the execution engine reads it."))

    story.append(_fig_to_image(_chart_timing(), width=W))
    story.append(_p(styles, "caption", "Figure 4: Hourly execution timeline"))

    story.append(_p(styles, "h2", "Cron Schedule (Recommended for v1)"))
    cron_text = """<pre>
# Data collector: every 5 minutes
*/5 * * * * cd /path/to/trend_crypto &amp;&amp; python scripts/collect_coinbase.py

# Signal service: minute 5 of every hour (after collector syncs)
5 * * * * cd /path/to/trend_crypto &amp;&amp; python scripts/run_medallion_live.py \\
    --state-dir /path/to/live_state &gt;&gt; /var/log/medallion_signal.log 2&gt;&amp;1

# Execution engine: minute 10 of every hour (after signal is written)
10 * * * * /path/to/execution_engine \\
    --signal-file /path/to/live_state/signal_output.json
</pre>"""
    story.append(Paragraph(cron_text, styles["code"]))

    story.append(Spacer(1, 12))
    story.append(_p(styles, "h1", "6. Safety Rails & Monitoring"))
    story.append(_p(styles, "body",
        "Both the signal service and execution engine implement layered safety checks. "
        "The system is designed to <b>fail safe</b> — when in doubt, do nothing."))

    story.append(_fig_to_image(_chart_safety(), width=W))
    story.append(_p(styles, "caption", "Figure 5: Safety conditions and automated responses"))

    story.append(PageBreak())

    # ── Page 6: State + Deployment ───────────────────────────────
    story.append(_p(styles, "h1", "7. State Management"))
    story.append(_p(styles, "body",
        "The signal service persists holdings state between cycles via <b>medallion_state.json</b>. "
        "This file contains current positions, entry timestamps, cumulative returns, and peak "
        "values for trailing stop calculation."))

    state_text = """<pre>
{
  "holdings": {
    "SOL-USD": {
      "symbol": "SOL-USD",
      "entry_ts": "2026-03-15T14:00:00+00:00",
      "entry_score": 0.78,
      "hours_held": 48,
      "cum_ret": 0.12,
      "peak_cum": 0.15
    },
    "ETH-USD": { ... },
    "AVAX-USD": { ... }
  },
  "last_cycle_ts": "2026-03-17T14:00:00+00:00",
  "last_rebalance_hour": 1224,
  "cycle_count": 1248
}
</pre>"""
    story.append(Paragraph(state_text, styles["code"]))

    story.append(_p(styles, "body",
        "<b>Recovery:</b> If the state file is deleted, the service starts fresh with no holdings. "
        "This is safe but will miss any existing positions — coordinate with the execution "
        "engine to reconcile. The execution engine itself is stateless and always converges."))

    story.append(Spacer(1, 10))
    story.append(_p(styles, "h1", "8. Deployment Options"))

    deploy_data = [
        ["Option", "Signal Service", "Execution Engine", "Best For"],
        ["A: Cron (recommended)",
         "cron at minute 5 of each hour",
         "cron at minute 10 of each hour",
         "Production v1 — simple, robust, easy to monitor"],
        ["B: Daemon",
         "Continuous loop with 3600s sleep",
         "Polls signal file for changes",
         "Always-on deployments, lower latency"],
        ["C: LiveRunner",
         "Embedded via MedallionEmbeddedAdapter",
         "Uses existing OMS + PaperBroker",
         "Paper trading, single-process development"],
    ]
    story.append(_make_table(deploy_data, [1.2 * inch, 1.6 * inch, 1.6 * inch, 2.1 * inch], styles))

    story.append(Spacer(1, 12))
    story.append(_p(styles, "h1", "9. Data Dependencies"))

    dep_data = [
        ["Dependency", "Frequency", "Source", "Lookback"],
        ["Hourly OHLCV (bars_1h)", "Continuous", "Coinbase API → Collector → DuckDB", "5,000 hours"],
        ["BTC daily close", "Derived", "Resampled from hourly in signal service", "200 days"],
        ["Universe (~50 tokens)", "Each cycle", "SQL median ADV filter", "—"],
        ["Portfolio state", "Each cycle", "medallion_state.json", "—"],
    ]
    story.append(_make_table(dep_data, [1.5 * inch, 1.0 * inch, 2.5 * inch, 1.5 * inch], styles))

    story.append(Spacer(1, 12))
    story.append(_p(styles, "h1", "10. File Index"))

    files_data = [
        ["File", "Purpose"],
        ["src/live/medallion_signal.py", "Signal service core — state management, factor computation, publishing"],
        ["src/strategy/medallion_portfolio.py", "PortfolioStrategy adapter for LiveRunner integration"],
        ["src/data/live_feed.py", "DuckDB-backed DataFeed for LiveRunner"],
        ["scripts/run_medallion_live.py", "CLI entry point — cron / daemon / paper modes"],
        ["scripts/collect_coinbase.py", "Data collector — Coinbase API to DuckDB"],
        ["docs/medallion_integration.md", "Full integration guide (markdown)"],
        ["scripts/research/medallion_lite/", "Research backtest code (reference implementation)"],
    ]
    story.append(_make_table(files_data, [2.8 * inch, 3.7 * inch], styles))

    # ── Build ────────────────────────────────────────────────────
    doc.build(story)
    print(f"PDF written to {out_path}")
    return out_path


if __name__ == "__main__":
    build_pdf()
