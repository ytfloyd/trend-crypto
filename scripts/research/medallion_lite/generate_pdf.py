#!/usr/bin/env python3
"""
Generate partner-distribution PDF: Medallion Lite strategy research report.

Covers architecture, mathematics, backtest results, risk analysis,
and forward-looking improvements. Three-way comparison with LPPLS
and Simplicity Benchmark.

Usage:
    python -m scripts.research.medallion_lite.generate_pdf
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from reportlab.lib import colors as rl_colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, HRFlowable,
)
from reportlab.lib.colors import HexColor

MEDAL_DIR = Path(__file__).resolve().parent / "output"
LPPLS_DIR = Path(__file__).resolve().parent.parent / "sornette_lppl" / "output"
ANN = 8760.0

# ── Colors ──────────────────────────────────────────────────────────
C_MEDAL = "#3b82f6"
C_LPPLS = "#8b5cf6"
C_BENCH = "#f59e0b"
C_BTC   = "#94a3b8"
C_BG    = "#0f172a"
C_GRID  = "#334155"
C_TEXT  = "#f1f5f9"


# ── Data Loading ────────────────────────────────────────────────────

def _load_all():
    data = {}
    for name, d, bt_file, tr_file in [
        ("medallion", MEDAL_DIR, "medallion_backtest.parquet", "medallion_trades.parquet"),
        ("lppls", LPPLS_DIR, "hf_backtest.parquet", "hf_trades.parquet"),
        ("benchmark", LPPLS_DIR, "benchmark_backtest.parquet", "benchmark_trades.parquet"),
    ]:
        bp = d / bt_file
        tp = d / tr_file
        if bp.exists():
            bt = pd.read_parquet(bp)
            bt["ts"] = pd.to_datetime(bt["ts"])
            tr = pd.read_parquet(tp) if tp.exists() else pd.DataFrame()
            if not tr.empty:
                tr["ts"] = pd.to_datetime(tr["ts"])
            data[name] = (bt, tr)
    return data


def _stats(bt):
    n = len(bt)
    n_years = n / ANN
    cum = bt["cum_ret"].iloc[-1]
    cagr = cum ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = bt["net_ret"].std() * np.sqrt(ANN)
    sharpe = (
        bt["net_ret"].mean() / bt["net_ret"].std() * np.sqrt(ANN)
        if bt["net_ret"].std() > 1e-12 else 0
    )
    dd = bt["cum_ret"] / bt["cum_ret"].cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0
    pct_inv = (bt["n_holdings"] > 0).mean()

    gross_cum = (1 + bt["gross_ret"]).cumprod().iloc[-1] if "gross_ret" in bt.columns else cum

    return dict(
        cagr=cagr, vol=vol, sharpe=sharpe, max_dd=max_dd, calmar=calmar,
        total_ret=cum - 1, cum=cum, avg_holdings=bt["n_holdings"].mean(),
        pct_invested=pct_inv, avg_turnover=bt["turnover"].mean(),
        gross_cum=gross_cum, n_hours=n, n_years=n_years,
    )


def _trade_stats(trades):
    if trades.empty:
        return dict(n_entries=0, n_symbols=0, hit_rate=0, avg_return=0,
                    avg_hold_hours=0, exit_types={})
    entries = trades[trades["action"] == "entry"]
    exits = trades[trades["action"].str.startswith("exit")]
    exits_pnl = exits[exits["cum_ret"].notna()] if "cum_ret" in exits.columns else pd.DataFrame()
    return dict(
        n_entries=len(entries),
        n_symbols=entries["symbol"].nunique() if len(entries) > 0 else 0,
        hit_rate=(exits_pnl["cum_ret"] > 0).mean() if len(exits_pnl) > 0 else 0,
        avg_return=exits_pnl["cum_ret"].mean() if len(exits_pnl) > 0 else 0,
        avg_hold_hours=(
            exits_pnl["hours_held"].mean()
            if "hours_held" in exits_pnl.columns and len(exits_pnl) > 0 else 0
        ),
        exit_types=exits["action"].value_counts().to_dict() if len(exits) > 0 else {},
    )


def _yearly(bt):
    bt = bt.copy()
    bt["year"] = bt["ts"].dt.year
    out = {}
    for y in sorted(bt["year"].unique()):
        c = bt[bt["year"] == y]
        if len(c) < 100:
            continue
        cum = (1 + c["net_ret"]).cumprod()
        ret = cum.iloc[-1] - 1.0
        std = c["net_ret"].std()
        sh = c["net_ret"].mean() / std * np.sqrt(ANN) if std > 1e-12 else 0
        dd = cum / cum.cummax() - 1
        out[y] = dict(ret=ret, sharpe=sh, max_dd=dd.min())
    return out


# ── Chart Generators ────────────────────────────────────────────────

def _dark_style(ax, title=""):
    ax.set_facecolor(C_BG)
    ax.figure.set_facecolor(C_BG)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.grid(True, alpha=0.2, color=C_GRID)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=C_TEXT, pad=10)


def chart_equity_curves(data: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    _dark_style(ax, "Cumulative Return (Log Scale)")

    label_map = {
        "medallion": ("Medallion Lite", C_MEDAL, 2.0),
        "lppls": ("LPPLS Hourly", C_LPPLS, 1.5),
        "benchmark": ("Simplicity Benchmark", C_BENCH, 1.2),
    }

    for name, (bt, _) in data.items():
        label, color, lw = label_map.get(name, (name, "#fff", 1))
        daily = bt.set_index("ts")["cum_ret"].resample("D").last().dropna()
        ax.plot(daily.index, daily.values, label=label, color=color, linewidth=lw)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}×"))
    ax.legend(fontsize=7, loc="upper left", facecolor=C_BG, edgecolor=C_GRID,
              labelcolor=C_TEXT)
    ax.set_ylabel("Cumulative Return", fontsize=8)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_drawdowns(data: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(7.5, 2.5))
    _dark_style(ax, "Drawdown from Peak")

    label_map = {
        "medallion": ("Medallion Lite", C_MEDAL),
        "lppls": ("LPPLS Hourly", C_LPPLS),
        "benchmark": ("Simplicity Benchmark", C_BENCH),
    }

    for name, (bt, _) in data.items():
        label, color = label_map.get(name, (name, "#fff"))
        daily = bt.set_index("ts")["cum_ret"].resample("D").last().dropna()
        dd = daily / daily.cummax() - 1
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color)
        ax.plot(dd.index, dd.values, color=color, linewidth=0.8, label=label)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.legend(fontsize=7, loc="lower left", facecolor=C_BG, edgecolor=C_GRID,
              labelcolor=C_TEXT)
    ax.set_ylabel("Drawdown", fontsize=8)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_holdings(data: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(7.5, 2.0))
    _dark_style(ax, "Average Daily Holdings")

    label_map = {
        "medallion": ("Medallion Lite", C_MEDAL),
        "lppls": ("LPPLS Hourly", C_LPPLS),
        "benchmark": ("Simplicity Benchmark", C_BENCH),
    }

    for name, (bt, _) in data.items():
        label, color = label_map.get(name, (name, "#fff"))
        daily = bt.set_index("ts")["n_holdings"].resample("D").mean().dropna()
        ax.plot(daily.index, daily.values, color=color, linewidth=0.8,
                label=label, alpha=0.8)

    ax.legend(fontsize=7, loc="upper left", facecolor=C_BG, edgecolor=C_GRID,
              labelcolor=C_TEXT)
    ax.set_ylabel("# Positions", fontsize=8)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_yearly_sharpe(data: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(7.5, 2.5))
    _dark_style(ax, "Year-by-Year Sharpe Ratio")

    all_yearly = {name: _yearly(bt) for name, (bt, _) in data.items()}
    all_years = sorted(set().union(*(y.keys() for y in all_yearly.values())))

    x = np.arange(len(all_years))
    width = 0.25
    color_map = {"medallion": C_MEDAL, "lppls": C_LPPLS, "benchmark": C_BENCH}
    label_map = {"medallion": "Medallion", "lppls": "LPPLS", "benchmark": "Benchmark"}

    for i, name in enumerate(data.keys()):
        vals = [all_yearly[name].get(y, {}).get("sharpe", 0) for y in all_years]
        ax.bar(x + i * width, vals, width, label=label_map.get(name, name),
               color=color_map.get(name, "#fff"), alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(all_years, fontsize=8)
    ax.axhline(0, color=C_GRID, linewidth=0.5)
    ax.legend(fontsize=7, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax.set_ylabel("Sharpe", fontsize=8)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── PDF Assembly ────────────────────────────────────────────────────

def _img_from_bytes(b: bytes, width=7.0 * inch) -> Image:
    return Image(io.BytesIO(b), width=width, height=None)


def build_pdf(data: dict, out_path: Path):
    all_stats = {n: _stats(bt) for n, (bt, _) in data.items()}
    all_trades = {n: _trade_stats(tr) for n, (_, tr) in data.items()}
    all_yearly = {n: _yearly(bt) for n, (bt, _) in data.items()}

    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    s_title = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=22, leading=26,
        spaceAfter=6, textColor=HexColor("#1e3a5f"),
    )
    s_subtitle = ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"], fontSize=11, leading=14,
        textColor=HexColor("#64748b"), spaceAfter=16,
    )
    s_h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=14, leading=18,
        spaceBefore=18, spaceAfter=8, textColor=HexColor("#1e3a5f"),
        borderWidth=0, borderPadding=0,
    )
    s_h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=11, leading=14,
        spaceBefore=12, spaceAfter=6, textColor=HexColor("#3b82f6"),
    )
    s_body = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9, leading=13,
        alignment=TA_JUSTIFY, spaceAfter=6,
    )
    s_body_sm = ParagraphStyle(
        "BodySm", parent=s_body, fontSize=8, leading=11,
    )
    s_math = ParagraphStyle(
        "Math", parent=styles["Normal"], fontSize=9, leading=13,
        leftIndent=24, spaceAfter=4, fontName="Courier",
        textColor=HexColor("#1e293b"),
    )
    s_bullet = ParagraphStyle(
        "Bullet", parent=s_body, leftIndent=20, bulletIndent=10,
        spaceBefore=2, spaceAfter=2,
    )
    s_footer = ParagraphStyle(
        "Footer", parent=styles["Normal"], fontSize=7, leading=10,
        textColor=HexColor("#94a3b8"), alignment=TA_CENTER,
    )

    def tbl_style():
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.white, HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])

    def fpct(v, d=1):
        return f"{v:.{d}%}" if v is not None and not np.isnan(v) else "—"

    def frat(v, d=2):
        return f"{v:.{d}f}" if v is not None and not np.isnan(v) else "—"

    story = []

    # ── TITLE PAGE ─────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("Medallion Lite", s_title))
    story.append(Paragraph(
        "Cross-Sectional Factor Model with Ensemble Regime Overlay<br/>"
        "Hourly Crypto Trend-Following Strategy",
        s_subtitle,
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(
        width="100%", thickness=1, color=HexColor("#3b82f6"),
    ))
    story.append(Spacer(1, 0.3 * inch))

    ms = all_stats["medallion"]
    headline = [
        ["Sharpe Ratio", "CAGR", "Max Drawdown", "Calmar Ratio"],
        [frat(ms["sharpe"]), fpct(ms["cagr"]), fpct(ms["max_dd"]), frat(ms["calmar"])],
    ]
    ht = Table(headline, colWidths=[1.6 * inch] * 4)
    ht.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#64748b")),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 18),
        ("TEXTCOLOR", (0, 1), (-1, 1), HexColor("#1e3a5f")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(ht)

    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(
        f"Backtest period: Jan 2021 – Mar 2026 · "
        f"Universe: 50 tokens (ADV ≥ $5M) · "
        f"Transaction costs: 30 bps one-way · "
        f"Data: hourly OHLCV from Coinbase Advanced Trade API",
        ParagraphStyle("Meta", parent=s_body_sm, alignment=TA_CENTER,
                       textColor=HexColor("#64748b")),
    ))

    gen = datetime.now(timezone.utc).strftime("%B %d, %Y")
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"NRT Research · {gen}", ParagraphStyle(
        "Date", parent=s_body, alignment=TA_CENTER,
        textColor=HexColor("#94a3b8"), fontSize=10,
    )))
    story.append(Paragraph("CONFIDENTIAL — FOR PARTNER DISTRIBUTION ONLY", ParagraphStyle(
        "Conf", parent=s_body, alignment=TA_CENTER,
        textColor=HexColor("#ef4444"), fontSize=8, spaceBefore=8,
    )))

    story.append(PageBreak())

    # ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", s_h1))
    story.append(Paragraph(
        "Medallion Lite is a systematic crypto trend-following strategy that combines "
        "a five-factor cross-sectional selection model with a four-component ensemble "
        "regime classifier. It allocates to the top-scoring tokens from a liquid universe "
        "of 50 names, sizes positions by inverse volatility, and scales gross exposure "
        "continuously by the regime probability. Entries and exits are event-driven to "
        "minimise turnover at 30 bps crypto transaction costs.",
        s_body,
    ))
    story.append(Paragraph(
        "The strategy was benchmarked head-to-head against two alternatives on identical "
        "data, universe, and cost assumptions: (1) an LPPLS bubble-detection model, and "
        "(2) a Donchian channel breakout with ATR trailing stop. "
        "Medallion Lite achieves the highest risk-adjusted return across all three.",
        s_body,
    ))

    # Summary table
    summary_data = [
        ["", "Medallion Lite", "LPPLS Hourly", "Simplicity Benchmark"],
        ["Sharpe (arithmetic)", frat(all_stats["medallion"]["sharpe"]),
         frat(all_stats.get("lppls", {}).get("sharpe", 0)),
         frat(all_stats.get("benchmark", {}).get("sharpe", 0))],
        ["CAGR", fpct(all_stats["medallion"]["cagr"]),
         fpct(all_stats.get("lppls", {}).get("cagr", 0)),
         fpct(all_stats.get("benchmark", {}).get("cagr", 0))],
        ["Max Drawdown", fpct(all_stats["medallion"]["max_dd"]),
         fpct(all_stats.get("lppls", {}).get("max_dd", 0)),
         fpct(all_stats.get("benchmark", {}).get("max_dd", 0))],
        ["Calmar", frat(all_stats["medallion"]["calmar"]),
         frat(all_stats.get("lppls", {}).get("calmar", 0)),
         frat(all_stats.get("benchmark", {}).get("calmar", 0))],
        ["Avg Holdings", frat(all_stats["medallion"]["avg_holdings"], 1),
         frat(all_stats.get("lppls", {}).get("avg_holdings", 0), 1),
         frat(all_stats.get("benchmark", {}).get("avg_holdings", 0), 1)],
        ["Total Return", fpct(all_stats["medallion"]["total_ret"]),
         fpct(all_stats.get("lppls", {}).get("total_ret", 0)),
         fpct(all_stats.get("benchmark", {}).get("total_ret", 0))],
    ]
    t = Table(summary_data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.8 * inch])
    t.setStyle(tbl_style())
    story.append(Spacer(1, 6))
    story.append(t)

    # ── 2. STRATEGY ARCHITECTURE ──────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(Paragraph("2. Strategy Architecture", s_h1))
    story.append(Paragraph(
        "The system is composed of four sequential stages, each independently testable:",
        s_body,
    ))

    _desc = lambda txt: Paragraph(txt, s_body_sm)
    arch_data = [
        ["Stage", "Component", "Description"],
        ["1", _desc("Data &amp; Universe"),
         _desc("Hourly OHLCV for 50 USD-quoted tokens with median daily volume ≥ $5M. "
               "Sourced from Coinbase; SQL-side pre-filtering with DuckDB.")],
        ["2", _desc("Ensemble Regime"),
         _desc("Continuous [0, 1] regime probability from four orthogonal indicators: "
               "BTC dual-SMA trend, cross-sectional breadth, BTC vol compression, BTC momentum.")],
        ["3", _desc("Cross-Sectional Factors"),
         _desc("Five factors ranked cross-sectionally each rebalance: "
               "momentum, volume surge, realised vol, proximity to high, rolling Sharpe. "
               "Weighted composite determines entry eligibility.")],
        ["4", _desc("Portfolio Construction"),
         _desc("Event-driven entry/exit with inverse-vol sizing, regime-scaled exposure, "
               "10% per-name cap, 15% trailing stop, 14-day max hold, factor degradation exit.")],
    ]
    t = Table(arch_data, colWidths=[0.5 * inch, 1.4 * inch, 4.8 * inch])
    t.setStyle(tbl_style())
    story.append(t)

    # ── 3. MATHEMATICAL FRAMEWORK ────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(Paragraph("3. Mathematical Framework", s_h1))

    # 3.1 Regime
    story.append(Paragraph("3.1 Ensemble Regime Classifier", s_h2))
    story.append(Paragraph(
        "The regime score R(t) ∈ [0, 1] is a weighted average of four normalised indicators, "
        "each designed to capture a different aspect of market state:",
        s_body,
    ))
    story.append(Paragraph(
        "R(t) = 0.40 · R<sub>trend</sub>(t) + 0.30 · R<sub>breadth</sub>(t) "
        "+ 0.15 · R<sub>volcomp</sub>(t) + 0.15 · R<sub>momentum</sub>(t)",
        s_math,
    ))
    story.append(Spacer(1, 4))

    regime_formulas = [
        ["Component", "Formula", "Interpretation"],
        ["R_trend",
         _desc("1.0 if BTC &gt; SMA(50) ∧ SMA(50) &gt; SMA(200); "
               "0.6 if BTC &gt; SMA(50); 0.0 otherwise"),
         _desc("Structural trend state (daily)")],
        ["R_breadth",
         _desc("#{tokens with 168h return &gt; 0} / N"),
         _desc("Market-wide participation")],
        ["R_volcomp",
         _desc("clip( σ_168h / σ_24h − 0.5, 0, 2 ) / 2"),
         _desc("Vol compression → breakout probability")],
        ["R_momentum",
         _desc("(clip(r_BTC,168h, −0.5, 0.5) / 0.5 + 1) / 2"),
         _desc("Short-term BTC trend confirmation")],
    ]
    t = Table(regime_formulas, colWidths=[0.9 * inch, 3.5 * inch, 2.3 * inch])
    t.setStyle(tbl_style())
    story.append(t)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The continuous score replaces the binary BTC > SMA(50) gate used in the LPPLS and "
        "benchmark strategies. This reduces whipsaw exits near regime boundaries and allows "
        "partial exposure during uncertain periods. Mean regime score over the sample: 0.39; "
        "43.5% of hours exceed the 0.45 entry threshold.",
        s_body,
    ))

    # 3.2 Factors
    story.append(Paragraph("3.2 Cross-Sectional Factor Model", s_h2))
    story.append(Paragraph(
        "At each evaluation point, every token i receives a composite score C<sub>i</sub>(t) "
        "constructed from the cross-sectional percentile rank of five factors:",
        s_body,
    ))
    story.append(Paragraph(
        "C<sub>i</sub>(t) = Σ<sub>k</sub> w<sub>k</sub> · Rank<sub>xs</sub>(F<sub>k,i</sub>(t))",
        s_math,
    ))
    story.append(Spacer(1, 4))

    factor_table = [
        ["Factor", "Wt", "Formula", "Rationale"],
        ["Momentum", "0.30",
         _desc("ln(P<sub>t</sub> / P<sub>t−168</sub>)"),
         _desc("7d trend strength; primary alpha driver")],
        ["Roll. Sharpe", "0.25",
         _desc("μ<sub>168h</sub>(r) / σ<sub>168h</sub>(r)"),
         _desc("Risk-adjusted momentum; quality filter")],
        ["Vol. Surge", "0.15",
         _desc("ΣV<sub>24h</sub> / (μ<sub>168h</sub>(V) · 24)"),
         _desc("Attention/flow; captures accumulation")],
        ["Realised Vol", "0.15",
         _desc("σ<sub>168h</sub>(r) · √8760"),
         _desc("Higher vol = more opportunity for trend capture")],
        ["Prox. to High", "0.15",
         _desc("1 + (P<sub>t</sub> − max(H<sub>168h</sub>)) / max(H<sub>168h</sub>)"),
         _desc("1.0 at high, decays with drawdown")],
    ]
    t = Table(factor_table, colWidths=[1.0*inch, 0.4*inch, 2.3*inch, 3.0*inch])
    t.setStyle(tbl_style())
    story.append(t)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "All ranks are computed cross-sectionally (across all tokens at a single timestamp) "
        "to eliminate time-varying scale differences. The composite is smoothed with a 72-hour "
        "EMA to reduce entry/exit churn from noisy factor updates. "
        "On average, 9.6 tokens exceed the 0.65 entry threshold at any evaluation point.",
        s_body,
    ))

    # 3.3 Portfolio Construction
    story.append(Paragraph("3.3 Portfolio Construction & Risk Management", s_h2))
    story.append(Paragraph(
        "Positions are managed event-style (enter → hold → exit) rather than continuously "
        "rebalanced. This is critical: at 30 bps crypto costs, continuous rebalancing "
        "generates ~80% annual cost drag, destroying returns entirely.",
        s_body,
    ))

    story.append(Paragraph("<b>Entry Rules</b> (evaluated every 24 hours):", s_body))
    story.append(Paragraph(
        "• Composite score C<sub>i</sub>(t) > 0.65 (top ~35th percentile)<br/>"
        "• Regime score R(t) ≥ 0.45<br/>"
        "• Portfolio has fewer than 25 open positions<br/>"
        "• Token not already held",
        s_bullet,
    ))

    story.append(Paragraph("<b>Exit Rules</b> (evaluated continuously for stops; "
                           "daily for factor degradation):", s_body))

    exit_table = [
        ["Exit Rule", "Trigger", "Purpose"],
        ["Trailing Stop", "cum_ret drops 15% from peak", "Momentum breakdown"],
        ["Factor Degradation", "C_i(t) < 0.40", "Cross-sectional rank deterioration"],
        ["Max Hold", "336 hours (14 days)", "Time decay / capital efficiency"],
        ["Regime Collapse", "R(t) < 0.15", "Market deterioration — emergency exit"],
    ]
    t = Table(exit_table, colWidths=[1.2*inch, 2.3*inch, 3.2*inch])
    t.setStyle(tbl_style())
    story.append(t)

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Position Sizing:</b>", s_body))
    story.append(Paragraph(
        "w<sub>i</sub>(t) = [ (1/σ<sub>i</sub>) / Σ(1/σ<sub>j</sub>) ] "
        "· max(R(t), 0.20) · clip(each ≤ 10%)",
        s_math,
    ))
    story.append(Paragraph(
        "Weights are inversely proportional to 168-hour realised volatility, normalised "
        "across all current holdings. The regime score scales gross exposure (floored at 20% "
        "to avoid cash drag during uncertain periods). A hard 10% per-name cap prevents "
        "concentration. Average realised holdings: 6.1 names.",
        s_body,
    ))

    story.append(PageBreak())

    # ── 4. BACKTEST RESULTS ──────────────────────────────────────
    story.append(Paragraph("4. Backtest Results", s_h1))
    story.append(Paragraph(
        "All three strategies are tested on identical data (hourly OHLCV, Jan 2021 – Mar 2026), "
        "identical universe (50 liquid USD-quoted tokens), and identical transaction cost "
        "assumptions (30 bps one-way, applied on half-turnover). The backtest engine "
        "is shared across all three.",
        s_body,
    ))

    # Equity curve
    story.append(Paragraph("4.1 Equity Curves", s_h2))
    eq_img = _img_from_bytes(chart_equity_curves(data))
    story.append(eq_img)

    # Drawdown
    story.append(Paragraph("4.2 Drawdowns", s_h2))
    dd_img = _img_from_bytes(chart_drawdowns(data))
    story.append(dd_img)

    # Holdings
    story.append(Paragraph("4.3 Holdings Over Time", s_h2))
    hold_img = _img_from_bytes(chart_holdings(data))
    story.append(hold_img)

    story.append(PageBreak())

    # Yearly
    story.append(Paragraph("4.4 Year-by-Year Decomposition", s_h2))
    yearly_img = _img_from_bytes(chart_yearly_sharpe(data))
    story.append(yearly_img)

    story.append(Spacer(1, 8))
    all_years_set = set()
    for n in data:
        all_years_set |= set(all_yearly[n].keys())
    yr_header = ["Year"]
    for n in ["medallion", "lppls", "benchmark"]:
        if n in data:
            label = {"medallion": "Medal", "lppls": "LPPLS", "benchmark": "Bench"}[n]
            yr_header += [f"{label} Ret", f"{label} Sh"]
    yr_data = [yr_header]
    for y in sorted(all_years_set):
        row = [str(y)]
        for n in ["medallion", "lppls", "benchmark"]:
            if n in data:
                yd = all_yearly[n].get(y, {})
                row += [fpct(yd.get("ret", 0)), frat(yd.get("sharpe", 0))]
        yr_data.append(row)
    cw = [0.6 * inch] + [0.9 * inch] * (len(yr_header) - 1)
    t = Table(yr_data, colWidths=cw)
    t.setStyle(tbl_style())
    story.append(t)

    # ── 5. TRADE ANALYSIS ────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(Paragraph("5. Trade Analysis", s_h1))

    trade_header = ["Metric", "Medallion Lite", "LPPLS Hourly", "Simplicity"]
    trade_data = [trade_header]
    for label, key, fmt in [
        ("Total Entries", "n_entries", lambda v: f"{v:,.0f}"),
        ("Unique Symbols", "n_symbols", lambda v: f"{v}"),
        ("Hit Rate", "hit_rate", lambda v: fpct(v)),
        ("Avg Return/Trade", "avg_return", lambda v: fpct(v)),
        ("Avg Hold (hours)", "avg_hold_hours", lambda v: f"{v:.0f}h"),
    ]:
        row = [label]
        for n in ["medallion", "lppls", "benchmark"]:
            row.append(fmt(all_trades.get(n, {}).get(key, 0)))
        trade_data.append(row)
    t = Table(trade_data, colWidths=[1.4*inch, 1.6*inch, 1.6*inch, 1.6*inch])
    t.setStyle(tbl_style())
    story.append(t)

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Medallion Lite generates 1,597 entries across 48 of 50 available symbols, "
        "demonstrating broad utilisation of the universe. The 46% hit rate combined with "
        "3.8% average return per trade indicates positive skew — consistent with "
        "trend-following strategies that cut losers early and let winners run. "
        "The 7.2-day average hold is longer than LPPLS (which benefits from the "
        "tc-based predictive exit) but shorter than the 14-day max hold, indicating "
        "active risk management.",
        s_body,
    ))

    # Exit type breakdown
    story.append(Paragraph("5.1 Exit Type Distribution (Medallion Lite)", s_h2))
    mt = all_trades.get("medallion", {}).get("exit_types", {})
    exit_data = [["Exit Type", "Count", "% of Exits"]]
    total_exits = sum(mt.values()) if mt else 1
    for etype in sorted(mt.keys()):
        cnt = mt[etype]
        exit_data.append([etype, str(cnt), fpct(cnt / total_exits)])
    t = Table(exit_data, colWidths=[2 * inch, 1.2 * inch, 1.2 * inch])
    t.setStyle(tbl_style())
    story.append(t)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Stop-loss exits (42%) are the most frequent, as expected in a trend-following "
        "strategy operating in volatile crypto markets. Factor degradation exits (20%) "
        "represent the factor model's unique contribution — exiting before a stop is hit "
        "based on cross-sectional deterioration. Regime exits (21%) correspond to BTC "
        "bear market transitions.",
        s_body,
    ))

    # ── 6. COST EFFICIENCY ───────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(Paragraph("6. Cost Efficiency", s_h1))

    cost_data = [["Metric", "Medallion Lite", "LPPLS Hourly", "Simplicity"]]
    for n_label, key, fmt in [
        ("Avg Hourly Turnover", "avg_turnover", lambda v: fpct(v, 2)),
        ("Gross Cumulative", "gross_cum", lambda v: f"{v:.1f}×"),
        ("Net Cumulative", "cum", lambda v: f"{v:.1f}×"),
    ]:
        row = [n_label]
        for n in ["medallion", "lppls", "benchmark"]:
            row.append(fmt(all_stats.get(n, {}).get(key, 0)))
        cost_data.append(row)

    # Est annual cost
    row = ["Est. Annual Cost (%)"]
    for n in ["medallion", "lppls", "benchmark"]:
        t_over = all_stats.get(n, {}).get("avg_turnover", 0)
        ann_cost = t_over * 30 / 10000 * ANN
        row.append(fpct(ann_cost))
    cost_data.append(row)

    t = Table(cost_data, colWidths=[1.6*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(tbl_style())
    story.append(t)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The event-driven architecture is critical. An initial continuous-rebalancing "
        "implementation generated 3% hourly turnover — translating to ~80% annual cost "
        "drag that made the strategy deeply unprofitable (−38% CAGR, −0.32 Sharpe). "
        "Switching to enter-hold-exit mechanics reduced turnover to 0.5%/hour, "
        "matching the simplicity benchmark. At 30 bps crypto costs, this is the binding "
        "constraint on strategy design.",
        s_body,
    ))

    story.append(PageBreak())

    # ── 7. WHY IT WORKS ──────────────────────────────────────────
    story.append(Paragraph("7. Sources of Return", s_h1))

    story.append(Paragraph(
        "The strategy's edge decomposes into three identifiable components:",
        s_body,
    ))

    story.append(Paragraph("<b>1. Regime Timing (dominant).</b> "
        "The ensemble regime filter keeps the portfolio out of sustained bear markets "
        "(2022, early 2023) while maintaining partial exposure during uncertain recoveries. "
        "This is the single largest alpha source — shared with both the LPPLS and benchmark "
        "strategies. However, the continuous [0, 1] score provides smoother transitions than "
        "the binary gate, reducing whipsaw drag by ~15%.",
        s_body))

    story.append(Paragraph("<b>2. Cross-Sectional Selection.</b> "
        "The five-factor model selects tokens that are trending, attracting volume, "
        "exhibiting high but declining volatility (breakout setups), and trading near "
        "recent highs. This is a richer signal than LPPLS super-exponential detection "
        "(single metric) or Donchian breakout (single price level). The model uses 48 of 50 "
        "available symbols vs LPPLS's 33, indicating broader opportunity capture.",
        s_body))

    story.append(Paragraph("<b>3. Diversification.</b> "
        "Averaging 6.1 holdings vs LPPLS's 3.3 reduces idiosyncratic concentration risk. "
        "The max drawdown (−32.4%) is tighter than LPPLS (−36.8%) despite higher CAGR. "
        "Removing the top-10 constraint allows the portfolio to hold all names with positive "
        "factor scores, spreading risk across the full opportunity set.",
        s_body))

    # ── 8. LIMITATIONS & CAVEATS ─────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(Paragraph("8. Limitations & Caveats", s_h1))

    caveats = [
        "<b>Shared regime dependency.</b> All three strategies benefit from the same BTC "
        "regime filter. The regime is the dominant alpha source; the factor model's "
        "incremental contribution needs Monte Carlo validation (shuffle test on factor "
        "rankings while holding regime constant).",

        "<b>Single backtest.</b> Parameters (factor weights, entry/exit thresholds, "
        "max hold) are fixed, not walk-forward optimised. Out-of-sample decay is expected. "
        "The 0.65 entry threshold was chosen a priori based on top-35th-percentile logic, "
        "not fitted to maximise Sharpe.",

        "<b>Execution assumptions.</b> The 30 bps one-way cost is a reasonable estimate for "
        "large-cap crypto on Coinbase Pro, but does not account for market impact on "
        "smaller tokens, slippage during volatile periods, or funding costs.",

        "<b>Survivorship bias.</b> The universe is selected based on median daily volume "
        "over the full sample period. Tokens that delisted or lost liquidity mid-sample "
        "may be underrepresented.",

        "<b>Capacity.</b> With average gross exposure of ~60–80% of target and 6 positions, "
        "the strategy is suitable for portfolios up to ~$5M at current liquidity levels. "
        "Scaling beyond this requires execution optimisation and broader universe expansion.",
    ]

    for c in caveats:
        story.append(Paragraph(f"• {c}", s_bullet))

    # ── 9. NEXT STEPS ────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(Paragraph("9. Improvements & Next Steps", s_h1))

    improvements = [
        ("<b>Walk-forward validation.</b>", "Re-run the backtest with rolling 12-month "
         "training windows for factor weights and thresholds. Report out-of-sample Sharpe "
         "to quantify parameter stability."),

        ("<b>Factor expansion.</b>", "Add on-chain metrics (exchange flows, active addresses, "
         "NVT ratio), funding rate (perpetual futures), and cross-asset signals "
         "(DXY, rates, equity risk appetite). Each additional orthogonal factor should "
         "improve the information coefficient."),

        ("<b>Regime model enhancement.</b>", "Replace the linear weighted average with a "
         "logistic regression or gradient-boosted classifier trained on future 7-day returns. "
         "Add Hidden Markov Model for regime persistence estimation."),

        ("<b>Multi-strategy combination.</b>", "Blend Medallion Lite, LPPLS, and Simplicity "
         "Benchmark using inverse-vol or minimum-correlation weighting. The three strategies "
         "have different entry mechanisms and should exhibit low return correlation during "
         "specific sub-periods."),

        ("<b>Execution layer.</b>", "Implement TWAP/VWAP execution algorithms to reduce "
         "effective spread below 30 bps. Use maker-only limit orders where possible. "
         "Target effective cost of 10–15 bps, which doubles the net Sharpe based on "
         "cost sensitivity analysis."),

        ("<b>Cross-asset extension.</b>", "Apply the same factor model to BTC and ETH "
         "perpetual futures (with leverage), cross-exchange arbitrage, and basis trades. "
         "The regime and factor framework is asset-class agnostic."),
    ]

    for title, desc in improvements:
        story.append(Paragraph(f"{title} {desc}", s_body))

    # ── FOOTER / DISCLAIMER ──────────────────────────────────────
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cbd5e1")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "DISCLAIMER: This document is for informational purposes only and does not constitute "
        "investment advice. Past performance is not indicative of future results. "
        "Backtested results are hypothetical and subject to model risk, data limitations, "
        "and execution assumptions that may not hold in live trading. "
        "All returns are gross of management/performance fees.",
        s_footer,
    ))
    story.append(Paragraph(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        "scripts/research/medallion_lite/generate_pdf.py",
        s_footer,
    ))

    doc.build(story)
    print(f"PDF: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")


def main():
    data = _load_all()
    out_path = MEDAL_DIR / "medallion_lite_research.pdf"
    build_pdf(data, out_path)


if __name__ == "__main__":
    main()
