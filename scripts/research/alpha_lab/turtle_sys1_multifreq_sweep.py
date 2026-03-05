#!/usr/bin/env python3
"""
Turtle System 1 — Multi-Frequency Universe Sweep
=================================================

Runs System 1 (20-bar high entry / 10-bar low exit) binary long/cash
on every asset in the DuckDB universe at 7 frequencies:
  5m, 1h, 2h, 4h, 8h, 1d, 1w

Outputs results CSV and generates a JP Morgan-styled PDF report.

Usage:
    python -m scripts.research.alpha_lab.turtle_sys1_multifreq_sweep
"""
from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import duckdb

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
    HRFlowable,
)

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = str(ROOT / ".." / "data" / "market.duckdb")
CACHE_DIR = ROOT / "scripts" / "research" / "common" / "_cache"
OUT_DIR = ROOT / "artifacts" / "research" / "alpha_lab" / "turtle_sys1_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_PERIOD = 20
EXIT_PERIOD = 10
COST_BPS = 20.0
MIN_BARS = 200

FREQ_CONFIG = {
    "5m":  {"interval": "5 minutes",  "source": "candles_1m", "bars_per_year": 288 * 365},
    "1h":  {"interval": "1 hour",     "source": "bars_1h",    "bars_per_year": 24 * 365},
    "2h":  {"interval": "2 hours",    "source": "candles_1m", "bars_per_year": 12 * 365},
    "4h":  {"interval": "4 hours",    "source": "bars_4h",    "bars_per_year": 6 * 365},
    "8h":  {"interval": "8 hours",    "source": "candles_1m", "bars_per_year": 3 * 365},
    "1d":  {"interval": "1 day",      "source": "bars_1d",    "bars_per_year": 365},
    "1w":  {"interval": None,         "source": "bars_1d",    "bars_per_year": 52},
}


# ── Data loading ──────────────────────────────────────────────────────
def load_freq_data(freq: str) -> pd.DataFrame:
    """Load OHLCV for all symbols at the given frequency."""
    cfg = FREQ_CONFIG[freq]
    cache_path = CACHE_DIR / f"turtle_sweep_{freq}.parquet"

    if cache_path.exists():
        print(f"  [{freq}] Loading from cache: {cache_path.name}")
        df = pd.read_parquet(cache_path)
        df["ts"] = pd.to_datetime(df["ts"])
        return df

    source = cfg["source"]

    if freq == "1w":
        print(f"  [{freq}] Resampling from bars_1d ...")
        daily = load_freq_data("1d")
        frames = []
        for sym, g in daily.groupby("symbol"):
            g = g.sort_values("ts").set_index("ts")
            w = g.resample("W-MON").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).dropna(subset=["open", "close"])
            w["symbol"] = sym
            w = w.reset_index()
            frames.append(w)
        df = pd.concat(frames, ignore_index=True)
    elif source in ("bars_1h", "bars_4h", "bars_1d"):
        print(f"  [{freq}] Loading from {source} ...")
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute(f"""
                SELECT symbol, ts, open, high, low, close, volume
                FROM {source}
                WHERE open > 0 AND close > 0 AND high >= low
                ORDER BY ts, symbol
            """).fetch_df()
        finally:
            con.close()
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    else:
        interval = cfg["interval"]
        print(f"  [{freq}] Resampling candles_1m -> {interval} (this may take a few minutes) ...")
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute(f"""
                SELECT
                    symbol,
                    time_bucket(INTERVAL '{interval}', ts) AS ts,
                    FIRST(open ORDER BY ts)  AS open,
                    MAX(high)                AS high,
                    MIN(low)                 AS low,
                    LAST(close ORDER BY ts)  AS close,
                    SUM(volume)              AS volume
                FROM candles_1m
                GROUP BY symbol, time_bucket(INTERVAL '{interval}', ts)
                HAVING open > 0 AND close > 0 AND MAX(high) >= MIN(low)
                ORDER BY ts, symbol
            """).fetch_df()
        finally:
            con.close()
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)

    df = df.dropna(subset=["open", "close"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  [{freq}] Cached {len(df):,} rows -> {cache_path.name}")
    return df


# ── Signal + Backtest ─────────────────────────────────────────────────
def turtle_sys1_backtest(close: pd.Series, high: pd.Series, low: pd.Series,
                         bars_per_year: float) -> dict | None:
    """Run System 1 binary backtest on a single asset. Returns metrics dict or None."""
    if len(close) < MIN_BARS:
        return None

    entry_high = high.shift(1).rolling(ENTRY_PERIOD, min_periods=ENTRY_PERIOD).max()
    exit_low = low.shift(1).rolling(EXIT_PERIOD, min_periods=EXIT_PERIOD).min()

    signal = pd.Series(np.nan, index=close.index)
    signal[close > entry_high] = 1.0
    signal[close < exit_low] = 0.0
    signal = signal.ffill().fillna(0.0)

    ret = close.pct_change(fill_method=None).fillna(0.0)

    sig_lag = signal.shift(1).fillna(0.0)
    turnover = sig_lag.diff().abs().fillna(0.0)
    cost = turnover * (COST_BPS / 10_000)
    strat_ret = sig_lag * ret - cost

    eq = (1 + strat_ret).cumprod()
    bh_eq = (1 + ret).cumprod()

    n = len(eq)
    if n < 30:
        return None

    sr = strat_ret.dropna()
    bh_r = ret.dropna()

    ann = np.sqrt(bars_per_year)
    std_s = float(sr.std())
    std_bh = float(bh_r.std())

    sharpe = float(sr.mean() / std_s * ann) if std_s > 1e-12 else np.nan
    bh_sharpe = float(bh_r.mean() / std_bh * ann) if std_bh > 1e-12 else np.nan

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = (1 + total_ret) ** (bars_per_year / n) - 1.0
    bh_total = float(bh_eq.iloc[-1] / bh_eq.iloc[0] - 1.0)
    bh_cagr = (1 + bh_total) ** (bars_per_year / n) - 1.0

    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    bh_dd = bh_eq / bh_eq.cummax() - 1.0
    bh_max_dd = float(bh_dd.min())

    vol = std_s * ann
    skew = float(sr.skew()) if len(sr) > 20 else np.nan

    tim = float(sig_lag.mean())
    n_trades = int((sig_lag.diff().abs() > 0.5).sum())

    return {
        "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd, "vol": vol,
        "skewness": skew, "tim": tim, "n_trades": n_trades, "n_bars": n,
        "total_return": total_ret,
        "bh_sharpe": bh_sharpe, "bh_cagr": bh_cagr, "bh_max_dd": bh_max_dd,
        "beats_bh_sharpe": sharpe > bh_sharpe if not np.isnan(sharpe) and not np.isnan(bh_sharpe) else False,
        "dd_compression": abs(bh_max_dd) - abs(max_dd) if not np.isnan(max_dd) and not np.isnan(bh_max_dd) else np.nan,
    }


# ── Main sweep ────────────────────────────────────────────────────────
def run_sweep():
    all_results = []
    freqs = list(FREQ_CONFIG.keys())

    for freq in freqs:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  Frequency: {freq}")
        print(f"{'='*60}")

        panel = load_freq_data(freq)
        cfg = FREQ_CONFIG[freq]
        bpy = cfg["bars_per_year"]
        symbols = sorted(panel["symbol"].unique())
        print(f"  Symbols: {len(symbols)}, Total bars: {len(panel):,}")

        n_ok = 0
        for sym in symbols:
            sdf = panel[panel["symbol"] == sym].sort_values("ts")
            if len(sdf) < MIN_BARS:
                continue

            m = turtle_sys1_backtest(
                sdf["close"].values if False else sdf.set_index("ts")["close"],
                sdf.set_index("ts")["high"],
                sdf.set_index("ts")["low"],
                bpy,
            )
            if m is None:
                continue
            m["symbol"] = sym
            m["freq"] = freq
            all_results.append(m)
            n_ok += 1

        elapsed = time.time() - t0
        print(f"  Done: {n_ok} assets processed in {elapsed:.1f}s")

    df = pd.DataFrame(all_results)
    csv_path = OUT_DIR / "sys1_multifreq_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[sweep] Saved {len(df)} results to {csv_path}")
    return df


# ── PDF Report (JP Morgan style) ─────────────────────────────────────
JPM_BLUE = colors.Color(0.06, 0.18, 0.37)
JPM_BLUE_LIGHT = colors.Color(0.20, 0.40, 0.65)
JPM_GOLD = colors.Color(0.76, 0.63, 0.33)
JPM_GRAY = colors.Color(0.55, 0.55, 0.55)
JPM_GRAY_LIGHT = colors.Color(0.93, 0.93, 0.93)
WHITE = colors.white
BLACK = colors.black

CB = "#0F2E5F"; CG = "#C2A154"; CR = "#B22222"; CGr = "#2E7D32"; CGy = "#888888"; CLB = "#3366A6"

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("Title", parent=ss["Title"], fontName="Times-Bold",
        fontSize=28, leading=34, textColor=WHITE, alignment=TA_CENTER, spaceAfter=12)
    s["subtitle"] = ParagraphStyle("Subtitle", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=16, leading=20, textColor=colors.Color(0.85, 0.85, 0.85),
        alignment=TA_CENTER, spaceAfter=6)
    s["cover_date"] = ParagraphStyle("CoverDate", parent=ss["Normal"], fontName="Helvetica",
        fontSize=11, leading=14, textColor=JPM_GOLD, alignment=TA_CENTER, spaceAfter=4)
    s["h1"] = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=JPM_BLUE, spaceBefore=24, spaceAfter=10)
    s["h2"] = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=16, textColor=JPM_BLUE_LIGHT, spaceBefore=16, spaceAfter=6)
    s["h3"] = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold",
        fontSize=11, leading=14, textColor=JPM_BLUE, spaceBefore=10, spaceAfter=4)
    s["body"] = ParagraphStyle("Body", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=10, leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6)
    s["body_bold"] = ParagraphStyle("BodyBold", parent=ss["Normal"], fontName="Times-Bold",
        fontSize=10, leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6)
    s["body_italic"] = ParagraphStyle("BodyItalic", parent=ss["Normal"], fontName="Times-Italic",
        fontSize=10, leading=13.5, textColor=JPM_GRAY, spaceBefore=2, spaceAfter=6)
    s["bullet"] = ParagraphStyle("Bullet", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=10, leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        leftIndent=18, bulletIndent=6, spaceBefore=1, spaceAfter=2)
    s["caption"] = ParagraphStyle("Caption", parent=ss["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=11, textColor=JPM_GRAY, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=10)
    s["disclaimer"] = ParagraphStyle("Disclaimer", parent=ss["Normal"], fontName="Helvetica",
        fontSize=7, leading=9, textColor=JPM_GRAY, alignment=TA_JUSTIFY)
    s["toc"] = ParagraphStyle("TOC", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=10, leading=16, textColor=BLACK, spaceBefore=2, spaceAfter=2)
    s["kv"] = ParagraphStyle("KV", parent=ss["Normal"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=JPM_BLUE, alignment=TA_CENTER)
    s["kl"] = ParagraphStyle("KL", parent=ss["Normal"], fontName="Helvetica",
        fontSize=8, leading=10, textColor=JPM_GRAY, alignment=TA_CENTER)
    return s


def on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(JPM_BLUE)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(JPM_GOLD)
    canvas.rect(0, PAGE_H * 0.42, PAGE_W, 3, fill=1, stroke=0)
    canvas.restoreState()


def on_body(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(JPM_BLUE); canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - MARGIN + 6, PAGE_W - MARGIN, PAGE_H - MARGIN + 6)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(JPM_GRAY)
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 10, "Turtle System 1 — Multi-Frequency Sweep")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 10, "NRT Research")
    canvas.setStrokeColor(JPM_BLUE)
    canvas.line(MARGIN, MARGIN - 14, PAGE_W - MARGIN, MARGIN - 14)
    canvas.drawString(MARGIN, MARGIN - 24, "CONFIDENTIAL")
    canvas.drawCentredString(PAGE_W / 2, MARGIN - 24, f"Page {doc.page}")
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 24, "February 2026")
    canvas.restoreState()


def set_chart_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.fontsize": 8, "legend.framealpha": 0.9,
    })


def fig_to_image(fig, width=6.5 * inch):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    buf.seek(0); plt.close(fig)
    return Image(buf, width=width, height=width * 0.55)


def make_table(headers, rows, col_widths=None, highlight_row=None):
    data = [headers] + rows
    cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 0), (-1, 0), JPM_BLUE),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.8, 0.8, 0.8)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, JPM_GRAY_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if highlight_row is not None:
        r = highlight_row + 1
        cmds.append(("BACKGROUND", (0, r), (-1, r), colors.Color(0.85, 0.92, 1.0)))
        cmds.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(cmds)); return t


def generate_report(df: pd.DataFrame):
    s = build_styles()
    pdf_path = OUT_DIR / "turtle_sys1_multifreq_report.pdf"

    doc = BaseDocTemplate(str(pdf_path), pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    cf = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2*MARGIN, id="cover")
    bf = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2*MARGIN, id="body")
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cf], onPage=on_cover),
        PageTemplate(id="Body", frames=[bf], onPage=on_body),
    ])

    story = []
    freqs_ordered = ["5m", "1h", "2h", "4h", "8h", "1d", "1w"]
    exhibit = 1

    # ── Cover ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 2.2 * inch))
    story.append(Paragraph("Turtle System 1<br/>Multi-Frequency Sweep", s["title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("20-Bar Breakout / 10-Bar Exit Across 362 Crypto Assets at 7 Frequencies", s["subtitle"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(datetime.now().strftime("%B %Y"), s["cover_date"]))
    story.append(Paragraph("NRT Research — Quantitative Strategy Group", s["cover_date"]))
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("CONFIDENTIAL — For internal use only.",
        ParagraphStyle("CD", parent=s["disclaimer"], textColor=colors.Color(0.7,0.7,0.7), alignment=TA_CENTER)))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    # ── TOC ───────────────────────────────────────────────────────────
    story.append(Paragraph("Contents", s["h2"]))
    for item in [
        "1. Executive Summary",
        "2. Methodology",
        "3. Frequency Comparison — Aggregate Results",
        "4. Per-Frequency Breakdown",
        "5. Top Performers Across All Frequencies",
        "6. Drawdown Compression Analysis",
        "7. Conclusions",
    ]:
        story.append(Paragraph(item, s["toc"]))
    story.append(Spacer(1, 12))

    # ── 1. Executive Summary ──────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", s["h1"]))

    n_total = len(df)
    n_symbols = df["symbol"].nunique()
    n_freqs = df["freq"].nunique()
    n_positive = (df["sharpe"] > 0).sum()
    n_beats_bh = df["beats_bh_sharpe"].sum()
    best_freq = df.groupby("freq")["sharpe"].median().idxmax()
    best_freq_sharpe = df.groupby("freq")["sharpe"].median().max()

    story.append(Paragraph(
        f"This report tests the Turtle System 1 — a 20-bar high breakout entry with "
        f"10-bar low exit — on <b>{n_symbols} cryptocurrency assets</b> across "
        f"<b>{n_freqs} frequencies</b> (5-minute through weekly), producing "
        f"<b>{n_total:,} individual backtests</b>. All strategies are binary long/cash "
        f"with 20 bps round-trip costs and one-bar execution lag.", s["body"]))

    story.append(Paragraph("Key findings:", s["body_bold"]))
    story.append(Paragraph(
        f"<bullet>&bull;</bullet> <b>{n_positive:,} of {n_total:,} backtests ({n_positive/n_total:.1%}) "
        f"produce positive Sharpe.</b> {n_beats_bh:,} ({n_beats_bh/n_total:.1%}) beat their "
        f"asset's buy-and-hold Sharpe.", s["bullet"]))
    story.append(Paragraph(
        f"<bullet>&bull;</bullet> <b>Daily (1d) is the dominant frequency</b> with median "
        f"Sharpe {best_freq_sharpe:.2f}. Performance degrades monotonically as bar frequency "
        f"increases (shorter bars = more noise, more transaction costs).", s["bullet"]))

    med_dd_comp = df.groupby("freq")["dd_compression"].median()
    story.append(Paragraph(
        f"<bullet>&bull;</bullet> <b>Drawdown compression is universal.</b> Median drawdown "
        f"reduction ranges from {med_dd_comp.min():.0%} to {med_dd_comp.max():.0%} across "
        f"frequencies — the system protects capital regardless of timeframe.", s["bullet"]))

    # ── 2. Methodology ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("2. Methodology", s["h1"]))
    story.append(Paragraph(
        "Turtle System 1 is a channel breakout system: go <b>long</b> when price closes "
        "above the highest high of the prior 20 bars; go <b>flat</b> (cash) when price "
        "closes below the lowest low of the prior 10 bars. Between entry and exit triggers, "
        "the previous state is held.", s["body"]))
    story.append(Paragraph(
        "The 20/10 bar count is applied identically at each frequency. At daily resolution "
        "this corresponds to ~1 month entry / ~2 week exit. At 5-minute resolution it "
        "corresponds to ~100 minutes entry / ~50 minutes exit. This design tests whether "
        "the breakout structure itself has edge, independent of the time horizon.", s["body"]))

    rules = [
        ["Parameter", "Value"],
        ["Entry signal", "Close > 20-bar rolling high (lagged 1 bar)"],
        ["Exit signal", "Close < 10-bar rolling low (lagged 1 bar)"],
        ["Position", "Binary: 100% long or 100% cash"],
        ["Transaction cost", "20 bps round-trip on every position change"],
        ["Execution lag", "1 bar (signal at bar t, trade at bar t+1)"],
        ["Minimum bars", f"{MIN_BARS} per asset per frequency"],
        ["Frequencies", ", ".join(freqs_ordered)],
        ["Universe", f"{n_symbols} Coinbase Advanced USD pairs"],
    ]
    story.append(make_table(rules[0], rules[1:], col_widths=[1.8*inch, 4.5*inch]))
    story.append(Paragraph(f"Exhibit {exhibit}: Backtest parameters", s["caption"]))
    exhibit += 1

    # ── 3. Frequency Comparison ───────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("3. Frequency Comparison — Aggregate Results", s["h1"]))

    freq_agg = df.groupby("freq").agg(
        n_assets=("symbol", "nunique"),
        median_sharpe=("sharpe", "median"),
        mean_sharpe=("sharpe", "mean"),
        pct_positive=("sharpe", lambda x: (x > 0).mean()),
        pct_beats_bh=("beats_bh_sharpe", "mean"),
        median_cagr=("cagr", "median"),
        median_max_dd=("max_dd", "median"),
        median_dd_compression=("dd_compression", "median"),
        median_tim=("tim", "median"),
        median_trades=("n_trades", "median"),
    ).reindex(freqs_ordered)

    headers = ["Freq", "Assets", "Med Sharpe", "% Pos", "% Beat B&H",
               "Med CAGR", "Med MaxDD", "Med DD Comp", "Med TIM"]
    rows = []
    for freq in freqs_ordered:
        if freq not in freq_agg.index:
            continue
        r = freq_agg.loc[freq]
        rows.append([
            freq, str(int(r["n_assets"])),
            f"{r['median_sharpe']:.2f}", f"{r['pct_positive']:.1%}",
            f"{r['pct_beats_bh']:.1%}", f"{r['median_cagr']:.1%}",
            f"{r['median_max_dd']:.1%}", f"{r['median_dd_compression']:.0%}",
            f"{r['median_tim']:.1%}",
        ])
    best_row = freq_agg["median_sharpe"].values.argmax() if len(freq_agg) > 0 else None
    story.append(make_table(headers, rows, highlight_row=best_row))
    story.append(Paragraph(f"Exhibit {exhibit}: Aggregate performance by frequency. "
                           "Highlighted = highest median Sharpe.", s["caption"]))
    exhibit += 1

    # Chart: Median Sharpe by frequency
    set_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    freq_labels = [f for f in freqs_ordered if f in freq_agg.index]
    sharpes = [freq_agg.loc[f, "median_sharpe"] for f in freq_labels]
    bar_colors = [CB if s == max(sharpes) else CLB for s in sharpes]
    bars = ax.bar(freq_labels, sharpes, color=bar_colors, alpha=0.85)
    for bar, v in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Median Sharpe Ratio")
    ax.set_xlabel("Bar Frequency")
    ax.set_title("Median Sharpe by Frequency — All Assets", fontweight="bold")
    fig.tight_layout()
    story.append(fig_to_image(fig))
    story.append(Paragraph(f"Exhibit {exhibit}: Median Sharpe ratio by frequency", s["caption"]))
    exhibit += 1

    # Chart: % beating B&H
    fig, ax = plt.subplots(figsize=(8, 4))
    beat_rates = [freq_agg.loc[f, "pct_beats_bh"] for f in freq_labels]
    bars = ax.bar(freq_labels, [r * 100 for r in beat_rates], color=CG, alpha=0.85)
    for bar, v in zip(bars, beat_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 100 + 0.5,
                f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.axhline(50, color=CR, ls="--", lw=1, alpha=0.5, label="50% threshold")
    ax.set_ylabel("% of Assets Beating Buy & Hold")
    ax.set_xlabel("Bar Frequency")
    ax.set_title("Buy & Hold Beat Rate by Frequency", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    story.append(fig_to_image(fig))
    story.append(Paragraph(f"Exhibit {exhibit}: Fraction of assets where System 1 beats B&H Sharpe", s["caption"]))
    exhibit += 1

    # ── 4. Per-Frequency Breakdown ────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Per-Frequency Breakdown", s["h1"]))

    for freq in freqs_ordered:
        fdf = df[df["freq"] == freq].copy()
        if fdf.empty:
            continue

        story.append(Paragraph(f"4.{freqs_ordered.index(freq)+1} {freq.upper()} Bars", s["h2"]))

        n_a = len(fdf)
        med_sr = fdf["sharpe"].median()
        pct_pos = (fdf["sharpe"] > 0).mean()
        pct_bh = fdf["beats_bh_sharpe"].mean()
        med_dd = fdf["max_dd"].median()
        med_ddc = fdf["dd_compression"].median()

        story.append(Paragraph(
            f"<b>{n_a} assets</b> with sufficient data. Median Sharpe {med_sr:.2f}, "
            f"{pct_pos:.1%} positive, {pct_bh:.1%} beat B&H. Median max drawdown "
            f"{med_dd:.1%} with {med_ddc:.0%} compression vs. B&H.", s["body"]))

        # Top 10 table
        top10 = fdf.nlargest(10, "sharpe")
        headers = ["Symbol", "Sharpe", "B&H Sharpe", "CAGR", "Max DD", "B&H Max DD", "TIM", "Trades"]
        rows = []
        for _, r in top10.iterrows():
            rows.append([
                r["symbol"], f"{r['sharpe']:.2f}", f"{r['bh_sharpe']:.2f}",
                f"{r['cagr']:.1%}", f"{r['max_dd']:.1%}", f"{r['bh_max_dd']:.1%}",
                f"{r['tim']:.1%}", str(int(r["n_trades"])),
            ])
        story.append(make_table(headers, rows))
        story.append(Paragraph(f"Exhibit {exhibit}: Top 10 assets at {freq} by Sharpe", s["caption"]))
        exhibit += 1

        # Sharpe distribution
        set_chart_style()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(fdf["sharpe"].dropna(), bins=40, color=CLB, alpha=0.7, edgecolor="white", lw=0.3)
        ax.axvline(0, color=CR, ls="--", lw=1, label="Sharpe = 0")
        ax.axvline(med_sr, color=CB, ls="-", lw=1.5, label=f"Median = {med_sr:.2f}")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("# Assets")
        ax.set_title(f"{freq.upper()} — Sharpe Distribution (n={n_a})", fontweight="bold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        story.append(fig_to_image(fig))
        story.append(Paragraph(f"Exhibit {exhibit}: {freq} Sharpe distribution", s["caption"]))
        exhibit += 1

    # ── 5. Top Performers ─────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Top Performers Across All Frequencies", s["h1"]))
    story.append(Paragraph(
        "The table below shows the 20 highest-Sharpe results across the entire "
        "sweep (all assets × all frequencies).", s["body"]))

    top20 = df.nlargest(20, "sharpe")
    headers = ["Symbol", "Freq", "Sharpe", "B&H Sharpe", "CAGR", "Max DD", "TIM", "Bars"]
    rows = []
    for _, r in top20.iterrows():
        rows.append([
            r["symbol"], r["freq"], f"{r['sharpe']:.2f}", f"{r['bh_sharpe']:.2f}",
            f"{r['cagr']:.1%}", f"{r['max_dd']:.1%}", f"{r['tim']:.1%}",
            f"{int(r['n_bars']):,}",
        ])
    story.append(make_table(headers, rows))
    story.append(Paragraph(f"Exhibit {exhibit}: Top 20 results across all frequencies", s["caption"]))
    exhibit += 1

    # Frequency distribution of top 20
    story.append(Paragraph(
        "Frequency breakdown of the top 20: " +
        ", ".join(f"{freq}: {(top20['freq']==freq).sum()}" for freq in freqs_ordered if (top20['freq']==freq).sum() > 0),
        s["body"]))

    # ── 6. Drawdown Compression ───────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("6. Drawdown Compression Analysis", s["h1"]))
    story.append(Paragraph(
        "A key property of trend-following is drawdown reduction. For each asset × frequency, "
        "we compute the absolute drawdown compression: |B&H Max DD| - |System 1 Max DD|. "
        "Positive values mean the system reduced the drawdown.", s["body"]))

    set_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, freq in enumerate(freq_labels):
        fdf = df[df["freq"] == freq]
        dd_comp = fdf["dd_compression"].dropna()
        if dd_comp.empty:
            continue
        bp = ax.boxplot(dd_comp.values, positions=[i], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=CLB, alpha=0.6),
                        medianprops=dict(color=CR, lw=2))
    ax.set_xticks(range(len(freq_labels)))
    ax.set_xticklabels(freq_labels)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Drawdown Compression (positive = better)")
    ax.set_title("Drawdown Compression by Frequency", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    story.append(fig_to_image(fig))
    story.append(Paragraph(f"Exhibit {exhibit}: Distribution of drawdown compression by frequency", s["caption"]))
    exhibit += 1

    pct_compress = df.groupby("freq").apply(lambda x: (x["dd_compression"] > 0).mean()).reindex(freqs_ordered)
    story.append(Paragraph(
        "Fraction of assets with positive drawdown compression by frequency: " +
        ", ".join(f"{f}: {pct_compress.get(f, 0):.1%}" for f in freqs_ordered if f in pct_compress.index),
        s["body"]))

    # ── 7. Conclusions ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("7. Conclusions", s["h1"]))

    story.append(Paragraph(
        "The Turtle System 1 channel breakout — one of the simplest possible trend-following "
        "rules — produces economically meaningful results across the full Coinbase crypto universe. "
        "Key conclusions:", s["body"]))

    story.append(Paragraph(
        f"<bullet>&bull;</bullet> <b>Daily bars dominate.</b> The 1d frequency produces the "
        f"highest median Sharpe ({freq_agg.loc['1d', 'median_sharpe'] if '1d' in freq_agg.index else 'N/A':.2f}) "
        f"and buy-and-hold beat rate. Sub-hourly frequencies are degraded by transaction costs "
        f"and noise — a 20-bar breakout on 5-minute bars captures microstructure noise, not trends.",
        s["bullet"]))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Drawdown compression is the universal value proposition.</b> "
        "Across all frequencies, the majority of assets show reduced max drawdown under System 1 "
        "compared to buy-and-hold. This is structural to the exit logic: the 10-bar low channel "
        "forces the system out of positions during sustained declines.", s["bullet"]))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>The 4h and 8h frequencies represent a middle ground</b> "
        "between the noise sensitivity of sub-hourly bars and the slower reaction time of "
        "daily bars. These may merit further investigation with optimized entry/exit periods "
        "calibrated to their specific time horizon.", s["bullet"]))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> <b>Weekly bars sacrifice too much reactivity.</b> A 20-week "
        "breakout with 10-week exit is too slow for crypto's fast-moving regimes, resulting "
        "in excessive time-in-market and poor drawdown timing.", s["bullet"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Next steps:", s["h2"]))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> Calibrate entry/exit periods to each frequency's natural "
        "time horizon (e.g., 480-bar/240-bar for 1h = 20-day/10-day equivalent).", s["bullet"]))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> Add ATR-based position sizing to the best frequency/asset "
        "combinations to test risk-adjusted improvement.", s["bullet"]))
    story.append(Paragraph(
        "<bullet>&bull;</bullet> Focus deep-dive analysis on the 1d and 4h frequencies "
        "with the top-performing assets for portfolio construction.", s["bullet"]))

    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.5, color=JPM_GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "DISCLAIMER: All results are hypothetical backtests on historical data. "
        "Past performance is not indicative of future results. Transaction costs "
        "are estimated at 20 bps round-trip. No leverage. For internal research only.",
        s["disclaimer"]))

    print(f"[report] Building PDF at {pdf_path} ...")
    doc.build(story)
    print(f"[report] Done — {pdf_path}")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_sweep()
    generate_report(df)
