"""K2-branded PDF: Medallion Lite strategy card + performance tearsheet (one doc).

Built from the HONEST point-in-time (survivorship-free) run. Matches the K2
reportlab/matplotlib branding used in medallion_lite/generate_pdf.py (deep-navy
headers #1e3a5f, blue accents #3b82f6, dark charts).

Run: PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/medallion_pdf.py
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(HERE), str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_medallion_walkforward as wf  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402
from reportlab.lib import colors as rc  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.lib.colors import HexColor  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# --- K2 brand (from medallion_lite/generate_pdf.py) ---
C_MEDAL, C_BTC, C_BG, C_GRID, C_TEXT = "#3b82f6", "#94a3b8", "#0f172a", "#334155", "#f1f5f9"
NAVY, BLUE, SLATE = "#1e3a5f", "#3b82f6", "#64748b"
OUT = ROOT / "scripts" / "research" / "medallion_lite" / "output" / "medallion_lite_card_and_tearsheet.pdf"
PARAMS = {"entry_threshold": 0.65, "trailing_stop_pct": 0.15}


def _dark_ax(ax, title):
    ax.set_facecolor(C_BG)
    ax.set_title(title, fontsize=10, fontweight="bold", color=C_TEXT, pad=8)
    ax.tick_params(colors=C_TEXT, labelsize=7)
    for s in ax.spines.values():
        s.set_color(C_GRID)
    ax.grid(True, color=C_GRID, alpha=0.4, lw=0.5)


def _save(fig, tmp, name):
    fig.patch.set_facecolor(C_BG)
    path = str(Path(tmp) / name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    return path


def make_charts(strat_eq, btc_eq, daily, tmp):
    paths = {}
    # 1. cumulative return (log) strategy vs BTC
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.plot(strat_eq.index, strat_eq.values, color=C_MEDAL, lw=1.6, label="Medallion Lite (PIT)")
    ax.plot(btc_eq.index, btc_eq.values, color=C_BTC, lw=1.2, label="BTC buy & hold")
    ax.set_yscale("log"); _dark_ax(ax, "Cumulative return (log) — survivorship-free")
    ax.legend(fontsize=7, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
    paths["equity"] = _save(fig, tmp, "equity.png")
    # 2. drawdown underwater
    dd = strat_eq / strat_eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.fill_between(dd.index, dd.values, 0, color=C_MEDAL, alpha=0.5)
    _dark_ax(ax, "Drawdown"); ax.set_ylabel("DD", color=C_TEXT, fontsize=8)
    paths["dd"] = _save(fig, tmp, "dd.png")
    # 3. monthly returns heatmap
    m = (1 + daily).resample("ME").prod() - 1
    md = pd.DataFrame({"y": m.index.year, "mo": m.index.month, "r": m.values})
    piv = md.pivot_table(index="y", columns="mo", values="r")
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-0.4, vmax=0.4)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=7, color=C_TEXT)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7, color=C_TEXT)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6,
                        color="#0f172a" if abs(v) < 0.25 else "white")
    _dark_ax(ax, "Monthly returns (%)"); ax.grid(False)
    paths["monthly"] = _save(fig, tmp, "monthly.png")
    # 4. rolling 1y Sortino
    roll = daily.rolling(365, min_periods=120)
    dn = daily.where(daily < 0)
    rs = (roll.mean() * 365) / (dn.rolling(365, min_periods=120).std() * np.sqrt(365))
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.plot(rs.index, rs.values, color=C_MEDAL, lw=1.3)
    ax.axhline(2.0, color="#22c55e", lw=0.8, ls="--", alpha=0.7)
    _dark_ax(ax, "Rolling 1y Sortino (dashed = 2.0 target)")
    paths["rsortino"] = _save(fig, tmp, "rsortino.png")
    return paths


def _metrics(daily, lo=None):
    d = daily if lo is None else daily[daily.index >= lo]
    return compute_metrics((1 + d.dropna()).cumprod())


def build():
    composite_pit, rw, regime = wf.load_pit()
    daily = wf.config_daily_returns(composite_pit, rw, regime, PARAMS)
    daily_vt = wf.vol_target(daily)
    btc = wf._daily(rw["BTC-USD"]).reindex(daily.index).fillna(0.0)
    strat_eq = (1 + daily.fillna(0.0)).cumprod()
    btc_eq = (1 + btc).cumprod()
    sf, so = _metrics(daily), _metrics(daily, "2023-01-01")
    bf, bo = _metrics(btc), _metrics(btc, "2023-01-01")
    vo = _metrics(daily_vt, "2023-01-01")

    st = getSampleStyleSheet()
    H = lambda n, **k: ParagraphStyle(n, parent=st["Normal"], **k)
    title = H("t", fontSize=22, leading=26, textColor=HexColor(NAVY), spaceAfter=2, fontName="Helvetica-Bold")
    sub = H("s", fontSize=10.5, leading=14, textColor=HexColor(SLATE), spaceAfter=12)
    h1 = H("h1", fontSize=13, leading=17, textColor=HexColor(NAVY), spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
    body = H("b", fontSize=9, leading=13, textColor=HexColor("#1e293b"), spaceAfter=4)
    small = H("sm", fontSize=7.5, leading=10, textColor=HexColor(SLATE))

    def tbl(data, widths, header=True):
        ts = [("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
              ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [rc.white, HexColor("#f8fafc")])]
        if header:
            ts += [("BACKGROUND", (0, 0), (-1, 0), HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), rc.white),
                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t = Table(data, colWidths=widths); t.setStyle(TableStyle(ts)); return t

    story = [
        Paragraph("MEDALLION LITE", title),
        Paragraph("Strategy Card &amp; Performance Tearsheet &nbsp;·&nbsp; K2 TRADE ATLAS", sub),
        Paragraph("Cross-sectional crypto factor · regime-gated · event-driven. Metrics are "
                  "survivorship-free (point-in-time top-50 universe), 30bps costs, daily.", body),
    ]
    # identity
    story += [Paragraph("Identity", h1), tbl([
        ["Registry ID", "2026-06-medallion-lite"], ["Route", "cross_sectional"],
        ["Universe", "Coinbase USD, point-in-time top-50 by ADV"],
        ["Bars / Costs", "Hourly (flagship) · 30 bps one-way"],
        ["Rulebook rules", "QF-01, QF-07, QF-10, CV-17, QF-21, MR-09"]],
        [1.5 * inch, 5.0 * inch], header=False)]
    # what it is
    story += [Paragraph("Strategy", h1),
              Paragraph("Ranks the liquid universe each bar on a 5-factor composite (momentum, "
                        "volume surge, realized vol, proximity-to-high, risk-adjusted momentum), "
                        "gates exposure with an ensemble BTC regime score, and runs an event-driven "
                        "portfolio: enter &gt; 0.65 / exit &lt; 0.40, ≤25 positions, ≤10%/name, 15% "
                        "trailing stop, ≤14-day hold. <b>Mandate note:</b> long-biased factor, not a "
                        "pure long-gamma vehicle — convexity comes from the trailing-stop + regime "
                        "left-tail truncation (−37% DD vs BTC −50%/−77%).", body)]
    # performance
    pf = lambda m: [f"{m['sortino']:.2f}", f"{m['sharpe']:.2f}", f"{m['cagr']:.0%}", f"{m['max_dd']:.0%}"]
    story += [Paragraph("Validated performance (honest)", h1), tbl(
        [["", "Sortino", "Sharpe", "CAGR", "MaxDD"],
         ["Medallion — OOS 2023-26"] + pf(so),
         ["Medallion — OOS + vol-target"] + [f"{vo['sortino']:.2f}", f"{vo['sharpe']:.2f}", f"{vo['cagr']:.0%}", f"{vo['max_dd']:.0%}"],
         ["Medallion — FULL 2021-26"] + pf(sf),
         ["BTC buy&hold — OOS"] + pf(bo),
         ["BTC buy&hold — FULL"] + pf(bf)],
        [2.4 * inch, 1.0 * inch, 1.0 * inch, 1.05 * inch, 1.05 * inch])]
    story += [Spacer(1, 4), Paragraph(
        "Honest arc: as-shipped flagship OOS Sortino 2.70 was look-ahead survivorship "
        "(full-period-ADV universe); point-in-time corrects to 1.97; param-frozen walk-forward "
        "= 2.03 (+vol-target 2.33). Per-fold OOS Sortino decays 3.49→1.97→1.11 (recent edge weakening).", small)]

    with tempfile.TemporaryDirectory() as tmp:
        ch = make_charts(strat_eq, btc_eq, daily, tmp)
        story += [Spacer(1, 8), Image(ch["equity"], width=6.6 * inch, height=2.45 * inch),
                  Image(ch["dd"], width=6.6 * inch, height=1.8 * inch), PageBreak()]
        story += [Paragraph("Monthly returns &amp; rolling risk", h1),
                  Image(ch["monthly"], width=6.6 * inch, height=2.15 * inch),
                  Image(ch["rsortino"], width=6.6 * inch, height=1.8 * inch)]
        story += [Paragraph("Risks &amp; caveats", h1)]
        for c in ["Recent edge decaying — per-fold OOS Sortino 3.49→1.97→1.11.",
                  "Vol-target uplift partly leverage (CAGR 101%→146%, MaxDD −39%→−42%).",
                  "Registry signal_fn is a simplified daily proxy; metrics are the flagship hourly pipeline.",
                  "Unverified: membership ADV ranking assumed point-in-time; flat 30bps; small param grid."]:
            story.append(Paragraph(f"•&nbsp; {c}", body))
        story += [Paragraph("Provenance", h1), Paragraph(
            "scripts/research/medallion_lite/ (flagship) · scripts/research/k2_atlas/"
            "{run_medallion_pit,run_medallion_walkforward,medallion_pdf}.py · "
            "registry/alphas/2026-06-medallion-lite.yaml · coinbase_crypto_ohlcv_lake.duckdb", small)]

        footer_dt = datetime.now().strftime("%Y-%m-%d")

        def _footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7); canvas.setFillColor(HexColor(SLATE))
            canvas.drawString(0.75 * inch, 0.5 * inch, f"K2 TRADE ATLAS · Medallion Lite · CONFIDENTIAL · {footer_dt}")
            canvas.drawRightString(7.75 * inch, 0.5 * inch, f"p. {doc.page}")
            canvas.setStrokeColor(HexColor("#cbd5e1")); canvas.line(0.75 * inch, 0.62 * inch, 7.75 * inch, 0.62 * inch)
            canvas.restoreState()

        OUT.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.7 * inch,
                                bottomMargin=0.75 * inch, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                                title="Medallion Lite — Strategy Card & Tearsheet")
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"PDF -> {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
