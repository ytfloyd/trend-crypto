"""K2-branded PDF: FULL Medallion Lite strategy card + FULL performance tearsheet.

Built from the HONEST point-in-time (survivorship-free) run. Matches K2 branding
(navy #1e3a5f headers, blue #3b82f6 accents, dark charts, CONFIDENTIAL footer).

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
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.lib.colors import HexColor  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

C_MEDAL, C_BTC, C_BG, C_GRID, C_TEXT = "#3b82f6", "#94a3b8", "#0f172a", "#334155", "#f1f5f9"
NAVY, BLUE, SLATE = "#1e3a5f", "#3b82f6", "#64748b"
GREEN, RED = "#22c55e", "#ef4444"
OUT = ROOT / "scripts" / "research" / "medallion_lite" / "output" / "medallion_lite_card_and_tearsheet.pdf"
PARAMS = {"entry_threshold": 0.65, "trailing_stop_pct": 0.15}

# ----------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------
def _ax(ax, title):
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
    p = {}
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.plot(strat_eq.index, strat_eq.values, color=C_MEDAL, lw=1.6, label="Medallion Lite (PIT)")
    ax.plot(btc_eq.index, btc_eq.values, color=C_BTC, lw=1.2, label="BTC buy & hold")
    ax.set_yscale("log"); _ax(ax, "Cumulative return (log) — survivorship-free")
    ax.legend(fontsize=7, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
    p["equity"] = _save(fig, tmp, "equity.png")

    dd = strat_eq / strat_eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.fill_between(dd.index, dd.values, 0, color=C_MEDAL, alpha=0.5)
    _ax(ax, "Drawdown (underwater)")
    p["dd"] = _save(fig, tmp, "dd.png")

    m = (1 + daily).resample("ME").prod() - 1
    piv = pd.DataFrame({"y": m.index.year, "mo": m.index.month, "r": m.values}).pivot_table(
        index="y", columns="mo", values="r")
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-0.4, vmax=0.4)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=7, color=C_TEXT)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7, color=C_TEXT)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6,
                        color="#0f172a" if abs(v) < 0.25 else "white")
    _ax(ax, "Monthly returns (%)"); ax.grid(False)
    p["monthly"] = _save(fig, tmp, "monthly.png")

    yr = (1 + daily).resample("YE").prod() - 1
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.bar([str(t.year) for t in yr.index], yr.values * 100,
           color=[GREEN if v >= 0 else RED for v in yr.values])
    _ax(ax, "Annual return (%)"); ax.tick_params(axis="x", rotation=45)
    p["annual"] = _save(fig, tmp, "annual.png")

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.hist(daily.dropna().values * 100, bins=60, color=C_MEDAL, alpha=0.8)
    _ax(ax, "Daily return distribution (%)")
    p["dist"] = _save(fig, tmp, "dist.png")

    dn = daily.where(daily < 0)
    rsor = (daily.rolling(365, min_periods=120).mean() * 365) / (
        dn.rolling(365, min_periods=120).std() * np.sqrt(365))
    rshp = (daily.rolling(365, min_periods=120).mean() * 365) / (
        daily.rolling(365, min_periods=120).std() * np.sqrt(365))
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.plot(rsor.index, rsor.values, color=C_MEDAL, lw=1.3, label="Sortino")
    ax.plot(rshp.index, rshp.values, color=C_BTC, lw=1.0, label="Sharpe")
    ax.axhline(2.0, color=GREEN, lw=0.8, ls="--", alpha=0.7)
    _ax(ax, "Rolling 1y Sortino & Sharpe (dashed = 2.0)")
    ax.legend(fontsize=7, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
    p["rolling"] = _save(fig, tmp, "rolling.png")
    return p


# ----------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------
def full_stats(daily, bench=None, lo=None) -> dict:
    d = (daily if lo is None else daily[daily.index >= lo]).dropna()
    m = compute_metrics((1 + d).cumprod())
    mo = (1 + d).resample("ME").prod() - 1
    out = {
        "Total return": f"{m['total_return']*100:,.0f}%", "CAGR": f"{m['cagr']*100:.0f}%",
        "Ann. volatility": f"{m['vol']*100:.0f}%", "Sharpe": f"{m['sharpe']:.2f}",
        "Sortino": f"{m['sortino']:.2f}", "Calmar": f"{m['calmar']:.2f}",
        "Max drawdown": f"{m['max_dd']*100:.0f}%", "Daily hit rate": f"{m['hit_rate']*100:.0f}%",
        "Best day": f"{d.max()*100:.1f}%", "Worst day": f"{d.min()*100:.1f}%",
        "Best month": f"{mo.max()*100:.0f}%", "Worst month": f"{mo.min()*100:.0f}%",
        "% positive months": f"{(mo > 0).mean()*100:.0f}%", "Skew (daily)": f"{m['skewness']:.2f}",
        "Excess kurtosis": f"{m['kurtosis']:.2f}", "Trading days": f"{m['n_days']:,}",
    }
    if bench is not None:
        b = bench.reindex(d.index).fillna(0.0)
        out["Corr to BTC"] = f"{d.corr(b):.2f}"
        out["Beta to BTC"] = f"{(np.cov(d, b)[0,1]/np.var(b)):.2f}" if np.var(b) > 0 else "—"
    return out


def build():
    composite_pit, rw, regime = wf.load_pit()
    daily = wf.config_daily_returns(composite_pit, rw, regime, PARAMS)
    btc = wf._daily(rw["BTC-USD"]).reindex(daily.index).fillna(0.0)
    strat_eq = (1 + daily.fillna(0.0)).cumprod()
    btc_eq = (1 + btc).cumprod()

    s_full, s_oos = full_stats(daily, btc), full_stats(daily, btc, "2023-01-01")
    b_full, b_oos = full_stats(btc), full_stats(btc, None, "2023-01-01")

    st = getSampleStyleSheet()
    P = lambda n, **k: ParagraphStyle(n, parent=st["Normal"], **k)
    s_title = P("t", fontSize=22, leading=26, textColor=HexColor(NAVY), fontName="Helvetica-Bold", spaceAfter=2)
    s_sub = P("s", fontSize=10.5, leading=14, textColor=HexColor(SLATE), spaceAfter=12)
    s_h1 = P("h1", fontSize=13, leading=17, textColor=HexColor(NAVY), fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    s_body = P("b", fontSize=9, leading=13, textColor=HexColor("#1e293b"), spaceAfter=4)
    s_small = P("sm", fontSize=7.5, leading=10, textColor=HexColor(SLATE))
    s_mono = P("m", fontSize=7.5, leading=11, fontName="Courier", textColor=HexColor("#1e293b"))

    def kv(rows, w0=1.6):
        t = Table(rows, colWidths=[w0 * inch, (6.9 - w0) * inch])
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cbd5e1")),
                               ("LEFTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                               ("BACKGROUND", (0, 0), (0, -1), HexColor("#f1f5f9"))]))
        return t

    def bullets(items):
        return ListFlowable([ListItem(Paragraph(x, s_body), leftIndent=10) for x in items],
                            bulletType="bullet", start="•", bulletColor=HexColor(BLUE))

    story = []
    # ===================== STRATEGY CARD =====================
    story += [Paragraph("MEDALLION LITE", s_title),
              Paragraph("Strategy Card &amp; Performance Tearsheet &nbsp;·&nbsp; K2 TRADE ATLAS", s_sub)]
    story += [Paragraph("Identity", s_h1), kv([
        ["Registry ID", "2026-06-medallion-lite"],
        ["Class", "Cross-sectional crypto factor (long-biased, regime-gated, event-driven)"],
        ["Route", "cross_sectional"],
        ["Universe", "Coinbase USD pairs, point-in-time top-50 by ADV (survivorship-free)"],
        ["Bar frequency", "Hourly (flagship pipeline); daily for the registry signal_fn proxy"],
        ["Costs", "30 bps one-way"],
        ["Status", "S3 · validated with caveats"]])]
    story += [Paragraph("What it is", s_h1), Paragraph(
        "A Renaissance-inspired crypto cross-sectional factor strategy. Each bar it ranks the "
        "liquid universe on a 5-factor composite, gates exposure with an ensemble market-regime "
        "score, and runs an event-driven portfolio (enter / hold / exit) rather than continuous "
        "rebalancing.", s_body),
        bullets([
            "<b>Factors</b> (cross-sectionally ranked → [0,1]): momentum, volume surge, realized "
            "vol, proximity-to-high, risk-adjusted momentum.",
            "<b>Regime gate</b>: ensemble score on BTC (trend / vol) scales or halts exposure.",
            "<b>Portfolio</b>: enter when composite &gt; 0.65, exit &lt; 0.40; ≤25 positions; ≤10% "
            "per name; 15% trailing stop; max hold 14 days; rebalance every 24h."]),
        Paragraph("<b>Mandate note (long convexity):</b> this is a long-biased directional factor "
                  "book, not a pure long-gamma vehicle. Its convexity is indirect — the trailing "
                  "stop + regime gate truncate the left tail (−37% max DD vs BTC −50%/−77%). Treat "
                  "it as a cross-sectional momentum sleeve, not the trend/options convexity core.", s_body)]
    story += [Paragraph("Rulebook rules applied (K2 TRADE ATLAS)", s_h1), bullets([
        "<b>QF-01</b> cross-sectional momentum · <b>QF-07</b> vol-targeting (overlay)",
        "<b>QF-10 / CV-17</b> time-series convexity framing (synthetic-straddle payoff)",
        "<b>QF-21</b> data-snooping discipline (param-frozen walk-forward)",
        "<b>MR-09</b> crypto regime · factor intent per QF-16 / QF-09"])]
    story += [PageBreak()]

    pf = lambda m: [m["Sortino"], m["Sharpe"], m["CAGR"], m["Max drawdown"]]
    perf = Table([["", "Sortino", "Sharpe", "CAGR", "MaxDD"],
                  ["Medallion — OOS 2023-26"] + pf(s_oos),
                  ["Medallion — OOS + vol-target"] + [f"{full_stats(wf.vol_target(daily), btc, '2023-01-01')['Sortino']}",
                                                       full_stats(wf.vol_target(daily), btc, "2023-01-01")["Sharpe"],
                                                       full_stats(wf.vol_target(daily), btc, "2023-01-01")["CAGR"],
                                                       full_stats(wf.vol_target(daily), btc, "2023-01-01")["Max drawdown"]],
                  ["Medallion — FULL 2021-26"] + pf(s_full),
                  ["BTC buy&hold — OOS"] + pf(b_oos),
                  ["BTC buy&hold — FULL"] + pf(b_full)],
                 colWidths=[2.4 * inch, 1.0 * inch, 1.0 * inch, 1.05 * inch, 1.05 * inch])
    perf.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cbd5e1")),
                              ("BACKGROUND", (0, 0), (-1, 0), HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), rc.white),
                              ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rc.white, HexColor("#f8fafc")])]))
    story += [Paragraph("Validated performance (honest)", s_h1), perf, Spacer(1, 4),
              Paragraph("<b>Honest arc:</b> as-shipped flagship OOS Sortino 2.70 was look-ahead "
                        "survivorship (full-period-ADV universe); point-in-time corrects to 1.97; "
                        "param-frozen walk-forward = 2.03 (+vol-target 2.33). Per-fold OOS Sortino "
                        "decays 3.49 (2023) → 1.97 (2024) → 1.11 (2025-26) — recent edge weakening.", s_small)]
    story += [Paragraph("Risks &amp; caveats", s_h1), bullets([
        "<b>Recent edge decaying</b> — per-fold OOS Sortino 3.49 → 1.97 → 1.11; the 2.03 aggregate leans on 2023.",
        "<b>Vol-target uplift is partly leverage</b> (CAGR 101%→146%, MaxDD −39%→−42%); risk-adjusted improves but not free.",
        "<b>Registry signal_fn ≠ validated strategy</b>: signals.cross_sectional.medallion_lite is a "
        "simplified daily proxy; these metrics are the flagship hourly pipeline. Reconcile before live.",
        "<b>Unverified</b>: membership-table ADV ranking assumed point-in-time; no fill/slippage beyond "
        "flat 30 bps; small (9-config) walk-forward grid."])]
    story += [Paragraph("Provenance &amp; reproduce", s_h1), bullets([
        "Flagship pipeline: scripts/research/medallion_lite/ (factors, regime_ensemble, portfolio)",
        "Harnesses: scripts/research/k2_atlas/{run_medallion_pit, run_medallion_walkforward, medallion_pdf}.py",
        "Registry validation block: registry/alphas/2026-06-medallion-lite.yaml",
        "Data: coinbase_crypto_ohlcv_lake.duckdb (bars_1h / top-50 membership table)"]),
        Paragraph("Regenerate this document:", s_body),
        Paragraph("PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/medallion_pdf.py", s_mono)]
    story += [PageBreak()]

    # ===================== TEARSHEET =====================
    story += [Paragraph("Performance Tearsheet — full statistics", s_h1),
              Paragraph("Point-in-time (survivorship-free) universe · 30 bps costs · daily series.", s_small)]
    order = list(s_full.keys())  # metric row order
    head = ["Metric", "Medallion FULL", "Medallion OOS", "BTC FULL", "BTC OOS"]
    rows = [head] + [[k, s_full.get(k, "—"), s_oos.get(k, "—"), b_full.get(k, "—"), b_oos.get(k, "—")]
                     for k in order]
    stat_tbl = Table(rows, colWidths=[1.9 * inch, 1.3 * inch, 1.3 * inch, 1.2 * inch, 1.2 * inch])
    stat_tbl.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cbd5e1")),
                                  ("BACKGROUND", (0, 0), (-1, 0), HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), rc.white),
                                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                                  ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                                  ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rc.white, HexColor("#f8fafc")])]))
    story += [Spacer(1, 6), stat_tbl, PageBreak()]

    with tempfile.TemporaryDirectory() as tmp:
        ch = make_charts(strat_eq, btc_eq, daily, tmp)
        story += [Paragraph("Equity &amp; drawdown", s_h1),
                  Image(ch["equity"], width=6.7 * inch, height=2.5 * inch),
                  Image(ch["dd"], width=6.7 * inch, height=1.8 * inch), PageBreak()]
        story += [Paragraph("Monthly &amp; annual returns", s_h1),
                  Image(ch["monthly"], width=6.7 * inch, height=2.1 * inch)]
        row = Table([[Image(ch["annual"], width=3.25 * inch, height=1.8 * inch),
                      Image(ch["dist"], width=3.25 * inch, height=1.8 * inch)]], colWidths=[3.35 * inch, 3.35 * inch])
        story += [Spacer(1, 4), row, PageBreak()]
        story += [Paragraph("Rolling risk", s_h1),
                  Image(ch["rolling"], width=6.7 * inch, height=1.8 * inch)]

        footer_dt = datetime.now().strftime("%Y-%m-%d")

        def _footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7); canvas.setFillColor(HexColor(SLATE))
            canvas.drawString(0.75 * inch, 0.5 * inch, f"K2 TRADE ATLAS · Medallion Lite · CONFIDENTIAL · {footer_dt}")
            canvas.drawRightString(7.75 * inch, 0.5 * inch, f"p. {doc.page}")
            canvas.setStrokeColor(HexColor("#cbd5e1")); canvas.line(0.75 * inch, 0.62 * inch, 7.75 * inch, 0.62 * inch)
            canvas.restoreState()

        OUT.parent.mkdir(parents=True, exist_ok=True)
        SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                          leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                          title="Medallion Lite — Strategy Card & Tearsheet").build(
            story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"PDF -> {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
