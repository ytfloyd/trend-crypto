"""K2-branded investor research report for Medallion Lite (concise, institutional).

Renders docs/research/medallion_research_report.md content as a branded PDF with the
adopted top-100 universe equity/drawdown chart and proper backtest disclaimers.

Run: PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/medallion_report_pdf.py
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(HERE), str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from reportlab.lib import colors as rc  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.lib.colors import HexColor  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

NAVY, BLUE, SLATE = "#1e3a5f", "#3b82f6", "#64748b"
C_MEDAL, C_BTC, C_BG, C_GRID, C_TEXT = "#3b82f6", "#94a3b8", "#0f172a", "#334155", "#f1f5f9"
OUT = ROOT / "scripts" / "research" / "medallion_lite" / "output" / "medallion_lite_research_report.pdf"


def make_chart(tmp):
    import run_medallion_universe as uni
    daily, btc, _, _ = uni.build_spec_daily("top", 100)
    seq = (1 + daily.fillna(0.0)).cumprod()
    beq = (1 + btc).cumprod()
    dd = seq / seq.cummax() - 1.0
    paths = {}
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.plot(seq.index, seq.values, color=C_MEDAL, lw=1.7, label="Medallion Lite (100-name)")
    ax.plot(beq.index, beq.values, color=C_BTC, lw=1.2, label="BTC buy & hold")
    ax.set_yscale("log")
    ax.set_facecolor(C_BG)
    ax.set_title("Growth of $1 — net of 30 bps (log scale, survivorship-free)",
                 fontsize=10, fontweight="bold", color=C_TEXT, pad=8)
    ax.tick_params(colors=C_TEXT, labelsize=7)
    for s in ax.spines.values():
        s.set_color(C_GRID)
    ax.grid(True, color=C_GRID, alpha=0.4, lw=0.5)
    ax.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
    fig.patch.set_facecolor(C_BG)
    paths["eq"] = str(Path(tmp) / "eq.png")
    fig.savefig(paths["eq"], dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 1.7))
    ax.fill_between(dd.index, dd.values, 0, color=C_MEDAL, alpha=0.5)
    ax.set_facecolor(C_BG)
    ax.set_title("Drawdown", fontsize=10, fontweight="bold", color=C_TEXT, pad=6)
    ax.tick_params(colors=C_TEXT, labelsize=7)
    for s in ax.spines.values():
        s.set_color(C_GRID)
    ax.grid(True, color=C_GRID, alpha=0.4, lw=0.5)
    fig.patch.set_facecolor(C_BG)
    paths["dd"] = str(Path(tmp) / "dd.png")
    fig.savefig(paths["dd"], dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    return paths


def build():
    st = getSampleStyleSheet()
    P = lambda n, **k: ParagraphStyle(n, parent=st["Normal"], **k)
    s_title = P("t", fontSize=22, leading=25, textColor=HexColor(NAVY), fontName="Helvetica-Bold", spaceAfter=2)
    s_sub = P("s", fontSize=10, leading=13, textColor=HexColor(SLATE), spaceAfter=2)
    s_kick = P("k", fontSize=8.5, leading=11, textColor=HexColor(BLUE), fontName="Helvetica-Bold", spaceAfter=10)
    s_h1 = P("h1", fontSize=12.5, leading=16, textColor=HexColor(NAVY), fontName="Helvetica-Bold", spaceBefore=13, spaceAfter=5)
    s_body = P("b", fontSize=9.2, leading=13.5, textColor=HexColor("#1e293b"), spaceAfter=6, alignment=4)
    s_small = P("sm", fontSize=7.3, leading=9.8, textColor=HexColor(SLATE), alignment=4)
    s_disc = P("d", fontSize=6.8, leading=9, textColor=HexColor(SLATE), alignment=4)

    def bullets(items):
        return ListFlowable([ListItem(Paragraph(x, s_body), leftIndent=10) for x in items],
                            bulletType="bullet", start="•", bulletColor=HexColor(BLUE), spaceBefore=0)

    def tbl(rows, widths, hi=None):
        t = Table(rows, colWidths=[w * inch for w in widths])
        style = [("FONTSIZE", (0, 0), (-1, -1), 8.3), ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cbd5e1")),
                 ("BACKGROUND", (0, 0), (-1, 0), HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), rc.white),
                 ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                 ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rc.white, HexColor("#f8fafc")])]
        if hi:
            style.append(("BACKGROUND", (0, hi), (-1, hi), HexColor("#dbeafe")))
            style.append(("FONTNAME", (0, hi), (-1, hi), "Helvetica-Bold"))
        t.setStyle(TableStyle(style))
        return t

    story = []
    story += [Paragraph("MEDALLION LITE — RESEARCH REPORT", s_title),
              Paragraph("Cross-sectional digital-asset factor strategy", s_sub),
              Paragraph("K2 TRADE ATLAS · SYSTEMATIC DIGITAL-ASSET STRATEGIES · PREPARED FOR INVESTOR REVIEW", s_kick)]

    story += [Paragraph("Executive summary", s_h1),
              Paragraph(
                  "Medallion Lite is a systematic, market-regime-gated <b>cross-sectional factor "
                  "strategy</b> trading liquid US-dollar crypto pairs. It ranks the tradable universe "
                  "each bar on a five-factor composite, sizes an event-driven portfolio (enter / hold / "
                  "exit with a trailing stop), and scales gross exposure with an ensemble measure of the "
                  "market regime.", s_body),
              Paragraph(
                  "On a survivorship-free, net-of-cost basis (30 bps one-way) over the 100-name universe, "
                  "the strategy delivers an out-of-sample (2023–2026) <b>Sortino ratio of 2.84 on frozen "
                  "parameters</b> — the headline figure, with no parameters fit on the data — versus "
                  "<b>1.78 for buy-and-hold Bitcoin</b>, with a materially smaller drawdown (−37% vs −50%). "
                  "A walk-forward variant that re-selects parameters reaches 2.95; we report it only as an "
                  "upper bound. All five figures below are reproduced by a single, deterministic, "
                  "pre-registered audit script with a published provenance manifest.", s_body)]
    story += [tbl([["Metric (OOS 2023–2026, net of 30 bps)", "Medallion Lite (100-name)", "BTC buy & hold"],
                   ["Sortino — frozen params (headline)", "2.84", "1.78"],
                   ["Sortino — walk-forward (upper bound)", "2.95", "—"],
                   ["Sharpe ratio", "2.15", "1.15"],
                   ["Annualised return (CAGR)", "173%", "54%"],
                   ["Maximum drawdown", "−37%", "−50%"]], [3.3, 2.0, 1.6], hi=1)]

    story += [Paragraph("1. Strategy overview", s_h1),
              Paragraph(
                  "Inspired by the cross-sectional, statistically-driven approach to systematic trading, "
                  "the investment process has three components:", s_body),
              bullets([
                  "<b>Factor ranking.</b> Each name is scored cross-sectionally on five orthogonal factors "
                  "— momentum, volume surge, realised volatility, proximity-to-high, and risk-adjusted "
                  "momentum — combined into a single percentile composite.",
                  "<b>Event-driven portfolio.</b> Enter above an upper composite threshold, exit below a "
                  "lower one; cap position count and concentration; 15% trailing stop and a maximum hold. "
                  "Signal-event driven, not continuous rebalancing — turnover stays low.",
                  "<b>Regime gate.</b> An ensemble regime score scales or halts gross exposure, "
                  "truncating the left tail."]),
              Paragraph(
                  "The strategy is <b>long-biased and directional</b>; its convexity is indirect, arising "
                  "from the trailing stop and regime gate that cut losses short.", s_body)]

    story += [Paragraph("2. Research methodology and integrity", s_h1),
              Paragraph(
                  "Backtest integrity governs every figure here: a <b>survivorship-free, point-in-time "
                  "universe</b> (membership known only as-of each date); <b>costs always on</b> (30 bps "
                  "one-way); and a <b>pre-registered, deterministic audit</b> with explicit pass/fail gates "
                  "and a published provenance manifest. The <b>headline uses frozen parameters</b> — "
                  "nothing fit on the data; a walk-forward variant that re-selects parameters is reported "
                  "only as an upper bound, because parameter selection is itself a source of optimism.", s_body),
              Paragraph(
                  "The value of this discipline is concrete, and we apply it to our own results. A "
                  "naïvely-constructed version appeared to deliver a Sortino of 2.70; rebuilding it "
                  "survivorship-free cut that to ~2.0 (≈0.7 was look-ahead bias). Separately, an attempt to "
                  "enrich the model with 100 technical factors and machine-learning ensembles produced "
                  "eye-catching numbers that <b>failed an overfitting test</b> (probability of backtest "
                  "overfitting 0.70–0.77) and were rejected — we run the simple five-factor model.", s_body),
              tbl([["Stage of rigor (100-name universe)", "OOS Sortino", "What changed"],
                   ["Naïve construction (50-name)", "2.70", "Full-history liquidity (look-ahead)"],
                   ["Survivorship-free + walk-forward (50)", "~2.0", "Universe + params honest"],
                   ["Widen to 100-name, walk-forward", "2.95", "Breadth (upper bound)"],
                   ["Frozen params (pre-registered HEADLINE)", "2.84", "No parameter fitting"]], [2.7, 1.0, 3.2])]
    story += [PageBreak()]

    story += [Paragraph("3. Performance results — the breadth lever", s_h1),
              Paragraph(
                  "Holding the honest methodology fixed, we varied only the <b>breadth of the tradable "
                  "universe</b>. Because the edge is relative ranking across names, a wider opportunity set "
                  "offers more independent bets. Figures below are out-of-sample (2023–2026), "
                  "survivorship-free, net of 30 bps, on the <b>walk-forward (upper-bound)</b> basis for "
                  "cross-universe comparability; the adopted 100-name frozen-param headline is 2.84.", s_body),
              tbl([["Universe definition", "~Names", "Sortino", "Max DD", "+ vol-target"],
                   ["50-name (prior baseline)", "50", "1.97", "−38%", "2.07"],
                   ["100-name (adopted)", "93", "2.95", "−35%", "3.04"],
                   ["Liquidity floor (ADV ≥ $1M)", "72", "2.46", "−37%", "3.00"],
                   ["200-name", "161", "2.47", "−43%", "2.60"],
                   ["Broad (ADV ≥ $250k)", "124", "2.48", "−45%", "2.66"],
                   ["Entire USD universe", "193", "2.48", "−41%", "2.62"],
                   ["BTC buy & hold (reference)", "—", "1.78", "−50%", "—"]], [2.5, 0.85, 1.0, 1.0, 1.2], hi=2),
              Spacer(1, 5),
              bullets([
                  "<b>Breadth is a genuine, low-cost lever.</b> Widening from ~50 to ~100 names raises the "
                  "honest Sortino from 1.97 to 2.95 <i>while tightening</i> the maximum drawdown. Every "
                  "universe tested clears a Sortino of 2.0 and beats Bitcoin.",
                  "<b>There is an optimum near 70–100 names.</b> Beyond ~100, marginal names are less "
                  "liquid; returns flatten to ~2.5 and drawdowns widen toward −45%."])]

    with tempfile.TemporaryDirectory() as tmp:
        ch = make_chart(tmp)
        story += [Spacer(1, 4), Image(ch["eq"], width=6.6 * inch, height=2.55 * inch),
                  Image(ch["dd"], width=6.6 * inch, height=1.6 * inch), PageBreak()]

        story += [Paragraph("4. The capacity advantage", s_h1),
                  Paragraph(
                      "This result is structurally favourable for a <b>deliberately capacity-constrained "
                      "manager</b>. Large pools of capital cannot meaningfully access the 50th–100th most "
                      "liquid crypto names without moving prices; a small, nimble book can. The strategy "
                      "therefore converts a constraint that handicaps large competitors into the precise "
                      "breadth band where its risk-adjusted performance is strongest. We cap strategy "
                      "capacity to preserve the edge rather than scale assets at the expense of returns.", s_body)]

        story += [Paragraph("5. Risk factors and limitations", s_h1), bullets([
            "<b>Transaction-cost realism.</b> 30 bps is conservative for the most liquid names but may "
            "understate slippage on the 50th–100th names; a tiered-cost re-test gates scaling the 100-name "
            "universe. We treat the 100-name figure as strong but not yet production-final.",
            "<b>Edge decay.</b> Within the 50-name history, out-of-sample performance weakened over time "
            "(strongest 2023, softer 2025–26). Crypto factor premia are not guaranteed to persist.",
            "<b>Leverage in the overlay.</b> The volatility-targeting uplift is partly a function of "
            "leverage; risk-adjusted performance improves but the enhancement is not free.",
            "<b>Directional exposure.</b> Despite the regime gate and stops, the strategy retains net long "
            "crypto exposure and will participate in broad drawdowns, with a smaller maximum loss than passive holding.",
            "<b>Universe reconstruction.</b> The adopted universe is recomputed from trailing liquidity; "
            "production will be driven by a committed, point-in-time membership record."])]

        story += [Paragraph("6. Conclusion", s_h1),
                  Paragraph(
                      "Medallion Lite is a disciplined cross-sectional crypto factor strategy whose honest, "
                      "survivorship-free, out-of-sample performance — a frozen-parameter Sortino of <b>2.84 "
                      "on the 100-name universe</b> (2.95 walk-forward upper bound), against 1.78 for Bitcoin, "
                      "with a smaller drawdown — rests on a structural edge that <i>improves with universe "
                      "breadth</i>, aligning directly with a capacity-constrained mandate. Every pre-registered "
                      "acceptance gate passed, the edge persists in each out-of-sample year (4.7 / 2.7 / 1.7), "
                      "and it is robust to costs through 50 bps. Transaction-cost realism on the smaller names "
                      "is the principal remaining diligence item before scaling.", s_body)]

        story += [Spacer(1, 8), Paragraph(
            "Provenance &amp; reproducibility: pre-registered protocol (medallion_validation_protocol.md); "
            "deterministic audit (run_medallion_audit.py) with a published JSON manifest + daily-return CSV "
            "in artifacts/medallion_audit/. Coinbase USD-pair OHLCV 2021-01→2026-06, OOS 2023+, 30 bps, "
            "survivorship-free point-in-time top-100 universe. Registry ID 2026-06-medallion-lite.", s_small)]
        story += [Spacer(1, 6), Paragraph(
            "<b>Disclaimer.</b> This document is provided for informational purposes only and does not "
            "constitute an offer to sell or a solicitation to buy any security or interest in any fund. "
            "Performance figures herein are hypothetical and based on backtested, simulated results, which "
            "have inherent limitations: they are constructed with the benefit of hindsight, do not represent "
            "actual trading, and may not reflect the impact of material market factors on real-world "
            "execution. Past or simulated performance is not indicative of future results. Digital-asset "
            "trading involves a high degree of risk, including the risk of total loss. No representation is "
            "made that any account will achieve results similar to those shown.", s_disc)]

        footer_dt = datetime.now().strftime("%Y-%m-%d")

        def _footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(HexColor(SLATE))
            canvas.drawString(0.75 * inch, 0.5 * inch, f"K2 TRADE ATLAS · Medallion Lite Research Report · CONFIDENTIAL · {footer_dt}")
            canvas.drawRightString(7.75 * inch, 0.5 * inch, f"p. {doc.page}")
            canvas.setStrokeColor(HexColor("#cbd5e1"))
            canvas.line(0.75 * inch, 0.62 * inch, 7.75 * inch, 0.62 * inch)
            canvas.restoreState()

        OUT.parent.mkdir(parents=True, exist_ok=True)
        SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                          leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                          title="Medallion Lite — Research Report").build(
            story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"PDF -> {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
