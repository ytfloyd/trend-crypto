"""Generate a professionally formatted PDF of the project status breakdown."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

MARGIN = 0.75 * inch
PAGE_W, PAGE_H = letter
CONTENT_W = PAGE_W - 2 * MARGIN

NAVY = colors.HexColor("#1B2A4A")
ACCENT = colors.HexColor("#2E86AB")
LIGHT_BG = colors.HexColor("#F4F6F9")
MID_GRAY = colors.HexColor("#6B7280")
DARK_TEXT = colors.HexColor("#1F2937")
WHITE = colors.white
GREEN = colors.HexColor("#059669")
AMBER = colors.HexColor("#D97706")
RED = colors.HexColor("#DC2626")
ROW_ALT = colors.HexColor("#F9FAFB")
BORDER = colors.HexColor("#D1D5DB")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=34,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#CBD5E1"),
            alignment=TA_LEFT,
        ),
        "cover_date": ParagraphStyle(
            "cover_date",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            textColor=colors.HexColor("#94A3B8"),
            alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=24,
            textColor=NAVY,
            spaceBefore=20,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=18,
            textColor=ACCENT,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=DARK_TEXT,
            spaceAfter=6,
        ),
        "body_bold": ParagraphStyle(
            "body_bold",
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=14,
            textColor=DARK_TEXT,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=DARK_TEXT,
            leftIndent=16,
            bulletIndent=4,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=DARK_TEXT,
            backColor=LIGHT_BG,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=MID_GRAY,
            spaceAfter=8,
        ),
        "toc_item": ParagraphStyle(
            "toc_item",
            fontName="Helvetica",
            fontSize=10,
            leading=18,
            textColor=DARK_TEXT,
            leftIndent=12,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=MID_GRAY,
            alignment=TA_CENTER,
        ),
        "header_right": ParagraphStyle(
            "header_right",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=MID_GRAY,
            alignment=TA_RIGHT,
        ),
        "th": ParagraphStyle(
            "th",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=WHITE,
        ),
        "td": ParagraphStyle(
            "td",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=DARK_TEXT,
        ),
        "td_code": ParagraphStyle(
            "td_code",
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=DARK_TEXT,
        ),
        "td_bold": ParagraphStyle(
            "td_bold",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=DARK_TEXT,
        ),
    }


def _table(headers: list[str], rows: list[list], col_widths: list[float], s: dict) -> Table:
    """Build a styled table with alternating row shading."""
    header_cells = [Paragraph(h, s["th"]) for h in headers]
    body_cells = []
    for row in rows:
        body_cells.append([Paragraph(str(c), s["td"]) if not isinstance(c, Paragraph) else c for c in row])

    t = Table([header_cells] + body_cells, colWidths=col_widths, repeatRows=1)
    style_cmds: list = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ]
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    t.setStyle(TableStyle(style_cmds))
    return t


def _status_badge(text: str, s: dict) -> Paragraph:
    color_map = {"STRONG": "#059669", "SOLID": "#2E86AB", "GAP": "#DC2626"}
    c = color_map.get(text, "#6B7280")
    return Paragraph(f'<font color="{c}"><b>{text}</b></font>', s["td"])


def _divider() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6, spaceBefore=6)


def _cover_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H * 0.38, PAGE_W, 3, fill=1, stroke=0)
    canvas.restoreState()


def _body_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - 0.5 * inch, PAGE_W - MARGIN, PAGE_H - 0.5 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(MARGIN, PAGE_H - 0.4 * inch, "Trend Crypto — Project Status Breakdown")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.4 * inch, f"March 2026  |  Confidential")
    canvas.line(MARGIN, 0.55 * inch, PAGE_W - MARGIN, 0.55 * inch)
    canvas.drawCentredString(PAGE_W / 2, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(output_path: Path) -> None:
    s = _styles()

    cover_frame = Frame(
        MARGIN + 0.25 * inch,
        PAGE_H * 0.42,
        CONTENT_W - 0.5 * inch,
        PAGE_H * 0.45,
        id="cover",
    )
    body_frame = Frame(
        MARGIN, MARGIN + 0.3 * inch, CONTENT_W, PAGE_H - 2 * MARGIN - 0.6 * inch, id="body"
    )

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[cover_frame], onPage=_cover_page),
            PageTemplate(id="body", frames=[body_frame], onPage=_body_page),
        ]
    )

    story: list = []

    # ── Cover page ──
    story.append(Paragraph("Trend Crypto", s["cover_title"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Project Status Breakdown", s["cover_title"]))
    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "Comprehensive audit of the platform: what is working, "
            "what needs attention, highest-leverage next steps, and candidates for removal.",
            s["cover_subtitle"],
        )
    )
    story.append(Spacer(1, 24))
    story.append(Paragraph(f"March 2026  ·  Internal Distribution", s["cover_date"]))

    story.append(NextPageTemplate("body"))
    story.append(PageBreak())

    # ── Table of Contents ──
    story.append(Paragraph("Contents", s["h1"]))
    story.append(_divider())
    toc_items = [
        "1. What Is Working Well",
        "2. What Is Not Working or Problematic",
        "3. Highest-Leverage Ideas for Pushing Forward",
        "4. Candidates for Removal",
        "5. Module Health Summary",
        "6. Research Module Inventory",
        "7. Notebook Inventory",
        "8. External Dependencies",
    ]
    for item in toc_items:
        story.append(Paragraph(item, s["toc_item"]))
    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "<b>Project Identity.</b>  <b>trend-crypto-backtest</b> is a crypto trend-following "
            "research and backtesting platform. It supports single-asset and multi-asset "
            "backtesting (BTC, ETH, mid-caps), systematic alpha research, and has a live/paper "
            "trading path. The codebase is Python 3.12, uses Polars + DuckDB for data, Pydantic "
            "for config, and runs CI via GitHub Actions (mypy, ruff, pytest).",
            s["body"],
        )
    )
    story.append(Spacer(1, 8))
    story.append(Paragraph("Repository layout", s["h2"]))
    layout_lines = [
        ("src/", "Core library — engine, strategies, risk, data, execution, alphas, pricing, vol"),
        ("scripts/", "Entry points — run_backtest, diagnostics, sweeps"),
        ("scripts/research/", "Research modules — JPM momentum, TSMOM, logreg, talib, alpha lab"),
        ("configs/", "YAML configs — backtest runs and research experiments"),
        ("notebooks/alpha/", "Jupyter notebooks (00–06) for interactive research"),
        ("tests/", "67 pytest files covering the core path"),
        ("docs/", "Strategy memos, alpha framework docs, validation guides"),
        ("deployments/", "Deployment specs (v2.5 incubation)"),
    ]
    story.append(
        _table(
            ["Directory", "Contents"],
            [[Paragraph(f"<font face='Courier'>{d}</font>", s["td"]), c] for d, c in layout_lines],
            [1.6 * inch, CONTENT_W - 1.6 * inch],
            s,
        )
    )

    story.append(PageBreak())

    # ── Section 1: What Is Working Well ──
    story.append(Paragraph("1. What Is Working Well", s["h1"]))
    story.append(_divider())

    story.append(Paragraph("Core Backtest Engine — Production-grade, heavily tested", s["h2"]))
    story.append(
        Paragraph(
            "The backtest engine (<font face='Courier'>src/backtest/engine.py</font>) implements "
            "strict <b>Model B</b> timing (signal at close, fill at open+1) with realistic cost "
            "modeling (fees, slippage, funding). It has <b>67 tests</b> covering timing, PnL, costs, "
            "rebalancing, impact, sell constraints, cash buffers, and deadbands. This is the "
            "strongest part of the codebase.",
            s["body"],
        )
    )
    engine_files = [
        ("src/backtest/engine.py", "Single-asset BacktestEngine"),
        ("src/backtest/portfolio_engine.py", "Multi-asset PortfolioEngine"),
        ("src/backtest/portfolio.py", "Cash/position tracking"),
        ("src/backtest/rebalance.py", "Target-weight rebalancing"),
        ("src/backtest/impact.py", "Dynamic slippage model"),
        ("src/backtest/metrics.py", "Performance statistics"),
    ]
    story.append(
        _table(
            ["File", "Role"],
            [[Paragraph(f"<font face='Courier'>{f}</font>", s["td"]), r] for f, r in engine_files],
            [2.8 * inch, CONTENT_W - 2.8 * inch],
            s,
        )
    )

    story.append(Paragraph("Data Pipeline — Reliable", s["h2"]))
    story.append(
        Paragraph(
            "<b>Coinbase collector</b> (<font face='Courier'>src/data/collector.py</font>) ingests "
            "1m OHLCV into DuckDB. <b>DataPortal</b> (<font face='Courier'>src/data/portal.py"
            "</font>) loads and resamples bars (1m → 1h, 4h, 1d). Both are tested. Schema is "
            "consistent across crypto and ETF databases.",
            s["body"],
        )
    )

    story.append(Paragraph("Risk Framework — Comprehensive", s["h2"]))
    risk_items = [
        "Vol targeting (vol_target.py, risk_manager.py)",
        "Kelly criterion (kelly.py)",
        "VaR — historical + parametric (var.py)",
        "Stress testing (stress.py)",
        "Factor attribution (attribution.py)",
        "Correlation regime detection (correlation.py, regime.py)",
        "Carver-style position sizing (position_sizing.py, handcraft.py, diversification.py)",
    ]
    for item in risk_items:
        story.append(Paragraph(f"•  {item}", s["bullet"]))
    story.append(
        Paragraph(
            "Tested via test_risk_framework.py, test_kelly.py, test_vol_target_scales_down.py, "
            "test_portfolio_engine.py.",
            s["caption"],
        )
    )

    story.append(Paragraph("Strategy Layer — Solid", s["h2"]))
    strat_rows = [
        ("MA crossover + vol hysteresis", "src/strategy/ma_cross_vol_hysteresis.py"),
        ("MA crossover long-only", "src/strategy/ma_crossover_long_only.py"),
        ("Buy-and-hold", "src/strategy/buy_and_hold.py"),
        ("Carver-style forecasts (EWMAC, breakout, carry)", "src/strategy/forecast.py"),
        ("Forecast combiner with FDM", "src/strategy/forecast_combiner.py"),
    ]
    story.append(
        _table(
            ["Strategy", "File"],
            [[n, Paragraph(f"<font face='Courier'>{f}</font>", s["td"])] for n, f in strat_rows],
            [3.0 * inch, CONTENT_W - 3.0 * inch],
            s,
        )
    )
    story.append(Paragraph("Tested and cataloged via a strategy registry.", s["caption"]))

    story.append(Paragraph("Alpha Factory — Well-architected", s["h2"]))
    story.append(
        Paragraph(
            "<font face='Courier'>src/alphas/</font> implements a formulaic alpha DSL: "
            "<b>Parser</b> (AST from string expressions), <b>Compiler</b> (Polars execution plans), "
            "<b>Primitives</b> (time-series and cross-sectional operators), <b>Signal processor</b> "
            "(z-score normalization, EMA smoothing, winsorization). Six test files cover parser, "
            "two-stage pipeline, warmup, signal processing, and table resolution.",
            s["body"],
        )
    )

    story.append(Paragraph("Live/Paper Trading Path — Tested", s["h2"]))
    story.append(
        Paragraph(
            "LiveRunner (<font face='Courier'>src/live/runner.py</font>), PaperBroker, OMS "
            "(<font face='Courier'>src/execution/</font>), and monitoring/alerts/reconciliation "
            "(<font face='Courier'>src/monitoring/</font>) form a complete paper-trading loop. "
            "Tested in test_live_trading.py.",
            s["body"],
        )
    )

    story.append(Paragraph("Research Infrastructure — Productive", s["h2"]))
    research_items = [
        "Common utilities (scripts/research/common/) — shared data loading, simple backtest, "
        "metrics, Bayesian evaluation, risk overlays, cost analysis.",
        "7 research notebooks (notebooks/alpha/00 through 06) — data explorer, signal sandbox, "
        "turtle trader, AVAX deep-dive, logreg filter, Bayesian evaluation, vol estimators.",
        "ExperimentTracker and ParameterOptimizer (src/research/).",
    ]
    for item in research_items:
        story.append(Paragraph(f"•  {item}", s["bullet"]))

    story.append(Paragraph("Documentation &amp; CI/CD", s["h2"]))
    story.append(
        Paragraph(
            "35 markdown files in <font face='Courier'>docs/research/</font> covering Transtrend "
            "(v0/v1/v2), JPM momentum (chapters 1–8), 101 alphas, Kuma trend, alpha framework, "
            "purged CV, and the options/vol program setup guide. GitHub Actions runs mypy (strict), "
            "ruff lint, strategy registry validation, and pytest with coverage on every push/PR "
            "to main.",
            s["body"],
        )
    )

    story.append(PageBreak())

    # ── Section 2: Not Working ──
    story.append(Paragraph("2. What Is Not Working or Problematic", s["h1"]))
    story.append(_divider())

    story.append(Paragraph("Options/Volatility Modules — Zero test coverage", s["h2"]))
    issue_rows = [
        ("src/pricing/", "Black-Scholes, Black-76, Bachelier", "None"),
        ("src/volatility/", "5 realized vol estimators, VolSurface", "None"),
        ("src/data/options/", "IB chain fetcher, vol surface collector", "None"),
    ]
    story.append(
        _table(
            ["Module", "Contents", "Tests"],
            [
                [
                    Paragraph(f"<font face='Courier'>{m}</font>", s["td"]),
                    c,
                    Paragraph(f'<font color="#DC2626"><b>{t}</b></font>', s["td"]),
                ]
                for m, c, t in issue_rows
            ],
            [1.5 * inch, 3.2 * inch, CONTENT_W - 4.7 * inch],
            s,
        )
    )
    story.append(
        Paragraph(
            "The IB integration requires the ib_insync optional dependency and has never been "
            "validated against a live or paper TWS connection.",
            s["body"],
        )
    )

    story.append(Paragraph("Notebook 04 — Broken imports", s["h2"]))
    story.append(
        Paragraph(
            "<font face='Courier'>04_logreg_probability_filter.ipynb</font> fails with "
            "ModuleNotFoundError because jpm_bigdata_ai/helpers.py uses absolute imports from "
            "scripts.research.common.data that don't resolve via the notebook's _setup.py "
            "path setup.",
            s["body"],
        )
    )

    story.append(Paragraph("Mypy Overrides — Suppressing rather than fixing", s["h2"]))
    story.append(
        Paragraph(
            "pyproject.toml has <font face='Courier'>ignore_errors = true</font> for: "
            "data.*, alphas.*, analysis.*, portfolio.*, volatility.*, pricing.*, strategy.*, "
            "risk.*. This masks real type errors. The strict mypy config loses most of its "
            "value when the majority of modules opt out.",
            s["body"],
        )
    )

    story.append(Paragraph("Stale GitHub Actions Run", s["h2"]))
    story.append(
        Paragraph(
            "A zombie queued run (ID 22727555100) is permanently stuck and cannot be cancelled "
            "or deleted (GitHub returns HTTP 500). Monitor and contact GitHub support if it "
            "persists.",
            s["body"],
        )
    )

    story.append(Paragraph("Inconsistent Import Conventions", s["h2"]))
    import_issues = [
        "test_ma_crossover_adx_default.py uses src.common.config while all other tests use "
        "common.config.",
        "src/data/options/snapshot.py and src/volatility/surface.py use absolute imports "
        "(from volatility.surface) after being patched from broken relative imports — correct "
        "given the sys.path setup, but a fragility point.",
    ]
    for item in import_issues:
        story.append(Paragraph(f"•  {item}", s["bullet"]))

    story.append(Paragraph("Missing Developer Documentation", s["h2"]))
    story.append(
        Paragraph(
            "No architecture overview, no module dependency diagram, no developer setup guide, "
            "no contribution guide. The README covers usage but not internals.",
            s["body"],
        )
    )

    story.append(Paragraph("Stray Files", s["h2"]))
    stray_items = [
        "snx_usd_timeseries.png — untracked PNG in repo root, referenced nowhere.",
        "scripts/research/common/_cache/*.parquet — cache files that should never be committed.",
    ]
    for item in stray_items:
        story.append(Paragraph(f"•  {item}", s["bullet"]))

    story.append(PageBreak())

    # ── Section 3: Highest-Leverage Ideas ──
    story.append(Paragraph("3. Highest-Leverage Ideas for Pushing Forward", s["h1"]))
    story.append(_divider())

    ideas = [
        (
            "A. Test the options/vol stack",
            "The pricing and volatility modules are mathematically dense and critical if the "
            "options program is to be trusted. Unit tests for Black-Scholes put/call parity, "
            "Greeks symmetries, IV round-tripping, and vol estimator sanity checks would take "
            "roughly a day and dramatically increase confidence.",
        ),
        (
            "B. Integrate Carver forecasts into the main backtest path",
            "forecast.py and forecast_combiner.py already implement EWMAC, breakout, and carry "
            "forecasts with proper scaling and combination. These are currently only used by the "
            "Carver position-sizing research path — wiring them into BacktestEngine as a "
            "first-class strategy type would unify the single-signal and multi-signal paths.",
        ),
        (
            "C. Multi-asset portfolio backtesting as the default",
            "PortfolioEngine exists and is tested, but most configs and workflows still target "
            "single-asset runs. Shifting the default workflow to multi-asset (with HRP or "
            "handcrafted weights) would better represent the actual portfolio being traded.",
        ),
        (
            "D. Standardize Bayesian evaluation",
            "The Bayesian toolkit in scripts/research/common/bayesian.py (posterior Sharpe, "
            "credible intervals, P(A beats B), Bayes factors) is powerful. Making it a mandatory "
            "step in every strategy evaluation — not just notebook 05 — would raise the research "
            "bar across the board.",
        ),
        (
            "E. Options data pipeline — validate with a real IB connection",
            "The IB integration code exists but has never been tested against a live or paper TWS "
            "connection. A single end-to-end test (connect, fetch chain, snapshot surface, store "
            "in DuckDB, reconstruct VolSurface) would validate the entire path.",
        ),
        (
            "F. Clean up mypy overrides",
            "Replace blanket ignore_errors = true overrides with targeted type: ignore comments "
            "or proper type annotations. This would catch real bugs that the strict mypy config "
            "was designed to surface.",
        ),
        (
            "G. Architecture documentation",
            "A one-page diagram showing the data flow and module dependency map would help "
            "onboarding and team communication.",
        ),
    ]
    for title, desc in ideas:
        story.append(Paragraph(title, s["h2"]))
        story.append(Paragraph(desc, s["body"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Data flow overview", s["body_bold"]))
    flow_lines = [
        "Coinbase API  →  Collector  →  DuckDB",
        "                                   ↓",
        "                              DataPortal  →  bars (1m / 1h / 4h / 1d)",
        "                                   ↓",
        "                              Strategy  →  target weights / forecasts",
        "                                   ↓",
        "                              RiskManager  →  vol-targeted weights",
        "                                   ↓",
        "                              BacktestEngine  →  fills, PnL, equity curve",
        "                                   ↓",
        "                              Artifacts  →  parquet, JSON, tearsheets",
    ]
    for line in flow_lines:
        story.append(Paragraph(line, s["code"]))

    story.append(PageBreak())

    # ── Section 4: Candidates for Removal ──
    story.append(Paragraph("4. Candidates for Removal", s["h1"]))
    story.append(_divider())

    story.append(Paragraph("Likely removable", s["h2"]))
    removal_rows = [
        ("snx_usd_timeseries.png", "Stray chart in repo root, referenced nowhere. Delete and add *.png to .gitignore."),
        ("scripts/research/ml4t_autoencoder/", "ML4T textbook exercise. Adds maintenance weight if not feeding production."),
        ("scripts/research/ml4t_garch/", "Pedagogical GARCH experiment from ML4T."),
        ("scripts/research/ml4t_pairs/", "Pairs trading experiment from ML4T."),
        ("scripts/research/sornette_lppl/", "LPPL bubble detection. Niche; if not informing risk overlays, it is dead weight."),
        ("scripts/research/crowding/", "Crowding overlay. Check if used downstream; if not, archive."),
    ]
    story.append(
        _table(
            ["Item", "Reason"],
            [
                [Paragraph(f"<font face='Courier'>{i}</font>", s["td"]), r]
                for i, r in removal_rows
            ],
            [2.4 * inch, CONTENT_W - 2.4 * inch],
            s,
        )
    )

    story.append(Paragraph("Review for consolidation", s["h2"]))
    consol_rows = [
        ("scripts/research/paper_strategies/", "If superseded by jpm_momentum, alpha_lab, or tsmom, archive."),
        ("scripts/research/paper_pipeline/", "Check for overlap with active research modules."),
        ("scripts/research/multifreq/", "If findings incorporated into the main strategy, archive."),
        ("configs/runs/ (24 YAML files)", "Many near-duplicates (v2 through v25). Consider parameterizing."),
    ]
    story.append(
        _table(
            ["Item", "Action"],
            [
                [Paragraph(f"<font face='Courier'>{i}</font>", s["td"]), a]
                for i, a in consol_rows
            ],
            [2.6 * inch, CONTENT_W - 2.6 * inch],
            s,
        )
    )

    story.append(Paragraph("Do not remove", s["h2"]))
    keep_rows = [
        ("All core src/ modules", "Production or actively used"),
        ("scripts/research/jpm_momentum/", "Active research, complete with runners"),
        ("scripts/research/logreg_filter/", "Active research, full pipeline"),
        ("scripts/research/talib_scanner/", "Active research, IC scanning"),
        ("scripts/research/tsmom/", "Active research, time-series momentum"),
        ("scripts/research/alpha_lab/", "Active research, forward simulation"),
        ("scripts/research/common/", "Shared utilities — everything depends on it"),
        ("scripts/research/etf_data/", "Needed for cross-asset research"),
    ]
    story.append(
        _table(
            ["Item", "Reason"],
            [[Paragraph(f"<font face='Courier'>{i}</font>", s["td"]), r] for i, r in keep_rows],
            [2.6 * inch, CONTENT_W - 2.6 * inch],
            s,
        )
    )

    story.append(PageBreak())

    # ── Section 5: Module Health Summary ──
    story.append(Paragraph("5. Module Health Summary", s["h1"]))
    story.append(_divider())

    health_rows = [
        ("src/backtest/", "STRONG", "15+ tests, core path, production-ready"),
        ("src/data/", "STRONG", "Collector + portal tested, clean schema"),
        ("src/risk/", "STRONG", "Comprehensive, tested"),
        ("src/strategy/", "STRONG", "Tested, forecast framework ready"),
        ("src/alphas/", "STRONG", "6 test files, DSL well-designed"),
        ("src/execution/", "SOLID", "Tested via live trading tests"),
        ("src/live/", "SOLID", "Tested"),
        ("src/monitoring/", "SOLID", "Tested"),
        ("src/portfolio/", "SOLID", "HRP tested"),
        ("src/research/", "SOLID", "ExperimentTracker, optimizer tested"),
        ("src/validation/", "SOLID", "Purged CV tested"),
        ("src/common/", "SOLID", "Shared config/utils, tested"),
        ("src/utils/", "SOLID", "DuckDB inspect, tested"),
        ("src/pricing/", "GAP", "No tests, used by options research only"),
        ("src/volatility/", "GAP", "No tests, used by options research only"),
        ("src/data/options/", "GAP", "No tests, requires ib_insync, experimental"),
    ]
    story.append(
        _table(
            ["Module", "Status", "Notes"],
            [
                [
                    Paragraph(f"<font face='Courier'>{m}</font>", s["td"]),
                    _status_badge(st, s),
                    n,
                ]
                for m, st, n in health_rows
            ],
            [1.7 * inch, 0.8 * inch, CONTENT_W - 2.5 * inch],
            s,
        )
    )

    story.append(PageBreak())

    # ── Section 6: Research Module Inventory ──
    story.append(Paragraph("6. Research Module Inventory", s["h1"]))
    story.append(_divider())

    research_rows = [
        ("jpm_momentum/", "Complete", "run_ch2_*.py, run_ch3_*.py", "Crypto + ETF, 6 signal types"),
        ("tsmom/", "Complete", "Multiple runners", "Time-series momentum, cross-asset"),
        ("logreg_filter/", "Complete", "python -m logreg_filter", "Walk-forward logistic regression"),
        ("talib_scanner/", "Complete", "python -m talib_scanner", "IC scan of ~95 TA-Lib features"),
        ("alpha_lab/", "Complete", "Multiple runners", "Turtle portfolio, forward sim"),
        ("etf_data/", "Complete", "python -m etf_data.ingest", "Tiingo API → DuckDB, ~64 ETFs"),
        ("common/", "Complete", "Imported by all", "Data, backtest, metrics, Bayesian"),
        ("sornette_lppl/", "Complete", "Standalone", "Review for removal"),
        ("ml4t_autoencoder/", "Experiment", "Standalone", "Review for removal"),
        ("ml4t_garch/", "Experiment", "Standalone", "Review for removal"),
        ("ml4t_pairs/", "Experiment", "Standalone", "Review for removal"),
        ("crowding/", "Experiment", "Standalone", "Review for removal"),
        ("paper_strategies/", "Unclear", "Multiple runners", "May overlap with jpm/tsmom"),
        ("paper_pipeline/", "Unclear", "Multiple runners", "May overlap with alpha_lab"),
        ("multifreq/", "Unclear", "Standalone", "Check if incorporated elsewhere"),
    ]
    story.append(
        _table(
            ["Module", "Status", "Entry Point", "Notes"],
            [
                [
                    Paragraph(f"<font face='Courier'>{m}</font>", s["td"]),
                    st,
                    Paragraph(f"<font face='Courier' size='7.5'>{ep}</font>", s["td"]),
                    n,
                ]
                for m, st, ep, n in research_rows
            ],
            [1.4 * inch, 0.85 * inch, 1.7 * inch, CONTENT_W - 3.95 * inch],
            s,
        )
    )

    story.append(Spacer(1, 16))

    # ── Section 7: Notebook Inventory ──
    story.append(Paragraph("7. Notebook Inventory", s["h1"]))
    story.append(_divider())

    nb_rows = [
        ("00_data_explorer", "OHLCV coverage, universe, ADV", "Working"),
        ("01_signal_sandbox", "Prototype signals, quick backtest", "Working"),
        ("02_turtle_trader", "Turtle rules on crypto (20/10, 55/20)", "Working"),
        ("03_avax_turtle_deep_dive", "AVAX-USD 8h vs 1d mechanics", "Working"),
        ("04_logreg_probability_filter", "LogReg probability engine", "Broken"),
        ("05_bayesian_strategy_evaluation", "Bayesian credible intervals, Bayes factors", "Working"),
        ("06_vol_estimators", "Realized vol estimators, vol cone, pricing", "Working"),
    ]

    def _nb_status(text: str) -> Paragraph:
        c = "#059669" if text == "Working" else "#DC2626"
        return Paragraph(f'<font color="{c}"><b>{text}</b></font>', s["td"])

    story.append(
        _table(
            ["Notebook", "Topic", "Status"],
            [
                [Paragraph(f"<font face='Courier'>{nb}</font>", s["td"]), topic, _nb_status(st)]
                for nb, topic, st in nb_rows
            ],
            [2.4 * inch, 3.0 * inch, CONTENT_W - 5.4 * inch],
            s,
        )
    )

    story.append(Spacer(1, 16))

    # ── Section 8: External Dependencies ──
    story.append(Paragraph("8. External Dependencies", s["h1"]))
    story.append(_divider())

    dep_rows = [
        ("Coinbase Advanced Trade API", "External API", "src/data/collector.py"),
        ("Tiingo API", "External API", "scripts/research/etf_data/ (requires TIINGO_API_KEY)"),
        ("Interactive Brokers TWS", "External API", "src/data/options/ (requires ib_insync)"),
        ("DuckDB (market.duckdb)", "Local database", "Core data path, all research"),
        ("DuckDB (etf_market.duckdb)", "Local database", "ETF research (JPM momentum, cross-asset)"),
        ("TA-Lib", "C library", "talib_scanner, logreg_filter, jpm_bigdata_ai"),
        ("scikit-learn", "Library", "logreg_filter, jpm_bigdata_ai, HRP covariance"),
        ("scipy", "Library", "HRP, Kelly, vol estimators, pricing"),
    ]
    story.append(
        _table(
            ["Dependency", "Type", "Used By"],
            dep_rows,
            [2.2 * inch, 1.1 * inch, CONTENT_W - 3.3 * inch],
            s,
        )
    )

    # ── Build ──
    doc.build(story)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "docs" / "project_status_breakdown.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(out)
    print(f"PDF written to {out}")
