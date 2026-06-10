"""String-contract tests for the standalone volbook HTML dashboard."""
from __future__ import annotations

from pathlib import Path

from volbook.bundle import Bar, OhlcvBundle, OhlcvSeries
from volbook.html_writer import render_html, write_html


def _toy_bundle() -> OhlcvBundle:
    bars = [
        Bar(t="2026-04-20", o=82.1, h=83.0, l=81.5, c=82.7, v=254010),
        Bar(t="2026-04-21", o=82.8, h=83.5, l=82.3, c=83.1, v=198442),
        Bar(t="2026-04-22", o=83.0, h=83.9, l=82.6, c=83.4, v=221100),
    ]
    series = OhlcvSeries(
        symbol="CL",
        expiry="202606",
        exchange="NYMEX",
        bar_size="1 day",
        duration="1 Y",
        what_to_show="TRADES",
        bars=bars,
        fetched_at="2026-04-24T00:00:00+00:00",
        indicators={
            "Momentum Indicators": [
                {
                    "function": "RSI",
                    "display_name": "Relative Strength Index",
                    "params": {"timeperiod": 14},
                    "outputs": {"real": [None, 51.2, 54.3]},
                    "error": None,
                }
            ],
            "Pattern Recognition": [
                {
                    "function": "CDLDOJI",
                    "display_name": "Doji",
                    "params": {},
                    "outputs": {"integer": [0, 0, 100]},
                    "error": None,
                }
            ],
        },
        setups=[
            {
                "name": "Trend continuation",
                "direction": "long",
                "entry": 83.4,
                "stop": 81.4,
                "target": 87.4,
                "risk": 2.0,
                "reward": 4.0,
                "rr": 2.0,
                "confidence": 0.7,
                "score": 1.4,
                "rationale": ["Close above SMA20", "MACD histogram rising"],
                "tags": ["trend", "long"],
            }
        ],
    )
    bundle = OhlcvBundle(generated_at="2026-04-24T00:00:00+00:00")
    bundle.upsert(series)
    return bundle


def test_render_html_inlines_data_and_sections() -> None:
    html = render_html(_toy_bundle())

    assert "<!doctype html>" in html
    assert "const BUNDLE =" in html
    assert "Volatility Book" in html
    assert "Discretionary watchlist" in html
    assert "Selected contract setups" in html
    assert "Technical indicators" in html
    assert "Pattern Recognition" in html
    assert "Momentum Indicators" in html
    assert "Jump to indicators" in html
    assert "CL Jun'26" in html
    assert "RSI" in html
    assert "Trend continuation" in html
    assert "Filter-then-group watchlist" in html
    assert "function allOpportunities()" in html
    assert "function selectOpportunity(contractKey, barSize)" in html
    assert "oppMinRr" in html
    assert "oppTradableOnly" in html
    assert "Setup Grade" in html
    assert "Data Quality" in html
    assert "Avg Vol20" in html
    assert "Download Excel" in html
    assert "function downloadOpportunitiesExcel()" in html
    assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in html
    assert "volbook-opportunities-" in html
    assert ".xlsx" in html
    assert "function makeXlsx(headers, rows)" in html
    assert "[Content_Types].xml" in html
    assert "Rank</th><th>Contract</th><th class=\"num\">Setup Grade" in html
    assert "RR</th><th class=\"rationale\">Rationale</th>" in html
    assert "Strategy Class" in html
    assert "View Contracts" in html
    assert "Conflict" in html
    assert "hard liquidity floor" in html or "liquidity floor" in html
    assert "fetch(" not in html
    assert "https://" not in html


def test_write_html_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "volbook.html"
    out = write_html(_toy_bundle(), target)

    assert out == target
    assert target.exists()
    assert target.read_text().startswith("<!doctype html>")


def test_empty_bundle_has_empty_state() -> None:
    html = render_html(OhlcvBundle())

    assert "No series available" in html
    assert "const BUNDLE = []" in html


def test_render_html_contains_multiple_contracts_and_timeframes() -> None:
    bundle = _toy_bundle()
    other = OhlcvSeries(
        symbol="NQ",
        expiry="202606",
        exchange="CME",
        bar_size="1 hour",
        duration="30 D",
        what_to_show="TRADES",
        bars=[
            Bar(t="2026-04-24T15:00:00+00:00", o=20100, h=20150, l=20080, c=20120, v=1000),
            Bar(t="2026-04-24T16:00:00+00:00", o=20120, h=20190, l=20110, c=20180, v=1100),
        ],
        fetched_at="2026-04-24T16:00:00+00:00",
    )
    bundle.upsert(other)

    html = render_html(bundle)

    assert "CL Jun'26" in html
    assert "NQ Jun'26" in html
    assert '"bar_size": "1 day"' in html
    assert '"bar_size": "1 hour"' in html
    assert "function timeframes(contractKey)" in html


def test_render_html_contains_global_opportunity_ranking_hooks() -> None:
    bundle = _toy_bundle()
    other = OhlcvSeries(
        symbol="NQ",
        expiry="202606",
        exchange="CME",
        bar_size="1 hour",
        duration="30 D",
        what_to_show="TRADES",
        bars=[
            Bar(t="2026-04-24T15:00:00+00:00", o=20100, h=20150, l=20080, c=20120, v=1000),
            Bar(t="2026-04-24T16:00:00+00:00", o=20120, h=20190, l=20110, c=20180, v=1100),
        ],
        fetched_at="2026-04-24T16:00:00+00:00",
        setups=[
            {
                "name": "Range breakout",
                "direction": "short",
                "entry": 20180.0,
                "stop": 20220.0,
                "target": 20060.0,
                "risk": 40.0,
                "reward": 120.0,
                "rr": 3.0,
                "confidence": 0.8,
                "score": 2.4,
                "rationale": ["Close below prior range", "MACD below signal line"],
                "tags": ["breakout", "short"],
            }
        ],
    )
    bundle.upsert(other)

    html = render_html(bundle)

    assert "Range breakout" in html
    assert "NQ Jun'26" in html
    assert "collapseByUnderlyingView(rows)" in html
    assert "applyDirectionConflictPenalty" in html
    assert "function groupedWatchlist(opportunities)" in html
    assert "(b.setup_grade_score - a.setup_grade_score)" in html
    assert "(b.tradability.avgVol20 - a.tradability.avgVol20)" in html
    assert "data-contract-key" in html
    assert "Click a row to inspect that chart" in html
    assert "opportunityExportRows(opportunities)" in html


def test_render_html_penalizes_thin_far_curve_opportunities() -> None:
    liquid_bars = [
        Bar(t=f"2026-04-{i + 1:02d}", o=80 + i, h=81 + i, l=79 + i, c=80.5 + i, v=150 + i)
        for i in range(25)
    ]
    thin_bars = [
        Bar(t=f"2026-04-{i + 1:02d}", o=100 + i, h=100 + i, l=100 + i, c=100 + i, v=0 if i < 23 else 1)
        for i in range(25)
    ]
    bundle = OhlcvBundle(generated_at="2026-04-24T00:00:00+00:00")
    bundle.upsert(
        OhlcvSeries(
            symbol="ZS",
            expiry="202605",
            exchange="CBOT",
            bar_size="1 day",
            duration="1 Y",
            what_to_show="TRADES",
            bars=liquid_bars,
            setups=[
                {
                    "name": "Trend continuation",
                    "direction": "long",
                    "entry": 104.5,
                    "stop": 100.0,
                    "target": 113.5,
                    "risk": 4.5,
                    "reward": 9.0,
                    "rr": 2.0,
                    "confidence": 0.6,
                    "score": 1.2,
                    "rationale": ["Liquid front contract trend"],
                    "tags": ["trend", "long"],
                }
            ],
        )
    )
    bundle.upsert(
        OhlcvSeries(
            symbol="ZS",
            expiry="202801",
            exchange="CBOT",
            bar_size="1 day",
            duration="1 Y",
            what_to_show="TRADES",
            bars=thin_bars,
            setups=[
                {
                    "name": "Trend continuation",
                    "direction": "long",
                    "entry": 124.0,
                    "stop": 120.0,
                    "target": 140.0,
                    "risk": 4.0,
                    "reward": 16.0,
                    "rr": 4.0,
                    "confidence": 0.9,
                    "score": 3.6,
                    "rationale": ["Thin far-curve price trend"],
                    "tags": ["trend", "long"],
                }
            ],
        )
    )

    html = render_html(bundle)

    assert "function seriesTradability(series, rankMap)" in html
    assert "adjustedScore = score * tradability.liquidityScore * tradability.dataQualityScore * tradability.curveScore" in html
    assert "far curve #" in html
    assert "thin/stale" in html
    assert "ok for 1-lot screen" in html
