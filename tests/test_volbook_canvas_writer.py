"""String-contract tests for the generated volbook canvas."""
from __future__ import annotations

from pathlib import Path

from volbook.bundle import Bar, OhlcvBundle, OhlcvSeries
from volbook.canvas_writer import render_canvas, write_canvas


def _toy_bundle() -> OhlcvBundle:
    bars = [
        Bar(t="2026-04-20T00:00:00+00:00", o=82.1, h=83.0, l=81.5, c=82.7, v=254010),
        Bar(t="2026-04-21T00:00:00+00:00", o=82.8, h=83.5, l=82.3, c=83.1, v=198442),
        Bar(t="2026-04-22T00:00:00+00:00", o=83.0, h=83.9, l=82.6, c=83.4, v=221100),
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
    )
    bundle = OhlcvBundle(generated_at="2026-04-24T00:00:00+00:00")
    bundle.upsert(series)
    return bundle


def test_render_canvas_includes_required_contract() -> None:
    tsx = render_canvas(_toy_bundle())

    # Required structural contract for a Cursor canvas.
    assert "export default function" in tsx
    assert 'from "cursor/canvas"' in tsx
    # Canvas can also pull hooks from "react" (allowed per sibling canvases).
    assert "fetch(" not in tsx, "canvas must not make network calls"

    # Bundle must be inlined as a const, not a dynamic import.
    assert "const BUNDLE" in tsx
    assert '"CL"' in tsx
    assert '"202606"' in tsx
    assert "83.9" in tsx  # a high from the toy bar
    assert "254010" in tsx  # a volume from the toy bar
    assert "82.1" in tsx  # an open price — candles need OHLC

    # Contract label (derived via FuturesSpec) should appear somewhere in
    # the inlined JSON so the dropdown can render it.
    assert "CL Jun'26" in tsx

    # Interactive controls + custom candle/volume renderers must be wired up.
    for symbol in (
        "Select",
        "Toggle",
        "Table",
        "useCanvasState",
        "useHostTheme",
        "Stat",
        "CandleChart",
        "VolumeChart",
        "onMouseDown",
        "wheel",
        # Technical Indicators section scaffolding.
        "IndicatorsSection",
        "Technical indicators",
        "Card",
        "CardHeader",
        "CardBody",
        "Pattern Recognition",
        # Best risk:reward setups section.
        "BestSetups",
        "Best technical risk:reward",
        "TradeSetup",
    ):
        assert symbol in tsx, f"canvas missing {symbol}"


def test_render_canvas_with_empty_bundle() -> None:
    # A no-series bundle should still render a valid file (the canvas
    # shows an instructional empty-state).
    tsx = render_canvas(OhlcvBundle())
    assert "export default function" in tsx
    assert "const BUNDLE: Series[] = []" in tsx


def test_write_canvas_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "volbook-futures-ohlcv.canvas.tsx"
    out = write_canvas(_toy_bundle(), target)
    assert out == target
    assert target.exists()
    assert target.read_text().startswith("import {")
