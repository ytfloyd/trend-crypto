"""Tests for the technical risk:reward setup builder."""
from __future__ import annotations

import math

import pytest

from volbook.bundle import Bar
from volbook.signals import TALIB_AVAILABLE, build_setups


pytestmark = pytest.mark.skipif(
    not TALIB_AVAILABLE, reason="TA-Lib not installed"
)


def _strong_downtrend(n: int = 80) -> list[Bar]:
    """A clean, persistent downtrend — should fire trend-short setup."""
    bars: list[Bar] = []
    px = 120.0
    for i in range(n):
        px -= 0.35  # steady drift down
        noise = math.sin(i / 3.0) * 0.2
        o = px + noise
        c = px - 0.25 + noise
        h = max(o, c) + 0.15
        l = min(o, c) - 0.15
        bars.append(
            Bar(
                t=f"2026-01-{(i % 28) + 1:02d}T00:00:00+00:00",
                o=o,
                h=h,
                l=l,
                c=c,
                v=1000 + (i * 13) % 200,
            )
        )
    return bars


def _strong_uptrend(n: int = 80) -> list[Bar]:
    bars: list[Bar] = []
    px = 50.0
    for i in range(n):
        px += 0.35
        noise = math.sin(i / 3.0) * 0.2
        o = px + noise
        c = px + 0.25 + noise
        h = max(o, c) + 0.15
        l = min(o, c) - 0.15
        bars.append(
            Bar(
                t=f"2026-01-{(i % 28) + 1:02d}T00:00:00+00:00",
                o=o,
                h=h,
                l=l,
                c=c,
                v=1000 + (i * 13) % 200,
            )
        )
    return bars


def test_downtrend_produces_short_setup() -> None:
    setups = build_setups(_strong_downtrend())
    assert setups, "expected at least one setup in a clean downtrend"
    top = setups[0]
    assert top["direction"] == "short"
    assert top["stop"] > top["entry"] > top["target"]
    assert top["rr"] >= 1.0
    assert 0.0 <= top["confidence"] <= 1.0
    assert top["rationale"], "setup must carry rationale bullets"


def test_uptrend_produces_long_setup() -> None:
    setups = build_setups(_strong_uptrend())
    assert setups
    top = setups[0]
    assert top["direction"] == "long"
    assert top["stop"] < top["entry"] < top["target"]
    assert top["rr"] >= 1.0


def test_setups_ranked_by_score() -> None:
    setups = build_setups(_strong_downtrend())
    scores = [s["score"] for s in setups]
    assert scores == sorted(scores, reverse=True)


def test_insufficient_bars_returns_empty() -> None:
    # Fewer than 60 bars skips setup generation entirely.
    assert build_setups(_strong_downtrend(n=40)) == []
