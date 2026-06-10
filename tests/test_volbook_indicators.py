"""Tests for the TA-Lib indicator computation helper."""
from __future__ import annotations

import math

import pytest

from volbook.bundle import Bar
from volbook.indicators import TALIB_AVAILABLE, compute_all_indicators


pytestmark = pytest.mark.skipif(
    not TALIB_AVAILABLE, reason="TA-Lib not installed (install with `pip install -e .[ta]`)"
)


def _synthetic_bars(n: int = 120) -> list[Bar]:
    """Slowly trending close with realistic OHLC spread."""
    bars: list[Bar] = []
    px = 100.0
    for i in range(n):
        px += math.sin(i / 8.0) + 0.05
        o = px
        c = px + math.cos(i / 5.0) * 0.4
        h = max(o, c) + 0.3
        l = min(o, c) - 0.3
        bars.append(
            Bar(
                t=f"2026-01-{(i % 28) + 1:02d}T00:00:00+00:00",
                o=o,
                h=h,
                l=l,
                c=c,
                v=1000 + (i * 37) % 500,
            )
        )
    return bars


def test_compute_all_indicators_returns_all_core_groups() -> None:
    bars = _synthetic_bars()
    by_cat = compute_all_indicators(bars, tail=10)

    # Ten well-known TA-Lib groups that should all have at least one row.
    expected_groups = {
        "Momentum Indicators",
        "Overlap Studies",
        "Volatility Indicators",
        "Volume Indicators",
        "Cycle Indicators",
        "Pattern Recognition",
        "Price Transform",
        "Statistic Functions",
        "Math Transform",
        "Math Operators",
    }
    missing = expected_groups - set(by_cat.keys())
    assert not missing, f"missing groups: {missing}"

    # Plenty of indicators overall.
    total = sum(len(v) for v in by_cat.values())
    assert total >= 100, f"expected 100+ indicators, got {total}"


def test_indicator_values_are_finite_or_none() -> None:
    bars = _synthetic_bars()
    by_cat = compute_all_indicators(bars, tail=5)
    for rows in by_cat.values():
        for rec in rows:
            for out_values in rec["outputs"].values():
                for v in out_values:
                    # Either a finite float or None (for leading-lookback NaNs).
                    assert v is None or (
                        isinstance(v, (int, float)) and math.isfinite(v)
                    ), f"{rec['function']} has non-finite value"


def test_tail_truncation_applied() -> None:
    bars = _synthetic_bars(200)
    by_cat = compute_all_indicators(bars, tail=7)
    lengths = {
        len(vals)
        for rows in by_cat.values()
        for rec in rows
        for vals in rec["outputs"].values()
    }
    assert lengths == {7}, f"expected all tails to be 7, got {lengths}"


def test_known_indicator_has_expected_outputs() -> None:
    bars = _synthetic_bars()
    by_cat = compute_all_indicators(bars, tail=10)
    momentum = {r["function"]: r for r in by_cat["Momentum Indicators"]}

    # MACD is canonical and stable across TA-Lib versions.
    assert "MACD" in momentum
    macd = momentum["MACD"]
    assert set(macd["outputs"].keys()) == {"macd", "macdsignal", "macdhist"}
    assert macd["params"]["fastperiod"] == 12
    assert macd["params"]["slowperiod"] == 26
    assert macd["error"] is None


def test_empty_bars_returns_empty_dict() -> None:
    assert compute_all_indicators([], tail=5) == {}
