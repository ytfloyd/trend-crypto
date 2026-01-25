"""Tests for dynamic impact model."""
import pytest
pytest.importorskip("polars")

import polars as pl
from datetime import datetime, timedelta, timezone

from backtest.impact import compute_dynamic_slippage


def _bars(n: int = 50) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for i in range(n):
        ts = start + timedelta(hours=i)
        price *= 1.001
        rows.append(
            {
                "ts": ts,
                "symbol": "BTC-USD",
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price * 1.0005,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


def test_compute_dynamic_slippage_bounds():
    bars = _bars(30)
    trade_sizes = [10000.0] * bars.height
    impact = compute_dynamic_slippage(bars, trade_sizes, impact_coeff=0.1, vol_window=5)
    assert len(impact) == bars.height
    assert all(0.0 <= x <= 500.0 for x in impact)
    # trades should be clipped to at least 2 bps when vol > 0
    assert all(x == 0.0 or x >= 2.0 for x in impact)
