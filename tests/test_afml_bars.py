"""Tests for src/afml/bars.py — alternative bar construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.bars import (
    bar_statistics,
    dollar_bars,
    tick_bars,
    tick_imbalance_bars,
    volume_bars,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_1m_candles(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic 1m OHLCV for testing."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="min")
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0, 0.5, n)
    low = close - rng.uniform(0, 0.5, n)
    opn = close + rng.normal(0, 0.1, n)
    volume = rng.uniform(10, 1000, n)
    return pd.DataFrame({
        "ts": ts,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def candles():
    return _make_1m_candles()


# ---------------------------------------------------------------------------
# Tick bars
# ---------------------------------------------------------------------------

class TestTickBars:
    def test_basic(self, candles):
        bars = tick_bars(candles, ticks_per_bar=60)
        # 500 / 60 = 8 full + 1 partial = 9 total bars
        assert len(bars) == 9
        assert list(bars.columns) == [
            "ts_start", "ts_end", "open", "high", "low", "close",
            "volume", "dollar_volume", "n_candles",
        ]

    def test_all_bars_have_correct_candle_count(self, candles):
        bars = tick_bars(candles, ticks_per_bar=50)
        # All bars except possibly the last should have 50 candles
        assert (bars["n_candles"].iloc[:-1] == 50).all()

    def test_ohlcv_aggregation(self, candles):
        bars = tick_bars(candles, ticks_per_bar=100)
        chunk = candles.iloc[:100]
        b = bars.iloc[0]
        assert b["open"] == chunk["open"].iloc[0]
        assert b["high"] == chunk["high"].max()
        assert b["low"] == chunk["low"].min()
        assert b["close"] == chunk["close"].iloc[-1]
        assert np.isclose(b["volume"], chunk["volume"].sum())


# ---------------------------------------------------------------------------
# Volume bars
# ---------------------------------------------------------------------------

class TestVolumeBars:
    def test_basic(self, candles):
        total_vol = candles["volume"].sum()
        target = total_vol / 10  # should get ~10 bars
        bars = volume_bars(candles, volume_per_bar=target)
        assert 8 <= len(bars) <= 12

    def test_volume_per_bar_approximately_correct(self, candles):
        target = 5000.0
        bars = volume_bars(candles, volume_per_bar=target)
        # Every bar must meet or exceed the threshold
        assert (bars["volume"] >= target * 0.99).all()

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        bars = volume_bars(empty, volume_per_bar=100)
        assert len(bars) == 0


# ---------------------------------------------------------------------------
# Dollar bars
# ---------------------------------------------------------------------------

class TestDollarBars:
    def test_basic(self, candles):
        total_dv = (candles["close"] * candles["volume"]).sum()
        target = total_dv / 8
        bars = dollar_bars(candles, dollars_per_bar=target)
        assert 6 <= len(bars) <= 10

    def test_dollar_volume_meets_threshold(self, candles):
        target = 50_000.0
        bars = dollar_bars(candles, dollars_per_bar=target)
        if len(bars) > 0:
            assert (bars["dollar_volume"] >= target * 0.99).all()


# ---------------------------------------------------------------------------
# Tick imbalance bars
# ---------------------------------------------------------------------------

class TestTickImbalanceBars:
    def test_basic(self, candles):
        bars = tick_imbalance_bars(candles, expected_t=30)
        assert len(bars) > 0
        assert "close" in bars.columns

    def test_adaptive_threshold(self, candles):
        bars = tick_imbalance_bars(candles, expected_t=30, ewma_span=20)
        # Bars should have varying lengths (adaptive)
        if len(bars) > 3:
            lengths = bars["n_candles"].values
            assert len(set(lengths)) > 1, "Expected varying bar lengths"

    def test_trending_vs_random_different_bar_lengths(self):
        """Trending and random markets produce bars with different
        length distributions, showing the adaptive threshold works."""
        n = 1000
        ts = pd.date_range("2024-01-01", periods=n, freq="min")
        close = 100.0 + np.arange(n) * 0.05  # monotonic up
        df = pd.DataFrame({
            "ts": ts,
            "open": close - 0.01,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(n, 100.0),
        })
        bars_trend = tick_imbalance_bars(df, expected_t=50)

        rng = np.random.default_rng(99)
        close_rw = 100.0 + np.cumsum(rng.choice([-0.05, 0.05], n))
        df_rw = df.copy()
        df_rw["close"] = close_rw
        df_rw["open"] = close_rw - 0.01
        bars_rw = tick_imbalance_bars(df_rw, expected_t=50)

        assert len(bars_trend) > 0
        assert len(bars_rw) > 0
        # Mean bar lengths should differ between trend and random
        mean_len_trend = bars_trend["n_candles"].mean()
        mean_len_rw = bars_rw["n_candles"].mean()
        assert mean_len_trend != mean_len_rw


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestBarStatistics:
    def test_basic(self, candles):
        bars = tick_bars(candles, ticks_per_bar=30)
        stats = bar_statistics(bars, label="tick_30")
        assert stats["label"] == "tick_30"
        assert stats["n_bars"] == len(bars)
        assert "skewness" in stats
        assert "jarque_bera" in stats

    def test_empty(self):
        empty = pd.DataFrame(columns=["close"])
        stats = bar_statistics(empty, label="empty")
        assert stats["n_bars"] == 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_column(self):
        df = pd.DataFrame({"ts": [1], "close": [100]})
        with pytest.raises(ValueError, match="Missing columns"):
            tick_bars(df, ticks_per_bar=1)
