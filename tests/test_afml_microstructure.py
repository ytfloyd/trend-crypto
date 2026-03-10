"""Tests for src/afml/microstructure.py — VPIN, Amihud, Kyle, Roll."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.microstructure import (
    amihud_illiquidity,
    kyle_lambda,
    roll_spread,
    vpin,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv():
    """Synthetic daily data with price and volume."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2022-01-03", periods=n)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    vol = rng.integers(100, 10000, n).astype(float)
    return pd.Series(prices, index=dates, name="close"), pd.Series(vol, index=dates, name="volume")


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------

class TestVPIN:
    def test_output_columns(self, ohlcv):
        close, volume = ohlcv
        result = vpin(close, volume, n_buckets=50, window=10)
        assert "vpin" in result.columns
        assert "bucket_end_date" in result.columns

    def test_vpin_bounded(self, ohlcv):
        close, volume = ohlcv
        result = vpin(close, volume, n_buckets=50, window=10)
        valid = result["vpin"].dropna()
        assert all(valid >= 0)
        assert all(valid <= 1)

    def test_more_buckets_more_rows(self, ohlcv):
        close, volume = ohlcv
        r1 = vpin(close, volume, n_buckets=20, window=5)
        r2 = vpin(close, volume, n_buckets=50, window=5)
        assert len(r2) >= len(r1)


# ---------------------------------------------------------------------------
# Amihud
# ---------------------------------------------------------------------------

class TestAmihud:
    def test_output_shape(self, ohlcv):
        close, volume = ohlcv
        result = amihud_illiquidity(close, volume, window=21)
        assert len(result) == len(close)

    def test_positive(self, ohlcv):
        close, volume = ohlcv
        result = amihud_illiquidity(close, volume, window=21).dropna()
        assert all(result >= 0)


# ---------------------------------------------------------------------------
# Kyle's lambda
# ---------------------------------------------------------------------------

class TestKyleLambda:
    def test_output_not_empty(self, ohlcv):
        close, volume = ohlcv
        result = kyle_lambda(close, volume, window=21)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Roll spread
# ---------------------------------------------------------------------------

class TestRollSpread:
    def test_output_shape(self, ohlcv):
        close, volume = ohlcv
        result = roll_spread(close, window=21)
        assert len(result) == len(close)

    def test_non_negative(self, ohlcv):
        close, volume = ohlcv
        result = roll_spread(close, window=21).dropna()
        assert all(result >= 0)
