"""Tests for src/afml/features.py — CUSUM, SADF, entropy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.features import (
    cusum_filter,
    encode_returns_binary,
    lempel_ziv_complexity,
    plugin_entropy,
    sadf,
    shannon_entropy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_close():
    """Strong uptrend — should trigger CUSUM events frequently."""
    dates = pd.bdate_range("2023-01-02", periods=200)
    rng = np.random.default_rng(42)
    return pd.Series(
        100 + np.cumsum(rng.normal(0.5, 1, 200)),
        index=dates,
    )


@pytest.fixture
def random_close():
    dates = pd.bdate_range("2023-01-02", periods=200)
    rng = np.random.default_rng(42)
    return pd.Series(
        100 + np.cumsum(rng.normal(0, 1, 200)),
        index=dates,
    )


# ---------------------------------------------------------------------------
# CUSUM filter
# ---------------------------------------------------------------------------

class TestCUSUM:
    def test_events_returned(self, trending_close):
        events = cusum_filter(trending_close, threshold=5.0)
        assert isinstance(events, pd.DatetimeIndex)
        assert len(events) > 0

    def test_higher_threshold_fewer_events(self, trending_close):
        e_low = cusum_filter(trending_close, threshold=3.0)
        e_high = cusum_filter(trending_close, threshold=10.0)
        assert len(e_low) >= len(e_high)

    def test_flat_series_no_events(self):
        dates = pd.bdate_range("2023-01-02", periods=100)
        flat = pd.Series(100.0, index=dates)
        events = cusum_filter(flat, threshold=1.0)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# SADF
# ---------------------------------------------------------------------------

class TestSADF:
    def test_output_shape(self, random_close):
        result = sadf(np.log(random_close), min_window=50, lags=1)
        assert "sadf_stat" in result.columns
        assert len(result) > 0

    def test_explosive_series_high_sadf(self):
        dates = pd.bdate_range("2023-01-02", periods=150)
        rng = np.random.default_rng(42)
        # Explosive: price doubles every ~20 periods
        prices = 100 * np.exp(np.cumsum(rng.normal(0.03, 0.01, 150)))
        log_p = pd.Series(np.log(prices), index=dates)
        result = sadf(log_p, min_window=30, lags=1)
        # Should have positive (explosive) SADF values
        assert result["sadf_stat"].max() > 0


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

class TestShannonEntropy:
    def test_uniform_high_entropy(self):
        rng = np.random.default_rng(42)
        series = pd.Series(rng.uniform(0, 1, 1000))
        h = shannon_entropy(series, n_bins=20)
        assert h > 2.0  # close to log(20) ≈ 3.0

    def test_constant_zero_entropy(self):
        series = pd.Series(np.ones(100))
        h = shannon_entropy(series, n_bins=20)
        assert h == 0.0


class TestPluginEntropy:
    def test_binary(self):
        h = plugin_entropy("0101010101")
        assert np.isclose(h, np.log(2), atol=0.01)

    def test_single_symbol(self):
        h = plugin_entropy("aaaaaaa")
        assert h == 0.0


class TestLempelZiv:
    def test_repetitive_low_complexity(self):
        c = lempel_ziv_complexity("0" * 100)
        assert c < 0.2

    def test_random_higher_complexity(self):
        rng = np.random.default_rng(42)
        seq = "".join(rng.choice(["0", "1"], size=200))
        c = lempel_ziv_complexity(seq)
        assert c > 0.3

    def test_empty(self):
        assert lempel_ziv_complexity("") == 0.0


class TestEncodeReturns:
    def test_basic(self):
        rets = pd.Series([0.01, -0.02, 0.03, -0.01])
        encoded = encode_returns_binary(rets)
        assert encoded == "1010"
