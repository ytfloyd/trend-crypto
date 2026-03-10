"""Tests for src/afml/fracdiff.py — fractional differentiation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.fracdiff import frac_diff, frac_diff_log, frac_diff_weights, find_min_d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def close():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", periods=1000)
    log_ret = rng.normal(0.0003, 0.02, len(dates))
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(prices, index=dates, name="close")


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

class TestFracDiffWeights:
    def test_d_zero(self):
        w = frac_diff_weights(0.0)
        assert len(w) == 1
        assert w[0] == 1.0

    def test_d_one(self):
        w = frac_diff_weights(1.0, threshold=1e-10)
        # d=1: w = [1, -1, 0, 0, ...] -> should be length 2
        assert len(w) == 2
        assert np.isclose(w[0], 1.0)
        assert np.isclose(w[1], -1.0)

    def test_d_half_decays(self):
        w = frac_diff_weights(0.5)
        # Weights should decay in absolute value
        abs_w = np.abs(w)
        assert all(abs_w[i] >= abs_w[i + 1] for i in range(len(abs_w) - 1))

    def test_threshold_controls_length(self):
        w_tight = frac_diff_weights(0.3, threshold=1e-3)
        w_loose = frac_diff_weights(0.3, threshold=1e-6)
        assert len(w_loose) > len(w_tight)


# ---------------------------------------------------------------------------
# Fractional differencing
# ---------------------------------------------------------------------------

class TestFracDiff:
    def test_d_one_is_diff(self, close):
        """d=1 should approximate first-differencing of log prices."""
        fd1 = frac_diff_log(close, d=1.0, threshold=1e-10)
        log_ret = np.log(close / close.shift(1))
        common = fd1.dropna().index.intersection(log_ret.dropna().index)
        corr = fd1.reindex(common).corr(log_ret.reindex(common))
        assert corr > 0.999

    def test_d_zero_preserves(self, close):
        """d=0 should return the original log series."""
        log_close = np.log(close)
        fd0 = frac_diff(log_close, d=0.0)
        # d=0 weights = [1.0], so output = input (shifted by window-1=0)
        assert np.allclose(fd0.dropna().values, log_close.values, atol=1e-10)

    def test_intermediate_d(self, close):
        """d=0.4 should produce something between log prices and log returns."""
        fd04 = frac_diff_log(close, d=0.4)
        log_close = np.log(close)
        log_ret = np.log(close / close.shift(1))

        common = fd04.dropna().index
        corr_price = fd04.reindex(common).corr(log_close.reindex(common))
        corr_ret = fd04.reindex(common).corr(log_ret.reindex(common))

        # Should correlate with both but not perfectly with either
        assert 0.1 < corr_price < 0.99
        assert 0.1 < corr_ret < 0.99

    def test_monotonic_d(self, close):
        """Higher d -> less correlation with original prices."""
        log_close = np.log(close)
        corrs = []
        for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
            fd = frac_diff(log_close, d)
            common = fd.dropna().index
            corrs.append(fd.reindex(common).corr(log_close.reindex(common)))
        # Correlation should decrease as d increases
        assert all(corrs[i] >= corrs[i + 1] - 0.01 for i in range(len(corrs) - 1))

    def test_output_length(self, close):
        fd = frac_diff_log(close, d=0.5)
        assert len(fd) == len(close)
        # Should have some leading NaNs
        assert fd.isna().sum() > 0
        assert fd.notna().sum() > 0


# ---------------------------------------------------------------------------
# find_min_d
# ---------------------------------------------------------------------------

class TestFindMinD:
    def test_basic(self, close):
        result = find_min_d(close, d_range=np.arange(0.0, 1.05, 0.1))
        assert "min_d" in result
        assert "results" in result
        assert len(result["results"]) == 11  # 0.0 to 1.0 in 0.1 steps

    def test_min_d_is_reasonable(self, close):
        result = find_min_d(close, d_range=np.arange(0.0, 1.05, 0.1))
        # For a random walk with drift, min_d should be > 0 and <= 1
        if not np.isnan(result["min_d"]):
            assert 0.0 < result["min_d"] <= 1.0

    def test_correlation_preserved(self, close):
        result = find_min_d(close, d_range=np.arange(0.0, 1.05, 0.1))
        if not np.isnan(result["min_d"]):
            # At min_d, should retain meaningful correlation with log prices
            assert result["correlation_at_min_d"] > 0.3
