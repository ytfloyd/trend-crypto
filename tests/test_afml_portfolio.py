"""Tests for src/afml/portfolio.py — HRP and weight utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.portfolio import (
    correlation_distance,
    equal_weights,
    hrp_weights,
    inverse_variance_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    n, k = 252, 5
    data = rng.normal(0, 0.02, (n, k))
    # Add correlation structure
    data[:, 1] += data[:, 0] * 0.5
    data[:, 3] += data[:, 2] * 0.7
    return pd.DataFrame(data, columns=["A", "B", "C", "D", "E"])


# ---------------------------------------------------------------------------
# correlation_distance
# ---------------------------------------------------------------------------

class TestCorrelationDistance:
    def test_self_distance_zero(self, returns):
        corr = returns.corr()
        dist = correlation_distance(corr)
        np.testing.assert_allclose(np.diag(dist.values), 0, atol=1e-10)

    def test_range(self, returns):
        corr = returns.corr()
        dist = correlation_distance(corr)
        assert (dist.values >= -1e-10).all()
        assert (dist.values <= 1 + 1e-10).all()


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------

class TestHRP:
    def test_weights_sum_to_one(self, returns):
        w = hrp_weights(returns)
        assert np.isclose(w.sum(), 1.0, atol=1e-8)

    def test_all_positive(self, returns):
        w = hrp_weights(returns)
        assert (w > 0).all()

    def test_correct_index(self, returns):
        w = hrp_weights(returns)
        assert list(w.index) == list(returns.columns)

    def test_different_linkage(self, returns):
        w1 = hrp_weights(returns, method="single")
        w2 = hrp_weights(returns, method="ward")
        # Both should produce valid weights regardless of method
        assert np.isclose(w1.sum(), 1.0)
        assert np.isclose(w2.sum(), 1.0)


# ---------------------------------------------------------------------------
# Inverse variance
# ---------------------------------------------------------------------------

class TestInverseVariance:
    def test_sum_to_one(self, returns):
        w = inverse_variance_weights(returns)
        assert np.isclose(w.sum(), 1.0)

    def test_all_positive(self, returns):
        w = inverse_variance_weights(returns)
        assert (w > 0).all()


# ---------------------------------------------------------------------------
# Equal weight
# ---------------------------------------------------------------------------

class TestEqualWeight:
    def test_equal(self, returns):
        w = equal_weights(returns)
        assert np.allclose(w.values, 0.2)
