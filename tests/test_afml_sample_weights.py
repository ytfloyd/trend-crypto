"""Tests for src/afml/sample_weights.py — concurrency, uniqueness, sequential bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.sample_weights import (
    average_uniqueness,
    label_concurrency,
    normalize_weights,
    sample_weight_by_return,
    sequential_bootstrap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dates():
    return pd.bdate_range("2023-01-02", periods=20)


@pytest.fixture
def close(dates):
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def t1_no_overlap(dates):
    """Three non-overlapping labels."""
    return pd.Series(
        [dates[2], dates[6], dates[10]],
        index=[dates[0], dates[4], dates[8]],
    )


@pytest.fixture
def t1_full_overlap(dates):
    """Three fully overlapping labels (same span)."""
    return pd.Series(
        [dates[5], dates[5], dates[5]],
        index=[dates[0], dates[0], dates[0]],
    )


@pytest.fixture
def t1_partial_overlap(dates):
    """Labels with partial overlap."""
    return pd.Series(
        [dates[4], dates[7], dates[10]],
        index=[dates[0], dates[3], dates[6]],
    )


# ---------------------------------------------------------------------------
# Label concurrency
# ---------------------------------------------------------------------------

class TestLabelConcurrency:
    def test_no_overlap(self, t1_no_overlap, dates):
        conc = label_concurrency(t1_no_overlap, dates)
        # Bars 0-2: label 1 active (conc=1), bars 3: nothing, bars 4-6: label 2, etc.
        assert conc.iloc[0] == 1
        assert conc.iloc[3] == 0
        assert conc.iloc[5] == 1

    def test_full_overlap(self, t1_full_overlap, dates):
        conc = label_concurrency(t1_full_overlap, dates)
        # All three labels span dates[0] to dates[5]
        for i in range(6):
            assert conc.iloc[i] == 3
        assert conc.iloc[6] == 0

    def test_partial_overlap(self, t1_partial_overlap, dates):
        conc = label_concurrency(t1_partial_overlap, dates)
        # dates[0]-[2]: only label 1 -> conc=1
        assert conc.iloc[0] == 1
        # dates[3]-[4]: labels 1 and 2 overlap -> conc=2
        assert conc.iloc[3] == 2
        assert conc.iloc[4] == 2
        # dates[6]-[7]: labels 2 and 3 overlap -> conc=2
        assert conc.iloc[6] == 2


# ---------------------------------------------------------------------------
# Average uniqueness
# ---------------------------------------------------------------------------

class TestAverageUniqueness:
    def test_no_overlap_is_one(self, t1_no_overlap, dates):
        uniq = average_uniqueness(t1_no_overlap, dates)
        # Non-overlapping labels should have uniqueness = 1.0
        assert np.allclose(uniq.dropna().values, 1.0)

    def test_full_overlap_is_one_third(self, t1_full_overlap, dates):
        uniq = average_uniqueness(t1_full_overlap, dates)
        # Three fully overlapping labels -> uniqueness = 1/3 each
        assert np.allclose(uniq.dropna().values, 1.0 / 3.0)

    def test_partial_overlap_between(self, t1_partial_overlap, dates):
        uniq = average_uniqueness(t1_partial_overlap, dates)
        # Partial overlap -> uniqueness between 1/3 and 1.0
        for v in uniq.dropna().values:
            assert 0.3 < v < 1.0


# ---------------------------------------------------------------------------
# Sample weights by return
# ---------------------------------------------------------------------------

class TestSampleWeightByReturn:
    def test_basic(self, t1_partial_overlap, close):
        weights = sample_weight_by_return(t1_partial_overlap, close)
        assert len(weights) == len(t1_partial_overlap)
        assert (weights.dropna() >= 0).all()

    def test_no_overlap_equal_weights(self, t1_no_overlap, close):
        weights = sample_weight_by_return(t1_no_overlap, close)
        # Non-overlapping labels with similar vol should have similar weights
        w = weights.dropna().values
        assert w.std() / w.mean() < 1.0  # CV < 1

    def test_normalize(self, t1_partial_overlap, close):
        weights = sample_weight_by_return(t1_partial_overlap, close)
        normed = normalize_weights(weights)
        assert np.isclose(normed.sum(), len(normed))


# ---------------------------------------------------------------------------
# Sequential bootstrap
# ---------------------------------------------------------------------------

class TestSequentialBootstrap:
    def test_basic(self, t1_partial_overlap, dates):
        idx = sequential_bootstrap(t1_partial_overlap, dates, n_samples=10)
        assert len(idx) == 10
        assert all(0 <= i < len(t1_partial_overlap) for i in idx)

    def test_no_overlap_uniform(self, t1_no_overlap, dates):
        """Non-overlapping labels should be drawn roughly uniformly."""
        idx = sequential_bootstrap(t1_no_overlap, dates, n_samples=3000,
                                   random_state=42)
        counts = np.bincount(idx, minlength=3)
        # Each should be ~1000; allow wide tolerance
        assert all(c > 500 for c in counts)

    def test_full_overlap_still_draws(self, t1_full_overlap, dates):
        """Even fully overlapping labels should produce valid draws."""
        idx = sequential_bootstrap(t1_full_overlap, dates, n_samples=10)
        assert len(idx) == 10

    def test_reproducible(self, t1_partial_overlap, dates):
        idx1 = sequential_bootstrap(t1_partial_overlap, dates, random_state=7)
        idx2 = sequential_bootstrap(t1_partial_overlap, dates, random_state=7)
        assert np.array_equal(idx1, idx2)

    def test_favors_unique_labels(self, dates):
        """Label A is non-overlapping, B and C overlap heavily.
        Sequential bootstrap should favor A."""
        t1 = pd.Series(
            [dates[2], dates[8], dates[8]],
            index=[dates[0], dates[3], dates[3]],
        )
        idx = sequential_bootstrap(t1, dates, n_samples=3000, random_state=42)
        counts = np.bincount(idx, minlength=3)
        # Label 0 (non-overlapping) should be picked more than either of 1 or 2
        assert counts[0] > counts[1]
        assert counts[0] > counts[2]
