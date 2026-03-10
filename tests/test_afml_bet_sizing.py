"""Tests for src/afml/bet_sizing.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.bet_sizing import (
    avg_active_signals,
    bet_size_from_prob,
    bet_size_sigmoid,
    discrete_signal,
    max_concurrent_signals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dates():
    return pd.bdate_range("2023-01-02", periods=20)


@pytest.fixture
def t1_short(dates):
    """Each bet spans 3 days."""
    return pd.Series([dates[min(i + 3, 19)] for i in range(20)], index=dates)


# ---------------------------------------------------------------------------
# discrete_signal
# ---------------------------------------------------------------------------

class TestDiscreteSignal:
    def test_step_size(self):
        signal = pd.Series([0.05, 0.15, -0.27, 0.99])
        result = discrete_signal(signal, step_size=0.1)
        # 0.05 → 0.0, 0.15 → 0.1 or 0.2 (float rounding), -0.27 → -0.3, 0.99 → 1.0
        assert all(abs(result - (signal / 0.1).round() * 0.1) < 1e-10)

    def test_identity_at_steps(self):
        signal = pd.Series([0.0, 0.1, -0.5, 1.0])
        result = discrete_signal(signal, step_size=0.1)
        pd.testing.assert_series_equal(result, signal)


# ---------------------------------------------------------------------------
# bet_size_sigmoid
# ---------------------------------------------------------------------------

class TestBetSizeSigmoid:
    def test_zero_signal(self):
        signal = pd.Series([0.0])
        result = bet_size_sigmoid(signal, w=10)
        assert np.isclose(result.iloc[0], 0.0, atol=1e-6)

    def test_positive_signal(self):
        signal = pd.Series([0.5, 1.0])
        result = bet_size_sigmoid(signal, w=10)
        assert all(result > 0)
        assert result.iloc[1] > result.iloc[0]  # higher signal → larger size

    def test_symmetry(self):
        signal = pd.Series([-0.3, 0.3])
        result = bet_size_sigmoid(signal, w=10)
        assert np.isclose(result.iloc[0], -result.iloc[1], atol=1e-6)

    def test_bounded(self):
        signal = pd.Series(np.linspace(-5, 5, 100))
        result = bet_size_sigmoid(signal, w=5)
        assert all(result > -1) and all(result < 1)


# ---------------------------------------------------------------------------
# bet_size_from_prob
# ---------------------------------------------------------------------------

class TestBetSizeFromProb:
    def test_high_prob_positive(self):
        prob = pd.Series([0.9])
        side = pd.Series([1])
        result = bet_size_from_prob(prob, side)
        assert result.iloc[0] > 0

    def test_high_prob_negative_side(self):
        prob = pd.Series([0.9])
        side = pd.Series([-1])
        result = bet_size_from_prob(prob, side)
        assert result.iloc[0] < 0

    def test_fifty_fifty(self):
        prob = pd.Series([0.5])
        result = bet_size_from_prob(prob)
        assert abs(result.iloc[0]) < 0.1  # near zero

    def test_no_side_defaults_long(self):
        prob = pd.Series([0.8, 0.9])
        result = bet_size_from_prob(prob)
        assert all(result > 0)


# ---------------------------------------------------------------------------
# avg_active_signals
# ---------------------------------------------------------------------------

class TestAvgActiveSignals:
    def test_basic(self, dates, t1_short):
        signals = pd.DataFrame({"signal": np.ones(20)}, index=dates)
        avg = avg_active_signals(signals, t1_short)
        assert len(avg) > 0
        assert all(np.isclose(avg, 1.0))  # all signals are 1 → avg is 1


# ---------------------------------------------------------------------------
# max_concurrent_signals
# ---------------------------------------------------------------------------

class TestMaxConcurrent:
    def test_basic(self, t1_short):
        mc = max_concurrent_signals(t1_short)
        assert mc >= 1

    def test_no_overlap(self, dates):
        t1 = pd.Series(dates[:10], index=dates[:10])
        mc = max_concurrent_signals(t1)
        assert mc == 1  # each bet ends at its own start

    def test_full_overlap(self, dates):
        t1 = pd.Series([dates[-1]] * 20, index=dates)
        mc = max_concurrent_signals(t1)
        assert mc == 20
