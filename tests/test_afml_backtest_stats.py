"""Tests for src/afml/backtest_stats.py — DSR, PBO, Sharpe utils."""
from __future__ import annotations

import numpy as np
import pandas as pd

from afml.backtest_stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    return_stats,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# expected_max_sharpe
# ---------------------------------------------------------------------------

class TestExpectedMaxSharpe:
    def test_increases_with_trials(self):
        e1 = expected_max_sharpe(10)
        e2 = expected_max_sharpe(100)
        e3 = expected_max_sharpe(1000)
        assert e1 < e2 < e3

    def test_single_trial(self):
        assert expected_max_sharpe(1) == 0.0

    def test_zero_trials(self):
        assert expected_max_sharpe(0) == 0.0


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio
# ---------------------------------------------------------------------------

class TestDSR:
    def test_high_sharpe_high_dsr(self):
        dsr = deflated_sharpe_ratio(
            observed_sr=3.0, sr_benchmark=1.0, n_obs=252,
        )
        assert dsr > 0.9

    def test_low_sharpe_low_dsr(self):
        dsr = deflated_sharpe_ratio(
            observed_sr=0.5, sr_benchmark=2.0, n_obs=252,
        )
        assert dsr < 0.1

    def test_non_normality_reduces_dsr(self):
        dsr_normal = deflated_sharpe_ratio(
            observed_sr=1.5, sr_benchmark=1.0, n_obs=252,
            skewness=0, excess_kurtosis=0,
        )
        dsr_fat = deflated_sharpe_ratio(
            observed_sr=1.5, sr_benchmark=1.0, n_obs=252,
            skewness=-1, excess_kurtosis=5,
        )
        # Fat tails + negative skew should reduce DSR
        assert dsr_fat < dsr_normal


# ---------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------

class TestPBO:
    def test_no_overfitting(self):
        # IS and OOS perfectly correlated
        is_scores = np.array([1, 2, 3, 4, 5], dtype=float)
        oos_scores = np.array([1, 2, 3, 4, 5], dtype=float)
        result = probability_of_backtest_overfitting(is_scores, oos_scores)
        assert result["pbo"] < 0.3  # best IS is also best OOS
        assert result["rank_corr"] > 0.9

    def test_overfitting(self):
        # IS and OOS inversely correlated
        is_scores = np.array([5, 4, 3, 2, 1], dtype=float)
        oos_scores = np.array([1, 2, 3, 4, 5], dtype=float)
        result = probability_of_backtest_overfitting(is_scores, oos_scores)
        assert result["pbo"] > 0.7
        assert result["rank_corr"] < -0.5


# ---------------------------------------------------------------------------
# Sharpe helpers
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_positive_sharpe(self):
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0.001, 0.01, 252))
        sr = sharpe_ratio(rets)
        assert sr > 0

    def test_zero_returns(self):
        rets = pd.Series(np.zeros(100))
        assert sharpe_ratio(rets) == 0.0

    def test_return_stats_keys(self):
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0, 0.01, 252))
        stats = return_stats(rets)
        assert "sharpe" in stats
        assert "n_obs" in stats
        assert "skewness" in stats
        assert "excess_kurtosis" in stats
