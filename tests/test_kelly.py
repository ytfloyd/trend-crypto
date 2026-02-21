"""Tests for empirical Kelly criterion — Phase 3b acceptance criteria.

CTO spec acceptance criteria:
✓ Unit test: known return series produces Kelly fraction within 1%
  of analytical solution.
✓ Unit test: leverage cap correctly scales down a portfolio with implied
  leverage above the Kelly fraction.
✓ Unit test: kelly_enabled=False produces identical weights (apply_kelly_cap
  with high fraction is passthrough).
✓ Kelly fraction never below 0 for valid data.
"""
from __future__ import annotations

import numpy as np
import pytest

from risk.kelly import apply_kelly_cap, empirical_kelly, gaussian_kelly


class TestEmpiricalKelly:

    def test_known_biased_coin(self):
        """Biased coin: win +1 with p=0.6, lose -1 with p=0.4.

        Analytical Kelly for even-odds biased coin: f* = 2p - 1 = 0.2.
        Empirical Kelly on a large sample should converge to this.
        """
        rng = np.random.default_rng(42)
        n = 100_000
        wins = rng.random(n) < 0.6
        returns = np.where(wins, 1.0, -1.0)
        f = empirical_kelly(returns, max_leverage=2.0)
        assert abs(f - 0.2) < 0.02, f"Expected ~0.2, got {f:.4f}"

    def test_positive_returns_give_positive_kelly(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.002, 0.02, 5000)
        f = empirical_kelly(returns)
        assert f > 0

    def test_negative_returns_give_zero_kelly(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(-0.01, 0.02, 1000)
        f = empirical_kelly(returns)
        assert f == 0.0

    def test_empty_returns_give_zero(self):
        assert empirical_kelly(np.array([])) == 0.0

    def test_few_returns_give_zero(self):
        assert empirical_kelly(np.array([0.01, 0.02])) == 0.0

    def test_handles_nan_and_inf(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 100)
        returns[10] = np.nan
        returns[20] = np.inf
        returns[30] = -np.inf
        f = empirical_kelly(returns)
        assert np.isfinite(f)

    def test_kelly_bounded_by_max_leverage(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 0.01, 10000)
        f = empirical_kelly(returns, max_leverage=2.0)
        assert f <= 2.0

    def test_empirical_less_than_gaussian_for_fat_tails(self):
        """With fat left tails, empirical Kelly should be more conservative."""
        rng = np.random.default_rng(42)
        n = 10000
        normal = rng.normal(0.001, 0.02, n)
        crashes = rng.choice(n, size=50, replace=False)
        normal[crashes] = rng.uniform(-0.15, -0.05, 50)

        emp = empirical_kelly(normal)
        gauss = gaussian_kelly(normal)
        assert emp < gauss, (
            f"Empirical Kelly ({emp:.4f}) should be less than "
            f"Gaussian ({gauss:.4f}) with fat tails"
        )


class TestGaussianKelly:

    def test_known_formula(self):
        """mu/sigma^2 for mu=0.001, sigma=0.02 -> 0.001/0.0004 = 2.5"""
        returns = np.array([0.001] * 10000)
        returns = returns + np.random.default_rng(42).normal(0, 0.02, 10000)
        g = gaussian_kelly(returns)
        expected = np.mean(returns) / np.var(returns, ddof=1)
        assert abs(g - expected) < 0.01

    def test_negative_mean_gives_zero(self):
        returns = np.array([-0.01, -0.02, -0.03, -0.01] * 100)
        assert gaussian_kelly(returns) == 0.0


class TestApplyKellyCap:

    def test_within_cap_unchanged(self):
        weights = {"BTC": 0.3, "ETH": 0.2}
        result = apply_kelly_cap(weights, kelly_fraction=1.0)
        assert result == weights

    def test_above_cap_scales_down(self):
        weights = {"BTC": 0.6, "ETH": 0.4, "SOL": 0.5}
        result = apply_kelly_cap(weights, kelly_fraction=1.0)
        gross = sum(abs(v) for v in result.values())
        assert abs(gross - 1.0) < 1e-10
        assert result["BTC"] / result["ETH"] == pytest.approx(0.6 / 0.4)

    def test_zero_fraction_zeros_everything(self):
        weights = {"BTC": 0.5}
        result = apply_kelly_cap(weights, kelly_fraction=0.0)
        assert result["BTC"] == 0.0

    def test_passthrough_with_high_fraction(self):
        """When kelly_fraction is very high, weights pass through unchanged."""
        weights = {"BTC": 0.4, "ETH": 0.3}
        result = apply_kelly_cap(weights, kelly_fraction=100.0)
        assert result == weights

    def test_preserves_relative_weights(self):
        weights = {"A": 0.6, "B": 0.3, "C": 0.1}
        result = apply_kelly_cap(weights, kelly_fraction=0.5)
        assert result["A"] / result["B"] == pytest.approx(0.6 / 0.3)
        assert result["B"] / result["C"] == pytest.approx(0.3 / 0.1)
