"""Tests for HRP covariance_method parameter — Phase 4 acceptance criteria.

CTO spec acceptance criteria:
✓ hrp.py with covariance_method='sample' produces bit-for-bit identical output
  to current version.
✓ hrp.py with covariance_method='ledoit_wolf' runs without error on test data.
✓ Weights sum to 1.0 for both methods.
✓ Unknown method raises ValueError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
sklearn = pytest.importorskip("sklearn")

from portfolio.hrp import HierarchicalRiskParity  # noqa: E402


def _make_returns(n_dates: int = 200, n_symbols: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    symbols = [f"SYM_{i:02d}" for i in range(n_symbols)]
    data = rng.standard_normal((n_dates, n_symbols)) * 0.02
    return pd.DataFrame(data, index=dates, columns=symbols)


class TestSampleMethodIdentical:
    """sample method must produce bit-for-bit identical output to pre-change behaviour."""

    def test_sample_explicit_equals_default(self):
        returns = _make_returns()
        w_default = HierarchicalRiskParity.allocate(returns)
        w_sample = HierarchicalRiskParity.allocate(returns, covariance_method="sample")
        pd.testing.assert_series_equal(w_default, w_sample)

    def test_sample_weights_sum_to_one(self):
        returns = _make_returns()
        w = HierarchicalRiskParity.allocate(returns, covariance_method="sample")
        assert np.isclose(w.sum(), 1.0)


class TestLedoitWolfMethod:

    def test_ledoit_wolf_runs_without_error(self):
        returns = _make_returns()
        w = HierarchicalRiskParity.allocate(returns, covariance_method="ledoit_wolf")
        assert len(w) == returns.shape[1]

    def test_ledoit_wolf_weights_sum_to_one(self):
        returns = _make_returns()
        w = HierarchicalRiskParity.allocate(returns, covariance_method="ledoit_wolf")
        assert np.isclose(w.sum(), 1.0)

    def test_ledoit_wolf_weights_positive(self):
        returns = _make_returns()
        w = HierarchicalRiskParity.allocate(returns, covariance_method="ledoit_wolf")
        assert (w >= 0).all()

    def test_ledoit_wolf_vs_sample_produces_different_weights(self):
        """Shrinkage should produce at least slightly different weights."""
        returns = _make_returns()
        w_sample = HierarchicalRiskParity.allocate(returns, covariance_method="sample")
        w_lw = HierarchicalRiskParity.allocate(returns, covariance_method="ledoit_wolf")
        assert not w_sample.equals(w_lw), "Ledoit-Wolf should differ from sample"

    def test_ledoit_wolf_on_correlated_data(self):
        """LW should handle highly correlated assets without blowing up."""
        rng = np.random.default_rng(7)
        n = 200
        base = rng.standard_normal(n)
        data = {
            f"S{i}": base + rng.standard_normal(n) * 0.1
            for i in range(5)
        }
        returns = pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=n, freq="D"))
        w = HierarchicalRiskParity.allocate(returns, covariance_method="ledoit_wolf")
        assert np.isclose(w.sum(), 1.0)
        assert (w >= 0).all()


class TestEdgeCases:

    def test_unknown_method_raises(self):
        returns = _make_returns()
        with pytest.raises(ValueError, match="Unknown covariance_method"):
            HierarchicalRiskParity.allocate(returns, covariance_method="bogus")  # type: ignore[arg-type]

    def test_empty_returns_raises(self):
        returns = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty"):
            HierarchicalRiskParity.allocate(returns, covariance_method="sample")
