"""Tests for information_horizon() in scripts/research/common/metrics.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

# ---------------------------------------------------------------------------
# Import research common package.
# scripts/research/common/ and src/common/ share the name "common".
# conftest.py already adds src/ to sys.path, which registers src/common/.
# We temporarily swap it out to load the research variant, then restore.
# ---------------------------------------------------------------------------
_RESEARCH_DIR = str(Path(__file__).resolve().parents[1] / "scripts" / "research")
_saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "common" or k.startswith("common.")}
sys.path.insert(0, _RESEARCH_DIR)
try:
    import common.metrics as _metrics_mod  # loads scripts/research/common/
finally:
    sys.path.remove(_RESEARCH_DIR)
    # Remove the research variants from cache, restore originals
    for k in list(sys.modules):
        if k == "common" or k.startswith("common."):
            sys.modules.pop(k, None)
    sys.modules.update(_saved)

information_horizon = _metrics_mod.information_horizon
_cross_sectional_ic = _metrics_mod._cross_sectional_ic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_panel(n_dates: int = 200, n_symbols: int = 10, seed: int = 42):
    """Create synthetic signal and returns panels for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    symbols = [f"SYM_{i:02d}" for i in range(n_symbols)]

    signal = pd.DataFrame(
        rng.standard_normal((n_dates, n_symbols)),
        index=dates, columns=symbols,
    )
    noise = rng.standard_normal((n_dates, n_symbols)) * 0.05
    returns = pd.DataFrame(
        signal.values * 0.01 + noise,
        index=dates, columns=symbols,
    )
    return signal, returns


def _make_perfect_signal(n_dates: int = 200, n_symbols: int = 10, seed: int = 0):
    """Signal perfectly predicts 1-period forward returns (high IC at horizon 1).

    signal(t) = returns(t+1), i.e. perfect foresight of next-period return.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    symbols = [f"SYM_{i:02d}" for i in range(n_symbols)]
    returns = pd.DataFrame(
        rng.standard_normal((n_dates, n_symbols)) * 0.02,
        index=dates, columns=symbols,
    )
    signal = returns.shift(-1).fillna(0.0)
    return signal, returns


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInformationHorizon:

    def test_output_shape_and_columns(self):
        signal, returns = _make_panel()
        horizons = [1, 5, 10, 20]
        result = information_horizon(signal, returns, horizons=horizons)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(horizons)
        expected_cols = {"horizon", "ic_mean", "ic_std", "ic_tstat", "ic_pval", "n_periods", "hit_rate"}
        assert set(result.columns) == expected_cols

    def test_ic_values_in_valid_range(self):
        signal, returns = _make_panel()
        result = information_horizon(signal, returns, horizons=[1, 5, 20])
        valid = result.dropna(subset=["ic_mean"])
        assert (valid["ic_mean"].between(-1.0, 1.0)).all(), "IC must be in [-1, 1]"

    def test_pval_in_valid_range(self):
        signal, returns = _make_panel()
        result = information_horizon(signal, returns, horizons=[1, 5, 20])
        valid = result.dropna(subset=["ic_pval"])
        assert (valid["ic_pval"].between(0.0, 1.0)).all(), "p-value must be in [0, 1]"

    def test_hit_rate_in_valid_range(self):
        signal, returns = _make_panel()
        result = information_horizon(signal, returns, horizons=[1, 5, 20])
        valid = result.dropna(subset=["hit_rate"])
        assert (valid["hit_rate"].between(0.0, 1.0)).all(), "hit_rate must be in [0, 1]"

    def test_perfect_signal_has_high_ic_at_short_horizon(self):
        """A signal that perfectly predicts 1-bar returns should have high IC at horizon 1."""
        signal, returns = _make_perfect_signal(n_dates=500, n_symbols=20)
        result = information_horizon(signal, returns, horizons=[1, 5, 20, 60])
        ic_h1 = result.loc[result["horizon"] == 1, "ic_mean"].iloc[0]
        assert ic_h1 > 0.3, f"Perfect signal should have high IC at h=1, got {ic_h1:.4f}"

    def test_random_signal_has_near_zero_ic(self):
        """Uncorrelated signal should have IC near zero at all horizons."""
        rng = np.random.default_rng(99)
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        symbols = [f"S{i}" for i in range(20)]
        signal = pd.DataFrame(rng.standard_normal((500, 20)), index=dates, columns=symbols)
        returns = pd.DataFrame(rng.standard_normal((500, 20)) * 0.02, index=dates, columns=symbols)
        result = information_horizon(signal, returns, horizons=[1, 5, 20])
        for _, row in result.iterrows():
            assert abs(row["ic_mean"]) < 0.15, (
                f"Random signal IC should be near zero, got {row['ic_mean']:.4f} at h={int(row['horizon'])}"
            )

    def test_too_few_symbols_raises(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        signal = pd.DataFrame({"A": np.ones(100)}, index=dates)
        returns = pd.DataFrame({"A": np.ones(100) * 0.01}, index=dates)
        with pytest.raises(ValueError, match="Need >= 2 common symbols"):
            information_horizon(signal, returns, horizons=[1])

    def test_n_periods_is_reasonable(self):
        signal, returns = _make_panel(n_dates=200)
        result = information_horizon(signal, returns, horizons=[1, 5, 50])
        for _, row in result.iterrows():
            h = int(row["horizon"])
            n = int(row["n_periods"])
            assert n <= 200 - h, f"n_periods={n} exceeds theoretical max for horizon {h}"
            assert n > 0, f"n_periods should be positive for horizon {h}"

    def test_horizon_ordering_preserved(self):
        signal, returns = _make_panel()
        horizons = [20, 1, 10, 5]
        result = information_horizon(signal, returns, horizons=horizons)
        assert list(result["horizon"]) == horizons


class TestCrossSectionalIC:

    def test_basic_positive_correlation(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        symbols = [f"S{i}" for i in range(10)]
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 10))
        signal = pd.DataFrame(data, index=dates, columns=symbols)
        forward_ret = pd.DataFrame(data * 0.5 + rng.standard_normal((50, 10)) * 0.1,
                                   index=dates, columns=symbols)
        ic = _cross_sectional_ic(signal, forward_ret, method="spearman")
        assert len(ic) > 0
        assert ic.mean() > 0.3, "Correlated data should produce positive IC"

    def test_handles_nan_gracefully(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        symbols = [f"S{i}" for i in range(10)]
        signal = pd.DataFrame(np.ones((50, 10)), index=dates, columns=symbols)
        signal.iloc[:, :5] = np.nan
        forward_ret = pd.DataFrame(np.ones((50, 10)) * 0.01, index=dates, columns=symbols)
        ic = _cross_sectional_ic(signal, forward_ret)
        assert isinstance(ic, pd.Series)
