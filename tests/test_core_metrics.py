"""Characterization tests for the unified ``core.metrics`` module.

These lock the *current* numeric behavior of the canonical equity-curve metrics
(moved verbatim from ``scripts/research/common/metrics.py``) so future refactors
can't silently drift the numbers, and verify the deprecated
``scripts/research/common/metrics.py`` shim re-exports the same objects.

See docs/RESEARCH_PIPELINE_REORGANIZATION.md (Phase 1: unify the core stack).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import metrics as core_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM_PATH = REPO_ROOT / "scripts" / "research" / "common" / "metrics.py"


def _fixed_equity() -> pd.Series:
    """Deterministic equity curve: 10 fixed daily returns -> 11 equity points."""
    idx = pd.date_range("2021-01-01", periods=11, freq="D")
    rets = np.array(
        [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.025, 0.04, -0.005, 0.01]
    )
    nav = np.concatenate([[1.0], np.cumprod(1 + rets)])
    return pd.Series(nav, index=idx)


# Golden master: values produced by core.metrics.compute_metrics on _fixed_equity()
# at the time of the verbatim move. ANN_FACTOR = 365.0. Do NOT update these to
# "fix" a number without an explicit decision recorded in the reorg plan — a
# change here means the metric definition changed.
_GOLDEN = {
    "total_return": 0.05952981034597049,
    "cagr": 5.812413208192085,
    "vol": 0.40477703052970376,
    "sharpe": 5.410386051634619,
    "sortino": 12.557069721873797,
    "calmar": 229.57325725640683,
    "max_dd": -0.025318337500000232,
    "hit_rate": 0.6,
    "skewness": 0.056515839757700796,
    "kurtosis": -0.9013789351884274,
    "n_days": 11,
}


def test_compute_metrics_golden_master():
    m = core_metrics.compute_metrics(_fixed_equity())
    assert set(m.keys()) == set(_GOLDEN.keys())
    for key, expected in _GOLDEN.items():
        assert m[key] == pytest.approx(expected, rel=1e-12), f"metric '{key}' drifted"


def test_ann_factor_is_365():
    assert core_metrics.ANN_FACTOR == 365.0


def test_compute_metrics_too_short_returns_all_nan():
    """Fewer than 2 points -> all-NaN dict with the expected keys."""
    m = core_metrics.compute_metrics(pd.Series([1.0]))
    expected_keys = {
        "cagr", "vol", "sharpe", "sortino", "calmar",
        "max_dd", "hit_rate", "skewness", "kurtosis",
        "n_days", "total_return",
    }
    assert set(m.keys()) == expected_keys
    assert all(isinstance(v, float) and math.isnan(v) for v in m.values())


def _load_shim():
    """Load the research shim by file path to avoid the `src/common` vs
    `scripts/research/common` top-level `common` package name collision."""
    spec = importlib.util.spec_from_file_location("_research_common_metrics_shim", SHIM_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shim_reexports_same_objects():
    """The deprecated common.metrics shim must re-export core.metrics verbatim."""
    shim = _load_shim()
    # Same function objects (identity) => zero behavior divergence by construction.
    assert shim.compute_metrics is core_metrics.compute_metrics
    assert shim.information_horizon is core_metrics.information_horizon
    assert shim.compute_ic_decay is core_metrics.compute_ic_decay
    assert shim.compute_yearly_sharpe_trend is core_metrics.compute_yearly_sharpe_trend
    assert shim.format_metrics_table is core_metrics.format_metrics_table
    assert shim.compute_regime is core_metrics.compute_regime
    assert shim._cross_sectional_ic is core_metrics._cross_sectional_ic
    assert shim.ANN_FACTOR == core_metrics.ANN_FACTOR


def test_shim_all_names_importable():
    shim = _load_shim()
    for name in shim.__all__:
        assert hasattr(shim, name), f"shim.__all__ lists '{name}' but it is not present"
