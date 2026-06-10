"""Characterization tests for the unified ``core.data`` module.

Locks the constants, the path-resolution contract (DEFAULT_DB unchanged across
the move), the pure helper behaviors that need no DuckDB file, and the
``scripts/research/common/data.py`` shim's re-export identity.

See docs/RESEARCH_PIPELINE_REORGANIZATION.md (Phase 1: unify the core stack).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import data as core_data

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM_PATH = REPO_ROOT / "scripts" / "research" / "common" / "data.py"


def test_constants():
    assert core_data.ANN_FACTOR == 365.0
    assert core_data.FREQ_INTERVALS == {
        "5m": "5 minutes", "30m": "30 minutes", "1h": "1 hour",
        "4h": "4 hours", "8h": "8 hours", "1d": "1 day",
    }
    assert core_data.BARS_PER_DAY == {
        "5m": 288.0, "30m": 48.0, "1h": 24.0, "4h": 6.0, "8h": 3.0, "1d": 1.0,
    }


def test_default_db_resolves_to_shared_data_dir():
    """DEFAULT_DB must still point at <repo>/../data/market.duckdb (unchanged by the move)."""
    expected = (REPO_ROOT / ".." / "data" / "market.duckdb").resolve()
    assert Path(core_data.DEFAULT_DB).resolve() == expected


def test_load_bars_rejects_unsupported_freq():
    with pytest.raises(ValueError, match="Unsupported freq"):
        core_data.load_bars("3m")


def test_compute_btc_benchmark_normalizes_to_one():
    panel = pd.DataFrame({
        "symbol": ["BTC-USD"] * 3 + ["ETH-USD"] * 3,
        "ts": list(pd.date_range("2022-01-01", periods=3)) * 2,
        "close": [100.0, 110.0, 120.0, 5.0, 6.0, 7.0],
    })
    eq = core_data.compute_btc_benchmark(panel)
    assert eq.name == "btc_equity"
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[1] == pytest.approx(1.1)
    assert eq.iloc[2] == pytest.approx(1.2)


def test_filter_universe_marks_membership():
    # 5 days, lenient thresholds so membership is deterministic after warmup.
    panel = pd.DataFrame({
        "symbol": ["AAA-USD"] * 5,
        "ts": pd.date_range("2022-01-01", periods=5),
        "close": [10.0, 10.0, 10.0, 10.0, 10.0],
        "volume": [1e6, 1e6, 1e6, 1e6, 1e6],
        "open": [10.0] * 5, "high": [10.0] * 5, "low": [10.0] * 5,
    })
    out = core_data.filter_universe(panel, min_adv_usd=0.0, min_history_days=3, adv_window=2)
    assert "in_universe" in out.columns
    assert out["in_universe"].dtype == bool
    # First two rows lack warmup (history < 3); later rows qualify.
    flags = out.sort_values("ts")["in_universe"].tolist()
    assert flags[:2] == [False, False]
    assert flags[-1] is True or flags[-1] == np.True_


def _load_shim():
    spec = importlib.util.spec_from_file_location("_research_common_data_shim", SHIM_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shim_reexports_same_objects():
    shim = _load_shim()
    assert shim.load_daily_bars is core_data.load_daily_bars
    assert shim.load_bars is core_data.load_bars
    assert shim.filter_universe is core_data.filter_universe
    assert shim.compute_btc_benchmark is core_data.compute_btc_benchmark
    assert shim.DEFAULT_DB == core_data.DEFAULT_DB
    assert shim.ANN_FACTOR == core_data.ANN_FACTOR
    for name in shim.__all__:
        assert hasattr(shim, name)
