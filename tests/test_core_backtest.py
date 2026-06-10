"""Characterization tests for the unified ``core.backtest`` / ``core.risk_overlays`` /
``core.cost_analysis`` modules.

Locks ``simple_backtest``'s numeric output (Model-B: signal at close t, held from
t+1) on a fixed weight/return panel, and verifies the deprecated
``scripts/research/common/{backtest,risk_overlays,cost_analysis}.py`` shims
re-export the implementations verbatim.

See docs/RESEARCH_PIPELINE_REORGANIZATION.md (Phase 1: unify the core stack).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from core import backtest as core_backtest
from core import cost_analysis as core_cost
from core import risk_overlays as core_risk

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON = REPO_ROOT / "scripts" / "research" / "common"


def _fixed_panel():
    ts = pd.date_range("2022-01-01", periods=4)
    w = pd.DataFrame({"AAA": [1.0, 1.0, 0.0, 0.5], "BBB": [0.0, 0.5, 0.5, 0.5]}, index=ts)
    r = pd.DataFrame({"AAA": [0.01, -0.02, 0.03, 0.00], "BBB": [0.00, 0.04, -0.01, 0.02]}, index=ts)
    return w, r


# Golden master at the time of the verbatim move (cost_bps=20, execution_lag=1).
_GOLDEN = {
    "portfolio_ret": [0.0, -0.022, 0.024, 0.008],
    "portfolio_equity": [1.0, 0.978, 1.001472, 1.009483776],
    "gross_exposure": [0.0, 1.0, 1.5, 0.5],
    "turnover": [0.0, 1.0, 0.5, 1.0],
    "cost_ret": [0.0, 0.002, 0.001, 0.002],
}


def test_default_cost_bps():
    assert core_backtest.DEFAULT_COST_BPS == 20.0


def test_simple_backtest_golden_master():
    w, r = _fixed_panel()
    out = core_backtest.simple_backtest(w, r, cost_bps=20.0)
    assert list(out.columns) == [
        "ts", "portfolio_ret", "portfolio_equity",
        "gross_exposure", "turnover", "cost_ret",
    ]
    for col, expected in _GOLDEN.items():
        got = [float(x) for x in out[col].tolist()]
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12), f"column '{col}' drifted"


def test_execution_lag_shifts_holding():
    """With execution_lag=1 the first bar is flat (nothing held yet)."""
    w, r = _fixed_panel()
    out = core_backtest.simple_backtest(w, r, cost_bps=0.0)
    assert out["portfolio_ret"].iloc[0] == pytest.approx(0.0)
    assert out["gross_exposure"].iloc[0] == pytest.approx(0.0)


def _load_shim(name: str):
    spec = importlib.util.spec_from_file_location(f"_research_common_{name}_shim", COMMON / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_backtest_shim_reexports_same_objects():
    shim = _load_shim("backtest")
    assert shim.simple_backtest is core_backtest.simple_backtest
    assert shim.DEFAULT_COST_BPS == core_backtest.DEFAULT_COST_BPS


def test_risk_overlays_shim_reexports_same_objects():
    shim = _load_shim("risk_overlays")
    for name in shim.__all__:
        assert getattr(shim, name) is getattr(core_risk, name)


def test_cost_analysis_shim_reexports_same_objects():
    shim = _load_shim("cost_analysis")
    for name in shim.__all__:
        assert getattr(shim, name) is getattr(core_cost, name)
