"""Tests for signals.cross_sectional.medallion_lite + the runner's L/S execute path."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import AlphaSpec, load_alpha_spec
from research import cli, runner
from signals.cross_sectional import medallion_lite


def _panel(n_syms: int = 12, n_bars: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n_bars, freq="D")
    frames = []
    for i in range(n_syms):
        steps = rng.normal(0.001 * (i - n_syms / 2), 0.02, n_bars)
        close = 100.0 * np.exp(np.cumsum(steps))
        frames.append(pd.DataFrame({
            "symbol": f"S{i:02d}-USD", "ts": ts, "close": close,
            "high": close * (1.0 + np.abs(rng.normal(0, 0.005, n_bars))),
            "volume": rng.uniform(1e6, 5e6, n_bars),
        }))
    return pd.concat(frames, ignore_index=True)


def _spec(**overrides) -> AlphaSpec:
    base = dict(
        registry_id="2026-06-test-xs", name="t", payoff_shape="linear",
        track="cross_sectional", horizon_bars=5,
        signal_fn="signals.cross_sectional.medallion_lite", signal_params={"lookback": 7},
        universe=["x"], hypothesis="h", rationale="r", falsification=["f"],
    )
    base.update(overrides)
    return AlphaSpec.model_validate(base)


# --- the signal ---
def test_medallion_lite_composite_scores():
    scores = medallion_lite(_panel(), lookback=7)
    assert scores.shape[1] == 12
    valid = scores.to_numpy()[~np.isnan(scores.to_numpy())]
    assert valid.min() >= 0.0 and valid.max() <= 1.0  # composite percentile in [0,1]


def test_medallion_lite_lookback_must_exceed_one():
    with pytest.raises(ValueError, match="lookback must be > 1"):
        medallion_lite(_panel(), lookback=1)


# --- cross-sectional execution ---
def test_execute_cross_sectional_end_to_end():
    resolved = runner.resolve_run(_spec())
    assert resolved.route == "cross_sectional"
    result = runner.execute_cross_sectional(resolved, _panel())
    for key in ("sharpe", "cagr", "max_dd", "total_return"):
        assert key in result["metrics"]
    assert len(result["equity"]) > 0
    assert list(result["backtest"].columns) == ["ts", "portfolio_ret", "portfolio_equity"]


def test_execute_dispatches_cross_sectional():
    resolved = runner.resolve_run(_spec())
    # execute() routes cross_sectional to the L/S path (no ValueError, unlike execute_screen)
    result = runner.execute(resolved, _panel())
    assert "sharpe" in result["metrics"]


def test_execute_cross_sectional_needs_breadth():
    resolved = runner.resolve_run(_spec())
    with pytest.raises(ValueError, match=">= 10 symbols"):
        runner.execute_cross_sectional(resolved, _panel(n_syms=5))


def test_execute_screen_still_refuses_cross_sectional():
    resolved = runner.resolve_run(_spec())
    with pytest.raises(ValueError, match="does not support the 'cross_sectional'"):
        runner.execute_screen(resolved, _panel())


# --- registry entry ---
def test_medallion_registry_entry_runnable():
    spec = load_alpha_spec(cli.DEFAULT_REGISTRY_DIR / "2026-06-medallion-lite.yaml")
    resolved = runner.resolve_run(spec)
    assert resolved.route == "cross_sectional"
    assert resolved.is_runnable
    assert resolved.signal_reason == "ok"
