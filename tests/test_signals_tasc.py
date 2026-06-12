"""Tests for signals.tasc.continuation_index (pure signal) + end-to-end run."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import AlphaSpec, load_alpha_spec
from research import cli, runner
from signals.tasc import continuation_index


def _bars(n: int = 60) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", periods=n, freq="D")
    up = pd.DataFrame({"symbol": "AAA-USD", "ts": ts, "close": np.linspace(100, 260, n)})
    osc = pd.DataFrame(
        {"symbol": "BBB-USD", "ts": ts, "close": 100 + 8 * np.sin(np.arange(n) / 4.0)}
    )
    return pd.concat([up, osc], ignore_index=True)


def test_shape_columns_and_warmup_is_nan():
    w = continuation_index(_bars(60), gamma=0.8, length=10)
    assert list(w.columns) == ["AAA-USD", "BBB-USD"]
    assert w.index.is_monotonic_increasing
    # leading warm-up (before the smoothed slope is defined) is NaN, not flat 0.0
    assert w.iloc[0].isna().all()


def test_long_only_equal_weight_values():
    w = continuation_index(_bars(60), gamma=0.8, length=10)
    post = w.values[~np.isnan(w.values)]
    assert set(np.unique(post)) <= {0.0, 0.5}  # 2-symbol universe -> 1/2 when long
    # the steadily-rising symbol is long by the end
    assert w["AAA-USD"].iloc[-1] == pytest.approx(0.5)


def test_length_must_exceed_one():
    with pytest.raises(ValueError, match="length must be > 1"):
        continuation_index(_bars(30), gamma=0.8, length=1)


def test_registry_continuation_index_now_runnable():
    spec = load_alpha_spec(cli.DEFAULT_REGISTRY_DIR / "2026-06-continuation-index.yaml")
    resolved = runner.resolve_run(spec)
    assert resolved.route == "convexity"
    assert resolved.is_runnable  # signals.tasc.continuation_index now imports
    assert resolved.signal_reason == "ok"


def test_execute_screen_end_to_end():
    spec = AlphaSpec.model_validate(dict(
        registry_id="2026-06-test-ci", name="t", payoff_shape="convex", track="trend",
        horizon_bars=20, signal_fn="signals.tasc.continuation_index",
        signal_params={"gamma": 0.8, "length": 10}, universe=["AAA-USD", "BBB-USD"],
        hypothesis="x", rationale="y", falsification=["z"],
    ))
    resolved = runner.resolve_run(spec)
    result = runner.execute_screen(resolved, _bars(60))
    assert "sharpe" in result["metrics"]
    assert 0 < len(result["equity"]) < 60  # warm-up dropped
