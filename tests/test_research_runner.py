"""Tests for the registry-driven research runner + CLI (`python -m research`).

See src/research/runner.py, src/research/cli.py, registry/README.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.registry import AlphaSpec, Stage, load_alpha_spec
from research import cli, runner


def _spec(**overrides) -> AlphaSpec:
    base = dict(
        registry_id="2026-06-test-alpha",
        name="Test Alpha",
        payoff_shape="linear",
        track="cross_sectional",
        horizon_bars=5,
        signal_fn="signals.test.foo",
        universe="midcap_usd",
        hypothesis="X predicts Y.",
        rationale="mechanism.",
        falsification=["IC <= 0"],
    )
    base.update(overrides)
    return AlphaSpec.model_validate(base)


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------
def test_route_cross_sectional():
    assert runner.route_for(_spec(payoff_shape="linear", track="cross_sectional")) == "cross_sectional"


def test_route_convexity():
    assert runner.route_for(_spec(payoff_shape="convex", track="trend")) == "convexity"


def test_route_time_series_default():
    assert runner.route_for(_spec(payoff_shape="ambiguous", track="N_A")) == "time_series"


def test_pipeline_modules_cover_all_routes():
    for route in ("cross_sectional", "convexity", "time_series"):
        assert route in runner.PIPELINE_MODULES


# ----------------------------------------------------------------------
# Signal resolution
# ----------------------------------------------------------------------
def test_resolve_signal_fn_importable():
    fn, reason = runner.resolve_signal_fn(_spec(signal_fn="math.sqrt"))
    assert callable(fn) and reason == "ok"


def test_resolve_signal_fn_missing_module():
    fn, reason = runner.resolve_signal_fn(_spec(signal_fn="signals.test.foo"))
    assert fn is None and "not importable" in reason


def test_resolve_signal_fn_missing_attr():
    fn, reason = runner.resolve_signal_fn(_spec(signal_fn="math.no_such_attr"))
    assert fn is None and "no attribute" in reason


# ----------------------------------------------------------------------
# resolve_run on the real backfilled registry entries
# ----------------------------------------------------------------------
def test_resolve_run_on_examples():
    specs = {s.registry_id: s for s in cli.load_registry(None).values()}
    # continuation-index has a built signal (signals.tasc) -> runnable, convexity route
    ci = runner.resolve_run(specs["2026-06-continuation-index"])
    assert ci.route == "convexity"
    assert ci.pipeline_module == "pipelines.convexity"
    assert ci.is_runnable
    # medallion-lite's signal (signals.cross_sectional.*) isn't built yet ->
    # not runnable, blocker reported (not faked)
    ml = runner.resolve_run(specs["2026-06-medallion-lite"])
    assert ml.route == "cross_sectional"
    assert not ml.is_runnable
    assert any("not importable" in b for b in ml.blockers)


# ----------------------------------------------------------------------
# Stage progression + promotion gate
# ----------------------------------------------------------------------
def test_next_stage():
    assert runner.next_stage(Stage.S1) == Stage.S2
    assert runner.next_stage(Stage.LIVE) is None


def test_promote_advances_stage():
    spec = _spec(stage="S1")
    assert runner.promote(spec) == Stage.S2
    assert spec.stage == Stage.S2


def test_promote_blocked_by_incomplete_preregistration():
    spec = _spec(stage="S1", hypothesis="", rationale="", falsification=[])
    with pytest.raises(ValueError, match="incomplete pre-registration"):
        runner.promote(spec)


def test_write_spec_roundtrip_persists_stage(tmp_path):
    spec = _spec(stage="S1")
    runner.promote(spec)  # -> S2
    path = tmp_path / f"{spec.registry_id}.yaml"
    runner.write_spec(spec, path)
    reloaded = load_alpha_spec(path)
    assert reloaded.stage == Stage.S2
    assert reloaded.registry_id == spec.registry_id


# ----------------------------------------------------------------------
# CLI exit codes
# ----------------------------------------------------------------------
def test_cli_list_and_validate_ok():
    assert cli.main(["list"]) == 0
    assert cli.main(["validate"]) == 0


def test_cli_run_reports_blocker_exit_1():
    # medallion-lite's signal isn't built yet -> blocker -> exit 1
    assert cli.main(["run", "2026-06-medallion-lite"]) == 1


def test_cli_promote_roundtrip(tmp_path):
    spec = _spec(stage="S1")
    path = tmp_path / f"{spec.registry_id}.yaml"
    runner.write_spec(spec, path)
    rc = cli.main(["--registry-dir", str(tmp_path), "promote", spec.registry_id])
    assert rc == 0
    assert load_alpha_spec(path).stage == Stage.S2


# ----------------------------------------------------------------------
# End-to-end execution (registry -> signal -> backtest -> metrics)
# ----------------------------------------------------------------------
def _synthetic_bars(n: int = 40) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", periods=n, freq="D")
    up = pd.DataFrame({"symbol": "AAA-USD", "ts": ts, "close": np.linspace(100, 220, n)})
    osc = pd.DataFrame(
        {"symbol": "BBB-USD", "ts": ts, "close": 100 + 10 * np.sin(np.arange(n) / 3.0)}
    )
    return pd.concat([up, osc], ignore_index=True)


def test_ma_5_40_entry_now_resolves_runnable():
    # signals.trend.ma_crossover now exists -> the real registry entry resolves.
    spec = load_alpha_spec(cli.DEFAULT_REGISTRY_DIR / "2026-06-ma-5-40-trend.yaml")
    resolved = runner.resolve_run(spec)
    assert resolved.is_runnable
    assert resolved.signal_reason == "ok"


def _trend_spec(**overrides):
    base = dict(
        payoff_shape="convex", track="trend",
        signal_fn="signals.trend.ma_crossover",
        signal_params={"fast": 3, "slow": 10},
        universe=["AAA-USD", "BBB-USD"],
    )
    base.update(overrides)
    return _spec(**base)


def test_execute_screen_end_to_end():
    resolved = runner.resolve_run(_trend_spec())
    assert resolved.is_runnable  # signal_fn imports

    result = runner.execute_screen(resolved, _synthetic_bars(40))
    metrics = result["metrics"]
    for key in ("sharpe", "cagr", "max_dd", "total_return"):
        assert key in metrics
    # warm-up (slow-1 = 9 bars) is dropped, not booked flat -> fewer than 40 bars
    assert 0 < len(result["equity"]) < 40
    assert list(result["backtest"].columns) == [
        "ts", "portfolio_ret", "portfolio_equity", "gross_exposure", "turnover", "cost_ret",
    ]


def test_execute_screen_tolerates_duplicate_bars():
    bars = pd.concat([_synthetic_bars(40), _synthetic_bars(40).iloc[:5]], ignore_index=True)
    resolved = runner.resolve_run(_trend_spec())
    result = runner.execute_screen(resolved, bars)  # must not raise on duplicate (ts,symbol)
    assert len(result["equity"]) > 0


def test_execute_screen_refuses_cross_sectional_route():
    spec = _spec(
        payoff_shape="linear", track="cross_sectional",
        signal_fn="signals.trend.ma_crossover",
        signal_params={"fast": 3, "slow": 10},
        universe=["AAA-USD", "BBB-USD"],
    )
    resolved = runner.resolve_run(spec)
    assert resolved.route == "cross_sectional"
    with pytest.raises(ValueError, match="does not support the 'cross_sectional'"):
        runner.execute_screen(resolved, _synthetic_bars(40))


def test_execute_screen_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    resolved = runner.resolve_run(_trend_spec())
    result = runner.execute_screen(resolved, _synthetic_bars(40))
    out = runner.write_results(spec_registry_id := resolved.spec.registry_id, result)
    assert spec_registry_id  # used
    assert (out / "metrics.json").exists()
    assert (out / "equity.csv").exists()


def test_execute_screen_refuses_unrunnable():
    spec = _spec(signal_fn="signals.missing.nope")
    resolved = runner.resolve_run(spec)
    with pytest.raises(ValueError, match="not runnable"):
        runner.execute_screen(resolved, _synthetic_bars(10))


def test_to_hypothesis_raises_for_cross_sectional_track():
    spec = _spec(payoff_shape="linear", track="cross_sectional")
    with pytest.raises(ValueError, match="no convexity-pipeline equivalent"):
        spec.to_hypothesis()


def test_cli_run_execute_data_unavailable_exits_2():
    # ma-5-40 is runnable (signal_fn imports), but the market lake is absent in this
    # env -> data-unavailable is exit 2, distinct from a blocked alpha (exit 1).
    assert cli.main(["run", "2026-06-ma-5-40-trend", "--execute"]) == 2
