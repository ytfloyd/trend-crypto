"""Tests for the registry-driven research runner + CLI (`python -m research`).

See src/research/runner.py, src/research/cli.py, registry/README.md.
"""
from __future__ import annotations

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
    r = runner.resolve_run(specs["2026-06-continuation-index"])
    assert r.route == "convexity"
    assert r.pipeline_module == "pipelines.convexity"
    # signals.* package not built yet -> not runnable, blocker reported (not faked)
    assert not r.is_runnable
    assert any("not importable" in b for b in r.blockers)


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
    # real entry; signals.* not built -> blocker -> exit 1
    assert cli.main(["run", "2026-06-continuation-index"]) == 1


def test_cli_promote_roundtrip(tmp_path):
    spec = _spec(stage="S1")
    path = tmp_path / f"{spec.registry_id}.yaml"
    runner.write_spec(spec, path)
    rc = cli.main(["--registry-dir", str(tmp_path), "promote", spec.registry_id])
    assert rc == 0
    assert load_alpha_spec(path).stage == Stage.S2
