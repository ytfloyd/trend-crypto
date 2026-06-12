"""Tests for the alpha registry schema, loader, and validation rules.

See docs/RESEARCH_PIPELINE_REORGANIZATION.md (the registry is the single source
of truth for alphas) and registry/README.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.registry import (
    DEFAULT_REGISTRY_DIR,
    AlphaSpec,
    PayoffShape,
    Stage,
    Track,
    Validation,
    load_alpha_spec,
    load_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _minimal(**overrides):
    base = dict(
        registry_id="2026-06-test-alpha",
        name="Test Alpha",
        payoff_shape="linear",
        track="cross_sectional",
        horizon_bars=5,
        signal_fn="signals.test.foo",
        universe="midcap_usd",
    )
    base.update(overrides)
    return base


# ----------------------------------------------------------------------
# Backfilled registry entries
# ----------------------------------------------------------------------
def test_load_registry_loads_all_examples():
    specs = load_registry()
    # the three backfilled reference alphas are present and keyed by id
    for rid in ("2026-06-continuation-index", "2026-06-ma-5-40-trend", "2026-06-medallion-lite"):
        assert rid in specs, f"missing backfilled alpha {rid}"
        assert specs[rid].registry_id == rid


def test_default_registry_dir_points_into_repo():
    assert DEFAULT_REGISTRY_DIR == REPO_ROOT / "registry" / "alphas"


def test_examples_are_preregistered_and_route():
    specs = load_registry()
    ci = specs["2026-06-continuation-index"]
    assert ci.payoff_shape == PayoffShape.CONVEX
    assert ci.track == Track.TREND
    assert ci.stage == Stage.S1
    assert ci.is_preregistration_complete()

    ml = specs["2026-06-medallion-lite"]
    assert ml.payoff_shape == PayoffShape.LINEAR
    assert ml.track == Track.CROSS_SECTIONAL


# ----------------------------------------------------------------------
# Validation rules
# ----------------------------------------------------------------------
def test_bad_registry_id_rejected():
    with pytest.raises(ValueError, match="date-prefixed"):
        AlphaSpec.model_validate(_minimal(registry_id="not_a_valid_id"))


def test_bad_signal_fn_rejected():
    with pytest.raises(ValueError, match="dotted import path"):
        AlphaSpec.model_validate(_minimal(signal_fn="not-a-path"))


def test_convex_requires_track():
    with pytest.raises(ValueError, match="must declare a track"):
        AlphaSpec.model_validate(_minimal(payoff_shape="convex", track="N_A"))


def test_non_positive_horizon_rejected():
    with pytest.raises(ValueError):
        AlphaSpec.model_validate(_minimal(horizon_bars=0))


def test_extra_fields_forbidden():
    with pytest.raises(ValueError):
        AlphaSpec.model_validate(_minimal(unknown_field="x"))


# ----------------------------------------------------------------------
# Pre-registration gate
# ----------------------------------------------------------------------
def test_require_preregistration_raises_when_incomplete():
    spec = AlphaSpec.model_validate(_minimal())  # no hypothesis/rationale/falsification
    assert not spec.is_preregistration_complete()
    with pytest.raises(ValueError, match="incomplete pre-registration"):
        spec.require_preregistration()


def test_require_preregistration_passes_when_complete():
    spec = AlphaSpec.model_validate(_minimal(
        hypothesis="X predicts Y over Z bars.",
        rationale="because mechanism.",
        falsification=["IC <= 0"],
    ))
    assert spec.is_preregistration_complete()
    spec.require_preregistration()  # no raise


# ----------------------------------------------------------------------
# Loader filename-stem rule + Hypothesis conversion
# ----------------------------------------------------------------------
def test_filename_stem_must_match_registry_id(tmp_path):
    p = tmp_path / "wrong-name.yaml"
    p.write_text(
        "registry_id: 2026-06-mismatch\nname: M\npayoff_shape: linear\n"
        "track: cross_sectional\nhorizon_bars: 3\nsignal_fn: signals.a.b\nuniverse: u\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must match filename stem"):
        load_alpha_spec(p)


def test_to_hypothesis_roundtrips_core_fields():
    spec = load_registry()["2026-06-continuation-index"]
    h = spec.to_hypothesis()
    assert h.name == spec.name
    assert h.statement == spec.hypothesis
    assert h.expected_payoff_shape.value == spec.payoff_shape.value
    assert h.horizon_bars == spec.horizon_bars
    assert h.universe == list(spec.universe)
    assert h.falsification_criteria == list(spec.falsification)


# ----------------------------------------------------------------------
# Validation block (realized OOS results + provenance)
# ----------------------------------------------------------------------
def test_validation_is_optional():
    spec = AlphaSpec.model_validate(_minimal())  # no validation block
    assert spec.validation is None


def test_medallion_validation_block_loads():
    ml = load_registry()["2026-06-medallion-lite"]
    assert isinstance(ml.validation, Validation)
    assert ml.validation.oos_sortino == 2.03
    assert ml.validation.benchmark_oos_sortino == 1.78
    assert ml.validation.costs_bps == 30.0
    assert ml.validation.method and ml.validation.caveats and ml.validation.provenance
