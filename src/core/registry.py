"""Machine-readable alpha registry: schema, loader, and validator.

The registry is the single source of truth for every alpha. Each entry is one
YAML file under ``registry/alphas/<registry_id>.yaml`` describing the
pre-registered hypothesis, the executable signal, the universe/costs, and the
lifecycle stage/status. The reorganization plan
(docs/RESEARCH_PIPELINE_REORGANIZATION.md) explains why this converges the
previously split sources (data/alpha_registry/*.xlsx, the markdown hypotheses,
and the in-code Hypothesis/Candidate objects) into one schema.

Two things this buys us:
  * Routing for free — ``payoff_shape``/``track`` decide which pipeline runs.
  * Pre-registration as a code check — a candidate cannot be promoted past S0
    until its hypothesis/rationale/falsification fields are filled in, so the
    "rewrite-after-backtest -> back to S0" rule is enforced, not honor-system.

Enum string values intentionally match ``src/convexity_pipeline/types.py`` so
an :class:`AlphaSpec` can be converted straight into that pipeline's
``Hypothesis`` via :meth:`AlphaSpec.to_hypothesis`. (A future cleanup can have
the pipeline import these enums from core rather than redefining them.)
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Repo-root-relative default location of the registry YAML files.
# parents[2] of src/core/registry.py == repo root.
DEFAULT_REGISTRY_DIR = Path(__file__).resolve().parents[2] / "registry" / "alphas"

_REGISTRY_ID_RE = re.compile(r"^\d{4}-\d{2}-[a-z0-9][a-z0-9-]*$")
_DOTTED_PATH_RE = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)+$")


class PayoffShape(str, Enum):
    """Expected payoff shape. Drives pipeline routing."""
    CONVEX = "convex"
    LINEAR = "linear"
    CONCAVE = "concave"
    AMBIGUOUS = "ambiguous"
    NA = "N_A"


class Track(str, Enum):
    """Which sub-pipeline / track this candidate runs in."""
    TREND = "trend"
    VOL_EXPANSION = "vol_expansion"
    BOTH = "both"
    CROSS_SECTIONAL = "cross_sectional"
    NA = "N_A"


class Stage(str, Enum):
    """Pipeline stage. S0..S4 are in scope for the live pipeline; the rest are status-only."""
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"
    LIVE = "Live"
    RETIRED = "Retired"
    KILLED = "Killed"


class Status(str, Enum):
    """Lifecycle status, orthogonal to stage."""
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    KILLED = "killed"
    LIVE = "live"
    RETIRED = "retired"


class PreRegMetric(BaseModel):
    """A single pre-registered expected metric."""
    model_config = ConfigDict(extra="forbid")
    expected: float
    confidence: Optional[str] = None  # e.g. "low" | "medium" | "high"


class Validation(BaseModel):
    """Realized, out-of-sample backtest results + provenance (filled AFTER testing).

    Distinct from pre_registered_metrics (the EXPECTED values set before backtest):
    this records what was actually measured, how, and the caveats — the honest
    scorecard. Per the mandate, every promoted result carries its provenance.
    """
    model_config = ConfigDict(extra="forbid")
    method: str                                   # e.g. "param-frozen walk-forward, point-in-time universe"
    data_range: str                               # e.g. "2021-01..2026-06; OOS 2023-2026"
    universe: str = ""
    costs_bps: Optional[float] = None
    oos_sortino: Optional[float] = None
    oos_sharpe: Optional[float] = None
    oos_cagr: Optional[float] = None
    oos_max_dd: Optional[float] = None
    benchmark: Optional[str] = None
    benchmark_oos_sortino: Optional[float] = None
    caveats: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)  # scripts, rule ids, git commit


class AlphaSpec(BaseModel):
    """Validated schema for one registry entry (registry/alphas/<id>.yaml)."""
    model_config = ConfigDict(extra="forbid")

    # --- identity ---
    registry_id: str
    name: str
    researcher: str = ""
    registered: str = ""            # ISO date
    source: str = ""

    # --- routing ---
    payoff_shape: PayoffShape
    track: Track = Track.NA
    horizon_bars: int = Field(gt=0)

    # --- pre-registration (gates promotion past S0) ---
    hypothesis: str = ""
    rationale: str = ""
    falsification: list[str] = Field(default_factory=list)

    # --- implementation ---
    signal_fn: str                  # dotted path into src/signals (or any importable module)
    signal_params: dict[str, Any] = Field(default_factory=dict)
    universe: str | list[str]
    bar_frequency: str = "1d"
    cost_profile: str = "crypto_default"

    # --- pre-registered expectations ---
    pre_registered_metrics: dict[str, PreRegMetric] = Field(default_factory=dict)

    # --- realized validation (filled after honest OOS testing) ---
    validation: Optional[Validation] = None

    # --- lifecycle ---
    stage: Stage = Stage.S0
    status: Status = Status.QUEUED

    @field_validator("registry_id")
    @classmethod
    def _check_registry_id(cls, v: str) -> str:
        if not _REGISTRY_ID_RE.match(v):
            raise ValueError(
                f"registry_id {v!r} must be date-prefixed kebab-case, e.g. '2026-06-continuation-index'"
            )
        return v

    @field_validator("signal_fn")
    @classmethod
    def _check_signal_fn(cls, v: str) -> str:
        if not _DOTTED_PATH_RE.match(v):
            raise ValueError(f"signal_fn {v!r} must be a dotted import path, e.g. 'signals.trend.ewmac'")
        return v

    @model_validator(mode="after")
    def _check_convex_has_track(self) -> "AlphaSpec":
        if self.payoff_shape == PayoffShape.CONVEX and self.track in (Track.NA,):
            raise ValueError(
                "a convex payoff_shape must declare a track (trend / vol_expansion / both)"
            )
        return self

    # ------------------------------------------------------------------
    # Behavior
    # ------------------------------------------------------------------
    def is_preregistration_complete(self) -> bool:
        """True iff the fields required to promote past S0 are filled in."""
        return bool(self.hypothesis.strip() and self.rationale.strip() and self.falsification)

    def require_preregistration(self) -> None:
        """Raise if pre-registration is incomplete (used as a promotion gate)."""
        if not self.is_preregistration_complete():
            missing = [
                f for f, ok in (
                    ("hypothesis", bool(self.hypothesis.strip())),
                    ("rationale", bool(self.rationale.strip())),
                    ("falsification", bool(self.falsification)),
                ) if not ok
            ]
            raise ValueError(
                f"{self.registry_id}: cannot run/promote — incomplete pre-registration: {missing}. "
                "Fill in the hypothesis before backtesting (rewriting after results returns to S0)."
            )

    def to_hypothesis(self) -> Any:
        """Convert to a convexity_pipeline Hypothesis (lazy import to avoid a hard dep)."""
        from convexity_pipeline.types import Hypothesis
        from convexity_pipeline.types import PayoffShape as CPayoff
        from convexity_pipeline.types import Track as CTrack

        universe = [self.universe] if isinstance(self.universe, str) else list(self.universe)
        try:
            convexity_track = CTrack(self.track.value)
        except ValueError as exc:
            raise ValueError(
                f"{self.registry_id}: track {self.track.value!r} has no convexity-pipeline "
                "equivalent (e.g. cross_sectional alphas do not convert to a convexity "
                "Hypothesis) — route it to its own pipeline instead."
            ) from exc
        return Hypothesis(
            name=self.name,
            statement=self.hypothesis,
            rationale=self.rationale,
            expected_payoff_shape=CPayoff(self.payoff_shape.value),
            convexity_track=convexity_track,
            horizon_bars=self.horizon_bars,
            universe=universe,
            bar_frequency=self.bar_frequency,
            params=dict(self.signal_params),
            expected_metrics={k: (v.expected, 0.0) for k, v in self.pre_registered_metrics.items()},
            falsification_criteria=list(self.falsification),
            researcher=self.researcher,
            registration_date=self.registered,
            source_reference=self.source,
        )


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def load_alpha_spec(path: str | Path) -> AlphaSpec:
    """Load and validate a single registry YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: registry entry must be a YAML mapping, got {type(raw).__name__}")
    spec = AlphaSpec.model_validate(raw)
    if spec.registry_id != path.stem:
        raise ValueError(
            f"{path}: registry_id {spec.registry_id!r} must match filename stem {path.stem!r}"
        )
    return spec


def load_registry(registry_dir: str | Path | None = None) -> dict[str, AlphaSpec]:
    """Load and validate every ``*.yaml`` in the registry directory.

    Returns a dict keyed by registry_id. Raises on the first invalid entry.
    """
    registry_dir = Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
    specs: dict[str, AlphaSpec] = {}
    for path in sorted(registry_dir.glob("*.yaml")):
        spec = load_alpha_spec(path)
        specs[spec.registry_id] = spec
    return specs
