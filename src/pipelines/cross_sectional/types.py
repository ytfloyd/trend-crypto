"""Data types for the alpha pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

# Shared stage-result contracts now live in pipelines.common; re-exported here so
# `cross_sectional.types.StageResult/StageVerdict` keeps working.
from ..common.types import StageResult, StageVerdict


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for each pipeline gate.  All can be overridden per-run."""

    # Stage 1 — IC screening
    min_abs_tstat: float = 2.0
    min_abs_mean_ic: float = 0.005
    min_ic_days: int = 200

    # Stage 2 — IC decay
    max_decay_horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    require_monotonic_decay: bool = True

    # Stage 3 — redundancy (with optional orthogonalization)
    max_correlation_with_existing: float = 0.70
    orthogonalize_redundancy: bool = True
    residual_min_abs_tstat: float = 1.5

    # Stage 3.5 — turnover / cost
    turnover_cost_bps: float = 20.0
    min_net_ic: float = 0.002
    turnover_lookback: int = 20

    # Stage 4 — walk-forward (CPCV)
    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    cpcv_pct_embargo: float = 0.01
    max_pbo: float = 0.50
    invvol_weight: bool = True
    vol_lookback: int = 20

    # Stage 5 — deflated Sharpe
    min_deflated_sharpe_pval: float = 0.95
    n_trials_for_deflation: int | None = None  # auto-set from candidate count


# Callable that takes (close_wide, volume_wide, returns_wide) panel DataFrames
# and returns a (symbol x ts) DataFrame of alpha scores.
AlphaFn = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]


@dataclass
class AlphaCandidate:
    """A candidate alpha signal to evaluate."""

    name: str
    family: str  # e.g. "momentum", "mean_reversion", "volatility"
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    # The signal itself: either a pre-computed DataFrame or a callable
    scores: pd.DataFrame | None = None   # (ts, symbol) → score
    compute_fn: AlphaFn | None = None

    def get_scores(
        self,
        close_wide: pd.DataFrame,
        volume_wide: pd.DataFrame,
        returns_wide: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the alpha score panel, computing lazily if needed."""
        if self.scores is not None:
            return self.scores
        if self.compute_fn is not None:
            self.scores = self.compute_fn(close_wide, volume_wide, returns_wide)
            return self.scores
        raise ValueError(f"Alpha '{self.name}' has no scores or compute_fn")


@dataclass
class PipelineReport:
    """Full pipeline results for one alpha candidate."""

    candidate: AlphaCandidate
    stages: list[StageResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.verdict == StageVerdict.PASS for s in self.stages)

    @property
    def final_verdict(self) -> StageVerdict:
        for s in self.stages:
            if s.verdict == StageVerdict.FAIL:
                return StageVerdict.FAIL
        return StageVerdict.PASS

    @property
    def failed_stage(self) -> str | None:
        for s in self.stages:
            if s.verdict == StageVerdict.FAIL:
                return s.stage
        return None

    def summary_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "alpha": self.candidate.name,
            "family": self.candidate.family,
            "verdict": self.final_verdict.value,
            "failed_stage": self.failed_stage,
        }
        for s in self.stages:
            for k, v in s.metrics.items():
                d[f"{s.stage}__{k}"] = v
        return d
