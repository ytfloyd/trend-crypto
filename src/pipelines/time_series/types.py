"""Data types for the time-series alpha pipeline.

Parallel to ``src/alpha_pipeline/types.py`` but designed for
per-asset directional (trend-following) signal evaluation rather
than cross-sectional ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from ..cross_sectional.types import StageResult, StageVerdict  # reuse


# ── Signal callable signature ─────────────────────────────────────────
# Takes (close, high, low, volume, returns) wide DataFrames and returns
# a wide DataFrame of *per-asset* forecasts (not cross-sectional ranks).
# Positive = long, negative = short, zero = flat.
TSAlphaFn = Callable[
    [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    pd.DataFrame,
]


@dataclass
class TSCandidate:
    """A candidate time-series (trend) signal to evaluate."""

    name: str
    family: str  # e.g. "ewmac", "breakout", "tsmom", "composite"
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    scores: pd.DataFrame | None = None
    compute_fn: TSAlphaFn | None = None

    def get_scores(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        volume: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the signal panel (ts x symbol), computing lazily if needed."""
        if self.scores is not None:
            return self.scores
        if self.compute_fn is not None:
            self.scores = self.compute_fn(close, high, low, volume, returns)
            return self.scores
        raise ValueError(f"TSCandidate '{self.name}' has no scores or compute_fn")


@dataclass(frozen=True)
class TSGateConfig:
    """Thresholds for each pipeline gate.  All can be overridden per-run."""

    # Stage 1 — per-asset time-series IC
    min_abs_tstat: float = 2.0
    min_abs_median_ic: float = 0.01
    min_ic_assets: int = 10
    ic_horizon: int = 1

    # Stage 2 — signal persistence
    min_median_autocorr: float = 0.80

    # Stage 3 — IC horizon profile
    ic_horizons: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 60])
    require_h1_positive: bool = True
    min_positive_horizons: int = 3

    # Stage 4 — vol-targeted portfolio backtest
    vol_target: float = 0.15  # annualised target vol per instrument
    vol_lookback: int = 60  # days for vol estimate
    max_weight: float = 0.40  # per-asset cap
    max_gross_leverage: float = 2.0
    cost_bps: float = 20.0  # round-trip cost
    min_net_sharpe: float = 0.30

    # Stage 5 — walk-forward (CPCV)
    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    cpcv_pct_embargo: float = 0.01
    max_pbo: float = 0.50

    # Stage 6 — deflated Sharpe
    min_deflated_sharpe_pval: float = 0.95
    n_trials_for_deflation: int | None = None

    # Annualisation factor (bars per year): 365 for daily, 2190 for 4h, 8760 for 1h
    ann_factor: float = 365.0

    # Stage 7 — blend diversification (informational, no gate)
    max_blend_correlation: float = 0.30


@dataclass
class TSPipelineReport:
    """Full pipeline results for one TS alpha candidate."""

    candidate: TSCandidate
    stages: list[StageResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(
            s.verdict in (StageVerdict.PASS, StageVerdict.SKIP)
            for s in self.stages
        )

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
