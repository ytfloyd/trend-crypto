"""Type definitions for the convexity pipeline.

Pure declarative module. No business logic lives here. Defines the data
contracts between Stage evaluators, the Runner, and the Alpha Registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Protocol, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Track(str, Enum):
    """Which sub-pipeline this candidate runs in."""
    TREND = "trend"
    VOL_EXPANSION = "vol_expansion"
    BOTH = "both"
    NA = "N_A"


class PayoffShape(str, Enum):
    """Expected payoff shape. Drives routing at Stage 0."""
    CONVEX = "convex"
    LINEAR = "linear"
    CONCAVE = "concave"
    AMBIGUOUS = "ambiguous"
    NA = "N_A"


class Stage(str, Enum):
    """Pipeline stages. S0..S4 are in scope for Phase 1; S5..Retired are status only."""
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"            # Phase 2 - pre-production (status only in Phase 1)
    S6 = "S6"            # Phase 2 - live shadow (status only)
    LIVE = "Live"        # Phase 2
    RETIRED = "Retired"
    KILLED = "Killed"


# ---------------------------------------------------------------------------
# Hypothesis (Stage 0 input)
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """Pre-registered hypothesis. One per candidate. See alpha_hypothesis_template.md.

    All fields except `notes` are required for routing. Empty / None on a
    required field will be flagged by Stage0Evaluator.
    """
    name: str
    statement: str                                  # falsifiable sentence
    rationale: str                                  # economic / behavioral mechanism
    expected_payoff_shape: PayoffShape
    convexity_track: Track
    horizon_bars: int                               # expected trade horizon in bars
    universe: List[str]                             # tickers / contracts
    bar_frequency: str                              # "1min", "5min", "60min", "1d", "1w"
    params: Dict[str, Any] = field(default_factory=dict)
    cost_assumptions: Dict[str, Any] = field(default_factory=dict)
    expected_metrics: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    falsification_criteria: List[str] = field(default_factory=list)
    blowup_scenario: str = ""
    blowup_mitigation: str = ""
    researcher: str = ""
    registration_date: str = ""
    source_reference: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Candidate (registry entry + signal)
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A registered alpha candidate. Combines a hypothesis with an executable signal."""
    registry_id: str
    hypothesis: Hypothesis
    signal_fn: Callable[..., pd.Series]         # produces target positions
    backtest_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BacktestResult (Stage input)
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Standardized backtest output. Stage evaluators consume this.

    Schema:
        alpha_returns: pd.Series      # net strategy returns, indexed by datetime
        underlying_returns: pd.Series # aligned underlying / benchmark returns
        equity: pd.Series             # cumulative equity curve
        trade_pnls: pd.Series         # per-trade $ or % PnL, indexed by trade id
        trade_durations: pd.Series    # per-trade duration in bars
        per_instrument: Dict[str, BacktestResult]  # nested per-instrument
        meta: Dict[str, Any]          # backtest engine version, costs applied, etc.
    """
    alpha_returns: pd.Series
    underlying_returns: pd.Series
    equity: pd.Series
    trade_pnls: pd.Series
    trade_durations: pd.Series
    per_instrument: Dict[str, "BacktestResult"] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Output of a single stage evaluation."""
    stage: Stage
    passed: bool
    metrics: Dict[str, Any]
    kill_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class KillDecision:
    """Final kill record for a candidate that did not pass."""
    registry_id: str
    stage: Stage
    criteria_violated: List[str]
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class Stage1Thresholds:
    """Thresholds for Stage 1 fast vectorized screen.

    Defaults are ILLUSTRATIVE. Must be calibrated against the first cohort
    per the implementation plan (Step 6). Update thresholds.py with
    calibrated values, do not edit defaults here.
    """
    min_aggregate_skew: float = 0.0
    catastrophic_skew_floor: float = -1.0
    catastrophic_skew_max_fraction: float = 0.33   # > 1/3 of universe below floor = kill
    min_ccs_aggregate: float = 0.5
    min_universe_positive_fraction: float = 0.60
    min_convexity_beta: float = 0.0
    max_consecutive_losses: int = 40
    trade_duration_tolerance_factor: float = 5.0   # actual_duration in [horizon/5, horizon*5]


@dataclass
class Stage2Thresholds:
    max_ccs_drop_fraction: float = 0.50            # CCS_post_cost / CCS_pre_cost >= 0.50
    min_net_return_to_cost_ratio: float = 1.5
    max_per_instrument_net_negative_fraction: float = 0.30
    max_convexity_beta_p: float = 0.10


@dataclass
class Stage3Thresholds:
    min_ccs_oos_to_is_ratio: float = 0.5
    min_oos_folds: int = 8
    catastrophic_fold_dd_multiplier: float = 2.0   # any fold > 2x aggregate DD = kill
    max_consecutive_negative_ccs_folds: int = 3
    # Whether to require the MEDIAN per-fold skew to be positive (in addition to the
    # always-on aggregate-OOS-skew>0 guard). Defaults True to preserve the original
    # strict behavior. Cohort-01 calibration disables it: for genuinely convex,
    # lumpy alphas the positive skew is concentrated in a few fold windows, so the
    # median fold skew is often negative even when the pooled OOS distribution is
    # strongly positively skewed -- requiring positive median fold skew reintroduces
    # the Sharpe-punishes-convexity trap at the fold level. See thresholds.py.
    require_positive_median_fold_skew: bool = True


@dataclass
class Stage4Thresholds:
    min_param_perturbation_stability_fraction: float = 0.70
    perturbation_pct: float = 0.20
    perturbation_ccs_tolerance: float = 0.30       # CCS within 30% of base in N% of perturbations
    min_regimes_positive: int = 3
    regimes_total: int = 5
    universe_drop_min_ccs_fraction: float = 0.60
    cost_2x_must_be_positive: bool = True
    require_plateau_width: int = 3                 # peak must extend across at least 3 grid cells


@dataclass
class PipelineConfig:
    """Top-level configuration for a pipeline run."""
    stage1: Stage1Thresholds = field(default_factory=Stage1Thresholds)
    stage2: Stage2Thresholds = field(default_factory=Stage2Thresholds)
    stage3: Stage3Thresholds = field(default_factory=Stage3Thresholds)
    stage4: Stage4Thresholds = field(default_factory=Stage4Thresholds)
    periods_per_year: int = 252
    min_dd_floor: float = 0.05                     # CCS denominator floor
    tail_decile: float = 0.10                      # top decile for tail capture
    notes: str = "defaults; calibrate on first cohort"


def default_config() -> PipelineConfig:
    """Convenience: return a fresh default config."""
    return PipelineConfig()


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    """Cost model. Stage 1 uses a simplified version; Stage 2 the full version."""
    commission_per_unit: float = 0.0              # $ per share / contract
    bid_ask_bps: float = 1.0                       # half-spread in bps
    impact_k: float = 5.0                          # bps at 100% participation
    borrow_rate_annual: float = 0.0                # short borrow cost
    futures_roll_bps_annual: float = 0.0
    options_vega_carry_bps_annual: float = 0.0


# ---------------------------------------------------------------------------
# BacktestEngine protocol
# ---------------------------------------------------------------------------

class BacktestEngine(Protocol):
    """Protocol the existing backtest engine must satisfy.

    Adapter responsibility: wrap our existing engine so it produces
    BacktestResult instances with all fields populated, including
    per_instrument breakdown.
    """
    def run(
        self,
        candidate: "Candidate",
        cost_model: CostModel,
        **kwargs: Any,
    ) -> BacktestResult: ...
