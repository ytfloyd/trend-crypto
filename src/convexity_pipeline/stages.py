"""Stage evaluators for the convexity pipeline.

Each evaluator implements evaluate(candidate, backtest) -> StageResult.

Phase 1 implements Stages 0-4 deeply. Stages 5-7 are tracked as status only
in the registry; their evaluators are not implemented here.

Kill criteria are read from PipelineConfig (types.py); they are NOT hardcoded.
This allows calibration on the first cohort per the implementation plan.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import metrics as M
from .types import (
    BacktestResult,
    Candidate,
    Hypothesis,
    PayoffShape,
    PipelineConfig,
    Stage,
    StageResult,
    Track,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class StageEvaluator(ABC):
    """Base class for stage evaluators."""

    stage: Stage

    def __init__(self, config: PipelineConfig):
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        candidate: Candidate,
        backtest: Optional[BacktestResult] = None,
        **kwargs: Any,
    ) -> StageResult: ...


# ---------------------------------------------------------------------------
# Stage 0 - hypothesis registration validation
# ---------------------------------------------------------------------------

class Stage0Evaluator(StageEvaluator):
    """Validates hypothesis completeness and routing.

    Does NOT touch backtest data. Pure metadata check.
    """

    stage = Stage.S0

    def evaluate(
        self,
        candidate: Candidate,
        backtest: Optional[BacktestResult] = None,
        **kwargs: Any,
    ) -> StageResult:
        h: Hypothesis = candidate.hypothesis
        kill: List[str] = []
        warn: List[str] = []
        m: Dict[str, Any] = {}

        # Required fields
        for field_name in ("name", "statement", "rationale", "researcher",
                           "registration_date", "bar_frequency"):
            if not getattr(h, field_name):
                kill.append(f"missing_field:{field_name}")

        if not h.universe:
            kill.append("missing_field:universe")

        if h.horizon_bars <= 0:
            kill.append("invalid_horizon_bars")

        # Routing: convex shape required for convexity pipeline
        if h.expected_payoff_shape == PayoffShape.LINEAR:
            kill.append("wrong_pipeline:linear_alpha_belongs_to_alpha_pipeline")
        elif h.expected_payoff_shape == PayoffShape.CONCAVE:
            kill.append("wrong_pipeline:concave_alpha_out_of_scope")
        elif h.expected_payoff_shape == PayoffShape.AMBIGUOUS:
            warn.append("payoff_shape_ambiguous_requires_research_lead_decision")

        # Track must be set if convex
        if h.expected_payoff_shape == PayoffShape.CONVEX:
            if h.convexity_track == Track.NA:
                kill.append("convex_alpha_must_specify_track")

        # Convex alpha must include stops / breakout / vol vehicle hint
        if h.expected_payoff_shape == PayoffShape.CONVEX:
            rationale_lc = (h.rationale + " " + h.statement).lower()
            convexity_hints = [
                "stop", "breakout", "compression", "volatility", "vol",
                "gamma", "option", "vix", "tail", "trend", "momentum",
            ]
            if not any(hint in rationale_lc for hint in convexity_hints):
                warn.append("convex_alpha_no_convexity_mechanism_in_rationale")

        # Expected metrics pre-registration
        if not h.expected_metrics:
            warn.append("no_expected_metrics_pre_registered")

        m["payoff_shape"] = h.expected_payoff_shape.value
        m["convexity_track"] = h.convexity_track.value
        m["universe_size"] = len(h.universe)
        m["horizon_bars"] = h.horizon_bars

        passed = len(kill) == 0
        return StageResult(
            stage=self.stage,
            passed=passed,
            metrics=m,
            kill_reasons=kill,
            warnings=warn,
            notes="hypothesis_registration_validation",
        )


# ---------------------------------------------------------------------------
# Stage 1 - fast vectorized screen
# ---------------------------------------------------------------------------

class Stage1Evaluator(StageEvaluator):
    """Fast vectorized screen on standard universe with simplified cost model.

    Kill criteria from spec Section 5 / Stage 1 (dev-modified):
    - aggregate skew < min_aggregate_skew
    - per-instrument: skew below catastrophic_skew_floor on > catastrophic_skew_max_fraction of universe
    - CCS_aggregate < min_ccs_aggregate
    - % of universe with positive CCS < min_universe_positive_fraction
    - convexity beta b <= min_convexity_beta on aggregate
    - trade duration not within horizon * tolerance_factor band
    - max consecutive losses > max_consecutive_losses
    """

    stage = Stage.S1

    def evaluate(
        self,
        candidate: Candidate,
        backtest: Optional[BacktestResult] = None,
        **kwargs: Any,
    ) -> StageResult:
        if backtest is None:
            return StageResult(
                stage=self.stage,
                passed=False,
                metrics={},
                kill_reasons=["no_backtest_provided"],
                notes="backtest required for Stage 1",
            )

        cfg = self.config.stage1
        m: Dict[str, Any] = {}
        kill: List[str] = []
        warn: List[str] = []

        # Aggregate metrics
        agg = M.calculate_all(
            alpha_returns=backtest.alpha_returns,
            underlying_returns=backtest.underlying_returns,
            equity=backtest.equity,
            trade_pnls=backtest.trade_pnls,
            trade_durations=backtest.trade_durations,
            periods_per_year=self.config.periods_per_year,
            min_dd_floor=self.config.min_dd_floor,
            tail_decile=self.config.tail_decile,
        )
        m["aggregate"] = agg

        # Per-instrument
        per_inst_ccs: Dict[str, Optional[float]] = {}
        per_inst_skew: Dict[str, Optional[float]] = {}
        for sym, sub_bt in (backtest.per_instrument or {}).items():
            sub_ccs = M.composite_convexity_score(
                returns=sub_bt.alpha_returns,
                trade_pnls=sub_bt.trade_pnls,
                alpha_returns_for_capture=sub_bt.alpha_returns,
                underlying_returns_for_capture=sub_bt.underlying_returns,
                periods_per_year=self.config.periods_per_year,
                min_dd_floor=self.config.min_dd_floor,
                tail_decile=self.config.tail_decile,
            )
            per_inst_ccs[sym] = sub_ccs
            per_inst_skew[sym] = M.skew(sub_bt.alpha_returns)
        m["per_instrument_ccs"] = per_inst_ccs
        m["per_instrument_skew"] = per_inst_skew

        # Kill checks
        agg_skew = agg["skew"]
        if agg_skew is None or agg_skew < cfg.min_aggregate_skew:
            kill.append(f"aggregate_skew_below_threshold:{agg_skew}")

        # Catastrophic per-instrument skew
        if per_inst_skew:
            n_catastrophic = sum(
                1 for s in per_inst_skew.values()
                if s is not None and s < cfg.catastrophic_skew_floor
            )
            frac_catastrophic = n_catastrophic / max(len(per_inst_skew), 1)
            m["catastrophic_skew_fraction"] = frac_catastrophic
            if frac_catastrophic > cfg.catastrophic_skew_max_fraction:
                kill.append(
                    f"catastrophic_skew_per_instrument:{frac_catastrophic:.2f}"
                )

        agg_ccs = agg["ccs"]
        if agg_ccs is None or agg_ccs < cfg.min_ccs_aggregate:
            kill.append(f"ccs_aggregate_below_threshold:{agg_ccs}")

        # Universe positive fraction
        if per_inst_ccs:
            n_pos = sum(1 for c in per_inst_ccs.values() if c is not None and c > 0)
            frac_pos = n_pos / max(len(per_inst_ccs), 1)
            m["universe_positive_fraction"] = frac_pos
            if frac_pos < cfg.min_universe_positive_fraction:
                kill.append(
                    f"universe_positive_below_threshold:{frac_pos:.2f}"
                )

        # Convexity beta
        b = agg["convexity_beta_b"]
        if b is None or b <= cfg.min_convexity_beta:
            kill.append(f"convexity_beta_below_threshold:{b}")

        # Max consecutive losses
        mcl = agg["max_consecutive_losses"]
        if mcl > cfg.max_consecutive_losses:
            kill.append(f"max_consecutive_losses:{mcl}")

        # Trade duration vs hypothesis horizon
        h = candidate.hypothesis
        median_dur = agg["median_trade_duration"]
        lo = h.horizon_bars / cfg.trade_duration_tolerance_factor
        hi = h.horizon_bars * cfg.trade_duration_tolerance_factor
        if median_dur > 0 and not (lo <= median_dur <= hi):
            kill.append(
                f"trade_duration_mismatch:{median_dur} not in [{lo:.1f},{hi:.1f}]"
            )

        passed = len(kill) == 0
        return StageResult(
            stage=self.stage,
            passed=passed,
            metrics=m,
            kill_reasons=kill,
            warnings=warn,
            notes="fast_vectorized_screen",
        )


# ---------------------------------------------------------------------------
# Stage 2 - realistic backtest
# ---------------------------------------------------------------------------

class Stage2Evaluator(StageEvaluator):
    """Apply realistic costs; verify alpha survives.

    Caller passes BOTH the Stage-1 backtest and the Stage-2 (post-cost) backtest.
    """

    stage = Stage.S2

    def evaluate(
        self,
        candidate: Candidate,
        backtest: Optional[BacktestResult] = None,
        **kwargs: Any,
    ) -> StageResult:
        backtest_pre = kwargs.get("backtest_pre_cost")
        backtest_post = backtest or kwargs.get("backtest_post_cost")
        if backtest_pre is None or backtest_post is None:
            return StageResult(
                stage=self.stage,
                passed=False,
                metrics={},
                kill_reasons=["missing_pre_or_post_cost_backtest"],
                notes="Stage 2 requires backtest_pre_cost and backtest_post_cost",
            )

        cfg = self.config.stage2
        kill: List[str] = []
        warn: List[str] = []
        m: Dict[str, Any] = {}

        agg_pre = M.calculate_all(
            alpha_returns=backtest_pre.alpha_returns,
            underlying_returns=backtest_pre.underlying_returns,
            equity=backtest_pre.equity,
            trade_pnls=backtest_pre.trade_pnls,
            trade_durations=backtest_pre.trade_durations,
            periods_per_year=self.config.periods_per_year,
            min_dd_floor=self.config.min_dd_floor,
            tail_decile=self.config.tail_decile,
        )
        agg_post = M.calculate_all(
            alpha_returns=backtest_post.alpha_returns,
            underlying_returns=backtest_post.underlying_returns,
            equity=backtest_post.equity,
            trade_pnls=backtest_post.trade_pnls,
            trade_durations=backtest_post.trade_durations,
            periods_per_year=self.config.periods_per_year,
            min_dd_floor=self.config.min_dd_floor,
            tail_decile=self.config.tail_decile,
        )
        m["aggregate_pre"] = agg_pre
        m["aggregate_post"] = agg_post

        # CCS drop
        ccs_pre = agg_pre["ccs"]
        ccs_post = agg_post["ccs"]
        if ccs_pre is None or ccs_pre <= 0 or ccs_post is None:
            kill.append("invalid_ccs_for_drop_check")
        else:
            drop = (ccs_pre - ccs_post) / ccs_pre
            m["ccs_drop_fraction"] = drop
            if drop > cfg.max_ccs_drop_fraction:
                kill.append(f"ccs_drop_too_large:{drop:.2f}")

        # Convexity beta significance after costs
        b_p = agg_post["convexity_beta_p"]
        if b_p is None or b_p > cfg.max_convexity_beta_p:
            kill.append(f"convexity_beta_loses_significance:p={b_p}")

        # Per-instrument net-negative fraction
        if backtest_post.per_instrument:
            n_neg = 0
            for sym, sub_bt in backtest_post.per_instrument.items():
                tot_ret = sub_bt.alpha_returns.sum()
                if tot_ret < 0:
                    n_neg += 1
            frac_neg = n_neg / max(len(backtest_post.per_instrument), 1)
            m["per_instrument_net_negative_fraction"] = frac_neg
            if frac_neg > cfg.max_per_instrument_net_negative_fraction:
                kill.append(f"per_instrument_net_negative:{frac_neg:.2f}")

        passed = len(kill) == 0
        return StageResult(
            stage=self.stage,
            passed=passed,
            metrics=m,
            kill_reasons=kill,
            warnings=warn,
            notes="realistic_backtest_with_costs",
        )


# ---------------------------------------------------------------------------
# Stage 3 - walk-forward OOS
# ---------------------------------------------------------------------------

class Stage3Evaluator(StageEvaluator):
    """Walk-forward OOS evaluation.

    Caller passes the IS BacktestResult and a list of OOS BacktestResults
    (one per fold). Uses CCS ratio for the IS/OOS comparison (dev modification).
    """

    stage = Stage.S3

    def evaluate(
        self,
        candidate: Candidate,
        backtest: Optional[BacktestResult] = None,
        **kwargs: Any,
    ) -> StageResult:
        backtest_is = backtest or kwargs.get("backtest_is")
        backtest_oos_folds = kwargs.get("backtest_oos_folds", [])

        if backtest_is is None or not backtest_oos_folds:
            return StageResult(
                stage=self.stage,
                passed=False,
                metrics={},
                kill_reasons=["missing_is_or_oos_backtests"],
                notes="Stage 3 requires backtest_is + backtest_oos_folds",
            )

        cfg = self.config.stage3
        kill: List[str] = []
        warn: List[str] = []
        m: Dict[str, Any] = {}

        n_folds = len(backtest_oos_folds)
        m["n_oos_folds"] = n_folds
        if n_folds < cfg.min_oos_folds:
            kill.append(f"insufficient_oos_folds:{n_folds}<{cfg.min_oos_folds}")

        ccs_is = M.composite_convexity_score(
            returns=backtest_is.alpha_returns,
            trade_pnls=backtest_is.trade_pnls,
            alpha_returns_for_capture=backtest_is.alpha_returns,
            underlying_returns_for_capture=backtest_is.underlying_returns,
            periods_per_year=self.config.periods_per_year,
            min_dd_floor=self.config.min_dd_floor,
            tail_decile=self.config.tail_decile,
        )
        m["ccs_is"] = ccs_is

        fold_ccs: List[Optional[float]] = []
        fold_skew: List[Optional[float]] = []
        fold_dd: List[float] = []
        for f in backtest_oos_folds:
            fold_ccs.append(M.composite_convexity_score(
                returns=f.alpha_returns,
                trade_pnls=f.trade_pnls,
                alpha_returns_for_capture=f.alpha_returns,
                underlying_returns_for_capture=f.underlying_returns,
                periods_per_year=self.config.periods_per_year,
                min_dd_floor=self.config.min_dd_floor,
                tail_decile=self.config.tail_decile,
            ))
            fold_skew.append(M.skew(f.alpha_returns))
            fold_dd.append(M.max_drawdown(f.equity))
        m["fold_ccs"] = fold_ccs
        m["fold_skew"] = fold_skew

        valid_ccs = [c for c in fold_ccs if c is not None]
        if valid_ccs:
            ccs_oos_mean = float(np.mean(valid_ccs))
            m["ccs_oos_mean"] = ccs_oos_mean
            if ccs_is is not None and ccs_is > 0:
                ratio = ccs_oos_mean / ccs_is
                m["ccs_oos_to_is_ratio"] = ratio
                if ratio < cfg.min_ccs_oos_to_is_ratio:
                    kill.append(f"ccs_oos_to_is_ratio_low:{ratio:.2f}")
        else:
            kill.append("no_valid_oos_ccs")

        # Skew: median AND aggregate must be positive (dev modification)
        valid_skew = [s for s in fold_skew if s is not None]
        if valid_skew:
            median_skew = float(np.median(valid_skew))
            m["median_oos_skew"] = median_skew
            if cfg.require_positive_median_fold_skew and median_skew <= 0:
                kill.append(f"median_oos_skew_not_positive:{median_skew:.3f}")
            agg_oos_returns = pd.concat(
                [f.alpha_returns for f in backtest_oos_folds], axis=0
            ).dropna()
            agg_oos_skew = M.skew(agg_oos_returns)
            m["aggregate_oos_skew"] = agg_oos_skew
            if agg_oos_skew is None or agg_oos_skew <= 0:
                kill.append(f"aggregate_oos_skew_not_positive:{agg_oos_skew}")

        # Catastrophic fold check (DD > N x aggregate DD)
        if fold_dd:
            agg_dd = M.max_drawdown(backtest_is.equity)
            for i, fdd in enumerate(fold_dd):
                if agg_dd > 0 and fdd > cfg.catastrophic_fold_dd_multiplier * agg_dd:
                    kill.append(
                        f"catastrophic_fold:{i} dd={fdd:.3f} > "
                        f"{cfg.catastrophic_fold_dd_multiplier} * {agg_dd:.3f}"
                    )

        # Consecutive negative CCS folds
        consec_neg = 0
        max_consec_neg = 0
        for c in fold_ccs:
            if c is None or c <= 0:
                consec_neg += 1
                max_consec_neg = max(max_consec_neg, consec_neg)
            else:
                consec_neg = 0
        m["max_consecutive_negative_folds"] = max_consec_neg
        if max_consec_neg > cfg.max_consecutive_negative_ccs_folds:
            kill.append(f"too_many_consecutive_negative_folds:{max_consec_neg}")

        passed = len(kill) == 0
        return StageResult(
            stage=self.stage,
            passed=passed,
            metrics=m,
            kill_reasons=kill,
            warnings=warn,
            notes="walk_forward_oos",
        )


# ---------------------------------------------------------------------------
# Stage 4 - robustness battery
# ---------------------------------------------------------------------------

class Stage4Evaluator(StageEvaluator):
    """Robustness battery.

    Caller must provide:
        backtest:        base case BacktestResult
        perturbation_results: List[BacktestResult] from perturbed parameters
        regime_results: Dict[str, BacktestResult] keyed by regime name
        universe_drop_result: BacktestResult with top instrument dropped
        cost_2x_result: BacktestResult with 2x cost model
        grid_search:    pd.DataFrame of param-grid CCS values for plateau detection (optional)
        look_ahead_signoff: bool (set by reviewer)
    """

    stage = Stage.S4

    def evaluate(
        self,
        candidate: Candidate,
        backtest: Optional[BacktestResult] = None,
        **kwargs: Any,
    ) -> StageResult:
        cfg = self.config.stage4
        kill: List[str] = []
        warn: List[str] = []
        m: Dict[str, Any] = {}

        perturbations: List[BacktestResult] = kwargs.get("perturbation_results", []) or []
        regimes: Dict[str, BacktestResult] = kwargs.get("regime_results", {}) or {}
        universe_drop: Optional[BacktestResult] = kwargs.get("universe_drop_result")
        cost_2x: Optional[BacktestResult] = kwargs.get("cost_2x_result")
        grid: Optional[pd.DataFrame] = kwargs.get("grid_search")
        look_ahead_signoff: bool = bool(kwargs.get("look_ahead_signoff", False))

        if backtest is None:
            return StageResult(
                stage=self.stage,
                passed=False,
                metrics={},
                kill_reasons=["no_base_backtest"],
                notes="Stage 4 requires base BacktestResult",
            )

        base_ccs = M.composite_convexity_score(
            returns=backtest.alpha_returns,
            trade_pnls=backtest.trade_pnls,
            alpha_returns_for_capture=backtest.alpha_returns,
            underlying_returns_for_capture=backtest.underlying_returns,
            periods_per_year=self.config.periods_per_year,
            min_dd_floor=self.config.min_dd_floor,
            tail_decile=self.config.tail_decile,
        )
        m["base_ccs"] = base_ccs

        # 1. Parameter perturbation
        if perturbations:
            within_tol = 0
            for p_bt in perturbations:
                p_ccs = M.composite_convexity_score(
                    returns=p_bt.alpha_returns,
                    trade_pnls=p_bt.trade_pnls,
                    alpha_returns_for_capture=p_bt.alpha_returns,
                    underlying_returns_for_capture=p_bt.underlying_returns,
                    periods_per_year=self.config.periods_per_year,
                    min_dd_floor=self.config.min_dd_floor,
                    tail_decile=self.config.tail_decile,
                )
                if base_ccs and p_ccs is not None:
                    if abs(p_ccs - base_ccs) <= cfg.perturbation_ccs_tolerance * abs(base_ccs):
                        within_tol += 1
            frac_stable = within_tol / len(perturbations)
            m["parameter_stability_fraction"] = frac_stable
            if frac_stable < cfg.min_param_perturbation_stability_fraction:
                kill.append(f"parameter_unstable:{frac_stable:.2f}")
        else:
            warn.append("no_perturbation_results_provided")

        # 2. Regime decomposition
        if regimes:
            n_positive = 0
            regime_ccs: Dict[str, Optional[float]] = {}
            for name, r_bt in regimes.items():
                rc = M.composite_convexity_score(
                    returns=r_bt.alpha_returns,
                    trade_pnls=r_bt.trade_pnls,
                    alpha_returns_for_capture=r_bt.alpha_returns,
                    underlying_returns_for_capture=r_bt.underlying_returns,
                    periods_per_year=self.config.periods_per_year,
                    min_dd_floor=self.config.min_dd_floor,
                    tail_decile=self.config.tail_decile,
                )
                regime_ccs[name] = rc
                if rc is not None and rc > 0:
                    n_positive += 1
            m["regime_ccs"] = regime_ccs
            if n_positive < cfg.min_regimes_positive:
                kill.append(f"too_few_positive_regimes:{n_positive}<{cfg.min_regimes_positive}")
        else:
            warn.append("no_regime_results_provided")

        # 3. Universe drop
        if universe_drop is not None:
            ud_ccs = M.composite_convexity_score(
                returns=universe_drop.alpha_returns,
                trade_pnls=universe_drop.trade_pnls,
                alpha_returns_for_capture=universe_drop.alpha_returns,
                underlying_returns_for_capture=universe_drop.underlying_returns,
                periods_per_year=self.config.periods_per_year,
                min_dd_floor=self.config.min_dd_floor,
                tail_decile=self.config.tail_decile,
            )
            m["universe_drop_ccs"] = ud_ccs
            if base_ccs and ud_ccs is not None:
                ratio = ud_ccs / base_ccs
                m["universe_drop_ratio"] = ratio
                if ratio < cfg.universe_drop_min_ccs_fraction:
                    kill.append(f"universe_drop_too_destructive:{ratio:.2f}")
        else:
            warn.append("no_universe_drop_result_provided")

        # 4. Cost 2x
        if cost_2x is not None and cfg.cost_2x_must_be_positive:
            cost_2x_ret = cost_2x.alpha_returns.sum()
            m["cost_2x_total_return"] = float(cost_2x_ret)
            if cost_2x_ret <= 0:
                kill.append("cost_2x_negative_total_return")
        elif cost_2x is None:
            warn.append("no_cost_2x_result_provided")

        # 5. Look-ahead audit (reviewer sign-off)
        m["look_ahead_signoff"] = look_ahead_signoff
        if not look_ahead_signoff:
            kill.append("look_ahead_audit_sign_off_missing")

        # 6. Curve-fit detection from grid
        if grid is not None and not grid.empty:
            grid_arr = grid.values
            max_val = np.nanmax(grid_arr)
            mask = grid_arr >= 0.9 * max_val
            plateau = int(mask.sum())
            m["plateau_width"] = plateau
            if plateau < cfg.require_plateau_width:
                kill.append(f"narrow_plateau:{plateau}")

        passed = len(kill) == 0
        return StageResult(
            stage=self.stage,
            passed=passed,
            metrics=m,
            kill_reasons=kill,
            warnings=warn,
            notes="robustness_battery",
        )
