"""Pipeline orchestrator.

Takes a list of Candidates + a backtest engine, advances each through
Stages 0-4 in order, stops at first kill, emits a scorecard DataFrame.

Backtest engine integration:
    Caller provides a callable that produces a BacktestResult given a Candidate
    and a CostModel. Phase 2 will wrap the existing backtest engine via adapter.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .stages import (
    Stage0Evaluator,
    Stage1Evaluator,
    Stage2Evaluator,
    Stage3Evaluator,
    Stage4Evaluator,
)
from .types import (
    BacktestResult,
    Candidate,
    PipelineConfig,
    Stage,
    StageResult,
    default_config,
)


# Type alias: a backtest fn takes a candidate + a string variant tag and returns a BacktestResult.
# Variants: "stage1_simple_cost", "stage1_pre_cost", "stage2_realistic",
#           "is", "oos_fold_<N>", "perturb_<idx>", "regime_<name>",
#           "universe_drop", "cost_2x".
BacktestFn = Callable[[Candidate, str], BacktestResult]


class ConvexityPipelineRunner:
    """Orchestrates progression of candidates through Stages 0-4."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config()
        self.s0 = Stage0Evaluator(self.config)
        self.s1 = Stage1Evaluator(self.config)
        self.s2 = Stage2Evaluator(self.config)
        self.s3 = Stage3Evaluator(self.config)
        self.s4 = Stage4Evaluator(self.config)

    def run_candidate(
        self,
        candidate: Candidate,
        backtest_fn: BacktestFn,
        oos_fold_count: int = 8,
        perturbation_count: int = 8,
        regimes: Optional[List[str]] = None,
        run_stage4: bool = True,
        look_ahead_signoff: bool = False,
    ) -> List[StageResult]:
        """Run a single candidate through stages. Stop at first kill."""
        regimes = regimes or ["bull", "bear", "sideways", "high_vol", "low_vol"]
        results: List[StageResult] = []

        # Stage 0 — hypothesis registration check
        r0 = self.s0.evaluate(candidate)
        results.append(r0)
        if not r0.passed:
            return results

        # Stage 1 — fast vectorized screen
        try:
            bt_s1 = backtest_fn(candidate, "stage1_simple_cost")
        except Exception as e:
            results.append(StageResult(
                stage=Stage.S1, passed=False, metrics={},
                kill_reasons=[f"backtest_failure:{type(e).__name__}:{e}"],
                notes="Stage 1 backtest threw",
            ))
            return results
        r1 = self.s1.evaluate(candidate, bt_s1)
        results.append(r1)
        if not r1.passed:
            return results

        # Stage 2 — realistic costs
        try:
            bt_s2_pre = backtest_fn(candidate, "stage2_pre_cost")
            bt_s2_post = backtest_fn(candidate, "stage2_realistic")
        except Exception as e:
            results.append(StageResult(
                stage=Stage.S2, passed=False, metrics={},
                kill_reasons=[f"backtest_failure:{type(e).__name__}:{e}"],
                notes="Stage 2 backtest threw",
            ))
            return results
        r2 = self.s2.evaluate(
            candidate,
            backtest=bt_s2_post,
            backtest_pre_cost=bt_s2_pre,
            backtest_post_cost=bt_s2_post,
        )
        results.append(r2)
        if not r2.passed:
            return results

        # Stage 3 — walk-forward OOS
        try:
            bt_is = backtest_fn(candidate, "is")
            bt_oos_folds = [
                backtest_fn(candidate, f"oos_fold_{i}")
                for i in range(oos_fold_count)
            ]
        except Exception as e:
            results.append(StageResult(
                stage=Stage.S3, passed=False, metrics={},
                kill_reasons=[f"backtest_failure:{type(e).__name__}:{e}"],
                notes="Stage 3 backtest threw",
            ))
            return results
        r3 = self.s3.evaluate(
            candidate,
            backtest=bt_is,
            backtest_is=bt_is,
            backtest_oos_folds=bt_oos_folds,
        )
        results.append(r3)
        if not r3.passed or not run_stage4:
            return results

        # Stage 4 — robustness battery
        try:
            perts = [
                backtest_fn(candidate, f"perturb_{i}")
                for i in range(perturbation_count)
            ]
            regime_bts = {
                name: backtest_fn(candidate, f"regime_{name}") for name in regimes
            }
            ud = backtest_fn(candidate, "universe_drop")
            c2 = backtest_fn(candidate, "cost_2x")
        except Exception as e:
            results.append(StageResult(
                stage=Stage.S4, passed=False, metrics={},
                kill_reasons=[f"backtest_failure:{type(e).__name__}:{e}"],
                notes="Stage 4 backtest threw",
            ))
            return results

        r4 = self.s4.evaluate(
            candidate,
            backtest=bt_s2_post,
            perturbation_results=perts,
            regime_results=regime_bts,
            universe_drop_result=ud,
            cost_2x_result=c2,
            look_ahead_signoff=look_ahead_signoff,
        )
        results.append(r4)
        return results

    def run_cohort(
        self,
        candidates: List[Candidate],
        backtest_fn: BacktestFn,
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, Dict[str, List[StageResult]]]:
        """Run all candidates and emit a scorecard.

        Returns:
            scorecard: one row per candidate, columns for each stage's pass/fail
                       and key metrics.
            per_candidate_results: {registry_id: List[StageResult]}
        """
        rows: List[Dict[str, Any]] = []
        per_cand: Dict[str, List[StageResult]] = {}
        for cand in candidates:
            results = self.run_candidate(cand, backtest_fn, **kwargs)
            per_cand[cand.registry_id] = results
            row: Dict[str, Any] = {
                "registry_id": cand.registry_id,
                "name": cand.hypothesis.name,
                "track": cand.hypothesis.convexity_track.value,
                "shape": cand.hypothesis.expected_payoff_shape.value,
            }
            final_stage = results[-1] if results else None
            row["final_stage"] = final_stage.stage.value if final_stage else None
            row["passed_all_run"] = (
                final_stage.passed if final_stage is not None else False
            )
            for r in results:
                row[f"{r.stage.value}_passed"] = r.passed
                if r.kill_reasons:
                    row[f"{r.stage.value}_kill"] = ";".join(r.kill_reasons)
            # Pull headline metrics from Stage 1 if reached
            for r in results:
                if r.stage == Stage.S1 and "aggregate" in r.metrics:
                    agg = r.metrics["aggregate"]
                    for k in ("ccs", "skew", "sharpe", "calmar",
                              "tail_capture", "convexity_beta_b",
                              "hit_rate", "max_drawdown"):
                        row[f"s1_{k}"] = agg.get(k)
                if r.stage == Stage.S3:
                    if "ccs_oos_mean" in r.metrics:
                        row["s3_ccs_oos_mean"] = r.metrics["ccs_oos_mean"]
                    if "ccs_oos_to_is_ratio" in r.metrics:
                        row["s3_ccs_oos_to_is_ratio"] = r.metrics["ccs_oos_to_is_ratio"]
            rows.append(row)
        scorecard = pd.DataFrame(rows)
        return scorecard, per_cand

    def persist(
        self,
        scorecard: pd.DataFrame,
        per_cand: Dict[str, List[StageResult]],
        base_path: str = "data/alpha_registry/runs",
    ) -> str:
        """Persist run output to a timestamped folder.

        Returns the output folder path.
        """
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        folder = Path(base_path) / ts
        folder.mkdir(parents=True, exist_ok=True)
        scorecard.to_csv(folder / "scorecard.csv", index=False)
        try:
            scorecard.to_parquet(folder / "scorecard.parquet", index=False)
        except (ImportError, ValueError):
            # parquet engine not installed; csv + json are enough for V1
            pass

        # Per-candidate JSON
        import json
        per_cand_json: Dict[str, List[Dict[str, Any]]] = {}
        for rid, results in per_cand.items():
            per_cand_json[rid] = [
                {
                    "stage": r.stage.value,
                    "passed": r.passed,
                    "metrics": {k: _to_json(v) for k, v in r.metrics.items()},
                    "kill_reasons": r.kill_reasons,
                    "warnings": r.warnings,
                    "notes": r.notes,
                    "timestamp": r.timestamp,
                }
                for r in results
            ]
        with (folder / "per_candidate.json").open("w") as f:
            json.dump(per_cand_json, f, indent=2, default=_to_json)
        return str(folder)


def _to_json(v: Any) -> Any:
    """Make NumPy and pandas types JSON-serializable."""
    try:
        import numpy as _np
        if isinstance(v, (_np.floating, _np.integer)):
            return float(v)
        if isinstance(v, _np.ndarray):
            return v.tolist()
    except ImportError:
        pass
    if isinstance(v, dict):
        return {k: _to_json(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_to_json(vv) for vv in v]
    return v
