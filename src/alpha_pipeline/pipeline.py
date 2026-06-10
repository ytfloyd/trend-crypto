"""Alpha pipeline orchestrator — the conveyor belt.

V2 changes (2026-03):
  - New turnover/cost gate between Redundancy and Walk-Forward
  - Passes returns_wide to stages for inverse-vol weighting
  - Passes close_wide to redundancy for orthogonalized IC testing
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.alpha_pipeline.stages import (
    stage_deflated_sharpe,
    stage_ic_decay,
    stage_ic_screen,
    stage_redundancy,
    stage_turnover,
    stage_walk_forward,
)
from src.alpha_pipeline.types import (
    AlphaCandidate,
    GateConfig,
    PipelineReport,
    StageVerdict,
)

logger = logging.getLogger(__name__)


class AlphaPipeline:
    """Automated alpha research pipeline.

    Stages (each has a gate that can reject):
        1. IC Screen     — cross-sectional Spearman IC vs forward returns
        2. IC Decay      — verify signal decays with horizon
        3. Redundancy    — orthogonalized check against existing alphas
        3.5 Turnover     — net-of-cost IC must be positive
        4. Walk-Forward  — CPCV with embargo + inverse-vol L/S + PBO
        5. Deflated SR   — multiple-testing correction

    Candidates that pass all stages are added to the approved set.
    """

    def __init__(
        self,
        close_wide: pd.DataFrame,
        volume_wide: pd.DataFrame,
        config: GateConfig | None = None,
        approved_alphas: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.close_wide = close_wide
        self.volume_wide = volume_wide
        self.returns_wide = close_wide.pct_change(fill_method=None)
        self.config = config or GateConfig()
        self.approved_alphas: dict[str, pd.DataFrame] = approved_alphas or {}
        self.reports: list[PipelineReport] = []
        self._n_tested = 0

    def evaluate(
        self,
        candidate: AlphaCandidate,
        *,
        stop_on_fail: bool = True,
    ) -> PipelineReport:
        """Run a single candidate through all pipeline stages.

        Args:
            candidate: The alpha to evaluate.
            stop_on_fail: If True, skip remaining stages after first failure.
        """
        self._n_tested += 1
        report = PipelineReport(candidate=candidate)

        scores = candidate.get_scores(
            self.close_wide, self.volume_wide, self.returns_wide,
        )

        logger.info("── Evaluating: %s (%s) ──", candidate.name, candidate.family)

        # Stage 1: IC Screen
        result = stage_ic_screen(scores, self.close_wide, self.config)
        report.stages.append(result)
        self._log_stage(result)
        if result.verdict == StageVerdict.FAIL and stop_on_fail:
            self.reports.append(report)
            return report

        # Stage 2: IC Decay
        result = stage_ic_decay(scores, self.close_wide, self.config)
        report.stages.append(result)
        self._log_stage(result)
        if result.verdict == StageVerdict.FAIL and stop_on_fail:
            self.reports.append(report)
            return report

        # Stage 3: Redundancy (with orthogonalization)
        result = stage_redundancy(
            scores, self.approved_alphas, self.config,
            close_wide=self.close_wide,
        )
        report.stages.append(result)
        self._log_stage(result)
        if result.verdict == StageVerdict.FAIL and stop_on_fail:
            self.reports.append(report)
            return report

        # Stage 3.5: Turnover / Cost
        result = stage_turnover(scores, self.close_wide, self.config)
        report.stages.append(result)
        self._log_stage(result)
        if result.verdict == StageVerdict.FAIL and stop_on_fail:
            self.reports.append(report)
            return report

        # Stage 4: Walk-Forward (CPCV + PBO)
        result = stage_walk_forward(
            scores, self.close_wide, self.config,
            returns_wide=self.returns_wide,
        )
        report.stages.append(result)
        self._log_stage(result)
        if result.verdict == StageVerdict.FAIL and stop_on_fail:
            self.reports.append(report)
            return report

        # Stage 5: Deflated Sharpe
        result = stage_deflated_sharpe(
            scores, self.close_wide, self.config,
            n_candidates_tested=self._n_tested,
            returns_wide=self.returns_wide,
        )
        report.stages.append(result)
        self._log_stage(result)

        # If all stages passed, add to approved set
        if report.passed:
            self.approved_alphas[candidate.name] = scores
            logger.info(
                "✓ APPROVED: %s (total approved: %d)",
                candidate.name, len(self.approved_alphas),
            )
        else:
            logger.info("✗ REJECTED: %s at stage '%s'",
                         candidate.name, report.failed_stage)

        self.reports.append(report)
        return report

    def evaluate_batch(
        self,
        candidates: list[AlphaCandidate],
        *,
        stop_on_fail: bool = True,
    ) -> list[PipelineReport]:
        """Evaluate a batch of candidates sequentially.

        Redundancy check naturally accumulates as approved alphas grow.
        """
        logger.info(
            "Starting pipeline: %d candidates, gates=%s",
            len(candidates), self.config,
        )

        if self.config.n_trials_for_deflation is None:
            object.__setattr__(
                self.config, "n_trials_for_deflation", len(candidates),
            )

        reports = []
        for i, c in enumerate(candidates, 1):
            logger.info("─── Candidate %d/%d ───", i, len(candidates))
            r = self.evaluate(c, stop_on_fail=stop_on_fail)
            reports.append(r)

        self._log_summary()
        return reports

    def summary_table(self) -> pd.DataFrame:
        """Build a DataFrame summarizing all evaluated candidates."""
        rows = [r.summary_dict() for r in self.reports]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def write_catalog(self, output_dir: str | Path) -> Path:
        """Write pipeline results to a JSONL catalog + summary CSV."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        jsonl_path = out / f"alpha_pipeline_{ts}.jsonl"
        with open(jsonl_path, "w") as f:
            for r in self.reports:
                line = r.summary_dict()
                line["stages"] = [
                    {"stage": s.stage, "verdict": s.verdict.value,
                     "metrics": s.metrics, "detail": s.detail}
                    for s in r.stages
                ]
                f.write(json.dumps(line, default=str) + "\n")

        csv_path = out / f"alpha_pipeline_{ts}.csv"
        df = self.summary_table()
        if not df.empty:
            df.to_csv(csv_path, index=False)

        approved_path = out / f"approved_alphas_{ts}.txt"
        approved = [r.candidate.name for r in self.reports if r.passed]
        with open(approved_path, "w") as f:
            for name in approved:
                f.write(name + "\n")

        logger.info("Catalog written to %s", out)
        logger.info("  JSONL:    %s", jsonl_path.name)
        logger.info("  CSV:      %s", csv_path.name)
        logger.info("  Approved: %s (%d alphas)", approved_path.name, len(approved))

        return out

    def _log_stage(self, result: Any) -> None:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "⊘"}[result.verdict.value]
        logger.info(
            "  %s %s: %s  %s",
            icon, result.stage, result.verdict.value, result.detail,
        )

    def _log_summary(self) -> None:
        n = len(self.reports)
        passed = sum(1 for r in self.reports if r.passed)
        logger.info("═" * 60)
        logger.info("Pipeline complete: %d/%d candidates approved", passed, n)
        logger.info("Approved alphas: %s",
                     [r.candidate.name for r in self.reports if r.passed])

        fail_stages: dict[str, int] = {}
        for r in self.reports:
            if not r.passed and r.failed_stage:
                fail_stages[r.failed_stage] = fail_stages.get(r.failed_stage, 0) + 1
        if fail_stages:
            logger.info("Rejection breakdown: %s", fail_stages)
        logger.info("═" * 60)
