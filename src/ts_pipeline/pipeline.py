"""Time-Series Alpha Pipeline orchestrator.

Evaluates trend-following signals through a 7-stage sequential fast-fail
pipeline, mirroring how AHL, Systematica, and Transtrend validate signals:

  1. Per-asset time-series IC (does the signal predict each asset's own returns?)
  2. Signal persistence (is the signal slow enough to trade?)
  3. IC horizon profile (does predictive power span multiple horizons?)
  4. Vol-targeted portfolio backtest (does it make money after costs?)
  5. Walk-forward validation — CPCV + PBO (is the edge real OOS?)
  6. Deflated Sharpe ratio (does it survive multiple-testing correction?)
  7. Blend diversification (informational — how correlated is it with approved signals?)

Stages 1-6 are hard gates.  A candidate that fails any gate is rejected
immediately (fast-fail).  Stage 7 is informational only.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from src.alpha_pipeline.types import StageResult, StageVerdict
from src.ts_pipeline.types import TSCandidate, TSGateConfig, TSPipelineReport
from src.ts_pipeline.stages import (
    stage_ts_ic,
    stage_persistence,
    stage_horizon_profile,
    stage_portfolio_backtest,
    stage_walk_forward,
    stage_deflated_sharpe,
    stage_blend_diversification,
)

logger = logging.getLogger(__name__)


class TSAlphaPipeline:
    """Orchestrates time-series alpha evaluation through 7 sequential stages.

    Parameters
    ----------
    close : (ts x symbol) wide DataFrame of close prices.
    high : (ts x symbol) wide DataFrame of high prices.
    low : (ts x symbol) wide DataFrame of low prices.
    volume : (ts x symbol) wide DataFrame of volume.
    returns : (ts x symbol) wide DataFrame of daily returns.
    config : TSGateConfig with all gate thresholds.
    """

    def __init__(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        volume: pd.DataFrame,
        returns: pd.DataFrame,
        config: TSGateConfig | None = None,
    ) -> None:
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.returns = returns
        self.config = config or TSGateConfig()
        self._approved: dict[str, pd.DataFrame] = {}
        self._reports: list[TSPipelineReport] = []
        self._n_evaluated: int = 0

    @property
    def reports(self) -> list[TSPipelineReport]:
        return list(self._reports)

    @property
    def approved_signals(self) -> dict[str, pd.DataFrame]:
        return dict(self._approved)

    def evaluate(self, candidate: TSCandidate) -> TSPipelineReport:
        """Evaluate a single candidate through all 7 stages.

        Returns a TSPipelineReport with per-stage results.
        """
        self._n_evaluated += 1
        report = TSPipelineReport(candidate=candidate)

        t0 = time.time()
        logger.info("TS Pipeline: evaluating '%s' (%s)", candidate.name, candidate.family)

        # Compute signal scores
        try:
            forecasts = candidate.get_scores(
                self.close, self.high, self.low, self.volume, self.returns,
            )
        except Exception as e:
            logger.error("Failed to compute scores for '%s': %s", candidate.name, e)
            report.stages.append(StageResult(
                stage="compute", verdict=StageVerdict.FAIL,
                metrics={}, detail=f"Error computing scores: {e}",
            ))
            self._reports.append(report)
            return report

        cfg = self.config

        # ── Stage 1: Per-asset TS IC ──
        result = stage_ts_ic(forecasts, self.returns, cfg)
        report.stages.append(result)
        logger.info("  Stage 1 (TS IC): %s — %s", result.verdict.value, result.detail)
        if result.verdict == StageVerdict.FAIL:
            self._reports.append(report)
            return report

        # ── Stage 2: Signal Persistence ──
        result = stage_persistence(forecasts, cfg)
        report.stages.append(result)
        logger.info("  Stage 2 (Persistence): %s — %s", result.verdict.value, result.detail)
        if result.verdict == StageVerdict.FAIL:
            self._reports.append(report)
            return report

        # ── Stage 3: IC Horizon Profile ──
        result = stage_horizon_profile(forecasts, self.returns, cfg)
        report.stages.append(result)
        logger.info("  Stage 3 (Horizon): %s — %s", result.verdict.value, result.detail)
        if result.verdict == StageVerdict.FAIL:
            self._reports.append(report)
            return report

        # ── Stage 4: Vol-Targeted Portfolio Backtest ──
        result = stage_portfolio_backtest(forecasts, self.returns, cfg)
        report.stages.append(result)
        logger.info("  Stage 4 (Portfolio): %s — %s", result.verdict.value, result.detail)
        if result.verdict == StageVerdict.FAIL:
            self._reports.append(report)
            return report

        # ── Stage 5: Walk-Forward (CPCV + PBO) ──
        result = stage_walk_forward(forecasts, self.returns, cfg)
        report.stages.append(result)
        logger.info("  Stage 5 (Walk-Forward): %s — %s", result.verdict.value, result.detail)
        if result.verdict == StageVerdict.FAIL:
            self._reports.append(report)
            return report

        # ── Stage 6: Deflated Sharpe Ratio ──
        result = stage_deflated_sharpe(
            forecasts, self.returns, cfg, self._n_evaluated,
        )
        report.stages.append(result)
        logger.info("  Stage 6 (DSR): %s — %s", result.verdict.value, result.detail)
        if result.verdict == StageVerdict.FAIL:
            self._reports.append(report)
            return report

        # ── Stage 7: Blend Diversification (informational) ──
        result = stage_blend_diversification(
            forecasts, self._approved, self.returns, cfg, candidate.name,
        )
        report.stages.append(result)
        logger.info("  Stage 7 (Blend): %s — %s", result.verdict.value, result.detail)

        # All gates passed — add to approved set
        self._approved[candidate.name] = forecasts
        elapsed = time.time() - t0
        logger.info(
            "  ✓ '%s' APPROVED in %.1fs (%d total approved)",
            candidate.name, elapsed, len(self._approved),
        )

        self._reports.append(report)
        return report

    def evaluate_batch(self, candidates: list[TSCandidate]) -> list[TSPipelineReport]:
        """Evaluate a batch of candidates sequentially."""
        reports = []
        for i, c in enumerate(candidates, 1):
            logger.info("━━━ Candidate %d/%d: %s ━━━", i, len(candidates), c.name)
            report = self.evaluate(c)
            reports.append(report)
        return reports

    def summary_df(self) -> pd.DataFrame:
        """Return a summary DataFrame of all evaluated candidates."""
        rows = [r.summary_dict() for r in self._reports]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
