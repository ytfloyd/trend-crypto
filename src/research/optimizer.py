"""Parameter optimization with walk-forward validation.

Provides grid search with walk-forward splits and Deflated Sharpe Ratio
(Bailey & Lopez de Prado) correction for multiple testing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Optional

import polars as pl

from common.logging import get_logger

logger = get_logger("optimizer")


@dataclass(frozen=True)
class OptimizationResult:
    """Result of parameter optimization.

    Attributes:
        best_params: Best parameter combination.
        best_metric: Best metric value (e.g. Sharpe).
        all_results: All evaluated combinations with metrics.
        deflated_sharpe: Deflated Sharpe Ratio (corrected for multiple testing).
        n_trials: Number of parameter combinations tested.
    """

    best_params: dict[str, Any]
    best_metric: float
    all_results: list[dict[str, Any]]
    deflated_sharpe: Optional[float]
    n_trials: int


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_bars: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Corrects the observed Sharpe ratio for the number of trials
    (parameter combinations) tested.

    Args:
        observed_sharpe: The best Sharpe ratio from the grid search.
        n_trials: Number of parameter combinations tested.
        n_bars: Number of observations.
        skew: Skewness of returns (0 for normal).
        kurtosis: Kurtosis of returns (3 for normal).

    Returns:
        Deflated Sharpe Ratio (probability that the Sharpe is genuine).
    """
    if n_trials <= 1 or n_bars <= 1:
        return 1.0

    # Expected maximum Sharpe under the null (i.i.d. normal)
    # E[max(SR)] ≈ (1 - gamma) * z(1 - 1/N) + gamma * z(1 - 1/(N*e))
    # Simplified: E[max] ≈ sqrt(2 * log(N)) approximation
    e_max_sr = math.sqrt(2 * math.log(n_trials))

    # Standard error of Sharpe ratio
    sr_se = math.sqrt(
        (1 + 0.5 * observed_sharpe ** 2 - skew * observed_sharpe
         + ((kurtosis - 3) / 4) * observed_sharpe ** 2) / n_bars
    )

    if sr_se <= 0:
        return 0.0

    # Z-score: how many standard errors is the observed SR above expected max
    z = (observed_sharpe - e_max_sr) / sr_se

    # Convert to probability using normal CDF approximation
    return _norm_cdf(z)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t
             - 0.356563782) * t + 0.319381530) * t
    cdf = 1.0 - d * math.exp(-0.5 * x * x) * poly
    return cdf if x >= 0 else 1.0 - cdf


@dataclass
class WalkForwardSplit:
    """A single walk-forward train/test split.

    Attributes:
        train_start: Index of first training bar.
        train_end: Index of last training bar (inclusive).
        test_start: Index of first test bar.
        test_end: Index of last test bar (inclusive).
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_splits(
    n_bars: int,
    n_splits: int = 5,
    train_frac: float = 0.7,
    gap: int = 0,
) -> list[WalkForwardSplit]:
    """Generate walk-forward (expanding or rolling window) splits.

    Args:
        n_bars: Total number of bars.
        n_splits: Number of walk-forward periods.
        train_frac: Fraction of each period used for training.
        gap: Number of bars between train and test (purge gap).

    Returns:
        List of WalkForwardSplit.
    """
    if n_splits < 1 or n_bars < 10:
        return []

    period_size = n_bars // n_splits
    if period_size < 4:
        return []

    splits: list[WalkForwardSplit] = []
    for i in range(n_splits):
        period_start = i * period_size
        period_end = min((i + 1) * period_size - 1, n_bars - 1)
        train_size = int(train_frac * (period_end - period_start + 1))
        train_end = period_start + train_size - 1
        test_start = min(train_end + 1 + gap, period_end)
        if test_start > period_end:
            continue
        splits.append(WalkForwardSplit(
            train_start=period_start,
            train_end=train_end,
            test_start=test_start,
            test_end=period_end,
        ))

    return splits


class ParameterOptimizer:
    """Grid search with walk-forward validation.

    Args:
        bars: OHLCV DataFrame.
        evaluate_fn: Function that takes (bars_slice, params) and returns
            a metric value (e.g. Sharpe ratio).
        param_grid: Dict mapping parameter name → list of values.
        n_splits: Number of walk-forward splits.
        train_frac: Fraction of each split for training.
        gap: Purge gap between train and test.
    """

    def __init__(
        self,
        bars: pl.DataFrame,
        evaluate_fn: Callable[[pl.DataFrame, dict[str, Any]], float],
        param_grid: dict[str, list[Any]],
        n_splits: int = 5,
        train_frac: float = 0.7,
        gap: int = 0,
    ) -> None:
        self.bars = bars
        self.evaluate_fn = evaluate_fn
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.train_frac = train_frac
        self.gap = gap

    def optimize(self, metric_name: str = "sharpe") -> OptimizationResult:
        """Run grid search with walk-forward validation.

        For each parameter combination:
        1. For each walk-forward split, evaluate on train set.
        2. Average the metric across splits.
        3. Select the best combination.
        4. Evaluate best on all test sets for out-of-sample metric.

        Returns:
            OptimizationResult with best params and deflated Sharpe.
        """
        keys = sorted(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        combinations = list(product(*values))
        splits = walk_forward_splits(
            self.bars.height, self.n_splits, self.train_frac, self.gap,
        )

        if not splits:
            logger.warning("Not enough data for walk-forward splits")
            return OptimizationResult(
                best_params={}, best_metric=0.0,
                all_results=[], deflated_sharpe=None, n_trials=0,
            )

        all_results: list[dict[str, Any]] = []

        for combo in combinations:
            params = dict(zip(keys, combo))
            train_metrics: list[float] = []
            test_metrics: list[float] = []

            for split in splits:
                train_bars = self.bars.slice(
                    split.train_start, split.train_end - split.train_start + 1
                )
                test_bars = self.bars.slice(
                    split.test_start, split.test_end - split.test_start + 1
                )

                try:
                    train_metric = self.evaluate_fn(train_bars, params)
                    train_metrics.append(train_metric)
                except Exception:
                    train_metrics.append(0.0)

                try:
                    test_metric = self.evaluate_fn(test_bars, params)
                    test_metrics.append(test_metric)
                except Exception:
                    test_metrics.append(0.0)

            avg_train = sum(train_metrics) / len(train_metrics) if train_metrics else 0.0
            avg_test = sum(test_metrics) / len(test_metrics) if test_metrics else 0.0

            row: dict[str, Any] = dict(params)
            row["avg_train_metric"] = avg_train
            row["avg_test_metric"] = avg_test
            row["n_splits"] = len(splits)
            all_results.append(row)

        # Select best by average training metric
        best_result = max(all_results, key=lambda r: r.get("avg_train_metric", 0.0))
        best_params = {k: best_result[k] for k in keys}
        best_metric = best_result.get("avg_train_metric", 0.0)

        # Compute Deflated Sharpe Ratio
        dsr = deflated_sharpe_ratio(
            observed_sharpe=best_metric,
            n_trials=len(combinations),
            n_bars=self.bars.height,
        )

        logger.info(
            "Optimization complete: %d combos, best=%s (metric=%.4f, DSR=%.4f)",
            len(combinations), best_params, best_metric, dsr,
        )

        return OptimizationResult(
            best_params=best_params,
            best_metric=best_metric,
            all_results=all_results,
            deflated_sharpe=dsr,
            n_trials=len(combinations),
        )
