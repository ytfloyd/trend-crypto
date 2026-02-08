"""Correlation regime detection using EWMA-based Dynamic Conditional Correlation.

Provides a simplified DCC estimator (Engle 2002) using EWMA rather than
the full GARCH-based DCC, avoiding numpy/scipy dependencies.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import polars as pl

from common.logging import get_logger

logger = get_logger("regime")


class RegimeState(str, Enum):
    """Correlation regime classification."""

    CRISIS = "crisis"
    NORMAL = "normal"
    DIVERSIFIED = "diversified"


def ewma_volatility(
    returns: list[float], halflife: int = 30
) -> list[float]:
    """Compute EWMA volatility (standard deviation) from returns.

    Args:
        returns: Return series.
        halflife: EWMA halflife in bars.

    Returns:
        List of EWMA volatilities (same length as returns, first = 0).
    """
    if not returns:
        return []
    alpha = 1 - math.exp(-math.log(2) / halflife)
    var_ewma = 0.0
    vols: list[float] = [0.0]
    for i in range(1, len(returns)):
        var_ewma = (1 - alpha) * var_ewma + alpha * returns[i] ** 2
        vols.append(math.sqrt(max(0.0, var_ewma)))
    return vols


def ewma_correlation(
    returns_a: list[float],
    returns_b: list[float],
    halflife: int = 30,
) -> list[float]:
    """Compute EWMA rolling correlation between two return series.

    Args:
        returns_a: Return series for asset A.
        returns_b: Return series for asset B.
        halflife: EWMA halflife in bars.

    Returns:
        List of EWMA correlations (same length, first = 0).
    """
    n = min(len(returns_a), len(returns_b))
    if n < 2:
        return [0.0] * n
    alpha = 1 - math.exp(-math.log(2) / halflife)
    cov_ewma = 0.0
    var_a = 0.0
    var_b = 0.0
    corrs: list[float] = [0.0]
    for i in range(1, n):
        cov_ewma = (1 - alpha) * cov_ewma + alpha * returns_a[i] * returns_b[i]
        var_a = (1 - alpha) * var_a + alpha * returns_a[i] ** 2
        var_b = (1 - alpha) * var_b + alpha * returns_b[i] ** 2
        denom = math.sqrt(max(0.0, var_a)) * math.sqrt(max(0.0, var_b))
        if denom > 0:
            corrs.append(max(-1.0, min(1.0, cov_ewma / denom)))
        else:
            corrs.append(0.0)
    return corrs


def rolling_dcc_correlation(
    returns_by_symbol: dict[str, list[float]],
    halflife: int = 30,
) -> list[dict[str, dict[str, float]]]:
    """Simplified DCC correlation matrix estimation using EWMA.

    Returns one correlation matrix per bar (after warmup).

    Args:
        returns_by_symbol: Symbol → list of returns.
        halflife: EWMA halflife for correlation estimation.

    Returns:
        List of correlation matrices (nested dicts).
    """
    symbols = sorted(returns_by_symbol.keys())
    if len(symbols) < 2:
        return []

    n = min(len(v) for v in returns_by_symbol.values())
    if n < 2:
        return []

    # Pre-compute pairwise EWMA correlations
    pairwise: dict[tuple[str, str], list[float]] = {}
    for i, si in enumerate(symbols):
        for j, sj in enumerate(symbols):
            if i < j:
                corrs = ewma_correlation(
                    returns_by_symbol[si], returns_by_symbol[sj], halflife
                )
                pairwise[(si, sj)] = corrs

    matrices: list[dict[str, dict[str, float]]] = []
    for t in range(n):
        matrix: dict[str, dict[str, float]] = {}
        for si in symbols:
            matrix[si] = {}
            for sj in symbols:
                if si == sj:
                    matrix[si][sj] = 1.0
                elif (si, sj) in pairwise:
                    matrix[si][sj] = pairwise[(si, sj)][t]
                else:
                    matrix[si][sj] = pairwise[(sj, si)][t]
        matrices.append(matrix)

    return matrices


def detect_correlation_regime(
    returns_by_symbol: dict[str, list[float]],
    halflife: int = 30,
    crisis_threshold: float = 0.7,
    diversified_threshold: float = 0.2,
) -> Optional[RegimeState]:
    """Classify current correlation regime from the latest DCC estimate.

    Args:
        returns_by_symbol: Symbol → list of returns.
        halflife: EWMA halflife.
        crisis_threshold: Average correlation above this → CRISIS.
        diversified_threshold: Average correlation below this → DIVERSIFIED.

    Returns:
        RegimeState or None if insufficient data.
    """
    matrices = rolling_dcc_correlation(returns_by_symbol, halflife)
    if not matrices:
        return None
    latest = matrices[-1]
    symbols = sorted(latest.keys())
    total = 0.0
    count = 0
    for i, si in enumerate(symbols):
        for j, sj in enumerate(symbols):
            if i < j:
                total += latest[si][sj]
                count += 1
    if count == 0:
        return None
    avg_corr = total / count
    if avg_corr >= crisis_threshold:
        return RegimeState.CRISIS
    if avg_corr <= diversified_threshold:
        return RegimeState.DIVERSIFIED
    return RegimeState.NORMAL
