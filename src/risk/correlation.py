"""Cross-asset correlation analysis for portfolio risk management."""
from __future__ import annotations

from enum import Enum
from typing import Optional

import polars as pl


class CorrelationRegime(str, Enum):
    """Classification of the correlation regime across assets."""

    CRISIS = "crisis"
    NORMAL = "normal"
    DIVERSIFIED = "diversified"


def rolling_correlation_matrix(
    returns_by_symbol: dict[str, pl.Series],
    window: int,
) -> list[dict[str, dict[str, float]]]:
    """Compute rolling pairwise correlation matrices.

    Args:
        returns_by_symbol: Symbol → return series (all same length).
        window: Rolling window size in bars.

    Returns:
        List of correlation matrices (one per bar after the warmup period).
        Each matrix is a nested dict: corr[symbol_i][symbol_j].
    """
    symbols = sorted(returns_by_symbol.keys())
    if len(symbols) < 2:
        return []

    # Build a combined DataFrame for rolling correlation
    combined = pl.DataFrame({
        sym: returns_by_symbol[sym] for sym in symbols
    })
    n = combined.height
    if n < window:
        return []

    matrices: list[dict[str, dict[str, float]]] = []
    for t in range(window - 1, n):
        start = t - window + 1
        window_df = combined.slice(start, window)
        matrix: dict[str, dict[str, float]] = {}
        for si in symbols:
            matrix[si] = {}
            for sj in symbols:
                if si == sj:
                    matrix[si][sj] = 1.0
                else:
                    corr_val = window_df.select(
                        pl.corr(si, sj)
                    ).item()
                    matrix[si][sj] = float(corr_val) if corr_val is not None else 0.0
        matrices.append(matrix)

    return matrices


def average_correlation(corr_matrix: dict[str, dict[str, float]]) -> float:
    """Compute average pairwise correlation from a correlation matrix.

    Args:
        corr_matrix: Nested dict corr[i][j] for all symbol pairs.

    Returns:
        Average off-diagonal correlation.
    """
    symbols = sorted(corr_matrix.keys())
    if len(symbols) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, si in enumerate(symbols):
        for j, sj in enumerate(symbols):
            if i < j:
                total += corr_matrix.get(si, {}).get(sj, 0.0)
                count += 1
    return total / count if count > 0 else 0.0


def correlation_regime_indicator(
    returns_by_symbol: dict[str, pl.Series],
    window: int = 60,
    crisis_threshold: float = 0.7,
    diversified_threshold: float = 0.2,
) -> Optional[CorrelationRegime]:
    """Classify current correlation regime based on recent rolling correlation.

    Args:
        returns_by_symbol: Symbol → return series.
        window: Rolling window in bars for correlation estimation.
        crisis_threshold: Average correlation above this → CRISIS.
        diversified_threshold: Average correlation below this → DIVERSIFIED.

    Returns:
        CorrelationRegime or None if insufficient data.
    """
    matrices = rolling_correlation_matrix(returns_by_symbol, window)
    if not matrices:
        return None
    latest = matrices[-1]
    avg_corr = average_correlation(latest)
    if avg_corr >= crisis_threshold:
        return CorrelationRegime.CRISIS
    if avg_corr <= diversified_threshold:
        return CorrelationRegime.DIVERSIFIED
    return CorrelationRegime.NORMAL
