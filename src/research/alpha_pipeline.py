"""Alpha research pipeline.

Standardized workflow for evaluating alpha signals: compute → IC analysis →
backtest → capacity estimate. Wraps existing alpha factory computations into
a formalized, repeatable pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from common.logging import get_logger

logger = get_logger("alpha_pipeline")


@dataclass(frozen=True)
class AlphaResult:
    """Result of evaluating a single alpha signal.

    Attributes:
        name: Alpha signal name.
        ic_mean: Mean information coefficient (rank correlation with forward returns).
        ic_std: Standard deviation of IC.
        ic_ir: IC information ratio (ic_mean / ic_std).
        hit_rate: Fraction of bars where signal direction matched return direction.
        turnover: Average per-bar turnover of the signal.
        sharpe: Annualized Sharpe ratio of the signal-weighted returns.
        n_bars: Number of bars evaluated.
        metadata: Additional metadata.
    """

    name: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    hit_rate: float
    turnover: float
    sharpe: float
    n_bars: int
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_ic(
    signal: pl.Series, forward_returns: pl.Series
) -> tuple[float, float]:
    """Compute mean and std of rank information coefficient.

    Uses Spearman rank correlation approximated via Pearson on ranks.

    Args:
        signal: Alpha signal values.
        forward_returns: Forward returns aligned to signals.

    Returns:
        (mean_ic, std_ic)
    """
    if signal.len() < 10 or forward_returns.len() < 10:
        return 0.0, 0.0

    # Rank both series
    sig_rank = signal.rank()
    ret_rank = forward_returns.rank()

    # Pearson correlation on ranks = Spearman correlation
    df = pl.DataFrame({"sig": sig_rank, "ret": ret_rank}).drop_nulls()
    if df.height < 10:
        return 0.0, 0.0

    corr_val = df.select(pl.corr("sig", "ret")).item()
    ic = float(corr_val) if corr_val is not None else 0.0

    # For a single cross-section, IC std is approximated
    # In multi-period case, compute rolling IC and take std
    ic_std = 1.0 / math.sqrt(max(1, df.height))

    return ic, ic_std


def compute_hit_rate(signal: pl.Series, forward_returns: pl.Series) -> float:
    """Fraction of bars where signal and return have the same sign.

    Args:
        signal: Alpha signal.
        forward_returns: Forward returns.
    """
    df = pl.DataFrame({"sig": signal, "ret": forward_returns}).drop_nulls()
    if df.height == 0:
        return 0.0
    same_sign = df.filter(
        (pl.col("sig") > 0) & (pl.col("ret") > 0)
        | (pl.col("sig") < 0) & (pl.col("ret") < 0)
    ).height
    return same_sign / df.height


def compute_signal_turnover(signal: pl.Series) -> float:
    """Average absolute change in signal per bar."""
    if signal.len() < 2:
        return 0.0
    changes = signal.diff().drop_nulls().abs()
    mean_val = changes.mean()
    return float(mean_val) if mean_val is not None else 0.0  # type: ignore[arg-type]


def evaluate_alpha(
    name: str,
    signal: pl.Series,
    forward_returns: pl.Series,
    periods_per_year: float = 8760.0,
) -> AlphaResult:
    """Full alpha evaluation pipeline.

    Args:
        name: Alpha signal name.
        signal: Signal values (higher = more bullish).
        forward_returns: 1-bar forward returns aligned to signal.
        periods_per_year: Annualization factor.

    Returns:
        AlphaResult with all computed metrics.
    """
    ic_mean, ic_std = compute_ic(signal, forward_returns)
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit_rate = compute_hit_rate(signal, forward_returns)
    turnover = compute_signal_turnover(signal)

    # Signal-weighted returns for Sharpe
    df = pl.DataFrame({"sig": signal, "ret": forward_returns}).drop_nulls()
    if df.height > 1:
        weighted_ret = (df["sig"] * df["ret"])
        mean_wr = float(weighted_ret.mean() or 0.0)  # type: ignore[arg-type]
        std_wr = float(weighted_ret.std(ddof=1) or 0.0)  # type: ignore[arg-type]
        sharpe = (mean_wr / std_wr) * math.sqrt(periods_per_year) if std_wr > 0 else 0.0
    else:
        sharpe = 0.0

    logger.info(
        "Alpha '%s': IC=%.4f, IR=%.4f, hit=%.2f%%, turnover=%.4f, sharpe=%.2f",
        name, ic_mean, ic_ir, hit_rate * 100, turnover, sharpe,
    )

    return AlphaResult(
        name=name,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        hit_rate=hit_rate,
        turnover=turnover,
        sharpe=sharpe,
        n_bars=df.height,
    )
