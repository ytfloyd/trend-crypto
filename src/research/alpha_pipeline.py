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

# ---------------------------------------------------------------------------
# Huber regression for robust linear signal fitting
# ---------------------------------------------------------------------------
HUBER_DELTA_DEFAULT = 1.345


def huber_regression(
    x: list[float] | pl.Series,
    y: list[float] | pl.Series,
    delta: float = HUBER_DELTA_DEFAULT,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Iteratively-reweighted least squares with Huber loss.

    Robust alternative to OLS for fat-tailed return data.  The default
    delta=1.345 gives 95% efficiency at the normal while strongly
    down-weighting outliers — the same convention used in the TSFM
    benchmarks of Rahimikia et al. (2025).

    Args:
        x: Independent variable (e.g. time index).
        y: Dependent variable (e.g. log prices or returns).
        delta: Huber threshold; residuals beyond ±delta are L1-penalised.
        max_iter: IRLS iterations.
        tol: Convergence tolerance on coefficient change.

    Returns:
        (slope, intercept)
    """
    if isinstance(x, pl.Series):
        x = x.to_list()
    if isinstance(y, pl.Series):
        y = y.to_list()

    n = len(x)
    if n < 3 or len(y) != n:
        return 0.0, 0.0

    # Initialise with OLS
    sx = sum(x)
    sy = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sx2 = sum(xi * xi for xi in x)
    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-15:
        return 0.0, 0.0

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    for _ in range(max_iter):
        residuals = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
        weights = [
            1.0 if abs(r) <= delta else delta / abs(r) for r in residuals
        ]

        wsum = sum(weights)
        if wsum < 1e-15:
            break
        wsx = sum(w * xi for w, xi in zip(weights, x))
        wsy = sum(w * yi for w, yi in zip(weights, y))
        wsxy = sum(w * xi * yi for w, xi, yi in zip(weights, x, y))
        wsx2 = sum(w * xi * xi for w, xi in zip(weights, x))

        d = wsum * wsx2 - wsx * wsx
        if abs(d) < 1e-15:
            break

        new_slope = (wsum * wsxy - wsx * wsy) / d
        new_intercept = (wsy - new_slope * wsx) / wsum

        if abs(new_slope - slope) + abs(new_intercept - intercept) < tol:
            slope, intercept = new_slope, new_intercept
            break
        slope, intercept = new_slope, new_intercept

    return slope, intercept


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
        r2_oos: Out-of-sample R² vs zero (Gu et al. 2020).  Positive means
            the signal's implicit prediction beats always-predict-zero.
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
    r2_oos: float
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

    from backtest.metrics import r2_oos_vs_zero  # deferred to avoid circular import

    r2_oos = r2_oos_vs_zero(df["ret"], df["sig"]) if df.height > 1 else 0.0

    logger.info(
        "Alpha '%s': IC=%.4f, IR=%.4f, hit=%.2f%%, turnover=%.4f, "
        "sharpe=%.2f, R2_oos=%.4f",
        name, ic_mean, ic_ir, hit_rate * 100, turnover, sharpe, r2_oos,
    )

    return AlphaResult(
        name=name,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        hit_rate=hit_rate,
        turnover=turnover,
        sharpe=sharpe,
        r2_oos=r2_oos,
        n_bars=df.height,
    )
