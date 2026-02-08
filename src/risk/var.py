"""Portfolio Value-at-Risk (VaR) computation.

Provides historical simulation, parametric (Gaussian), and component VaR
using Euler decomposition.
"""
from __future__ import annotations

import math
from typing import Optional

import polars as pl


def historical_var(
    portfolio_returns: pl.Series,
    confidence: float = 0.95,
) -> float:
    """Historical simulation VaR.

    Args:
        portfolio_returns: Portfolio return series.
        confidence: Confidence level (e.g. 0.95 for 95% VaR).

    Returns:
        VaR as a positive number (loss magnitude at the given confidence).
    """
    if portfolio_returns.len() < 2:
        return 0.0
    quantile = 1 - confidence
    var_val = portfolio_returns.quantile(quantile, interpolation="linear")
    if var_val is None:
        return 0.0
    return -float(var_val)  # type: ignore[arg-type]


def parametric_var(
    weights: dict[str, float],
    cov_matrix: dict[str, dict[str, float]],
    confidence: float = 0.95,
    holding_period: float = 1.0,
) -> float:
    """Parametric (Gaussian) VaR using the covariance matrix.

    Args:
        weights: Symbol → weight mapping.
        cov_matrix: Nested dict of covariances, cov_matrix[i][j].
        confidence: Confidence level.
        holding_period: Holding period in bars (for scaling).

    Returns:
        VaR as a positive number.
    """
    symbols = sorted(weights.keys())
    if not symbols:
        return 0.0

    # Compute portfolio variance: w' * Sigma * w
    port_var = 0.0
    for si in symbols:
        for sj in symbols:
            wi = weights.get(si, 0.0)
            wj = weights.get(sj, 0.0)
            cov_ij = cov_matrix.get(si, {}).get(sj, 0.0)
            port_var += wi * wj * cov_ij

    port_vol = math.sqrt(max(0.0, port_var))

    # Z-score for Gaussian VaR
    z = _norm_ppf(confidence)
    return z * port_vol * math.sqrt(holding_period)


def component_var(
    weights: dict[str, float],
    cov_matrix: dict[str, dict[str, float]],
    confidence: float = 0.95,
) -> dict[str, float]:
    """Component VaR via Euler decomposition.

    Each component VaR represents the marginal contribution of an asset
    to the total portfolio VaR. The sum of component VaRs equals total VaR.

    Args:
        weights: Symbol → weight mapping.
        cov_matrix: Nested dict of covariances.
        confidence: Confidence level.

    Returns:
        Dict mapping symbol → component VaR contribution.
    """
    symbols = sorted(weights.keys())
    if not symbols:
        return {}

    total_var = parametric_var(weights, cov_matrix, confidence)
    if total_var <= 0:
        return {s: 0.0 for s in symbols}

    # Portfolio variance for denominator
    port_var = 0.0
    for si in symbols:
        for sj in symbols:
            wi = weights.get(si, 0.0)
            wj = weights.get(sj, 0.0)
            cov_ij = cov_matrix.get(si, {}).get(sj, 0.0)
            port_var += wi * wj * cov_ij

    if port_var <= 0:
        return {s: 0.0 for s in symbols}

    port_vol = math.sqrt(port_var)

    # Component VaR_i = w_i * sum_j(w_j * cov_ij) / port_vol * z
    z = _norm_ppf(confidence)
    result: dict[str, float] = {}
    for si in symbols:
        wi = weights.get(si, 0.0)
        marginal = 0.0
        for sj in symbols:
            wj = weights.get(sj, 0.0)
            cov_ij = cov_matrix.get(si, {}).get(sj, 0.0)
            marginal += wj * cov_ij
        result[si] = wi * marginal / port_vol * z

    return result


def _norm_ppf(confidence: float) -> float:
    """Approximate inverse CDF of standard normal using rational approximation.

    Accurate to ~1e-4 for confidence in [0.5, 0.999].
    Avoids scipy dependency.
    """
    if confidence <= 0.5:
        return 0.0
    if confidence >= 0.9999:
        confidence = 0.9999

    # Beasley-Springer-Moro approximation
    p = confidence
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    # Rational approximation coefficients
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
