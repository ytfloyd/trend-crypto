"""Factor risk attribution using rolling OLS decomposition.

Decomposes portfolio returns into factor exposures (beta) and idiosyncratic
components without numpy/scipy by implementing simple OLS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FactorAttribution:
    """Result of factor risk attribution for a single window.

    Attributes:
        factor_betas: Factor name → beta exposure.
        r_squared: Fraction of variance explained by factors.
        residual_vol: Annualized idiosyncratic volatility.
        factor_contributions: Factor name → fraction of total risk.
    """

    factor_betas: dict[str, float]
    r_squared: float
    residual_vol: float
    factor_contributions: dict[str, float]


def _simple_ols(y: list[float], x: list[float]) -> tuple[float, float, float]:
    """Simple OLS regression: y = alpha + beta * x.

    Returns (alpha, beta, r_squared).
    """
    n = len(y)
    if n < 3 or len(x) != n:
        return 0.0, 0.0, 0.0

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-15:
        return 0.0, 0.0, 0.0

    beta = (n * sum_xy - sum_x * sum_y) / denom
    alpha = (sum_y - beta * sum_x) / n

    # R-squared
    mean_y = sum_y / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - alpha - beta * xi) ** 2 for xi, yi in zip(x, y))
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return alpha, beta, max(0.0, r_sq)


def _multi_ols(
    y: list[float], factors: dict[str, list[float]]
) -> tuple[dict[str, float], float]:
    """Multi-factor OLS using iterative single-factor regression.

    This is an approximation (not true multivariate OLS) that works
    reasonably well when factors are not too correlated. Avoids matrix
    inversion without numpy.

    Returns (betas, r_squared).
    """
    if not factors:
        return {}, 0.0

    # Simple sequential approach: regress on each factor independently
    betas: dict[str, float] = {}
    residuals = list(y)

    for name, factor_vals in factors.items():
        _, beta, r_sq = _simple_ols(residuals, factor_vals)
        betas[name] = beta
        # Update residuals
        residuals = [r - beta * f for r, f in zip(residuals, factor_vals)]

    # Compute overall R-squared from original y
    n = len(y)
    mean_y = sum(y) / n if n > 0 else 0.0
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum(r * r for r in residuals)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return betas, max(0.0, r_squared)


def factor_risk_attribution(
    portfolio_returns: list[float],
    factor_returns: dict[str, list[float]],
    periods_per_year: float = 8760.0,
) -> FactorAttribution:
    """Compute factor risk attribution for portfolio returns.

    Args:
        portfolio_returns: Portfolio return series.
        factor_returns: Factor name → return series (same length).
        periods_per_year: Annualization factor.

    Returns:
        FactorAttribution with betas, R², residual vol, and factor contributions.
    """
    n = len(portfolio_returns)
    if n < 3:
        return FactorAttribution(
            factor_betas={}, r_squared=0.0,
            residual_vol=0.0, factor_contributions={},
        )

    betas, r_squared = _multi_ols(portfolio_returns, factor_returns)

    # Compute residuals
    residuals = list(portfolio_returns)
    for name, beta in betas.items():
        factor_vals = factor_returns[name]
        residuals = [r - beta * f for r, f in zip(residuals, factor_vals)]

    # Residual volatility (annualized)
    mean_resid = sum(residuals) / n
    resid_var = sum((r - mean_resid) ** 2 for r in residuals) / (n - 1) if n > 1 else 0.0
    residual_vol = math.sqrt(max(0.0, resid_var)) * math.sqrt(periods_per_year)

    # Factor contributions to total variance
    mean_port = sum(portfolio_returns) / n
    port_var = sum((r - mean_port) ** 2 for r in portfolio_returns) / (n - 1) if n > 1 else 0.0

    factor_contributions: dict[str, float] = {}
    if port_var > 0:
        for name, beta in betas.items():
            factor_vals = factor_returns[name]
            mean_f = sum(factor_vals) / n
            factor_var = sum((f - mean_f) ** 2 for f in factor_vals) / (n - 1) if n > 1 else 0.0
            factor_contributions[name] = (beta ** 2 * factor_var) / port_var
    else:
        factor_contributions = {name: 0.0 for name in betas}

    return FactorAttribution(
        factor_betas=betas,
        r_squared=r_squared,
        residual_vol=residual_vol,
        factor_contributions=factor_contributions,
    )
