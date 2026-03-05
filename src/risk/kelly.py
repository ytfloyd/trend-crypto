"""Empirical Kelly criterion for growth-optimal leverage sizing.

Pure function: takes a return series, outputs the optimal leverage fraction.
No side effects, no shared state. Sits above vol_target.py as a portfolio-level
leverage cap.

Reference:
    Max Dama, "Automated Trading", Section 6.4 — Empirical Kelly Code.
    The Gaussian approximation mu/sigma^2 underestimates tail risk in crypto.
    Empirical Kelly uses the actual return distribution (fat tails, skew).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def empirical_kelly(
    returns: np.ndarray,
    max_leverage: float = 5.0,
) -> float:
    """Compute the growth-optimal leverage fraction from realized returns.

    Maximizes E[log(1 + f * r)] over the empirical distribution of returns,
    where f is the leverage fraction and r is a single-period return.

    Parameters
    ----------
    returns : np.ndarray
        Array of arithmetic single-period returns (e.g. daily).
    max_leverage : float
        Upper bound on the search range for f.  Defaults to 5.0.

    Returns
    -------
    float
        Optimal leverage fraction.  Values > 1.0 imply leveraging up;
        values < 1.0 imply holding cash.  Returns 0.0 if the strategy
        has negative expected return.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 10:
        return 0.0

    if np.mean(returns) <= 0:
        return 0.0

    def neg_expected_log_growth(f: float) -> float:
        growth = np.log1p(f * returns)
        valid = np.isfinite(growth)
        if valid.sum() < len(returns) * 0.9:
            return 1e10
        return -float(np.mean(growth[valid]))

    result = minimize_scalar(
        neg_expected_log_growth,
        bounds=(0.0, max_leverage),
        method="bounded",
    )

    if not result.success or result.fun >= 0:
        return 0.0

    return float(result.x)


def gaussian_kelly(returns: np.ndarray) -> float:
    """Continuous/Gaussian Kelly approximation: mu / sigma^2.

    Useful as a comparison benchmark. In crypto with heavy left tails,
    empirical Kelly should be materially lower than this value.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 10:
        return 0.0
    mu = np.mean(returns)
    var = np.var(returns, ddof=1)
    if var <= 0 or mu <= 0:
        return 0.0
    return float(mu / var)


def apply_kelly_cap(
    weights: dict[str, float],
    kelly_fraction: float,
) -> dict[str, float]:
    """Scale portfolio weights down if implied leverage exceeds Kelly fraction.

    Parameters
    ----------
    weights : dict[str, float]
        Symbol -> target weight.
    kelly_fraction : float
        Maximum allowed gross leverage from Kelly criterion.

    Returns
    -------
    dict[str, float]
        Scaled weights. If gross leverage <= kelly_fraction, returned unchanged.
    """
    if kelly_fraction <= 0:
        return {k: 0.0 for k in weights}

    gross = sum(abs(w) for w in weights.values())
    if gross <= kelly_fraction:
        return dict(weights)

    scale = kelly_fraction / gross
    return {k: v * scale for k, v in weights.items()}
