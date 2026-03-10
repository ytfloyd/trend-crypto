"""
Backtesting statistics: detecting overfitting.

Implements tools from AFML Chapter 11 (and Bailey/de Prado papers):

  - **Deflated Sharpe Ratio (DSR)**: adjusts the observed Sharpe for
    the number of trials conducted (selection bias), non-normality
    (skewness, kurtosis), and sample length.

  - **Probability of Backtest Overfitting (PBO)**: given multiple
    backtest paths (from CPCV), estimates the probability that the
    best in-sample strategy will underperform OOS.

  - **Expected maximum Sharpe (Haircut)**: the expected maximum Sharpe
    among N i.i.d. strategies — the benchmark against which the
    observed Sharpe should be compared.

Reference:
    Bailey, D.H. & López de Prado, M. (2014)
    "The Deflated Sharpe Ratio", J. Portfolio Management.
    López de Prado, M. (2018) AFML Chapter 11.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


# =====================================================================
# Expected maximum Sharpe  (AFML eq. 11.3)
# =====================================================================

def expected_max_sharpe(
    n_trials: int,
    mean_sharpe: float = 0.0,
    std_sharpe: float = 1.0,
) -> float:
    """Expected maximum Sharpe among ``n_trials`` independent strategies.

    Under the null that all strategies have true Sharpe = 0 (and
    estimated Sharpes are i.i.d. Normal), the expected maximum
    observed Sharpe grows as ~sqrt(2 * log(N)).

    This is the benchmark: your observed Sharpe must exceed this
    to be evidence of genuine skill.

    Parameters
    ----------
    n_trials : int
        Number of strategies / parameter combinations tried.
    mean_sharpe : float
        Mean of the null distribution of Sharpe estimates (usually 0).
    std_sharpe : float
        Std of the null distribution.

    Returns
    -------
    float — E[max(SR)] under the null.
    """
    if n_trials <= 0:
        return 0.0
    emc = 0.5772156649  # Euler-Mascheroni constant
    z = norm.ppf(1 - 1.0 / n_trials) if n_trials > 1 else 0.0
    e_max = mean_sharpe + std_sharpe * (
        z * (1 - emc) + emc * norm.ppf(1 - 1.0 / (n_trials * np.e))
        if n_trials > 1
        else 0.0
    )
    return e_max


# =====================================================================
# Deflated Sharpe Ratio  (AFML Snippet 11.2)
# =====================================================================

def deflated_sharpe_ratio(
    observed_sr: float,
    sr_benchmark: float,
    n_obs: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio deflated by selection bias.

    Computes the probability that the *true* Sharpe exceeds the
    benchmark, accounting for non-normality and sample size.

    Parameters
    ----------
    observed_sr : float
        Observed (annualised) Sharpe ratio of the best strategy.
    sr_benchmark : float
        Benchmark Sharpe — typically from ``expected_max_sharpe()``.
    n_obs : int
        Number of return observations.
    skewness : float
        Sample skewness of returns.
    excess_kurtosis : float
        Sample excess kurtosis of returns.

    Returns
    -------
    float — p-value in [0, 1].  High values → the Sharpe is likely genuine.
    """
    # Lo (2002) / Bailey & de Prado (2014) standard error of SR
    var = (1 - skewness * observed_sr + (excess_kurtosis + 2) / 4 * observed_sr**2) / (n_obs - 1)
    sr_std = np.sqrt(max(var, 1e-12))
    if sr_std <= 0:
        return 0.0

    z = (observed_sr - sr_benchmark) / sr_std
    return float(norm.cdf(z))


# =====================================================================
# Probability of Backtest Overfitting  (AFML / Bailey & de Prado)
# =====================================================================

def probability_of_backtest_overfitting(
    is_scores: np.ndarray,
    oos_scores: np.ndarray,
) -> dict[str, float]:
    """Estimate the Probability of Backtest Overfitting (PBO).

    Given paired in-sample and out-of-sample scores from CPCV
    (one pair per path), PBO measures how often the best IS strategy
    underperforms OOS.

    Parameters
    ----------
    is_scores : np.ndarray of shape (n_paths,)
        In-sample performance for each CPCV path.
    oos_scores : np.ndarray of shape (n_paths,)
        Corresponding out-of-sample performance.

    Returns
    -------
    dict with keys:
        pbo : float — P(overfit) ∈ [0, 1]
        rank_corr : float — Spearman correlation between IS and OOS ranks
        best_is_oos : float — OOS score of the best IS path
    """
    from scipy.stats import spearmanr

    n = len(is_scores)
    if n == 0:
        return {"pbo": 1.0, "rank_corr": 0.0, "best_is_oos": np.nan}

    # For each path, check if the best-IS path is below median OOS
    best_is_idx = np.argmax(is_scores)
    best_is_oos = oos_scores[best_is_idx]

    # Lambda_c: relative rank of best-IS in OOS
    oos_rank = (oos_scores < best_is_oos).sum() / n
    pbo = 1 - oos_rank  # P(overfit) = 1 - relative rank

    rank_corr = spearmanr(is_scores, oos_scores).statistic

    return {
        "pbo": float(pbo),
        "rank_corr": float(rank_corr),
        "best_is_oos": float(best_is_oos),
    }


# =====================================================================
# Sharpe ratio helpers
# =====================================================================

def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def return_stats(returns: pd.Series) -> dict[str, float | int]:
    """Compute return statistics needed for DSR."""
    return {
        "sharpe": sharpe_ratio(returns),
        "n_obs": len(returns),
        "skewness": float(skew(returns.dropna())),
        "excess_kurtosis": float(kurtosis(returns.dropna())),
    }
