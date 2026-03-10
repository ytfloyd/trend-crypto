"""
Fractionally differentiated features.

Implements the fixed-width window fractional differentiation from AFML Chapter 5.
The key insight: integer differencing (d=1, i.e. log returns) achieves
stationarity but destroys memory.  Fractional differencing with d < 1
preserves long-range dependence while achieving stationarity.

The goal is to find the minimum *d* that makes a series stationary
(ADF test rejects the unit root) — this retains maximum memory.

Reference:
    López de Prado, M. (2018) *Advances in Financial Machine Learning*,
    Chapter 5: Fractionally Differentiated Features (pp. 87–107).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Fractional differencing weights  (AFML Snippet 5.1)
# -----------------------------------------------------------------------

def frac_diff_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Compute the weights for fractional differencing of order *d*.

    Weights decay as: w_k = -w_{k-1} * (d - k + 1) / k
    We truncate when |w_k| < threshold (fixed-width window approach).

    Parameters
    ----------
    d : float
        Fractional differencing order. 0 < d < 1 for partial differencing,
        d = 1 for standard differencing.
    threshold : float
        Minimum absolute weight to keep.

    Returns
    -------
    np.ndarray of weights, starting with w_0 = 1.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w)


# -----------------------------------------------------------------------
# Fixed-width window fracdiff  (AFML Snippet 5.2)
# -----------------------------------------------------------------------

def frac_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """Apply fixed-width window fractional differencing.

    Parameters
    ----------
    series : pd.Series
        Input series (typically log prices).
    d : float
        Differencing order. 0 < d < 1.
    threshold : float
        Weight truncation threshold.  Default 1e-4 balances
        accuracy against window length (see ``frac_diff_weights``).

    Returns
    -------
    pd.Series of fractionally differenced values.
        Leading NaNs where the weight window exceeds available data.
    """
    w = frac_diff_weights(d, threshold)
    width = len(w)

    values = series.values.astype(np.float64)
    n = len(values)
    result = np.full(n, np.nan)

    for i in range(width - 1, n):
        window = values[i - width + 1 : i + 1][::-1]
        result[i] = np.dot(w, window)

    return pd.Series(result, index=series.index, name=f"fracdiff_d{d:.2f}")


def frac_diff_log(
    close: pd.Series,
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """Fractionally difference the log of a price series.

    Convenience wrapper: applies ``frac_diff`` to ``log(close)``.
    At d=1.0 this is equivalent to log returns.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    d : float
        Differencing order.
    threshold : float
        Weight truncation threshold.

    Returns
    -------
    pd.Series of fractionally differenced log prices.
    """
    return frac_diff(np.log(close), d, threshold)


# -----------------------------------------------------------------------
# Minimum d for stationarity  (AFML Snippet 5.3)
# -----------------------------------------------------------------------

def find_min_d(
    close: pd.Series,
    d_range: np.ndarray | None = None,
    threshold: float = 1e-4,
    significance: float = 0.05,
) -> dict[str, object]:
    """Find the minimum fractional differencing order for stationarity.

    Sweeps over d values and runs the ADF test on each.
    Returns the minimum d where the ADF p-value < significance.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    d_range : np.ndarray | None
        Array of d values to test. Default: 0.0 to 1.0 in steps of 0.05.
    threshold : float
        Weight truncation threshold for fracdiff.
    significance : float
        ADF significance level.

    Returns
    -------
    dict with keys:
        - ``min_d``: minimum d for stationarity (or NaN if none found)
        - ``results``: list of dicts with d, adf_stat, p_value, is_stationary
        - ``correlation_at_min_d``: correlation between fracdiff(min_d) and log(close)
    """
    from statsmodels.tsa.stattools import adfuller

    if d_range is None:
        d_range = np.arange(0.0, 1.05, 0.05)

    log_close = np.log(close)
    results = []
    min_d = np.nan
    min_d_corr = np.nan

    for d in d_range:
        fd = frac_diff(log_close, d, threshold) if d > 0 else log_close
        fd_clean = fd.dropna()

        if len(fd_clean) < 30:
            results.append({"d": d, "adf_stat": np.nan, "p_value": np.nan, "is_stationary": False})
            continue

        adf_result = adfuller(fd_clean, maxlag=1, regression="c", autolag=None)
        adf_stat, p_value = adf_result[0], adf_result[1]
        is_stationary = p_value < significance

        results.append({
            "d": round(d, 3),
            "adf_stat": adf_stat,
            "p_value": p_value,
            "is_stationary": is_stationary,
        })

        if is_stationary and np.isnan(min_d):
            min_d = d
            # Correlation with original log prices — measures memory retained
            common = log_close.reindex(fd_clean.index)
            min_d_corr = common.corr(fd_clean)

    return {
        "min_d": min_d,
        "results": results,
        "correlation_at_min_d": min_d_corr,
    }
