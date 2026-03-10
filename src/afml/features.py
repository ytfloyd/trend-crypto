"""
Structural breaks, entropy, and event-driven features.

Implements tools from AFML Chapters 14 and 15:

**Structural breaks (Ch 14):**
  - CUSUM filter — detect mean-shifts in a stream of returns
  - Supremum ADF (SADF) — test for explosive / bubble behaviour

**Entropy (Ch 15):**
  - Shannon entropy
  - Plug-in (maximum-likelihood) entropy
  - Lempel-Ziv complexity (compression-based)

These features help detect regime changes and measure information
content in financial time series — useful as both features and
filters for ML models.

Reference:
    López de Prado, M. (2018) *Advances in Financial Machine Learning*,
    Chapters 14–15.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =====================================================================
# CUSUM Filter  (AFML Chapter 2 / 14)
# =====================================================================

def cusum_filter(
    close: pd.Series,
    threshold: float,
) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter — event sampling.

    Fires an event whenever the cumulative deviation from the running
    mean exceeds ``threshold``.  This is a non-parametric way to
    detect structural breaks and sample at meaningful events rather
    than fixed time intervals.

    Parameters
    ----------
    close : pd.Series
        Price or log-price series.
    threshold : float
        Cumulative deviation threshold (e.g. daily vol × multiplier).

    Returns
    -------
    pd.DatetimeIndex — event timestamps.
    """
    diff = close.diff().dropna()
    s_pos, s_neg = 0.0, 0.0
    events = []

    for t, val in diff.items():
        s_pos = max(0, s_pos + val)
        s_neg = min(0, s_neg + val)

        if s_pos > threshold:
            events.append(t)
            s_pos = 0.0
        elif s_neg < -threshold:
            events.append(t)
            s_neg = 0.0

    return pd.DatetimeIndex(events)


# =====================================================================
# Supremum ADF  (AFML Chapter 14)
# =====================================================================

def sadf(
    log_prices: pd.Series,
    min_window: int = 50,
    max_window: int | None = None,
    lags: int = 1,
) -> pd.DataFrame:
    """Supremum Augmented Dickey-Fuller (SADF) test.

    For each date, runs ADF tests over expanding windows
    (from ``min_window`` to the current date) and reports the
    supremum (largest) ADF statistic.  High SADF values indicate
    explosive / bubble-like behaviour.

    Parameters
    ----------
    log_prices : pd.Series
        Log-price series (or any I(1) series).
    min_window : int
        Minimum window for the first ADF test.
    max_window : int | None
        Maximum window.  If None, uses the full history up to each date.
    lags : int
        Number of lags in ADF regression.

    Returns
    -------
    pd.DataFrame with columns: sadf_stat, indexed by date.
    """
    from statsmodels.tsa.stattools import adfuller

    results = []
    index = log_prices.index

    for end in range(min_window, len(log_prices)):
        start_min = 0 if max_window is None else max(0, end - max_window)
        sup_adf = -np.inf

        for start in range(start_min, end - min_window + 1):
            window = log_prices.iloc[start : end + 1]
            try:
                adf_stat = adfuller(window, maxlag=lags, regression="c", autolag=None)[0]
                sup_adf = max(sup_adf, adf_stat)
            except Exception:
                continue

        if sup_adf > -np.inf:
            results.append({"date": index[end], "sadf_stat": sup_adf})

    return pd.DataFrame(results).set_index("date")


# =====================================================================
# Entropy  (AFML Chapter 15)
# =====================================================================

def shannon_entropy(
    series: pd.Series,
    n_bins: int = 20,
) -> float:
    """Shannon entropy of a continuous series (via binning).

    Parameters
    ----------
    series : pd.Series
        Numeric values.
    n_bins : int
        Number of bins for the histogram.

    Returns
    -------
    float — entropy in nats.
    """
    counts, _ = np.histogram(series.dropna(), bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def plugin_entropy(
    message: str | np.ndarray,
) -> float:
    """Plug-in (maximum-likelihood) entropy estimator.

    Computes the empirical distribution of symbols and returns the
    Shannon entropy.  Works on discrete sequences (strings or arrays
    of integers).

    Parameters
    ----------
    message : str or np.ndarray
        Discrete sequence.

    Returns
    -------
    float — entropy in nats.
    """
    if isinstance(message, np.ndarray):
        message = message.astype(str)
        values, counts = np.unique(message, return_counts=True)
    else:
        values, counts = np.unique(list(message), return_counts=True)

    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def lempel_ziv_complexity(
    sequence: str | np.ndarray,
) -> float:
    """Lempel-Ziv complexity (normalised).

    Counts the number of distinct substrings found by the LZ76
    parsing algorithm, normalised by the theoretical maximum
    ``n / log2(n)``.  Higher values → more complex (random-like).

    Parameters
    ----------
    sequence : str or np.ndarray
        Binary or discrete sequence.  Arrays are converted to strings.

    Returns
    -------
    float — normalised LZ complexity in [0, ~1].
    """
    if isinstance(sequence, np.ndarray):
        s = "".join(sequence.astype(str))
    else:
        s = str(sequence)

    n = len(s)
    if n == 0:
        return 0.0

    # LZ76 parsing
    i = 0
    c = 1  # complexity counter
    mlen = 1  # current match length
    while i + mlen <= n:
        if s[i : i + mlen] in s[0 : i + mlen - 1]:
            mlen += 1
        else:
            c += 1
            i += mlen
            mlen = 1

    # Normalise
    return c / (n / np.log2(n)) if n > 1 else 1.0


def encode_returns_binary(
    returns: pd.Series,
) -> str:
    """Encode a return series as a binary string (1 = positive, 0 = negative).

    Useful as input to ``lempel_ziv_complexity`` and ``plugin_entropy``.
    """
    return "".join(["1" if r > 0 else "0" for r in returns.dropna()])
