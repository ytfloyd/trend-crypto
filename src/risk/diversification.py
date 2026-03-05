"""Instrument Diversification Multiplier (IDM).

When trading N imperfectly correlated instruments, the portfolio's risk is
less than the sum of individual risks.  The IDM scales up per-instrument
positions so that the *portfolio* (not each leg) hits the vol target.

IDM = 1 / sqrt(w' @ C @ w)

where w = vector of instrument weights and C = correlation matrix of
instrument returns.

Reference: Robert Carver, *Systematic Trading*, Chapter 11.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_idm(
    instrument_weights: dict[str, float],
    correlation_matrix: pd.DataFrame | np.ndarray,
    symbols: list[str] | None = None,
    max_idm: float = 2.5,
) -> float:
    """Compute the Instrument Diversification Multiplier.

    Parameters
    ----------
    instrument_weights : dict mapping symbol to its portfolio weight.
    correlation_matrix : pairwise correlation matrix of instrument returns.
        If a DataFrame, columns/index should be symbol names.
        If an ndarray, ``symbols`` must be provided.
    symbols : required if correlation_matrix is an ndarray.
    max_idm : upper cap to prevent unrealistic scaling.
    """
    if isinstance(correlation_matrix, pd.DataFrame):
        syms = list(correlation_matrix.columns)
        corr = correlation_matrix.values
    else:
        if symbols is None:
            raise ValueError("symbols required when correlation_matrix is ndarray")
        syms = symbols
        corr = np.asarray(correlation_matrix)

    common = [s for s in syms if s in instrument_weights]
    if len(common) < 2:
        return 1.0

    idx = [syms.index(s) for s in common]
    sub_corr = corr[np.ix_(idx, idx)]
    w = np.array([instrument_weights[s] for s in common])
    w = w / w.sum()

    port_var = float(w @ sub_corr @ w)
    if port_var <= 0:
        return 1.0

    idm = 1.0 / math.sqrt(port_var)
    return min(idm, max_idm)


def rolling_idm(
    returns_wide: pd.DataFrame,
    instrument_weights: dict[str, float],
    corr_window: int = 125,
    max_idm: float = 2.5,
) -> pd.Series:
    """Compute a time-varying IDM from rolling instrument correlations.

    Parameters
    ----------
    returns_wide : DataFrame, index=ts, columns=symbols, values=returns.
    instrument_weights : dict mapping symbol to weight.
    corr_window : rolling window for correlation estimation.
    max_idm : upper cap.

    Returns
    -------
    pd.Series of IDM values, indexed by ts.
    """
    common = [s for s in returns_wide.columns if s in instrument_weights]
    if len(common) < 2:
        return pd.Series(1.0, index=returns_wide.index, name="idm")

    r = returns_wide[common].copy()
    w = np.array([instrument_weights[s] for s in common])
    w = w / w.sum()

    idm_values = np.full(len(r), np.nan)

    for t in range(corr_window - 1, len(r)):
        window = r.iloc[t - corr_window + 1 : t + 1]
        corr = window.corr().values
        if np.any(np.isnan(corr)):
            corr = np.nan_to_num(corr, nan=1.0)
            np.fill_diagonal(corr, 1.0)
        port_var = float(w @ corr @ w)
        if port_var > 0:
            idm_values[t] = min(1.0 / math.sqrt(port_var), max_idm)
        else:
            idm_values[t] = 1.0

    result = pd.Series(idm_values, index=returns_wide.index, name="idm")
    return result.ffill().fillna(1.0)


# ── Quick reference IDM table ────────────────────────────────────────────
# Carver provides approximate IDM values based on # instruments and
# average pairwise correlation.  Useful as a sanity check.

def approximate_idm(n_instruments: int, avg_correlation: float) -> float:
    """Quick IDM estimate from number of instruments and average correlation.

    Assumes equal weighting and a uniform correlation structure.
    """
    if n_instruments < 1:
        return 1.0
    if n_instruments == 1:
        return 1.0
    w = 1.0 / n_instruments
    port_var = w**2 * n_instruments + w**2 * n_instruments * (n_instruments - 1) * avg_correlation
    if port_var <= 0:
        return 1.0
    return min(1.0 / math.sqrt(port_var), 2.5)
