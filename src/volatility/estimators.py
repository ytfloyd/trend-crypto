"""Realized volatility estimators.

Each estimator takes OHLCV price data and produces a rolling annualised
volatility estimate.  The estimators differ in statistical efficiency —
how many bars of data they need for a given level of accuracy.

Relative efficiency (vs close-to-close, higher = better):
    Close-to-close:   1.0  (baseline)
    Parkinson:         5.2  (uses high-low range)
    Garman-Klass:      7.4  (uses OHLC)
    Rogers-Satchell:   8.0  (handles drift, uses OHLC)
    Yang-Zhang:       14.0  (combines overnight + intraday, uses OHLC)

All functions return pd.Series of annualised volatility, indexed the
same as the input.  NaN where insufficient data.

Reference: Natenberg, *Option Volatility and Pricing*, Ch. 6-7;
           Yang & Zhang (2000), "Drift Independent Volatility Estimation".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _annualise(per_bar_var: pd.Series, ann_factor: float) -> pd.Series:
    """Convert per-bar variance to annualised volatility."""
    return np.sqrt(per_bar_var * ann_factor)


# ── Close-to-close (standard) ────────────────────────────────────────────

def close_to_close(
    close: pd.Series,
    window: int = 20,
    ann_factor: float = 365.0,
    ewm: bool = False,
) -> pd.Series:
    """Classical close-to-close realized volatility.

    This is the simplest estimator — rolling std of log returns.
    Statistically inefficient but unbiased and well-understood.
    """
    log_ret = np.log(close / close.shift(1))
    if ewm:
        var = log_ret.ewm(span=window, min_periods=max(10, window // 2)).var()
    else:
        var = log_ret.rolling(window, min_periods=window).var()
    return _annualise(var, ann_factor)


# ── Parkinson (1980) ─────────────────────────────────────────────────────

def parkinson(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    ann_factor: float = 365.0,
) -> pd.Series:
    """Parkinson (1980) high-low range estimator.

    5.2x more efficient than close-to-close.  Assumes no drift and
    continuous trading (biased downward by discrete sampling / gaps).

        sigma^2 = (1 / 4 ln 2) * E[(ln H/L)^2]
    """
    log_hl = np.log(high / low)
    hl_sq = log_hl ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    var = factor * hl_sq.rolling(window, min_periods=window).mean()
    return _annualise(var, ann_factor)


# ── Garman-Klass (1980) ──────────────────────────────────────────────────

def garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    ann_factor: float = 365.0,
) -> pd.Series:
    """Garman-Klass (1980) OHLC estimator.

    7.4x more efficient than close-to-close.  Uses all four OHLC prices
    per bar.  Assumes no drift and continuous trading.

        sigma^2 = 0.5 * (ln H/L)^2 - (2 ln 2 - 1) * (ln C/O)^2
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    term1 = 0.5 * log_hl ** 2
    term2 = (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    var = (term1 - term2).rolling(window, min_periods=window).mean()
    var = var.clip(lower=0.0)
    return _annualise(var, ann_factor)


# ── Rogers-Satchell (1991) ───────────────────────────────────────────────

def rogers_satchell(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    ann_factor: float = 365.0,
) -> pd.Series:
    """Rogers-Satchell (1991) estimator.

    ~8x more efficient than close-to-close.  Handles non-zero drift,
    which Parkinson and Garman-Klass do not.

        sigma^2 = E[ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)]
    """
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_)
    rs = log_hc * log_ho + log_lc * log_lo
    var = rs.rolling(window, min_periods=window).mean()
    var = var.clip(lower=0.0)
    return _annualise(var, ann_factor)


# ── Yang-Zhang (2000) ────────────────────────────────────────────────────

def yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    ann_factor: float = 365.0,
) -> pd.Series:
    """Yang-Zhang (2000) estimator.

    ~14x more efficient than close-to-close.  The most efficient
    estimator that is independent of drift and opening jumps.

    Combines overnight variance (close-to-open), open-to-close
    variance, and Rogers-Satchell intraday variance with weights
    that minimise overall estimator variance.

        sigma^2 = sigma_overnight^2 + k * sigma_open_close^2
                  + (1 - k) * sigma_RS^2

    where k = 0.34 / (1.34 + (n+1)/(n-1)) and n = window.
    """
    n = window

    log_oc = np.log(open_ / close.shift(1))
    log_co = np.log(close / open_)

    sigma_overnight_sq = log_oc.rolling(n, min_periods=n).var()
    sigma_openclose_sq = log_co.rolling(n, min_periods=n).var()

    # Rogers-Satchell component
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_)
    rs = log_hc * log_ho + log_lc * log_lo
    sigma_rs_sq = rs.rolling(n, min_periods=n).mean()

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = sigma_overnight_sq + k * sigma_openclose_sq + (1 - k) * sigma_rs_sq
    var = var.clip(lower=0.0)
    return _annualise(var, ann_factor)


# ── Utilities ─────────────────────────────────────────────────────────────

def vol_cone(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    windows: list[int] | None = None,
    ann_factor: float = 365.0,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> pd.DataFrame:
    """Volatility cone: distribution of realized vol at multiple horizons.

    For each window, compute Yang-Zhang vol over the full history, then
    report percentiles.  Useful for gauging whether current vol is rich
    or cheap relative to its own history.
    """
    if windows is None:
        windows = [5, 10, 20, 40, 60, 120, 252]

    rows = []
    for w in windows:
        yz = yang_zhang(open_, high, low, close, window=w, ann_factor=ann_factor)
        yz_clean = yz.dropna()
        if yz_clean.empty:
            continue
        pcts = np.percentile(yz_clean.values, percentiles)
        row = {"window": w, "current": float(yz_clean.iloc[-1])}
        for p, v in zip(percentiles, pcts):
            row[f"p{int(p)}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")


def compare_estimators(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    ann_factor: float = 365.0,
) -> pd.DataFrame:
    """Run all five estimators and return them side-by-side.

    Convenient for research notebooks to compare estimator behavior
    on the same underlying data.
    """
    return pd.DataFrame({
        "close_to_close": close_to_close(close, window, ann_factor),
        "parkinson": parkinson(high, low, window, ann_factor),
        "garman_klass": garman_klass(open_, high, low, close, window, ann_factor),
        "rogers_satchell": rogers_satchell(open_, high, low, close, window, ann_factor),
        "yang_zhang": yang_zhang(open_, high, low, close, window, ann_factor),
    })
