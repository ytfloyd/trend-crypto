#!/usr/bin/env python
"""Research primitives for the TASC 2025 long-convex trend experiments.

The functions in this module intentionally live under ``scripts/research`` first:
they are stable, vectorized building blocks, but not yet production API. Once a
component survives the BTC / futures / ETF tests it can be promoted into ``src``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def autocorr_regime_score(
    returns: pd.Series,
    *,
    window: int = 126,
    min_lag: int = 2,
    max_lag: int = 20,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Rolling autocorrelation regime score over lags ``min_lag..max_lag``.

    Returns:
      - ``ac_abs_mean``: mean absolute lag autocorrelation.
      - ``ac_breadth``: fraction of lags whose absolute AC exceeds threshold.
      - ``ac_score``: average of the two, clipped to [0, 1].

    All values at timestamp ``t`` use data through ``t``. Strategy harnesses
    should shift the resulting score before trading if they act at next open.
    """
    if min_lag < 1 or max_lag < min_lag:
        raise ValueError("Require 1 <= min_lag <= max_lag")
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan)
    ac_cols: list[pd.Series] = []
    for lag in range(min_lag, max_lag + 1):
        ac = r.rolling(window=window, min_periods=max(30, window // 3)).corr(r.shift(lag))
        ac_cols.append(ac.rename(f"ac_lag_{lag}"))
    ac_df = pd.concat(ac_cols, axis=1)
    ac_abs = ac_df.abs()
    out = pd.DataFrame(index=r.index)
    out["ac_abs_mean"] = ac_abs.mean(axis=1)
    out["ac_breadth"] = (ac_abs >= threshold).mean(axis=1)
    out["ac_score"] = ((out["ac_abs_mean"] + out["ac_breadth"]) / 2.0).clip(0.0, 1.0)
    return out


def no_lag_ema(series: pd.Series, span: int) -> pd.Series:
    """Double-EMA no-lag approximation used as a simple Ehlers-style comparator."""
    if span <= 0:
        raise ValueError("span must be positive")
    x = pd.Series(series, dtype=float)
    ema1 = x.ewm(span=span, adjust=False, min_periods=span).mean()
    ema2 = ema1.ewm(span=span, adjust=False, min_periods=span).mean()
    return 2.0 * ema1 - ema2


def laguerre_filter(series: pd.Series, gamma: float = 0.8) -> pd.Series:
    """Ehlers Laguerre filter.

    The recursive form is:
      L0 = (1-gamma)*x + gamma*L0[-1]
      L1 = -gamma*L0 + L0[-1] + gamma*L1[-1]
      L2 = -gamma*L1 + L1[-1] + gamma*L2[-1]
      L3 = -gamma*L2 + L2[-1] + gamma*L3[-1]
      filt = (L0 + 2*L1 + 2*L2 + L3) / 6
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    x = pd.Series(series, dtype=float)
    values = x.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    l0 = l1 = l2 = l3 = np.nan
    prev_l0 = prev_l1 = prev_l2 = np.nan
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        if not np.isfinite(l0):
            l0 = l1 = l2 = l3 = v
            prev_l0 = prev_l1 = prev_l2 = v
        else:
            old_l0, old_l1, old_l2 = l0, l1, l2
            l0 = (1.0 - gamma) * v + gamma * l0
            l1 = -gamma * l0 + old_l0 + gamma * l1
            l2 = -gamma * l1 + old_l1 + gamma * l2
            l3 = -gamma * l2 + old_l2 + gamma * l3
            prev_l0, prev_l1, prev_l2 = old_l0, old_l1, old_l2
        # Keep references live for readability and future debugging of the recursion.
        _ = (prev_l0, prev_l1, prev_l2)
        out[i] = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0
    return pd.Series(out, index=x.index, name="laguerre")


def continuation_index(
    close: pd.Series,
    *,
    gamma: float = 0.8,
    length: int = 20,
) -> pd.Series:
    """Binary +1/-1 trend state based on Laguerre slope persistence.

    This is a practical approximation for the platform tests: the Laguerre slope
    is smoothed over ``length`` bars and the state is the sign of that smoothed
    slope, carried forward through zero/NaN regions.
    """
    if length <= 1:
        raise ValueError("length must be > 1")
    filt = laguerre_filter(close, gamma=gamma)
    slope = filt.diff()
    smoothed = slope.ewm(span=length, adjust=False, min_periods=length).mean()
    state = np.sign(smoothed)
    state = state.replace(0.0, np.nan).ffill().fillna(0.0)
    return state.rename("continuation_index")


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range for OHLC data with columns ``high``, ``low``, ``close``."""
    _require_cols(df, {"high", "low", "close"})
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rename("true_range")


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder-style ATR."""
    if length <= 0:
        raise ValueError("length must be positive")
    return true_range(df).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean().rename("atr")


def supertrend(
    df: pd.DataFrame,
    *,
    atr_len: int = 13,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """ATR-based SuperTrend state and trailing band.

    Returns columns ``supertrend``, ``supertrend_dir`` (+1/-1), ``st_upper``,
    and ``st_lower``.
    """
    _require_cols(df, {"high", "low", "close"})
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    hl2 = (high + low) / 2.0
    atr_s = atr(df, atr_len)
    basic_upper = hl2 + multiplier * atr_s
    basic_lower = hl2 - multiplier * atr_s
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = pd.Series(np.nan, index=df.index, dtype=float)
    st = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0 or not np.isfinite(atr_s.iloc[i]):
            continue
        prev_close = close.iloc[i - 1]
        prev_upper = final_upper.iloc[i - 1]
        prev_lower = final_lower.iloc[i - 1]
        if np.isfinite(prev_upper):
            if basic_upper.iloc[i] < prev_upper or prev_close > prev_upper:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = prev_upper
        if np.isfinite(prev_lower):
            if basic_lower.iloc[i] > prev_lower or prev_close < prev_lower:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = prev_lower

        prior_dir = direction.iloc[i - 1] if np.isfinite(direction.iloc[i - 1]) else 1.0
        if prior_dir > 0:
            direction.iloc[i] = -1.0 if close.iloc[i] < final_lower.iloc[i] else 1.0
        else:
            direction.iloc[i] = 1.0 if close.iloc[i] > final_upper.iloc[i] else -1.0
        st.iloc[i] = final_lower.iloc[i] if direction.iloc[i] > 0 else final_upper.iloc[i]

    return pd.DataFrame(
        {
            "supertrend": st,
            "supertrend_dir": direction.ffill().fillna(0.0),
            "st_upper": final_upper,
            "st_lower": final_lower,
        },
        index=df.index,
    )


def linear_regression_channel(
    close: pd.Series,
    *,
    length: int = 40,
    width: float = 2.0,
) -> pd.DataFrame:
    """Rolling linear-regression slope and channel bounds."""
    if length <= 2:
        raise ValueError("length must be > 2")
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    c = pd.Series(close, dtype=float)

    def _fit(vals: np.ndarray) -> pd.Series:
        y = vals.astype(float)
        y_mean = y.mean()
        slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
        intercept = float(y_mean - slope * x_mean)
        fitted = intercept + slope * x
        resid = y - fitted
        sigma = float(np.sqrt(np.mean(resid**2)))
        center = float(fitted[-1])
        return pd.Series(
            {
                "lr_center": center,
                "lr_slope": slope,
                "lr_upper": center + width * sigma,
                "lr_lower": center - width * sigma,
            }
        )

    out = c.rolling(length, min_periods=length).apply(lambda _v: np.nan)
    rows = []
    idx = []
    for i in range(length - 1, len(c)):
        vals = c.iloc[i - length + 1 : i + 1].to_numpy(dtype=float)
        if np.isfinite(vals).all():
            rows.append(_fit(vals))
            idx.append(c.index[i])
    result = pd.DataFrame(rows, index=idx)
    return result.reindex(c.index)


def ulcer_index(nav: pd.Series) -> float:
    """Ulcer Index: RMS percentage drawdown."""
    n = pd.Series(nav, dtype=float).dropna()
    if n.empty:
        return 0.0
    dd = n / n.cummax() - 1.0
    return float(np.sqrt(np.mean(np.square(dd.clip(upper=0.0)))))


def drawdown_duration(nav: pd.Series) -> dict[str, int]:
    """Drawdown duration in bars: max time underwater and current time underwater."""
    n = pd.Series(nav, dtype=float).dropna()
    if n.empty:
        return {"max_drawdown_duration": 0, "current_drawdown_duration": 0}
    underwater = n < n.cummax()
    max_dur = 0
    cur = 0
    for flag in underwater:
        cur = cur + 1 if bool(flag) else 0
        max_dur = max(max_dur, cur)
    return {"max_drawdown_duration": int(max_dur), "current_drawdown_duration": int(cur)}


def _require_cols(df: pd.DataFrame, cols: set[str]) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

