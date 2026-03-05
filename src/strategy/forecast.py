"""Carver-style continuous forecast generators with standard scaling.

Every trading rule produces a raw forecast, which is then scaled so the
average absolute value equals ``TARGET_ABS_FORECAST`` (default 10).
Forecasts are capped at +/- ``FORECAST_CAP`` (default 20).

Reference: Robert Carver, *Systematic Trading*, Chapters 7-8.

The functions here operate on numpy arrays for portability across
the Polars engine stack and the Pandas research stack.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_ABS_FORECAST = 10.0
FORECAST_CAP = 20.0


def _ewma(x: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted moving average via pandas (for numerical stability)."""
    return pd.Series(x).ewm(span=span, min_periods=span).mean().to_numpy()


def _sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(x).rolling(window, min_periods=window).mean().to_numpy()


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation."""
    return pd.Series(x).rolling(window, min_periods=window).std(ddof=1).to_numpy()


def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=window).max().to_numpy()


def _rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=window).min().to_numpy()


# ── EWMAC ────────────────────────────────────────────────────────────────
def ewmac_raw(
    close: np.ndarray,
    fast_span: int,
    slow_span: int,
    vol_lookback: int | None = None,
) -> np.ndarray:
    """Raw EWMAC forecast: (fast_EMA - slow_EMA) / sigma_price.

    Parameters
    ----------
    close : array of closing prices
    fast_span : EMA span for the fast MA
    slow_span : EMA span for the slow MA (must be > fast_span)
    vol_lookback : window for rolling price volatility estimate.
        Defaults to ``slow_span`` if not provided.
    """
    if vol_lookback is None:
        vol_lookback = slow_span

    fast_ma = _ewma(close, fast_span)
    slow_ma = _ewma(close, slow_span)
    raw_cross = fast_ma - slow_ma

    returns = np.diff(close, prepend=np.nan)
    sigma = _rolling_std(returns, vol_lookback)
    sigma = np.where((sigma == 0) | np.isnan(sigma), np.nan, sigma)

    return raw_cross / sigma


def ewmac_forecast(
    close: np.ndarray,
    fast_span: int,
    slow_span: int,
    vol_lookback: int | None = None,
    scalar: float | None = None,
    cap: float = FORECAST_CAP,
) -> np.ndarray:
    """Scaled and capped EWMAC forecast.

    If ``scalar`` is None, it is estimated from the data so that
    the average |forecast| ≈ ``TARGET_ABS_FORECAST``.
    """
    raw = ewmac_raw(close, fast_span, slow_span, vol_lookback)
    if scalar is None:
        scalar = estimate_forecast_scalar(raw)
    scaled = raw * scalar
    return np.clip(scaled, -cap, cap)


# ── Breakout / Channel ───────────────────────────────────────────────────
def breakout_raw(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Raw breakout forecast: channel position mapped to [-1, +1].

    ``forecast = (close - rolling_low) / (rolling_high - rolling_low) * 2 - 1``

    At the top of the channel → +1, at the bottom → -1, midpoint → 0.
    Uses shifted windows (data through t-1) to avoid lookahead.
    """
    shifted_high = np.roll(high, 1)
    shifted_high[0] = np.nan
    shifted_low = np.roll(low, 1)
    shifted_low[0] = np.nan

    roll_high = _rolling_max(shifted_high, lookback)
    roll_low = _rolling_min(shifted_low, lookback)

    channel_width = roll_high - roll_low
    channel_width = np.where(channel_width <= 0, np.nan, channel_width)

    position = (close - roll_low) / channel_width
    return position * 2.0 - 1.0


def breakout_forecast(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    lookback: int,
    scalar: float | None = None,
    cap: float = FORECAST_CAP,
) -> np.ndarray:
    """Scaled and capped breakout forecast.

    The raw breakout is in [-1, +1], so a typical scalar is ~10
    (mapping +1 to +10 average).
    """
    raw = breakout_raw(close, high, low, lookback)
    if scalar is None:
        scalar = estimate_forecast_scalar(raw)
    scaled = raw * scalar
    return np.clip(scaled, -cap, cap)


# ── Forecast scaling ─────────────────────────────────────────────────────
def estimate_forecast_scalar(
    raw_forecast: np.ndarray,
    target_abs: float = TARGET_ABS_FORECAST,
    backfill: bool = True,
) -> float:
    """Estimate the scalar needed so mean(|forecast|) ≈ target_abs.

    Uses the median of |raw| (robust to outliers) as the denominator.
    Carver uses a similar approach, expanding-window or fixed.
    """
    valid = raw_forecast[np.isfinite(raw_forecast)]
    if len(valid) < 10:
        return 1.0
    avg_abs = np.nanmedian(np.abs(valid))
    if avg_abs < 1e-12:
        return 1.0
    return target_abs / avg_abs


def cap_forecast(
    forecast: np.ndarray,
    cap: float = FORECAST_CAP,
) -> np.ndarray:
    """Clip forecast to +/- cap."""
    return np.clip(forecast, -cap, cap)


# ── Long-only variant ────────────────────────────────────────────────────
def long_only_forecast(
    forecast: np.ndarray,
    cap: float = FORECAST_CAP,
) -> np.ndarray:
    """Clamp forecast to [0, cap] for long-only strategies."""
    return np.clip(forecast, 0.0, cap)


# ── Convenience: generate a suite of EWMAC forecasts ─────────────────────
DEFAULT_EWMAC_PAIRS = [
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256),
]

DEFAULT_BREAKOUT_LOOKBACKS = [20, 40, 80, 160]


def ewmac_suite(
    close: np.ndarray,
    pairs: list[tuple[int, int]] | None = None,
    long_only: bool = True,
) -> dict[str, np.ndarray]:
    """Generate multiple EWMAC forecasts at different speeds.

    Returns dict mapping rule name (e.g. ``'ewmac_8_32'``) to forecast array.
    """
    if pairs is None:
        pairs = DEFAULT_EWMAC_PAIRS
    result: dict[str, np.ndarray] = {}
    for fast, slow in pairs:
        name = f"ewmac_{fast}_{slow}"
        fc = ewmac_forecast(close, fast, slow)
        if long_only:
            fc = long_only_forecast(fc)
        result[name] = fc
    return result


def breakout_suite(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    lookbacks: list[int] | None = None,
    long_only: bool = True,
) -> dict[str, np.ndarray]:
    """Generate multiple breakout forecasts at different lookbacks.

    Returns dict mapping rule name (e.g. ``'breakout_20'``) to forecast array.
    """
    if lookbacks is None:
        lookbacks = DEFAULT_BREAKOUT_LOOKBACKS
    result: dict[str, np.ndarray] = {}
    for lb in lookbacks:
        name = f"breakout_{lb}"
        fc = breakout_forecast(close, high, low, lb)
        if long_only:
            fc = long_only_forecast(fc)
        result[name] = fc
    return result
