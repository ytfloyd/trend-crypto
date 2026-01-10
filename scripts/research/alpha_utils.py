#!/usr/bin/env python
from __future__ import annotations

"""
Indicator utilities used across research (daily and intraday bars).

Implemented for V1.5 Growth Sleeve:
- calc_atr: Wilder ATR
- calc_adx: Wilder ADX
- calc_ichimoku: 9/26/52 Ichimoku cloud with cloud boundaries and flags
- calc_keltner: Keltner channel (EMA/SMA mid ± ATR band)
- calc_dewma: Double EWMA (DEMA)
- calc_bollinger_width: (upper-lower)/mid for Bollinger bands

All functions are vectorized and NaN-safe; they operate per-symbol if a
`symbol` column is present, otherwise on the full series.
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd


def _group(df: pd.DataFrame):
    return df.groupby("symbol") if "symbol" in df.columns else [(None, df)]


def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Average True Range (Wilder). Returns ATR series aligned to df index.
    Expects columns high, low, close; optional symbol for per-asset rolling.
    """
    highs = df["high"]
    lows = df["low"]
    closes = df["close"]
    prev_close = closes.shift(1)
    tr = pd.DataFrame(
        {
            "hl": highs - lows,
            "hc": (highs - prev_close).abs(),
            "lc": (lows - prev_close).abs(),
        }
    ).max(axis=1)

    atr_list = []
    for _, sub in _group(pd.DataFrame({"tr": tr, "symbol": df["symbol"] if "symbol" in df.columns else None})):
        if "symbol" in sub.columns:
            sub = sub.drop(columns="symbol")
        atr = sub["tr"].ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        atr_list.append(atr)
    atr_out = pd.concat(atr_list).sort_index()
    atr_out.name = "atr"
    return atr_out


def calc_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Average Directional Index (Wilder).
    Returns ADX in [0,100]; NaN during warmup.
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("calc_adx requires high, low, close columns")

    highs = df["high"]
    lows = df["low"]
    prev_high = highs.shift(1)
    prev_low = lows.shift(1)

    dm_pos = (highs - prev_high).clip(lower=0)
    dm_neg = (prev_low - lows).clip(lower=0)
    dm_pos = dm_pos.where(dm_pos > dm_neg, 0.0)
    dm_neg = dm_neg.where(dm_neg > dm_pos, 0.0)

    atr = calc_atr(df, n)

    adx_list = []
    for _, sub in _group(pd.DataFrame({"dm_pos": dm_pos, "dm_neg": dm_neg, "atr": atr, "symbol": df["symbol"] if "symbol" in df.columns else None})):
        if "symbol" in sub.columns:
            sub = sub.drop(columns="symbol")
        atr_sub = sub["atr"]
        plus_di = 100 * sub["dm_pos"].ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_sub
        minus_di = 100 * sub["dm_neg"].ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_sub
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
        adx = dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        adx_list.append(adx)

    adx_out = pd.concat(adx_list).sort_index()
    adx_out.name = "adx"
    adx_out = adx_out.clip(lower=0.0, upper=100.0)
    return adx_out


def calc_ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    Ichimoku cloud components.
    Returns columns: tenkan, kijun, senkou_a, senkou_b, cloud_top, cloud_bottom, in_cloud, above_cloud, below_cloud.
    cloud_top/bottom are computed after the displacement shift.
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("calc_ichimoku requires high, low, close columns")

    def mid(series: pd.Series, window: int) -> pd.Series:
        roll_max = series.rolling(window=window, min_periods=window).max()
        roll_min = series.rolling(window=window, min_periods=window).min()
        return (roll_max + roll_min) / 2.0

    tenkan_line = mid(df["high"], tenkan).combine(mid(df["low"], tenkan), func=lambda a, b: (a + b) / 2)
    kijun_line = mid(df["high"], kijun).combine(mid(df["low"], kijun), func=lambda a, b: (a + b) / 2)
    senkou_a = ((tenkan_line + kijun_line) / 2.0).shift(displacement)
    senkou_b = mid(df["high"], senkou).combine(mid(df["low"], senkou), func=lambda a, b: (a + b) / 2).shift(displacement)

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    close = df["close"]
    in_cloud = (close >= cloud_bottom) & (close <= cloud_top)
    above_cloud = close > cloud_top
    below_cloud = close < cloud_bottom

    return pd.DataFrame(
        {
            "tenkan": tenkan_line,
            "kijun": kijun_line,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "in_cloud": in_cloud,
            "above_cloud": above_cloud,
            "below_cloud": below_cloud,
        }
    )


def calc_keltner(
    df: pd.DataFrame,
    n: int = 20,
    atr_mult: float = 2.0,
    ma: Literal["ema", "sma"] = "ema",
) -> pd.DataFrame:
    """
    Keltner Channel.
    mid = EMA/SMA(close, n)
    upper/lower = mid ± atr_mult * ATR(n)
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("calc_keltner requires high, low, close columns")
    close = df["close"]
    if ma == "ema":
        mid = close.ewm(span=n, adjust=False, min_periods=n).mean()
    else:
        mid = close.rolling(window=n, min_periods=n).mean()
    atr = calc_atr(df, n)
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def calc_dewma(series: pd.Series, n: int) -> pd.Series:
    """
    Double EWMA (DEMA): 2*EMA(x,n) - EMA(EMA(x,n),n)
    """
    ema1 = series.ewm(span=n, adjust=False, min_periods=n).mean()
    ema2 = ema1.ewm(span=n, adjust=False, min_periods=n).mean()
    return 2 * ema1 - ema2


def calc_bollinger_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """
    Bollinger width = (upper - lower) / mid, where mid = SMA(close, n).
    """
    mid = close.rolling(window=n, min_periods=n).mean()
    std = close.rolling(window=n, min_periods=n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    width = (upper - lower) / mid
    return width
