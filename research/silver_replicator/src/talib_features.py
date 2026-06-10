"""Wide TA-Lib feature matrix for silver OHLCV bars.

``compute_features(df)`` returns a tidy DataFrame indexed by the input's
``ts`` index with one column per indicator output. Families covered:
    - Overlap studies (SMA/EMA suites, BBANDS, KAMA, T3, HT_TRENDLINE)
    - Momentum (RSI, MACD, ADX, AROON, CCI, STOCH, STOCHRSI, MOM, ROC,
      WILLR, MFI)
    - Volatility (ATR, NATR, TRANGE, STDDEV)
    - Cycle (HT_DCPERIOD, HT_DCPHASE, HT_SINE)
    - Pattern recognition (candlestick CDL*)

Default periods follow standard convention but every period is overridable
via ``**kwargs`` (e.g. ``rsi_period=21``, ``sma_periods=(20, 50)``).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import talib


DEFAULT_SMA_PERIODS: tuple[int, ...] = (10, 20, 50, 100, 200)
DEFAULT_EMA_PERIODS: tuple[int, ...] = (10, 20, 50, 100, 200)


def _as_f64(s: pd.Series) -> np.ndarray:
    return s.astype("float64").to_numpy()


def _overlap(
    out: dict[str, np.ndarray],
    close: np.ndarray,
    sma_periods: Sequence[int],
    ema_periods: Sequence[int],
    bbands_period: int,
    kama_period: int,
    t3_period: int,
) -> None:
    for p in sma_periods:
        out[f"sma_{p}"] = talib.SMA(close, timeperiod=p)
    for p in ema_periods:
        out[f"ema_{p}"] = talib.EMA(close, timeperiod=p)
    upper, middle, lower = talib.BBANDS(close, timeperiod=bbands_period)
    out[f"bband_upper_{bbands_period}"] = upper
    out[f"bband_middle_{bbands_period}"] = middle
    out[f"bband_lower_{bbands_period}"] = lower
    out[f"kama_{kama_period}"] = talib.KAMA(close, timeperiod=kama_period)
    out[f"t3_{t3_period}"] = talib.T3(close, timeperiod=t3_period)
    out["ht_trendline"] = talib.HT_TRENDLINE(close)


def _momentum(
    out: dict[str, np.ndarray],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    rsi_period: int,
    adx_period: int,
    aroon_period: int,
    cci_periods: Sequence[int],
    mom_period: int,
    roc_period: int,
    willr_period: int,
    mfi_period: int,
) -> None:
    out[f"rsi_{rsi_period}"] = talib.RSI(close, timeperiod=rsi_period)
    macd, macd_signal, macd_hist = talib.MACD(close)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out[f"adx_{adx_period}"] = talib.ADX(high, low, close, timeperiod=adx_period)
    aroon_down, aroon_up = talib.AROON(high, low, timeperiod=aroon_period)
    out[f"aroon_up_{aroon_period}"] = aroon_up
    out[f"aroon_down_{aroon_period}"] = aroon_down
    for p in cci_periods:
        out[f"cci_{p}"] = talib.CCI(high, low, close, timeperiod=p)
    slowk, slowd = talib.STOCH(high, low, close)
    out["stoch_k"] = slowk
    out["stoch_d"] = slowd
    fk, fd = talib.STOCHRSI(close)
    out["stochrsi_k"] = fk
    out["stochrsi_d"] = fd
    out[f"mom_{mom_period}"] = talib.MOM(close, timeperiod=mom_period)
    out[f"roc_{roc_period}"] = talib.ROC(close, timeperiod=roc_period)
    out[f"willr_{willr_period}"] = talib.WILLR(high, low, close, timeperiod=willr_period)
    out[f"mfi_{mfi_period}"] = talib.MFI(high, low, close, volume, timeperiod=mfi_period)


def _volatility(
    out: dict[str, np.ndarray],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_period: int,
    stddev_period: int,
) -> None:
    out[f"atr_{atr_period}"] = talib.ATR(high, low, close, timeperiod=atr_period)
    out[f"natr_{atr_period}"] = talib.NATR(high, low, close, timeperiod=atr_period)
    out["trange"] = talib.TRANGE(high, low, close)
    out[f"stddev_{stddev_period}"] = talib.STDDEV(close, timeperiod=stddev_period)


def _cycle(out: dict[str, np.ndarray], close: np.ndarray) -> None:
    out["ht_dcperiod"] = talib.HT_DCPERIOD(close)
    out["ht_dcphase"] = talib.HT_DCPHASE(close)
    sine, leadsine = talib.HT_SINE(close)
    out["ht_sine"] = sine
    out["ht_leadsine"] = leadsine


_PATTERN_FUNCS = {
    "cdl_engulfing": "CDLENGULFING",
    "cdl_hammer": "CDLHAMMER",
    "cdl_doji": "CDLDOJI",
    "cdl_morningstar": "CDLMORNINGSTAR",
    "cdl_eveningstar": "CDLEVENINGSTAR",
    "cdl_shootingstar": "CDLSHOOTINGSTAR",
    "cdl_harami": "CDLHARAMI",
    "cdl_3whitesoldiers": "CDL3WHITESOLDIERS",
    "cdl_3blackcrows": "CDL3BLACKCROWS",
}


def _patterns(
    out: dict[str, np.ndarray],
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> None:
    for col, fname in _PATTERN_FUNCS.items():
        func = getattr(talib, fname)
        out[col] = func(open_, high, low, close)


def compute_features(
    df_ohlcv: pd.DataFrame,
    *,
    sma_periods: Iterable[int] = DEFAULT_SMA_PERIODS,
    ema_periods: Iterable[int] = DEFAULT_EMA_PERIODS,
    bbands_period: int = 20,
    kama_period: int = 30,
    t3_period: int = 5,
    rsi_period: int = 14,
    adx_period: int = 14,
    aroon_period: int = 14,
    cci_periods: Iterable[int] = (14, 20),
    mom_period: int = 10,
    roc_period: int = 10,
    willr_period: int = 14,
    mfi_period: int = 14,
    atr_period: int = 14,
    stddev_period: int = 20,
) -> pd.DataFrame:
    """Compute a wide TA-Lib feature matrix from an OHLCV DataFrame.

    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        Must contain columns ``o, h, l, c, v`` and be indexed by ts
        (or have a ``ts`` column).
    """
    if df_ohlcv.empty:
        return pd.DataFrame(index=df_ohlcv.index)

    df = df_ohlcv.copy()
    if "ts" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("ts")
    df = df.sort_index()

    open_ = _as_f64(df["o"])
    high = _as_f64(df["h"])
    low = _as_f64(df["l"])
    close = _as_f64(df["c"])
    volume = _as_f64(df["v"])

    out: dict[str, np.ndarray] = {}

    _overlap(
        out, close,
        sma_periods=tuple(sma_periods),
        ema_periods=tuple(ema_periods),
        bbands_period=bbands_period,
        kama_period=kama_period,
        t3_period=t3_period,
    )
    _momentum(
        out, high, low, close, volume,
        rsi_period=rsi_period,
        adx_period=adx_period,
        aroon_period=aroon_period,
        cci_periods=tuple(cci_periods),
        mom_period=mom_period,
        roc_period=roc_period,
        willr_period=willr_period,
        mfi_period=mfi_period,
    )
    _volatility(
        out, high, low, close,
        atr_period=atr_period,
        stddev_period=stddev_period,
    )
    _cycle(out, close)
    _patterns(out, open_, high, low, close)

    feat = pd.DataFrame(out, index=df.index)
    feat.index.name = "ts"
    return feat
