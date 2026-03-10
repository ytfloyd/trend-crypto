"""
Alternative bar construction from high-frequency data.

Implements the financial data structures from AFML Chapter 2:
  - Time bars  (standard OHLCV at fixed time intervals — baseline)
  - Tick bars  (sample every N trades / 1m candles)
  - Volume bars (sample every V units of volume)
  - Dollar bars (sample every D units of dollar volume)
  - Tick imbalance bars (sample when cumulative signed-tick imbalance
    exceeds a dynamic threshold)

All functions accept a pandas DataFrame of 1-minute candles
(columns: ts, open, high, low, close, volume) for a **single symbol**
and return a DataFrame of OHLCV bars at the requested sampling.

Reference:
    López de Prado, M. (2018) *Advances in Financial Machine Learning*,
    Chapter 2: Financial Data Structures (pp. 23–37).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _threshold_bars(
    df: pd.DataFrame,
    values: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Build bars by accumulating *values* until *threshold* is reached.

    Uses numpy arrays internally for speed.  Resets accumulator after
    each bar (matching the book's definition exactly).
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    ts = df["ts"].values
    opn = df["open"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    vol = df["volume"].values
    dv = close.astype(np.float64) * vol.astype(np.float64)

    records: list[dict[str, object]] = []
    cum = 0.0
    chunk_start = 0

    for i in range(n):
        cum += values[i]
        if cum >= threshold:
            s = chunk_start
            records.append({
                "ts_start": ts[s],
                "ts_end": ts[i],
                "open": opn[s],
                "high": high[s : i + 1].max(),
                "low": low[s : i + 1].min(),
                "close": close[i],
                "volume": vol[s : i + 1].sum(),
                "dollar_volume": dv[s : i + 1].sum(),
                "n_candles": i - s + 1,
            })
            cum = 0.0
            chunk_start = i + 1

    return pd.DataFrame(records)


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns and sort by timestamp."""
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.sort_values("ts").reset_index(drop=True)
    return out


def _aggregate_bar(chunk: pd.DataFrame) -> dict[str, object]:
    """Aggregate a chunk of 1m candles into a single OHLCV bar."""
    return {
        "ts_start": chunk["ts"].iloc[0],
        "ts_end": chunk["ts"].iloc[-1],
        "open": chunk["open"].iloc[0],
        "high": chunk["high"].max(),
        "low": chunk["low"].min(),
        "close": chunk["close"].iloc[-1],
        "volume": chunk["volume"].sum(),
        "dollar_volume": (chunk["close"] * chunk["volume"]).sum(),
        "n_candles": len(chunk),
    }


# -----------------------------------------------------------------------
# Tick bars (fixed number of 1m candles per bar)
# -----------------------------------------------------------------------

def tick_bars(df: pd.DataFrame, ticks_per_bar: int = 60) -> pd.DataFrame:
    """Sample a bar every *ticks_per_bar* 1-minute candles.

    Since we use 1m candles as our atomic unit (proxy for ticks),
    ``ticks_per_bar=60`` yields roughly hourly bars, while
    ``ticks_per_bar=1440`` yields roughly daily bars.

    Parameters
    ----------
    df : pd.DataFrame
        Single-symbol 1m OHLCV (columns: ts, open, high, low, close, volume).
    ticks_per_bar : int
        Number of 1m candles per output bar.

    Returns
    -------
    pd.DataFrame with columns: ts_start, ts_end, open, high, low, close,
        volume, dollar_volume, n_candles.
    """
    df = _validate_input(df)
    bars = []
    for start in range(0, len(df), ticks_per_bar):
        chunk = df.iloc[start : start + ticks_per_bar]
        if len(chunk) == 0:
            continue
        bars.append(_aggregate_bar(chunk))
    return pd.DataFrame(bars)


# -----------------------------------------------------------------------
# Volume bars
# -----------------------------------------------------------------------

def volume_bars(df: pd.DataFrame, volume_per_bar: float) -> pd.DataFrame:
    """Sample a bar every time cumulative volume reaches *volume_per_bar*.

    Volume bars tend to produce returns closer to IID Normal than time bars
    because they sample proportionally to trading activity.

    Parameters
    ----------
    df : pd.DataFrame
        Single-symbol 1m OHLCV.
    volume_per_bar : float
        Cumulative volume threshold for emitting a bar.

    Returns
    -------
    pd.DataFrame of OHLCV bars.
    """
    df = _validate_input(df)
    return _threshold_bars(df, df["volume"].values.astype(np.float64), volume_per_bar)


# -----------------------------------------------------------------------
# Dollar bars
# -----------------------------------------------------------------------

def dollar_bars(df: pd.DataFrame, dollars_per_bar: float) -> pd.DataFrame:
    """Sample a bar every time cumulative dollar volume reaches *dollars_per_bar*.

    Dollar bars normalise for price changes over time — a bar always
    represents roughly the same notional traded value.

    Parameters
    ----------
    df : pd.DataFrame
        Single-symbol 1m OHLCV.
    dollars_per_bar : float
        Cumulative dollar-volume threshold for emitting a bar.

    Returns
    -------
    pd.DataFrame of OHLCV bars.
    """
    df = _validate_input(df)
    dv = (df["close"].values * df["volume"].values).astype(np.float64)
    return _threshold_bars(df, dv, dollars_per_bar)


# -----------------------------------------------------------------------
# Tick Imbalance Bars  (AFML Snippet 2.2 / 2.3)
# -----------------------------------------------------------------------

def _signed_tick_rule(close: np.ndarray) -> np.ndarray:
    """Apply the tick rule to classify each 1m candle as +1 or -1.

    Uses close-to-close differences.  If the price is unchanged, the
    previous sign carries forward.
    """
    diff = np.diff(close, prepend=close[0])
    signs = np.sign(diff)
    # Carry forward on zero-diff
    last_sign = 1.0
    for i in range(len(signs)):
        if signs[i] == 0:
            signs[i] = last_sign
        else:
            last_sign = signs[i]
    return np.asarray(signs)


def tick_imbalance_bars(
    df: pd.DataFrame,
    expected_t: int = 60,
    *,
    ewma_span: int = 100,
) -> pd.DataFrame:
    """Construct Tick Imbalance Bars (TIBs).

    A new bar is emitted when the absolute cumulative signed-tick
    imbalance exceeds a dynamic threshold.  The threshold adapts
    via an EWMA of prior bar lengths and prior imbalance magnitudes.

    Parameters
    ----------
    df : pd.DataFrame
        Single-symbol 1m OHLCV.
    expected_t : int
        Initial expected bar length (number of 1m candles).
    ewma_span : int
        EWMA span for adaptive threshold estimation.

    Returns
    -------
    pd.DataFrame of OHLCV bars.

    Reference
    ---------
    AFML Section 2.3.2.1, pp. 29–31.
    """
    df = _validate_input(df)
    close = df["close"].values.astype(np.float64)
    b_t = _signed_tick_rule(close)

    alpha = 2.0 / (ewma_span + 1)

    # Initial estimates — use warmup window for imbalance, but cap it
    # to avoid an overly aggressive initial threshold
    warmup = min(expected_t, len(b_t))
    exp_t = float(expected_t)
    raw_imb = np.abs(b_t[:warmup].mean()) if warmup > 0 else 0.5
    exp_imbalance = max(raw_imb, 0.1)

    ts = df["ts"].values
    opn = df["open"].values
    high = df["high"].values
    low = df["low"].values
    cls = close
    vol = df["volume"].values
    dv = close * vol.astype(np.float64)

    bars: list[dict[str, object]] = []
    cum_theta = 0.0
    chunk_start = 0

    for i in range(len(df)):
        cum_theta += b_t[i]
        threshold = exp_t * exp_imbalance

        if abs(cum_theta) >= threshold and i > chunk_start:
            s, e = chunk_start, i
            bars.append({
                "ts_start": ts[s],
                "ts_end": ts[e],
                "open": opn[s],
                "high": high[s : e + 1].max(),
                "low": low[s : e + 1].min(),
                "close": cls[e],
                "volume": vol[s : e + 1].sum(),
                "dollar_volume": dv[s : e + 1].sum(),
                "n_candles": e - s + 1,
            })

            bar_len = float(e + 1 - s)
            bar_imbalance = abs(cum_theta) / bar_len

            exp_t = alpha * bar_len + (1 - alpha) * exp_t
            exp_imbalance = alpha * bar_imbalance + (1 - alpha) * exp_imbalance

            cum_theta = 0.0
            chunk_start = i + 1

    return pd.DataFrame(bars)


# -----------------------------------------------------------------------
# Diagnostics (used in notebooks)
# -----------------------------------------------------------------------

def bar_statistics(bars: pd.DataFrame, label: str = "") -> dict[str, object]:
    """Compute summary statistics on bar returns for comparison.

    Returns dict with: label, n_bars, mean_return, std_return,
    skewness, kurtosis, jarque_bera_stat, mean_bar_duration_minutes,
    serial_corr_lag1.
    """
    if bars.empty or "close" not in bars.columns:
        return {"label": label, "n_bars": 0}

    log_ret = np.log(bars["close"] / bars["close"].shift(1)).dropna()

    from scipy import stats as sp_stats

    jb_stat, _ = sp_stats.jarque_bera(log_ret) if len(log_ret) > 10 else (np.nan, np.nan)

    duration_min = np.nan
    if "ts_start" in bars.columns and "ts_end" in bars.columns:
        durations = (
            pd.to_datetime(bars["ts_end"]) - pd.to_datetime(bars["ts_start"])
        ).dt.total_seconds() / 60.0
        duration_min = durations.mean()

    return {
        "label": label,
        "n_bars": len(bars),
        "mean_return": float(log_ret.mean()),
        "std_return": float(log_ret.std()),
        "skewness": float(log_ret.skew()),
        "kurtosis": float(log_ret.kurtosis()),
        "jarque_bera": float(jb_stat),
        "mean_duration_min": float(duration_min),
        "serial_corr_lag1": float(log_ret.autocorr(lag=1)) if len(log_ret) > 2 else np.nan,
    }
