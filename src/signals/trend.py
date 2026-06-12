"""Trend-following signal functions (pure, no I/O).

Each returns a wide target-weight frame (index=ts, columns=symbol) that the
research runner feeds to core.backtest.simple_backtest.
"""
from __future__ import annotations

import pandas as pd


def ma_crossover(bars: pd.DataFrame, fast: int = 5, slow: int = 40) -> pd.DataFrame:
    """Long-only moving-average crossover, equal-weight across the universe.

    For each symbol, go long when the ``fast``-bar simple moving average of close
    is above the ``slow``-bar SMA, else flat. Active positions are equal-weighted
    so the gross book is <= 1.0 (weight = 1/N per in-trend symbol, N = universe
    size). Causal: the SMAs use only trailing closes; the runner lags execution.

    Parameters
    ----------
    bars : pd.DataFrame
        Long format with columns: symbol, ts, close.
    fast, slow : int
        Fast/slow SMA windows in bars (fast < slow).

    Returns
    -------
    pd.DataFrame
        Wide target weights, index=ts (sorted), columns=symbol.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    close = bars.pivot(index="ts", columns="symbol", values="close").sort_index()

    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()

    in_trend = (fast_ma > slow_ma).astype(float)  # 1.0 long, 0.0 flat
    # Keep the warm-up (before the slow SMA is defined) as NaN rather than 0.0, so
    # downstream backtests drop it instead of booking flat zero-return bars that
    # would dilute the metrics.
    in_trend = in_trend.where(slow_ma.notna())
    n_symbols = close.shape[1]
    weights = in_trend / float(n_symbols) if n_symbols else in_trend
    return weights
