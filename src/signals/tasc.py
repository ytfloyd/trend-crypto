"""TASC indicator signal functions (pure, no I/O).

Self-contained ports of the Ehlers indicators used as registry signals. Each
returns a wide target-weight frame (index=ts, columns=symbol) for the research
runner's fast screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _laguerre_filter(series: pd.Series, gamma: float = 0.8) -> pd.Series:
    """Ehlers Laguerre filter (recursive 4-pole smoother).

    Ported verbatim from scripts/research/common/tasc2025_indicators.laguerre_filter.
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    values = pd.Series(series, dtype=float).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    l0 = l1 = l2 = l3 = np.nan
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        if not np.isfinite(l0):
            l0 = l1 = l2 = l3 = v
        else:
            old_l0, old_l1, old_l2 = l0, l1, l2
            l0 = (1.0 - gamma) * v + gamma * l0
            l1 = -gamma * l0 + old_l0 + gamma * l1
            l2 = -gamma * l1 + old_l1 + gamma * l2
            l3 = -gamma * l2 + old_l2 + gamma * l3
        out[i] = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0
    return pd.Series(out, index=pd.Series(series).index, name="laguerre")


def _continuation_state(close: pd.Series, gamma: float, length: int) -> pd.Series:
    """+1/-1 trend state from the smoothed Laguerre slope; NaN during warm-up.

    Mirrors tasc2025_indicators.continuation_index but keeps the leading warm-up
    as NaN (rather than filling 0.0) so the backtest drops it instead of booking
    flat bars that dilute the metrics.
    """
    filt = _laguerre_filter(close, gamma=gamma)
    slope = filt.diff()
    smoothed = slope.ewm(span=length, adjust=False, min_periods=length).mean()
    state = np.sign(smoothed)
    # carry the regime forward through zero/flat regions, but leave the leading
    # warm-up (before the first defined state) as NaN.
    return state.replace(0.0, np.nan).ffill()


def continuation_index(bars: pd.DataFrame, gamma: float = 0.8, length: int = 20) -> pd.DataFrame:
    """Long-only Ehlers Continuation Index, equal-weight across the universe.

    Go long a symbol when its continuation-index state is +1, else flat. Active
    positions are equal-weighted (weight = 1/N per long symbol). Causal: the
    Laguerre filter and slope use only trailing closes; the runner lags execution.

    Parameters
    ----------
    bars : pd.DataFrame
        Long format with columns: symbol, ts, close.
    gamma : float
        Laguerre filter damping in [0, 1).
    length : int
        Slope-smoothing window in bars (> 1).

    Returns
    -------
    pd.DataFrame
        Wide target weights, index=ts (sorted), columns=symbol; NaN during warm-up.
    """
    if length <= 1:
        raise ValueError(f"length must be > 1, got {length}")

    close = bars.pivot(index="ts", columns="symbol", values="close").sort_index()
    states = close.apply(lambda s: _continuation_state(s, gamma=gamma, length=length))

    # long (1.0) when state == +1, flat (0.0) otherwise; preserve NaN warm-up.
    long = states.gt(0).astype(float).where(states.notna())
    n_symbols = close.shape[1]
    weights = long / float(n_symbols) if n_symbols else long
    return weights
