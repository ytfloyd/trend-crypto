"""Cross-sectional signal functions (pure, no I/O).

These return a wide DataFrame of cross-sectional *rank scores* (index=ts,
columns=symbol, values in [0, 1]) — NOT portfolio weights. The research runner's
cross-sectional execute path turns the ranks into an inverse-vol long/short
quantile book. Contrast with signals.trend / signals.tasc, which return weights
for the long-only directional screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Daily-native adaptation of the Medallion Lite factor intent
# (scripts/research/medallion_lite/factors.py is the hourly original).
_DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum": 0.30,
    "volume_surge": 0.15,
    "realized_vol": 0.15,
    "proximity_to_high": 0.15,
    "rolling_sharpe": 0.25,
}
_ANN_DAILY = 365.0


def medallion_lite(
    bars: pd.DataFrame,
    lookback: int = 7,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Medallion-Lite cross-sectional composite score (daily).

    Ranks the universe each bar on five factors and returns the weighted-average
    cross-sectional percentile (∈ [0, 1]). Higher = more attractive long. All
    inputs are trailing, so the score at bar t uses only data up to t.

    Factors (each cross-sectionally ranked to [0, 1] per bar):
      momentum          — ``lookback``-bar log return
      volume_surge      — volume vs its ``lookback``-bar average
      realized_vol      — ``lookback``-bar annualized vol
      proximity_to_high — closeness to the ``lookback``-bar high
      rolling_sharpe    — ``lookback``-bar return / vol

    Parameters
    ----------
    bars : pd.DataFrame
        Long format with columns: symbol, ts, close, high, volume.
    lookback : int
        Factor lookback in bars (> 1).
    weights : dict | None
        Per-factor weights; defaults to the Medallion Lite blend.

    Returns
    -------
    pd.DataFrame
        Wide composite scores ∈ [0, 1], index=ts (sorted), columns=symbol.
    """
    if lookback <= 1:
        raise ValueError(f"lookback must be > 1, got {lookback}")
    w = dict(_DEFAULT_WEIGHTS if weights is None else weights)

    close = bars.pivot(index="ts", columns="symbol", values="close").sort_index()
    high = bars.pivot(index="ts", columns="symbol", values="high").sort_index()
    volume = bars.pivot(index="ts", columns="symbol", values="volume").sort_index()
    ret = close.pct_change()

    min_p = max(lookback // 2, 2)
    factors: dict[str, pd.DataFrame] = {
        "momentum": np.log(close / close.shift(lookback)),
        "volume_surge": (
            volume / volume.rolling(lookback, min_periods=min_p).mean().clip(lower=1e-8)
        ).clip(0, 5),
        "realized_vol": ret.rolling(lookback, min_periods=min_p).std() * np.sqrt(_ANN_DAILY),
        "proximity_to_high": 1.0
        + (close - high.rolling(lookback, min_periods=min_p).max())
        / high.rolling(lookback, min_periods=min_p).max().clip(lower=1e-8),
        "rolling_sharpe": (
            ret.rolling(lookback, min_periods=min_p).mean()
            / ret.rolling(lookback, min_periods=min_p).std().clip(lower=1e-8)
        ),
    }

    composite = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for name, weight in w.items():
        if name in factors:
            composite += weight * factors[name].rank(axis=1, pct=True).fillna(0.5)
    return composite
