"""
Market regime filter for the Jumpers strategy.

The key insight from Sornette: bubble dynamics only apply when the
market IS in a bubble (or recovery) regime.  During sustained bear
markets, the LPPL/super-exponential signals generate false positives.

Regime classification
---------------------
We use a simple, robust dual-SMA filter on Bitcoin:

    BULL:   BTC close > SMA(50) AND SMA(50) > SMA(200)
    RISK-ON: BTC close > SMA(50) (allows early-recovery entries)
    BEAR:   otherwise

Only allocate capital during BULL or RISK-ON regimes.

Additionally, a cross-sectional breadth indicator:
    % of universe with 20d return > 0
    → breadth > 50% indicates broad-based uptrend
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime(
    btc_close: pd.Series,
    fast_sma: int = 50,
    slow_sma: int = 200,
) -> pd.DataFrame:
    """Classify market regime from BTC price.

    Parameters
    ----------
    btc_close : pd.Series
        BTC daily close, indexed by date.
    fast_sma, slow_sma : int
        SMA lookback periods.

    Returns
    -------
    pd.DataFrame
        Columns: [date, regime, regime_score].
        regime ∈ {"bull", "risk_on", "bear"}.
        regime_score ∈ [0, 1] — higher = more bullish.
    """
    close = btc_close.sort_index().copy()
    sma_fast = close.rolling(fast_sma, min_periods=fast_sma).mean()
    sma_slow = close.rolling(slow_sma, min_periods=slow_sma).mean()

    above_fast = close > sma_fast
    fast_above_slow = sma_fast > sma_slow

    regime = pd.Series("bear", index=close.index)
    regime[above_fast] = "risk_on"
    regime[above_fast & fast_above_slow] = "bull"

    # Regime score: distance of price from slow SMA, normalised
    dist = (close - sma_slow) / sma_slow
    score = dist.clip(-0.5, 0.5) / 0.5  # ∈ [-1, 1]
    score = (score + 1) / 2  # → [0, 1]

    df = pd.DataFrame({
        "date": close.index,
        "regime": regime.values,
        "regime_score": score.values,
        "btc_close": close.values,
        "sma_fast": sma_fast.values,
        "sma_slow": sma_slow.values,
    })
    return df


def compute_breadth(
    returns_wide: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """Fraction of symbols with positive trailing return.

    Returns a pd.Series (index=date) in [0, 1].
    """
    cum = returns_wide.rolling(lookback, min_periods=lookback).sum()
    breadth = (cum > 0).mean(axis=1)
    return breadth


def regime_mask(
    regime_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    allowed: set[str] | None = None,
) -> pd.Series:
    """Boolean mask: True on dates where regime is favorable.

    Parameters
    ----------
    regime_df : pd.DataFrame
        Output of ``compute_regime``.
    dates : pd.DatetimeIndex
        Dates to evaluate.
    allowed : set
        Allowed regime labels (default: {"bull", "risk_on"}).

    Returns
    -------
    pd.Series
        Boolean, indexed by dates.
    """
    if allowed is None:
        allowed = {"bull", "risk_on"}

    reg = regime_df.set_index("date")["regime"]
    reg = reg.reindex(dates, method="ffill")
    return reg.isin(allowed)
