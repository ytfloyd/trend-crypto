"""
Ensemble regime classifier — continuous [0, 1] probability score.

Replaces the binary BTC > SMA(50) gate with a smooth probability surface
built from four orthogonal regime indicators:

  1. BTC trend (dual-SMA)        → structural trend state
  2. Cross-sectional breadth     → market-wide participation
  3. BTC volatility compression  → calm-before-storm detector
  4. BTC momentum                → short-term trend confirmation

The continuous score scales portfolio exposure rather than toggling it,
which reduces whipsaw losses near regime boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _btc_trend_score(
    btc_daily: pd.Series, fast: int = 50, slow: int = 200,
) -> pd.Series:
    """BTC dual-SMA → continuous score: bull=1.0, risk_on=0.6, bear=0.0."""
    sma_f = btc_daily.rolling(fast, min_periods=fast).mean()
    sma_s = btc_daily.rolling(slow, min_periods=slow).mean()

    score = pd.Series(0.0, index=btc_daily.index)
    score[btc_daily > sma_f] = 0.6
    score[(btc_daily > sma_f) & (sma_f > sma_s)] = 1.0
    return score


def _breadth_score(
    returns_wide: pd.DataFrame, lookback: int = 168,
) -> pd.Series:
    """Fraction of tokens with positive trailing 7d return. ∈ [0, 1]."""
    cum = returns_wide.rolling(lookback, min_periods=lookback // 2).sum()
    return (cum > 0).mean(axis=1)


def _vol_compression_score(
    btc_hourly: pd.Series, short: int = 24, long: int = 168,
) -> pd.Series:
    """When short-term vol < long-term vol, score is high.

    Compressed vol often precedes directional breakouts — favorable for
    trend-following strategies that need movement to generate returns.
    """
    ret = btc_hourly.pct_change()
    vol_short = ret.rolling(short, min_periods=short).std()
    vol_long = ret.rolling(long, min_periods=long // 2).std()

    ratio = vol_long / vol_short.clip(lower=1e-8)
    return (ratio - 0.5).clip(0, 2) / 2.0


def _btc_momentum_score(
    btc_hourly: pd.Series, lookback: int = 168,
) -> pd.Series:
    """BTC 7d return normalised to [0, 1]."""
    ret = btc_hourly.pct_change(lookback)
    return (ret.clip(-0.5, 0.5) / 0.5 + 1) / 2


def compute_ensemble_regime(
    btc_daily: pd.Series,
    btc_hourly: pd.Series,
    returns_wide: pd.DataFrame,
    *,
    w_trend: float = 0.40,
    w_breadth: float = 0.30,
    w_volcomp: float = 0.15,
    w_momentum: float = 0.15,
) -> pd.Series:
    """Continuous regime probability ∈ [0, 1] at hourly frequency.

    Higher values indicate environments favorable for long momentum
    positions; the portfolio scales its gross exposure by this score.
    """
    trend = _btc_trend_score(btc_daily)
    trend_h = trend.reindex(returns_wide.index, method="ffill")

    breadth = _breadth_score(returns_wide)

    vol_comp = _vol_compression_score(btc_hourly)
    vol_comp = vol_comp.reindex(returns_wide.index, method="ffill")

    momentum = _btc_momentum_score(btc_hourly)
    momentum = momentum.reindex(returns_wide.index, method="ffill")

    score = (
        w_trend * trend_h.fillna(0)
        + w_breadth * breadth.fillna(0)
        + w_volcomp * vol_comp.fillna(0.5)
        + w_momentum * momentum.fillna(0.5)
    ).clip(0, 1)

    return score
