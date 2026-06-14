"""Causal features for the spot-convexity baseline (subset of the full taxonomy).

Every column is backward-looking and valid AT the signal date (the label window starts at
entry = signal+1, so there is no overlap). Breakout references use the PRIOR N-day extreme
(shift(1)) to avoid counting today's bar in its own breakout. These feed baseline_score.py;
the full feature library is built out in the empirical phase.

Operates on one asset's daily OHLCV DataFrame (DatetimeIndex, columns open/high/low/close/volume).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


def _tz(s: pd.Series, win: int = 252) -> pd.Series:
    """Trailing causal z-score (no full-sample stats), clipped to [-3, 3]."""
    mu = s.rolling(win, min_periods=win // 4).mean()
    sd = s.rolling(win, min_periods=win // 4).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-3, 3)


def _tpct(s: pd.Series, win: int = 252) -> pd.Series:
    """Trailing percentile rank in [0,1] of the last value within the window."""
    return s.rolling(win, min_periods=win // 4).apply(
        lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan, raw=True)


def compute_features(df: pd.DataFrame, *, stop_mult: float = 2.0) -> pd.DataFrame:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)
    ret1 = c.pct_change()

    # A. returns & momentum
    f["ret_21"] = c / c.shift(21) - 1
    f["ret_63"] = c / c.shift(63) - 1
    f["mom_accel"] = f["ret_21"] - f["ret_63"] / 3.0           # short vs medium pace

    # B. trend state
    sma50, sma100, sma200 = c.rolling(50).mean(), c.rolling(100).mean(), c.rolling(200).mean()
    f["dist_ma50"] = c / sma50 - 1
    f["slope_ma50"] = sma50 / sma50.shift(21) - 1
    f["trend_stack"] = ((c > sma50) & (sma50 > sma100) & (sma100 > sma200)).astype(float)

    # C. breakout & range position (prior N-day high => no self-inclusion)
    f["donchian20"] = (c > h.rolling(20).max().shift(1)).astype(float)
    f["dist_252high"] = c / h.rolling(252).max() - 1

    # D. realized vol / ATR
    atr = atr_wilder(h, l, c, 14)
    f["atr_pct"] = atr / c
    f["atr_expansion"] = atr / atr.shift(21) - 1

    # E. compression -> expansion
    bb_width = (4 * c.rolling(20).std()) / c.rolling(20).mean()
    f["compression_score"] = 1.0 - _tpct(bb_width)            # high when quiet (low BB width pctile)
    rw5 = (h.rolling(5).max() - l.rolling(5).min()) / c
    rw21 = (h.rolling(21).max() - l.rolling(21).min()) / c
    f["range_expansion"] = rw5 / rw21

    # F. stop viability (proxy: asset's trailing up-tail vs planned stop size)
    stop_pct = (stop_mult * atr) / c
    support = l.rolling(20).min()
    f["support_in_atr"] = (c - support) / atr                 # >stop_mult => stop sits under support
    up_tail = (ret1.where(ret1 > 0)).rolling(252, min_periods=63).quantile(0.95)
    f["right_tail_to_stop"] = up_tail / stop_pct.replace(0, np.nan)

    # G. gap & stop-slippage risk
    gap = o / c.shift(1) - 1
    f["worst_gap_63"] = gap.rolling(63).min()
    f["gap_vs_stop"] = f["worst_gap_63"].abs() / stop_pct.replace(0, np.nan)

    # H. upside/downside asymmetry
    up_vol = ret1.where(ret1 > 0).rolling(63).std()
    down_vol = ret1.where(ret1 < 0).rolling(63).std()
    f["downside_to_upside_vol"] = down_vol / up_vol.replace(0, np.nan)

    # I. path quality / whipsaw (Kaufman efficiency ratio over 21d)
    direction = (c - c.shift(21)).abs()
    volatility = ret1.abs().rolling(21).sum() * c               # ~path length in price terms
    f["efficiency_ratio_21"] = (direction / volatility.replace(0, np.nan)).clip(0, 1)
    f["whipsaw"] = 1.0 - f["efficiency_ratio_21"]

    # K. volume & liquidity
    f["vol_ratio"] = v / v.rolling(20).mean()
    dollar_vol = c * v
    f["dollar_adv20"] = dollar_vol.rolling(20).mean()
    f["amihud"] = (ret1.abs() / dollar_vol.replace(0, np.nan)).rolling(21).mean()

    return f
