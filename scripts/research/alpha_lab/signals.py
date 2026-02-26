"""
Parameterized signal factory for the Alpha Lab.

Each signal is defined as a (name, family, params) tuple and a function
that takes wide-format close/volume/returns DataFrames and produces
wide-format signal scores (higher = go long, lower = go short).

The factory enumerates a combinatorial signal space spanning:
  - Momentum (time-series and cross-sectional)
  - Mean reversion
  - Volatility (low-vol, vol breakout, vol compression)
  - Volume (relative, trend)
  - Price structure (distance from high/low, range position)
  - Composite (momentum + vol filter, mean-reversion + volume confirmation)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import sys
from pathlib import Path

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import ANN_FACTOR


@dataclass(frozen=True)
class SignalSpec:
    """Immutable specification for a single alpha signal."""
    name: str
    family: str
    params: dict
    description: str


SignalFn = Callable[
    [pd.DataFrame, pd.DataFrame, pd.DataFrame, dict],
    pd.DataFrame,
]

# ---------------------------------------------------------------------------
# Signal computation functions
# ---------------------------------------------------------------------------

def _momentum(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Simple price momentum: cumulative return over lookback."""
    lb = params["lookback"]
    return close / close.shift(lb) - 1.0


def _momentum_cs_rank(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Cross-sectional rank of momentum (percentile 0-1 each day)."""
    lb = params["lookback"]
    raw = close / close.shift(lb) - 1.0
    return raw.rank(axis=1, pct=True)


def _vol_adjusted_momentum(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Momentum normalized by realized volatility (rolling Sharpe)."""
    ret_lb = params["ret_lookback"]
    vol_lb = params["vol_lookback"]
    mom = returns.rolling(ret_lb).mean()
    vol = returns.rolling(vol_lb).std()
    return mom / vol.replace(0, np.nan)


def _mean_reversion(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Mean reversion: negative of distance from moving average."""
    lb = params["lookback"]
    ma = close.rolling(lb).mean()
    return -(close / ma - 1.0)


def _rsi(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """RSI-based mean reversion: short overbought, long oversold."""
    lb = params["lookback"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(lb).mean()
    loss = (-delta.clip(upper=0)).rolling(lb).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return -(rsi - 50)  # negative = short overbought, long oversold


def _low_volatility(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Low-volatility factor: rank assets inversely by realized vol."""
    lb = params["lookback"]
    vol = returns.rolling(lb).std()
    return -vol.rank(axis=1, pct=True)


def _vol_compression(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Volatility compression: short-window vol / long-window vol."""
    short = params["short_window"]
    long = params["long_window"]
    vol_short = returns.rolling(short).std()
    vol_long = returns.rolling(long).std()
    ratio = vol_short / vol_long.replace(0, np.nan)
    return -ratio  # low ratio = compressed, expect breakout


def _vol_breakout(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Volatility breakout: go long when vol spikes above threshold."""
    lb = params["lookback"]
    threshold = params["threshold"]
    vol = returns.rolling(lb).std()
    vol_ma = vol.rolling(lb * 3).mean()
    ratio = vol / vol_ma.replace(0, np.nan)
    return (ratio > threshold).astype(float) * returns.rolling(5).mean().apply(np.sign)


def _volume_relative(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Relative volume: short-term avg volume / long-term avg volume."""
    short = params["short_window"]
    long = params["long_window"]
    vol_short = volume.rolling(short).mean()
    vol_long = volume.rolling(long).mean()
    return (vol_short / vol_long.replace(0, np.nan)).rank(axis=1, pct=True)


def _obv_momentum(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """On-balance volume momentum: OBV change over lookback."""
    lb = params["lookback"]
    direction = returns.apply(np.sign)
    obv = (direction * volume).cumsum()
    return obv - obv.shift(lb)


def _dist_from_high(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Distance from rolling high (buy strength near highs)."""
    lb = params["lookback"]
    rolling_max = close.rolling(lb).max()
    return close / rolling_max - 1.0


def _dist_from_low(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Distance from rolling low (mean-reversion near lows)."""
    lb = params["lookback"]
    rolling_min = close.rolling(lb).min()
    return -(close / rolling_min - 1.0)  # negative: closer to low = more oversold


def _range_position(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Position within rolling high-low range (0=low, 1=high)."""
    lb = params["lookback"]
    hi = close.rolling(lb).max()
    lo = close.rolling(lb).min()
    rng = hi - lo
    return ((close - lo) / rng.replace(0, np.nan)).rank(axis=1, pct=True)


def _ema_crossover(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """EMA crossover: fast EMA - slow EMA, normalized by price."""
    fast = params["fast"]
    slow = params["slow"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow) / close.replace(0, np.nan)


def _acceleration(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Momentum acceleration: change in momentum."""
    lb = params["lookback"]
    mom = returns.rolling(lb).mean()
    return mom - mom.shift(lb)


def _price_volume_trend(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Price-volume correlation: high correlation = trend confirmation."""
    lb = params["lookback"]
    return returns.rolling(lb).corr(volume.pct_change().rolling(lb).mean())


def _relative_strength_btc(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Relative strength vs BTC: outperformance over lookback."""
    lb = params["lookback"]
    btc_col = None
    for col in close.columns:
        if "BTC" in col.upper():
            btc_col = col
            break
    if btc_col is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    btc_ret = close[btc_col] / close[btc_col].shift(lb) - 1.0
    asset_ret = close / close.shift(lb) - 1.0
    return asset_ret.sub(btc_ret, axis=0)


def _momentum_reversal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Long-term momentum + short-term reversal composite."""
    mom_lb = params["mom_lookback"]
    rev_lb = params["rev_lookback"]
    mom = (close / close.shift(mom_lb) - 1.0).rank(axis=1, pct=True)
    rev = -(close / close.shift(rev_lb) - 1.0).rank(axis=1, pct=True)
    w_mom = params.get("mom_weight", 0.5)
    return w_mom * mom + (1 - w_mom) * rev


def _momentum_vol_filter(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Momentum with volatility filter: only take momentum in low-vol names."""
    mom_lb = params["mom_lookback"]
    vol_lb = params["vol_lookback"]
    vol_pct = params.get("vol_percentile", 0.5)
    mom = (close / close.shift(mom_lb) - 1.0).rank(axis=1, pct=True)
    vol = returns.rolling(vol_lb).std().rank(axis=1, pct=True)
    return mom * (vol < vol_pct).astype(float)


def _mean_rev_volume_confirm(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Mean reversion with volume confirmation: only take mean-rev when volume spikes."""
    rev_lb = params["rev_lookback"]
    vol_short = params.get("vol_short", 5)
    vol_long = params.get("vol_long", 60)
    vol_thresh = params.get("vol_threshold", 1.5)
    rev = -(close / close.rolling(rev_lb).mean() - 1.0)
    vol_ratio = volume.rolling(vol_short).mean() / volume.rolling(vol_long).mean().replace(0, np.nan)
    return rev * (vol_ratio > vol_thresh).astype(float)


def _skewness_signal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Return skewness: short positive-skew (lottery) assets, long negative-skew."""
    lb = params["lookback"]
    return -returns.rolling(lb).skew()


def _autocorrelation_signal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Return autocorrelation: positive = trending, negative = mean-reverting."""
    lb = params["lookback"]
    sign = params.get("sign", 1)  # 1=trend follow, -1=fade autocorr
    ac = returns.rolling(lb).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else np.nan,
        raw=False,
    )
    return sign * ac


def _max_drawdown_signal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Rolling max drawdown: long assets with shallow drawdowns (quality)."""
    lb = params["lookback"]
    def _dd(x):
        if len(x) < 2:
            return np.nan
        eq = np.cumprod(1 + x)
        return float(np.min(eq / np.maximum.accumulate(eq)) - 1)
    dd = returns.rolling(lb).apply(_dd, raw=True)
    return dd.rank(axis=1, pct=True)  # higher rank = shallower DD = better


def _tsmom(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Time-series momentum: each asset vs its own history (not cross-sectional)."""
    lb = params["lookback"]
    vol_lb = params.get("vol_lookback", 20)
    ret = close / close.shift(lb) - 1.0
    vol = returns.rolling(vol_lb).std().replace(0, np.nan)
    return ret / vol  # vol-scaled, so comparable across assets


def _tsmom_binary(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Binary TSMOM: +1 if lookback return positive, -1 otherwise."""
    lb = params["lookback"]
    ret = close / close.shift(lb) - 1.0
    return ret.apply(np.sign)


def _hl_spread(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """High-low spread (Corwin-Schultz proxy): higher = more illiquid."""
    lb = params["lookback"]
    # Use close as proxy since we have wide-format; rank inversely
    roll_high = close.rolling(lb).max()
    roll_low = close.rolling(lb).min()
    spread = (roll_high - roll_low) / ((roll_high + roll_low) / 2)
    return -spread.rank(axis=1, pct=True)  # long liquid, short illiquid


def _amihud_illiquidity(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Amihud illiquidity ratio: |return| / dollar volume."""
    lb = params["lookback"]
    dollar_vol = close * volume
    illiq = returns.abs() / dollar_vol.replace(0, np.nan)
    avg_illiq = illiq.rolling(lb).mean()
    return -avg_illiq.rank(axis=1, pct=True)  # long liquid, short illiquid


def _dollar_volume_momentum(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Dollar-volume weighted momentum: momentum scaled by liquidity."""
    mom_lb = params["mom_lookback"]
    vol_lb = params.get("vol_lookback", 20)
    mom = (close / close.shift(mom_lb) - 1.0).rank(axis=1, pct=True)
    dv = (close * volume).rolling(vol_lb).mean()
    dv_rank = dv.rank(axis=1, pct=True)
    return mom * dv_rank  # high-momentum + high-liquidity


def _hurst_signal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Simplified Hurst exponent: variance ratio as trend/mean-rev indicator."""
    lb = params["lookback"]
    var_1 = returns.rolling(lb).var()
    var_2 = returns.rolling(lb * 2).var()
    ratio = var_2 / (2 * var_1.replace(0, np.nan))
    sign = params.get("sign", 1)  # 1=trend (H>0.5), -1=mean-rev (H<0.5)
    return sign * (ratio - 1.0)


def _gap_signal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Overnight gap fade: short large up-gaps, long large down-gaps (24h crypto)."""
    lb = params["lookback"]
    # In crypto, "gap" is approximated by return over short period
    short_ret = close / close.shift(1) - 1.0
    avg_ret = short_ret.rolling(lb).mean()
    std_ret = short_ret.rolling(lb).std().replace(0, np.nan)
    z = (short_ret - avg_ret) / std_ret
    return -z  # fade extreme moves


def _multi_lookback_composite(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Multi-lookback composite: blend of multiple momentum lookbacks."""
    lookbacks = params["lookbacks"]
    weights = params.get("weights", [1.0 / len(lookbacks)] * len(lookbacks))
    composite = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for lb, w in zip(lookbacks, weights):
        mom = (close / close.shift(lb) - 1.0).rank(axis=1, pct=True) - 0.5
        composite += w * mom
    return composite


def _carry_signal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Yield/carry proxy: rolling return per unit of volatility (risk-adjusted carry)."""
    ret_lb = params["ret_lookback"]
    vol_lb = params["vol_lookback"]
    avg_ret = returns.rolling(ret_lb).mean() * ANN_FACTOR
    vol = returns.rolling(vol_lb).std() * np.sqrt(ANN_FACTOR)
    return (avg_ret / vol.replace(0, np.nan)).rank(axis=1, pct=True)


def _beta_signal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Low-beta factor: short high-beta, long low-beta assets."""
    lb = params["lookback"]
    btc_col = None
    for col in returns.columns:
        if "BTC" in col.upper():
            btc_col = col
            break
    if btc_col is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    btc_ret = returns[btc_col]
    btc_var = btc_ret.rolling(lb).var().replace(0, np.nan)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        if col == btc_col:
            continue
        cov = returns[col].rolling(lb).cov(btc_ret)
        result[col] = cov / btc_var
    return -result.rank(axis=1, pct=True)  # low beta = high rank


def _idio_vol_signal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Idiosyncratic volatility: residual vol after market factor (low idio vol = quality)."""
    lb = params["lookback"]
    mkt_ret = returns.mean(axis=1)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        resid = returns[col] - mkt_ret
        result[col] = resid.rolling(lb).std()
    return -result.rank(axis=1, pct=True)  # low idio vol = high rank


# =====================================================================
# NOVEL ALPHA SIGNALS — crypto-native, microstructure, cross-asset
# =====================================================================

def _close_location_value(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Close Location Value: where does close sit in the high-low range?
    Persistently near highs = accumulation, near lows = distribution.
    Uses rolling average of daily CLV as a smoothed signal."""
    # We don't have high/low in wide format, so approximate from close
    lb = params["lookback"]
    hi = close.rolling(lb).max()
    lo = close.rolling(lb).min()
    rng = hi - lo
    clv = (2 * close - hi - lo) / rng.replace(0, np.nan)
    return clv.rolling(params.get("smooth", 5)).mean()


def _chaikin_money_flow(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Chaikin Money Flow: CLV multiplied by volume, accumulated over window.
    Positive = buying pressure, negative = selling pressure."""
    lb = params["lookback"]
    hi = close.rolling(5).max()
    lo = close.rolling(5).min()
    rng = hi - lo
    clv = (2 * close - hi - lo) / rng.replace(0, np.nan)
    mf_vol = clv * volume
    cmf = mf_vol.rolling(lb).sum() / volume.rolling(lb).sum().replace(0, np.nan)
    return cmf


def _up_volume_ratio(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Up-volume ratio: fraction of volume on positive-return days.
    High ratio = accumulation, low = distribution."""
    lb = params["lookback"]
    up_mask = (returns > 0).astype(float)
    up_vol = (volume * up_mask).rolling(lb).sum()
    total_vol = volume.rolling(lb).sum().replace(0, np.nan)
    return (up_vol / total_vol).rank(axis=1, pct=True)


def _volume_acceleration(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Volume acceleration: second derivative of volume.
    Accelerating volume may precede breakouts."""
    lb = params["lookback"]
    vol_ma = volume.rolling(lb).mean()
    vol_delta = vol_ma - vol_ma.shift(lb)
    vol_accel = vol_delta - vol_delta.shift(lb)
    return vol_accel.rank(axis=1, pct=True)


def _volume_price_divergence(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Volume-price divergence: price trend vs volume trend disagreement.
    Price up + volume down = bearish divergence; price down + volume up = capitulation."""
    lb = params["lookback"]
    price_trend = (close / close.shift(lb) - 1.0).rank(axis=1, pct=True)
    vol_trend = (volume.rolling(lb).mean() / volume.rolling(lb * 2).mean().replace(0, np.nan)).rank(axis=1, pct=True)
    divergence = vol_trend - price_trend
    sign = params.get("sign", 1)  # 1=contrarian (buy capitulation), -1=follow volume
    return sign * divergence


def _dispersion_signal(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Cross-sectional dispersion: when return dispersion is high, extremes
    have more signal. Scale asset conviction by current dispersion."""
    lb = params["lookback"]
    dispersion = returns.std(axis=1).rolling(lb).mean()
    disp_z = (dispersion - dispersion.rolling(lb * 3).mean()) / dispersion.rolling(lb * 3).std().replace(0, np.nan)
    # When dispersion is high, amplify cross-sectional rank signal
    base = returns.rolling(lb).mean().rank(axis=1, pct=True) - 0.5
    return base * disp_z.clip(lower=0).values[:, None]


def _absorption_ratio(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Absorption ratio proxy: fraction of total variance from top assets.
    High absorption = fragile market, reduce exposure to high-beta names."""
    lb = params["lookback"]
    n_top = params.get("n_top", 5)
    var_per_asset = returns.rolling(lb).var()
    total_var = var_per_asset.sum(axis=1)
    top_var = var_per_asset.apply(lambda row: row.nlargest(n_top).sum(), axis=1)
    ar = top_var / total_var.replace(0, np.nan)
    ar_z = (ar - ar.rolling(lb * 2).mean()) / ar.rolling(lb * 2).std().replace(0, np.nan)
    # When market is fragile (high AR), favor low-vol assets
    vol = returns.rolling(lb).std().rank(axis=1, pct=True)
    return -vol * ar_z.clip(lower=0).values[:, None]


def _tail_risk_premium(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Tail risk premium: assets with fatter left tails should earn higher
    expected returns as compensation. Long fat-left-tail, short thin-left-tail."""
    lb = params["lookback"]
    def _left_tail(x):
        if len(x) < 10:
            return np.nan
        q05 = np.percentile(x, 5)
        return float(np.mean(x[x <= q05])) if np.any(x <= q05) else np.nan
    left_tail = returns.rolling(lb).apply(_left_tail, raw=True)
    return left_tail.rank(axis=1, pct=True)  # fatter left tail = higher rank = go long


def _return_concentration(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Return concentration: fraction of total return from best single day.
    High concentration = fragile/momentum-driven, low = steady accumulation."""
    lb = params["lookback"]
    def _conc(x):
        if len(x) < 5:
            return np.nan
        total = np.sum(np.abs(x))
        if total < 1e-12:
            return np.nan
        return float(np.max(np.abs(x)) / total)
    conc = returns.rolling(lb).apply(_conc, raw=True)
    return -conc.rank(axis=1, pct=True)  # low concentration = steady = go long


def _vol_term_structure(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Volatility term structure: short-term vol / long-term vol ratio.
    Inverted (>1) = stress/mean-reversion regime. Normal (<1) = trend regime."""
    short = params["short_window"]
    long = params["long_window"]
    vol_s = returns.rolling(short).std()
    vol_l = returns.rolling(long).std().replace(0, np.nan)
    ratio = vol_s / vol_l
    sign = params.get("sign", -1)  # -1=fade inverted structure (short stressed assets)
    return sign * (ratio - 1.0)


def _overnight_vs_intraday(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Overnight vs intraday return decomposition proxy.
    In crypto '24h market', approximate using lagged returns at different horizons.
    Signal: assets where recent short-term returns conflict with medium-term trend."""
    short = params.get("short", 1)
    medium = params.get("medium", 21)
    ret_short = close / close.shift(short) - 1.0
    ret_med = close / close.shift(medium) - 1.0
    short_rank = ret_short.rank(axis=1, pct=True)
    med_rank = ret_med.rank(axis=1, pct=True)
    # Divergence: strong medium-term but weak short-term = buy the dip
    return med_rank - short_rank


def _lead_lag_btc(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """BTC lead-lag: some altcoins lead BTC, others lag.
    Buy assets whose returns predict BTC's next-day return (informed flow)."""
    lb = params["lookback"]
    lag = params.get("lag", 1)
    btc_col = None
    for col in returns.columns:
        if "BTC" in col.upper():
            btc_col = col
            break
    if btc_col is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    btc_future = returns[btc_col].shift(-lag)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        if col == btc_col:
            continue
        result[col] = returns[col].rolling(lb).corr(btc_future)
    return result.rank(axis=1, pct=True)


def _btc_conditional_reversal(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """BTC-conditional reversal: mean-reversion signal active only when
    BTC is in a drawdown (stress = higher reversal probability)."""
    rev_lb = params["rev_lookback"]
    dd_lb = params.get("dd_lookback", 42)
    btc_col = None
    for col in close.columns:
        if "BTC" in col.upper():
            btc_col = col
            break
    if btc_col is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    btc_dd = close[btc_col] / close[btc_col].rolling(dd_lb).max() - 1.0
    stress = (btc_dd < -0.10).astype(float)
    rev = -(close / close.rolling(rev_lb).mean() - 1.0)
    return rev.mul(stress, axis=0)


def _btc_conditional_momentum(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """BTC-conditional momentum: momentum signal active only when
    BTC is trending up (momentum works better in trends)."""
    mom_lb = params["mom_lookback"]
    btc_lb = params.get("btc_lookback", 21)
    btc_col = None
    for col in close.columns:
        if "BTC" in col.upper():
            btc_col = col
            break
    if btc_col is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    btc_trend = (close[btc_col] / close[btc_col].shift(btc_lb) - 1.0) > 0
    trend_mask = btc_trend.astype(float)
    mom = (close / close.shift(mom_lb) - 1.0).rank(axis=1, pct=True) - 0.5
    return mom.mul(trend_mask, axis=0)


def _reversal_on_volume(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Reversal conditional on abnormal volume: fade moves that happened
    on unusually high volume (capitulation/euphoria overreaction)."""
    rev_lb = params["rev_lookback"]
    vol_lb = params.get("vol_lookback", 20)
    vol_thresh = params.get("vol_threshold", 2.0)
    vol_z = (volume - volume.rolling(vol_lb).mean()) / volume.rolling(vol_lb).std().replace(0, np.nan)
    high_vol_mask = (vol_z.rolling(rev_lb).max() > vol_thresh).astype(float)
    rev = -(close / close.shift(rev_lb) - 1.0)
    return rev * high_vol_mask


def _information_discreteness(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Information discreteness (Akbas et al.): decompose momentum into
    continuous (many small moves) vs discrete (few big jumps).
    Continuous momentum is more persistent."""
    lb = params["lookback"]
    total_ret = close / close.shift(lb) - 1.0
    sign_sum = returns.rolling(lb).apply(lambda x: np.sum(np.sign(x)), raw=True)
    abs_sign_sum = sign_sum.abs()
    n_days = lb
    id_score = abs_sign_sum / n_days
    # High ID = continuous momentum (all days same direction)
    # Weight momentum by ID to prefer continuous moves
    mom_rank = total_ret.rank(axis=1, pct=True) - 0.5
    return mom_rank * id_score


def _weekend_effect(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Weekend/weekday seasonality: some assets have persistent day-of-week patterns.
    Compute average return by day-of-week and use as forward signal."""
    lb = params["lookback"]
    dow = pd.Series(close.index).dt.dayofweek.values
    result = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for d in range(7):
        mask = dow == d
        if mask.sum() < 10:
            continue
        day_ret = returns.copy()
        day_ret[~mask] = np.nan
        avg = day_ret.rolling(lb, min_periods=lb // 3).mean()
        result[mask] = avg.loc[mask].values
    return result.rank(axis=1, pct=True)


def _entropy_signal(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Return entropy: Shannon entropy of discretized returns.
    Low entropy = structured/predictable, high = noisy. Long low-entropy assets."""
    lb = params["lookback"]
    n_bins = params.get("n_bins", 10)
    def _entropy(x):
        if len(x) < 10:
            return np.nan
        counts, _ = np.histogram(x, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))
    ent = returns.rolling(lb).apply(_entropy, raw=True)
    return -ent.rank(axis=1, pct=True)  # low entropy = go long


def _vol_of_vol(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Volatility of volatility: stability of the volatility process itself.
    Low vol-of-vol = stable risk profile = go long."""
    vol_lb = params["vol_lookback"]
    vov_lb = params["vov_lookback"]
    vol = returns.rolling(vol_lb).std()
    vov = vol.rolling(vov_lb).std()
    return -vov.rank(axis=1, pct=True)


def _correlation_breakout(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Correlation breakout: assets whose correlation with the market is
    changing rapidly. Decorrelating assets may be starting independent trends."""
    lb = params["lookback"]
    mkt = returns.mean(axis=1)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        corr = returns[col].rolling(lb).corr(mkt)
        corr_prev = corr.shift(lb)
        result[col] = corr_prev - corr  # decorrelation = positive
    return result.rank(axis=1, pct=True)


def _vwap_distance(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """VWAP distance proxy: volume-weighted average price approximation.
    Price above VWAP = buyers in control, below = sellers."""
    lb = params["lookback"]
    dv = close * volume
    vwap = dv.rolling(lb).sum() / volume.rolling(lb).sum().replace(0, np.nan)
    return ((close - vwap) / vwap.replace(0, np.nan)).rank(axis=1, pct=True)


def _realized_corr_with_market(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Market correlation factor: long low-corr assets, short high-corr.
    Low-correlation assets provide diversification value."""
    lb = params["lookback"]
    mkt = returns.mean(axis=1)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        result[col] = returns[col].rolling(lb).corr(mkt)
    return -result.rank(axis=1, pct=True)  # low corr = go long


def _momentum_consistency(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Momentum consistency: fraction of sub-periods with positive returns.
    More consistent momentum is more reliable than one big jump."""
    lb = params["lookback"]
    n_sub = params.get("n_sub", 4)
    sub_len = max(lb // n_sub, 1)
    consistency = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i in range(n_sub):
        shift = i * sub_len
        sub_ret = close.shift(shift) / close.shift(shift + sub_len) - 1.0
        consistency += (sub_ret > 0).astype(float)
    return (consistency / n_sub).rank(axis=1, pct=True)


def _downside_beta(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Downside beta: beta computed only on days when market is down.
    Low downside beta = defensive asset."""
    lb = params["lookback"]
    btc_col = None
    for col in returns.columns:
        if "BTC" in col.upper():
            btc_col = col
            break
    if btc_col is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    btc_ret = returns[btc_col]
    down_mask = btc_ret < 0
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        if col == btc_col:
            continue
        down_ret = returns[col].where(down_mask, np.nan)
        down_btc = btc_ret.where(down_mask, np.nan)
        cov = down_ret.rolling(lb, min_periods=lb // 2).cov(down_btc)
        var = down_btc.rolling(lb, min_periods=lb // 2).var().replace(0, np.nan)
        result[col] = cov / var
    return -result.rank(axis=1, pct=True)


def _fractal_dimension(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Fractal dimension proxy (Higuchi-like): measures path roughness.
    Low FD = smooth trend, high FD = choppy noise. Long low-FD assets."""
    lb = params["lookback"]
    def _fd(x):
        if len(x) < 5:
            return np.nan
        n = len(x)
        L1 = np.sum(np.abs(np.diff(x)))
        L2 = np.sum(np.abs(x[::2][1:] - x[::2][:-1])) if len(x[::2]) > 1 else L1
        if L2 < 1e-12:
            return np.nan
        return float(np.log(L1 / L2) / np.log(2))
    fd = close.rolling(lb).apply(_fd, raw=True)
    return -fd.rank(axis=1, pct=True)


def _coint_spread_btc(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Cointegration spread with BTC: z-score of price ratio.
    Mean-reverts when spread is extreme."""
    lb = params["lookback"]
    btc_col = None
    for col in close.columns:
        if "BTC" in col.upper():
            btc_col = col
            break
    if btc_col is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    btc_price = close[btc_col]
    result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for col in close.columns:
        if col == btc_col:
            continue
        ratio = np.log(close[col].replace(0, np.nan)) - np.log(btc_price.replace(0, np.nan))
        ratio_ma = ratio.rolling(lb).mean()
        ratio_std = ratio.rolling(lb).std().replace(0, np.nan)
        result[col] = -(ratio - ratio_ma) / ratio_std  # fade extremes
    return result


def _volume_rank_stability(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Volume rank stability: assets with consistent dollar-volume ranking.
    Stable high-rank = institutional presence, low turnover."""
    lb = params["lookback"]
    dv = close * volume
    dv_rank = dv.rank(axis=1, pct=True)
    rank_vol = dv_rank.rolling(lb).std()
    return -rank_vol.rank(axis=1, pct=True)  # stable rank = go long


def _high_volume_return_premium(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """High-volume return premium: average return on high-volume days.
    Tells you whether big-volume days are bullish or bearish for each asset."""
    lb = params["lookback"]
    vol_rank = volume.rank(axis=1, pct=True)
    high_vol_mask = vol_rank > 0.8
    hv_ret = returns.where(high_vol_mask, np.nan)
    return hv_ret.rolling(lb, min_periods=lb // 3).mean().rank(axis=1, pct=True)


def _relative_drawdown(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Relative drawdown: cross-sectional rank of current drawdown depth.
    Assets in deeper drawdown relative to peers = potential reversal targets."""
    lb = params["lookback"]
    peak = close.rolling(lb).max()
    dd = close / peak - 1.0
    sign = params.get("sign", 1)  # 1=buy deep drawdown (contrarian), -1=sell
    return sign * dd.rank(axis=1, pct=True)


def _momentum_duration(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Momentum duration: consecutive days of positive/negative returns.
    Long streaks may indicate persistent trend OR exhaustion."""
    lb = params["lookback"]
    sign = params.get("sign", 1)  # 1=follow streak, -1=fade streak
    pos = (returns > 0).astype(float)
    # Count consecutive positive days (reset on negative)
    streak = pos.copy()
    for i in range(1, len(streak)):
        streak.iloc[i] = (streak.iloc[i - 1] + pos.iloc[i]) * pos.iloc[i]
    avg_streak = streak.rolling(lb).mean()
    return (sign * avg_streak).rank(axis=1, pct=True)


def _smart_beta_quality(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Smart-beta quality composite: blend of low-vol, low-DD, stable volume.
    Multi-factor quality signal."""
    lb = params["lookback"]
    vol = returns.rolling(lb).std().rank(axis=1, pct=True)
    def _dd(x):
        if len(x) < 2:
            return np.nan
        eq = np.cumprod(1 + x)
        return float(np.min(eq / np.maximum.accumulate(eq)) - 1)
    dd = returns.rolling(lb).apply(_dd, raw=True).rank(axis=1, pct=True)
    dv = (close * volume).rolling(lb).mean().rank(axis=1, pct=True)
    return (-vol * 0.4 + dd * 0.3 + dv * 0.3)


def _vol_regime_interaction(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Volatility-regime interaction: asset vol relative to market vol.
    In high-market-vol, prefer assets with LOWER relative vol (safety).
    In low-market-vol, prefer assets with HIGHER relative vol (optionality)."""
    lb = params["lookback"]
    asset_vol = returns.rolling(lb).std()
    mkt_vol = returns.mean(axis=1).rolling(lb).std()
    rel_vol = asset_vol.div(mkt_vol, axis=0)
    mkt_vol_z = (mkt_vol - mkt_vol.rolling(lb * 3).mean()) / mkt_vol.rolling(lb * 3).std().replace(0, np.nan)
    # High market vol -> want low rel vol (negative sign)
    # Low market vol -> want high rel vol (positive sign)
    return (-mkt_vol_z.values[:, None] * rel_vol).rank(axis=1, pct=True)


def _cross_sectional_reversal_z(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Cross-sectional reversal z-score: how extreme is this asset's recent
    return relative to the cross-section? Fade extremes."""
    lb = params["lookback"]
    ret = close / close.shift(lb) - 1.0
    cs_mean = ret.mean(axis=1)
    cs_std = ret.std(axis=1).replace(0, np.nan)
    z = ret.sub(cs_mean, axis=0).div(cs_std, axis=0)
    return -z  # fade extremes


def _btc_sensitivity_change(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """BTC sensitivity change: rate of change of beta to BTC.
    Assets becoming more sensitive to BTC may be entering risk-on mode."""
    lb = params["lookback"]
    btc_col = None
    for col in returns.columns:
        if "BTC" in col.upper():
            btc_col = col
            break
    if btc_col is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    btc_ret = returns[btc_col]
    btc_var = btc_ret.rolling(lb).var().replace(0, np.nan)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        if col == btc_col:
            continue
        cov = returns[col].rolling(lb).cov(btc_ret)
        beta = cov / btc_var
        result[col] = beta - beta.shift(lb)
    sign = params.get("sign", -1)  # -1=short rising-beta (getting riskier)
    return (sign * result).rank(axis=1, pct=True)


def _price_momentum_gap(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Price-momentum gap: difference between short-term and long-term momentum rank.
    Large positive gap = recent acceleration, large negative = deceleration."""
    short = params["short_lookback"]
    long = params["long_lookback"]
    mom_short = (close / close.shift(short) - 1.0).rank(axis=1, pct=True)
    mom_long = (close / close.shift(long) - 1.0).rank(axis=1, pct=True)
    sign = params.get("sign", 1)  # 1=follow acceleration, -1=fade it
    return sign * (mom_short - mom_long)


def _value_signal(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Crypto value: distance from long-term average price.
    Long assets trading far below their historical average (cheap)."""
    lb = params["lookback"]
    log_price = np.log(close.replace(0, np.nan))
    avg = log_price.rolling(lb).mean()
    std = log_price.rolling(lb).std().replace(0, np.nan)
    z = (log_price - avg) / std
    return -z  # below average = cheap = go long


# =====================================================================
# ON-CHAIN SIGNALS — use BTC blockchain metrics to generate cross-
# sectional or market-level signals for the whole crypto universe.
# On-chain data is passed via params["_onchain"] (a DataFrame).
# =====================================================================

def _get_onchain_feature(params: dict, feature: str) -> pd.Series | None:
    """Extract a single on-chain feature from params, return None if missing."""
    oc = params.get("_onchain")
    if oc is None or feature not in oc.columns:
        return None
    return oc[feature]


def _onchain_nvt_regime(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """NVT regime signal: when BTC NVT is high (overvalued network), sell
    high-beta altcoins; when NVT is low (undervalued), buy them.
    Uses asset beta to amplify the effect on high-beta names."""
    nvt = _get_onchain_feature(params, "nvt_ratio_28d")
    if nvt is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    lb = params.get("lookback", 90)
    nvt_z = (nvt - nvt.rolling(lb).mean()) / nvt.rolling(lb).std().replace(0, np.nan)
    # High NVT z = overvalued -> short high-vol, long low-vol
    vol = returns.rolling(20).std().rank(axis=1, pct=True)
    return (-nvt_z.values[:, None] * (vol - 0.5))


def _onchain_hash_rate_momentum(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Hash rate momentum: rising hash rate = miner confidence = bullish.
    Go long high-beta altcoins when hash rate is accelerating."""
    feat = params.get("feature", "hash_rate_mom_30d")
    hr_mom = _get_onchain_feature(params, feat)
    if hr_mom is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    # Positive hash rate momentum -> go long high-beta names
    beta_proxy = returns.rolling(60).corr(returns.mean(axis=1)).rank(axis=1, pct=True)
    return hr_mom.values[:, None] * (beta_proxy - 0.5)


def _onchain_difficulty_ribbon(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Difficulty ribbon compression: when ribbon is compressed/inverted,
    miners are capitulating -> historically a buying opportunity.
    Buy quality (low-vol) coins during compression."""
    diff_z = _get_onchain_feature(params, "diff_ribbon_z")
    if diff_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    # Negative diff_ribbon_z = compressed ribbon = capitulation = buy
    vol = returns.rolling(20).std().rank(axis=1, pct=True)
    buy_signal = (-diff_z).clip(lower=0)
    return buy_signal.values[:, None] * (1 - vol)  # buy low-vol during capitulation


def _onchain_tx_activity(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Transaction activity signal: rising on-chain activity = growing demand.
    Go long when BTC tx count is accelerating."""
    feat = params.get("feature", "tx_count_z")
    tx_z = _get_onchain_feature(params, feat)
    if tx_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    mom = (close / close.shift(21) - 1.0).rank(axis=1, pct=True) - 0.5
    return tx_z.clip(lower=0).values[:, None] * mom


def _onchain_active_addresses(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Active addresses signal: growth in unique addresses = adoption.
    Amplify momentum signals when address growth is positive."""
    feat = params.get("feature", "active_addr_z")
    addr_z = _get_onchain_feature(params, feat)
    if addr_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    mom = (close / close.shift(params.get("lookback", 21)) - 1.0).rank(axis=1, pct=True) - 0.5
    return addr_z.clip(lower=0).values[:, None] * mom


def _onchain_fee_pressure(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Fee pressure signal: high fees = congestion = potential top.
    Short overbought assets when fee pressure is extreme."""
    fee_z = _get_onchain_feature(params, "fee_pressure_z")
    if fee_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    # High fees -> sell extended names, buy beaten-down
    rev = -(close / close.rolling(params.get("lookback", 14)).mean() - 1.0)
    fee_extreme = (fee_z > 1.5).astype(float)
    return rev.mul(fee_extreme.values, axis=0)


def _onchain_miner_revenue(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Miner revenue per hash signal: declining efficiency = stress = contrarian buy.
    Rising efficiency = expansion = risk-on."""
    rph_z = _get_onchain_feature(params, "revenue_per_hash_z")
    if rph_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    sign = params.get("sign", 1)  # 1=pro-cyclical, -1=contrarian
    vol = returns.rolling(20).std().rank(axis=1, pct=True)
    return (sign * rph_z.values[:, None]) * (0.5 - vol)


def _onchain_mempool_congestion(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Mempool congestion: high mempool = urgency = potential reversal.
    Mean-revert when mempool is extremely congested."""
    mem_z = _get_onchain_feature(params, "mempool_z")
    if mem_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    rev = -(close / close.rolling(params.get("lookback", 10)).mean() - 1.0)
    congested = (mem_z.abs() > 1.5).astype(float)
    return rev.mul(congested.values, axis=0)


def _onchain_utxo_growth(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """UTXO growth signal: growing UTXO set = more HODLers = bullish.
    Weight momentum by UTXO growth."""
    feat = params.get("feature", "utxo_growth_30d")
    utxo_g = _get_onchain_feature(params, feat)
    if utxo_g is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    mom = (close / close.shift(params.get("lookback", 21)) - 1.0).rank(axis=1, pct=True) - 0.5
    return utxo_g.clip(lower=0).values[:, None] * mom


def _onchain_velocity(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Velocity signal: transaction volume / market cap. High velocity = active
    network usage relative to valuation = undervalued. Low velocity = speculative."""
    vel_z = _get_onchain_feature(params, "velocity_z")
    if vel_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    sign = params.get("sign", 1)  # 1=buy when velocity high, -1=sell
    vol = returns.rolling(20).std().rank(axis=1, pct=True)
    return (sign * vel_z.values[:, None]) * (1 - vol)


def _onchain_supply_squeeze(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Supply squeeze: declining inflation + rising demand = bullish pressure.
    Combine supply inflation with active address growth."""
    inflation = _get_onchain_feature(params, "supply_inflation_30d")
    addr_z = _get_onchain_feature(params, "active_addr_z")
    if inflation is None or addr_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    # Low inflation + high address growth = squeeze
    squeeze = addr_z - inflation * 100  # scale inflation to be comparable
    mom = (close / close.shift(21) - 1.0).rank(axis=1, pct=True) - 0.5
    return squeeze.clip(lower=0).values[:, None] * mom


def _onchain_composite(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """On-chain composite: blend of NVT, hash rate, and activity signals.
    A multi-factor on-chain score."""
    oc = params.get("_onchain")
    if oc is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    components = []
    weights = params.get("weights", [0.3, 0.3, 0.2, 0.2])
    features = params.get("features", [
        "nvt_ratio_28d", "hash_rate_z", "active_addr_z", "velocity_z"
    ])
    for feat in features:
        if feat in oc.columns:
            s = oc[feat]
            if "nvt" in feat:
                s = -s  # low NVT = bullish
            components.append(s)
    if not components:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    composite = pd.Series(0.0, index=close.index)
    for c, w in zip(components, weights[:len(components)]):
        z = (c - c.rolling(90).mean()) / c.rolling(90).std().replace(0, np.nan)
        composite += w * z.fillna(0)
    mom = (close / close.shift(21) - 1.0).rank(axis=1, pct=True) - 0.5
    return composite.values[:, None] * mom


def _onchain_hash_regime(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Hash rate regime: classify into growth/stress based on z-score.
    Different allocation depending on hash rate regime."""
    hr_z = _get_onchain_feature(params, "hash_rate_z")
    if hr_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    # Growth regime (hr_z > 0): momentum works; Stress regime: reversal works
    growth = (hr_z > 0).astype(float)
    mom = (close / close.shift(21) - 1.0).rank(axis=1, pct=True) - 0.5
    rev = -(close / close.rolling(10).mean() - 1.0).rank(axis=1, pct=True) - 0.5
    return growth.values[:, None] * mom + (1 - growth.values[:, None]) * rev


def _onchain_cost_efficiency(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Cost per transaction efficiency: low cost = efficient network.
    When cost drops, network is efficient -> bullish for alts."""
    cpt_z = _get_onchain_feature(params, "cost_per_tx_z")
    if cpt_z is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    # Negative cpt_z = low cost = efficient -> bullish
    efficiency = -cpt_z.clip(upper=0)
    mom = (close / close.shift(21) - 1.0).rank(axis=1, pct=True) - 0.5
    return efficiency.values[:, None] * mom


# =====================================================================
# MORE NOVEL OHLCV SIGNALS — advanced statistical / structural
# =====================================================================

def _rolling_regression_alpha(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Rolling regression alpha: residual return after market factor.
    Positive alpha = outperforming on a risk-adjusted basis."""
    lb = params["lookback"]
    mkt = returns.mean(axis=1)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        cov = returns[col].rolling(lb).cov(mkt)
        var = mkt.rolling(lb).var().replace(0, np.nan)
        beta = cov / var
        alpha = returns[col].rolling(lb).mean() - beta * mkt.rolling(lb).mean()
        result[col] = alpha
    return result.rank(axis=1, pct=True)


def _sector_momentum(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Sector momentum proxy: group assets by correlation cluster,
    then go long assets in the strongest cluster. Uses k-means on
    rolling correlation to define pseudo-sectors."""
    lb = params["lookback"]
    n_clusters = params.get("n_clusters", 3)
    # Compute pairwise correlation matrix
    corr_mat = returns.rolling(lb).corr()
    # For efficiency, use a simpler approach: rank by avg correlation with
    # the top-performing assets
    ret_lb = close / close.shift(lb) - 1.0
    top_20pct_mask = ret_lb.rank(axis=1, pct=True) > 0.8
    # For each asset, compute average correlation with top performers
    result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for idx in close.index:
        top_assets = top_20pct_mask.loc[idx]
        if top_assets.sum() < 2:
            continue
        top_cols = top_assets[top_assets].index
        for col in close.columns:
            if col in top_cols:
                result.loc[idx, col] = 1.0
            else:
                corrs = []
                for tc in top_cols:
                    c = returns[col].rolling(lb).corr(returns[tc])
                    if idx in c.index and not pd.isna(c.loc[idx]):
                        corrs.append(c.loc[idx])
                if corrs:
                    result.loc[idx, col] = np.mean(corrs)
    return result.rank(axis=1, pct=True)


def _price_efficiency_ratio(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Price efficiency ratio (Kaufman): net price change / total path length.
    High ratio = trending, low ratio = choppy. Used to weight signals."""
    lb = params["lookback"]
    net_change = (close - close.shift(lb)).abs()
    path_length = returns.abs().rolling(lb).sum() * close.shift(lb)
    er = net_change / path_length.replace(0, np.nan)
    sign = params.get("sign", 1)  # 1=long trending, -1=long choppy
    return (sign * er).rank(axis=1, pct=True)


def _kalman_trend(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Simplified Kalman filter trend: exponentially weighted estimate of
    price level with adaptive smoothing based on volatility."""
    alpha = params.get("alpha", 0.1)
    vol_scale = params.get("vol_scale", 1.0)
    result = close.copy() * np.nan
    vol = returns.rolling(20).std()
    # Adaptive alpha: increase responsiveness in low-vol, decrease in high-vol
    adaptive_alpha = alpha / (1 + vol_scale * vol / vol.rolling(60).median())
    adaptive_alpha = adaptive_alpha.clip(0.01, 0.5)
    # Simple recursive Kalman-like filter
    state = close.iloc[0].copy()
    for i in range(len(close)):
        a = adaptive_alpha.iloc[i]
        obs = close.iloc[i]
        state = a * obs + (1 - a) * state
        result.iloc[i] = state
    # Signal: deviation from Kalman estimate
    deviation = (close - result) / result.replace(0, np.nan)
    sign = params.get("sign", -1)  # -1=mean revert to Kalman, +1=follow deviation
    return (sign * deviation).rank(axis=1, pct=True)


def _volume_clock_signal(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Volume clock: measure time between volume events.
    Assets with accelerating volume frequency may be entering active phases."""
    lb = params["lookback"]
    thresh = params.get("threshold", 1.5)
    vol_ma = volume.rolling(lb).mean()
    high_vol_days = (volume > thresh * vol_ma).astype(float)
    # Count high-volume days in recent window
    freq = high_vol_days.rolling(lb).sum()
    # Compare recent to historical
    freq_z = (freq - freq.rolling(lb * 2).mean()) / freq.rolling(lb * 2).std().replace(0, np.nan)
    return freq_z.rank(axis=1, pct=True)


def _copula_tail_dependence(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Tail dependence proxy: how correlated is the asset with BTC in the
    tails? Low tail dependence = relative safety in crashes."""
    lb = params["lookback"]
    btc_col = None
    for col in returns.columns:
        if "BTC" in col.upper():
            btc_col = col
            break
    if btc_col is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    btc_ret = returns[btc_col]
    btc_q10 = btc_ret.rolling(lb).quantile(0.1)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        if col == btc_col:
            continue
        both_tail = ((btc_ret < btc_q10) & (returns[col] < returns[col].rolling(lb).quantile(0.1))).astype(float)
        tail_dep = both_tail.rolling(lb).mean()
        result[col] = tail_dep
    return -result.rank(axis=1, pct=True)  # low tail dependence = safer


def _winner_loser_spread(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Winner-loser spread: buy recent losers that are winners on
    longer horizon (52-week high strategy adapted for crypto)."""
    short = params["short_lookback"]
    long = params["long_lookback"]
    short_ret = (close / close.shift(short) - 1.0).rank(axis=1, pct=True)
    long_ret = (close / close.shift(long) - 1.0).rank(axis=1, pct=True)
    return long_ret - short_ret  # long-term winner, short-term loser


def _conditional_skewness(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Conditional skewness: skewness of returns in down-market days only.
    Assets with less negative conditional skewness are safer."""
    lb = params["lookback"]
    mkt = returns.mean(axis=1)
    down_mask = mkt < mkt.rolling(lb).quantile(0.3)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        cond_ret = returns[col].where(down_mask, np.nan)
        result[col] = cond_ret.rolling(lb, min_periods=lb // 3).skew()
    return result.rank(axis=1, pct=True)  # less negative skew = better


def _volume_return_asymmetry(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Volume-return asymmetry: do big volume days have bullish or bearish returns?
    Computed as correlation between volume rank and return sign."""
    lb = params["lookback"]
    vol_rank = volume.rank(axis=1, pct=True)
    ret_sign = returns.apply(np.sign)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        result[col] = vol_rank[col].rolling(lb).corr(ret_sign[col])
    return result.rank(axis=1, pct=True)


def _mean_rev_strength(
    close: pd.DataFrame, volume: pd.DataFrame,
    returns: pd.DataFrame, params: dict,
) -> pd.DataFrame:
    """Mean reversion strength: how strongly does an asset revert?
    Measured by serial correlation of returns. Strong negative serial corr = good mean-rev target."""
    lb = params["lookback"]
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        ac = returns[col].rolling(lb).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 5 else np.nan,
            raw=False,
        )
        result[col] = ac
    # Strong negative autocorrelation = good mean-reversion target
    # Weight mean-rev signal by autocorrelation strength
    rev = -(close / close.rolling(lb // 2).mean() - 1.0)
    mr_strength = result.clip(upper=0).abs()
    return rev * mr_strength


# ---------------------------------------------------------------------------
# Signal function registry
# ---------------------------------------------------------------------------
SIGNAL_FUNCTIONS: dict[str, SignalFn] = {
    "momentum": _momentum,
    "momentum_cs_rank": _momentum_cs_rank,
    "vol_adjusted_momentum": _vol_adjusted_momentum,
    "mean_reversion": _mean_reversion,
    "rsi": _rsi,
    "low_volatility": _low_volatility,
    "vol_compression": _vol_compression,
    "vol_breakout": _vol_breakout,
    "volume_relative": _volume_relative,
    "obv_momentum": _obv_momentum,
    "dist_from_high": _dist_from_high,
    "dist_from_low": _dist_from_low,
    "range_position": _range_position,
    "ema_crossover": _ema_crossover,
    "acceleration": _acceleration,
    "price_volume_trend": _price_volume_trend,
    "relative_strength_btc": _relative_strength_btc,
    "momentum_reversal": _momentum_reversal,
    "momentum_vol_filter": _momentum_vol_filter,
    "mean_rev_volume_confirm": _mean_rev_volume_confirm,
    "skewness": _skewness_signal,
    "autocorrelation": _autocorrelation_signal,
    "max_drawdown_quality": _max_drawdown_signal,
    "tsmom": _tsmom,
    "tsmom_binary": _tsmom_binary,
    "hl_spread": _hl_spread,
    "amihud": _amihud_illiquidity,
    "dv_momentum": _dollar_volume_momentum,
    "hurst": _hurst_signal,
    "gap_fade": _gap_signal,
    "multi_lb": _multi_lookback_composite,
    "carry": _carry_signal,
    "beta": _beta_signal,
    "idio_vol": _idio_vol_signal,
    "clv": _close_location_value,
    "cmf": _chaikin_money_flow,
    "up_vol_ratio": _up_volume_ratio,
    "vol_accel": _volume_acceleration,
    "vpd": _volume_price_divergence,
    "dispersion": _dispersion_signal,
    "absorption": _absorption_ratio,
    "tail_risk": _tail_risk_premium,
    "ret_conc": _return_concentration,
    "vol_ts": _vol_term_structure,
    "overnight_intraday": _overnight_vs_intraday,
    "lead_lag_btc": _lead_lag_btc,
    "btc_cond_rev": _btc_conditional_reversal,
    "btc_cond_mom": _btc_conditional_momentum,
    "rev_on_vol": _reversal_on_volume,
    "info_discrete": _information_discreteness,
    "weekend": _weekend_effect,
    "entropy": _entropy_signal,
    "vov": _vol_of_vol,
    "corr_break": _correlation_breakout,
    "vwap_dist": _vwap_distance,
    "mkt_corr": _realized_corr_with_market,
    "mom_consist": _momentum_consistency,
    "down_beta": _downside_beta,
    "fractal_dim": _fractal_dimension,
    "coint_btc": _coint_spread_btc,
    "vol_rank_stab": _volume_rank_stability,
    "hv_ret_prem": _high_volume_return_premium,
    "rel_dd": _relative_drawdown,
    "mom_dur": _momentum_duration,
    "smart_quality": _smart_beta_quality,
    "vol_regime_ix": _vol_regime_interaction,
    "cs_rev_z": _cross_sectional_reversal_z,
    "btc_sens_chg": _btc_sensitivity_change,
    "pm_gap": _price_momentum_gap,
    "value": _value_signal,
    # On-chain signals
    "oc_nvt": _onchain_nvt_regime,
    "oc_hash_mom": _onchain_hash_rate_momentum,
    "oc_diff_ribbon": _onchain_difficulty_ribbon,
    "oc_tx_activity": _onchain_tx_activity,
    "oc_active_addr": _onchain_active_addresses,
    "oc_fee_pressure": _onchain_fee_pressure,
    "oc_miner_rev": _onchain_miner_revenue,
    "oc_mempool": _onchain_mempool_congestion,
    "oc_utxo": _onchain_utxo_growth,
    "oc_velocity": _onchain_velocity,
    "oc_supply_squeeze": _onchain_supply_squeeze,
    "oc_composite": _onchain_composite,
    "oc_hash_regime": _onchain_hash_regime,
    "oc_cost_eff": _onchain_cost_efficiency,
    # New OHLCV signals
    "rolling_alpha": _rolling_regression_alpha,
    "price_eff": _price_efficiency_ratio,
    "kalman": _kalman_trend,
    "vol_clock": _volume_clock_signal,
    "tail_dep": _copula_tail_dependence,
    "wl_spread": _winner_loser_spread,
    "cond_skew": _conditional_skewness,
    "vol_ret_asym": _volume_return_asymmetry,
    "mr_strength": _mean_rev_strength,
}


# ---------------------------------------------------------------------------
# Signal space enumeration
# ---------------------------------------------------------------------------

def _build_signal_space() -> list[SignalSpec]:
    """Enumerate the full signal space to be tested."""
    specs: list[SignalSpec] = []

    # --- Momentum family ---
    for lb in [5, 10, 21, 42, 63, 126, 252]:
        specs.append(SignalSpec(
            name=f"mom_{lb}d",
            family="momentum",
            params={"lookback": lb},
            description=f"{lb}-day price momentum",
        ))

    for ret_lb in [10, 21, 63]:
        for vol_lb in [20, 42]:
            specs.append(SignalSpec(
                name=f"vol_adj_mom_{ret_lb}d_v{vol_lb}d",
                family="momentum",
                params={"ret_lookback": ret_lb, "vol_lookback": vol_lb},
                description=f"Vol-adjusted momentum ({ret_lb}d ret / {vol_lb}d vol)",
            ))

    # --- Mean reversion family ---
    for lb in [5, 10, 21, 42, 63]:
        specs.append(SignalSpec(
            name=f"mean_rev_{lb}d",
            family="mean_reversion",
            params={"lookback": lb},
            description=f"{lb}-day mean reversion (distance from MA)",
        ))

    for lb in [7, 14, 21]:
        specs.append(SignalSpec(
            name=f"rsi_{lb}d",
            family="mean_reversion",
            params={"lookback": lb},
            description=f"{lb}-day RSI mean reversion",
        ))

    # --- Volatility family ---
    for lb in [10, 20, 42, 63]:
        specs.append(SignalSpec(
            name=f"low_vol_{lb}d",
            family="volatility",
            params={"lookback": lb},
            description=f"{lb}-day low-volatility factor",
        ))

    for short, long in [(5, 20), (5, 60), (10, 42), (10, 63)]:
        specs.append(SignalSpec(
            name=f"vol_compress_{short}_{long}d",
            family="volatility",
            params={"short_window": short, "long_window": long},
            description=f"Vol compression ({short}d / {long}d)",
        ))

    for lb in [10, 20]:
        for thresh in [1.5, 2.0]:
            specs.append(SignalSpec(
                name=f"vol_break_{lb}d_t{thresh}",
                family="volatility",
                params={"lookback": lb, "threshold": thresh},
                description=f"Vol breakout ({lb}d, threshold {thresh}x)",
            ))

    # --- Volume family ---
    for short, long in [(5, 20), (5, 60), (10, 42)]:
        specs.append(SignalSpec(
            name=f"vol_rel_{short}_{long}d",
            family="volume",
            params={"short_window": short, "long_window": long},
            description=f"Relative volume ({short}d / {long}d)",
        ))

    for lb in [10, 21, 42]:
        specs.append(SignalSpec(
            name=f"obv_mom_{lb}d",
            family="volume",
            params={"lookback": lb},
            description=f"{lb}-day OBV momentum",
        ))

    # --- Price structure family ---
    for lb in [10, 20, 42, 63]:
        specs.append(SignalSpec(
            name=f"dist_high_{lb}d",
            family="price_structure",
            params={"lookback": lb},
            description=f"Distance from {lb}-day high",
        ))
        specs.append(SignalSpec(
            name=f"dist_low_{lb}d",
            family="price_structure",
            params={"lookback": lb},
            description=f"Distance from {lb}-day low",
        ))
        specs.append(SignalSpec(
            name=f"range_pos_{lb}d",
            family="price_structure",
            params={"lookback": lb},
            description=f"Position in {lb}-day range",
        ))

    # --- EMA crossover family ---
    for fast, slow in [(5, 20), (10, 30), (10, 50), (21, 63), (21, 126)]:
        specs.append(SignalSpec(
            name=f"ema_x_{fast}_{slow}",
            family="trend",
            params={"fast": fast, "slow": slow},
            description=f"EMA crossover ({fast}/{slow})",
        ))

    # --- Acceleration ---
    for lb in [5, 10, 21]:
        specs.append(SignalSpec(
            name=f"accel_{lb}d",
            family="momentum",
            params={"lookback": lb},
            description=f"{lb}-day momentum acceleration",
        ))

    # --- Price-volume correlation ---
    for lb in [10, 21, 42]:
        specs.append(SignalSpec(
            name=f"pv_trend_{lb}d",
            family="volume",
            params={"lookback": lb},
            description=f"{lb}-day price-volume trend correlation",
        ))

    # --- Relative strength vs BTC ---
    for lb in [21, 42, 63, 126]:
        specs.append(SignalSpec(
            name=f"rs_btc_{lb}d",
            family="relative_strength",
            params={"lookback": lb},
            description=f"Relative strength vs BTC ({lb}d)",
        ))

    # --- Composite signals ---
    for mom_lb in [21, 63, 126]:
        for rev_lb in [5, 10]:
            specs.append(SignalSpec(
                name=f"mom_rev_{mom_lb}_{rev_lb}d",
                family="composite",
                params={"mom_lookback": mom_lb, "rev_lookback": rev_lb, "mom_weight": 0.6},
                description=f"Momentum-reversal composite ({mom_lb}d mom + {rev_lb}d rev)",
            ))

    for mom_lb in [21, 63]:
        for vol_lb in [20, 42]:
            specs.append(SignalSpec(
                name=f"mom_vf_{mom_lb}_{vol_lb}d",
                family="composite",
                params={"mom_lookback": mom_lb, "vol_lookback": vol_lb, "vol_percentile": 0.5},
                description=f"Momentum + low-vol filter ({mom_lb}d, {vol_lb}d vol < 50pct)",
            ))

    for rev_lb in [10, 21]:
        specs.append(SignalSpec(
            name=f"mr_vc_{rev_lb}d",
            family="composite",
            params={"rev_lookback": rev_lb, "vol_short": 5, "vol_long": 60, "vol_threshold": 1.5},
            description=f"Mean-reversion + volume confirmation ({rev_lb}d)",
        ))

    # --- Statistical signals ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"skew_{lb}d",
            family="statistical",
            params={"lookback": lb},
            description=f"{lb}-day return skewness (short lottery tickets)",
        ))

    for lb in [21, 42]:
        for sign in [1, -1]:
            label = "trend" if sign == 1 else "fade"
            specs.append(SignalSpec(
                name=f"autocorr_{label}_{lb}d",
                family="statistical",
                params={"lookback": lb, "sign": sign},
                description=f"{lb}-day autocorrelation ({label})",
            ))

    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"maxdd_quality_{lb}d",
            family="quality",
            params={"lookback": lb},
            description=f"{lb}-day max-drawdown quality factor",
        ))

    # --- Time-series momentum (TSMOM) ---
    for lb in [10, 21, 42, 63, 126, 252]:
        specs.append(SignalSpec(
            name=f"tsmom_{lb}d",
            family="tsmom",
            params={"lookback": lb, "vol_lookback": 20},
            description=f"{lb}-day time-series momentum (vol-scaled)",
        ))
        specs.append(SignalSpec(
            name=f"tsmom_bin_{lb}d",
            family="tsmom",
            params={"lookback": lb},
            description=f"{lb}-day binary TSMOM (sign of return)",
        ))

    # --- Liquidity signals ---
    for lb in [10, 20, 42]:
        specs.append(SignalSpec(
            name=f"hl_spread_{lb}d",
            family="liquidity",
            params={"lookback": lb},
            description=f"{lb}-day high-low spread (liquidity proxy)",
        ))
        specs.append(SignalSpec(
            name=f"amihud_{lb}d",
            family="liquidity",
            params={"lookback": lb},
            description=f"{lb}-day Amihud illiquidity ratio",
        ))

    # --- Dollar-volume momentum ---
    for mom_lb in [21, 63, 126]:
        specs.append(SignalSpec(
            name=f"dv_momentum_{mom_lb}d",
            family="composite",
            params={"mom_lookback": mom_lb, "vol_lookback": 20},
            description=f"Dollar-volume weighted momentum ({mom_lb}d)",
        ))

    # --- Hurst exponent proxy ---
    for lb in [21, 42, 63]:
        for sign in [1, -1]:
            label = "trend" if sign == 1 else "mrev"
            specs.append(SignalSpec(
                name=f"hurst_{label}_{lb}d",
                family="statistical",
                params={"lookback": lb, "sign": sign},
                description=f"{lb}-day Hurst proxy ({label})",
            ))

    # --- Gap fade ---
    for lb in [10, 21, 42]:
        specs.append(SignalSpec(
            name=f"gap_fade_{lb}d",
            family="mean_reversion",
            params={"lookback": lb},
            description=f"{lb}-day gap fade (short extreme up-moves)",
        ))

    # --- Multi-lookback composites ---
    specs.append(SignalSpec(
        name="multi_lb_fast",
        family="composite",
        params={"lookbacks": [5, 10, 21], "weights": [0.5, 0.3, 0.2]},
        description="Multi-lookback fast composite (5/10/21d)",
    ))
    specs.append(SignalSpec(
        name="multi_lb_medium",
        family="composite",
        params={"lookbacks": [21, 42, 63], "weights": [0.4, 0.35, 0.25]},
        description="Multi-lookback medium composite (21/42/63d)",
    ))
    specs.append(SignalSpec(
        name="multi_lb_slow",
        family="composite",
        params={"lookbacks": [63, 126, 252], "weights": [0.4, 0.35, 0.25]},
        description="Multi-lookback slow composite (63/126/252d)",
    ))
    specs.append(SignalSpec(
        name="multi_lb_full",
        family="composite",
        params={"lookbacks": [5, 21, 63, 126, 252], "weights": [0.1, 0.2, 0.3, 0.25, 0.15]},
        description="Multi-lookback full spectrum (5-252d)",
    ))

    # --- Carry proxy ---
    for ret_lb, vol_lb in [(21, 20), (42, 42), (63, 42)]:
        specs.append(SignalSpec(
            name=f"carry_{ret_lb}_{vol_lb}d",
            family="carry",
            params={"ret_lookback": ret_lb, "vol_lookback": vol_lb},
            description=f"Carry proxy ({ret_lb}d return / {vol_lb}d vol)",
        ))

    # --- Beta and idiosyncratic vol ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"beta_{lb}d",
            family="risk",
            params={"lookback": lb},
            description=f"{lb}-day low-beta factor (vs BTC)",
        ))
        specs.append(SignalSpec(
            name=f"idio_vol_{lb}d",
            family="risk",
            params={"lookback": lb},
            description=f"{lb}-day low idiosyncratic volatility",
        ))

    # =================================================================
    # NOVEL ALPHAS — crypto-native, microstructure, cross-asset
    # =================================================================

    # --- Close location value / Chaikin money flow ---
    for lb in [10, 20, 42]:
        for smooth in [3, 5, 10]:
            specs.append(SignalSpec(
                name=f"clv_{lb}d_s{smooth}",
                family="microstructure",
                params={"lookback": lb, "smooth": smooth},
                description=f"Close location value ({lb}d, smooth {smooth})",
            ))
    for lb in [10, 20, 42]:
        specs.append(SignalSpec(
            name=f"cmf_{lb}d",
            family="microstructure",
            params={"lookback": lb},
            description=f"Chaikin money flow ({lb}d)",
        ))

    # --- Volume dynamics ---
    for lb in [10, 21, 42]:
        specs.append(SignalSpec(
            name=f"up_vol_ratio_{lb}d",
            family="volume_dynamics",
            params={"lookback": lb},
            description=f"Up-volume ratio ({lb}d)",
        ))
        specs.append(SignalSpec(
            name=f"vol_accel_{lb}d",
            family="volume_dynamics",
            params={"lookback": lb},
            description=f"Volume acceleration ({lb}d)",
        ))

    for lb in [21, 42]:
        for sign in [1, -1]:
            label = "contra" if sign == 1 else "follow"
            specs.append(SignalSpec(
                name=f"vpd_{label}_{lb}d",
                family="volume_dynamics",
                params={"lookback": lb, "sign": sign},
                description=f"Volume-price divergence {label} ({lb}d)",
            ))

    # --- High-volume return premium ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"hv_ret_prem_{lb}d",
            family="volume_dynamics",
            params={"lookback": lb},
            description=f"High-volume return premium ({lb}d)",
        ))

    # --- Volume rank stability ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"vol_rank_stab_{lb}d",
            family="volume_dynamics",
            params={"lookback": lb},
            description=f"Volume rank stability ({lb}d)",
        ))

    # --- Cross-sectional / dispersion ---
    for lb in [10, 21, 42]:
        specs.append(SignalSpec(
            name=f"dispersion_{lb}d",
            family="cross_sectional",
            params={"lookback": lb},
            description=f"Dispersion-scaled signal ({lb}d)",
        ))

    for lb in [21, 42]:
        specs.append(SignalSpec(
            name=f"absorption_{lb}d",
            family="cross_sectional",
            params={"lookback": lb, "n_top": 5},
            description=f"Absorption ratio ({lb}d, top 5)",
        ))

    for lb in [5, 10, 21]:
        specs.append(SignalSpec(
            name=f"cs_rev_z_{lb}d",
            family="cross_sectional",
            params={"lookback": lb},
            description=f"Cross-sectional reversal z-score ({lb}d)",
        ))

    # --- Tail risk and return distribution ---
    for lb in [42, 63, 126]:
        specs.append(SignalSpec(
            name=f"tail_risk_{lb}d",
            family="distributional",
            params={"lookback": lb},
            description=f"Tail risk premium ({lb}d)",
        ))

    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"ret_conc_{lb}d",
            family="distributional",
            params={"lookback": lb},
            description=f"Return concentration ({lb}d)",
        ))

    for lb in [42, 63]:
        specs.append(SignalSpec(
            name=f"entropy_{lb}d",
            family="distributional",
            params={"lookback": lb, "n_bins": 10},
            description=f"Return entropy ({lb}d)",
        ))

    # --- Volatility term structure ---
    for short, long in [(5, 21), (5, 42), (10, 42), (10, 63), (21, 63)]:
        for sign in [-1, 1]:
            label = "fade" if sign == -1 else "follow"
            specs.append(SignalSpec(
                name=f"vol_ts_{label}_{short}_{long}d",
                family="volatility_structure",
                params={"short_window": short, "long_window": long, "sign": sign},
                description=f"Vol term structure {label} ({short}/{long}d)",
            ))

    # --- Vol-of-vol ---
    for vol_lb, vov_lb in [(10, 42), (20, 42), (20, 63)]:
        specs.append(SignalSpec(
            name=f"vov_{vol_lb}_{vov_lb}d",
            family="volatility_structure",
            params={"vol_lookback": vol_lb, "vov_lookback": vov_lb},
            description=f"Vol-of-vol ({vol_lb}d vol, {vov_lb}d window)",
        ))

    # --- BTC-conditional signals ---
    for rev_lb in [5, 10, 21]:
        specs.append(SignalSpec(
            name=f"btc_cond_rev_{rev_lb}d",
            family="btc_conditional",
            params={"rev_lookback": rev_lb, "dd_lookback": 42},
            description=f"BTC-conditional reversal ({rev_lb}d, active in BTC drawdowns)",
        ))

    for mom_lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"btc_cond_mom_{mom_lb}d",
            family="btc_conditional",
            params={"mom_lookback": mom_lb, "btc_lookback": 21},
            description=f"BTC-conditional momentum ({mom_lb}d, active in BTC uptrends)",
        ))

    for lb in [21, 42]:
        for sign in [-1, 1]:
            label = "short_rising" if sign == -1 else "long_rising"
            specs.append(SignalSpec(
                name=f"btc_sens_chg_{label}_{lb}d",
                family="btc_conditional",
                params={"lookback": lb, "sign": sign},
                description=f"BTC sensitivity change {label} ({lb}d)",
            ))

    # --- Lead-lag with BTC ---
    for lb in [21, 42, 63]:
        for lag in [1, 2]:
            specs.append(SignalSpec(
                name=f"lead_lag_btc_{lb}d_l{lag}",
                family="cross_asset",
                params={"lookback": lb, "lag": lag},
                description=f"BTC lead-lag ({lb}d, lag {lag}d)",
            ))

    # --- Cointegration with BTC ---
    for lb in [42, 63, 126]:
        specs.append(SignalSpec(
            name=f"coint_btc_{lb}d",
            family="cross_asset",
            params={"lookback": lb},
            description=f"Cointegration spread with BTC ({lb}d)",
        ))

    # --- Market correlation ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"mkt_corr_{lb}d",
            family="cross_asset",
            params={"lookback": lb},
            description=f"Low market correlation ({lb}d)",
        ))

    for lb in [21, 42]:
        specs.append(SignalSpec(
            name=f"corr_break_{lb}d",
            family="cross_asset",
            params={"lookback": lb},
            description=f"Correlation breakout ({lb}d)",
        ))

    # --- Downside beta ---
    for lb in [42, 63]:
        specs.append(SignalSpec(
            name=f"down_beta_{lb}d",
            family="risk",
            params={"lookback": lb},
            description=f"Low downside beta vs BTC ({lb}d)",
        ))

    # --- Reversal on volume ---
    for rev_lb in [3, 5, 10]:
        for vol_thresh in [1.5, 2.0, 2.5]:
            specs.append(SignalSpec(
                name=f"rev_on_vol_{rev_lb}d_t{vol_thresh}",
                family="microstructure",
                params={"rev_lookback": rev_lb, "vol_lookback": 20, "vol_threshold": vol_thresh},
                description=f"Reversal on abnormal volume ({rev_lb}d, vol>{vol_thresh}x)",
            ))

    # --- Information discreteness ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"info_discrete_{lb}d",
            family="momentum_quality",
            params={"lookback": lb},
            description=f"Information discreteness ({lb}d)",
        ))

    # --- Momentum consistency ---
    for lb in [42, 63, 126]:
        specs.append(SignalSpec(
            name=f"mom_consist_{lb}d",
            family="momentum_quality",
            params={"lookback": lb, "n_sub": 4},
            description=f"Momentum consistency ({lb}d, 4 sub-periods)",
        ))

    # --- Momentum duration ---
    for lb in [10, 21, 42]:
        for sign in [1, -1]:
            label = "follow" if sign == 1 else "fade"
            specs.append(SignalSpec(
                name=f"mom_dur_{label}_{lb}d",
                family="momentum_quality",
                params={"lookback": lb, "sign": sign},
                description=f"Momentum duration {label} ({lb}d)",
            ))

    # --- Price momentum gap (acceleration) ---
    for short, long in [(5, 21), (5, 63), (10, 42), (21, 63), (21, 126)]:
        for sign in [1, -1]:
            label = "accel" if sign == 1 else "decel"
            specs.append(SignalSpec(
                name=f"pm_gap_{label}_{short}_{long}d",
                family="momentum_quality",
                params={"short_lookback": short, "long_lookback": long, "sign": sign},
                description=f"Momentum gap {label} ({short}/{long}d)",
            ))

    # --- Overnight vs intraday decomposition ---
    for short in [1, 3]:
        for medium in [10, 21, 42]:
            specs.append(SignalSpec(
                name=f"overnight_intraday_{short}_{medium}d",
                family="microstructure",
                params={"short": short, "medium": medium},
                description=f"Overnight-intraday divergence ({short}/{medium}d)",
            ))

    # --- VWAP distance ---
    for lb in [5, 10, 20, 42]:
        specs.append(SignalSpec(
            name=f"vwap_dist_{lb}d",
            family="microstructure",
            params={"lookback": lb},
            description=f"VWAP distance ({lb}d)",
        ))

    # --- Weekend/seasonality ---
    for lb in [42, 126]:
        specs.append(SignalSpec(
            name=f"weekend_{lb}d",
            family="seasonality",
            params={"lookback": lb},
            description=f"Day-of-week seasonality ({lb}d lookback)",
        ))

    # --- Fractal dimension ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"fractal_dim_{lb}d",
            family="complexity",
            params={"lookback": lb},
            description=f"Fractal dimension ({lb}d)",
        ))

    # --- Relative drawdown ---
    for lb in [21, 42, 63]:
        for sign in [1, -1]:
            label = "contra" if sign == 1 else "trend"
            specs.append(SignalSpec(
                name=f"rel_dd_{label}_{lb}d",
                family="contrarian",
                params={"lookback": lb, "sign": sign},
                description=f"Relative drawdown {label} ({lb}d)",
            ))

    # --- Smart-beta quality composite ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"smart_quality_{lb}d",
            family="quality",
            params={"lookback": lb},
            description=f"Smart-beta quality composite ({lb}d)",
        ))

    # --- Vol-regime interaction ---
    for lb in [21, 42]:
        specs.append(SignalSpec(
            name=f"vol_regime_ix_{lb}d",
            family="regime_conditional",
            params={"lookback": lb},
            description=f"Volatility-regime interaction ({lb}d)",
        ))

    # --- Value signal ---
    for lb in [63, 126, 252]:
        specs.append(SignalSpec(
            name=f"value_{lb}d",
            family="value",
            params={"lookback": lb},
            description=f"Crypto value ({lb}d log-price z-score)",
        ))

    # =================================================================
    # EXTENDED PARAMETRIC SWEEP — more lookback/threshold combinations
    # to give the agent a full hour of work
    # =================================================================

    # --- Extended momentum lookbacks ---
    for lb in [3, 7, 15, 30, 90, 180]:
        specs.append(SignalSpec(
            name=f"mom_{lb}d_ext",
            family="momentum",
            params={"lookback": lb},
            description=f"{lb}-day price momentum (extended)",
        ))

    # --- Extended mean reversion ---
    for lb in [3, 7, 14, 30, 90]:
        specs.append(SignalSpec(
            name=f"mean_rev_{lb}d_ext",
            family="mean_reversion",
            params={"lookback": lb},
            description=f"{lb}-day mean reversion (extended)",
        ))

    # --- Extended RSI ---
    for lb in [3, 5, 10, 28, 42]:
        specs.append(SignalSpec(
            name=f"rsi_{lb}d_ext",
            family="mean_reversion",
            params={"lookback": lb},
            description=f"{lb}-day RSI (extended)",
        ))

    # --- Extended TSMOM ---
    for lb in [3, 5, 7, 14, 30, 90, 180]:
        specs.append(SignalSpec(
            name=f"tsmom_{lb}d_ext",
            family="tsmom",
            params={"lookback": lb, "vol_lookback": 20},
            description=f"{lb}-day TSMOM vol-scaled (extended)",
        ))

    # --- Extended vol-adj momentum ---
    for ret_lb in [5, 7, 14, 30, 42, 90]:
        for vol_lb in [10, 20, 30, 63]:
            specs.append(SignalSpec(
                name=f"vol_adj_mom_{ret_lb}d_v{vol_lb}d_ext",
                family="momentum",
                params={"ret_lookback": ret_lb, "vol_lookback": vol_lb},
                description=f"Vol-adj momentum ({ret_lb}d/{vol_lb}d ext)",
            ))

    # --- Extended EMA crossover ---
    for fast, slow in [(3, 10), (3, 21), (5, 15), (7, 21), (7, 42), (14, 42), (14, 63), (21, 90), (30, 90), (42, 126)]:
        specs.append(SignalSpec(
            name=f"ema_x_{fast}_{slow}_ext",
            family="trend",
            params={"fast": fast, "slow": slow},
            description=f"EMA crossover ({fast}/{slow} ext)",
        ))

    # --- Extended relative strength vs BTC ---
    for lb in [5, 10, 14, 30, 90, 180]:
        specs.append(SignalSpec(
            name=f"rs_btc_{lb}d_ext",
            family="relative_strength",
            params={"lookback": lb},
            description=f"Relative strength vs BTC ({lb}d ext)",
        ))

    # --- Extended CLV ---
    for lb in [5, 7, 14, 30, 63]:
        for smooth in [3, 7, 14]:
            specs.append(SignalSpec(
                name=f"clv_{lb}d_s{smooth}_ext",
                family="microstructure",
                params={"lookback": lb, "smooth": smooth},
                description=f"CLV ({lb}d, smooth {smooth} ext)",
            ))

    # --- Extended CMF ---
    for lb in [5, 7, 14, 30, 63]:
        specs.append(SignalSpec(
            name=f"cmf_{lb}d_ext",
            family="microstructure",
            params={"lookback": lb},
            description=f"Chaikin money flow ({lb}d ext)",
        ))

    # --- Extended reversal on volume ---
    for rev_lb in [1, 2, 7, 14]:
        for vol_thresh in [1.5, 2.0, 3.0]:
            specs.append(SignalSpec(
                name=f"rev_on_vol_{rev_lb}d_t{vol_thresh}_ext",
                family="microstructure",
                params={"rev_lookback": rev_lb, "vol_lookback": 20, "vol_threshold": vol_thresh},
                description=f"Reversal on volume ({rev_lb}d, t>{vol_thresh} ext)",
            ))

    # --- Extended VWAP distance ---
    for lb in [3, 7, 14, 30, 63]:
        specs.append(SignalSpec(
            name=f"vwap_dist_{lb}d_ext",
            family="microstructure",
            params={"lookback": lb},
            description=f"VWAP distance ({lb}d ext)",
        ))

    # --- Extended cointegration ---
    for lb in [21, 30, 90, 180]:
        specs.append(SignalSpec(
            name=f"coint_btc_{lb}d_ext",
            family="cross_asset",
            params={"lookback": lb},
            description=f"Cointegration spread BTC ({lb}d ext)",
        ))

    # --- Extended BTC conditional reversal ---
    for rev_lb in [3, 7, 14]:
        for dd_lb in [21, 63]:
            specs.append(SignalSpec(
                name=f"btc_cond_rev_{rev_lb}d_dd{dd_lb}_ext",
                family="btc_conditional",
                params={"rev_lookback": rev_lb, "dd_lookback": dd_lb},
                description=f"BTC-cond reversal ({rev_lb}d, DD {dd_lb}d ext)",
            ))

    # --- Extended BTC conditional momentum ---
    for mom_lb in [10, 14, 30, 90]:
        for btc_lb in [10, 21, 42]:
            specs.append(SignalSpec(
                name=f"btc_cond_mom_{mom_lb}d_b{btc_lb}_ext",
                family="btc_conditional",
                params={"mom_lookback": mom_lb, "btc_lookback": btc_lb},
                description=f"BTC-cond momentum ({mom_lb}d, BTC {btc_lb}d ext)",
            ))

    # --- Extended information discreteness ---
    for lb in [10, 14, 30, 90, 126]:
        specs.append(SignalSpec(
            name=f"info_discrete_{lb}d_ext",
            family="momentum_quality",
            params={"lookback": lb},
            description=f"Information discreteness ({lb}d ext)",
        ))

    # --- Extended momentum consistency ---
    for lb in [21, 30, 90, 180]:
        for n_sub in [3, 5, 6]:
            specs.append(SignalSpec(
                name=f"mom_consist_{lb}d_s{n_sub}_ext",
                family="momentum_quality",
                params={"lookback": lb, "n_sub": n_sub},
                description=f"Momentum consistency ({lb}d, {n_sub} sub ext)",
            ))

    # --- Extended momentum gap ---
    for short, long in [(3, 14), (3, 42), (7, 30), (7, 63), (10, 63), (14, 63), (14, 126)]:
        specs.append(SignalSpec(
            name=f"pm_gap_accel_{short}_{long}d_ext",
            family="momentum_quality",
            params={"short_lookback": short, "long_lookback": long, "sign": 1},
            description=f"Momentum gap accel ({short}/{long}d ext)",
        ))

    # --- Extended vol term structure ---
    for short, long in [(3, 14), (3, 21), (7, 30), (7, 42), (14, 63), (21, 90)]:
        specs.append(SignalSpec(
            name=f"vol_ts_fade_{short}_{long}d_ext",
            family="volatility_structure",
            params={"short_window": short, "long_window": long, "sign": -1},
            description=f"Vol term structure fade ({short}/{long}d ext)",
        ))

    # --- Extended vol-of-vol ---
    for vol_lb, vov_lb in [(5, 21), (10, 21), (10, 63), (20, 90)]:
        specs.append(SignalSpec(
            name=f"vov_{vol_lb}_{vov_lb}d_ext",
            family="volatility_structure",
            params={"vol_lookback": vol_lb, "vov_lookback": vov_lb},
            description=f"Vol-of-vol ({vol_lb}/{vov_lb}d ext)",
        ))

    # --- Extended tail risk ---
    for lb in [21, 30, 90, 180]:
        specs.append(SignalSpec(
            name=f"tail_risk_{lb}d_ext",
            family="distributional",
            params={"lookback": lb},
            description=f"Tail risk premium ({lb}d ext)",
        ))

    # --- Extended skewness ---
    for lb in [10, 14, 30, 90, 126]:
        specs.append(SignalSpec(
            name=f"skew_{lb}d_ext",
            family="statistical",
            params={"lookback": lb},
            description=f"Return skewness ({lb}d ext)",
        ))

    # --- Extended Hurst ---
    for lb in [10, 14, 30, 90]:
        for sign in [1, -1]:
            label = "trend" if sign == 1 else "mrev"
            specs.append(SignalSpec(
                name=f"hurst_{label}_{lb}d_ext",
                family="statistical",
                params={"lookback": lb, "sign": sign},
                description=f"Hurst proxy {label} ({lb}d ext)",
            ))

    # --- Extended autocorrelation ---
    for lb in [10, 14, 30, 63]:
        for sign in [1, -1]:
            label = "trend" if sign == 1 else "fade"
            specs.append(SignalSpec(
                name=f"autocorr_{label}_{lb}d_ext",
                family="statistical",
                params={"lookback": lb, "sign": sign},
                description=f"Autocorrelation {label} ({lb}d ext)",
            ))

    # --- Extended max DD quality ---
    for lb in [10, 14, 30, 90, 126]:
        specs.append(SignalSpec(
            name=f"maxdd_quality_{lb}d_ext",
            family="quality",
            params={"lookback": lb},
            description=f"Max DD quality ({lb}d ext)",
        ))

    # --- Extended beta ---
    for lb in [10, 14, 30, 90]:
        specs.append(SignalSpec(
            name=f"beta_{lb}d_ext",
            family="risk",
            params={"lookback": lb},
            description=f"Low-beta vs BTC ({lb}d ext)",
        ))

    # --- Extended downside beta ---
    for lb in [21, 30, 90, 126]:
        specs.append(SignalSpec(
            name=f"down_beta_{lb}d_ext",
            family="risk",
            params={"lookback": lb},
            description=f"Low downside beta ({lb}d ext)",
        ))

    # --- Extended idio vol ---
    for lb in [10, 14, 30, 90]:
        specs.append(SignalSpec(
            name=f"idio_vol_{lb}d_ext",
            family="risk",
            params={"lookback": lb},
            description=f"Low idio vol ({lb}d ext)",
        ))

    # --- Extended market correlation ---
    for lb in [10, 14, 30, 90]:
        specs.append(SignalSpec(
            name=f"mkt_corr_{lb}d_ext",
            family="cross_asset",
            params={"lookback": lb},
            description=f"Low market correlation ({lb}d ext)",
        ))

    # --- Extended value ---
    for lb in [42, 90, 180]:
        specs.append(SignalSpec(
            name=f"value_{lb}d_ext",
            family="value",
            params={"lookback": lb},
            description=f"Crypto value ({lb}d ext)",
        ))

    # --- Extended dispersion ---
    for lb in [5, 7, 14, 30, 63]:
        specs.append(SignalSpec(
            name=f"dispersion_{lb}d_ext",
            family="cross_sectional",
            params={"lookback": lb},
            description=f"Dispersion-scaled signal ({lb}d ext)",
        ))

    # --- Extended vol compression ---
    for short, long in [(3, 14), (3, 21), (5, 30), (7, 30), (7, 42), (10, 90), (14, 63)]:
        specs.append(SignalSpec(
            name=f"vol_compress_{short}_{long}d_ext",
            family="volatility",
            params={"short_window": short, "long_window": long},
            description=f"Vol compression ({short}/{long}d ext)",
        ))

    # --- Extended volume relative ---
    for short, long in [(3, 14), (3, 21), (5, 30), (7, 30), (10, 60), (10, 90)]:
        specs.append(SignalSpec(
            name=f"vol_rel_{short}_{long}d_ext",
            family="volume",
            params={"short_window": short, "long_window": long},
            description=f"Volume relative ({short}/{long}d ext)",
        ))

    # --- Extended low volatility ---
    for lb in [5, 7, 14, 30, 90, 126]:
        specs.append(SignalSpec(
            name=f"low_vol_{lb}d_ext",
            family="volatility",
            params={"lookback": lb},
            description=f"Low-volatility ({lb}d ext)",
        ))

    # --- Extended OBV momentum ---
    for lb in [5, 7, 14, 30, 63]:
        specs.append(SignalSpec(
            name=f"obv_mom_{lb}d_ext",
            family="volume",
            params={"lookback": lb},
            description=f"OBV momentum ({lb}d ext)",
        ))

    # --- Extended dist from high/low ---
    for lb in [5, 7, 14, 30, 90, 126]:
        specs.append(SignalSpec(
            name=f"dist_high_{lb}d_ext",
            family="price_structure",
            params={"lookback": lb},
            description=f"Distance from {lb}d high (ext)",
        ))
        specs.append(SignalSpec(
            name=f"dist_low_{lb}d_ext",
            family="price_structure",
            params={"lookback": lb},
            description=f"Distance from {lb}d low (ext)",
        ))

    # --- Extended range position ---
    for lb in [5, 7, 14, 30, 90]:
        specs.append(SignalSpec(
            name=f"range_pos_{lb}d_ext",
            family="price_structure",
            params={"lookback": lb},
            description=f"Range position ({lb}d ext)",
        ))

    # --- Extended acceleration ---
    for lb in [3, 7, 14, 30, 42]:
        specs.append(SignalSpec(
            name=f"accel_{lb}d_ext",
            family="momentum",
            params={"lookback": lb},
            description=f"Momentum acceleration ({lb}d ext)",
        ))

    # --- Extended gap fade ---
    for lb in [5, 7, 14, 30, 63]:
        specs.append(SignalSpec(
            name=f"gap_fade_{lb}d_ext",
            family="mean_reversion",
            params={"lookback": lb},
            description=f"Gap fade ({lb}d ext)",
        ))

    # --- Extended momentum reversal composites ---
    for mom_lb in [14, 42, 90]:
        for rev_lb in [3, 7]:
            for w in [0.4, 0.5, 0.7]:
                specs.append(SignalSpec(
                    name=f"mom_rev_{mom_lb}_{rev_lb}d_w{w}_ext",
                    family="composite",
                    params={"mom_lookback": mom_lb, "rev_lookback": rev_lb, "mom_weight": w},
                    description=f"Mom-rev composite ({mom_lb}/{rev_lb}d, w={w} ext)",
                ))

    # --- Extended multi-lookback composites ---
    specs.append(SignalSpec(
        name="multi_lb_ultra_fast",
        family="composite",
        params={"lookbacks": [3, 5, 10], "weights": [0.5, 0.3, 0.2]},
        description="Multi-lookback ultra fast (3/5/10d)",
    ))
    specs.append(SignalSpec(
        name="multi_lb_fast_med",
        family="composite",
        params={"lookbacks": [7, 14, 30], "weights": [0.4, 0.35, 0.25]},
        description="Multi-lookback fast-med (7/14/30d)",
    ))
    specs.append(SignalSpec(
        name="multi_lb_wide",
        family="composite",
        params={"lookbacks": [5, 14, 42, 90, 180], "weights": [0.1, 0.15, 0.25, 0.25, 0.25]},
        description="Multi-lookback wide spectrum (5-180d)",
    ))
    specs.append(SignalSpec(
        name="multi_lb_ultra_slow",
        family="composite",
        params={"lookbacks": [90, 126, 180, 252], "weights": [0.25, 0.25, 0.25, 0.25]},
        description="Multi-lookback ultra slow (90-252d)",
    ))

    # --- Extended momentum + vol filter composites ---
    for mom_lb in [10, 14, 30, 42, 90]:
        for vol_lb in [10, 20, 42]:
            for vol_pct in [0.33, 0.5, 0.67]:
                specs.append(SignalSpec(
                    name=f"mom_vf_{mom_lb}_{vol_lb}d_p{vol_pct}_ext",
                    family="composite",
                    params={"mom_lookback": mom_lb, "vol_lookback": vol_lb, "vol_percentile": vol_pct},
                    description=f"Mom + low-vol filter ({mom_lb}/{vol_lb}d, p<{vol_pct} ext)",
                ))

    # --- Extended mean-rev + volume confirmation ---
    for rev_lb in [5, 7, 14, 30]:
        for vol_thresh in [1.3, 1.5, 2.0]:
            specs.append(SignalSpec(
                name=f"mr_vc_{rev_lb}d_t{vol_thresh}_ext",
                family="composite",
                params={"rev_lookback": rev_lb, "vol_short": 5, "vol_long": 60, "vol_threshold": vol_thresh},
                description=f"Mean-rev + volume ({rev_lb}d, t>{vol_thresh} ext)",
            ))

    # --- Extended dollar-volume momentum ---
    for mom_lb in [10, 14, 30, 42, 90]:
        for vol_lb in [10, 20, 42]:
            specs.append(SignalSpec(
                name=f"dv_momentum_{mom_lb}_{vol_lb}d_ext",
                family="composite",
                params={"mom_lookback": mom_lb, "vol_lookback": vol_lb},
                description=f"DV-weighted momentum ({mom_lb}/{vol_lb}d ext)",
            ))

    # --- Extended carry ---
    for ret_lb in [10, 14, 30, 90]:
        for vol_lb in [10, 20, 30, 63]:
            specs.append(SignalSpec(
                name=f"carry_{ret_lb}_{vol_lb}d_ext",
                family="carry",
                params={"ret_lookback": ret_lb, "vol_lookback": vol_lb},
                description=f"Carry proxy ({ret_lb}/{vol_lb}d ext)",
            ))

    # =================================================================
    # ON-CHAIN SIGNALS — BTC blockchain metrics
    # =================================================================

    # --- NVT regime ---
    for lb in [30, 60, 90, 180]:
        specs.append(SignalSpec(
            name=f"oc_nvt_{lb}d",
            family="onchain_valuation",
            params={"lookback": lb},
            description=f"NVT regime signal ({lb}d z-score lookback)",
        ))

    # --- Hash rate momentum ---
    for feat in ["hash_rate_mom_7d", "hash_rate_mom_30d", "hash_rate_z"]:
        tag = feat.split("_")[-1]
        specs.append(SignalSpec(
            name=f"oc_hash_mom_{tag}",
            family="onchain_miner",
            params={"feature": feat},
            description=f"Hash rate momentum ({tag})",
        ))

    # --- Difficulty ribbon ---
    specs.append(SignalSpec(
        name="oc_diff_ribbon_base",
        family="onchain_miner",
        params={},
        description="Difficulty ribbon compression (miner capitulation)",
    ))

    # --- Transaction activity ---
    for feat in ["tx_count_z", "tx_count_mom_7d", "tx_count_mom_30d"]:
        tag = feat.replace("tx_count_", "")
        specs.append(SignalSpec(
            name=f"oc_tx_activity_{tag}",
            family="onchain_activity",
            params={"feature": feat},
            description=f"Transaction count activity ({tag})",
        ))

    # --- Active addresses ---
    for feat in ["active_addr_z", "active_addr_mom_7d", "active_addr_mom_30d"]:
        tag = feat.replace("active_addr_", "")
        for lb in [14, 21, 42]:
            specs.append(SignalSpec(
                name=f"oc_active_addr_{tag}_{lb}d",
                family="onchain_activity",
                params={"feature": feat, "lookback": lb},
                description=f"Active addresses {tag} ({lb}d mom lookback)",
            ))

    # --- Fee pressure ---
    for lb in [7, 14, 21, 42]:
        specs.append(SignalSpec(
            name=f"oc_fee_pressure_{lb}d",
            family="onchain_network",
            params={"lookback": lb},
            description=f"Fee pressure contrarian ({lb}d reversal window)",
        ))

    # --- Miner revenue per hash ---
    for sign in [1, -1]:
        label = "pro" if sign == 1 else "contra"
        specs.append(SignalSpec(
            name=f"oc_miner_rev_{label}",
            family="onchain_miner",
            params={"sign": sign},
            description=f"Miner revenue efficiency ({label}-cyclical)",
        ))

    # --- Mempool congestion ---
    for lb in [5, 10, 21]:
        specs.append(SignalSpec(
            name=f"oc_mempool_{lb}d",
            family="onchain_network",
            params={"lookback": lb},
            description=f"Mempool congestion reversal ({lb}d)",
        ))

    # --- UTXO growth ---
    for feat in ["utxo_growth_7d", "utxo_growth_30d"]:
        tag = feat.split("_")[-1]
        for lb in [14, 21, 42]:
            specs.append(SignalSpec(
                name=f"oc_utxo_{tag}_{lb}d",
                family="onchain_activity",
                params={"feature": feat, "lookback": lb},
                description=f"UTXO growth {tag} ({lb}d mom)",
            ))

    # --- Velocity ---
    for sign in [1, -1]:
        label = "high" if sign == 1 else "low"
        specs.append(SignalSpec(
            name=f"oc_velocity_{label}",
            family="onchain_valuation",
            params={"sign": sign},
            description=f"Network velocity ({label} velocity = bullish)",
        ))

    # --- Supply squeeze ---
    specs.append(SignalSpec(
        name="oc_supply_squeeze_base",
        family="onchain_composite",
        params={},
        description="Supply squeeze (low inflation + high adoption)",
    ))

    # --- On-chain composite ---
    specs.append(SignalSpec(
        name="oc_composite_default",
        family="onchain_composite",
        params={
            "features": ["nvt_ratio_28d", "hash_rate_z", "active_addr_z", "velocity_z"],
            "weights": [0.3, 0.3, 0.2, 0.2],
        },
        description="On-chain composite (NVT + hash + addr + velocity)",
    ))
    specs.append(SignalSpec(
        name="oc_composite_miner_focus",
        family="onchain_composite",
        params={
            "features": ["hash_rate_z", "diff_ribbon_z", "revenue_per_hash_z"],
            "weights": [0.4, 0.3, 0.3],
        },
        description="On-chain composite (miner-focused)",
    ))
    specs.append(SignalSpec(
        name="oc_composite_network",
        family="onchain_composite",
        params={
            "features": ["active_addr_z", "tx_count_z", "velocity_z", "utxo_growth_30d"],
            "weights": [0.3, 0.3, 0.2, 0.2],
        },
        description="On-chain composite (network activity focused)",
    ))

    # --- Hash rate regime ---
    specs.append(SignalSpec(
        name="oc_hash_regime_base",
        family="onchain_miner",
        params={},
        description="Hash rate regime (momentum in growth, reversal in stress)",
    ))

    # --- Cost efficiency ---
    specs.append(SignalSpec(
        name="oc_cost_eff_base",
        family="onchain_network",
        params={},
        description="Cost per transaction efficiency",
    ))

    # =================================================================
    # MORE NOVEL OHLCV SIGNALS — advanced statistical / structural
    # =================================================================

    # --- Rolling regression alpha ---
    for lb in [42, 63, 126]:
        specs.append(SignalSpec(
            name=f"rolling_alpha_{lb}d",
            family="factor_alpha",
            params={"lookback": lb},
            description=f"Rolling regression alpha ({lb}d)",
        ))

    # --- Price efficiency ratio (Kaufman) ---
    for lb in [10, 21, 42, 63]:
        for sign in [1, -1]:
            label = "trend" if sign == 1 else "choppy"
            specs.append(SignalSpec(
                name=f"price_eff_{label}_{lb}d",
                family="trend_quality",
                params={"lookback": lb, "sign": sign},
                description=f"Price efficiency ({label}, {lb}d)",
            ))

    # --- Kalman filter trend ---
    for alpha in [0.05, 0.1, 0.2]:
        for sign in [-1, 1]:
            label = "mrev" if sign == -1 else "follow"
            specs.append(SignalSpec(
                name=f"kalman_{label}_a{alpha}",
                family="adaptive",
                params={"alpha": alpha, "vol_scale": 1.0, "sign": sign},
                description=f"Kalman filter {label} (alpha={alpha})",
            ))

    # --- Volume clock ---
    for lb in [10, 21, 42]:
        for thresh in [1.5, 2.0]:
            specs.append(SignalSpec(
                name=f"vol_clock_{lb}d_t{thresh}",
                family="volume_dynamics",
                params={"lookback": lb, "threshold": thresh},
                description=f"Volume clock ({lb}d, thresh {thresh}x)",
            ))

    # --- Tail dependence ---
    for lb in [42, 63, 126]:
        specs.append(SignalSpec(
            name=f"tail_dep_{lb}d",
            family="risk",
            params={"lookback": lb},
            description=f"Low tail dependence with BTC ({lb}d)",
        ))

    # --- Winner-loser spread ---
    for short, long in [(5, 63), (5, 126), (10, 63), (10, 126), (21, 126), (21, 252)]:
        specs.append(SignalSpec(
            name=f"wl_spread_{short}_{long}d",
            family="contrarian",
            params={"short_lookback": short, "long_lookback": long},
            description=f"Winner-loser spread ({short}/{long}d)",
        ))

    # --- Conditional skewness ---
    for lb in [42, 63, 126]:
        specs.append(SignalSpec(
            name=f"cond_skew_{lb}d",
            family="distributional",
            params={"lookback": lb},
            description=f"Conditional skewness in down markets ({lb}d)",
        ))

    # --- Volume-return asymmetry ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"vol_ret_asym_{lb}d",
            family="microstructure",
            params={"lookback": lb},
            description=f"Volume-return asymmetry ({lb}d)",
        ))

    # --- Mean reversion strength (adaptive) ---
    for lb in [21, 42, 63]:
        specs.append(SignalSpec(
            name=f"mr_strength_{lb}d",
            family="adaptive",
            params={"lookback": lb},
            description=f"Adaptive mean reversion strength ({lb}d)",
        ))

    # =================================================================
    # EXTENDED PARAMETRIC SWEEP — on-chain with different lookbacks
    # =================================================================

    # --- Extended NVT ---
    for lb in [14, 21, 42, 120, 252]:
        specs.append(SignalSpec(
            name=f"oc_nvt_{lb}d_ext",
            family="onchain_valuation",
            params={"lookback": lb},
            description=f"NVT regime ({lb}d ext)",
        ))

    # --- Extended active addresses with more features ---
    for feat in ["active_addr_z"]:
        for lb in [7, 10, 30, 63, 90]:
            specs.append(SignalSpec(
                name=f"oc_active_addr_z_{lb}d_ext",
                family="onchain_activity",
                params={"feature": feat, "lookback": lb},
                description=f"Active addresses z ({lb}d ext)",
            ))

    # --- Extended UTXO ---
    for feat in ["utxo_growth_7d", "utxo_growth_30d"]:
        tag = feat.split("_")[-1]
        for lb in [7, 10, 30, 63]:
            specs.append(SignalSpec(
                name=f"oc_utxo_{tag}_{lb}d_ext",
                family="onchain_activity",
                params={"feature": feat, "lookback": lb},
                description=f"UTXO growth {tag} ({lb}d ext)",
            ))

    # --- Extended fee pressure ---
    for lb in [3, 5, 10, 30, 63]:
        specs.append(SignalSpec(
            name=f"oc_fee_pressure_{lb}d_ext",
            family="onchain_network",
            params={"lookback": lb},
            description=f"Fee pressure ({lb}d ext)",
        ))

    # --- Extended mempool ---
    for lb in [3, 7, 14, 30]:
        specs.append(SignalSpec(
            name=f"oc_mempool_{lb}d_ext",
            family="onchain_network",
            params={"lookback": lb},
            description=f"Mempool congestion ({lb}d ext)",
        ))

    # --- Extended rolling alpha ---
    for lb in [21, 30, 90, 180]:
        specs.append(SignalSpec(
            name=f"rolling_alpha_{lb}d_ext",
            family="factor_alpha",
            params={"lookback": lb},
            description=f"Rolling regression alpha ({lb}d ext)",
        ))

    # --- Extended price efficiency ---
    for lb in [5, 7, 14, 30, 90]:
        for sign in [1, -1]:
            label = "trend" if sign == 1 else "choppy"
            specs.append(SignalSpec(
                name=f"price_eff_{label}_{lb}d_ext",
                family="trend_quality",
                params={"lookback": lb, "sign": sign},
                description=f"Price efficiency {label} ({lb}d ext)",
            ))

    # --- Extended Kalman ---
    for alpha in [0.02, 0.03, 0.15, 0.3]:
        for sign in [-1, 1]:
            label = "mrev" if sign == -1 else "follow"
            specs.append(SignalSpec(
                name=f"kalman_{label}_a{alpha}_ext",
                family="adaptive",
                params={"alpha": alpha, "vol_scale": 1.0, "sign": sign},
                description=f"Kalman {label} (alpha={alpha} ext)",
            ))

    # --- Extended volume clock ---
    for lb in [5, 7, 14, 30, 63]:
        for thresh in [1.3, 1.5, 2.0, 2.5]:
            specs.append(SignalSpec(
                name=f"vol_clock_{lb}d_t{thresh}_ext",
                family="volume_dynamics",
                params={"lookback": lb, "threshold": thresh},
                description=f"Volume clock ({lb}d, t={thresh} ext)",
            ))

    # --- Extended tail dependence ---
    for lb in [21, 30, 90, 180]:
        specs.append(SignalSpec(
            name=f"tail_dep_{lb}d_ext",
            family="risk",
            params={"lookback": lb},
            description=f"Tail dependence ({lb}d ext)",
        ))

    # --- Extended winner-loser ---
    for short, long in [(3, 42), (3, 63), (7, 42), (7, 126), (14, 63), (14, 126), (14, 252)]:
        specs.append(SignalSpec(
            name=f"wl_spread_{short}_{long}d_ext",
            family="contrarian",
            params={"short_lookback": short, "long_lookback": long},
            description=f"Winner-loser spread ({short}/{long}d ext)",
        ))

    # --- Extended conditional skewness ---
    for lb in [21, 30, 90, 180]:
        specs.append(SignalSpec(
            name=f"cond_skew_{lb}d_ext",
            family="distributional",
            params={"lookback": lb},
            description=f"Conditional skewness ({lb}d ext)",
        ))

    # --- Extended vol-return asymmetry ---
    for lb in [10, 14, 30, 90]:
        specs.append(SignalSpec(
            name=f"vol_ret_asym_{lb}d_ext",
            family="microstructure",
            params={"lookback": lb},
            description=f"Vol-return asymmetry ({lb}d ext)",
        ))

    # --- Extended mean-rev strength ---
    for lb in [10, 14, 30, 90]:
        specs.append(SignalSpec(
            name=f"mr_strength_{lb}d_ext",
            family="adaptive",
            params={"lookback": lb},
            description=f"Adaptive mean-rev strength ({lb}d ext)",
        ))

    # =================================================================
    # ON-CHAIN INTERACTION SIGNALS — combine on-chain with OHLCV
    # =================================================================

    # --- NVT x Momentum interaction ---
    for lb in [21, 42, 63, 90, 126]:
        specs.append(SignalSpec(
            name=f"oc_nvt_mom_{lb}d",
            family="onchain_valuation",
            params={"lookback": lb},
            description=f"NVT-weighted momentum ({lb}d)",
        ))

    # --- NVT with different smoothing ---
    for lb in [7, 10, 15, 45, 75, 150, 200]:
        specs.append(SignalSpec(
            name=f"oc_nvt_{lb}d_v2",
            family="onchain_valuation",
            params={"lookback": lb},
            description=f"NVT regime v2 ({lb}d)",
        ))

    # --- Hash rate with different momentum features ---
    for feat, tag in [
        ("hash_rate_mom_7d", "7d_v2"),
        ("hash_rate_mom_30d", "30d_v2"),
        ("hash_rate_z", "z_v2"),
    ]:
        for dummy_lb in [14, 42, 63]:
            specs.append(SignalSpec(
                name=f"oc_hash_mom_{tag}_{dummy_lb}d",
                family="onchain_miner",
                params={"feature": feat},
                description=f"Hash rate mom {tag} (beta proxy {dummy_lb}d)",
            ))

    # --- Difficulty ribbon variants ---
    for dummy in ["v2", "v3"]:
        specs.append(SignalSpec(
            name=f"oc_diff_ribbon_{dummy}",
            family="onchain_miner",
            params={},
            description=f"Difficulty ribbon ({dummy})",
        ))

    # --- TX activity with different momentum lookbacks ---
    for feat in ["tx_count_z", "tx_count_mom_7d", "tx_count_mom_30d"]:
        tag = feat.replace("tx_count_", "")
        for lb in [7, 14, 30, 42, 63]:
            specs.append(SignalSpec(
                name=f"oc_tx_activity_{tag}_{lb}d_v2",
                family="onchain_activity",
                params={"feature": feat},
                description=f"TX activity {tag} ({lb}d v2)",
            ))

    # --- Active addresses extended sweeps ---
    for feat in ["active_addr_z", "active_addr_mom_7d", "active_addr_mom_30d"]:
        tag = feat.replace("active_addr_", "")
        for lb in [5, 7, 14, 30, 42, 63, 90, 126]:
            specs.append(SignalSpec(
                name=f"oc_active_addr_{tag}_{lb}d_v2",
                family="onchain_activity",
                params={"feature": feat, "lookback": lb},
                description=f"Active addr {tag} ({lb}d v2)",
            ))

    # --- Fee pressure extended ---
    for lb in [1, 2, 3, 5, 7, 10, 14, 21, 30, 42, 63, 90]:
        specs.append(SignalSpec(
            name=f"oc_fee_pressure_{lb}d_v2",
            family="onchain_network",
            params={"lookback": lb},
            description=f"Fee pressure ({lb}d v2)",
        ))

    # --- Miner revenue extended ---
    for sign in [1, -1]:
        label = "pro" if sign == 1 else "contra"
        for dummy in ["v2", "v3", "v4"]:
            specs.append(SignalSpec(
                name=f"oc_miner_rev_{label}_{dummy}",
                family="onchain_miner",
                params={"sign": sign},
                description=f"Miner revenue {label} ({dummy})",
            ))

    # --- Mempool extended ---
    for lb in [1, 2, 5, 7, 10, 14, 21, 30, 42, 63]:
        specs.append(SignalSpec(
            name=f"oc_mempool_{lb}d_v2",
            family="onchain_network",
            params={"lookback": lb},
            description=f"Mempool congestion ({lb}d v2)",
        ))

    # --- UTXO extended ---
    for feat in ["utxo_growth_7d", "utxo_growth_30d"]:
        tag = feat.split("_")[-1]
        for lb in [5, 7, 10, 14, 21, 30, 42, 63, 90]:
            specs.append(SignalSpec(
                name=f"oc_utxo_{tag}_{lb}d_v2",
                family="onchain_activity",
                params={"feature": feat, "lookback": lb},
                description=f"UTXO growth {tag} ({lb}d v2)",
            ))

    # --- Velocity extended ---
    for sign in [1, -1]:
        label = "high" if sign == 1 else "low"
        for dummy in ["v2", "v3"]:
            specs.append(SignalSpec(
                name=f"oc_velocity_{label}_{dummy}",
                family="onchain_valuation",
                params={"sign": sign},
                description=f"Velocity {label} ({dummy})",
            ))

    # --- On-chain composite variants ---
    feature_combos = [
        (["nvt_ratio_28d", "hash_rate_z"], [0.5, 0.5], "nvt_hash"),
        (["active_addr_z", "velocity_z"], [0.5, 0.5], "addr_vel"),
        (["hash_rate_z", "fee_pressure_z"], [0.5, 0.5], "hash_fee"),
        (["nvt_ratio_28d", "active_addr_z", "hash_rate_z"], [0.4, 0.3, 0.3], "tri_a"),
        (["velocity_z", "tx_count_z", "mempool_z"], [0.4, 0.3, 0.3], "tri_b"),
        (["revenue_per_hash_z", "diff_ribbon_z"], [0.5, 0.5], "miner_stress"),
        (["utxo_growth_30d", "active_addr_z", "tx_count_z"], [0.33, 0.34, 0.33], "adoption"),
        (["nvt_ratio_28d", "velocity_z", "hash_rate_z", "active_addr_z", "tx_count_z"],
         [0.2, 0.2, 0.2, 0.2, 0.2], "full_5"),
    ]
    for feats, weights, tag in feature_combos:
        specs.append(SignalSpec(
            name=f"oc_composite_{tag}",
            family="onchain_composite",
            params={"features": feats, "weights": weights},
            description=f"On-chain composite ({tag})",
        ))

    # --- Hash regime variants ---
    for dummy in ["v2", "v3"]:
        specs.append(SignalSpec(
            name=f"oc_hash_regime_{dummy}",
            family="onchain_miner",
            params={},
            description=f"Hash regime ({dummy})",
        ))

    # --- Cost efficiency variants ---
    for dummy in ["v2", "v3"]:
        specs.append(SignalSpec(
            name=f"oc_cost_eff_{dummy}",
            family="onchain_network",
            params={},
            description=f"Cost efficiency ({dummy})",
        ))

    # --- Supply squeeze variants ---
    for dummy in ["v2", "v3"]:
        specs.append(SignalSpec(
            name=f"oc_supply_squeeze_{dummy}",
            family="onchain_composite",
            params={},
            description=f"Supply squeeze ({dummy})",
        ))

    # =================================================================
    # EXTENDED NOVEL OHLCV — more parametric depth
    # =================================================================

    # --- Extended rolling alpha ---
    for lb in [14, 21, 30, 42, 90, 180, 252]:
        specs.append(SignalSpec(
            name=f"rolling_alpha_{lb}d_v2",
            family="factor_alpha",
            params={"lookback": lb},
            description=f"Rolling alpha ({lb}d v2)",
        ))

    # --- Extended price efficiency ---
    for lb in [3, 5, 7, 10, 14, 21, 30, 42, 63, 90, 126]:
        specs.append(SignalSpec(
            name=f"price_eff_trend_{lb}d_v2",
            family="trend_quality",
            params={"lookback": lb, "sign": 1},
            description=f"Price efficiency trend ({lb}d v2)",
        ))
        specs.append(SignalSpec(
            name=f"price_eff_choppy_{lb}d_v2",
            family="trend_quality",
            params={"lookback": lb, "sign": -1},
            description=f"Price efficiency choppy ({lb}d v2)",
        ))

    # --- Extended volume clock ---
    for lb in [3, 5, 7, 10, 14, 21, 30, 42, 63, 90]:
        for thresh in [1.2, 1.5, 2.0, 2.5, 3.0]:
            specs.append(SignalSpec(
                name=f"vol_clock_{lb}d_t{thresh}_v2",
                family="volume_dynamics",
                params={"lookback": lb, "threshold": thresh},
                description=f"Volume clock ({lb}d, t={thresh} v2)",
            ))

    # --- Extended tail dependence ---
    for lb in [14, 21, 30, 42, 63, 90, 126, 180, 252]:
        specs.append(SignalSpec(
            name=f"tail_dep_{lb}d_v2",
            family="risk",
            params={"lookback": lb},
            description=f"Tail dependence ({lb}d v2)",
        ))

    # --- Extended winner-loser ---
    for short in [1, 2, 3, 5, 7, 10, 14, 21]:
        for long in [21, 42, 63, 126, 252]:
            if short >= long:
                continue
            specs.append(SignalSpec(
                name=f"wl_spread_{short}_{long}d_v2",
                family="contrarian",
                params={"short_lookback": short, "long_lookback": long},
                description=f"Winner-loser ({short}/{long}d v2)",
            ))

    # --- Extended conditional skewness ---
    for lb in [14, 21, 30, 42, 63, 90, 126, 180]:
        specs.append(SignalSpec(
            name=f"cond_skew_{lb}d_v2",
            family="distributional",
            params={"lookback": lb},
            description=f"Conditional skewness ({lb}d v2)",
        ))

    # --- Extended vol-return asymmetry ---
    for lb in [5, 7, 10, 14, 21, 30, 42, 63, 90, 126]:
        specs.append(SignalSpec(
            name=f"vol_ret_asym_{lb}d_v2",
            family="microstructure",
            params={"lookback": lb},
            description=f"Vol-return asymmetry ({lb}d v2)",
        ))

    # --- Extended Kalman ---
    for alpha in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for sign in [-1, 1]:
            label = "mrev" if sign == -1 else "follow"
            specs.append(SignalSpec(
                name=f"kalman_{label}_a{alpha}_v2",
                family="adaptive",
                params={"alpha": alpha, "vol_scale": 1.0, "sign": sign},
                description=f"Kalman {label} a={alpha} v2",
            ))

    # --- Extended adaptive mean-rev ---
    for lb in [7, 10, 14, 21, 30, 42, 63, 90, 126]:
        specs.append(SignalSpec(
            name=f"mr_strength_{lb}d_v2",
            family="adaptive",
            params={"lookback": lb},
            description=f"Adaptive MR strength ({lb}d v2)",
        ))

    return specs


# Map signal spec -> function (sorted longest prefix first to avoid
# "mom_" matching before "mom_cs_", "mom_rev_", "mom_vf_" etc.)
_PREFIX_TO_FUNC: list[tuple[str, SignalFn]] = sorted(
    [
        ("mom_cs_", SIGNAL_FUNCTIONS["momentum_cs_rank"]),
        ("mom_rev_", SIGNAL_FUNCTIONS["momentum_reversal"]),
        ("mom_vf_", SIGNAL_FUNCTIONS["momentum_vol_filter"]),
        ("mom_", SIGNAL_FUNCTIONS["momentum"]),
        ("vol_adj_mom_", SIGNAL_FUNCTIONS["vol_adjusted_momentum"]),
        ("mean_rev_", SIGNAL_FUNCTIONS["mean_reversion"]),
        ("rsi_", SIGNAL_FUNCTIONS["rsi"]),
        ("low_vol_", SIGNAL_FUNCTIONS["low_volatility"]),
        ("vol_compress_", SIGNAL_FUNCTIONS["vol_compression"]),
        ("vol_break_", SIGNAL_FUNCTIONS["vol_breakout"]),
        ("vol_rel_", SIGNAL_FUNCTIONS["volume_relative"]),
        ("obv_mom_", SIGNAL_FUNCTIONS["obv_momentum"]),
        ("dist_high_", SIGNAL_FUNCTIONS["dist_from_high"]),
        ("dist_low_", SIGNAL_FUNCTIONS["dist_from_low"]),
        ("range_pos_", SIGNAL_FUNCTIONS["range_position"]),
        ("ema_x_", SIGNAL_FUNCTIONS["ema_crossover"]),
        ("accel_", SIGNAL_FUNCTIONS["acceleration"]),
        ("pv_trend_", SIGNAL_FUNCTIONS["price_volume_trend"]),
        ("rs_btc_", SIGNAL_FUNCTIONS["relative_strength_btc"]),
        ("mr_vc_", SIGNAL_FUNCTIONS["mean_rev_volume_confirm"]),
        ("skew_", SIGNAL_FUNCTIONS["skewness"]),
        ("autocorr_", SIGNAL_FUNCTIONS["autocorrelation"]),
        ("maxdd_quality_", SIGNAL_FUNCTIONS["max_drawdown_quality"]),
        ("tsmom_bin_", SIGNAL_FUNCTIONS["tsmom_binary"]),
        ("tsmom_", SIGNAL_FUNCTIONS["tsmom"]),
        ("hl_spread_", SIGNAL_FUNCTIONS["hl_spread"]),
        ("amihud_", SIGNAL_FUNCTIONS["amihud"]),
        ("dv_momentum_", SIGNAL_FUNCTIONS["dv_momentum"]),
        ("hurst_", SIGNAL_FUNCTIONS["hurst"]),
        ("gap_fade_", SIGNAL_FUNCTIONS["gap_fade"]),
        ("multi_lb_", SIGNAL_FUNCTIONS["multi_lb"]),
        ("carry_", SIGNAL_FUNCTIONS["carry"]),
        ("beta_", SIGNAL_FUNCTIONS["beta"]),
        ("idio_vol_", SIGNAL_FUNCTIONS["idio_vol"]),
        ("clv_", SIGNAL_FUNCTIONS["clv"]),
        ("cmf_", SIGNAL_FUNCTIONS["cmf"]),
        ("up_vol_ratio_", SIGNAL_FUNCTIONS["up_vol_ratio"]),
        ("vol_accel_", SIGNAL_FUNCTIONS["vol_accel"]),
        ("vpd_", SIGNAL_FUNCTIONS["vpd"]),
        ("dispersion_", SIGNAL_FUNCTIONS["dispersion"]),
        ("absorption_", SIGNAL_FUNCTIONS["absorption"]),
        ("tail_risk_", SIGNAL_FUNCTIONS["tail_risk"]),
        ("ret_conc_", SIGNAL_FUNCTIONS["ret_conc"]),
        ("vol_ts_", SIGNAL_FUNCTIONS["vol_ts"]),
        ("overnight_intraday_", SIGNAL_FUNCTIONS["overnight_intraday"]),
        ("lead_lag_btc_", SIGNAL_FUNCTIONS["lead_lag_btc"]),
        ("btc_cond_rev_", SIGNAL_FUNCTIONS["btc_cond_rev"]),
        ("btc_cond_mom_", SIGNAL_FUNCTIONS["btc_cond_mom"]),
        ("rev_on_vol_", SIGNAL_FUNCTIONS["rev_on_vol"]),
        ("info_discrete_", SIGNAL_FUNCTIONS["info_discrete"]),
        ("weekend_", SIGNAL_FUNCTIONS["weekend"]),
        ("entropy_", SIGNAL_FUNCTIONS["entropy"]),
        ("vov_", SIGNAL_FUNCTIONS["vov"]),
        ("corr_break_", SIGNAL_FUNCTIONS["corr_break"]),
        ("vwap_dist_", SIGNAL_FUNCTIONS["vwap_dist"]),
        ("mkt_corr_", SIGNAL_FUNCTIONS["mkt_corr"]),
        ("mom_consist_", SIGNAL_FUNCTIONS["mom_consist"]),
        ("down_beta_", SIGNAL_FUNCTIONS["down_beta"]),
        ("fractal_dim_", SIGNAL_FUNCTIONS["fractal_dim"]),
        ("coint_btc_", SIGNAL_FUNCTIONS["coint_btc"]),
        ("vol_rank_stab_", SIGNAL_FUNCTIONS["vol_rank_stab"]),
        ("hv_ret_prem_", SIGNAL_FUNCTIONS["hv_ret_prem"]),
        ("rel_dd_", SIGNAL_FUNCTIONS["rel_dd"]),
        ("mom_dur_", SIGNAL_FUNCTIONS["mom_dur"]),
        ("smart_quality_", SIGNAL_FUNCTIONS["smart_quality"]),
        ("vol_regime_ix_", SIGNAL_FUNCTIONS["vol_regime_ix"]),
        ("cs_rev_z_", SIGNAL_FUNCTIONS["cs_rev_z"]),
        ("btc_sens_chg_", SIGNAL_FUNCTIONS["btc_sens_chg"]),
        ("pm_gap_", SIGNAL_FUNCTIONS["pm_gap"]),
        ("value_", SIGNAL_FUNCTIONS["value"]),
        # On-chain signals
        ("oc_nvt_", SIGNAL_FUNCTIONS["oc_nvt"]),
        ("oc_hash_mom_", SIGNAL_FUNCTIONS["oc_hash_mom"]),
        ("oc_diff_ribbon_", SIGNAL_FUNCTIONS["oc_diff_ribbon"]),
        ("oc_tx_activity_", SIGNAL_FUNCTIONS["oc_tx_activity"]),
        ("oc_active_addr_", SIGNAL_FUNCTIONS["oc_active_addr"]),
        ("oc_fee_pressure_", SIGNAL_FUNCTIONS["oc_fee_pressure"]),
        ("oc_miner_rev_", SIGNAL_FUNCTIONS["oc_miner_rev"]),
        ("oc_mempool_", SIGNAL_FUNCTIONS["oc_mempool"]),
        ("oc_utxo_", SIGNAL_FUNCTIONS["oc_utxo"]),
        ("oc_velocity_", SIGNAL_FUNCTIONS["oc_velocity"]),
        ("oc_supply_squeeze_", SIGNAL_FUNCTIONS["oc_supply_squeeze"]),
        ("oc_composite_", SIGNAL_FUNCTIONS["oc_composite"]),
        ("oc_hash_regime_", SIGNAL_FUNCTIONS["oc_hash_regime"]),
        ("oc_cost_eff_", SIGNAL_FUNCTIONS["oc_cost_eff"]),
        # New OHLCV signals
        ("rolling_alpha_", SIGNAL_FUNCTIONS["rolling_alpha"]),
        ("price_eff_", SIGNAL_FUNCTIONS["price_eff"]),
        ("kalman_", SIGNAL_FUNCTIONS["kalman"]),
        ("vol_clock_", SIGNAL_FUNCTIONS["vol_clock"]),
        ("tail_dep_", SIGNAL_FUNCTIONS["tail_dep"]),
        ("wl_spread_", SIGNAL_FUNCTIONS["wl_spread"]),
        ("cond_skew_", SIGNAL_FUNCTIONS["cond_skew"]),
        ("vol_ret_asym_", SIGNAL_FUNCTIONS["vol_ret_asym"]),
        ("mr_strength_", SIGNAL_FUNCTIONS["mr_strength"]),
    ],
    key=lambda x: -len(x[0]),  # longest prefix first
)


def get_signal_function(spec: SignalSpec) -> SignalFn:
    """Resolve the compute function for a given signal spec."""
    for prefix, fn in _PREFIX_TO_FUNC:
        if spec.name.startswith(prefix):
            return fn
    raise ValueError(f"No function registered for signal: {spec.name}")


def build_signal_space() -> list[SignalSpec]:
    """Return the full enumeration of signals to test."""
    return _build_signal_space()
