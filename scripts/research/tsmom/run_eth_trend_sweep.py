#!/usr/bin/env python3
"""
ETH-USD Exhaustive Trend Model Sweep.

Runs every conceivable trend-following model on ETH-USD with full parameter
sweeps across daily, 4h, and 1h frequencies.  Pure signal discovery — no
overfitting constraints.

Usage:
    python -m scripts.research.tsmom.run_eth_trend_sweep
"""
from __future__ import annotations

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import talib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import load_daily_bars, load_bars, ANN_FACTOR

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts" / "research" / "tsmom" / "eth_trend_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "ETH-USD"
COST_BPS = 20


# =====================================================================
# Data loading
# =====================================================================

def load_eth(freq: str = "1d") -> pd.DataFrame:
    """Load ETH-USD OHLCV at requested frequency, return single-asset DF."""
    if freq == "1d":
        panel = load_daily_bars()
    else:
        panel = load_bars(freq)
    eth = panel[panel["symbol"] == SYMBOL].copy()
    eth = eth.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    eth = eth[["open", "high", "low", "close", "volume"]].astype(float)
    return eth


def resample_to_daily(intraday_signal: pd.Series, daily_close: pd.Series) -> pd.Series:
    """Take an intraday signal and pick end-of-day value for daily evaluation."""
    sig_daily = intraday_signal.resample("1D").last().dropna()
    sig_daily = sig_daily.reindex(daily_close.index, method="ffill")
    return sig_daily


# =====================================================================
# Single-asset backtest
# =====================================================================

def backtest_signal(signal: pd.Series, returns: pd.Series, cost_bps: int = COST_BPS):
    """
    Binary long/cash backtest on a single asset.
    signal: 1 = long, 0 = cash.  Applied with 1-bar lag.
    returns: daily close-to-close returns.
    """
    sig = signal.reindex(returns.index).fillna(0)
    pos = sig.shift(1).fillna(0)

    trades = pos.diff().abs()
    cost = trades * (cost_bps / 10_000)

    gross_ret = pos * returns
    net_ret = gross_ret - cost

    equity = (1 + net_ret).cumprod()
    return equity, net_ret, pos


def compute_perf(equity: pd.Series, net_ret: pd.Series, pos: pd.Series):
    """Compute standard performance metrics."""
    ret = net_ret.dropna()
    if len(ret) < 60:
        return None

    total_days = len(ret)
    std = ret.std()
    if std < 1e-12:
        return None

    sharpe = float(ret.mean() / std * np.sqrt(ANN_FACTOR))
    skewness = float(ret.skew())
    cagr = float(equity.iloc[-1] ** (ANN_FACTOR / total_days) - 1)
    maxdd = float((equity / equity.cummax() - 1).min())

    in_market = (pos.abs() > 1e-6).mean()
    trades = (pos.diff().abs() > 1e-6).sum()
    n_entries = int(trades / 2)

    invested_ret = ret[pos.shift(1).abs() > 1e-6]
    invested_skew = float(invested_ret.skew()) if len(invested_ret) > 30 else np.nan

    avg_trade_bars = total_days * in_market / max(n_entries, 1)

    return {
        "sharpe": round(sharpe, 4),
        "skewness": round(skewness, 4),
        "invested_skew": round(invested_skew, 4) if not np.isnan(invested_skew) else np.nan,
        "cagr": round(cagr, 4),
        "max_dd": round(maxdd, 4),
        "time_in_market": round(float(in_market), 4),
        "n_trades": n_entries,
        "avg_trade_days": round(float(avg_trade_bars), 1),
        "total_return": round(float(equity.iloc[-1] - 1), 4),
        "sortino": round(float(ret.mean() / ret[ret < 0].std() * np.sqrt(ANN_FACTOR)) if ret[ret < 0].std() > 1e-12 else np.nan, 4),
    }


# =====================================================================
# Signal generators — each returns pd.Series of 0/1
# =====================================================================

def _sma(close, n):
    return close.rolling(n, min_periods=n).mean()

def _ema(close, n):
    return close.ewm(span=n, adjust=False).mean()

def _dema(close, n):
    e1 = _ema(close, n)
    e2 = _ema(e1, n)
    return 2 * e1 - e2

def _hull_ma(close, n):
    half = max(int(n / 2), 1)
    sqrt_n = max(int(np.sqrt(n)), 1)
    wma_half = close.rolling(half, min_periods=half).mean()
    wma_full = close.rolling(n, min_periods=n).mean()
    diff = 2 * wma_half - wma_full
    return diff.rolling(sqrt_n, min_periods=sqrt_n).mean()


# ── Moving Average Crossovers ────────────────────────────────────────

def sig_sma_cross(close, fast, slow):
    return (_sma(close, fast) > _sma(close, slow)).astype(float)

def sig_ema_cross(close, fast, slow):
    return (_ema(close, fast) > _ema(close, slow)).astype(float)

def sig_dema_cross(close, fast, slow):
    return (_dema(close, fast) > _dema(close, slow)).astype(float)

def sig_hull_cross(close, fast, slow):
    return (_hull_ma(close, fast) > _hull_ma(close, slow)).astype(float)

def sig_price_above_sma(close, n):
    return (close > _sma(close, n)).astype(float)

def sig_price_above_ema(close, n):
    return (close > _ema(close, n)).astype(float)


# ── Channel Breakouts ────────────────────────────────────────────────

def sig_donchian(close, high, n):
    upper = high.rolling(n, min_periods=n).max()
    return (close > upper.shift(1)).astype(float)

def sig_bollinger(close, n, num_std):
    sma = _sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    upper = sma + num_std * std
    return (close > upper).astype(float)

def sig_keltner(close, high, low, n, atr_mult):
    mid = _ema(close, n)
    atr = talib.ATR(high.values, low.values, close.values, timeperiod=n)
    atr = pd.Series(atr, index=close.index)
    upper = mid + atr_mult * atr
    return (close > upper).astype(float)

def sig_supertrend(close, high, low, period, mult):
    atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
    atr = pd.Series(atr, index=close.index)
    hl2 = (high + low) / 2
    upper_band = hl2 + mult * atr
    lower_band = hl2 - mult * atr

    supertrend = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1.0, index=close.index)

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1.0
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1.0 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1.0 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

    return (direction > 0).astype(float)


# ── Momentum / ROC ───────────────────────────────────────────────────

def sig_momentum(close, n):
    ret = close.pct_change(n)
    return (ret > 0).astype(float)

def sig_momentum_threshold(close, n, threshold):
    ret = close.pct_change(n)
    return (ret > threshold).astype(float)

def sig_vol_scaled_mom(close, n, vol_lookback):
    ret = close.pct_change(n)
    vol = close.pct_change().rolling(vol_lookback, min_periods=vol_lookback).std()
    scaled = ret / (vol * np.sqrt(n) + 1e-12)
    return (scaled > 0).astype(float)

def sig_lreg(close, n):
    log_close = np.log(close)
    x = np.arange(n, dtype=float)
    x_dm = x - x.mean()
    ss_x = (x_dm ** 2).sum()

    def _tstat(window):
        if len(window) < n or np.isnan(window).any():
            return np.nan
        y = window - window.mean()
        beta = np.dot(x_dm, y) / ss_x
        resid = y - beta * x_dm
        se = np.sqrt(np.sum(resid ** 2) / (n - 2) / ss_x) if n > 2 else 1e-12
        return beta / (se + 1e-12)

    tstat = log_close.rolling(n, min_periods=n).apply(_tstat, raw=True)
    return (tstat > 0).astype(float)


# ── TA-Lib Indicators ────────────────────────────────────────────────

def sig_macd(close, fast, slow, signal_period):
    macd, signal, hist = talib.MACD(close.values, fastperiod=fast, slowperiod=slow, signalperiod=signal_period)
    return pd.Series((macd > signal).astype(float), index=close.index)

def sig_rsi(close, period, threshold):
    rsi = talib.RSI(close.values, timeperiod=period)
    return pd.Series((rsi > threshold).astype(float), index=close.index)

def sig_adx_trend(close, high, low, period, threshold):
    adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
    plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=period)
    minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=period)
    signal = ((adx > threshold) & (plus_di > minus_di)).astype(float)
    return pd.Series(signal, index=close.index)

def sig_cci(close, high, low, period):
    cci = talib.CCI(high.values, low.values, close.values, timeperiod=period)
    return pd.Series((cci > 0).astype(float), index=close.index)

def sig_aroon(close, high, low, period):
    aroon_down, aroon_up = talib.AROON(high.values, low.values, timeperiod=period)
    return pd.Series((aroon_up > aroon_down).astype(float), index=close.index)

def sig_stoch(close, high, low, fastk, slowk, slowd):
    slowk_val, slowd_val = talib.STOCH(
        high.values, low.values, close.values,
        fastk_period=fastk, slowk_period=slowk, slowd_period=slowd,
    )
    return pd.Series((slowk_val > slowd_val).astype(float), index=close.index)

def sig_sar(close, high, low):
    sar = talib.SAR(high.values, low.values)
    return pd.Series((close.values > sar).astype(float), index=close.index)

def sig_williams_r(close, high, low, period, threshold):
    wr = talib.WILLR(high.values, low.values, close.values, timeperiod=period)
    return pd.Series((wr > threshold).astype(float), index=close.index)

def sig_mfi(close, high, low, volume, period, threshold):
    mfi = talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=period)
    return pd.Series((mfi > threshold).astype(float), index=close.index)

def sig_trix(close, period):
    trix = talib.TRIX(close.values, timeperiod=period)
    return pd.Series((trix > 0).astype(float), index=close.index)

def sig_ultosc(close, high, low, p1, p2, p3, threshold):
    ult = talib.ULTOSC(high.values, low.values, close.values,
                       timeperiod1=p1, timeperiod2=p2, timeperiod3=p3)
    return pd.Series((ult > threshold).astype(float), index=close.index)

def sig_ppo(close, fast, slow):
    ppo = talib.PPO(close.values, fastperiod=fast, slowperiod=slow)
    return pd.Series((ppo > 0).astype(float), index=close.index)

def sig_apo(close, fast, slow):
    apo = talib.APO(close.values, fastperiod=fast, slowperiod=slow)
    return pd.Series((apo > 0).astype(float), index=close.index)

def sig_adosc(close, high, low, volume, fast, slow):
    adosc = talib.ADOSC(high.values, low.values, close.values, volume.values,
                        fastperiod=fast, slowperiod=slow)
    return pd.Series((adosc > 0).astype(float), index=close.index)

def sig_dx_trend(close, high, low, period, threshold):
    dx = talib.DX(high.values, low.values, close.values, timeperiod=period)
    plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=period)
    minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=period)
    return pd.Series(((dx > threshold) & (plus_di > minus_di)).astype(float), index=close.index)

def sig_mom(close, period):
    mom = talib.MOM(close.values, timeperiod=period)
    return pd.Series((mom > 0).astype(float), index=close.index)

def sig_roc(close, period):
    roc = talib.ROC(close.values, timeperiod=period)
    return pd.Series((roc > 0).astype(float), index=close.index)

def sig_cmo(close, period):
    cmo = talib.CMO(close.values, timeperiod=period)
    return pd.Series((cmo > 0).astype(float), index=close.index)


# ── Composite / Exotic ───────────────────────────────────────────────

def sig_ichimoku(close, high, low, tenkan=9, kijun=26, senkou_b=52):
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    cloud_top = pd.concat([senkou_a, senkou_span_b], axis=1).max(axis=1)
    return (close > cloud_top).astype(float)

def sig_obv_trend(close, volume, sma_period):
    obv = talib.OBV(close.values, volume.values)
    obv = pd.Series(obv, index=close.index)
    obv_ma = obv.rolling(sma_period, min_periods=sma_period).mean()
    return (obv > obv_ma).astype(float)

def sig_heikin_ashi(open_, high, low, close):
    ha_close = (open_ + high + low + close) / 4
    ha_open = pd.Series(np.nan, index=close.index)
    ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(close)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    return (ha_close > ha_open).astype(float)

def sig_kaufman_er(close, er_period, threshold):
    direction = (close - close.shift(er_period)).abs()
    volatility = close.diff().abs().rolling(er_period, min_periods=er_period).sum()
    er = direction / (volatility + 1e-12)
    trend_up = close > close.shift(er_period)
    return ((er > threshold) & trend_up).astype(float)

def sig_vwap_trend(close, volume, period):
    cum_vp = (close * volume).rolling(period, min_periods=period).sum()
    cum_v = volume.rolling(period, min_periods=period).sum()
    vwap = cum_vp / (cum_v + 1e-12)
    return (close > vwap).astype(float)

def sig_dual_momentum(close, fast, slow):
    """Long only if BOTH fast and slow momentum are positive."""
    ret_fast = close.pct_change(fast)
    ret_slow = close.pct_change(slow)
    return ((ret_fast > 0) & (ret_slow > 0)).astype(float)

def sig_triple_ma(close, fast, mid, slow):
    sma_fast = _sma(close, fast)
    sma_mid = _sma(close, mid)
    sma_slow = _sma(close, slow)
    return ((sma_fast > sma_mid) & (sma_mid > sma_slow)).astype(float)

def sig_ma_ribbon(close, periods):
    """All MAs aligned in ascending order."""
    mas = [_sma(close, p) for p in sorted(periods)]
    aligned = pd.Series(True, index=close.index)
    for i in range(1, len(mas)):
        aligned = aligned & (mas[i-1] > mas[i])
    return aligned.astype(float)

def sig_turtle_breakout(close, high, low, entry_n, exit_n):
    """Turtle-style: enter on N-day high, exit on M-day low."""
    upper = high.rolling(entry_n, min_periods=entry_n).max().shift(1)
    lower = low.rolling(exit_n, min_periods=exit_n).min().shift(1)
    pos = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i] if not np.isnan(upper.iloc[i]) else False:
            pos.iloc[i] = 1.0
        elif close.iloc[i] < lower.iloc[i] if not np.isnan(lower.iloc[i]) else False:
            pos.iloc[i] = 0.0
        else:
            pos.iloc[i] = pos.iloc[i-1]
    return pos

def sig_mean_reversion_band(close, n, num_std):
    """Counter-trend: long when price drops below lower Bollinger band."""
    sma = _sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    lower = sma - num_std * std
    return (close < lower).astype(float)

def sig_regime_filter_sma(close, trend_n, fast, slow):
    """SMA cross only when above longer-term trend."""
    trend = close > _sma(close, trend_n)
    cross = _sma(close, fast) > _sma(close, slow)
    return (trend & cross).astype(float)

def sig_atr_breakout(close, high, low, period, mult):
    """Long when close > yesterday's close + mult*ATR."""
    atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
    atr = pd.Series(atr, index=close.index)
    threshold = close.shift(1) + mult * atr.shift(1)
    return (close > threshold).astype(float)

def sig_close_above_high(close, high, n):
    """Close above the N-bar high (excluding current bar)."""
    prev_high = high.shift(1).rolling(n, min_periods=n).max()
    return (close > prev_high).astype(float)


# =====================================================================
# Configuration grid
# =====================================================================

def build_configs(freq: str, has_ohlcv: bool = True):
    """Build list of (label, signal_func, kwargs) tuples."""
    configs = []

    # ── MA Crossovers ─────────────────────────────────────────────────
    fast_vals = [3, 5, 7, 10, 15, 20, 30]
    slow_vals = [15, 20, 30, 50, 100, 150, 200]

    for fast, slow in product(fast_vals, slow_vals):
        if fast >= slow:
            continue
        configs.append((f"SMA_cross_{fast}_{slow}", "sma_cross", {"fast": fast, "slow": slow}))
        configs.append((f"EMA_cross_{fast}_{slow}", "ema_cross", {"fast": fast, "slow": slow}))

    for fast, slow in product([5, 10, 15, 20, 30], [30, 50, 100, 200]):
        if fast >= slow:
            continue
        configs.append((f"DEMA_cross_{fast}_{slow}", "dema_cross", {"fast": fast, "slow": slow}))

    for fast, slow in product([5, 10, 15, 20], [20, 30, 50, 100]):
        if fast >= slow:
            continue
        configs.append((f"Hull_cross_{fast}_{slow}", "hull_cross", {"fast": fast, "slow": slow}))

    # ── Price vs MA ───────────────────────────────────────────────────
    for n in [5, 10, 15, 20, 30, 50, 100, 150, 200]:
        configs.append((f"Price_above_SMA_{n}", "price_sma", {"n": n}))
        configs.append((f"Price_above_EMA_{n}", "price_ema", {"n": n}))

    # ── Channel Breakouts ─────────────────────────────────────────────
    if has_ohlcv:
        for n in [5, 10, 15, 20, 30, 50, 100]:
            configs.append((f"Donchian_{n}", "donchian", {"n": n}))

        for n, std in product([10, 15, 20, 30, 50], [1.0, 1.5, 2.0, 2.5, 3.0]):
            configs.append((f"Boll_{n}_{std}", "bollinger", {"n": n, "num_std": std}))

        for n, mult in product([7, 10, 14, 20, 30], [1.0, 1.5, 2.0, 2.5, 3.0]):
            configs.append((f"Keltner_{n}_{mult}", "keltner", {"n": n, "atr_mult": mult}))

        for period, mult in product([7, 10, 14, 20, 30], [1.0, 1.5, 2.0, 2.5, 3.0]):
            configs.append((f"Supertrend_{period}_{mult}", "supertrend", {"period": period, "mult": mult}))

    # ── Momentum / ROC ────────────────────────────────────────────────
    for n in [3, 5, 7, 10, 15, 20, 30, 50, 63, 100, 126, 200]:
        configs.append((f"Momentum_{n}", "momentum", {"n": n}))

    for n, thresh in product([5, 10, 20, 30, 50, 100], [0.01, 0.02, 0.05, 0.10]):
        configs.append((f"MomThresh_{n}_{thresh}", "mom_threshold", {"n": n, "threshold": thresh}))

    for n, vl in product([5, 10, 20, 30, 50, 63], [21, 42, 63]):
        configs.append((f"VolScaledMom_{n}_{vl}", "vol_scaled_mom", {"n": n, "vol_lookback": vl}))

    for n in [5, 7, 10, 15, 20, 30, 50]:
        configs.append((f"LREG_{n}", "lreg", {"n": n}))

    # ── TA-Lib indicators ─────────────────────────────────────────────
    for fast, slow, sig in [(8, 21, 5), (12, 26, 9), (5, 35, 5), (10, 50, 10),
                            (5, 13, 1), (3, 10, 16), (8, 17, 9), (12, 26, 5)]:
        configs.append((f"MACD_{fast}_{slow}_{sig}", "macd", {"fast": fast, "slow": slow, "signal_period": sig}))

    for period, thresh in product([5, 7, 10, 14, 21, 30], [30, 40, 50, 55, 60]):
        configs.append((f"RSI_{period}_{thresh}", "rsi", {"period": period, "threshold": thresh}))

    if has_ohlcv:
        for period, thresh in product([7, 10, 14, 21], [15, 20, 25, 30]):
            configs.append((f"ADX_{period}_{thresh}", "adx", {"period": period, "threshold": thresh}))

        for period in [10, 14, 20, 30, 50]:
            configs.append((f"CCI_{period}", "cci", {"period": period}))

        for period in [10, 14, 20, 25, 50]:
            configs.append((f"Aroon_{period}", "aroon", {"period": period}))

        for fk, sk, sd in [(5, 3, 3), (14, 3, 3), (14, 5, 5), (21, 7, 7), (9, 3, 3)]:
            configs.append((f"Stoch_{fk}_{sk}_{sd}", "stoch", {"fastk": fk, "slowk": sk, "slowd": sd}))

        configs.append(("SAR", "sar", {}))

        for period, thresh in product([7, 14, 21], [-50, -40, -30, -20]):
            configs.append((f"WilliamsR_{period}_{thresh}", "williams_r", {"period": period, "threshold": thresh}))

        for period, thresh in product([7, 14, 21], [40, 50, 60]):
            configs.append((f"MFI_{period}_{thresh}", "mfi", {"period": period, "threshold": thresh}))

        for period, mult in product([7, 10, 14, 20], [1.0, 1.5, 2.0, 3.0]):
            configs.append((f"ATR_breakout_{period}_{mult}", "atr_breakout", {"period": period, "mult": mult}))

    for period in [5, 10, 15, 20, 30]:
        configs.append((f"TRIX_{period}", "trix", {"period": period}))

    for fast, slow in product([5, 10, 12], [20, 26, 50]):
        if fast >= slow:
            continue
        configs.append((f"PPO_{fast}_{slow}", "ppo", {"fast": fast, "slow": slow}))
        configs.append((f"APO_{fast}_{slow}", "apo", {"fast": fast, "slow": slow}))

    for period in [5, 10, 14, 20, 30, 50]:
        configs.append((f"MOM_{period}", "talib_mom", {"period": period}))
        configs.append((f"ROC_{period}", "talib_roc", {"period": period}))
        configs.append((f"CMO_{period}", "talib_cmo", {"period": period}))

    if has_ohlcv:
        for p1, p2, p3, thresh in [(7, 14, 28, 50), (5, 10, 20, 50), (7, 14, 28, 40)]:
            configs.append((f"UltOsc_{p1}_{p2}_{p3}_{thresh}", "ultosc",
                          {"p1": p1, "p2": p2, "p3": p3, "threshold": thresh}))

        for fast, slow in [(3, 10), (5, 20), (10, 30)]:
            configs.append((f"ADOSC_{fast}_{slow}", "adosc", {"fast": fast, "slow": slow}))

    # ── Composite / Exotic ────────────────────────────────────────────
    if has_ohlcv:
        for t, k, s in [(9, 26, 52), (7, 22, 44), (12, 30, 60)]:
            configs.append((f"Ichimoku_{t}_{k}_{s}", "ichimoku", {"tenkan": t, "kijun": k, "senkou_b": s}))

        configs.append(("HeikinAshi", "heikin_ashi", {}))

        for n in [5, 10, 15, 20, 30, 50]:
            configs.append((f"Close_above_high_{n}", "close_above_high", {"n": n}))

    for sma_p in [10, 20, 30, 50, 100]:
        configs.append((f"OBV_trend_{sma_p}", "obv_trend", {"sma_period": sma_p}))

    for er_p, thresh in product([10, 20, 30, 50], [0.3, 0.5, 0.6, 0.7]):
        configs.append((f"Kaufman_{er_p}_{thresh}", "kaufman", {"er_period": er_p, "threshold": thresh}))

    for p in [10, 20, 30, 50]:
        configs.append((f"VWAP_{p}", "vwap", {"period": p}))

    # ── Multi-signal composites ───────────────────────────────────────
    for fast, slow in [(5, 30), (10, 50), (10, 100), (20, 100), (30, 200)]:
        configs.append((f"DualMom_{fast}_{slow}", "dual_mom", {"fast": fast, "slow": slow}))

    for fast, mid, slow in [(5, 20, 50), (10, 30, 100), (10, 50, 200), (20, 50, 200)]:
        configs.append((f"TripleMA_{fast}_{mid}_{slow}", "triple_ma", {"fast": fast, "mid": mid, "slow": slow}))

    for entry, exit_ in [(20, 10), (30, 15), (50, 20), (50, 25), (100, 50), (55, 20)]:
        configs.append((f"Turtle_{entry}_{exit_}", "turtle", {"entry_n": entry, "exit_n": exit_}))

    for trend, fast, slow in [(200, 10, 50), (100, 5, 30), (200, 20, 50)]:
        configs.append((f"RegimeSMA_{trend}_{fast}_{slow}", "regime_sma",
                       {"trend_n": trend, "fast": fast, "slow": slow}))

    for n, std in product([10, 20, 30], [1.5, 2.0, 2.5]):
        configs.append((f"MeanRevBand_{n}_{std}", "mean_rev_band", {"n": n, "num_std": std}))

    return configs


# =====================================================================
# Signal dispatcher
# =====================================================================

def dispatch_signal(sig_type, close, high, low, open_, volume, **kwargs):
    """Route to the correct signal generator."""
    c, h, l, o, v = close, high, low, open_, volume

    if sig_type == "sma_cross":
        return sig_sma_cross(c, kwargs["fast"], kwargs["slow"])
    elif sig_type == "ema_cross":
        return sig_ema_cross(c, kwargs["fast"], kwargs["slow"])
    elif sig_type == "dema_cross":
        return sig_dema_cross(c, kwargs["fast"], kwargs["slow"])
    elif sig_type == "hull_cross":
        return sig_hull_cross(c, kwargs["fast"], kwargs["slow"])
    elif sig_type == "price_sma":
        return sig_price_above_sma(c, kwargs["n"])
    elif sig_type == "price_ema":
        return sig_price_above_ema(c, kwargs["n"])
    elif sig_type == "donchian":
        return sig_donchian(c, h, kwargs["n"])
    elif sig_type == "bollinger":
        return sig_bollinger(c, kwargs["n"], kwargs["num_std"])
    elif sig_type == "keltner":
        return sig_keltner(c, h, l, kwargs["n"], kwargs["atr_mult"])
    elif sig_type == "supertrend":
        return sig_supertrend(c, h, l, kwargs["period"], kwargs["mult"])
    elif sig_type == "momentum":
        return sig_momentum(c, kwargs["n"])
    elif sig_type == "mom_threshold":
        return sig_momentum_threshold(c, kwargs["n"], kwargs["threshold"])
    elif sig_type == "vol_scaled_mom":
        return sig_vol_scaled_mom(c, kwargs["n"], kwargs["vol_lookback"])
    elif sig_type == "lreg":
        return sig_lreg(c, kwargs["n"])
    elif sig_type == "macd":
        return sig_macd(c, kwargs["fast"], kwargs["slow"], kwargs["signal_period"])
    elif sig_type == "rsi":
        return sig_rsi(c, kwargs["period"], kwargs["threshold"])
    elif sig_type == "adx":
        return sig_adx_trend(c, h, l, kwargs["period"], kwargs["threshold"])
    elif sig_type == "cci":
        return sig_cci(c, h, l, kwargs["period"])
    elif sig_type == "aroon":
        return sig_aroon(c, h, l, kwargs["period"])
    elif sig_type == "stoch":
        return sig_stoch(c, h, l, kwargs["fastk"], kwargs["slowk"], kwargs["slowd"])
    elif sig_type == "sar":
        return sig_sar(c, h, l)
    elif sig_type == "williams_r":
        return sig_williams_r(c, h, l, kwargs["period"], kwargs["threshold"])
    elif sig_type == "mfi":
        return sig_mfi(c, h, l, v, kwargs["period"], kwargs["threshold"])
    elif sig_type == "trix":
        return sig_trix(c, kwargs["period"])
    elif sig_type == "ultosc":
        return sig_ultosc(c, h, l, kwargs["p1"], kwargs["p2"], kwargs["p3"], kwargs["threshold"])
    elif sig_type == "ppo":
        return sig_ppo(c, kwargs["fast"], kwargs["slow"])
    elif sig_type == "apo":
        return sig_apo(c, kwargs["fast"], kwargs["slow"])
    elif sig_type == "adosc":
        return sig_adosc(c, h, l, v, kwargs["fast"], kwargs["slow"])
    elif sig_type == "talib_mom":
        return sig_mom(c, kwargs["period"])
    elif sig_type == "talib_roc":
        return sig_roc(c, kwargs["period"])
    elif sig_type == "talib_cmo":
        return sig_cmo(c, kwargs["period"])
    elif sig_type == "dx":
        return sig_dx_trend(c, h, l, kwargs["period"], kwargs["threshold"])
    elif sig_type == "ichimoku":
        return sig_ichimoku(c, h, l, kwargs["tenkan"], kwargs["kijun"], kwargs["senkou_b"])
    elif sig_type == "obv_trend":
        return sig_obv_trend(c, v, kwargs["sma_period"])
    elif sig_type == "heikin_ashi":
        return sig_heikin_ashi(o, h, l, c)
    elif sig_type == "kaufman":
        return sig_kaufman_er(c, kwargs["er_period"], kwargs["threshold"])
    elif sig_type == "vwap":
        return sig_vwap_trend(c, v, kwargs["period"])
    elif sig_type == "dual_mom":
        return sig_dual_momentum(c, kwargs["fast"], kwargs["slow"])
    elif sig_type == "triple_ma":
        return sig_triple_ma(c, kwargs["fast"], kwargs["mid"], kwargs["slow"])
    elif sig_type == "turtle":
        return sig_turtle_breakout(c, h, l, kwargs["entry_n"], kwargs["exit_n"])
    elif sig_type == "mean_rev_band":
        return sig_mean_reversion_band(c, kwargs["n"], kwargs["num_std"])
    elif sig_type == "regime_sma":
        return sig_regime_filter_sma(c, kwargs["trend_n"], kwargs["fast"], kwargs["slow"])
    elif sig_type == "atr_breakout":
        return sig_atr_breakout(c, h, l, kwargs["period"], kwargs["mult"])
    elif sig_type == "close_above_high":
        return sig_close_above_high(c, h, kwargs["n"])
    elif sig_type == "ma_ribbon":
        return sig_ma_ribbon(c, kwargs["periods"])
    else:
        raise ValueError(f"Unknown signal type: {sig_type}")


# =====================================================================
# Main sweep
# =====================================================================

def run_sweep_on_freq(freq: str, daily_eth: pd.DataFrame | None = None):
    """Run all configs for a given frequency. Returns list of result dicts."""
    print(f"\n{'='*70}")
    print(f"  FREQUENCY: {freq}")
    print(f"{'='*70}")

    eth = load_eth(freq)
    print(f"  Loaded {len(eth)} bars for {SYMBOL} at {freq}")
    print(f"  Date range: {eth.index.min()} to {eth.index.max()}")

    close = eth["close"]
    high = eth["high"]
    low = eth["low"]
    open_ = eth["open"]
    volume = eth["volume"]
    returns = close.pct_change(fill_method=None)

    if freq != "1d" and daily_eth is not None:
        daily_close = daily_eth["close"]
        daily_returns = daily_close.pct_change(fill_method=None)
    else:
        daily_close = close
        daily_returns = returns

    has_ohlcv = True
    configs = build_configs(freq, has_ohlcv)
    print(f"  Configurations to test: {len(configs)}")

    results = []
    t0 = time.time()

    for i, (label, sig_type, kwargs) in enumerate(configs):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(configs) - i - 1) / rate
            print(f"  [{i+1}/{len(configs)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining ...")

        try:
            signal = dispatch_signal(sig_type, close, high, low, open_, volume, **kwargs)

            if freq != "1d" and daily_eth is not None:
                signal = resample_to_daily(signal, daily_close)
                eval_returns = daily_returns
            else:
                eval_returns = returns

            signal = signal.dropna()
            if len(signal) < 100:
                continue

            equity, net_ret, pos = backtest_signal(signal, eval_returns)
            perf = compute_perf(equity, net_ret, pos)

            if perf is not None:
                perf["label"] = label
                perf["signal_family"] = label.split("_")[0]
                perf["freq"] = freq
                perf["params"] = str(kwargs)
                results.append(perf)

        except Exception as e:
            pass

    elapsed = time.time() - t0
    print(f"  Completed {len(results)}/{len(configs)} valid configs in {elapsed:.1f}s")
    return results


def main():
    print("=" * 70)
    print("  ETH-USD EXHAUSTIVE TREND MODEL SWEEP")
    print("=" * 70)

    daily_eth = load_eth("1d")
    print(f"  Daily ETH-USD: {len(daily_eth)} bars, {daily_eth.index.min()} to {daily_eth.index.max()}")

    # Buy and hold baseline
    bh_ret = daily_eth["close"].pct_change(fill_method=None).dropna()
    bh_equity = (1 + bh_ret).cumprod()
    bh_perf = compute_perf(bh_equity, bh_ret, pd.Series(1.0, index=bh_ret.index))
    print(f"\n  BUY & HOLD BASELINE:")
    if bh_perf:
        print(f"  Sharpe: {bh_perf['sharpe']:.3f}  CAGR: {bh_perf['cagr']:.1%}  "
              f"MaxDD: {bh_perf['max_dd']:.1%}  Skew: {bh_perf['skewness']:.3f}")

    all_results = []

    # Daily
    daily_results = run_sweep_on_freq("1d", daily_eth)
    all_results.extend(daily_results)

    # 4h
    try:
        results_4h = run_sweep_on_freq("4h", daily_eth)
        all_results.extend(results_4h)
    except Exception as e:
        print(f"  [WARN] 4h sweep failed: {e}")

    # 1h
    try:
        results_1h = run_sweep_on_freq("1h", daily_eth)
        all_results.extend(results_1h)
    except Exception as e:
        print(f"  [WARN] 1h sweep failed: {e}")

    if not all_results:
        print("\n  NO VALID RESULTS")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values("sharpe", ascending=False)
    df.to_csv(OUT_DIR / "results.csv", index=False, float_format="%.4f")
    print(f"\n  Total valid results: {len(df)}")
    print(f"  Saved to {OUT_DIR / 'results.csv'}")

    # ── Rankings ──────────────────────────────────────────────────────
    def print_top(metric, n=25, ascending=False):
        print(f"\n  {'─'*70}")
        print(f"  TOP {n} BY {metric.upper()}")
        print(f"  {'─'*70}")
        top = df.nlargest(n, metric) if not ascending else df.nsmallest(n, metric)
        print(f"  {'Rank':>4s} {'Label':<40s} {'Freq':>4s} {'Sharpe':>8s} {'Skew':>8s} "
              f"{'CAGR':>8s} {'MaxDD':>8s} {'TIM':>6s} {'Trades':>7s}")
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            print(f"  {rank:>4d} {row['label']:<40s} {row['freq']:>4s} "
                  f"{row['sharpe']:>8.3f} {row['skewness']:>8.3f} "
                  f"{row['cagr']:>7.1%} {row['max_dd']:>7.1%} "
                  f"{row['time_in_market']:>5.0%} {row['n_trades']:>7d}")

    print_top("sharpe")
    print_top("skewness")
    print_top("cagr")

    # Best by Sharpe within each signal family
    print(f"\n  {'─'*70}")
    print(f"  BEST PER SIGNAL FAMILY (by Sharpe)")
    print(f"  {'─'*70}")
    best_per_family = df.loc[df.groupby("signal_family")["sharpe"].idxmax()]
    best_per_family = best_per_family.sort_values("sharpe", ascending=False)
    print(f"  {'Family':<20s} {'Best Config':<35s} {'Freq':>4s} {'Sharpe':>8s} {'Skew':>8s} "
          f"{'CAGR':>8s} {'MaxDD':>8s} {'TIM':>6s}")
    for _, row in best_per_family.head(30).iterrows():
        print(f"  {row['signal_family']:<20s} {row['label']:<35s} {row['freq']:>4s} "
              f"{row['sharpe']:>8.3f} {row['skewness']:>8.3f} "
              f"{row['cagr']:>7.1%} {row['max_dd']:>7.1%} "
              f"{row['time_in_market']:>5.0%}")

    # ── Chart ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    families = df["signal_family"].unique()
    cmap = plt.cm.tab20(np.linspace(0, 1, len(families)))
    fam_colors = {f: cmap[i] for i, f in enumerate(families)}

    for _, row in df.iterrows():
        c = fam_colors.get(row["signal_family"], "gray")
        axes[0].scatter(row["skewness"], row["sharpe"], c=[c], alpha=0.4, s=15, edgecolors="none")
    axes[0].set_xlabel("Skewness")
    axes[0].set_ylabel("Sharpe")
    axes[0].set_title("Sharpe vs Skewness (all configs)")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=0.5)
    if bh_perf:
        axes[0].scatter(bh_perf["skewness"], bh_perf["sharpe"], c="red", s=100, marker="*",
                       zorder=10, label="Buy & Hold")
        axes[0].legend()

    for _, row in df.iterrows():
        c = fam_colors.get(row["signal_family"], "gray")
        axes[1].scatter(row["time_in_market"], row["sharpe"], c=[c], alpha=0.4, s=15, edgecolors="none")
    axes[1].set_xlabel("Time in Market")
    axes[1].set_ylabel("Sharpe")
    axes[1].set_title("Sharpe vs Time in Market")

    sharpe_by_family = df.groupby("signal_family")["sharpe"].max().sort_values(ascending=True)
    y_pos = np.arange(len(sharpe_by_family))
    colors = [fam_colors.get(f, "gray") for f in sharpe_by_family.index]
    axes[2].barh(y_pos, sharpe_by_family.values, color=colors, alpha=0.8)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(sharpe_by_family.index, fontsize=7)
    axes[2].set_xlabel("Max Sharpe")
    axes[2].set_title("Best Sharpe per Signal Family")
    axes[2].axvline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep_summary.png", dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved to {OUT_DIR / 'sweep_summary.png'}")

    # ── Summary stats ─────────────────────────────────────────────────
    print(f"\n  {'='*70}")
    print(f"  SUMMARY")
    print(f"  {'='*70}")
    print(f"  Total configurations tested: {len(df)}")
    print(f"  Configs with Sharpe > 1.0: {(df['sharpe'] > 1.0).sum()}")
    print(f"  Configs with Sharpe > 1.5: {(df['sharpe'] > 1.5).sum()}")
    print(f"  Configs with Sharpe > 2.0: {(df['sharpe'] > 2.0).sum()}")
    print(f"  Configs with positive skewness: {(df['skewness'] > 0).sum()}")
    print(f"  Configs beating Buy & Hold Sharpe ({bh_perf['sharpe']:.2f}): "
          f"{(df['sharpe'] > bh_perf['sharpe']).sum()}" if bh_perf else "")
    print(f"\n  Median Sharpe: {df['sharpe'].median():.3f}")
    print(f"  Median Skewness: {df['skewness'].median():.3f}")
    print(f"  Median Time-in-Market: {df['time_in_market'].median():.1%}")

    print(f"\n  {'='*70}")
    print(f"  SWEEP COMPLETE")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
