#!/usr/bin/env python3
"""
Exhaustive Trend Sweep v2 — with trailing stop overlays.

For each base signal, tests multiple trailing stop levels.
Includes buy & hold as baseline. Drawdowns emphasized throughout.

Usage:
    python -m scripts.research.tsmom.run_eth_trend_sweep_v2 [--symbol ETH-USD]
"""
from __future__ import annotations

import argparse
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
COST_BPS = 20

SYMBOL = "ETH-USD"  # default; overridden by --symbol CLI arg

def _out_dir(symbol: str = None) -> Path:
    sym = (symbol or SYMBOL).replace("-", "").lower()
    d = ROOT / "artifacts" / "research" / "tsmom" / f"{sym}_trend_sweep"
    d.mkdir(parents=True, exist_ok=True)
    return d

OUT_DIR = _out_dir()

PCT_STOP_LEVELS = [None, 0.05, 0.10, 0.20]
PCT_STOP_LABELS = ["none", "pct5", "pct10", "pct20"]

ATR_MULTS = [1.5, 2.0, 2.5, 3.0, 4.0]
ATR_LABELS = ["atr1.5", "atr2.0", "atr2.5", "atr3.0", "atr4.0"]

ALL_STOP_LABELS = PCT_STOP_LABELS + ATR_LABELS


# =====================================================================
# Data loading
# =====================================================================

def load_asset(freq: str = "1d", symbol: str = None) -> pd.DataFrame:
    sym = symbol or SYMBOL
    if freq == "1d":
        panel = load_daily_bars()
    else:
        panel = load_bars(freq)
    asset = panel[panel["symbol"] == sym].copy()
    asset = asset.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    asset = asset[["open", "high", "low", "close", "volume"]].astype(float)
    return asset

# Backward-compat alias used by extension scripts
load_eth = load_asset


def resample_to_daily(intraday_signal: pd.Series, daily_close: pd.Series) -> pd.Series:
    sig_daily = intraday_signal.resample("1D").last().dropna()
    sig_daily = sig_daily.reindex(daily_close.index, method="ffill")
    return sig_daily


# =====================================================================
# Trailing stop overlay
# =====================================================================

def apply_trailing_stop(signal: pd.Series, close: pd.Series, stop_pct: float) -> pd.Series:
    """
    Overlay a trailing stop on a binary 0/1 signal.

    When long (signal=1), tracks highest close since entry.
    If close drops stop_pct below the high-water mark, forces exit.
    Stays out until the base signal goes to 0 and then back to 1 (fresh entry).
    """
    sig = signal.reindex(close.index).fillna(0).values
    px = close.values
    n = len(sig)
    pos = np.zeros(n)
    hwm = 0.0
    stopped_out = False

    for i in range(n):
        base_sig = sig[i] > 0.5

        if not base_sig:
            stopped_out = False
            pos[i] = 0.0
            hwm = 0.0
            continue

        if stopped_out:
            pos[i] = 0.0
            continue

        if pos[i - 1] < 0.5 if i > 0 else True:
            hwm = px[i]

        if px[i] > hwm:
            hwm = px[i]

        drawdown_from_peak = (hwm - px[i]) / hwm if hwm > 0 else 0

        if drawdown_from_peak >= stop_pct:
            pos[i] = 0.0
            stopped_out = True
            hwm = 0.0
        else:
            pos[i] = 1.0

    return pd.Series(pos, index=close.index)


def apply_atr_trailing_stop(
    signal: pd.Series, close: pd.Series,
    high: pd.Series, low: pd.Series,
    atr_mult: float, atr_period: int = 14,
) -> pd.Series:
    """
    Vol-adaptive trailing stop using ATR at entry.

    When entering long, records ATR at entry date.
    Tracks highest close since entry.
    Exits if close drops below HWM - atr_mult * ATR_entry.
    Stays out until base signal goes to 0 then back to 1.
    """
    atr_arr = talib.ATR(high.values, low.values, close.values, timeperiod=atr_period)
    sig = signal.reindex(close.index).fillna(0).values
    px = close.values
    n = len(sig)
    pos = np.zeros(n)
    hwm = 0.0
    entry_atr = 0.0
    stopped_out = False

    for i in range(n):
        base_sig = sig[i] > 0.5

        if not base_sig:
            stopped_out = False
            pos[i] = 0.0
            hwm = 0.0
            entry_atr = 0.0
            continue

        if stopped_out:
            pos[i] = 0.0
            continue

        if pos[i - 1] < 0.5 if i > 0 else True:
            hwm = px[i]
            entry_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0

        if px[i] > hwm:
            hwm = px[i]

        stop_distance = atr_mult * entry_atr
        if stop_distance > 0 and (hwm - px[i]) >= stop_distance:
            pos[i] = 0.0
            stopped_out = True
            hwm = 0.0
            entry_atr = 0.0
        else:
            pos[i] = 1.0

    return pd.Series(pos, index=close.index)


# =====================================================================
# Single-asset backtest
# =====================================================================

def backtest_signal(signal: pd.Series, returns: pd.Series, cost_bps: int = COST_BPS):
    sig = signal.reindex(returns.index).fillna(0)
    pos = sig.shift(1).fillna(0)
    trades = pos.diff().abs()
    cost = trades * (cost_bps / 10_000)
    gross_ret = pos * returns
    net_ret = gross_ret - cost
    equity = (1 + net_ret).cumprod()
    return equity, net_ret, pos


def compute_perf(equity: pd.Series, net_ret: pd.Series, pos: pd.Series):
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
    calmar = abs(cagr / maxdd) if abs(maxdd) > 1e-6 else np.nan

    in_market = (pos.abs() > 1e-6).mean()
    trades = (pos.diff().abs() > 1e-6).sum()
    n_entries = int(trades / 2)
    avg_trade_bars = total_days * in_market / max(n_entries, 1)

    return {
        "sharpe": round(sharpe, 4),
        "skewness": round(skewness, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(maxdd, 4),
        "calmar": round(calmar, 4) if not np.isnan(calmar) else np.nan,
        "total_return": round(float(equity.iloc[-1] - 1), 4),
        "sortino": round(float(ret.mean() / ret[ret < 0].std() * np.sqrt(ANN_FACTOR)) if ret[ret < 0].std() > 1e-12 else np.nan, 4),
        "time_in_market": round(float(in_market), 4),
        "n_trades": n_entries,
        "avg_trade_days": round(float(avg_trade_bars), 1),
    }


# =====================================================================
# Signal generators (same as v1 — kept inline for self-containment)
# =====================================================================

def _sma(close, n):
    return close.rolling(n, min_periods=n).mean()

def _ema(close, n):
    return close.ewm(span=n, adjust=False).mean()

def _dema(close, n):
    e1 = _ema(close, n)
    return 2 * e1 - _ema(e1, n)

def _hull_ma(close, n):
    half = max(int(n / 2), 1)
    sqrt_n = max(int(np.sqrt(n)), 1)
    wma_half = close.rolling(half, min_periods=half).mean()
    wma_full = close.rolling(n, min_periods=n).mean()
    return (2 * wma_half - wma_full).rolling(sqrt_n, min_periods=sqrt_n).mean()

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

def sig_donchian(close, high, n):
    return (close > high.rolling(n, min_periods=n).max().shift(1)).astype(float)

def sig_bollinger(close, n, num_std):
    sma = _sma(close, n)
    return (close > sma + num_std * close.rolling(n, min_periods=n).std()).astype(float)

def sig_keltner(close, high, low, n, atr_mult):
    mid = _ema(close, n)
    atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=n), index=close.index)
    return (close > mid + atr_mult * atr).astype(float)

def sig_supertrend(close, high, low, period, mult):
    atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    hl2 = (high + low) / 2
    upper_band = hl2 + mult * atr
    lower_band = hl2 - mult * atr
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

def sig_momentum(close, n):
    return (close.pct_change(n) > 0).astype(float)

def sig_momentum_threshold(close, n, threshold):
    return (close.pct_change(n) > threshold).astype(float)

def sig_vol_scaled_mom(close, n, vol_lookback):
    ret = close.pct_change(n)
    vol = close.pct_change().rolling(vol_lookback, min_periods=vol_lookback).std()
    return (ret / (vol * np.sqrt(n) + 1e-12) > 0).astype(float)

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

def sig_macd(close, fast, slow, signal_period):
    macd, signal, _ = talib.MACD(close.values, fastperiod=fast, slowperiod=slow, signalperiod=signal_period)
    return pd.Series((macd > signal).astype(float), index=close.index)

def sig_rsi(close, period, threshold):
    return pd.Series((talib.RSI(close.values, timeperiod=period) > threshold).astype(float), index=close.index)

def sig_adx_trend(close, high, low, period, threshold):
    adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
    plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=period)
    minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=period)
    return pd.Series(((adx > threshold) & (plus_di > minus_di)).astype(float), index=close.index)

def sig_cci(close, high, low, period):
    return pd.Series((talib.CCI(high.values, low.values, close.values, timeperiod=period) > 0).astype(float), index=close.index)

def sig_aroon(close, high, low, period):
    down, up = talib.AROON(high.values, low.values, timeperiod=period)
    return pd.Series((up > down).astype(float), index=close.index)

def sig_stoch(close, high, low, fastk, slowk, slowd):
    sk, sd = talib.STOCH(high.values, low.values, close.values, fastk_period=fastk, slowk_period=slowk, slowd_period=slowd)
    return pd.Series((sk > sd).astype(float), index=close.index)

def sig_sar(close, high, low):
    return pd.Series((close.values > talib.SAR(high.values, low.values)).astype(float), index=close.index)

def sig_williams_r(close, high, low, period, threshold):
    return pd.Series((talib.WILLR(high.values, low.values, close.values, timeperiod=period) > threshold).astype(float), index=close.index)

def sig_mfi(close, high, low, volume, period, threshold):
    return pd.Series((talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=period) > threshold).astype(float), index=close.index)

def sig_trix(close, period):
    return pd.Series((talib.TRIX(close.values, timeperiod=period) > 0).astype(float), index=close.index)

def sig_ppo(close, fast, slow):
    return pd.Series((talib.PPO(close.values, fastperiod=fast, slowperiod=slow) > 0).astype(float), index=close.index)

def sig_apo(close, fast, slow):
    return pd.Series((talib.APO(close.values, fastperiod=fast, slowperiod=slow) > 0).astype(float), index=close.index)

def sig_adosc(close, high, low, volume, fast, slow):
    return pd.Series((talib.ADOSC(high.values, low.values, close.values, volume.values, fastperiod=fast, slowperiod=slow) > 0).astype(float), index=close.index)

def sig_mom(close, period):
    return pd.Series((talib.MOM(close.values, timeperiod=period) > 0).astype(float), index=close.index)

def sig_roc(close, period):
    return pd.Series((talib.ROC(close.values, timeperiod=period) > 0).astype(float), index=close.index)

def sig_cmo(close, period):
    return pd.Series((talib.CMO(close.values, timeperiod=period) > 0).astype(float), index=close.index)

def sig_ichimoku(close, high, low, tenkan=9, kijun=26, senkou_b=52):
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    cloud_top = pd.concat([senkou_a, senkou_span_b], axis=1).max(axis=1)
    return (close > cloud_top).astype(float)

def sig_obv_trend(close, volume, sma_period):
    obv = pd.Series(talib.OBV(close.values, volume.values), index=close.index)
    return (obv > obv.rolling(sma_period, min_periods=sma_period).mean()).astype(float)

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
    return ((er > threshold) & (close > close.shift(er_period))).astype(float)

def sig_vwap_trend(close, volume, period):
    cum_vp = (close * volume).rolling(period, min_periods=period).sum()
    cum_v = volume.rolling(period, min_periods=period).sum()
    return (close > cum_vp / (cum_v + 1e-12)).astype(float)

def sig_dual_momentum(close, fast, slow):
    return ((close.pct_change(fast) > 0) & (close.pct_change(slow) > 0)).astype(float)

def sig_triple_ma(close, fast, mid, slow):
    return ((_sma(close, fast) > _sma(close, mid)) & (_sma(close, mid) > _sma(close, slow))).astype(float)

def sig_turtle_breakout(close, high, low, entry_n, exit_n):
    upper = high.rolling(entry_n, min_periods=entry_n).max().shift(1)
    lower = low.rolling(exit_n, min_periods=exit_n).min().shift(1)
    pos = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        if not np.isnan(upper.iloc[i]) and close.iloc[i] > upper.iloc[i]:
            pos.iloc[i] = 1.0
        elif not np.isnan(lower.iloc[i]) and close.iloc[i] < lower.iloc[i]:
            pos.iloc[i] = 0.0
        else:
            pos.iloc[i] = pos.iloc[i-1]
    return pos

def sig_regime_filter_sma(close, trend_n, fast, slow):
    return ((close > _sma(close, trend_n)) & (_sma(close, fast) > _sma(close, slow))).astype(float)

def sig_atr_breakout(close, high, low, period, mult):
    atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    return (close > close.shift(1) + mult * atr.shift(1)).astype(float)

def sig_close_above_high(close, high, n):
    return (close > high.shift(1).rolling(n, min_periods=n).max()).astype(float)

def sig_mean_reversion_band(close, n, num_std):
    sma = _sma(close, n)
    return (close < sma - num_std * close.rolling(n, min_periods=n).std()).astype(float)


# =====================================================================
# Signal dispatcher
# =====================================================================

def dispatch_signal(sig_type, close, high, low, open_, volume, **kw):
    c, h, l, o, v = close, high, low, open_, volume
    m = {
        "sma_cross":      lambda: sig_sma_cross(c, kw["fast"], kw["slow"]),
        "ema_cross":      lambda: sig_ema_cross(c, kw["fast"], kw["slow"]),
        "dema_cross":     lambda: sig_dema_cross(c, kw["fast"], kw["slow"]),
        "hull_cross":     lambda: sig_hull_cross(c, kw["fast"], kw["slow"]),
        "price_sma":      lambda: sig_price_above_sma(c, kw["n"]),
        "price_ema":      lambda: sig_price_above_ema(c, kw["n"]),
        "donchian":       lambda: sig_donchian(c, h, kw["n"]),
        "bollinger":      lambda: sig_bollinger(c, kw["n"], kw["num_std"]),
        "keltner":        lambda: sig_keltner(c, h, l, kw["n"], kw["atr_mult"]),
        "supertrend":     lambda: sig_supertrend(c, h, l, kw["period"], kw["mult"]),
        "momentum":       lambda: sig_momentum(c, kw["n"]),
        "mom_threshold":  lambda: sig_momentum_threshold(c, kw["n"], kw["threshold"]),
        "vol_scaled_mom": lambda: sig_vol_scaled_mom(c, kw["n"], kw["vol_lookback"]),
        "lreg":           lambda: sig_lreg(c, kw["n"]),
        "macd":           lambda: sig_macd(c, kw["fast"], kw["slow"], kw["signal_period"]),
        "rsi":            lambda: sig_rsi(c, kw["period"], kw["threshold"]),
        "adx":            lambda: sig_adx_trend(c, h, l, kw["period"], kw["threshold"]),
        "cci":            lambda: sig_cci(c, h, l, kw["period"]),
        "aroon":          lambda: sig_aroon(c, h, l, kw["period"]),
        "stoch":          lambda: sig_stoch(c, h, l, kw["fastk"], kw["slowk"], kw["slowd"]),
        "sar":            lambda: sig_sar(c, h, l),
        "williams_r":     lambda: sig_williams_r(c, h, l, kw["period"], kw["threshold"]),
        "mfi":            lambda: sig_mfi(c, h, l, v, kw["period"], kw["threshold"]),
        "trix":           lambda: sig_trix(c, kw["period"]),
        "ppo":            lambda: sig_ppo(c, kw["fast"], kw["slow"]),
        "apo":            lambda: sig_apo(c, kw["fast"], kw["slow"]),
        "adosc":          lambda: sig_adosc(c, h, l, v, kw["fast"], kw["slow"]),
        "talib_mom":      lambda: sig_mom(c, kw["period"]),
        "talib_roc":      lambda: sig_roc(c, kw["period"]),
        "talib_cmo":      lambda: sig_cmo(c, kw["period"]),
        "ichimoku":       lambda: sig_ichimoku(c, h, l, kw["tenkan"], kw["kijun"], kw["senkou_b"]),
        "obv_trend":      lambda: sig_obv_trend(c, v, kw["sma_period"]),
        "heikin_ashi":    lambda: sig_heikin_ashi(o, h, l, c),
        "kaufman":        lambda: sig_kaufman_er(c, kw["er_period"], kw["threshold"]),
        "vwap":           lambda: sig_vwap_trend(c, v, kw["period"]),
        "dual_mom":       lambda: sig_dual_momentum(c, kw["fast"], kw["slow"]),
        "triple_ma":      lambda: sig_triple_ma(c, kw["fast"], kw["mid"], kw["slow"]),
        "turtle":         lambda: sig_turtle_breakout(c, h, l, kw["entry_n"], kw["exit_n"]),
        "regime_sma":     lambda: sig_regime_filter_sma(c, kw["trend_n"], kw["fast"], kw["slow"]),
        "atr_breakout":   lambda: sig_atr_breakout(c, h, l, kw["period"], kw["mult"]),
        "close_above_high": lambda: sig_close_above_high(c, h, kw["n"]),
        "mean_rev_band":  lambda: sig_mean_reversion_band(c, kw["n"], kw["num_std"]),
    }
    return m[sig_type]()


# =====================================================================
# Configuration grid (same as v1)
# =====================================================================

def build_configs(freq: str, has_ohlcv: bool = True):
    configs = []
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

    for n in [5, 10, 15, 20, 30, 50, 100, 150, 200]:
        configs.append((f"Price_above_SMA_{n}", "price_sma", {"n": n}))
        configs.append((f"Price_above_EMA_{n}", "price_ema", {"n": n}))

    if has_ohlcv:
        for n in [5, 10, 15, 20, 30, 50, 100]:
            configs.append((f"Donchian_{n}", "donchian", {"n": n}))
        for n, std in product([10, 15, 20, 30, 50], [1.0, 1.5, 2.0, 2.5, 3.0]):
            configs.append((f"Boll_{n}_{std}", "bollinger", {"n": n, "num_std": std}))
        for n, mult in product([7, 10, 14, 20, 30], [1.0, 1.5, 2.0, 2.5, 3.0]):
            configs.append((f"Keltner_{n}_{mult}", "keltner", {"n": n, "atr_mult": mult}))
        for period, mult in product([7, 10, 14, 20, 30], [1.0, 1.5, 2.0, 2.5, 3.0]):
            configs.append((f"Supertrend_{period}_{mult}", "supertrend", {"period": period, "mult": mult}))

    for n in [3, 5, 7, 10, 15, 20, 30, 50, 63, 100, 126, 200]:
        configs.append((f"Momentum_{n}", "momentum", {"n": n}))
    for n, thresh in product([5, 10, 20, 30, 50, 100], [0.01, 0.02, 0.05, 0.10]):
        configs.append((f"MomThresh_{n}_{thresh}", "mom_threshold", {"n": n, "threshold": thresh}))
    for n, vl in product([5, 10, 20, 30, 50, 63], [21, 42, 63]):
        configs.append((f"VolScaledMom_{n}_{vl}", "vol_scaled_mom", {"n": n, "vol_lookback": vl}))
    for n in [5, 7, 10, 15, 20, 30, 50]:
        configs.append((f"LREG_{n}", "lreg", {"n": n}))

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
            configs.append((f"UltOsc_{p1}_{p2}_{p3}_{thresh}", "ultosc_not_impl", {}))
        for fast, slow in [(3, 10), (5, 20), (10, 30)]:
            configs.append((f"ADOSC_{fast}_{slow}", "adosc", {"fast": fast, "slow": slow}))
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

    # Filter out stubs
    configs = [(l, s, k) for l, s, k in configs if s != "ultosc_not_impl"]
    return configs


# =====================================================================
# Main sweep with stop overlays
# =====================================================================

def run_sweep_on_freq(freq: str, daily_eth: pd.DataFrame, symbol: str = None):
    sym = symbol or SYMBOL
    print(f"\n{'='*70}")
    print(f"  FREQUENCY: {freq}")
    print(f"{'='*70}")

    eth = load_asset(freq, symbol=sym)
    print(f"  Loaded {len(eth)} bars for {sym} at {freq}")

    close = eth["close"]
    high = eth["high"]
    low = eth["low"]
    open_ = eth["open"]
    volume = eth["volume"]

    daily_close = daily_eth["close"]
    daily_high = daily_eth["high"]
    daily_low = daily_eth["low"]
    daily_returns = daily_close.pct_change(fill_method=None)

    configs = build_configs(freq, has_ohlcv=True)
    n_stops = len(PCT_STOP_LABELS) + len(ATR_LABELS)
    total_with_stops = len(configs) * n_stops
    print(f"  Base signals: {len(configs)}")
    print(f"  Stop variants: {n_stops} ({', '.join(ALL_STOP_LABELS)})")
    print(f"  Total configs: {total_with_stops}")

    results = []
    t0 = time.time()

    for i, (label, sig_type, kwargs) in enumerate(configs):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(configs) - i - 1) / rate
            print(f"  [{i+1}/{len(configs)} signals] {elapsed:.0f}s elapsed, ~{eta:.0f}s remain "
                  f"({len(results)} results so far)")

        try:
            raw_signal = dispatch_signal(sig_type, close, high, low, open_, volume, **kwargs)

            if freq != "1d":
                base_signal = resample_to_daily(raw_signal, daily_close)
            else:
                base_signal = raw_signal

            base_signal = base_signal.dropna()
            if len(base_signal) < 100:
                continue

            # Fixed-% stops
            for stop_pct, stop_label in zip(PCT_STOP_LEVELS, PCT_STOP_LABELS):
                if stop_pct is None:
                    signal = base_signal
                else:
                    signal = apply_trailing_stop(base_signal, daily_close, stop_pct)

                equity, net_ret, pos = backtest_signal(signal, daily_returns)
                perf = compute_perf(equity, net_ret, pos)

                if perf is not None:
                    perf["label"] = label
                    perf["stop"] = stop_label
                    perf["stop_type"] = "none" if stop_pct is None else "pct"
                    perf["full_label"] = f"{label}|stop={stop_label}"
                    perf["signal_family"] = label.split("_")[0]
                    perf["freq"] = freq
                    perf["params"] = str(kwargs)
                    results.append(perf)

            # ATR-based vol-adaptive stops
            for atr_mult, atr_label in zip(ATR_MULTS, ATR_LABELS):
                signal = apply_atr_trailing_stop(
                    base_signal, daily_close, daily_high, daily_low,
                    atr_mult=atr_mult, atr_period=14,
                )
                equity, net_ret, pos = backtest_signal(signal, daily_returns)
                perf = compute_perf(equity, net_ret, pos)

                if perf is not None:
                    perf["label"] = label
                    perf["stop"] = atr_label
                    perf["stop_type"] = "atr"
                    perf["full_label"] = f"{label}|stop={atr_label}"
                    perf["signal_family"] = label.split("_")[0]
                    perf["freq"] = freq
                    perf["params"] = str(kwargs)
                    results.append(perf)

        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"  Completed: {len(results)} valid configs in {elapsed:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETH-USD", help="Asset symbol (e.g. BTC-USD)")
    parser.add_argument("--daily-only", action="store_true", help="Skip 4h/1h frequencies")
    args = parser.parse_args()

    global SYMBOL, OUT_DIR
    SYMBOL = args.symbol
    OUT_DIR = _out_dir(SYMBOL)

    print("=" * 70)
    print(f"  {SYMBOL} TREND SWEEP v2 — WITH TRAILING STOPS")
    print("=" * 70)

    daily_asset = load_asset("1d", symbol=SYMBOL)
    print(f"  Daily {SYMBOL}: {len(daily_asset)} bars, {daily_asset.index.min()} to {daily_asset.index.max()}")

    daily_close = daily_asset["close"]
    daily_returns = daily_close.pct_change(fill_method=None).dropna()

    # Buy & hold baseline
    bh_equity = (1 + daily_returns).cumprod()
    bh_perf = compute_perf(bh_equity, daily_returns, pd.Series(1.0, index=daily_returns.index))

    all_results = []

    freqs = ["1d"] if args.daily_only else ["1d", "4h", "1h"]
    for freq in freqs:
        try:
            results = run_sweep_on_freq(freq, daily_asset, symbol=SYMBOL)
            all_results.extend(results)
        except Exception as e:
            print(f"  [WARN] {freq} sweep failed: {e}")

    if not all_results:
        print("\n  NO VALID RESULTS")
        return

    df = pd.DataFrame(all_results)

    # Add B&H row
    if bh_perf:
        bh_row = bh_perf.copy()
        bh_row["label"] = "BUY_AND_HOLD"
        bh_row["stop"] = "none"
        bh_row["full_label"] = "BUY_AND_HOLD"
        bh_row["signal_family"] = "BENCHMARK"
        bh_row["freq"] = "1d"
        bh_row["params"] = "{}"
        df = pd.concat([df, pd.DataFrame([bh_row])], ignore_index=True)

    df = df.sort_values("sharpe", ascending=False)
    df.to_csv(OUT_DIR / "results_v2.csv", index=False, float_format="%.4f")
    print(f"\n  Total valid results: {len(df)}")
    print(f"  Saved to {OUT_DIR / 'results_v2.csv'}")

    # ── B&H baseline ─────────────────────────────────────────────────
    if bh_perf:
        print(f"\n  {'='*90}")
        print(f"  BUY & HOLD BASELINE")
        print(f"  Sharpe: {bh_perf['sharpe']:.3f}  CAGR: {bh_perf['cagr']:.1%}  "
              f"MaxDD: {bh_perf['max_dd']:.1%}  Calmar: {bh_perf['calmar']:.2f}  "
              f"Skew: {bh_perf['skewness']:.3f}")
        print(f"  {'='*90}")

    # ── Rankings ──────────────────────────────────────────────────────
    hdr = (f"  {'Rk':>3s} {'Signal':<30s} {'Stop':>5s} {'Freq':>4s} "
           f"{'Sharpe':>7s} {'CAGR':>7s} {'MaxDD':>7s} {'Calmar':>7s} "
           f"{'Skew':>7s} {'TIM':>5s} {'#Tr':>5s}")
    sep = f"  {'─'*3} {'─'*30} {'─'*5} {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*5} {'─'*5}"

    def fmt_row(rank, r):
        return (f"  {rank:>3d} {r['label']:<30s} {r['stop']:>5s} {r['freq']:>4s} "
                f"{r['sharpe']:>7.3f} {r['cagr']:>6.1%} {r['max_dd']:>6.1%} "
                f"{r['calmar']:>7.2f} {r['skewness']:>7.3f} "
                f"{r['time_in_market']:>4.0%} {r['n_trades']:>5d}")

    def print_top(metric, n=30, ascending=False):
        print(f"\n  {'─'*90}")
        print(f"  TOP {n} BY {metric.upper()}")
        print(f"  {'─'*90}")
        print(hdr)
        print(sep)
        top = df.nlargest(n, metric) if not ascending else df.nsmallest(n, metric)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            print(fmt_row(rank, row))
        if bh_perf:
            print(sep)
            bh = df[df["label"] == "BUY_AND_HOLD"].iloc[0]
            print(f"  {'B&H':>3s} {'BUY_AND_HOLD':<30s} {'none':>5s} {'1d':>4s} "
                  f"{bh['sharpe']:>7.3f} {bh['cagr']:>6.1%} {bh['max_dd']:>6.1%} "
                  f"{bh['calmar']:>7.2f} {bh['skewness']:>7.3f} "
                  f"{bh['time_in_market']:>4.0%} {bh['n_trades']:>5d}")

    print_top("sharpe")
    print_top("calmar")
    print_top("max_dd", ascending=False)

    # ── Best drawdown by stop level ──────────────────────────────────
    print(f"\n  {'─'*90}")
    print(f"  DRAWDOWN IMPROVEMENT BY STOP LEVEL (best config per stop by Calmar)")
    print(f"  {'─'*90}")
    print(f"  {'Stop':>8s} {'Type':>4s} {'Best Signal':<28s} {'Freq':>4s} {'MaxDD':>7s} {'Sharpe':>7s} "
          f"{'CAGR':>7s} {'Calmar':>7s} {'Skew':>7s} {'TIM':>5s}")
    print(f"  {'─'*8} {'─'*4} {'─'*28} {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*5}")

    strategies_only = df[df["label"] != "BUY_AND_HOLD"]
    for stop_label in ALL_STOP_LABELS:
        subset = strategies_only[strategies_only["stop"] == stop_label]
        if subset.empty:
            continue
        best_by_calmar = subset.loc[subset["calmar"].idxmax()]
        r = best_by_calmar
        stype = r.get("stop_type", "?")
        print(f"  {stop_label:>8s} {stype:>4s} {r['label']:<28s} {r['freq']:>4s} {r['max_dd']:>6.1%} "
              f"{r['sharpe']:>7.3f} {r['cagr']:>6.1%} {r['calmar']:>7.2f} "
              f"{r['skewness']:>7.3f} {r['time_in_market']:>4.0%}")
    if bh_perf:
        print(f"  {'B&H':>8s} {'─':>4s} {'BUY_AND_HOLD':<28s} {'1d':>4s} {bh_perf['max_dd']:>6.1%} "
              f"{bh_perf['sharpe']:>7.3f} {bh_perf['cagr']:>6.1%} {bh_perf['calmar']:>7.2f} "
              f"{bh_perf['skewness']:>7.3f} {bh_perf['time_in_market']:>4.0%}")

    # ── Stop effect on top 10 base signals ────────────────────────────
    print(f"\n  {'─'*100}")
    print(f"  STOP EFFECT ON TOP 10 BASE SIGNALS (sorted by no-stop Sharpe)")
    print(f"  {'─'*100}")
    no_stop = strategies_only[strategies_only["stop"] == "none"].nlargest(10, "sharpe")

    for _, base_row in no_stop.iterrows():
        print(f"\n  {base_row['label']} ({base_row['freq']})")
        print(f"  {'Stop':>8s} {'Type':>4s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s} "
              f"{'Skew':>8s} {'TIM':>6s} {'Trades':>7s}")
        print(f"  {'─'*8} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*7}")

        variants = strategies_only[
            (strategies_only["label"] == base_row["label"]) &
            (strategies_only["freq"] == base_row["freq"])
        ]
        # Sort: none first, then pct ascending, then atr ascending
        order = {s: i for i, s in enumerate(ALL_STOP_LABELS)}
        variants = variants.copy()
        variants["_order"] = variants["stop"].map(order)
        variants = variants.sort_values("_order")

        for _, v in variants.iterrows():
            stype = v.get("stop_type", "?")
            marker = " ← base" if v["stop"] == "none" else ""
            print(f"  {v['stop']:>8s} {stype:>4s} {v['sharpe']:>8.3f} {v['cagr']:>7.1%} {v['max_dd']:>7.1%} "
                  f"{v['calmar']:>8.2f} {v['skewness']:>8.3f} "
                  f"{v['time_in_market']:>5.0%} {v['n_trades']:>7d}{marker}")

    # ── Best per family by Calmar ─────────────────────────────────────
    print(f"\n  {'─'*90}")
    print(f"  BEST PER SIGNAL FAMILY (by Calmar = CAGR / |MaxDD|)")
    print(f"  {'─'*90}")
    print(f"  {'Family':<16s} {'Config':<28s} {'Stop':>5s} {'Freq':>4s} "
          f"{'Calmar':>7s} {'Sharpe':>7s} {'CAGR':>7s} {'MaxDD':>7s} {'Skew':>7s}")
    valid = strategies_only.dropna(subset=["calmar"])
    best_per_family = valid.loc[valid.groupby("signal_family")["calmar"].idxmax()]
    best_per_family = best_per_family.sort_values("calmar", ascending=False)
    for _, r in best_per_family.head(25).iterrows():
        print(f"  {r['signal_family']:<16s} {r['label']:<28s} {r['stop']:>5s} {r['freq']:>4s} "
              f"{r['calmar']:>7.2f} {r['sharpe']:>7.3f} {r['cagr']:>6.1%} "
              f"{r['max_dd']:>6.1%} {r['skewness']:>7.3f}")

    # ── Charts ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Sharpe vs MaxDD
    for _, row in strategies_only.iterrows():
        c = "navy" if row["stop"] == "none" else "orange"
        a = 0.3 if row["stop"] == "none" else 0.15
        axes[0, 0].scatter(row["max_dd"], row["sharpe"], c=c, alpha=a, s=10, edgecolors="none")
    if bh_perf:
        axes[0, 0].scatter(bh_perf["max_dd"], bh_perf["sharpe"], c="red", s=120, marker="*", zorder=10, label="B&H")
    axes[0, 0].set_xlabel("Max Drawdown")
    axes[0, 0].set_ylabel("Sharpe")
    axes[0, 0].set_title("Sharpe vs Max Drawdown (navy=no stop, orange=with stop)")
    axes[0, 0].legend()
    axes[0, 0].axhline(0, color="black", linewidth=0.5)

    # Calmar vs Skewness
    valid_plot = strategies_only.dropna(subset=["calmar"])
    for _, row in valid_plot.iterrows():
        c = "navy" if row["stop"] == "none" else "orange"
        a = 0.3 if row["stop"] == "none" else 0.15
        axes[0, 1].scatter(row["skewness"], row["calmar"], c=c, alpha=a, s=10, edgecolors="none")
    if bh_perf:
        axes[0, 1].scatter(bh_perf["skewness"], bh_perf["calmar"], c="red", s=120, marker="*", zorder=10, label="B&H")
    axes[0, 1].set_xlabel("Skewness")
    axes[0, 1].set_ylabel("Calmar (CAGR/|MaxDD|)")
    axes[0, 1].set_title("Calmar vs Skewness")
    axes[0, 1].legend()

    # MaxDD distribution by stop level
    stop_dd = []
    labels_present = []
    for sl in ALL_STOP_LABELS:
        subset = strategies_only[strategies_only["stop"] == sl]["max_dd"]
        if len(subset) > 0:
            stop_dd.append(subset.values)
            labels_present.append(sl)
    bp = axes[1, 0].boxplot(stop_dd, labels=labels_present, patch_artist=True)
    pct_count = len(PCT_STOP_LABELS)
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("lightblue" if idx < pct_count else "lightyellow")
    if bh_perf:
        axes[1, 0].axhline(bh_perf["max_dd"], color="red", linewidth=1.5, linestyle="--", label=f"B&H ({bh_perf['max_dd']:.0%})")
    axes[1, 0].set_ylabel("Max Drawdown")
    axes[1, 0].set_xlabel("Stop (blue=fixed%, yellow=ATR)")
    axes[1, 0].set_title("Drawdown Distribution by Stop Level")
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Sharpe distribution by stop level
    stop_sharpe = []
    for sl in labels_present:
        subset = strategies_only[strategies_only["stop"] == sl]["sharpe"]
        if len(subset) > 0:
            stop_sharpe.append(subset.values)
    bp2 = axes[1, 1].boxplot(stop_sharpe, labels=labels_present, patch_artist=True)
    for idx, patch in enumerate(bp2["boxes"]):
        patch.set_facecolor("lightblue" if idx < pct_count else "lightyellow")
    if bh_perf:
        axes[1, 1].axhline(bh_perf["sharpe"], color="red", linewidth=1.5, linestyle="--", label=f"B&H ({bh_perf['sharpe']:.2f})")
    axes[1, 1].set_ylabel("Sharpe")
    axes[1, 1].set_xlabel("Stop (blue=fixed%, yellow=ATR)")
    axes[1, 1].set_title("Sharpe Distribution by Stop Level")
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep_v2_summary.png", dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved to {OUT_DIR / 'sweep_v2_summary.png'}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n  {'='*90}")
    print(f"  SUMMARY")
    print(f"  {'='*90}")
    print(f"  Total configs tested: {len(df)}")

    print(f"\n  {'Stop':>8s} {'Type':>4s} {'N':>6s} {'Med Sharpe':>11s} {'Med MaxDD':>10s} "
          f"{'Med Calmar':>11s} {'Med Skew':>9s} {'Med TIM':>8s}")
    print(f"  {'─'*8} {'─'*4} {'─'*6} {'─'*11} {'─'*10} {'─'*11} {'─'*9} {'─'*8}")
    for sl in ALL_STOP_LABELS:
        subset = strategies_only[strategies_only["stop"] == sl]
        if subset.empty:
            continue
        stype = "pct" if sl.startswith("pct") else ("atr" if sl.startswith("atr") else "none")
        med_sharpe = subset["sharpe"].median()
        med_dd = subset["max_dd"].median()
        med_calmar = subset["calmar"].median()
        med_skew = subset["skewness"].median()
        med_tim = subset["time_in_market"].median()
        print(f"  {sl:>8s} {stype:>4s} {len(subset):>6d} {med_sharpe:>11.3f} {med_dd:>9.1%} "
              f"{med_calmar:>11.2f} {med_skew:>9.3f} {med_tim:>7.0%}")

    if bh_perf:
        print(f"  {'':>12s} B&H:        Sharpe={bh_perf['sharpe']:>6.3f}  "
              f"MaxDD={bh_perf['max_dd']:>6.1%}  Calmar={bh_perf['calmar']:>6.2f}")

    print(f"\n  {'='*90}")
    print(f"  SWEEP v2 COMPLETE")
    print(f"  {'='*90}")


if __name__ == "__main__":
    main()
