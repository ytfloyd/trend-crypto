"""
Expanded TA-Lib feature computation (~95 indicators).

Extends the 54-feature set from jpm_bigdata_ai/helpers.py with additional
TA-Lib indicators and multi-period variants. All features are shifted by 1
bar to prevent lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from common.data import ANN_FACTOR

RETURN_LOOKBACKS = [5, 10, 21, 42, 63]

# Populated by compute_all_features()
ALL_FEATURE_COLS: list[str] = []


def compute_all_features(
    panel: pd.DataFrame,
    lookbacks: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build expanded feature matrix from daily OHLCV panel.

    Returns (panel_with_features, list_of_feature_column_names).
    """
    if lookbacks is None:
        lookbacks = RETURN_LOOKBACKS

    def _per_sym(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        o = g["open"].values.astype(np.float64)
        h = g["high"].values.astype(np.float64)
        lo = g["low"].values.astype(np.float64)
        c = g["close"].values.astype(np.float64)
        v = g["volume"].values.astype(np.float64)
        log_ret = np.log(g["close"] / g["close"].shift(1))

        # ── Returns, volatility, volume ratios ────────────────────────
        for lb in lookbacks:
            g[f"ret_{lb}d"] = g["close"].shift(1) / g["close"].shift(1 + lb) - 1.0
            g[f"vol_{lb}d"] = (
                log_ret.shift(1).rolling(lb, min_periods=lb).std()
                * np.sqrt(ANN_FACTOR)
            )
            vol_ma = g["volume"].shift(1).rolling(lb, min_periods=lb).mean()
            g[f"vol_ratio_{lb}d"] = g["volume"].shift(1) / vol_ma.clip(lower=1.0)

        # ── Trend ─────────────────────────────────────────────────────
        for p in [7, 14, 28, 42]:
            g[f"adx_{p}"] = talib.ADX(h, lo, c, timeperiod=p)
        g[f"dx_14"] = talib.DX(h, lo, c, timeperiod=14)
        g["plus_di_14"] = talib.PLUS_DI(h, lo, c, timeperiod=14)
        g["minus_di_14"] = talib.MINUS_DI(h, lo, c, timeperiod=14)
        g["di_spread"] = g["plus_di_14"] - g["minus_di_14"]

        for fast, slow, sig in [(12, 26, 9), (5, 15, 5)]:
            macd, macd_sig, macd_hist = talib.MACD(
                c, fastperiod=fast, slowperiod=slow, signalperiod=sig
            )
            sfx = f"_{fast}_{slow}"
            g[f"macd{sfx}"] = macd
            g[f"macd_signal{sfx}"] = macd_sig
            g[f"macd_hist{sfx}"] = macd_hist

        aroon_down, aroon_up = talib.AROON(h, lo, timeperiod=14)
        g["aroon_osc"] = aroon_up - aroon_down
        g["aroon_up"] = aroon_up
        g["aroon_down"] = aroon_down

        for p in [14, 42]:
            g[f"linearreg_slope_{p}"] = talib.LINEARREG_SLOPE(c, timeperiod=p)

        g["sar"] = talib.SAR(h, lo, acceleration=0.02, maximum=0.2)
        g["sar_dist"] = (c - g["sar"].values) / np.where(c > 0, c, np.nan)

        g["ht_trendline"] = talib.HT_TRENDLINE(c)
        g["ht_trend_dist"] = (c - g["ht_trendline"].values) / np.where(c > 0, c, np.nan)
        g["ht_trendmode"] = talib.HT_TRENDMODE(c).astype(np.float64)

        for p in [10, 21, 50]:
            g[f"kama_{p}"] = talib.KAMA(c, timeperiod=p)
            g[f"kama_dist_{p}"] = (c - g[f"kama_{p}"].values) / np.where(c > 0, c, np.nan)

        # ── Momentum / oscillators ────────────────────────────────────
        for p in [7, 14, 21, 28, 42]:
            g[f"rsi_{p}"] = talib.RSI(c, timeperiod=p)

        slowk, slowd = talib.STOCH(h, lo, c, fastk_period=14, slowk_period=3, slowd_period=3)
        g["stoch_k"] = slowk
        g["stoch_d"] = slowd
        fastk, fastd = talib.STOCHF(h, lo, c, fastk_period=14, fastd_period=3)
        g["stochf_k"] = fastk
        g["stochf_d"] = fastd
        srsi_k, srsi_d = talib.STOCHRSI(c, timeperiod=14, fastk_period=5, fastd_period=3)
        g["stochrsi_k"] = srsi_k
        g["stochrsi_d"] = srsi_d

        for p in [7, 14, 28, 42]:
            g[f"cci_{p}"] = talib.CCI(h, lo, c, timeperiod=p)

        g["willr_14"] = talib.WILLR(h, lo, c, timeperiod=14)
        g["mfi_14"] = talib.MFI(h, lo, c, v, timeperiod=14)
        g["mfi_28"] = talib.MFI(h, lo, c, v, timeperiod=28)

        for p in [5, 10, 21, 42, 63]:
            g[f"roc_{p}"] = talib.ROC(c, timeperiod=p)
        for p in [10, 21]:
            g[f"mom_{p}"] = talib.MOM(c, timeperiod=p)

        g["ultosc"] = talib.ULTOSC(h, lo, c, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        g["apo"] = talib.APO(c, fastperiod=12, slowperiod=26)
        g["ppo"] = talib.PPO(c, fastperiod=12, slowperiod=26)
        g["cmo_14"] = talib.CMO(c, timeperiod=14)
        g["bop"] = talib.BOP(o, h, lo, c)
        g["trix_14"] = talib.TRIX(c, timeperiod=14)
        g["trix_28"] = talib.TRIX(c, timeperiod=28)

        # ── Volatility ────────────────────────────────────────────────
        for p in [7, 14, 28]:
            g[f"atr_{p}"] = talib.ATR(h, lo, c, timeperiod=p)
            g[f"natr_{p}"] = talib.NATR(h, lo, c, timeperiod=p)

        for p in [10, 20, 30]:
            upper, mid, lower = talib.BBANDS(c, timeperiod=p, nbdevup=2, nbdevdn=2)
            g[f"bb_width_{p}"] = (upper - lower) / np.where(mid > 0, mid, np.nan)
            g[f"bb_pctb_{p}"] = (c - lower) / np.where(
                (upper - lower) > 0, upper - lower, np.nan
            )

        g["hl_range"] = (h - lo) / np.where(c > 0, c, np.nan)
        g["trange"] = talib.TRANGE(h, lo, c)
        g["trange_norm"] = g["trange"].values / np.where(c > 0, c, np.nan)

        # ── Volume ────────────────────────────────────────────────────
        obv = talib.OBV(c, v)
        g["obv_slope_14"] = talib.LINEARREG_SLOPE(obv, timeperiod=14)
        g["obv_slope_42"] = talib.LINEARREG_SLOPE(obv, timeperiod=42)
        ad = talib.AD(h, lo, c, v)
        g["ad_slope_14"] = talib.LINEARREG_SLOPE(ad, timeperiod=14)

        # ── Price structure ───────────────────────────────────────────
        for p in [14, 42]:
            chan_lo = talib.MIN(lo, p)
            chan_hi = talib.MAX(h, p)
            rng = chan_hi - chan_lo
            g[f"channel_pos_{p}"] = (c - chan_lo) / np.where(rng > 0, rng, np.nan)

        # ── Open-price / candle features ──────────────────────────────
        prev_c = np.roll(c, 1); prev_c[0] = np.nan
        g["overnight_gap"] = (o - prev_c) / np.where(prev_c > 0, prev_c, np.nan)
        body = c - o
        full_range = h - lo
        g["body_ratio"] = body / np.where(full_range > 0, full_range, np.nan)
        g["upper_shadow"] = (h - np.maximum(o, c)) / np.where(full_range > 0, full_range, np.nan)
        g["lower_shadow"] = (np.minimum(o, c) - lo) / np.where(full_range > 0, full_range, np.nan)

        # ── Candlestick patterns (keep the standard 7) ───────────────
        g["cdl_doji"] = talib.CDLDOJI(o, h, lo, c).astype(np.float64)
        g["cdl_hammer"] = talib.CDLHAMMER(o, h, lo, c).astype(np.float64)
        g["cdl_engulfing"] = talib.CDLENGULFING(o, h, lo, c).astype(np.float64)
        g["cdl_morningstar"] = talib.CDLMORNINGSTAR(o, h, lo, c).astype(np.float64)
        g["cdl_eveningstar"] = talib.CDLEVENINGSTAR(o, h, lo, c).astype(np.float64)
        g["cdl_3whitesoldiers"] = talib.CDL3WHITESOLDIERS(o, h, lo, c).astype(np.float64)
        g["cdl_3blackcrows"] = talib.CDL3BLACKCROWS(o, h, lo, c).astype(np.float64)

        # ── Shift all TA-Lib features by 1 bar ───────────────────────
        skip = {"symbol", "ts", "open", "high", "low", "close", "volume",
                "dollar_volume", "in_universe"}
        ret_vol_cols = set()
        for lb in lookbacks:
            ret_vol_cols |= {f"ret_{lb}d", f"vol_{lb}d", f"vol_ratio_{lb}d"}

        for col in g.columns:
            if col not in skip and col not in ret_vol_cols:
                g[col] = g[col].shift(1)

        return g

    result = panel.groupby("symbol", group_keys=False).apply(_per_sym)

    skip = {"symbol", "ts", "open", "high", "low", "close", "volume",
            "dollar_volume", "in_universe"}
    feat_cols = [c for c in result.columns if c not in skip]

    ALL_FEATURE_COLS.clear()
    ALL_FEATURE_COLS.extend(feat_cols)

    return result, feat_cols
