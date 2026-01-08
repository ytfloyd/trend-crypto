#!/usr/bin/env python
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ANN_FACTOR = 365.0


def _win(d: float) -> int:
    return max(1, int(math.floor(d)))


def _to_panel(df: pd.DataFrame) -> pd.DataFrame:
    if not {"ts", "symbol"}.issubset(df.columns):
        raise ValueError("Panel must include ts and symbol columns")
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    out = out.sort_values(["symbol", "ts"])
    out = out.set_index(["symbol", "ts"])
    return out


# ----------------------------------------------------------------------
# Core operators (Appendix A.2 inspired)
# ----------------------------------------------------------------------
def cs_rank(s: pd.Series) -> pd.Series:
    """
    Cross-sectional rank within each timestamp.

    Expects a Series indexed by [ts, symbol] (MultiIndex) or by ts with
    duplicate timestamps (one per symbol). Returns a Series of ranks in
    [0, 1], aligned with the original index.
    """
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    idx = s.index
    if isinstance(idx, pd.MultiIndex) and "ts" in idx.names:
        grouped = s.groupby(level="ts")
    else:
        grouped = s.groupby(idx)

    ranked = grouped.rank(method="average", pct=True)
    ranked.name = s.name
    return ranked


def delay(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    return s.groupby(level="symbol").shift(d)


def delta(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    return s - delay(s, d)


def ts_sum(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    return (
        s.groupby(level="symbol")
        .rolling(window=d, min_periods=d)
        .sum()
        .reset_index(level=0, drop=True)
    )


def ts_product(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    return (
        s.groupby(level="symbol")
        .rolling(window=d, min_periods=d)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
    )


def ts_min(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    return (
        s.groupby(level="symbol")
        .rolling(window=d, min_periods=d)
        .min()
        .reset_index(level=0, drop=True)
    )


def ts_max(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    return (
        s.groupby(level="symbol")
        .rolling(window=d, min_periods=d)
        .max()
        .reset_index(level=0, drop=True)
    )


def ts_argmin(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    return (
        s.groupby(level="symbol")
        .rolling(window=d, min_periods=d)
        .apply(lambda x: float(np.argmin(x)), raw=True)
        .reset_index(level=0, drop=True)
    )


def ts_argmax(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    return (
        s.groupby(level="symbol")
        .rolling(window=d, min_periods=d)
        .apply(lambda x: float(np.argmax(x)), raw=True)
        .reset_index(level=0, drop=True)
    )


def ts_rank(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    def _rank_last(x: Iterable[float]) -> float:
        arr = pd.Series(x)
        return float(arr.rank(pct=True, method="average").iloc[-1])

    return (
        s.groupby(level="symbol")
        .rolling(window=d, min_periods=d)
        .apply(_rank_last, raw=False)
        .reset_index(level=0, drop=True)
    )


def correlation(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    """
    Time-series correlation of x and y over a rolling window.

    - If index is MultiIndex with a 'symbol' level, compute correlation per symbol.
    - Otherwise, compute a single rolling correlation over the flat index.

    Returns a Series aligned to x's index.
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Align indices first
    x, y = x.align(y, join="inner")

    idx = x.index

    if isinstance(idx, pd.MultiIndex) and "symbol" in idx.names:
        df = pd.DataFrame({"x": x, "y": y})

        def _corr(group: pd.DataFrame) -> pd.Series:
            return group["x"].rolling(window=d, min_periods=d).corr(group["y"])

        out = df.groupby(level="symbol", group_keys=False).apply(_corr)
    else:
        out = x.sort_index().rolling(window=d, min_periods=d).corr(y.sort_index())

    out.name = x.name
    return out


def covariance(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    x, y = x.align(y, join="inner")
    idx = x.index

    if isinstance(idx, pd.MultiIndex) and "symbol" in idx.names:
        df = pd.DataFrame({"x": x, "y": y})

        def _cov(group: pd.DataFrame) -> pd.Series:
            return group["x"].rolling(window=d, min_periods=d).cov(group["y"])

        out = df.groupby(level="symbol", group_keys=False).apply(_cov)
    else:
        out = x.sort_index().rolling(window=d, min_periods=d).cov(y.sort_index())

    out.name = x.name
    return out


def stddev(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    return (
        s.groupby(level="symbol")
        .rolling(window=d, min_periods=d)
        .std()
        .reset_index(level=0, drop=True)
    )


def decay_linear(s: pd.Series, d: int) -> pd.Series:
    d = _win(d)
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()

    def _apply(x: np.ndarray) -> float:
        return float(np.dot(x, weights[-len(x) :]))  # align shortest windows

    return (
        s.groupby(level="symbol")
        .rolling(window=d, min_periods=d)
        .apply(_apply, raw=True)
        .reset_index(level=0, drop=True)
    )


def scale(s: pd.Series, a: float = 1.0) -> pd.Series:
    """
    Cross-sectional scaling per ts:
    For each timestamp ts, rescale values so that sum(abs(x_ts)) = a.
    Works on a Series indexed by (ts, symbol) MultiIndex or plain ts index.
    """
    if not isinstance(s, pd.Series):
        s = s.squeeze()

    idx = s.index

    if isinstance(idx, pd.MultiIndex) and "ts" in idx.names:
        grouped = s.groupby(level="ts")
    else:
        grouped = s.groupby(idx)

    def _scale_block(x: pd.Series) -> pd.Series:
        denom = x.abs().sum()
        if denom == 0 or not np.isfinite(denom):
            return 0.0 * x
        return a * x / denom

    return grouped.transform(_scale_block)


def signedpower(s: pd.Series, a: float) -> pd.Series:
    return np.sign(s) * (np.abs(s) ** a)


# simple alias for consistency with specs
def sum_ts(s: pd.Series, d: int) -> pd.Series:
    return ts_sum(s, d)


def signed_power(s: pd.Series, a: float) -> pd.Series:
    return signedpower(s, a)


def sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


def indneutralize(s: pd.Series, group: Optional[str] = None) -> pd.Series:
    """
    Cross-sectional demean per timestamp. Ignores group because we do not have
    sector/industry data in this crypto setting.
    """
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    idx = s.index

    if isinstance(idx, pd.MultiIndex) and "ts" in idx.names:
        ts_level = idx.names.index("ts")
        grouped = s.groupby(level=ts_level)
    else:
        grouped = s.groupby(idx)

    return grouped.transform(lambda x: x - x.mean())


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def compute_returns(close: pd.Series) -> pd.Series:
    return close.groupby(level="symbol").pct_change()


def adv(df_panel: pd.DataFrame, window: int) -> pd.Series:
    dollar_vol = df_panel["close"] * df_panel["volume"]
    return (
        dollar_vol.groupby(level="symbol")
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


# ----------------------------------------------------------------------
# Alpha definitions (v0 subset 1-10, approximated to Appendix A)
# ----------------------------------------------------------------------
def alpha_001(df: pd.DataFrame, ret: pd.Series) -> pd.Series:
    cond = ret < 0
    x = pd.Series(np.where(cond, stddev(ret, 20), df["close"]), index=ret.index)
    sp = signedpower(x, 2.0)
    return cs_rank(ts_argmax(sp, 5)) - 0.5


def alpha_002(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"]
    open_ = df["open"]
    close = df["close"]
    x = cs_rank(delta(np.log(vol.replace(0, np.nan)), 2))
    y = cs_rank((close - open_) / open_)
    return -1 * correlation(x, y, 6)


def alpha_003(df: pd.DataFrame) -> pd.Series:
    return -1 * correlation(cs_rank(df["open"]), cs_rank(df["volume"]), 10)


def alpha_004(df: pd.DataFrame) -> pd.Series:
    return -1 * ts_rank(cs_rank(df["low"]), 9)


def alpha_005(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    vwap = df["vwap"]
    close = df["close"]
    part1 = cs_rank(open_ - ts_sum(vwap, 10) / 10.0)
    part2 = -1 * np.abs(cs_rank(close - vwap))
    return part1 * part2


def alpha_006(df: pd.DataFrame) -> pd.Series:
    return -1 * correlation(df["open"], df["volume"], 10)


def alpha_007(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    ret_7 = delta(c, 7)
    signal = -1 * ts_rank(np.abs(ret_7), 60) * np.sign(ret_7)
    return signal


def alpha_008(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    return -1 * ts_rank(delta(vwap, 3), 100)


def alpha_009(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    open_ = df["open"]
    return -1 * delta(close * 0.6 + open_ * 0.4, 1)


def alpha_010(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    returns = compute_returns(close)
    return -1 * cs_rank(stddev(returns, 20) / stddev(returns, 5))


def alpha_011(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    close = df["close"]
    vol = df["volume"]
    term = (vwap - close)
    res = (cs_rank(ts_max(term, 3)) + cs_rank(ts_min(term, 3))) * cs_rank(delta(vol, 3))
    return res


def alpha_012(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"]
    close = df["close"]
    return sign(delta(vol, 1)) * (-1 * delta(close, 1))


def alpha_013(df: pd.DataFrame) -> pd.Series:
    return -1 * cs_rank(covariance(cs_rank(df["close"]), cs_rank(df["volume"]), 5))


def alpha_014(df: pd.DataFrame) -> pd.Series:
    ret = df["returns"]
    return (-1 * cs_rank(delta(ret, 3))) * correlation(df["open"], df["volume"], 10)


def alpha_015(df: pd.DataFrame) -> pd.Series:
    return -1 * ts_sum(cs_rank(correlation(cs_rank(df["high"]), cs_rank(df["volume"]), 3)), 3)


def alpha_016(df: pd.DataFrame) -> pd.Series:
    return -1 * cs_rank(covariance(cs_rank(df["high"]), cs_rank(df["volume"]), 5))


def alpha_017(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vol = df["volume"]
    adv20 = df["adv20"]
    part1 = -1 * cs_rank(ts_rank(close, 10))
    part2 = cs_rank(delta(delta(close, 1), 1))
    part3 = cs_rank(ts_rank(vol / adv20, 5))
    return part1 * part2 * part3


def alpha_018(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    open_ = df["open"]
    term = stddev(abs(close - open_), 5) + (close - open_)
    return -1 * cs_rank(term + correlation(close, open_, 10))


def alpha_019(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    ret = df["returns"]
    term = (close - delay(close, 7)) + delta(close, 7)
    return (-1 * sign(term)) * (1 + cs_rank(1 + ts_sum(ret, 250)))


def alpha_020(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    return (
        (-1 * cs_rank(open_ - delay(df["high"], 1)))
        * cs_rank(open_ - delay(df["close"], 1))
        * cs_rank(open_ - delay(df["low"], 1))
    )


def alpha_021(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vol = df["volume"]
    adv20 = df["adv20"]
    cond1 = ((ts_sum(close, 8) / 8) + stddev(close, 8)) < (ts_sum(close, 2) / 2)
    cond2 = (ts_sum(close, 2) / 2) < ((ts_sum(close, 8) / 8) - stddev(close, 8))
    cond3 = ((vol / adv20) > 1) | ((vol / adv20) == 1)
    return np.where(cond1, -1.0, np.where(cond2, 1.0, np.where(cond3, 1.0, -1.0)))


def alpha_022(df: pd.DataFrame) -> pd.Series:
    return -1 * (delta(correlation(df["high"], df["volume"], 5), 5) * cs_rank(stddev(df["close"], 20)))


def alpha_023(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    cond = (ts_sum(high, 20) / 20) < high
    return np.where(cond, -1 * delta(high, 2), 0.0)


def alpha_024(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    lhs = delta(ts_sum(close, 100) / 100, 100) / delay(close, 100)
    cond = (lhs < 0.05) | (lhs == 0.05)
    return np.where(cond, -1 * (close - ts_min(close, 100)), -1 * delta(close, 3))


def alpha_025(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    high = df["high"]
    vwap = df["vwap"]
    ret = df["returns"]
    adv20 = df["adv20"]
    return cs_rank(((-1 * ret) * adv20 * vwap * (high - close)))


def alpha_026(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"]
    high = df["high"]
    corr = correlation(ts_rank(vol, 5), ts_rank(high, 5), 5)
    return -1 * ts_max(corr, 3)


def alpha_027(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"]
    vwap = df["vwap"]
    corr_sum = ts_sum(correlation(cs_rank(vol), cs_rank(vwap), 6), 2) / 2.0
    return np.where(cs_rank(corr_sum) > 0.5, -1.0, 1.0)


def alpha_028(df: pd.DataFrame) -> pd.Series:
    adv20 = df["adv20"]
    low = df["low"]
    high = df["high"]
    close = df["close"]
    return scale(correlation(adv20, low, 5) + ((high + low) / 2) - close)


def alpha_029(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    ret = df["returns"]
    # Interpret (close - 1) as close; keep structure per published formula
    inner = ts_min(cs_rank(cs_rank(-1 * cs_rank(delta(close, 5)))), 2)
    sum_inner = ts_sum(inner, 1)
    log_inner = np.log(sum_inner.replace(0, np.nan))
    scaled = scale(log_inner)
    r1 = cs_rank(scaled)
    r2 = cs_rank(r1)
    prod = ts_product(r2, 1)
    part1 = ts_min(prod, 5)
    part2 = ts_rank(delay(-1 * ret, 6), 5)
    return part1 + part2


def alpha_030(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vol = df["volume"]
    term = (
        np.sign(close - delay(close, 1))
        + np.sign(delay(close, 1) - delay(close, 2))
        + np.sign(delay(close, 2) - delay(close, 3))
    )
    return ((1.0 - cs_rank(term)) * ts_sum(vol, 5)) / ts_sum(vol, 20)


def alpha_031(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    term1 = cs_rank(cs_rank(cs_rank(decay_linear(-1 * cs_rank(cs_rank(delta(close, 10))), 10))))
    term2 = cs_rank(-1 * delta(close, 3))
    term3 = np.sign(scale(correlation(df["adv20"], df["low"], 12)))
    return term1 + term2 + term3


def alpha_032(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vwap = df["vwap"]
    term1 = scale((ts_sum(close, 7) / 7) - close)
    term2 = 20 * scale(correlation(vwap, delay(close, 5), 230))
    return term1 + term2


def alpha_033(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    close = df["close"]
    return cs_rank(-1 * (1 - (open_ / close)))


def alpha_034(df: pd.DataFrame) -> pd.Series:
    ret = df["returns"]
    close = df["close"]
    term1 = 1 - cs_rank(stddev(ret, 2) / stddev(ret, 5))
    term2 = 1 - cs_rank(delta(close, 1))
    return cs_rank(term1 + term2)


def alpha_035(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"]
    close = df["close"]
    high = df["high"]
    low = df["low"]
    return (ts_rank(vol, 32) * (1 - ts_rank((close + high) - low, 16))) * (1 - ts_rank(df["returns"], 32))


def alpha_036(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    open_ = df["open"]
    vol = df["volume"]
    vwap = df["vwap"]
    adv20 = df["adv20"]

    term1 = 2.21 * cs_rank(correlation(close - open_, delay(vol, 1), 15))
    term2 = 0.7 * cs_rank(open_ - close)
    term3 = 0.73 * cs_rank(ts_rank(delay(-1 * df["returns"], 6), 5))
    term4 = cs_rank(abs(correlation(vwap, adv20, 6)))
    term5 = 0.6 * cs_rank(((ts_sum(close, 200) / 200) - open_) * (close - open_))
    return term1 + term2 + term3 + term4 + term5


def alpha_037(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    close = df["close"]
    return cs_rank(correlation(delay(open_ - close, 1), close, 200)) + cs_rank(open_ - close)


def alpha_038(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    open_ = df["open"]
    return -1 * cs_rank(ts_rank(close, 10)) * cs_rank(close / open_)


def alpha_039(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vol = df["volume"]
    adv20 = df["adv20"]
    ret = df["returns"]
    term = delta(close, 7) * (1 - cs_rank(decay_linear(vol / adv20, 9)))
    return -1 * cs_rank(term) * (1 + cs_rank(ts_sum(ret, 250)))


def alpha_040(df: pd.DataFrame) -> pd.Series:
    return -1 * cs_rank(stddev(df["high"], 10)) * correlation(df["high"], df["volume"], 10)


def alpha_041(df: pd.DataFrame) -> pd.Series:
    return (df["high"] * df["low"]) ** 0.5 - df["vwap"]


def alpha_042(df: pd.DataFrame) -> pd.Series:
    return cs_rank(df["vwap"] - df["close"]) / cs_rank(df["vwap"] + df["close"])


def alpha_043(df: pd.DataFrame) -> pd.Series:
    return ts_rank(df["volume"] / df["adv20"], 20) * ts_rank(-1 * delta(df["close"], 7), 8)


def alpha_044(df: pd.DataFrame) -> pd.Series:
    return -1 * correlation(df["high"], cs_rank(df["volume"]), 5)


def alpha_045(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vol = df["volume"]
    term1 = cs_rank(ts_sum(delay(close, 5), 20) / 20)
    term2 = correlation(close, vol, 2)
    term3 = cs_rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))
    return -1 * (term1 * term2 * term3)


def alpha_046(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    term = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    cond1 = term > 0.25
    cond2 = term < 0
    res = np.where(cond1, -1.0, np.where(cond2, 1.0, -1.0 * (close - delay(close, 1))))
    return pd.Series(res, index=close.index)


def alpha_047(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    high = df["high"]
    vwap = df["vwap"]
    vol = df["volume"]
    adv20 = df["adv20"]
    term1 = (cs_rank(1 / close) * vol) / adv20
    term2 = (high * cs_rank(high - close)) / (ts_sum(high, 5) / 5)
    return (term1 * term2) - cs_rank(vwap - delay(vwap, 5))


def alpha_048(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    num = correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)
    num = num / close
    den = ts_sum((delta(close, 1) / delay(close, 1)) ** 2, 250)
    return indneutralize(num, "subindustry") / den


def alpha_049(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    term = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    cond = term < -0.1
    res = np.where(cond, 1.0, -1.0 * (close - delay(close, 1)))
    return pd.Series(res, index=close.index)


def alpha_050(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"]
    vwap = df["vwap"]
    return -1 * ts_max(cs_rank(correlation(cs_rank(vol), cs_rank(vwap), 5)), 5)


def alpha_051(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    term = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    cond = term < -0.05
    res = np.where(cond, 1.0, -1.0 * (close - delay(close, 1)))
    return pd.Series(res, index=close.index)


def alpha_052(df: pd.DataFrame) -> pd.Series:
    low = df["low"]
    ret = df["returns"]
    vol = df["volume"]
    term1 = (-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)
    term2 = cs_rank((ts_sum(ret, 240) - ts_sum(ret, 20)) / 220)
    term3 = ts_rank(vol, 5)
    return term1 * term2 * term3


def alpha_053(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    low = df["low"]
    high = df["high"]
    term = ((close - low) - (high - close)) / (close - low)
    return -1 * delta(term, 9)


def alpha_054(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    low = df["low"]
    high = df["high"]
    open_ = df["open"]
    return (-1 * (low - close) * (open_ ** 5)) / ((low - high) * (close ** 5))


def alpha_055(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    low = df["low"]
    high = df["high"]
    vol = df["volume"]
    term = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))
    return -1 * correlation(cs_rank(term), cs_rank(vol), 6)


def alpha_056(df: pd.DataFrame) -> pd.Series:
    ret = df["returns"]
    cap = df["cap"]
    term1 = cs_rank(ts_sum(ret, 10) / ts_sum(ts_sum(ret, 2), 3))
    term2 = cs_rank(ret * cap)
    return -1 * term1 * term2


def alpha_057(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vwap = df["vwap"]
    denom = decay_linear(cs_rank(ts_argmax(close, 30)), 2)
    return -1 * (close - vwap) / denom


def alpha_058(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    vol = df["volume"]
    corr = correlation(indneutralize(vwap, "sector"), vol, 3.92795)
    dl = decay_linear(corr, 7.89291)
    return -1 * ts_rank(dl, 5.50322)


def alpha_059(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    vol = df["volume"]
    comb = (vwap * 0.728317) + (vwap * (1 - 0.728317))
    corr = correlation(indneutralize(comb, "industry"), vol, 4.25197)
    dl = decay_linear(corr, 16.2289)
    return -1 * ts_rank(dl, 8.19648)


def alpha_060(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    term1 = ((close - low) - (high - close)) / (high - low)
    part1 = 2 * scale(cs_rank(term1 * vol))
    part2 = scale(cs_rank(ts_argmax(close, 10)))
    return -1 * (part1 - part2)


def alpha_061(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    adv180 = df["adv180"]
    term1 = cs_rank(vwap - ts_min(vwap, 16.1219))
    term2 = cs_rank(correlation(vwap, adv180, 17.9282))
    return (term1 < term2).astype(float)


def alpha_062(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    adv20 = df["adv20"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    lhs = cs_rank(correlation(vwap, ts_sum(adv20, 22.4101), 9.91009))
    rhs = cs_rank((cs_rank(open_) + cs_rank(open_)) < (cs_rank((high + low) / 2) + cs_rank(high)))
    return ((lhs < rhs) * -1).astype(float)


def alpha_063(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vwap = df["vwap"]
    open_ = df["open"]
    term1 = cs_rank(decay_linear(delta(indneutralize(close, "industry"), 2.25164), 8.22237))
    term2 = cs_rank(
        decay_linear(
            correlation((vwap * 0.318108) + (open_ * (1 - 0.318108)), ts_sum(df["adv180"], 37.2467), 13.557),
            12.2883,
        )
    )
    return (term1 - term2) * -1


def alpha_064(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    low = df["low"]
    high = df["high"]
    vwap = df["vwap"]
    term1 = cs_rank(
        correlation(
            ts_sum((open_ * 0.178404) + (low * (1 - 0.178404)), 12.7054),
            ts_sum(df["adv120"], 12.7054),
            16.6208,
        )
    )
    term2 = cs_rank(
        delta((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404)), 3.69741)
    )
    return ((term1 < term2) * -1).astype(float)


def alpha_065(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    vwap = df["vwap"]
    term1 = cs_rank(
        correlation(
            (open_ * 0.00817205) + (vwap * (1 - 0.00817205)),
            ts_sum(df["adv60"], 8.6911),
            6.40374,
        )
    )
    term2 = cs_rank(open_ - ts_min(open_, 13.635))
    return ((term1 < term2) * -1).astype(float)


def alpha_066(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    low = df["low"]
    open_ = df["open"]
    high = df["high"]
    term1 = cs_rank(decay_linear(delta(vwap, 3.51013), 7.23052))
    num = low - vwap
    denom = open_ - ((high + low) / 2)
    ratio = num / denom.replace(0.0, np.nan)
    term2 = ts_rank(decay_linear(ratio, 11.4157), 6.72611)
    return (term1 + term2) * -1


def alpha_067(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    vwap = df["vwap"]
    adv20 = df["adv20"]
    term1 = cs_rank(high - ts_min(high, 2.14593))
    term2 = cs_rank(correlation(indneutralize(vwap, "sector"), indneutralize(adv20, "subindustry"), 6.02936))
    return -1 * (term1 ** term2)


def alpha_068(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    adv15 = df["adv15"]
    close = df["close"]
    low = df["low"]
    term1 = ts_rank(correlation(cs_rank(high), cs_rank(adv15), 8.91644), 13.9333)
    term2 = cs_rank(delta((close * 0.518371) + (low * (1 - 0.518371)), 1.06157))
    return ((term1 < term2) * -1).astype(float)


def alpha_069(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    close = df["close"]
    adv20 = df["adv20"]
    term1 = cs_rank(ts_max(delta(indneutralize(vwap, "industry"), 2.72412), 4.79344))
    term2 = ts_rank(
        correlation((close * 0.490655) + (vwap * (1 - 0.490655)), adv20, 4.92416),
        9.0615,
    )
    return -1 * (term1 ** term2)


def alpha_070(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    close = df["close"]
    adv50 = df["adv50"]
    term1 = cs_rank(delta(vwap, 1.29456))
    term2 = ts_rank(correlation(indneutralize(close, "industry"), adv50, 17.8256), 17.9171)
    return -1 * (term1 ** term2)


def alpha_071(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    adv180 = df["adv180"]
    low = df["low"]
    open_ = df["open"]
    vwap = df["vwap"]
    term1 = ts_rank(
        decay_linear(correlation(ts_rank(close, 3.43976), ts_rank(adv180, 12.0647), 18.0175), 4.20501),
        15.6948,
    )
    term2 = ts_rank(
        decay_linear((cs_rank((low + open_) - (vwap + vwap)) ** 2), 16.4662),
        4.4388,
    )
    return np.maximum(term1, term2)


def alpha_072(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    adv40 = df["adv40"]
    vwap = df["vwap"]
    vol = df["volume"]
    num = cs_rank(decay_linear(correlation((high + low) / 2, adv40, 8.93345), 10.1519))
    den = cs_rank(
        decay_linear(
            correlation(ts_rank(vwap, 3.72469), ts_rank(vol, 18.5188), 6.86671),
            2.95011,
        )
    )
    return num / den


def alpha_073(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    open_ = df["open"]
    low = df["low"]
    term1 = cs_rank(decay_linear(delta(vwap, 4.72775), 2.91864))
    denom = (open_ * 0.147155) + (low * (1 - 0.147155))
    ratio = delta(denom, 2.03608) / denom.replace(0.0, np.nan)
    term2 = ts_rank(decay_linear(ratio * -1, 3.33829), 16.7411)
    return -1 * np.maximum(term1, term2)


def alpha_074(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    adv30 = df["adv30"]
    high = df["high"]
    vwap = df["vwap"]
    vol = df["volume"]
    term1 = cs_rank(correlation(close, ts_sum(adv30, 37.4843), 15.1365))
    term2 = cs_rank(correlation(cs_rank((high * 0.0261661) + (vwap * (1 - 0.0261661))), cs_rank(vol), 11.4791))
    return ((term1 < term2) * -1).astype(float)


def alpha_075(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    vol = df["volume"]
    low = df["low"]
    adv50 = df["adv50"]
    term1 = cs_rank(correlation(vwap, vol, 4.24304))
    term2 = cs_rank(correlation(cs_rank(low), cs_rank(adv50), 12.4413))
    return (term1 < term2).astype(float)


def alpha_076(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    low = df["low"]
    adv81 = df["adv81"]
    term1 = cs_rank(decay_linear(delta(vwap, 1.24383), 11.8259))
    term2 = ts_rank(
        decay_linear(
            ts_rank(correlation(indneutralize(low, "sector"), adv81, 8.14941), 19.569),
            17.1543,
        ),
        19.383,
    )
    return -1 * np.maximum(term1, term2)


def alpha_077(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    vwap = df["vwap"]
    adv40 = df["adv40"]
    term1 = cs_rank(decay_linear((((high + low) / 2) + high) - (vwap + high), 20.0451))
    term2 = cs_rank(decay_linear(correlation((high + low) / 2, adv40, 3.1614), 5.64125))
    return np.minimum(term1, term2)


def alpha_078(df: pd.DataFrame) -> pd.Series:
    low = df["low"]
    vwap = df["vwap"]
    adv40 = df["adv40"]
    vol = df["volume"]
    term1 = cs_rank(
        correlation(
            ts_sum((low * 0.352233) + (vwap * (1 - 0.352233)), 19.7428),
            ts_sum(adv40, 19.7428),
            6.83313,
        )
    )
    term2 = cs_rank(correlation(cs_rank(vwap), cs_rank(vol), 5.77492))
    return term1 ** term2


def alpha_079(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    open_ = df["open"]
    vwap = df["vwap"]
    adv150 = df["adv150"]
    term1 = cs_rank(delta(indneutralize((close * 0.60733) + (open_ * (1 - 0.60733)), "sector"), 1.23438))
    term2 = cs_rank(correlation(ts_rank(vwap, 3.60973), ts_rank(adv150, 9.18637), 14.6644))
    return (term1 < term2).astype(float)


def alpha_080(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    high = df["high"]
    adv10 = df["adv10"]
    term1 = cs_rank(np.sign(delta(indneutralize((open_ * 0.868128) + (high * (1 - 0.868128)), "industry"), 4.04545)))
    term2 = ts_rank(correlation(high, adv10, 5.11456), 5.53756)
    return -1 * (term1 ** term2)


def alpha_081(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    adv10 = df["adv10"]
    vol = df["volume"]
    corr = correlation(vwap, ts_sum(adv10, 49.6054), 8.47743)
    term1 = cs_rank(np.log(ts_product(cs_rank(cs_rank(corr) ** 4), 14.9655)))
    term2 = cs_rank(correlation(cs_rank(vwap), cs_rank(vol), 5.07914))
    return ((term1 < term2) * -1).astype(float)


def alpha_082(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    vol = df["volume"]
    term1 = cs_rank(decay_linear(delta(open_, 1.46063), 14.8717))
    term2 = ts_rank(
        decay_linear(
            correlation(indneutralize(vol, "sector"), open_ * 0.634196 + open_ * (1 - 0.634196), 17.4842),
            6.92131,
        ),
        13.4283,
    )
    return -1 * np.minimum(term1, term2)


def alpha_083(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    vol = df["volume"]
    vwap = df["vwap"]
    num = cs_rank(delay((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(vol))
    den = ((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close)
    return num / den


def alpha_084(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    close = df["close"]
    term1 = ts_rank(vwap - ts_max(vwap, 15.3217), 20.7127)
    term2 = delta(close, 4.96796)
    return signed_power(term1, term2)


def alpha_085(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#85:
    (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331)) ^
     rank(correlation(Ts_Rank(((high + low) / 2), 3.70596),
                      Ts_Rank(volume, 10.1595), 7.11408)))
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    adv30 = df["adv30"]

    part1 = correlation(
        (high * 0.876703) + (close * (1.0 - 0.876703)),
        adv30,
        _win(9.61331),
    )

    part2 = correlation(
        ts_rank((high + low) / 2.0, 3.70596),
        ts_rank(volume, 10.1595),
        _win(7.11408),
    )

    return cs_rank(part1) ** cs_rank(part2)


def alpha_086(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    open_ = df["open"]
    vwap = df["vwap"]
    adv20 = df["adv20"]
    term1 = ts_rank(correlation(close, ts_sum(adv20, 14.7444), 6.00049), 20.4195)
    term2 = cs_rank((open_ + close) - (vwap + open_))
    return ((term1 < term2) * -1).astype(float)


def alpha_087(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vwap = df["vwap"]
    adv81 = df["adv81"]
    term1 = cs_rank(decay_linear(delta((close * 0.369701) + (vwap * (1 - 0.369701)), 1.91233), 2.65461))
    term2 = ts_rank(
        decay_linear(abs(correlation(indneutralize(adv81, "industry"), close, 13.4132)), 4.89768),
        14.4535,
    )
    return -1 * np.maximum(term1, term2)


def alpha_088(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    low = df["low"]
    high = df["high"]
    close = df["close"]
    adv60 = df["adv60"]
    term1 = cs_rank(decay_linear((cs_rank(open_) + cs_rank(low)) - (cs_rank(high) + cs_rank(close)), 8.06882))
    term2 = ts_rank(
        decay_linear(
            correlation(ts_rank(close, 8.44728), ts_rank(adv60, 20.6966), 8.01266),
            6.65053,
        ),
        2.61957,
    )
    return np.minimum(term1, term2)


def alpha_089(df: pd.DataFrame) -> pd.Series:
    low = df["low"]
    vwap = df["vwap"]
    adv10 = df["adv10"]
    term1 = ts_rank(
        decay_linear(correlation(low, adv10, 6.94279), 5.51607),
        3.79744,
    )
    term2 = ts_rank(decay_linear(delta(indneutralize(vwap, "industry"), 3.48158), 10.1466), 15.3012)
    return term1 - term2


def alpha_090(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    adv40 = df["adv40"]
    low = df["low"]
    term1 = cs_rank(close - ts_max(close, 4.66719))
    term2 = ts_rank(correlation(indneutralize(adv40, "subindustry"), low, 5.38375), 3.21856)
    return -1 * (term1 ** term2)


def alpha_091(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    vol = df["volume"]
    vwap = df["vwap"]
    adv30 = df["adv30"]
    term1 = ts_rank(
        decay_linear(
            decay_linear(correlation(indneutralize(close, "industry"), vol, 9.74928), 16.398),
            3.83219,
        ),
        4.8667,
    )
    term2 = cs_rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))
    return (term1 - term2) * -1


def alpha_092(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]
    adv30 = df["adv30"]
    term1 = ts_rank(
        decay_linear((((high + low) / 2) + close) < (low + open_), 14.7221),
        18.8683,
    )
    term2 = ts_rank(
        decay_linear(correlation(cs_rank(low), cs_rank(adv30), 7.58555), 6.94024),
        6.80584,
    )
    return np.minimum(term1, term2)


def alpha_093(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    adv81 = df["adv81"]
    close = df["close"]
    term1 = ts_rank(
        decay_linear(correlation(indneutralize(vwap, "industry"), adv81, 17.4193), 19.848),
        7.54455,
    )
    term2 = cs_rank(
        decay_linear(delta((close * 0.524434) + (vwap * (1 - 0.524434)), 2.77377), 16.2664)
    )
    return term1 / term2


def alpha_094(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    adv60 = df["adv60"]
    term1 = cs_rank(vwap - ts_min(vwap, 11.5783))
    term2 = ts_rank(
        correlation(ts_rank(vwap, 19.6462), ts_rank(adv60, 4.02992), 18.0926),
        2.70756,
    )
    return -1 * (term1 ** term2)


def alpha_095(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    adv40 = df["adv40"]
    term1 = cs_rank(open_ - ts_min(open_, 12.4105))
    term2 = ts_rank(
        cs_rank(correlation(ts_sum((high + low) / 2, 19.1351), ts_sum(adv40, 19.1351), 12.8742)) ** 5,
        11.7584,
    )
    return (term1 < term2).astype(float)


def alpha_096(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    vol = df["volume"]
    close = df["close"]
    adv60 = df["adv60"]
    term1 = ts_rank(
        decay_linear(correlation(cs_rank(vwap), cs_rank(vol), 3.83878), 4.16783),
        8.38151,
    )
    term2 = ts_rank(
        decay_linear(
            ts_argmax(correlation(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556),
            14.0365,
        ),
        13.4143,
    )
    return -1 * np.maximum(term1, term2)


def alpha_097(df: pd.DataFrame) -> pd.Series:
    low = df["low"]
    vwap = df["vwap"]
    adv60 = df["adv60"]
    term1 = cs_rank(
        decay_linear(delta(indneutralize((low * 0.721001) + (vwap * (1 - 0.721001)), "industry"), 3.3705), 20.4523)
    )
    term2 = ts_rank(
        decay_linear(
            ts_rank(correlation(ts_rank(low, 7.87871), ts_rank(adv60, 17.255), 4.97547), 18.5925),
            15.7152,
        ),
        6.71659,
    )
    return (term1 - term2) * -1


def alpha_098(df: pd.DataFrame) -> pd.Series:
    vwap = df["vwap"]
    adv5 = df["adv5"]
    open_ = df["open"]
    adv15 = df["adv15"]
    term1 = cs_rank(decay_linear(correlation(vwap, ts_sum(adv5, 26.4719), 4.58418), 7.18088))
    term2 = cs_rank(
        decay_linear(
            ts_rank(ts_argmin(correlation(cs_rank(open_), cs_rank(adv15), 20.8187), 8.62571), 6.95668),
            8.07206,
        )
    )
    return term1 - term2


def alpha_099(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    adv60 = df["adv60"]
    vol = df["volume"]
    term1 = cs_rank(
        correlation(
            ts_sum((high + low) / 2, 19.8975),
            ts_sum(adv60, 19.8975),
            8.8136,
        )
    )
    term2 = cs_rank(correlation(low, vol, 6.28259))
    return ((term1 < term2) * -1).astype(float)


def alpha_100(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    low = df["low"]
    high = df["high"]
    vol = df["volume"]
    adv20 = df["adv20"]
    term1 = ((close - low) - (high - close)) / (high - low)
    r1 = cs_rank(term1 * vol)
    inner = indneutralize(indneutralize(r1, "subindustry"), "subindustry")
    termA = 1.5 * scale(inner)
    termB = scale(
        indneutralize(
            correlation(close, cs_rank(adv20), 5) - cs_rank(ts_argmin(close, 30)),
            "subindustry",
        )
    )
    return -1 * ((termA - termB) * (vol / adv20))


def alpha_101(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    return (close - open_) / ((high - low) + 0.001)


# === Custom N-series alphas: alpha_201..alpha_220 ===
# These are research-only extensions; they rely only on existing helpers and panel fields.


def alpha_201(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_201: 5-day close momentum normalized by 20-day return volatility.
    """
    close = panel["close"]
    ret1 = close / delay(close, 1) - 1.0
    mom5 = ts_sum(ret1, 5)
    vol20 = stddev(ret1, 20)
    signal = mom5 / (1e-6 + vol20)
    return cs_rank(signal)


def alpha_202(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_202: Trend-following with shallow pullback near 20-day high.
    """
    close = panel["close"]
    high = panel["high"]
    hh20 = ts_max(high, 20)
    trend20 = close / delay(close, 20) - 1.0
    pullback = hh20 - close
    signal = cs_rank(trend20) * cs_rank(-pullback)
    return signal


def alpha_203(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_203: Volume-confirmed 1-day breakout.
    """
    close = panel["close"]
    volume = panel["volume"]
    ret1 = close / delay(close, 1) - 1.0
    adv20 = ts_sum(volume, 20) / 20.0
    vol_ratio = volume / (1e-6 + adv20)
    signal = cs_rank(ret1) * cs_rank(vol_ratio)
    return signal


def alpha_204(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_204: Trend-following VWAP reversion (buy strong names on VWAP dips).
    """
    close = panel["close"]
    vwap = panel.get("vwap", close)
    spread = (close - vwap) / (1e-6 + stddev(close, 10))
    trend20 = close / delay(close, 20) - 1.0
    signal = cs_rank(trend20) * cs_rank(-spread)
    return signal


def alpha_205(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_205: Range-compression breakout (strong 10-day trend with tight range).
    """
    close = panel["close"]
    high = panel["high"]
    low = panel["low"]
    range10 = ts_max(high, 10) - ts_min(low, 10)
    mom10 = close / delay(close, 10) - 1.0
    signal = cs_rank(mom10) * cs_rank(-range10)
    return signal


def alpha_206(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_206: Path efficiency (net 10-day move vs sum of absolute 1-day moves).
    """
    close = panel["close"]
    ret1 = close / delay(close, 1) - 1.0
    abs_path10 = ts_sum(abs(ret1), 10)
    net10 = close / delay(close, 10) - 1.0
    eff = net10 / (1e-6 + abs_path10)
    return cs_rank(eff)


def alpha_207(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_207: Gap reversion modulated by short-term volatility.
    """
    open_ = panel["open"]
    close = panel["close"]
    prev_close = delay(close, 1)
    gap = (open_ - prev_close) / (1e-6 + prev_close)
    ret1 = close / delay(close, 1) - 1.0
    vol5 = stddev(ret1, 5)
    signal = -cs_rank(gap) * cs_rank(vol5)
    return signal


def alpha_208(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_208: Intraday skew; fade closes skewed away from the mid-range.
    """
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    rng = high - low
    body = close - (high + low) / 2.0
    signal = -cs_rank(body / (1e-6 + rng))
    return signal


def alpha_209(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_209: Volatility-of-volume relative to average volume.
    """
    volume = panel["volume"]
    vol_vol20 = stddev(volume, 20)
    adv20 = ts_sum(volume, 20) / 20.0
    signal = cs_rank(vol_vol20 / (1e-6 + adv20))
    return signal


def alpha_210(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_210: Balance of winners vs losers over 10 days.
    """
    close = panel["close"]
    ret1 = close / delay(close, 1) - 1.0
    pos = ts_sum((ret1 > 0).astype(float), 10)
    neg = ts_sum((ret1 < 0).astype(float), 10)
    raw = (pos - neg) / (1.0 + pos + neg)
    return cs_rank(raw)


def alpha_211(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_211: Volatility breakout (short-term vol vs long-term vol in an uptrend).
    """
    close = panel["close"]
    ret1 = close / delay(close, 1) - 1.0
    vol10 = stddev(ret1, 10)
    vol50 = stddev(ret1, 50)
    trend20 = close / delay(close, 20) - 1.0
    signal = cs_rank(vol10 / (1e-6 + vol50)) * cs_rank(trend20)
    return signal


def alpha_212(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_212: Short-term vs long-term momentum divergence.
    """
    close = panel["close"]
    ret1 = close / delay(close, 1) - 1.0
    mom3 = ts_sum(ret1, 3)
    mom20 = ts_sum(ret1, 20)
    signal = cs_rank(mom3 - mom20)
    return signal


def alpha_213(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_213: Volume-weighted recent momentum (5-day decayed).
    """
    close = panel["close"]
    volume = panel["volume"]
    ret1 = close / delay(close, 1) - 1.0
    vw_ret = decay_linear(ret1 * volume, 5)
    signal = cs_rank(vw_ret)
    return signal


def alpha_214(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_214: Down-volume exhaustion vs up-volume (contrarian).
    """
    close = panel["close"]
    volume = panel["volume"]
    ret1 = close / delay(close, 1) - 1.0
    down_vol = decay_linear((ret1 < 0).astype(float) * volume, 5)
    up_vol = decay_linear((ret1 > 0).astype(float) * volume, 5)
    signal = cs_rank(-down_vol / (1e-6 + up_vol))
    return signal


def alpha_215(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_215: Bollinger-band style z-score of close.
    """
    close = panel["close"]
    ma20 = ts_sum(close, 20) / 20.0
    vol20 = stddev(close, 20)
    z = (close - ma20) / (1e-6 + vol20)
    return cs_rank(z)


def alpha_216(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_216: Mean reversion of 1-day return z-score vs 20-day history.
    """
    close = panel["close"]
    ret1 = close / delay(close, 1) - 1.0
    ma20_ret = ts_sum(ret1, 20) / 20.0
    vol20_ret = stddev(ret1, 20)
    z = (ret1 - ma20_ret) / (1e-6 + vol20_ret)
    return -cs_rank(z)


def alpha_217(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_217: Momentum acceleration (5-day minus 10-day).
    """
    close = panel["close"]
    mom5 = close / delay(close, 5) - 1.0
    mom10 = close / delay(close, 10) - 1.0
    signal = cs_rank(mom5 - mom10)
    return signal


def alpha_218(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_218: Close location in range times 5-day momentum.
    """
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    rng = high - low
    loc = (close - low) / (1e-6 + rng)
    mom5 = close / delay(close, 5) - 1.0
    signal = cs_rank(loc * mom5)
    return signal


def alpha_219(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_219: Volatility per unit volume (illiquidity tilt).
    """
    close = panel["close"]
    volume = panel["volume"]
    ret1 = close / delay(close, 1) - 1.0
    vol20 = stddev(ret1, 20)
    adv20 = ts_sum(volume, 20) / 20.0
    signal = cs_rank(vol20 / (1e-6 + adv20))
    return -signal


def alpha_220(panel: pd.DataFrame) -> pd.Series:
    """
    alpha_220: Correlation of returns with volume over 20 days.
    """
    close = panel["close"]
    volume = panel["volume"]
    ret1 = close / delay(close, 1) - 1.0
    corr20 = correlation(ret1, volume, 20)
    return cs_rank(corr20)


ALPHA_FUNCS_CUSTOM = {
    "alpha_201": alpha_201,
    "alpha_202": alpha_202,
    "alpha_203": alpha_203,
    "alpha_204": alpha_204,
    "alpha_205": alpha_205,
    "alpha_206": alpha_206,
    "alpha_207": alpha_207,
    "alpha_208": alpha_208,
    "alpha_209": alpha_209,
    "alpha_210": alpha_210,
    "alpha_211": alpha_211,
    "alpha_212": alpha_212,
    "alpha_213": alpha_213,
    "alpha_214": alpha_214,
    "alpha_215": alpha_215,
    "alpha_216": alpha_216,
    "alpha_217": alpha_217,
    "alpha_218": alpha_218,
    "alpha_219": alpha_219,
    "alpha_220": alpha_220,
}


ALPHA_FUNCS = {
    "alpha_001": alpha_001,
    "alpha_002": alpha_002,
    "alpha_003": alpha_003,
    "alpha_004": alpha_004,
    "alpha_005": alpha_005,
    "alpha_006": alpha_006,
    "alpha_007": alpha_007,
    "alpha_008": alpha_008,
    "alpha_009": alpha_009,
    "alpha_010": alpha_010,
    "alpha_011": alpha_011,
    "alpha_012": alpha_012,
    "alpha_013": alpha_013,
    "alpha_014": alpha_014,
    "alpha_015": alpha_015,
    "alpha_016": alpha_016,
    "alpha_017": alpha_017,
    "alpha_018": alpha_018,
    "alpha_019": alpha_019,
    "alpha_020": alpha_020,
    "alpha_021": alpha_021,
    "alpha_022": alpha_022,
    "alpha_023": alpha_023,
    "alpha_024": alpha_024,
    "alpha_025": alpha_025,
    "alpha_026": alpha_026,
    "alpha_027": alpha_027,
    "alpha_028": alpha_028,
    "alpha_029": alpha_029,
    "alpha_030": alpha_030,
    "alpha_031": alpha_031,
    "alpha_032": alpha_032,
    "alpha_033": alpha_033,
    "alpha_034": alpha_034,
    "alpha_035": alpha_035,
    "alpha_036": alpha_036,
    "alpha_037": alpha_037,
    "alpha_038": alpha_038,
    "alpha_039": alpha_039,
    "alpha_040": alpha_040,
    "alpha_041": alpha_041,
    "alpha_042": alpha_042,
    "alpha_043": alpha_043,
    "alpha_044": alpha_044,
    "alpha_045": alpha_045,
    "alpha_046": alpha_046,
    "alpha_047": alpha_047,
    "alpha_048": alpha_048,
    "alpha_049": alpha_049,
    "alpha_050": alpha_050,
    "alpha_051": alpha_051,
    "alpha_052": alpha_052,
    "alpha_053": alpha_053,
    "alpha_054": alpha_054,
    "alpha_055": alpha_055,
    "alpha_056": alpha_056,
    "alpha_057": alpha_057,
    "alpha_058": alpha_058,
    "alpha_059": alpha_059,
    "alpha_060": alpha_060,
    "alpha_061": alpha_061,
    "alpha_062": alpha_062,
    "alpha_063": alpha_063,
    "alpha_064": alpha_064,
    "alpha_065": alpha_065,
    "alpha_066": alpha_066,
    "alpha_067": alpha_067,
    "alpha_068": alpha_068,
    "alpha_069": alpha_069,
    "alpha_070": alpha_070,
    "alpha_071": alpha_071,
    "alpha_072": alpha_072,
    "alpha_073": alpha_073,
    "alpha_074": alpha_074,
    "alpha_075": alpha_075,
    "alpha_076": alpha_076,
    "alpha_077": alpha_077,
    "alpha_078": alpha_078,
    "alpha_079": alpha_079,
    "alpha_080": alpha_080,
    "alpha_081": alpha_081,
    "alpha_082": alpha_082,
    "alpha_083": alpha_083,
    "alpha_084": alpha_084,
    "alpha_085": alpha_085,
    "alpha_086": alpha_086,
    "alpha_087": alpha_087,
    "alpha_088": alpha_088,
    "alpha_089": alpha_089,
    "alpha_090": alpha_090,
    "alpha_091": alpha_091,
    "alpha_092": alpha_092,
    "alpha_093": alpha_093,
    "alpha_094": alpha_094,
    "alpha_095": alpha_095,
    "alpha_096": alpha_096,
    "alpha_097": alpha_097,
    "alpha_098": alpha_098,
    "alpha_099": alpha_099,
    "alpha_100": alpha_100,
    "alpha_101": alpha_101,
}


# ----------------------------------------------------------------------
# Custom C-alphas (C1-C20)
# ----------------------------------------------------------------------
def alpha_c01(panel: pd.DataFrame) -> pd.Series:
    df = panel
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    close = df["close"]
    vol = df["volume"]
    adv20 = df["adv20"]

    body_top = open_.where(open_ > close, close)
    denom = (high - low).replace(0.0, np.nan)
    wick_ratio = (high - body_top) / denom
    x1 = cs_rank(wick_ratio)
    x2 = cs_rank(vol / adv20.replace(0.0, np.nan))
    alpha = -1.0 * x1 * x2
    alpha.name = "alpha_c01"
    return alpha


def alpha_c02(panel: pd.DataFrame) -> pd.Series:
    df = panel
    high_rank = cs_rank(df["high"])
    vol_rank = cs_rank(df["volume"])
    alpha = -1.0 * correlation(high_rank, vol_rank, 5)
    alpha.name = "alpha_c02"
    return alpha


def alpha_c03(panel: pd.DataFrame) -> pd.Series:
    df = panel
    close = df["close"]
    vwap = df["vwap"]
    z = (close - vwap) / stddev(close, 5)
    alpha = -1.0 * cs_rank(z)
    alpha.name = "alpha_c03"
    return alpha


def alpha_c04(panel: pd.DataFrame) -> pd.Series:
    df = panel
    ret = df["returns"]
    close = df["close"]
    num = sum_ts(ret.abs(), 5)
    denom = (close - delay(close, 5)).replace(0.0, np.nan)
    eff = num / denom
    alpha = -1.0 * ts_rank(eff, 10)
    alpha.name = "alpha_c04"
    return alpha


def alpha_c05(panel: pd.DataFrame) -> pd.Series:
    df = panel
    close = df["close"]
    vol = df["volume"]
    adv20 = df["adv20"]
    dip = -1.0 * delta(close, 1)
    vol_ratio = vol / adv20.replace(0.0, np.nan)
    vol_rank = cs_rank(vol_ratio)
    alpha = dip * (1.0 - vol_rank)
    alpha.name = "alpha_c05"
    return alpha


def alpha_c06(panel: pd.DataFrame) -> pd.Series:
    df = panel
    open_ = df["open"]
    close = df["close"]
    gap = open_ - delay(close, 1)
    alpha = -1.0 * gap
    alpha.name = "alpha_c06"
    return alpha


def alpha_c07(panel: pd.DataFrame) -> pd.Series:
    df = panel
    close = df["close"]
    ret = df["returns"]
    vol_5 = stddev(close, 5)
    alpha = -1.0 * cs_rank(vol_5) * cs_rank(ret)
    alpha.name = "alpha_c07"
    return alpha


def alpha_c08(panel: pd.DataFrame) -> pd.Series:
    df = panel
    c_rank = ts_rank(df["close"], 14)
    v_rank = ts_rank(df["volume"], 14)
    alpha = -1.0 * (c_rank - v_rank)
    alpha.name = "alpha_c08"
    return alpha


def alpha_c09(panel: pd.DataFrame) -> pd.Series:
    df = panel
    alpha = cs_rank(delta(df["close"], 1)) * cs_rank(delta(df["volume"], 1))
    alpha.name = "alpha_c09"
    return alpha


def alpha_c10(panel: pd.DataFrame) -> pd.Series:
    df = panel
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    denom = (high - low + 0.001)
    eff = (close - open_) / denom
    alpha = cs_rank(eff)
    alpha.name = "alpha_c10"
    return alpha


def alpha_c11(panel: pd.DataFrame) -> pd.Series:
    df = panel
    ret = df["returns"]
    run_sum = sum_ts(np.sign(ret), 10)
    alpha = ts_rank(run_sum, 10)
    alpha.name = "alpha_c11"
    return alpha


def alpha_c12(panel: pd.DataFrame) -> pd.Series:
    df = panel
    vol10 = stddev(df["close"], 10)
    alpha = cs_rank(delta(vol10, 1))
    alpha.name = "alpha_c12"
    return alpha


def alpha_c13(panel: pd.DataFrame) -> pd.Series:
    df = panel
    ret = df["returns"]
    ret_rank = cs_rank(ret)
    alpha = cs_rank(ret - ret_rank)
    alpha.name = "alpha_c13"
    return alpha


def alpha_c14(panel: pd.DataFrame) -> pd.Series:
    df = panel
    close = df["close"]
    hi10 = ts_max(close, 10)
    alpha = (close == hi10).astype(float)
    alpha.name = "alpha_c14"
    return alpha


def alpha_c15(panel: pd.DataFrame) -> pd.Series:
    df = panel
    close = df["close"]
    vwap = df["vwap"]
    s = sum_ts(np.sign(close - vwap), 5)
    alpha = cs_rank(s)
    alpha.name = "alpha_c15"
    return alpha


def alpha_c16(panel: pd.DataFrame) -> pd.Series:
    df = panel
    ret = df["returns"]
    vol5 = stddev(ret, 5)
    vol20 = stddev(ret, 20).replace(0.0, np.nan)
    alpha = vol5 / vol20
    alpha.name = "alpha_c16"
    return alpha


def alpha_c17(panel: pd.DataFrame) -> pd.Series:
    df = panel
    high = df["high"]
    low = df["low"]
    close = df["close"]
    denom = (high - low).replace(0.0, np.nan)
    alpha = (high + low - 2.0 * close) / denom
    alpha.name = "alpha_c17"
    return alpha


def alpha_c18(panel: pd.DataFrame) -> pd.Series:
    df = panel
    ret = df["returns"]
    vol = df["volume"]
    close = df["close"]
    denom = (vol * close).replace(0.0, np.nan)
    alpha = ret.abs() / denom
    alpha.name = "alpha_c18"
    return alpha


def alpha_c19(panel: pd.DataFrame) -> pd.Series:
    df = panel
    ret = df["returns"]
    idx = ret.index
    mkt = ret.groupby(level="ts").mean()
    mkt_broadcast = mkt.reindex(idx.get_level_values("ts")).set_axis(idx)
    alpha = correlation(ret, mkt_broadcast, 10)
    alpha.name = "alpha_c19"
    return alpha


def alpha_c20(panel: pd.DataFrame) -> pd.Series:
    df = panel
    rng = df["high"] - df["low"]
    alpha = rng / delay(rng, 1).replace(0.0, np.nan)
    alpha.name = "alpha_c20"
    return alpha



def compute_all_alphas_v0(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    df_panel: columns ts, symbol, open, high, low, close, volume, vwap
    returns DataFrame indexed by [symbol, ts] with alpha columns.
    """
    panel = _to_panel(df_panel)

    # Pre-compute shared series
    ret = compute_returns(panel["close"])
    panel["returns"] = ret

    adv_windows = [5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180]
    for w in adv_windows:
        panel[f"adv{w}"] = adv(panel, w)

    # proxy for cap
    panel["cap"] = panel.get("adv20", adv(panel, 20))

    alpha_outputs: Dict[str, pd.Series] = {}
    for name, func in ALPHA_FUNCS.items():
        if name == "alpha_001":
            alpha_outputs[name] = func(panel, ret)
        else:
            alpha_outputs[name] = func(panel)

    # Custom N-series alphas
    for name, func in ALPHA_FUNCS_CUSTOM.items():
        alpha_outputs[name] = func(panel)

    # Custom C alphas
    alpha_outputs["alpha_c01"] = alpha_c01(panel)
    alpha_outputs["alpha_c02"] = alpha_c02(panel)
    alpha_outputs["alpha_c03"] = alpha_c03(panel)
    alpha_outputs["alpha_c04"] = alpha_c04(panel)
    alpha_outputs["alpha_c05"] = alpha_c05(panel)
    alpha_outputs["alpha_c06"] = alpha_c06(panel)
    alpha_outputs["alpha_c07"] = alpha_c07(panel)
    alpha_outputs["alpha_c08"] = alpha_c08(panel)
    alpha_outputs["alpha_c09"] = alpha_c09(panel)
    alpha_outputs["alpha_c10"] = alpha_c10(panel)
    alpha_outputs["alpha_c11"] = alpha_c11(panel)
    alpha_outputs["alpha_c12"] = alpha_c12(panel)
    alpha_outputs["alpha_c13"] = alpha_c13(panel)
    alpha_outputs["alpha_c14"] = alpha_c14(panel)
    alpha_outputs["alpha_c15"] = alpha_c15(panel)
    alpha_outputs["alpha_c16"] = alpha_c16(panel)
    alpha_outputs["alpha_c17"] = alpha_c17(panel)
    alpha_outputs["alpha_c18"] = alpha_c18(panel)
    alpha_outputs["alpha_c19"] = alpha_c19(panel)
    alpha_outputs["alpha_c20"] = alpha_c20(panel)

    out = pd.DataFrame(alpha_outputs)
    out.index.names = ["symbol", "ts"]
    return out.sort_index()

