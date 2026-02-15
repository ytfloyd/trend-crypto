#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.groupby_utils import apply_by_symbol, apply_by_ts


def _ensure_symbol_ts_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "symbol" in df.columns and "ts" in df.columns:
        return df
    if isinstance(df.index, pd.MultiIndex) and len(df.index.levels) == 2:
        df2 = df.reset_index()
        cols = list(df2.columns)
        cols[0] = "symbol"
        cols[1] = "ts"
        df2.columns = cols
        return df2
    raise ValueError("Expected columns ['symbol','ts'] or a 2-level MultiIndex.")


def compute_atr30(panel: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = _ensure_symbol_ts_columns(panel).copy().sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        high = group["high"]
        low = group["low"]
        close = group["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window, min_periods=window).mean().shift(1)
        out = group.copy()
        out["tr"] = tr
        out["atr30"] = atr
        return out

    return apply_by_symbol(df, _per_symbol)


def compute_vol31(panel: pd.DataFrame, window: int = 31) -> pd.DataFrame:
    df = _ensure_symbol_ts_columns(panel).copy().sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"]
        ret_cc = close.pct_change()
        vol = ret_cc.rolling(window, min_periods=window).std().shift(1)
        out = group.copy()
        out["vol31"] = vol
        return out

    return apply_by_symbol(df, _per_symbol)


def compute_breakout_and_ma_filter(
    panel: pd.DataFrame,
    breakout_lookback: int = 20,
    fast: int = 5,
    slow: int = 40,
) -> pd.DataFrame:
    df = _ensure_symbol_ts_columns(panel).copy().sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"]
        breakout_max = close.shift(1).rolling(breakout_lookback, min_periods=breakout_lookback).max()
        breakout20 = close > breakout_max
        sma5 = close.shift(1).rolling(fast, min_periods=fast).mean()
        sma40 = close.shift(1).rolling(slow, min_periods=slow).mean()
        trend_ok = sma5 > sma40
        entry_signal = breakout20 & trend_ok
        out = group.copy()
        out["breakout20"] = breakout20
        out["sma5"] = sma5
        out["sma40"] = sma40
        out["trend_ok"] = trend_ok
        out["entry_signal"] = entry_signal
        return out

    return apply_by_symbol(df, _per_symbol)


def apply_dynamic_atr_trailing_stop(
    panel_with_signals: pd.DataFrame,
    atr_mult: float = 3.0,
) -> pd.DataFrame:
    df = _ensure_symbol_ts_columns(panel_with_signals).copy().sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        stop_level = []
        stop_hit = []
        stop_block = []

        in_pos = False
        highest_close = np.nan

        for _, row in out.iterrows():
            close = float(row["close"])
            atr = row.get("atr30")
            entry_signal = bool(row.get("entry_signal", False))

            hit = False
            level = np.nan
            if in_pos:
                highest_close = close if not np.isfinite(highest_close) else max(highest_close, close)
                if np.isfinite(atr):
                    level = highest_close - atr_mult * float(atr)
                    if close <= level:
                        hit = True
                if hit:
                    in_pos = False
                    highest_close = np.nan
            else:
                level = np.nan

            block_active = hit
            if not in_pos and not block_active and entry_signal:
                in_pos = True
                highest_close = close

            stop_level.append(level)
            stop_hit.append(hit)
            stop_block.append(block_active)

        out["stop_level"] = stop_level
        out["stop_hit"] = stop_hit
        out["stop_block"] = stop_block
        return out

    return apply_by_symbol(df, _per_symbol)


def build_inverse_vol_weights(panel_features: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_symbol_ts_columns(panel_features).copy()

    def _per_ts(group: pd.DataFrame) -> pd.DataFrame:
        eligible = (
            group["entry_signal"].fillna(False)
            & (~group["stop_block"].fillna(False))
            & group["vol31"].notna()
        )
        vol = group["vol31"].where(eligible, np.nan)
        raw = 1.0 / vol
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if raw.sum() > 0:
            w = raw / raw.sum()
        else:
            w = raw * 0.0
        out = group.copy()
        out["w_signal"] = w.values
        return out[["symbol", "ts", "w_signal"]]

    weights = apply_by_ts(df, _per_ts)
    weights["w_signal"] = weights["w_signal"].fillna(0.0)
    return weights.reset_index(drop=True)


def simulate_portfolio(
    panel: pd.DataFrame,
    weights_signal: pd.DataFrame,
    cost_bps: float,
    execution_lag_bars: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if execution_lag_bars < 1:
        raise ValueError("execution_lag_bars must be >= 1")

    df = _ensure_symbol_ts_columns(panel).copy().sort_values(["ts", "symbol"])
    weights_signal = _ensure_symbol_ts_columns(weights_signal).copy()

    ret = df.copy()
    ret["ret_oc"] = ret["close"] / ret["open"] - 1.0
    returns_wide = ret.pivot(index="ts", columns="symbol", values="ret_oc").sort_index()
    weights_wide = weights_signal.pivot(index="ts", columns="symbol", values="w_signal").sort_index()

    common_ts = returns_wide.index.intersection(weights_wide.index)
    returns_wide = returns_wide.reindex(common_ts).fillna(0.0)
    weights_wide = weights_wide.reindex(common_ts).fillna(0.0)

    w_held = weights_wide.shift(execution_lag_bars).fillna(0.0)

    gross_exposure = w_held.sum(axis=1)
    cash_weight = (1.0 - gross_exposure).clip(lower=0.0)

    turnover_one = (w_held - w_held.shift(1).fillna(0.0)).abs().sum(axis=1)
    turnover_two = 0.5 * turnover_one

    cost_ret = turnover_one * (cost_bps / 10000.0)
    portfolio_ret = (w_held * returns_wide).sum(axis=1) - cost_ret
    portfolio_equity = (1 + portfolio_ret).cumprod()

    equity_df = pd.DataFrame(
        {
            "ts": common_ts,
            "portfolio_ret": portfolio_ret.values,
            "portfolio_equity": portfolio_equity.values,
            "gross_exposure": gross_exposure.values,
            "cash_weight": cash_weight.values,
            "turnover_one_sided": turnover_one.values,
            "turnover_two_sided": turnover_two.values,
            "cost_ret": cost_ret.values,
        }
    )

    weights_held = w_held.stack().reset_index()
    weights_held.columns = ["ts", "symbol", "w_held"]

    return equity_df, weights_held
