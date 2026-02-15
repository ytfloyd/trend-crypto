#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.research.groupby_utils import apply_by_symbol, apply_by_ts

@dataclass(frozen=True)
class KumaConfig:
    breakout_lookback: int = 20
    fast_ma: int = 5
    slow_ma: int = 40
    atr_window: int = 20
    vol_window: int = 20
    cash_yield_annual: float = 0.04
    cash_buffer: float = 0.05
    atr_k: float = 2.0


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


def _compute_indicators(group: pd.DataFrame, cfg: KumaConfig) -> pd.DataFrame:
    close = group["close"]
    high = group["high"]
    low = group["low"]

    fast_ma = close.shift(1).rolling(cfg.fast_ma, min_periods=cfg.fast_ma).mean()
    slow_ma = close.shift(1).rolling(cfg.slow_ma, min_periods=cfg.slow_ma).mean()
    breakout_max = close.shift(1).rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max()
    breakout = close > breakout_max

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(cfg.atr_window, min_periods=cfg.atr_window).mean().shift(1)

    ret_cc = close.pct_change()
    vol = ret_cc.rolling(cfg.vol_window, min_periods=cfg.vol_window).std().shift(1)

    out = group.copy()
    out["fast_ma"] = fast_ma
    out["slow_ma"] = slow_ma
    out["breakout"] = breakout
    out["atr"] = atr
    out["vol"] = vol
    out["signal"] = (fast_ma > slow_ma) & breakout
    return out


def _apply_trailing_stop(group: pd.DataFrame, cfg: KumaConfig) -> pd.DataFrame:
    out = group.copy()
    weights = []
    in_pos = False
    atr_entry = np.nan
    max_close = np.nan

    for _, row in out.iterrows():
        desired = float(row["weight"])
        close = float(row["close"])
        atr = float(row["atr"]) if np.isfinite(row["atr"]) else np.nan

        if desired > 0:
            if not in_pos:
                if not np.isfinite(atr):
                    desired = 0.0
                else:
                    in_pos = True
                    atr_entry = atr
                    max_close = close
            if in_pos:
                max_close = max(max_close, close)
                stop_level = max_close - cfg.atr_k * atr_entry
                if close <= stop_level:
                    desired = 0.0
                    in_pos = False
                    atr_entry = np.nan
                    max_close = np.nan
        else:
            in_pos = False
            atr_entry = np.nan
            max_close = np.nan

        weights.append(desired)

    out["weight"] = weights
    return out


def run_kuma_trend_backtest(
    panel: pd.DataFrame, cfg: KumaConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _ensure_symbol_ts_columns(panel).sort_values(["symbol", "ts"]).copy()
    df = apply_by_symbol(df, lambda g: _compute_indicators(g, cfg))

    def _per_ts(group: pd.DataFrame) -> pd.DataFrame:
        eligible = group["signal"].fillna(False) & group["vol"].notna()
        vol = group["vol"].where(eligible, np.nan)
        inv_vol = 1.0 / vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        inv_vol = inv_vol.fillna(0.0)
        gross_target = max(0.0, 1.0 - cfg.cash_buffer)
        if inv_vol.sum() > 0:
            w = inv_vol / inv_vol.sum() * gross_target
        else:
            w = inv_vol * 0.0
        out = group.copy()
        out["weight"] = w.values
        return out

    df = apply_by_ts(df, _per_ts)
    df = apply_by_symbol(df, lambda g: _apply_trailing_stop(g, cfg))

    weights_df = df.set_index(["symbol", "ts"])[["weight"]].sort_index()
    positions = df.set_index(["symbol", "ts"])[["close", "signal", "weight"]].sort_index()

    returns = df.copy()
    returns["ret_cc"] = returns.groupby("symbol")["close"].pct_change()
    returns_wide = returns.pivot(index="ts", columns="symbol", values="ret_cc").sort_index()
    weights_wide = weights_df.reset_index().pivot(index="ts", columns="symbol", values="weight").sort_index()

    common_ts = returns_wide.index.intersection(weights_wide.index)
    returns_wide = returns_wide.reindex(common_ts).fillna(0.0)
    weights_wide = weights_wide.reindex(common_ts).fillna(0.0)

    w_held = weights_wide.shift(1).fillna(0.0)
    gross_exposure = w_held.abs().sum(axis=1)
    cash_weight = (1.0 - gross_exposure).clip(lower=0.0)
    turnover = (w_held - w_held.shift(1).fillna(0.0)).abs().sum(axis=1)

    rf_bar = (1 + cfg.cash_yield_annual) ** (1 / 365.0) - 1.0
    portfolio_ret = (w_held * returns_wide).sum(axis=1) + cash_weight * rf_bar
    portfolio_equity = (1 + portfolio_ret).cumprod()

    equity_df = pd.DataFrame(
        {
            "ts": common_ts,
            "portfolio_ret": portfolio_ret.values,
            "portfolio_equity": portfolio_equity.values,
            "turnover": turnover.values,
        }
    )

    return weights_df, equity_df, positions
