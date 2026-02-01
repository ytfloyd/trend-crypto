#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    breakout_lookback: int
    fast_ma: int
    slow_ma: int


@dataclass(frozen=True)
class TranstrendConfigV1:
    horizons: list[HorizonSpec]
    target_vol_annual: float = 0.20
    danger_gross: float = 0.25
    cost_bps: float = 20.0
    fee_bps: float = 10.0
    slippage_bps: float = 10.0
    cash_yield_annual: float = 0.04
    cash_buffer: float = 0.05
    max_gross: float = 1.0
    vol_floor: float = 0.10
    vol_window: int = 20
    danger_btc_vol_threshold: float = 0.80
    danger_btc_dd20_threshold: float = -0.20
    danger_btc_ret5_threshold: float = -0.10
    execution_lag_bars: int = 1
    atr_window: int = 20
    atr_k: float = 3.0
    stop_cooldown_days: int = 5
    stop_use_atr_entry: bool = True

    def to_dict(self) -> dict:
        cfg = asdict(self)
        cfg["horizons"] = [asdict(h) for h in self.horizons]
        return cfg


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


def _annualize_vol(series: pd.Series, window: int) -> pd.Series:
    vol = series.rolling(window, min_periods=window).std()
    return vol * np.sqrt(365.0)


def compute_atr(panel: pd.DataFrame, window: int) -> pd.Series:
    df = _ensure_symbol_ts_columns(panel).sort_values(["symbol", "ts"]).copy()
    prev_close = df.groupby("symbol")["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.groupby(df["symbol"]).rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
    atr = atr.groupby(df["symbol"]).shift(1)
    atr.name = "atr"
    return atr


def compute_trend_scores(panel: pd.DataFrame, cfg: TranstrendConfigV1) -> pd.DataFrame:
    df0 = _ensure_symbol_ts_columns(panel).copy()
    df = df0.sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"]
        signal_sum = 0.0
        for h in cfg.horizons:
            breakout_max = close.shift(1).rolling(h.breakout_lookback, min_periods=h.breakout_lookback).max()
            breakout = (close > breakout_max).astype(float)
            sma_fast = close.shift(1).rolling(h.fast_ma, min_periods=h.fast_ma).mean()
            sma_slow = close.shift(1).rolling(h.slow_ma, min_periods=h.slow_ma).mean()
            ma_filter = (sma_fast > sma_slow).astype(float)
            signal_sum += breakout * ma_filter
        score = signal_sum / float(len(cfg.horizons))
        ret_cc = close.pct_change()
        vol_ann = _annualize_vol(ret_cc.shift(1), cfg.vol_window)
        group = group.copy()
        group["symbol"] = group["symbol"].iloc[0] if "symbol" in group.columns else group.name
        group["score"] = score
        group["vol_ann"] = vol_ann
        return group

    out = df.groupby("symbol", group_keys=False).apply(_per_symbol)
    if "symbol" not in out.columns:
        out["symbol"] = df0["symbol"].values
    if "ts" not in out.columns:
        out["ts"] = df0["ts"].values
    out["atr"] = compute_atr(out, cfg.atr_window)
    return out


def compute_danger_flags(panel: pd.DataFrame, cfg: TranstrendConfigV1) -> pd.Series:
    df = _ensure_symbol_ts_columns(panel).copy()
    df = df.sort_values(["symbol", "ts"])
    btc = df[df["symbol"] == "BTC-USD"]
    ts_index = pd.Index(sorted(df["ts"].unique()), name="ts")
    if btc.empty:
        return pd.Series(False, index=ts_index, name="danger")

    btc = btc.sort_values("ts")
    close = btc["close"]
    ret_cc = close.pct_change()
    vol_ann = _annualize_vol(ret_cc.shift(1), 20)
    roll_max = close.shift(1).rolling(20, min_periods=20).max()
    dd20 = close / roll_max - 1.0
    ret5 = close / close.shift(5) - 1.0

    danger = (
        (vol_ann > cfg.danger_btc_vol_threshold)
        | (dd20 < cfg.danger_btc_dd20_threshold)
        | (ret5 < cfg.danger_btc_ret5_threshold)
    )
    danger = danger.fillna(False).astype(bool)
    danger.index = btc["ts"].values
    danger = danger.reindex(ts_index).fillna(False).astype(bool)
    danger.name = "danger"
    return danger


def apply_atr_stops(panel: pd.DataFrame, cfg: TranstrendConfigV1) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = panel.sort_values(["symbol", "ts"]).copy()
    records = []
    events = []

    for symbol, group in df.groupby("symbol", sort=False):
        g = group.reset_index(drop=True)
        n = len(g)
        score = g["score"].fillna(0.0).values
        atr = g["atr"].values
        open_px = g["open"].values
        close_px = g["close"].values

        decision_queue = [False] * cfg.execution_lag_bars
        entry_atr_queue = [np.nan] * cfg.execution_lag_bars
        in_pos = False
        atr_entry = np.nan
        entry_price = np.nan
        max_close = np.nan
        stop_level = np.nan
        cooldown = 0

        for i in range(n):
            in_pos = bool(decision_queue[0])
            stop_hit = False

            if in_pos and np.isfinite(stop_level):
                max_close = max(max_close, close_px[i]) if np.isfinite(max_close) else close_px[i]
                stop_level = max_close - cfg.atr_k * atr_entry
                if close_px[i] <= stop_level:
                    stop_hit = True
                    cooldown = cfg.stop_cooldown_days

            allow = score[i] > 0
            if cooldown > 0:
                allow = False
            if stop_hit:
                allow = False

            if in_pos and np.isnan(atr_entry):
                atr_entry = entry_atr_queue[0]
                entry_price = open_px[i]
                max_close = close_px[i]
                if np.isfinite(atr_entry):
                    stop_level = entry_price - cfg.atr_k * atr_entry

            decision_queue.pop(0)
            entry_atr_queue.pop(0)
            decision_queue.append(bool(allow))
            entry_atr_queue.append(atr[i] if allow else np.nan)

            in_stop = bool(cooldown > 0)
            records.append(
                {
                    "ts": g.loc[i, "ts"],
                    "symbol": symbol,
                    "stop_level": stop_level,
                    "in_stop": in_stop,
                    "cooldown_remaining": cooldown,
                    "atr_entry": atr_entry,
                    "max_close_since_entry": max_close,
                }
            )
            if stop_hit:
                events.append(
                    {
                        "ts": g.loc[i, "ts"],
                        "symbol": symbol,
                        "stop_hit": True,
                        "stop_level": stop_level,
                        "close": close_px[i],
                        "atr_entry": atr_entry,
                    }
                )

            if cooldown > 0:
                cooldown -= 1
            if not in_pos and np.isfinite(atr_entry) and not allow:
                atr_entry = np.nan
                entry_price = np.nan
                max_close = np.nan
                stop_level = np.nan

    stop_levels = pd.DataFrame.from_records(records)
    stop_events = pd.DataFrame.from_records(events)
    return stop_levels, stop_events


def build_target_weights(panel: pd.DataFrame, cfg: TranstrendConfigV1) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    df = _ensure_symbol_ts_columns(panel).copy()
    if "score" not in df.columns or "vol_ann" not in df.columns or "atr" not in df.columns:
        df = compute_trend_scores(df, cfg)

    danger = compute_danger_flags(df, cfg)
    stop_levels, stop_events = apply_atr_stops(df, cfg)

    stop_mask = stop_levels[["ts", "symbol", "in_stop", "cooldown_remaining"]].copy()
    stop_mask["stop_block"] = stop_mask["in_stop"] | (stop_mask["cooldown_remaining"] > 0)

    df = df.merge(stop_mask[["ts", "symbol", "stop_block"]], on=["ts", "symbol"], how="left")
    df["stop_block"] = df["stop_block"].fillna(False)

    def _per_ts(group: pd.DataFrame) -> pd.DataFrame:
        score = group["score"].fillna(0.0).clip(lower=0.0)
        vol = group["vol_ann"].fillna(np.nan)
        vol = vol.where(np.isfinite(vol), np.nan)
        vol = vol.fillna(cfg.vol_floor)
        vol = vol.clip(lower=cfg.vol_floor)

        w_raw = score / vol
        w_raw = w_raw.where(np.isfinite(w_raw), 0.0)
        w_raw = w_raw.where(~group["stop_block"], 0.0)

        gross_raw = w_raw.sum()
        gross_target = min(cfg.max_gross, 1.0 - cfg.cash_buffer)
        if gross_raw > 0:
            w = w_raw / gross_raw * gross_target
        else:
            w = w_raw * 0.0

        port_vol = np.sqrt(np.sum((w * vol) ** 2))
        if port_vol > 0 and cfg.target_vol_annual > 0:
            scale = min(1.0, cfg.target_vol_annual / port_vol)
            w = w * scale

        gross = w.sum()
        if gross > cfg.max_gross and gross > 0:
            w = w * (cfg.max_gross / gross)
            gross = w.sum()

        if danger.get(group.name, False) and gross > 0:
            w = w * (cfg.danger_gross / gross)

        out = group.copy()
        out["ts"] = group["ts"].iloc[0] if "ts" in group.columns else group.name
        out["w_signal"] = w.values
        return out

    weights = df.groupby("ts", group_keys=False).apply(_per_ts)
    weights["w_signal"] = weights["w_signal"].clip(lower=0.0)
    return weights.reset_index(drop=True), danger, stop_levels, stop_events


def simulate_portfolio(
    panel: pd.DataFrame,
    weights_signal: pd.DataFrame,
    cfg: TranstrendConfigV1,
    danger: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.execution_lag_bars < 1:
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

    w_held = weights_wide.shift(cfg.execution_lag_bars).fillna(0.0)

    gross_exposure = w_held.sum(axis=1)
    cash_weight = (1.0 - gross_exposure).clip(lower=0.0)

    turnover_one = (w_held - w_held.shift(1).fillna(0.0)).abs().sum(axis=1)
    turnover_two = 0.5 * turnover_one

    cost_ret = turnover_one * (cfg.cost_bps / 10000.0)
    rf_bar = (1 + cfg.cash_yield_annual) ** (1 / 365.0) - 1.0

    portfolio_ret = (w_held * returns_wide).sum(axis=1) + cash_weight * rf_bar - cost_ret
    portfolio_equity = (1 + portfolio_ret).cumprod()

    danger_aligned = danger.reindex(common_ts).fillna(False).astype(bool)

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
            "danger": danger_aligned.values,
        }
    )

    weights_held = w_held.stack().reset_index()
    weights_held.columns = ["ts", "symbol", "w_held"]

    return equity_df, weights_held
