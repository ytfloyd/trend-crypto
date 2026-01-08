#!/usr/bin/env python
from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np
import pandas as pd


CASH_YIELD_ANNUAL_DEFAULT = 0.04
PPY = 365  # daily periods per year
RF_DAILY_DEFAULT = (1.0 + CASH_YIELD_ANNUAL_DEFAULT) ** (1.0 / PPY) - 1.0


@dataclasses.dataclass
class KumaConfig:
    breakout_lookback: int = 20  # breakout window
    fast_ma: int = 5
    slow_ma: int = 40
    atr_window: int = 20
    vol_window: int = 20
    cash_yield_annual: float = CASH_YIELD_ANNUAL_DEFAULT
    cash_buffer: float = 0.05  # keep 5% cash buffer


def _to_panel(df: pd.DataFrame) -> pd.DataFrame:
    if not {"ts", "symbol"}.issubset(df.columns):
        raise ValueError("Panel must include ts and symbol columns.")
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.sort_values(["symbol", "ts"]).set_index(["symbol", "ts"])
    return out


def compute_daily_returns(panel: pd.DataFrame) -> pd.Series:
    close = panel["close"]
    return close.groupby(level="symbol").pct_change()


def compute_ma(panel: pd.DataFrame, window: int) -> pd.Series:
    close = panel["close"]
    return close.groupby(level="symbol").rolling(window, min_periods=window).mean().droplevel(0)


def compute_atr(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    prev_close = close.groupby(level="symbol").shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = (
        tr.groupby(level="symbol")
        .rolling(window, min_periods=window)
        .mean()
        .droplevel(0)
    )
    return atr


def compute_realized_vol(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    ret = compute_daily_returns(panel)
    vol = (
        ret.groupby(level="symbol")
        .rolling(window, min_periods=window)
        .std()
        .droplevel(0)
    )
    return vol


def compute_kuma_positions(panel: pd.DataFrame, cfg: KumaConfig) -> pd.Series:
    """
    Compute long/cash position indicator per (ts, symbol).
    Returns Series indexed by [ts, symbol] with values {0.0, 1.0}.
    """
    out_list: List[pd.Series] = []
    for sym, g in panel.groupby(level="symbol", sort=False):
        g = g.reset_index()
        g = g.sort_values("ts")

        close = g["close"]
        ma_fast = close.rolling(cfg.fast_ma, min_periods=cfg.fast_ma).mean()
        ma_slow = close.rolling(cfg.slow_ma, min_periods=cfg.slow_ma).mean()
        hh_breakout = close.rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max().shift(1)

        # ATR on this single-symbol series
        high = g["high"]
        low = g["low"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(cfg.atr_window, min_periods=cfg.atr_window).mean()

        pos = []
        in_pos = False
        highest_close = np.nan
        entry_atr = np.nan
        stop = np.nan

        for i, row in g.iterrows():
            c = row["close"]
            bf = ma_fast.iloc[i]
            bs = ma_slow.iloc[i]
            br = hh_breakout.iloc[i]
            atr_i = atr.iloc[i]

            # default stay
            if not in_pos:
                can_enter = (
                    pd.notna(br)
                    and pd.notna(bf)
                    and pd.notna(bs)
                    and pd.notna(atr_i)
                    and (c > br)
                    and (bf > bs)
                )
                if can_enter:
                    in_pos = True
                    highest_close = c
                    entry_atr = atr_i
                    stop = c - 2.0 * entry_atr
            else:
                highest_close = max(highest_close, c)
                stop = highest_close - 2.0 * entry_atr
                if c <= stop:
                    in_pos = False
                    highest_close = np.nan
                    entry_atr = np.nan
                    stop = np.nan

            pos.append(1.0 if in_pos else 0.0)

        s = pd.Series(pos, index=g.set_index(["ts", "symbol"]).index, name="position")
        out_list.append(s)

    positions = pd.concat(out_list).sort_index()
    return positions


def build_kuma_weights_and_equity(
    panel: pd.DataFrame,
    positions: pd.Series,
    cfg: KumaConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.sort_index()
    symbols = panel.index.get_level_values("symbol").unique()
    dates = panel.index.get_level_values("ts").unique()

    ret = panel["close"].groupby(level="symbol").pct_change()
    vol = (
        ret.groupby(level="symbol")
        .rolling(cfg.vol_window, min_periods=cfg.vol_window)
        .std()
        .droplevel(0)
    )

    pos_df = positions.unstack("symbol").reindex(index=dates, columns=symbols).fillna(0.0)
    vol_df = vol.unstack("symbol").reindex(index=dates, columns=symbols)
    ret_df = ret.unstack("symbol").reindex(index=dates, columns=symbols)

    weights_list = []
    equity_rows = []

    rf_daily = (1.0 + cfg.cash_yield_annual) ** (1.0 / PPY) - 1.0
    w_prev = pd.Series(0.0, index=symbols)
    cash_prev = 1.0
    equity = 1.0

    for ts in dates:
        pos_t = pos_df.loc[ts]
        vol_t = vol_df.loc[ts]

        eligible = (pos_t == 1.0) & vol_t.notna() & (vol_t > 0)
        if eligible.any():
            inv_vol = 1.0 / (vol_t + 1e-8)
            inv_vol[~eligible] = 0.0
            if inv_vol.sum() > 0:
                w_raw = inv_vol / inv_vol.sum()
            else:
                w_raw = pd.Series(0.0, index=symbols)
        else:
            w_raw = pd.Series(0.0, index=symbols)

        weights_t = w_raw * (1.0 - cfg.cash_buffer)
        gross_long = weights_t.sum()
        cash_weight = 1.0 - gross_long

        # turnover vs prev weights
        turnover = 0.5 * (weights_t - w_prev).abs().sum()

        # apply previous weights to today's returns
        ret_t = ret_df.loc[ts].fillna(0.0)
        port_ret = float((w_prev * ret_t).sum() + cash_prev * rf_daily)
        equity *= (1.0 + port_ret)

        weights_list.append((ts, *weights_t.values))
        equity_rows.append(
            {
                "ts": ts,
                "portfolio_ret": port_ret,
                "portfolio_equity": equity,
                "gross_long": gross_long,
                "cash_weight": cash_weight,
                "turnover": turnover,
            }
        )

        w_prev = weights_t
        cash_prev = cash_weight

    weights_df = pd.DataFrame(weights_list, columns=["ts", *symbols]).set_index("ts")
    equity_df = pd.DataFrame(equity_rows).sort_values("ts").reset_index(drop=True)

    return weights_df, equity_df


def run_kuma_trend_backtest(
    panel: pd.DataFrame,
    cfg: KumaConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cfg = cfg or KumaConfig()
    positions = compute_kuma_positions(panel, cfg)
    weights_df, equity_df = build_kuma_weights_and_equity(panel, positions, cfg)
    return weights_df, equity_df, positions

