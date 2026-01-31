#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

UNIVERSE = [
    "BTC-USD",
    "ETH-USD",
    "ADA-USD",
    "BNB-USD",
    "XRP-USD",
    "SOL-USD",
    "DOT-USD",
    "DOGE-USD",
    "AVAX-USD",
    "UNI-USD",
    "LINK-USD",
    "LTC-USD",
    "ALGO-USD",
    "BCH-USD",
    "ATOM-USD",
    "ICP-USD",
    "XLM-USD",
    "FIL-USD",
    "TRX-USD",
    "VET-USD",
    "ETC-USD",
    "XTZ-USD",
]


@dataclass(frozen=True)
class SimpleBaselineConfig:
    fast_ma: int = 20
    slow_ma: int = 100
    cost_bps: float = 0.0
    execution_lag_bars: int = 1


def load_panel(db_path: str, table: str, start: str, end: str) -> pd.DataFrame:
    symbol_list = ",".join([f"'{s}'" for s in UNIVERSE])
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM {table}
            WHERE symbol IN ({symbol_list})
              AND ts >= ? AND ts <= ?
            ORDER BY ts, symbol
            """,
            [start, end],
        ).fetch_df()
    finally:
        con.close()

    required = {"symbol", "ts", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=["open", "close"])
    return df


def compute_signals(panel: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"]
        fast_ma = close.shift(1).rolling(fast, min_periods=fast).mean()
        slow_ma = close.shift(1).rolling(slow, min_periods=slow).mean()
        signal = (fast_ma > slow_ma).astype(float)
        out = group.copy()
        out["signal"] = signal
        return out

    out = df.groupby("symbol", group_keys=False).apply(_per_symbol)
    return out


def build_equal_weights(panel_with_signal: pd.DataFrame) -> pd.DataFrame:
    df = panel_with_signal.copy()
    if "signal" not in df.columns:
        raise ValueError("signal column missing; run compute_signals first")

    def _per_ts(group: pd.DataFrame) -> pd.DataFrame:
        signal = group["signal"].fillna(0.0).clip(lower=0.0)
        n_on = int((signal > 0).sum())
        if n_on > 0:
            w = signal / float(n_on)
        else:
            w = signal * 0.0
        out = group[["ts", "symbol"]].copy()
        out["w_signal"] = w.values
        return out

    weights = df.groupby("ts", group_keys=False).apply(_per_ts)
    weights["w_signal"] = weights["w_signal"].clip(lower=0.0)
    return weights.reset_index(drop=True)


def simulate_portfolio(
    panel: pd.DataFrame,
    weights_signal: pd.DataFrame,
    cfg: SimpleBaselineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.execution_lag_bars < 1:
        raise ValueError("execution_lag_bars must be >= 1")

    df = panel.copy().sort_values(["ts", "symbol"])
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
