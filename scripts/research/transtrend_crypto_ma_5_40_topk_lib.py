#!/usr/bin/env python
from __future__ import annotations

import duckdb
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


def load_panel(db_path: str, table: str, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if not symbols:
        raise ValueError("symbols list is empty")
    placeholders = ",".join(["?"] * len(symbols))
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM {table}
            WHERE symbol IN ({placeholders})
              AND ts >= ? AND ts <= ?
            ORDER BY ts, symbol
            """,
            symbols + [start, end],
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


def compute_signals(panel: pd.DataFrame, fast: int = 5, slow: int = 40) -> pd.DataFrame:
    df = _ensure_symbol_ts_columns(panel).copy().sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"]
        fast_ma = close.shift(1).rolling(fast, min_periods=fast).mean()
        slow_ma = close.shift(1).rolling(slow, min_periods=slow).mean()
        signal = (fast_ma > slow_ma).astype(float)
        out = group.copy()
        out["signal"] = signal
        return out

    return apply_by_symbol(df, _per_symbol)


def compute_rank_score(panel: pd.DataFrame, lookback: int = 60, method: str = "ret") -> pd.DataFrame:
    if method != "ret":
        raise ValueError(f"Unknown rank method: {method}")

    df = _ensure_symbol_ts_columns(panel).copy().sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"]
        score = close.shift(1) / close.shift(1 + lookback) - 1.0
        out = group.copy()
        out["score"] = score
        return out

    return apply_by_symbol(df, _per_symbol)


def build_topk_weights(panel_with_signal_and_score: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    df = _ensure_symbol_ts_columns(panel_with_signal_and_score).copy()

    def _per_ts(group: pd.DataFrame) -> pd.DataFrame:
        mask = (group["signal"] > 0) & np.isfinite(group["score"])
        candidates = group.loc[mask].copy()
        if not candidates.empty:
            top = candidates.nlargest(k, "score")
            selected = set(top["symbol"].tolist())
            n_selected = len(selected)
        else:
            selected = set()
            n_selected = 0

        out = group.copy()
        if n_selected > 0:
            out["w_signal"] = out["symbol"].apply(
                lambda s: 1.0 / float(n_selected) if s in selected else 0.0
            )
        else:
            out["w_signal"] = 0.0
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
