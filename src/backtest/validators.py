from __future__ import annotations

import polars as pl


def validate_bars(bars: pl.DataFrame) -> None:
    required_cols = ["ts", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in bars.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if bars.is_empty():
        raise ValueError("Bars cannot be empty")

    ts = bars["ts"]
    if not ts.is_sorted():
        raise ValueError("Bars must be sorted by ts ascending.")
    if ts.n_unique() != ts.len():
        raise ValueError("Duplicate ts values detected.")

    for col in ["open", "high", "low", "close"]:
        if (bars[col] <= 0).any():
            raise ValueError(f"Non-positive price found in {col}")

    if (bars["volume"] < 0).any():
        raise ValueError("Negative volume found")


def validate_context_bounds(history: pl.DataFrame, decision_ts: object) -> None:
    max_ts = history.select(pl.col("ts").max()).item()
    if max_ts != decision_ts:
        raise AssertionError("Context includes future data")
    if history.filter(pl.col("ts") > pl.lit(decision_ts)).height > 0:
        raise AssertionError("Context leaked future ts")


def validate_fill_timing(trades: pl.DataFrame, bars: pl.DataFrame) -> None:
    return

