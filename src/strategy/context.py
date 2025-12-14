from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import polars as pl


@dataclass
class MarketBar:
    ts: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class StrategyContext:
    history: pl.DataFrame
    decision_ts: datetime


def make_strategy_context(bars: pl.DataFrame, i: int, lookback: Optional[int]) -> StrategyContext:
    """
    Slice bars up to and including index i, enforcing no future leakage.
    """
    if i < 0 or i >= len(bars):
        raise IndexError("Index out of range when building strategy context")
    start = 0 if lookback is None else max(0, i + 1 - lookback)
    window = i - start + 1
    history = bars.slice(start, window)
    decision_ts = bars[i, "ts"]
    max_ts = history.select(pl.col("ts").max()).item()
    if max_ts != decision_ts:
        raise ValueError("StrategyContext contains data beyond decision timestamp")
    if history.select(pl.col("ts").max()).item() != decision_ts:
        raise ValueError("Context timing mismatch")
    if history.filter(pl.col("ts") > decision_ts).height > 0:
        raise ValueError("Context includes future bars")
    return StrategyContext(history=history, decision_ts=decision_ts)

