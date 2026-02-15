"""Data feed abstractions for live/paper trading.

Provides a replay data feed that replays historical bars in sequence,
and the base protocol used by LiveRunner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import polars as pl


@dataclass(frozen=True)
class BarData:
    """Single OHLCV bar."""

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFeed(ABC):
    """Abstract data feed consumed by LiveRunner."""

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[BarData]:
        """Return the most recent bar for *symbol*."""

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Return the latest close price for *symbol*."""

    @abstractmethod
    def get_history(self, symbol: str, n: int) -> pl.DataFrame:
        """Return up to *n* most-recent bars as a Polars DataFrame."""

    @property
    @abstractmethod
    def is_exhausted(self) -> bool:
        """True when the feed has no more bars to replay."""

    def advance(self, symbol: str) -> None:  # noqa: B027
        """Advance cursor for a single symbol (optional)."""

    def advance_all(self) -> None:  # noqa: B027
        """Advance cursors for all symbols (optional)."""


class ReplayDataFeed(DataFeed):
    """Replays pre-loaded Polars DataFrames bar-by-bar.

    Args:
        data: Mapping of symbol to a Polars DataFrame with columns
              ``ts, symbol, open, high, low, close, volume``.
    """

    def __init__(self, data: dict[str, pl.DataFrame]) -> None:
        self._data = data
        self._cursors: dict[str, int] = {sym: 0 for sym in data}

    def get_latest_bar(self, symbol: str) -> Optional[BarData]:
        df = self._data.get(symbol)
        if df is None or df.is_empty():
            return None
        idx = self._cursors.get(symbol, 0)
        row = df.row(idx, named=True)
        return BarData(
            symbol=row.get("symbol", symbol),
            ts=row["ts"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )

    def get_latest_price(self, symbol: str) -> float:
        bar = self.get_latest_bar(symbol)
        if bar is None:
            raise ValueError(f"No data for {symbol}")
        return bar.close

    def get_history(self, symbol: str, n: int) -> pl.DataFrame:
        df = self._data.get(symbol)
        if df is None:
            return pl.DataFrame()
        idx = self._cursors.get(symbol, 0)
        start = max(0, idx - n + 1)
        return df.slice(start, idx - start + 1)

    @property
    def is_exhausted(self) -> bool:
        return all(
            self._cursors[sym] >= df.height - 1
            for sym, df in self._data.items()
        )

    def advance(self, symbol: str) -> None:
        df = self._data.get(symbol)
        if df is None:
            return
        cur = self._cursors.get(symbol, 0)
        if cur < df.height - 1:
            self._cursors[symbol] = cur + 1

    def advance_all(self) -> None:
        for sym in self._data:
            self.advance(sym)
