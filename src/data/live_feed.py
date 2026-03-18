"""
DuckDB-backed live data feed.

Implements the DataFeed interface by querying bars_1h from DuckDB,
giving LiveRunner access to live/recent market data without needing
pre-loaded DataFrames.

Assumes the Coinbase collector keeps DuckDB updated on a schedule.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl

from .feed import BarData, DataFeed

logger = logging.getLogger("live_feed")

DEFAULT_DB = str(
    Path(__file__).resolve().parents[2] / ".." / "data" / "market.duckdb"
)


class DuckDBLiveDataFeed(DataFeed):
    """Live data feed backed by DuckDB hourly bars.

    Queries bars_1h on each call. Suitable for hourly-frequency
    strategies where latency is not critical.

    Parameters
    ----------
    db_path : str
        Path to the DuckDB database file.
    symbols : list[str]
        Symbols to serve data for.
    max_staleness_hours : float
        Log a warning if the most recent bar is older than this.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        symbols: Optional[list[str]] = None,
        max_staleness_hours: float = 3.0,
    ) -> None:
        self.db_path = db_path
        self._symbols = symbols or []
        self.max_staleness_hours = max_staleness_hours
        self._cache: dict[str, pl.DataFrame] = {}
        self._cache_ts: Optional[datetime] = None

    def _refresh_cache(self) -> None:
        """Reload recent bars from DuckDB."""
        now = datetime.now(timezone.utc)
        if (
            self._cache_ts is not None
            and (now - self._cache_ts).total_seconds() < 300
        ):
            return

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            for sym in self._symbols:
                df_pd = con.execute(
                    """
                    SELECT symbol, ts, open, high, low, close, volume
                    FROM bars_1h
                    WHERE symbol = ?
                      AND ts >= current_timestamp - INTERVAL '5000 hours'
                      AND open > 0 AND close > 0
                    ORDER BY ts
                    """,
                    [sym],
                ).fetch_df()
                if not df_pd.empty:
                    df_pd["ts"] = df_pd["ts"].dt.tz_localize(None)
                    self._cache[sym] = pl.from_pandas(df_pd)
        finally:
            con.close()

        self._cache_ts = now

        if self._cache:
            latest = max(
                df["ts"].max() for df in self._cache.values() if len(df) > 0
            )
            staleness = (now - latest.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            if staleness > self.max_staleness_hours:
                logger.warning(
                    "Data is %.1fh stale (threshold: %.1fh)",
                    staleness, self.max_staleness_hours,
                )

    def get_latest_bar(self, symbol: str) -> Optional[BarData]:
        self._refresh_cache()
        df = self._cache.get(symbol)
        if df is None or len(df) == 0:
            return None
        row = df.row(-1, named=True)
        return BarData(
            symbol=row["symbol"],
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
        self._refresh_cache()
        df = self._cache.get(symbol)
        if df is None:
            return pl.DataFrame()
        return df.tail(n)

    @property
    def is_exhausted(self) -> bool:
        return False  # live feed never exhausts

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)
