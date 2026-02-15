"""Coinbase Advanced Trade data collector.

Fetches 1-minute OHLCV candles for all USD spot products, stores in DuckDB,
and creates resampled views for backtesting at any timeframe.

Usage:
    collector = CoinbaseCollector(db_path="data/market.duckdb")
    collector.backfill_symbol("BTC-USD")
    collector.update_all()

    # Top-N by ADV
    top = collector.discover_top_n(n=200, lookback_days=90, workers=8)
    collector.backfill_all(symbols=[s for s, _ in top], workers=8)
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import duckdb
import polars as pl

from common.logging import get_logger

logger = get_logger("collector")

# Coinbase Advanced Trade granularity strings
GRANULARITY_1M = "ONE_MINUTE"
GRANULARITY_5M = "FIVE_MINUTE"
GRANULARITY_15M = "FIFTEEN_MINUTE"
GRANULARITY_1H = "ONE_HOUR"
GRANULARITY_1D = "ONE_DAY"

# API limits: max 300 candles per request
MAX_CANDLES_PER_REQUEST = 300

# For 1-minute candles: 300 candles = 5 hours
WINDOW_MINUTES_1M = MAX_CANDLES_PER_REQUEST  # 300 minutes = 5 hours

# Stablecoins to exclude (shared with coinbase_usd_universe.py)
STABLE_BASES: set[str] = {
    "USDC", "USDT", "DAI", "PAX", "TUSD", "GUSD",
    "BUSD", "USDP", "PYUSD", "FDUSD", "USDS",
}

# Additional exclusions (fiats, wrapped assets)
EXCLUDED_BASES: set[str] = STABLE_BASES | {
    "EUR", "GBP", "CAD", "AUD", "JPY", "GYEN", "WBTC", "UST",
}


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Allows up to ``max_rps`` requests per second across all threads.
    """

    def __init__(self, max_rps: float) -> None:
        self.max_rps = max_rps
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.max_rps <= 0:
            return
        while True:
            now = time.time()
            with self._lock:
                # Purge timestamps older than 1 second
                while self._timestamps and self._timestamps[0] <= now - 1.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_rps:
                    self._timestamps.append(now)
                    return
                sleep_time = max(0.0, (self._timestamps[0] + 1.0) - now)
            if sleep_time > 0:
                time.sleep(sleep_time)


class CoinbaseCollector:
    """Collect OHLCV candles from Coinbase Advanced Trade API into DuckDB.

    Stores raw 1-minute candles in ``candles_1m`` table and creates
    resampled views (``bars_1h``, ``bars_4h``, ``bars_1d``) that are
    directly compatible with the DataPortal.
    """

    def __init__(
        self,
        db_path: str,
        api_key: str | None = None,
        api_secret: str | None = None,
        max_rps: float = 25.0,
    ) -> None:
        self.db_path = db_path
        self.max_rps = max_rps
        self._rate_limiter = RateLimiter(max_rps)

        # Lazy import — only needed when actually fetching
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._api_key = api_key
        self._api_secret = api_secret

        # Open DuckDB and ensure schema
        self._conn = duckdb.connect(db_path)
        self.ensure_schema()

    def _create_client(self) -> Any:
        """Create a new Coinbase REST client instance."""
        from coinbase.rest import RESTClient

        if self._api_key and self._api_secret:
            return RESTClient(
                api_key=self._api_key,
                api_secret=self._api_secret,
            )
        # Try environment variables (COINBASE_API_KEY, COINBASE_API_SECRET)
        return RESTClient()

    def _get_client(self) -> Any:
        """Lazily initialize the main Coinbase REST client (not thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def _open_conn(self) -> duckdb.DuckDBPyConnection:
        """Open a new DuckDB connection (for use by worker threads)."""
        return duckdb.connect(self.db_path)

    # ──────────────────────────────────────────────
    # Schema
    # ──────────────────────────────────────────────

    def ensure_schema(self) -> None:
        """Create the candles_1m table and resampled views if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candles_1m (
                symbol    VARCHAR NOT NULL,
                ts        TIMESTAMPTZ NOT NULL,
                open      DOUBLE NOT NULL,
                high      DOUBLE NOT NULL,
                low       DOUBLE NOT NULL,
                close     DOUBLE NOT NULL,
                volume    DOUBLE NOT NULL
            );
        """)

        # Create unique index for upsert safety (if not exists)
        try:
            self._conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_1m_pk
                ON candles_1m (symbol, ts);
            """)
        except duckdb.CatalogException:
            pass  # Index already exists

        self._create_resampled_views()

    def _create_resampled_views(self) -> None:
        """Create resampled views compatible with DataPortal."""
        for interval, view_name in [
            ("1 hour", "bars_1h"),
            ("4 hours", "bars_4h"),
            ("1 day", "bars_1d"),
        ]:
            self._conn.execute(f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT
                    symbol,
                    time_bucket(INTERVAL '{interval}', ts) AS ts,
                    FIRST(open ORDER BY ts)  AS open,
                    MAX(high)                AS high,
                    MIN(low)                 AS low,
                    LAST(close ORDER BY ts)  AS close,
                    SUM(volume)              AS volume
                FROM candles_1m
                GROUP BY symbol, time_bucket(INTERVAL '{interval}', ts);
            """)

    def refresh_clean_tables(self) -> None:
        """Materialize resampled views into tables for fast backtest access.

        Creates bars_1h_clean, bars_4h_clean, bars_1d_clean tables.
        """
        for view_name in ["bars_1h", "bars_4h", "bars_1d"]:
            table_name = f"{view_name}_clean"
            self._conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM {view_name} ORDER BY symbol, ts;
            """)
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()
            rows = count[0] if count else 0
            logger.info("Materialized %s: %d rows", table_name, rows)

    # ──────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────

    def discover_products(self) -> list[str]:
        """Fetch all USD spot products from Coinbase, excluding stablecoins.

        Returns:
            Sorted list of product IDs (e.g. ["BTC-USD", "ETH-USD", ...]).
        """
        client = self._get_client()
        self._rate_limiter.wait()
        response = client.get_products(product_type="SPOT", get_all_products=True)

        symbols: list[str] = []
        for product in response.products or []:
            pid = getattr(product, "product_id", None)
            quote = getattr(product, "quote_currency_id", None)
            base = getattr(product, "base_currency_id", None)

            if not pid or not quote or not base:
                continue
            if quote != "USD":
                continue
            if base.upper() in EXCLUDED_BASES:
                continue

            symbols.append(pid)

        symbols = sorted(set(symbols))
        logger.info("Discovered %d USD spot products", len(symbols))
        return symbols

    # ──────────────────────────────────────────────
    # ADV Liquidity Scan
    # ──────────────────────────────────────────────

    def compute_adv(
        self,
        symbol: str,
        lookback_days: int = 90,
        *,
        _client: Any = None,
    ) -> float:
        """Compute average daily notional volume for a symbol.

        Fetches daily candles for the last ``lookback_days`` days and returns
        mean(close * volume) as a dollar-denominated ADV.

        Args:
            symbol: Product ID (e.g. "BTC-USD").
            lookback_days: Number of days to look back.
            _client: Override API client (used by parallel workers).

        Returns:
            Average daily notional volume in USD. 0.0 on failure.
        """
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=lookback_days)
            df = self._fetch_candles(
                symbol, start, end, GRANULARITY_1D, _client=_client,
            )
            if df.is_empty():
                return 0.0
            adv = (df["close"] * df["volume"]).mean()
            return float(adv) if adv is not None else 0.0
        except Exception as e:
            logger.debug("%s: ADV scan failed: %s", symbol, e)
            return 0.0

    def discover_top_n(
        self,
        n: int = 200,
        lookback_days: int = 90,
        workers: int = 8,
        min_adv: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Discover top N most liquid USD products by average daily notional.

        Args:
            n: Number of top products to return.
            lookback_days: Days of history for ADV calculation.
            workers: Number of parallel threads for the ADV scan.
            min_adv: Minimum ADV threshold (USD) to include.

        Returns:
            List of (symbol, adv) tuples sorted by ADV descending.
        """
        candidates = self.discover_products()
        logger.info(
            "Scanning %d USD pairs for %d-day ADV (workers=%d)...",
            len(candidates), lookback_days, workers,
        )

        adv_map: dict[str, float] = {}

        if workers <= 1:
            for i, sym in enumerate(candidates, 1):
                adv_map[sym] = self.compute_adv(sym, lookback_days)
                if i % 20 == 0:
                    logger.info("  ADV scan: %d/%d", i, len(candidates))
        else:
            # Each thread gets its own API client for thread safety
            _thread_clients: dict[int, Any] = {}
            _clients_lock = threading.Lock()

            def _get_thread_client() -> Any:
                tid = threading.get_ident()
                if tid not in _thread_clients:
                    with _clients_lock:
                        if tid not in _thread_clients:
                            _thread_clients[tid] = self._create_client()
                return _thread_clients[tid]

            def _scan_one(sym: str) -> tuple[str, float]:
                client = _get_thread_client()
                adv = self.compute_adv(sym, lookback_days, _client=client)
                return sym, adv

            done = 0
            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = {exe.submit(_scan_one, sym): sym for sym in candidates}
                for future in as_completed(futures):
                    sym, adv = future.result()
                    adv_map[sym] = adv
                    done += 1
                    if done % 20 == 0:
                        logger.info("  ADV scan: %d/%d", done, len(candidates))

        # Filter by min_adv and sort
        pairs = [(sym, adv) for sym, adv in adv_map.items() if adv >= min_adv]
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:n]

        if top:
            logger.info(
                "Top %d selected from %d candidates (min ADV cutoff: $%.0f/day)",
                len(top), len(candidates), top[-1][1] if top else 0,
            )
            logger.info(
                "  #1 %s: $%,.0f/day | #%d %s: $%,.0f/day",
                top[0][0], top[0][1], len(top), top[-1][0], top[-1][1],
            )
        else:
            logger.warning("No symbols passed ADV filter (min_adv=$%.0f)", min_adv)

        return top

    # ──────────────────────────────────────────────
    # Backfill
    # ──────────────────────────────────────────────

    def backfill_symbol(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        granularity: str = GRANULARITY_1M,
        *,
        _client: Any = None,
        _conn: duckdb.DuckDBPyConnection | None = None,
    ) -> int:
        """Fetch full history for one symbol.

        If ``start`` is None, auto-discovers the earliest available candle
        via binary search. If data already exists, resumes from the last
        known timestamp.

        Args:
            symbol: Product ID (e.g. "BTC-USD").
            start: Start datetime (UTC). None = auto-discover.
            end: End datetime (UTC). None = now.
            granularity: Coinbase granularity string.
            _client: Override API client (used by parallel workers).
            _conn: Override DuckDB connection (used by parallel workers).

        Returns:
            Total number of rows inserted.
        """
        conn = _conn or self._conn
        if end is None:
            end = datetime.now(timezone.utc)

        # Check for existing data to resume from
        resume_ts = self._get_last_timestamp(symbol, _conn=conn)
        if resume_ts is not None:
            # Resume from 1 minute after last known candle
            effective_start = resume_ts + timedelta(minutes=1)
            if effective_start >= end:
                logger.info("%s: already up to date (last=%s)", symbol, resume_ts)
                return 0
            logger.info("%s: resuming from %s", symbol, effective_start.isoformat())
        elif start is not None:
            effective_start = start
        else:
            # Auto-discover earliest available candle
            logger.info("%s: searching for earliest available data...", symbol)
            earliest = self._find_earliest_candle(
                symbol, granularity, _client=_client,
            )
            if earliest is None:
                logger.warning("%s: no data available", symbol)
                return 0
            effective_start = earliest
            logger.info("%s: earliest candle at %s", symbol, earliest.isoformat())

        return self._fetch_range(
            symbol, effective_start, end, granularity,
            _client=_client, _conn=conn,
        )

    def backfill_all(
        self,
        min_history_days: int = 0,
        symbols: list[str] | None = None,
        workers: int = 1,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, int]:
        """Backfill all discovered USD products.

        Args:
            min_history_days: Skip symbols with less than this many days of history.
            symbols: Override product list (default: discover from API).
            workers: Number of parallel threads (default 1 = sequential).
            start: Start datetime override for all symbols.
            end: End datetime override for all symbols.

        Returns:
            Dict of {symbol: rows_inserted}.
        """
        if symbols is None:
            symbols = self.discover_products()

        total = len(symbols)
        logger.info("Backfilling %d symbols (workers=%d)...", total, workers)

        if workers <= 1:
            return self._backfill_sequential(symbols, min_history_days, start, end)
        else:
            return self._backfill_parallel(
                symbols, min_history_days, workers, start, end,
            )

    def _backfill_sequential(
        self,
        symbols: list[str],
        min_history_days: int,
        start: datetime | None,
        end: datetime | None,
    ) -> dict[str, int]:
        """Sequential backfill (original behavior)."""
        results: dict[str, int] = {}
        cutoff = datetime.now(timezone.utc) - timedelta(days=min_history_days)
        total = len(symbols)

        for i, symbol in enumerate(symbols, 1):
            logger.info("[%d/%d] Backfilling %s...", i, total, symbol)
            try:
                rows = self.backfill_symbol(
                    symbol, start=start, end=end or datetime.now(timezone.utc),
                )
                results[symbol] = rows

                if min_history_days > 0 and rows > 0:
                    first_ts = self._get_first_timestamp(symbol)
                    if first_ts and first_ts > cutoff:
                        logger.info(
                            "%s: only %d days of history (need %d)",
                            symbol,
                            (datetime.now(timezone.utc) - first_ts).days,
                            min_history_days,
                        )
            except Exception as e:
                logger.error("%s: backfill failed: %s", symbol, e)
                results[symbol] = 0

        success = sum(1 for v in results.values() if v > 0)
        logger.info(
            "Backfill complete: %d/%d symbols with data, %d total rows",
            success, total, sum(results.values()),
        )
        return results

    def _backfill_parallel(
        self,
        symbols: list[str],
        min_history_days: int,
        workers: int,
        start: datetime | None,
        end: datetime | None,
    ) -> dict[str, int]:
        """Parallel backfill using thread pool.

        Each worker gets its own API client and DuckDB connection.
        """
        _thread_clients: dict[int, Any] = {}
        _thread_conns: dict[int, duckdb.DuckDBPyConnection] = {}
        _resources_lock = threading.Lock()

        def _get_thread_resources() -> tuple[Any, duckdb.DuckDBPyConnection]:
            tid = threading.get_ident()
            if tid not in _thread_clients:
                with _resources_lock:
                    if tid not in _thread_clients:
                        _thread_clients[tid] = self._create_client()
                        _thread_conns[tid] = self._open_conn()
            return _thread_clients[tid], _thread_conns[tid]

        def _worker(symbol: str) -> tuple[str, int]:
            client, conn = _get_thread_resources()
            try:
                rows = self.backfill_symbol(
                    symbol,
                    start=start,
                    end=end or datetime.now(timezone.utc),
                    _client=client,
                    _conn=conn,
                )
                return symbol, rows
            except Exception as e:
                logger.error("%s: backfill failed: %s", symbol, e)
                return symbol, 0

        results: dict[str, int] = {}
        done = 0
        total = len(symbols)

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(_worker, sym): sym for sym in symbols}
            for future in as_completed(futures):
                symbol, rows = future.result()
                results[symbol] = rows
                done += 1
                if done % 5 == 0 or done == total:
                    logger.info(
                        "Backfill progress: %d/%d symbols done", done, total,
                    )

        # Clean up worker connections
        for conn in _thread_conns.values():
            try:
                conn.close()
            except Exception:
                pass

        success = sum(1 for v in results.values() if v > 0)
        total_rows = sum(results.values())
        logger.info(
            "Backfill complete: %d/%d symbols with data, %d total rows",
            success, total, total_rows,
        )
        return results

    # ──────────────────────────────────────────────
    # Incremental Update
    # ──────────────────────────────────────────────

    def update_symbol(
        self,
        symbol: str,
        *,
        _client: Any = None,
        _conn: duckdb.DuckDBPyConnection | None = None,
    ) -> int:
        """Fetch new candles from last known timestamp to now.

        Returns:
            Number of rows inserted.
        """
        conn = _conn or self._conn
        last_ts = self._get_last_timestamp(symbol, _conn=conn)
        if last_ts is None:
            logger.warning("%s: no existing data, use backfill_symbol() first", symbol)
            return 0

        start = last_ts + timedelta(minutes=1)
        end = datetime.now(timezone.utc)

        if start >= end:
            logger.info("%s: already up to date", symbol)
            return 0

        return self._fetch_range(
            symbol, start, end, GRANULARITY_1M,
            _client=_client, _conn=conn,
        )

    def update_all(self, workers: int = 1) -> dict[str, int]:
        """Update all symbols that have existing data in the DB.

        Args:
            workers: Number of parallel threads (default 1 = sequential).

        Returns:
            Dict of {symbol: rows_inserted}.
        """
        symbols = self._get_symbols_in_db()
        if not symbols:
            logger.warning("No symbols in database. Run backfill first.")
            return {}

        logger.info("Updating %d symbols (workers=%d)...", len(symbols), workers)

        if workers <= 1:
            results: dict[str, int] = {}
            for i, symbol in enumerate(symbols, 1):
                logger.info("[%d/%d] Updating %s...", i, len(symbols), symbol)
                try:
                    results[symbol] = self.update_symbol(symbol)
                except Exception as e:
                    logger.error("%s: update failed: %s", symbol, e)
                    results[symbol] = 0
        else:
            _thread_clients: dict[int, Any] = {}
            _thread_conns: dict[int, duckdb.DuckDBPyConnection] = {}
            _resources_lock = threading.Lock()

            def _get_thread_resources() -> tuple[Any, duckdb.DuckDBPyConnection]:
                tid = threading.get_ident()
                if tid not in _thread_clients:
                    with _resources_lock:
                        if tid not in _thread_clients:
                            _thread_clients[tid] = self._create_client()
                            _thread_conns[tid] = self._open_conn()
                return _thread_clients[tid], _thread_conns[tid]

            def _worker(symbol: str) -> tuple[str, int]:
                client, conn = _get_thread_resources()
                try:
                    return symbol, self.update_symbol(
                        symbol, _client=client, _conn=conn,
                    )
                except Exception as e:
                    logger.error("%s: update failed: %s", symbol, e)
                    return symbol, 0

            results = {}
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = {exe.submit(_worker, sym): sym for sym in symbols}
                for future in as_completed(futures):
                    symbol, rows = future.result()
                    results[symbol] = rows
                    done += 1
                    if done % 10 == 0 or done == len(symbols):
                        logger.info(
                            "Update progress: %d/%d symbols done", done, len(symbols),
                        )

            for conn in _thread_conns.values():
                try:
                    conn.close()
                except Exception:
                    pass

        total_rows = sum(results.values())
        logger.info("Update complete: %d symbols, %d new rows", len(symbols), total_rows)
        return results

    # ──────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────

    def status(self) -> list[dict[str, Any]]:
        """Get summary of all symbols in the database.

        Returns:
            List of dicts with symbol, row_count, first_ts, last_ts, gap_hours.
        """
        result = self._conn.execute("""
            SELECT
                symbol,
                COUNT(*)                                    AS row_count,
                MIN(ts)                                     AS first_ts,
                MAX(ts)                                     AS last_ts,
                ROUND(
                    EXTRACT(EPOCH FROM (NOW() - MAX(ts))) / 3600.0, 1
                )                                           AS hours_behind
            FROM candles_1m
            GROUP BY symbol
            ORDER BY symbol;
        """).fetchall()

        rows: list[dict[str, Any]] = []
        for r in result:
            rows.append({
                "symbol": r[0],
                "row_count": r[1],
                "first_ts": r[2],
                "last_ts": r[3],
                "hours_behind": r[4],
            })
        return rows

    # ──────────────────────────────────────────────
    # Internal: API Interaction
    # ──────────────────────────────────────────────

    def _fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        granularity: str = GRANULARITY_1M,
        *,
        _client: Any = None,
    ) -> pl.DataFrame:
        """Fetch one window of candles from the Coinbase API.

        Returns a Polars DataFrame with columns:
            ts (datetime UTC), symbol, open, high, low, close, volume
        """
        client = _client or self._get_client()
        self._rate_limiter.wait()

        start_unix = str(int(start.timestamp()))
        end_unix = str(int(end.timestamp()))

        _empty_schema = {
            "ts": pl.Datetime(time_zone="UTC"),
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }

        try:
            response = client.get_candles(
                product_id=symbol,
                start=start_unix,
                end=end_unix,
                granularity=granularity,
            )
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                logger.warning("%s: rate limited, backing off 5s", symbol)
                time.sleep(5)
                self._rate_limiter.wait()
                response = client.get_candles(
                    product_id=symbol,
                    start=start_unix,
                    end=end_unix,
                    granularity=granularity,
                )
            else:
                raise

        candles = getattr(response, "candles", None) or []
        if not candles:
            return pl.DataFrame(schema=_empty_schema)

        records: list[dict[str, Any]] = []
        for c in candles:
            ts_val = getattr(c, "start", None)
            if ts_val is None:
                continue
            records.append({
                "ts": datetime.fromtimestamp(int(ts_val), tz=timezone.utc),
                "symbol": symbol,
                "open": float(getattr(c, "open", 0)),
                "high": float(getattr(c, "high", 0)),
                "low": float(getattr(c, "low", 0)),
                "close": float(getattr(c, "close", 0)),
                "volume": float(getattr(c, "volume", 0)),
            })

        if not records:
            return pl.DataFrame(schema=_empty_schema)

        return pl.DataFrame(records).sort("ts")

    def _fetch_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        granularity: str,
        *,
        _client: Any = None,
        _conn: duckdb.DuckDBPyConnection | None = None,
    ) -> int:
        """Fetch candles for a date range in windows, upserting into DuckDB.

        Returns total rows inserted.
        """
        conn = _conn or self._conn

        # Window size depends on granularity
        if granularity == GRANULARITY_1M:
            window = timedelta(minutes=WINDOW_MINUTES_1M)
        elif granularity == GRANULARITY_1H:
            window = timedelta(hours=MAX_CANDLES_PER_REQUEST)
        elif granularity == GRANULARITY_1D:
            window = timedelta(days=MAX_CANDLES_PER_REQUEST)
        else:
            window = timedelta(minutes=WINDOW_MINUTES_1M)

        total_rows = 0
        current_start = start
        window_count = 0
        total_windows = max(1, math.ceil((end - start) / window))

        while current_start < end:
            current_end = min(current_start + window, end)
            window_count += 1

            df = self._fetch_candles(
                symbol, current_start, current_end, granularity,
                _client=_client,
            )

            if not df.is_empty():
                # Ensure bounds are UTC for Polars comparison
                _start_utc = current_start.astimezone(timezone.utc) if current_start.tzinfo else current_start.replace(tzinfo=timezone.utc)
                _end_utc = current_end.astimezone(timezone.utc) if current_end.tzinfo else current_end.replace(tzinfo=timezone.utc)
                # Filter to exact window bounds
                df = df.filter(
                    (pl.col("ts") >= _start_utc) & (pl.col("ts") < _end_utc)
                ).unique(subset=["ts"])

                if not df.is_empty():
                    self._upsert_candles(df, _conn=conn)
                    total_rows += df.height

            # Progress logging
            if window_count % 50 == 0 or current_end >= end:
                pct = min(100.0, window_count / total_windows * 100)
                logger.info(
                    "%s: window %d/%d (%.0f%%), %d rows so far",
                    symbol, window_count, total_windows, pct, total_rows,
                )

            current_start = current_end

        if total_rows > 0:
            logger.info("%s: inserted %d rows", symbol, total_rows)

        return total_rows

    def _upsert_candles(
        self,
        df: pl.DataFrame,
        *,
        _conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Insert candles into candles_1m, handling duplicates via DELETE+INSERT."""
        if df.is_empty():
            return

        conn = _conn or self._conn
        sym = df["symbol"][0]
        ts_min = df["ts"].min()
        ts_max = df["ts"].max()

        # Delete existing rows in this range to avoid duplicates
        conn.execute(
            """
            DELETE FROM candles_1m
            WHERE symbol = ?
              AND ts >= ?
              AND ts <= ?;
            """,
            [sym, ts_min, ts_max],
        )

        # Insert new rows
        conn.register("_tmp_candles", df)
        conn.execute("""
            INSERT INTO candles_1m (symbol, ts, open, high, low, close, volume)
            SELECT symbol, ts, open, high, low, close, volume
            FROM _tmp_candles;
        """)
        conn.unregister("_tmp_candles")

    def _find_earliest_candle(
        self,
        symbol: str,
        granularity: str = GRANULARITY_1M,
        *,
        _client: Any = None,
    ) -> datetime | None:
        """Binary search for the earliest available candle.

        Probes forward from 2015-01-01 in 90-day jumps, then binary searches
        to narrow down to the exact first candle.
        """
        anchor = datetime(2015, 1, 1, tzinfo=timezone.utc)
        end = datetime.now(timezone.utc)
        coarse_step = timedelta(days=90)

        # Phase 1: Coarse scan — find first non-empty region
        probe_start = anchor
        found_region: datetime | None = None

        while probe_start < end:
            probe_end = min(probe_start + timedelta(hours=5), end)
            df = self._fetch_candles(
                symbol, probe_start, probe_end, granularity,
                _client=_client,
            )

            if not df.is_empty():
                found_region = probe_start
                break

            probe_start = probe_start + coarse_step

        if found_region is None:
            return None

        # Phase 2: Binary search for exact earliest candle
        lo = max(anchor, found_region - coarse_step)
        hi = found_region

        while (hi - lo) > timedelta(hours=5):
            mid = lo + (hi - lo) / 2
            # Snap to minute boundary
            mid = mid.replace(second=0, microsecond=0)
            mid_end = min(mid + timedelta(hours=5), end)

            df = self._fetch_candles(
                symbol, mid, mid_end, granularity,
                _client=_client,
            )

            if df.is_empty():
                lo = mid
            else:
                hi = mid

        # Fetch the final window to get the exact first timestamp
        df = self._fetch_candles(
            symbol, hi, min(hi + timedelta(hours=5), end), granularity,
            _client=_client,
        )
        if df.is_empty():
            return found_region

        return df["ts"].min()  # type: ignore[return-value]

    # ──────────────────────────────────────────────
    # Internal: DuckDB Queries
    # ──────────────────────────────────────────────

    def _get_last_timestamp(
        self,
        symbol: str,
        *,
        _conn: duckdb.DuckDBPyConnection | None = None,
    ) -> datetime | None:
        """Get the most recent timestamp for a symbol."""
        conn = _conn or self._conn
        result = conn.execute(
            "SELECT MAX(ts) FROM candles_1m WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result and result[0] is not None:
            ts: datetime = result[0]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            return ts
        return None

    def _get_first_timestamp(
        self,
        symbol: str,
        *,
        _conn: duckdb.DuckDBPyConnection | None = None,
    ) -> datetime | None:
        """Get the earliest timestamp for a symbol."""
        conn = _conn or self._conn
        result = conn.execute(
            "SELECT MIN(ts) FROM candles_1m WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result and result[0] is not None:
            ts: datetime = result[0]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            return ts
        return None

    def _get_symbols_in_db(self) -> list[str]:
        """Get all symbols that have data in the database."""
        result = self._conn.execute(
            "SELECT DISTINCT symbol FROM candles_1m ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in result]

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()
