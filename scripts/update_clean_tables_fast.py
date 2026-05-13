#!/usr/bin/env python3
"""
Fast daily/hourly bar refresh for the crypto data lake.

Skips the 1-minute granularity entirely and directly updates the
``bars_1d_clean`` / ``bars_1h_clean`` materialized tables by fetching
daily and hourly candles from Coinbase. This is *much* faster than
filling 1m candles for the same time range:

    Daily:  300 candles = 300 days  → 1 request covers ~54 days × 850 syms
    Hourly: 300 candles ≈ 12.5 days → ~5 requests per symbol for 54 days
    1m:     300 candles = 5 hours   → ~260 requests per symbol for 54 days

Backtests (Turtle, Relative Strength, etc.) only consume the daily/hourly
clean tables, so this brings them to current with minimal API quota.

The 1-minute raw table can be back-filled later via:

    python scripts/collect_coinbase.py update --workers 8 --max-rps 10

Usage:

    python scripts/update_clean_tables_fast.py \
        --db /Users/russellfloyd/Dropbox/NRT/nrt_dev/data/market.duckdb \
        --quotes USD,USDC \
        --workers 6 \
        --max-rps 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("update_fast")


def _load_creds() -> tuple[str, str]:
    """Read the live CDP key file (preferred) or fall back to env."""
    key_path = "/Users/russellfloyd/.secrets/cdp_api_key.json"
    if os.path.exists(key_path):
        with open(key_path) as f:
            k = json.load(f)
        return k["name"], k["privateKey"]
    return os.environ["COINBASE_API_KEY"], os.environ["COINBASE_API_SECRET"]


class FastUpdater:
    GRAN = {
        "1d": ("ONE_DAY",  86_400),
        "1h": ("ONE_HOUR",  3_600),
    }
    MAX_CANDLES = 300

    def __init__(self, db_path: str, max_rps: float) -> None:
        from data.collector import RateLimiter

        self.db_path = db_path
        self._rate_limiter = RateLimiter(max_rps)
        self._client_lock = threading.Lock()
        self._thread_clients: dict[int, Any] = {}
        self._db_lock = threading.Lock()

    def _client(self) -> Any:
        from coinbase.rest import RESTClient
        tid = threading.get_ident()
        if tid not in self._thread_clients:
            with self._client_lock:
                if tid not in self._thread_clients:
                    name, secret = _load_creds()
                    self._thread_clients[tid] = RESTClient(
                        api_key=name, api_secret=secret,
                    )
        return self._thread_clients[tid]

    def get_target_symbols(self, quotes: set[str] | None) -> list[str]:
        """Symbols already present in candles_1m, optionally filtered by quote."""
        with self._db_lock:
            conn = duckdb.connect(self.db_path)
            try:
                rows = conn.execute(
                    "SELECT DISTINCT symbol FROM candles_1m ORDER BY symbol"
                ).fetchall()
            finally:
                conn.close()
        syms = [r[0] for r in rows]
        if quotes is None:
            return syms
        return [s for s in syms if s.split("-", 1)[-1].upper() in quotes]

    def get_last_ts_bulk(self, table: str) -> dict[str, datetime]:
        """One-shot read of MAX(ts) per symbol — avoids per-worker DB hits."""
        out: dict[str, datetime] = {}
        with self._db_lock:
            conn = duckdb.connect(self.db_path)
            try:
                try:
                    rows = conn.execute(
                        f"SELECT symbol, MAX(ts) FROM {table} GROUP BY symbol"
                    ).fetchall()
                except duckdb.CatalogException:
                    rows = []
            finally:
                conn.close()
        for sym, ts in rows:
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            out[sym] = ts.astimezone(timezone.utc)
        return out

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        granularity: str,
    ) -> list[dict[str, Any]]:
        client = self._client()
        gran_str, gran_sec = self.GRAN[granularity]
        out: list[dict[str, Any]] = []
        cur = start
        while cur < end:
            window_end = min(cur + timedelta(seconds=gran_sec * self.MAX_CANDLES), end)
            self._rate_limiter.wait()
            try:
                resp = client.get_candles(
                    product_id=symbol,
                    start=str(int(cur.timestamp())),
                    end=str(int(window_end.timestamp())),
                    granularity=gran_str,
                )
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "rate" in msg.lower():
                    logger.warning("%s: rate limited, backing off", symbol)
                    time.sleep(5.0)
                    cur = cur  # retry same window
                    continue
                logger.warning("%s [%s]: %s", symbol, granularity, msg[:120])
                return out
            candles = getattr(resp, "candles", None) or []
            for c in candles:
                # Coinbase Candle objects expose attributes; fall back to dict.
                def _g(name: str) -> Any:
                    v = getattr(c, name, None)
                    if v is None and hasattr(c, "__getitem__"):
                        try:
                            v = c[name]
                        except Exception:
                            v = None
                    return v
                ts_val = _g("start")
                if ts_val is None:
                    continue
                out.append({
                    "symbol": symbol,
                    "ts": datetime.fromtimestamp(int(ts_val), tz=timezone.utc),
                    "open":   float(_g("open")),
                    "high":   float(_g("high")),
                    "low":    float(_g("low")),
                    "close":  float(_g("close")),
                    "volume": float(_g("volume")),
                })
            cur = window_end
        return out

    def update_table(
        self,
        table: str,
        symbols: list[str],
        granularity: str,
        workers: int,
        end: datetime | None = None,
    ) -> dict[str, int]:
        end = end or datetime.now(timezone.utc)
        results: dict[str, int] = {}
        last_ts_map = self.get_last_ts_bulk(table)
        gran_sec = self.GRAN[granularity][1]
        lookback_days = 365 if granularity == "1d" else 60

        def _worker(symbol: str) -> tuple[str, int]:
            last = last_ts_map.get(symbol)
            if last is None:
                start = end - timedelta(days=lookback_days)
            else:
                start = last + timedelta(seconds=gran_sec)
            if start >= end:
                return symbol, 0
            rows = self.fetch_candles(symbol, start, end, granularity)
            if not rows:
                return symbol, 0
            self._upsert(table, rows)
            return symbol, len(rows)

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(_worker, s): s for s in symbols}
            done = 0
            for fut in as_completed(futures):
                sym, n = fut.result()
                results[sym] = n
                done += 1
                if done % 25 == 0 or done == len(symbols):
                    logger.info(
                        "%s: %d/%d symbols (%d new rows so far)",
                        table, done, len(symbols), sum(results.values()),
                    )
        return results

    def _upsert(self, table: str, rows: list[dict[str, Any]]) -> None:
        with self._db_lock:
            conn = duckdb.connect(self.db_path)
            try:
                sym = rows[0]["symbol"]
                ts_min = min(r["ts"] for r in rows)
                ts_max = max(r["ts"] for r in rows)
                conn.execute(
                    f"DELETE FROM {table} WHERE symbol = ? AND ts >= ? AND ts <= ?",
                    [sym, ts_min, ts_max],
                )
                conn.executemany(
                    f"INSERT INTO {table} (symbol, ts, open, high, low, close, volume) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (r["symbol"], r["ts"], r["open"], r["high"],
                         r["low"], r["close"], r["volume"])
                        for r in rows
                    ],
                )
            finally:
                conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--quotes", default="USD,USDC",
                   help="Comma-separated quote currencies (default: USD,USDC).")
    p.add_argument("--granularities", default="1d,1h",
                   help="Comma-separated: 1d,1h (default: both).")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-rps", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap symbols (0 = all). For testing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    quotes = {q.strip().upper() for q in args.quotes.split(",")} if args.quotes else None

    upd = FastUpdater(args.db, args.max_rps)
    syms = upd.get_target_symbols(quotes)
    if args.limit:
        syms = syms[: args.limit]
    logger.info("Updating %d symbols (quotes=%s)", len(syms), quotes)

    grans = [g.strip() for g in args.granularities.split(",")]
    table_for = {"1d": "bars_1d_clean", "1h": "bars_1h_clean"}
    grand_total = 0
    for g in grans:
        if g not in table_for:
            logger.warning("unknown granularity %s — skip", g)
            continue
        t0 = time.time()
        results = upd.update_table(
            table=table_for[g], symbols=syms, granularity=g,
            workers=args.workers,
        )
        elapsed = time.time() - t0
        n_rows = sum(results.values())
        n_syms = sum(1 for v in results.values() if v > 0)
        grand_total += n_rows
        logger.info(
            "%s done: %d new rows across %d symbols (%.1fs)",
            table_for[g], n_rows, n_syms, elapsed,
        )

    # Refresh bars_4h_clean from the freshly updated bars_1h_clean
    if "1h" in grans:
        logger.info("Refreshing bars_4h_clean from bars_1h_clean …")
        conn = duckdb.connect(args.db)
        conn.execute("DROP TABLE IF EXISTS bars_4h_clean_tmp")
        conn.execute("""
            CREATE TABLE bars_4h_clean_tmp AS
            SELECT
                symbol,
                time_bucket(INTERVAL '4 hours', ts) AS ts,
                FIRST(open  ORDER BY ts) AS open,
                MAX(high)                AS high,
                MIN(low)                 AS low,
                LAST(close  ORDER BY ts) AS close,
                SUM(volume)              AS volume
            FROM bars_1h_clean
            GROUP BY symbol, time_bucket(INTERVAL '4 hours', ts)
            ORDER BY symbol, ts
        """)
        conn.execute("DROP TABLE IF EXISTS bars_4h_clean")
        conn.execute("ALTER TABLE bars_4h_clean_tmp RENAME TO bars_4h_clean")
        conn.close()
        logger.info("bars_4h_clean refreshed")

    print(f"\nTotal new rows: {grand_total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
