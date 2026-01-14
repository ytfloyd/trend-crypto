#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import datetime as dt
import os
import time
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl
import requests

from data.coinbase_usd_universe import get_usd_spot_universe_ex_stables

BASE_URL = "https://api.exchange.coinbase.com"
EXCHANGE_BASE = "https://api.exchange.coinbase.com"
HOURLY_GRANULARITY = 3600
WINDOW_SECONDS = 300 * HOURLY_GRANULARITY  # Coinbase limit 300 candles
DEFAULT_DB_PATH = Path("data/market.duckdb")


class RateLimiter:
    def __init__(self, max_rps: float) -> None:
        self.period = 1.0 / max_rps if max_rps > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.period <= 0:
            return
        now = time.time()
        sleep_for = max(0.0, self._last + self.period - now)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.time()


def resolve_db_path(args: argparse.Namespace) -> Path:
    if getattr(args, "db", None):
        db_path = Path(args.db)
    else:
        env_path = os.getenv("TREND_CRYPTO_DUCKDB_PATH")
        if env_path:
            db_path = Path(env_path)
        else:
            db_path = DEFAULT_DB_PATH
    db_path = db_path.expanduser().resolve()
    if not db_path.exists():
        candidates = list(Path("data").glob("*.duckdb")) if Path("data").exists() else []
        msg = [
            f"[sync_coinbase_usd_universe_hourly] ERROR: DuckDB file not found at: {db_path}",
            "",
            "Hints:",
            "- set TREND_CRYPTO_DUCKDB_PATH=/path/to/your.duckdb, or",
            "  pass --db /path/to/your.duckdb, or",
            "  symlink/copy it to data/market.duckdb",
        ]
        if candidates:
            msg.append("")
            msg.append("Discovered .duckdb files under ./data:")
            for c in candidates:
                msg.append(f"  - {c.resolve()}")
        raise FileNotFoundError("\n".join(msg))
    return db_path


def request_with_retry(
    session: requests.Session,
    url: str,
    params: dict,
    *,
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> requests.Response:
    for attempt in range(max_retries + 1):
        rate_limiter.wait()
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(float(retry_after))
                except ValueError:
                    pass
            else:
                backoff = min(60, 2**attempt)
                time.sleep(backoff)
            continue
        resp.raise_for_status()
    resp.raise_for_status()
    return resp


def fetch_hourly_candles(
    session: requests.Session,
    product_id: str,
    start_iso: str,
    end_iso: str,
    timeout: float = 10.0,
) -> list[dict]:
    """
    Fetch hourly OHLCV candles from Coinbase Exchange public market-data API.

    - product_id example: "SOL-USD"
    - granularity: 3600 seconds (1 hour)
    """
    url = f"{EXCHANGE_BASE}/products/{product_id}/candles"
    params = {
        "start": start_iso,
        "end": end_iso,
        "granularity": 3600,
    }
    data = request_with_retry(
        session,
        url,
        params,
        timeout=timeout,
        max_retries=3,
        rate_limiter=RateLimiter(0),  # already rate-limited outside
    ).json()

    candles = []
    if not data:
        return candles

    for t, low, high, open_, close, volume in sorted(data, key=lambda row: row[0]):
        ts = datetime.datetime.utcfromtimestamp(t).replace(tzinfo=datetime.timezone.utc)
        candles.append(
            {
                "ts": ts,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
        )
    return candles


def fetch_candles_chunk(
    session: requests.Session,
    product_id: str,
    start: dt.datetime,
    end: dt.datetime,
    granularity: int,
    *,
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> pl.DataFrame:
    # Keep window logic; actual fetch now uses Exchange public endpoint
    candles = fetch_hourly_candles(
        session,
        product_id=product_id,
        start_iso=start.isoformat(),
        end_iso=end.isoformat(),
        timeout=timeout,
    )
    if not candles:
        return pl.DataFrame(
            schema={
                "time": pl.Datetime(time_zone="UTC"),
                "low": pl.Float64,
                "high": pl.Float64,
                "open": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    df = pl.DataFrame(candles).rename({"ts": "time"})
    return df.sort("time")


def ensure_candles_table_no_pk(conn: duckdb.DuckDBPyConnection) -> None:
    pk_exists = False
    try:
        res = conn.execute(
            """
            SELECT 1
            FROM information_schema.table_constraints
            WHERE table_name = 'candles'
              AND constraint_type = 'PRIMARY KEY'
            LIMIT 1;
            """
        ).fetchone()
        pk_exists = bool(res and res[0] == 1)
    except Exception:
        pk_exists = False

    if pk_exists:
        conn.execute(
            """
            CREATE TABLE candles_new (
                product_id TEXT NOT NULL,
                time TIMESTAMP NOT NULL,
                low DOUBLE,
                high DOUBLE,
                open DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                granularity INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO candles_new
            SELECT product_id, time, low, high, open, close, volume, granularity
            FROM candles;
            """
        )
        conn.execute("DROP TABLE candles;")
        conn.execute("ALTER TABLE candles_new RENAME TO candles;")


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            product_id TEXT NOT NULL,
            time TIMESTAMP NOT NULL,
            low DOUBLE,
            high DOUBLE,
            open DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            granularity INTEGER NOT NULL
        );
        """
    )


def create_view(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE OR REPLACE VIEW hourly_bars AS
        SELECT
          product_id AS symbol,
          time AS ts,
          ANY_VALUE(open) AS open,
          ANY_VALUE(high) AS high,
          ANY_VALUE(low) AS low,
          ANY_VALUE(close) AS close,
          ANY_VALUE(volume) AS volume
        FROM candles
        WHERE granularity = 3600
        GROUP BY product_id, time;
        """
    )


def iter_windows(start: dt.datetime, end: dt.datetime, granularity: int):
    window = dt.timedelta(seconds=WINDOW_SECONDS)
    overlap = dt.timedelta(seconds=granularity)
    cur = start
    while cur < end:
        nxt = min(cur + window, end)
        yield cur, nxt
        cur = nxt - overlap


def ingest_symbol(
    conn: duckdb.DuckDBPyConnection,
    session: requests.Session,
    symbol: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    *,
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> int:
    total_rows = 0
    for w_start, w_end in iter_windows(start_dt, end_dt, HOURLY_GRANULARITY):
        df_chunk = fetch_candles_chunk(
            session,
            symbol,
            w_start,
            w_end,
            HOURLY_GRANULARITY,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
        if df_chunk.is_empty():
            continue

        w_start_lit = pl.lit(w_start).cast(pl.Datetime(time_unit="us", time_zone="UTC"))
        w_end_lit = pl.lit(w_end).cast(pl.Datetime(time_unit="us", time_zone="UTC"))

        df_chunk = (
            df_chunk.unique(subset=["time"])
            .filter((pl.col("time") >= w_start_lit) & (pl.col("time") < w_end_lit))
        )

        df_insert = (
            df_chunk.with_columns(
                [
                    pl.lit(symbol).alias("product_id"),
                    pl.lit(HOURLY_GRANULARITY).alias("granularity"),
                ]
            )[
                ["product_id", "time", "low", "high", "open", "close", "volume", "granularity"]
            ]
        )
        df_insert = df_insert.unique(subset=["product_id", "time", "granularity"], keep="first")

        if df_insert.height == 0:
            continue

        tmin = df_insert["time"].min()
        tmax = df_insert["time"].max()
        conn.execute(
            """
            DELETE FROM candles
            WHERE product_id = ?
              AND granularity = ?
              AND time >= ?
              AND time <= ?;
            """,
            [symbol, HOURLY_GRANULARITY, tmin, tmax],
        )
        conn.register("tmp_candles", df_insert)
        conn.execute(
            """
            INSERT INTO candles
            SELECT product_id, time, low, high, open, close, volume, granularity
            FROM tmp_candles;
            """
        )
        conn.unregister("tmp_candles")

        total_rows += df_insert.height

    return total_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync Coinbase USD spot universe (ex-stablecoins) into hourly_bars."
    )
    parser.add_argument("--db", type=str, default=None, help="Path to DuckDB database.")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="ISO8601 start (UTC). If omitted, will use existing max(ts)+1h per symbol or require start.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="ISO8601 end (UTC). Defaults to now (floored to hour).",
    )
    parser.add_argument("--max_parallel", type=int, default=1, help="Concurrency (currently sequential).")
    parser.add_argument("--max_rps", type=float, default=5.0, help="Max requests per second.")
    parser.add_argument("--max_retries", type=int, default=5, help="Max HTTP retries.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds.")
    args = parser.parse_args()

    db_path = resolve_db_path(args)
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    end_dt = (
        dt.datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        if args.end
        else now
    )

    session = requests.Session()
    rate_limiter = RateLimiter(args.max_rps)

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("SET TimeZone='UTC';")
        ensure_candles_table_no_pk(conn)
        ensure_schema(conn)

        universe = get_usd_spot_universe_ex_stables(session)
        if not universe:
            raise RuntimeError("No USD spot symbols discovered from Coinbase Advanced.")
        print(
            f"[sync_coinbase_usd_universe_hourly] Using DuckDB: {db_path} | symbols={len(universe)} | sample={universe[:5]}"
        )

        total_rows = 0
        for symbol in universe:
            res = conn.execute(
                """
                SELECT max(time) FROM candles WHERE product_id = ? AND granularity = ?
                """,
                [symbol, HOURLY_GRANULARITY],
            ).fetchone()
            last_ts = res[0] if res and res[0] is not None else None

            if last_ts is None:
                if not args.start:
                    raise RuntimeError(f"No existing data for {symbol} and no --start provided.")
                start_dt = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(
                    dt.timezone.utc
                )
            else:
                start_dt = last_ts.replace(tzinfo=dt.timezone.utc) + dt.timedelta(seconds=HOURLY_GRANULARITY)

            if start_dt >= end_dt:
                print(f"[sync_coinbase_usd_universe_hourly] {symbol}: up to date (start>=end).")
                continue

            print(
                f"[sync_coinbase_usd_universe_hourly] {symbol}: ingesting {start_dt.isoformat()} -> {end_dt.isoformat()}"
            )
            n_rows = ingest_symbol(
                conn,
                session,
                symbol,
                start_dt,
                end_dt,
                timeout=args.timeout,
                max_retries=args.max_retries,
                rate_limiter=rate_limiter,
            )
            total_rows += n_rows
            print(f"[sync_coinbase_usd_universe_hourly] {symbol}: inserted {n_rows} rows")

        create_view(conn)
        print(f"[sync_coinbase_usd_universe_hourly] DONE symbols={len(universe)} rows_inserted={total_rows}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

