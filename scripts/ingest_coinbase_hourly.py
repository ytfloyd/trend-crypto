from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Optional

import duckdb
import polars as pl
import requests

BASE_URL = "https://api.exchange.coinbase.com"
PRODUCT_ID = "BTC-USD"
HOURLY_GRANULARITY = 3600
WINDOW_SECONDS = 300 * HOURLY_GRANULARITY  # 300-candle limit


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


def parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc)


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def find_earliest_available_hour(
    session: requests.Session,
    product_id: str,
    granularity: int,
    end_dt: datetime,
    *,
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> datetime:
    anchor = datetime(2012, 1, 1, tzinfo=timezone.utc)
    coarse_step = timedelta(days=90)
    probe_window = timedelta(seconds=WINDOW_SECONDS)

    t = anchor
    found = False
    lo = anchor
    hi = end_dt

    while t < end_dt:
        probe_start = t
        probe_end = min(probe_start + probe_window, end_dt)
        try:
            df = fetch_candles_chunk(
                session,
                product_id,
                probe_start,
                probe_end,
                granularity,
                timeout=timeout,
                max_retries=max_retries,
                rate_limiter=rate_limiter,
            )
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code == 400:
                df = pl.DataFrame(
                    schema={
                        "time": pl.Datetime(time_zone="UTC"),
                        "low": pl.Float64,
                        "high": pl.Float64,
                        "open": pl.Float64,
                        "close": pl.Float64,
                        "volume": pl.Float64,
                    }
                )
            else:
                raise
        empty = df.is_empty()
        logging.info(
            "Earliest scan probe: %s -> %s (empty=%s)",
            probe_start.isoformat(),
            probe_end.isoformat(),
            empty,
        )
        if empty:
            t = probe_start + coarse_step
            lo = t
            continue
        # Found non-empty region
        found = True
        hi = probe_start
        lo = max(anchor, probe_start - coarse_step)
        break

    if not found:
        raise RuntimeError("No candles found for BTC-USD in the specified range.")

    while lo < hi:
        mid = floor_to_hour(lo + (hi - lo) / 2)
        if mid >= hi:
            break
        mid_end = min(mid + probe_window, end_dt)
        try:
            df = fetch_candles_chunk(
                session,
                product_id,
                mid,
                mid_end,
                granularity,
                timeout=timeout,
                max_retries=max_retries,
                rate_limiter=rate_limiter,
            )
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code == 400:
                df = pl.DataFrame(
                    schema={
                        "time": pl.Datetime(time_zone="UTC"),
                        "low": pl.Float64,
                        "high": pl.Float64,
                        "open": pl.Float64,
                        "close": pl.Float64,
                        "volume": pl.Float64,
                    }
                )
            else:
                raise
        if df.is_empty():
            lo = mid + timedelta(hours=1)
        else:
            hi = mid

    tight_end = min(hi + probe_window, end_dt)
    try:
        df_final = fetch_candles_chunk(
            session,
            product_id,
            hi,
            tight_end,
            granularity,
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
        )
    except requests.HTTPError as e:
        resp = getattr(e, "response", None)
        if resp is not None and resp.status_code == 400:
            df_final = pl.DataFrame(
                schema={
                    "time": pl.Datetime(time_zone="UTC"),
                    "low": pl.Float64,
                    "high": pl.Float64,
                    "open": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            )
        else:
            raise
    if df_final.is_empty():
        raise RuntimeError("Failed to locate earliest candle despite non-empty probe.")
    earliest = floor_to_hour(df_final["time"].min())
    logging.info("Earliest available candle hour: %s", earliest.isoformat())
    return earliest


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


def fetch_candles_chunk(
    session: requests.Session,
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: int,
    *,
    timeout: float,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> pl.DataFrame:
    url = f"{BASE_URL}/products/{product_id}/candles"
    params = {
        "granularity": granularity,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    resp = request_with_retry(
        session,
        url,
        params,
        timeout=timeout,
        max_retries=max_retries,
        rate_limiter=rate_limiter,
    )
    data = resp.json()
    if not data:
        return pl.DataFrame(
            schema={"time": pl.Datetime, "low": pl.Float64, "high": pl.Float64, "open": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
        )
    # Coinbase returns [time, low, high, open, close, volume]
    records = []
    for row in data:
        ts = datetime.fromtimestamp(row[0], tz=timezone.utc)
        records.append(
            {
                "time": ts,
                "low": float(row[1]),
                "high": float(row[2]),
                "open": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    return pl.DataFrame(records).sort("time")


def ensure_candles_table_no_pk(conn: duckdb.DuckDBPyConnection) -> None:
    # Check for existing primary key on candles
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
        logging.info("Migrating candles table to remove PRIMARY KEY")
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


def validate_canonical(df: pl.DataFrame) -> None:
    required = ["ts", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.is_empty():
        raise ValueError("No data to validate")
    if not df["ts"].is_sorted():
        raise ValueError("ts not sorted ascending")
    if df["ts"].n_unique() != df.height:
        raise ValueError("Duplicate ts detected")
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            raise ValueError(f"Non-positive price in {col}")
    if (df["volume"] < 0).any():
        raise ValueError("Negative volume detected")


def iter_windows(start: datetime, end: datetime, granularity: int) -> Iterable[tuple[datetime, datetime]]:
    window = timedelta(seconds=WINDOW_SECONDS)
    cur = start
    overlap = timedelta(seconds=granularity)  # 1-bar overlap
    while cur < end:
        nxt = min(cur + window, end)
        yield cur, nxt
        cur = nxt - overlap


def load_resume_start(conn: duckdb.DuckDBPyConnection, product_id: str, granularity: int) -> Optional[datetime]:
    res = conn.execute(
        "SELECT max(time) FROM candles WHERE product_id = ? AND granularity = ?",
        [product_id, granularity],
    ).fetchone()
    if res and res[0] is not None:
        return res[0].replace(tzinfo=timezone.utc)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Coinbase hourly candles into DuckDB and Parquet.")
    parser.add_argument("--symbol", default=PRODUCT_ID, help="Product ID (e.g., BTC-USD, ETH-USD)")
    parser.add_argument("--start", help="ISO8601 start (UTC). Required unless --resume and data exists.")
    parser.add_argument("--end", help="ISO8601 end (UTC). Defaults to now.")
    parser.add_argument("--db", default="data/market.duckdb")
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--granularity", type=int, default=HOURLY_GRANULARITY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-rps", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    session = requests.Session()
    rate_limiter = RateLimiter(args.max_rps)

    end_dt = parse_dt(args.end) if args.end else datetime.now(timezone.utc)
    symbol = args.symbol
    parquet_path = args.parquet or f"data/curated/hourly/{symbol}.parquet"

    conn = duckdb.connect(args.db)
    ensure_candles_table_no_pk(conn)
    ensure_schema(conn)

    start_dt: Optional[datetime] = None
    if args.resume:
        resume_start = load_resume_start(conn, symbol, args.granularity)
        if resume_start is not None:
            start_dt = resume_start + timedelta(seconds=args.granularity)
            logging.info("Resuming from %s", start_dt.isoformat())

    if start_dt is None:
        if args.start and args.start.lower() == "earliest":
            start_dt = find_earliest_available_hour(
                session,
                symbol,
                args.granularity,
                end_dt,
                timeout=args.timeout,
                max_retries=args.max_retries,
                rate_limiter=rate_limiter,
            )
        elif args.start:
            start_dt = parse_dt(args.start)
        else:
            logging.error("--start is required when no existing data and --resume not available")
            return 1

    if start_dt >= end_dt:
        logging.error("start must be before end")
        return 1

    start_dt = floor_to_hour(start_dt)
    end_dt = floor_to_hour(end_dt)

    total_rows = 0
    earliest_ts: Optional[datetime] = None
    latest_ts: Optional[datetime] = None

    window_delta = timedelta(seconds=WINDOW_SECONDS)
    w_start = start_dt
    while True:
        w_end = min(w_start + window_delta, end_dt)
        is_final_window = w_end >= end_dt

        logging.info("Fetching %s to %s", w_start.isoformat(), w_end.isoformat())
        df_chunk = fetch_candles_chunk(
            session,
            symbol,
            w_start,
            w_end,
            args.granularity,
            timeout=args.timeout,
            max_retries=args.max_retries,
            rate_limiter=rate_limiter,
        )
        if df_chunk.is_empty():
            if is_final_window:
                break
            w_start = w_end
            continue

        w_start_lit = pl.lit(w_start).cast(pl.Datetime(time_unit="us", time_zone="UTC"))
        w_end_lit = pl.lit(w_end).cast(pl.Datetime(time_unit="us", time_zone="UTC"))

        df_chunk = (
            df_chunk.unique(subset=["time"])
            .filter((pl.col("time") >= w_start_lit) & (pl.col("time") < w_end_lit))
        )
        total_rows += df_chunk.height
        earliest_ts = df_chunk["time"].min() if earliest_ts is None else min(earliest_ts, df_chunk["time"].min())
        latest_ts = df_chunk["time"].max() if latest_ts is None else max(latest_ts, df_chunk["time"].max())

        df_insert = df_chunk.with_columns(
            [
                pl.lit(symbol).alias("product_id"),
                pl.lit(args.granularity).alias("granularity"),
            ]
        )[
            ["product_id", "time", "low", "high", "open", "close", "volume", "granularity"]
        ]
        df_insert = df_insert.filter((pl.col("time") >= w_start_lit) & (pl.col("time") < w_end_lit))
        df_insert = df_insert.unique(subset=["product_id", "time", "granularity"], keep="first")
        n = df_insert.height
        nuniq = df_insert.select(pl.struct(["product_id", "time", "granularity"]).n_unique()).item()
        if n != nuniq:
            logging.warning("Batch still contains duplicate keys: rows=%d unique=%d", n, nuniq)

        if df_insert.height > 0:
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
                [symbol, args.granularity, tmin, tmax],
            )
            conn.register("tmp_candles", df_insert)
            conn.execute(
                """
                INSERT INTO candles
                SELECT product_id, time, low, high, open, close, volume, granularity
                FROM tmp_candles
                """
            )
            conn.unregister("tmp_candles")

        if is_final_window:
            break
        w_start = w_end

    create_view(conn)
    try:
        counts = conn.execute(
            """
            SELECT
              COUNT(*) AS rows,
              COUNT(DISTINCT time) AS distinct_ts
            FROM hourly_bars
            WHERE symbol = ?
            """,
            [symbol],
        ).fetchone()
        logging.info("hourly_bars deduped view ready: rows=%s distinct_ts=%s", counts[0], counts[1])
    except Exception:
        logging.info("hourly_bars view created")

    curated = conn.execute(
        """
        SELECT symbol, ts, open, high, low, close, volume
        FROM hourly_bars
        WHERE symbol = ?
          AND ts >= ?
          AND ts <= ?
        ORDER BY ts ASC
        """,
        [symbol, start_dt, end_dt],
    ).pl()

    curated = curated.unique(subset=["ts"], keep="first").sort("ts")
    dup_dropped = curated.height - curated["ts"].n_unique()
    curated = curated.unique(subset=["ts"], keep="first")

    curated = curated.select(
        [
            pl.col("ts"),
            pl.col("symbol"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ]
    )

    if args.strict:
        validate_canonical(curated)

    parquet_path = args.parquet
    curated.write_parquet(parquet_path)

    logging.info("Fetched rows: %s", total_rows)
    if earliest_ts and latest_ts:
        logging.info("Range: %s -> %s", earliest_ts.isoformat(), latest_ts.isoformat())
    logging.info("Duplicates dropped: %s", dup_dropped)
    logging.info("Parquet written: %s", parquet_path)
    logging.info("DuckDB path: %s", args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())

