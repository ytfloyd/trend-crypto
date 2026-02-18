#!/usr/bin/env python
"""
Ingest ETF daily data from Tiingo into DuckDB.

Fetches adjusted OHLCV for the curated ETF universe and stores it in
``data/etf_market.duckdb`` with the same ``bars_1d`` table schema used
by the crypto research infrastructure.

The ingestion is idempotent: re-running updates existing tickers with
any new data since the last fetch.

Usage::

    # Full ingest (all ~64 ETFs, from 2005 to today)
    python -m scripts.research.etf_data.ingest

    # Specific tickers and date range
    python -m scripts.research.etf_data.ingest --tickers SPY,QQQ,TLT --start 2010-01-01

    # Use core universe (fewer tickers, longer histories)
    python -m scripts.research.etf_data.ingest --universe core
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from .tiingo_client import TiingoDaily
from .universe import get_core_universe, get_full_universe

DEFAULT_DB = str(
    Path(__file__).resolve().parents[3] / ".." / "data" / "etf_market.duckdb"
)

# Store adjusted prices in the standard bars_1d columns so the momentum
# library works identically for crypto and ETFs.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bars_1d (
    symbol VARCHAR,
    ts     TIMESTAMP,
    open   DOUBLE,
    high   DOUBLE,
    low    DOUBLE,
    close  DOUBLE,
    volume DOUBLE,
    dividend     DOUBLE,
    split_factor DOUBLE,
    PRIMARY KEY (symbol, ts)
)
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest ETF daily data from Tiingo")
    p.add_argument("--db", default=DEFAULT_DB, help="DuckDB output path")
    p.add_argument("--start", default="2005-01-01", help="Start date")
    p.add_argument("--end", default=None, help="End date (default: today)")
    p.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated ticker list (overrides --universe)",
    )
    p.add_argument(
        "--universe",
        choices=["full", "core"],
        default="full",
        help="Which curated universe to use (default: full)",
    )
    p.add_argument("--api-key", default=None, help="Tiingo API key (or set TIINGO_API_KEY)")
    return p.parse_args()


def _get_last_date(con: duckdb.DuckDBPyConnection, ticker: str) -> str | None:
    """Return the last date we have for this ticker, or None."""
    result = con.execute(
        "SELECT MAX(ts) FROM bars_1d WHERE symbol = ?", [ticker]
    ).fetchone()
    if result and result[0] is not None:
        return pd.Timestamp(result[0]).strftime("%Y-%m-%d")
    return None


def ingest_ticker(
    client: TiingoDaily,
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    start: str,
    end: str | None,
) -> int:
    """Fetch and upsert data for a single ticker. Returns row count."""
    # Check for existing data → incremental update
    last_date = _get_last_date(con, ticker)
    fetch_start = start
    if last_date is not None:
        # Start from day after last stored date
        next_day = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fetch_start = next_day

    df = client.fetch(ticker, start=fetch_start, end=end)
    if df.empty:
        return 0

    # Use adjusted prices as the standard OHLCV columns
    records = pd.DataFrame({
        "symbol": df["symbol"],
        "ts": df["ts"],
        "open": df.get("adj_open", df["open"]),
        "high": df.get("adj_high", df["high"]),
        "low": df.get("adj_low", df["low"]),
        "close": df.get("adj_close", df["close"]),
        "volume": df.get("adj_volume", df["volume"]),
        "dividend": df.get("dividend", 0.0),
        "split_factor": df.get("split_factor", 1.0),
    })

    # Filter valid rows
    records = records[
        (records["open"] > 0) & (records["close"] > 0) & (records["high"] >= records["low"])
    ]
    if records.empty:
        return 0

    # Upsert: INSERT OR REPLACE
    con.execute(
        """
        INSERT OR REPLACE INTO bars_1d
            (symbol, ts, open, high, low, close, volume, dividend, split_factor)
        SELECT symbol, ts, open, high, low, close, volume, dividend, split_factor
        FROM records
        """
    )
    return len(records)


def main() -> None:
    args = parse_args()
    end = args.end or datetime.now().strftime("%Y-%m-%d")

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.universe == "core":
        tickers = get_core_universe()
    else:
        tickers = get_full_universe()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== ETF Data Ingestion ===")
    print(f"Database : {db_path}")
    print(f"Tickers  : {len(tickers)}")
    print(f"Range    : {args.start} → {end}")

    client = TiingoDaily(api_key=args.api_key)
    con = duckdb.connect(str(db_path))
    con.execute(CREATE_TABLE_SQL)

    total_rows = 0
    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        try:
            n = ingest_ticker(client, con, ticker, args.start, end)
            total_rows += n
            status = f"{n:,} rows" if n > 0 else "up to date"
            print(f"  [{i+1}/{len(tickers)}] {ticker}: {status}")
        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {ticker}: FAILED — {e}")
            failed.append(ticker)

    con.close()

    print(f"\n--- Summary ---")
    print(f"Total rows inserted/updated: {total_rows:,}")
    print(f"Tickers processed: {len(tickers) - len(failed)}/{len(tickers)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
