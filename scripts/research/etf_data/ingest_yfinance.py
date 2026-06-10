#!/usr/bin/env python
"""
Bulk-download ETF daily data via yfinance into DuckDB.

Uses yfinance's batch download for speed (~200 tickers in a single request).
Data is stored in the same ``bars_1d`` table schema as the Tiingo pipeline
so that alpha_brain.py works identically for crypto and ETFs.

Usage::

    # Full expanded universe (~148 ETFs, from 2005 to today)
    python -m scripts.research.etf_data.ingest_yfinance

    # Core universe only
    python -m scripts.research.etf_data.ingest_yfinance --universe core

    # Specific tickers
    python -m scripts.research.etf_data.ingest_yfinance --tickers SPY,QQQ,TLT
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

from .universe import get_core_universe, get_full_universe

DEFAULT_DB = str(
    Path(__file__).resolve().parents[3] / ".." / "data" / "etf_market.duckdb"
)

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

BATCH_SIZE = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk-download ETF data via yfinance")
    p.add_argument("--db", default=DEFAULT_DB, help="DuckDB output path")
    p.add_argument("--start", default="2005-01-01", help="Start date")
    p.add_argument("--end", default=None, help="End date (default: today)")
    p.add_argument(
        "--tickers", default=None,
        help="Comma-separated ticker list (overrides --universe)",
    )
    p.add_argument(
        "--universe", choices=["expanded", "full", "core"],
        default="expanded",
        help="Which curated universe to use (default: expanded)",
    )
    return p.parse_args()


def _download_batch(
    tickers: list[str], start: str, end: str,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for a batch of tickers via yfinance."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    if raw.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        ticker = tickers[0]
        df = raw.copy()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        df = df.rename(columns={"adj close": "close"})
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                return {}
        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        df["symbol"] = ticker
        df["ts"] = df.index
        df = df.reset_index(drop=True)
        results[ticker] = df
        return results

    # Multi-ticker: columns are MultiIndex (metric, ticker)
    for ticker in tickers:
        try:
            sub = raw.xs(ticker, level="Ticker", axis=1).copy()
            sub.columns = [c.lower() for c in sub.columns]
            sub = sub.rename(columns={"adj close": "close"})
            needed = ["open", "high", "low", "close", "volume"]
            if not all(c in sub.columns for c in needed):
                continue
            sub = sub[needed].dropna(subset=["close"])
            if len(sub) < 10:
                continue
            sub["symbol"] = ticker
            sub["ts"] = sub.index
            sub = sub.reset_index(drop=True)
            results[ticker] = sub
        except (KeyError, ValueError):
            continue

    return results


def upsert_ticker(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert or replace rows for a single ticker."""
    records = pd.DataFrame({
        "symbol": df["symbol"],
        "ts": pd.to_datetime(df["ts"]),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "dividend": 0.0,
        "split_factor": 1.0,
    })
    records = records[
        (records["open"] > 0) & (records["close"] > 0)
        & (records["high"] >= records["low"])
        & np.isfinite(records["close"])
    ]
    if records.empty:
        return 0
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
    elif args.universe == "full":
        tickers = get_full_universe()
    else:
        # 'expanded' was merged into the full curated universe.
        tickers = get_full_universe()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ETF Data Ingestion (yfinance)")
    print(f"{'='*60}")
    print(f"  Database : {db_path}")
    print(f"  Tickers  : {len(tickers)}")
    print(f"  Range    : {args.start} -> {end}")
    print(f"  Batches  : {(len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print()

    con = duckdb.connect(str(db_path))
    con.execute(CREATE_TABLE_SQL)

    total_rows = 0
    succeeded: list[str] = []
    failed: list[str] = []

    for batch_start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches}: downloading {len(batch)} tickers...")

        try:
            results = _download_batch(batch, args.start, end)
        except Exception as e:
            print(f"    Batch download failed: {e}")
            failed.extend(batch)
            continue

        for ticker in batch:
            if ticker in results and len(results[ticker]) > 0:
                try:
                    n = upsert_ticker(con, results[ticker])
                    total_rows += n
                    succeeded.append(ticker)
                    print(f"    {ticker}: {n:,} rows")
                except Exception as e:
                    print(f"    {ticker}: UPSERT FAILED - {e}")
                    failed.append(ticker)
            else:
                print(f"    {ticker}: no data returned")
                failed.append(ticker)

    con.close()

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Total rows inserted/updated: {total_rows:,}")
    print(f"  Succeeded: {len(succeeded)}/{len(tickers)}")
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(sorted(failed))}")
    print()


if __name__ == "__main__":
    main()
