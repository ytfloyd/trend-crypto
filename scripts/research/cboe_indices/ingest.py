#!/usr/bin/env python
"""Ingest Cboe Global Indices daily levels into ``indices_market.duckdb``.

Source: Cboe public end-of-day CSV files on CDN — one file per index, e.g.
``https://cdn.cboe.com/api/global/us_indices/daily_prices/VIXEQ_History.csv``.

Close-only indices (VIXEQ, DSPX, VVIX, …) are stored with
``open = high = low = close``.  Full OHLC indices (VIX, VIX9D, …) keep their
native columns.

Usage::

    # All Cboe vol + correlation indices on the public CDN (~40 symbols)
    python -m scripts.research.cboe_indices.ingest

    # Legacy minimal set
    python -m scripts.research.cboe_indices.ingest --universe core

    # Explicit list
    python -m scripts.research.cboe_indices.ingest --symbols VIX,COR3M,OVX
"""
from __future__ import annotations

import argparse
import io
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import duckdb
import pandas as pd

from .universe import VOL_CORRELATION_UNIVERSE, get_vol_correlation_universe

DEFAULT_DB = str(
    Path(__file__).resolve().parents[3] / ".." / "data" / "indices_market.duckdb"
)
CBOE_CDN = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
CORE_SYMBOLS = ("VIXEQ", "DSPX", "VIX")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bars_1d (
    symbol VARCHAR,
    ts     TIMESTAMP,
    open   DOUBLE,
    high   DOUBLE,
    low    DOUBLE,
    close  DOUBLE,
    volume DOUBLE,
    source VARCHAR,
    PRIMARY KEY (symbol, ts)
)
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=DEFAULT_DB, help="DuckDB output path")
    p.add_argument(
        "--universe",
        choices=["vol_correlation", "core"],
        default="vol_correlation",
        help="Curated symbol set (default: vol_correlation = all vol + correlation on CDN)",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated index symbols (overrides --universe)",
    )
    return p.parse_args()


def _fetch_csv(symbol: str) -> str:
    url = CBOE_CDN.format(symbol=symbol)
    req = Request(url, headers={"User-Agent": "trend_crypto/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"network error fetching {url}: {exc}") from exc


def _parse_cboe_csv(symbol: str, text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [c.strip().upper() for c in raw.columns]
    if "DATE" not in raw.columns:
        raise ValueError(f"{symbol}: expected DATE column, got {list(raw.columns)}")

    ts = pd.to_datetime(raw["DATE"], format="%m/%d/%Y", errors="coerce")
    out = pd.DataFrame({"symbol": symbol, "ts": ts})

    if {"OPEN", "HIGH", "LOW", "CLOSE"}.issubset(raw.columns):
        out["open"] = pd.to_numeric(raw["OPEN"], errors="coerce")
        out["high"] = pd.to_numeric(raw["HIGH"], errors="coerce")
        out["low"] = pd.to_numeric(raw["LOW"], errors="coerce")
        out["close"] = pd.to_numeric(raw["CLOSE"], errors="coerce")
    else:
        if symbol in raw.columns:
            level = pd.to_numeric(raw[symbol], errors="coerce")
        elif "CLOSE" in raw.columns:
            level = pd.to_numeric(raw["CLOSE"], errors="coerce")
        else:
            value_cols = [c for c in raw.columns if c != "DATE"]
            if len(value_cols) != 1:
                raise ValueError(f"{symbol}: unrecognized columns {list(raw.columns)}")
            level = pd.to_numeric(raw[value_cols[0]], errors="coerce")
        out["open"] = level
        out["high"] = level
        out["low"] = level
        out["close"] = level

    out["volume"] = 0.0
    out["source"] = "cboe_cdn"
    out = out.dropna(subset=["ts", "close"])
    out = out[out["close"] > 0]
    return out.sort_values("ts").reset_index(drop=True)


def upsert_symbol(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    con.register("records", df)
    con.execute(
        """
        INSERT OR REPLACE INTO bars_1d
            (symbol, ts, open, high, low, close, volume, source)
        SELECT symbol, ts, open, high, low, close, volume, source
        FROM records
        """
    )
    con.unregister("records")
    return len(df)


def main() -> None:
    args = parse_args()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe == "core":
        symbols = list(CORE_SYMBOLS)
    else:
        symbols = get_vol_correlation_universe()
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Cboe Indices Ingestion")
    print(f"{'='*60}")
    print(f"  Database : {db_path}")
    print(f"  Symbols  : {', '.join(symbols)}")
    print()

    con = duckdb.connect(str(db_path))
    con.execute(CREATE_TABLE_SQL)

    total = 0
    failed: list[str] = []
    for symbol in symbols:
        try:
            text = _fetch_csv(symbol)
            df = _parse_cboe_csv(symbol, text)
            n = upsert_symbol(con, df)
            total += n
            print(
                f"  {symbol}: {n:,} rows  "
                f"({df['ts'].min().date()} → {df['ts'].max().date()})"
            )
        except Exception as exc:
            print(f"  {symbol}: FAILED — {exc}")
            failed.append(symbol)

    con.close()
    print(f"\n  Total rows upserted: {total:,}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"  Finished {datetime.now().isoformat(timespec='seconds')}\n")


if __name__ == "__main__":
    main()
