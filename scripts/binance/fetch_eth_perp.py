#!/usr/bin/env python
"""
Fetch ETH-USDT / BTC-USDT perpetual data from Binance's public archive
(data.binance.vision), which is US-accessible (the live API at fapi.binance.com
is geofenced from the US, but the archive on CloudFront is not).

Strategy:
- For completed months: download monthly ZIPs
- For the current month: download daily ZIPs day-by-day

Endpoints:
- Klines:  https://data.binance.vision/data/futures/um/{daily,monthly}/klines/{SYMBOL}/{INTERVAL}/...
- Funding: https://data.binance.vision/data/futures/um/{daily,monthly}/fundingRate/{SYMBOL}/...

Output (parquet, written to data/binance_perp/):
- {sym}_{interval}.parquet         daily/hourly OHLCV
- {sym}_funding_8h.parquet         raw 8h funding rates
- {sym}_funding_1d.parquet         daily-aggregated funding (sum of 3 8h payments)
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests


ARCHIVE_BASE = "https://data.binance.vision/data/futures/um"


def month_iter(start: date, end_inclusive: date):
    """Yield (year, month) pairs from start month up to and including end_inclusive month."""
    y, m = start.year, start.month
    while (y, m) <= (end_inclusive.year, end_inclusive.month):
        yield y, m
        m += 1
        if m > 12:
            y += 1; m = 1


def day_iter(start: date, end_inclusive: date):
    d = start
    while d <= end_inclusive:
        yield d
        d += timedelta(days=1)


def fetch_zip_csv(url: str, *, optional: bool = False) -> Optional[pd.DataFrame]:
    """Download a ZIP from data.binance.vision and parse the single CSV inside.
    Returns None on 404 if optional=True.
    """
    try:
        resp = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        print(f"  [error] {url}: {exc}", file=sys.stderr)
        if optional: return None
        raise
    if resp.status_code == 404 and optional:
        return None
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    members = z.namelist()
    if not members:
        return None
    with z.open(members[0]) as f:
        head = f.read(80)  # peek to detect header
    text = resp.content
    z = zipfile.ZipFile(io.BytesIO(text))
    with z.open(members[0]) as f:
        sample = f.read(200).decode("utf-8", errors="replace")
    # Skip header only if first character looks like a header letter
    has_header = sample[:1].isalpha()
    with z.open(members[0]) as f:
        if has_header:
            df = pd.read_csv(f)
        else:
            df = pd.read_csv(f, header=None)
    return df


def parse_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    if list(df.columns) == cols:
        pass
    elif df.shape[1] == 12:
        df.columns = cols
    else:
        raise ValueError(f"unexpected kline schema: {df.columns.tolist()}")
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote_volume"):
        df[c] = df[c].astype(float)
    df["count"] = df["count"].astype("int64")
    return df[["ts", "open", "high", "low", "close", "volume",
               "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]]


def parse_funding_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    if list(df.columns) == cols:
        pass
    elif df.shape[1] == 3:
        df.columns = cols
    else:
        raise ValueError(f"unexpected funding schema: {df.columns.tolist()}")
    df["ts"] = pd.to_datetime(df["calc_time"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = df["last_funding_rate"].astype(float)
    df["funding_interval_hours"] = df["funding_interval_hours"].astype("int64")
    return df[["ts", "funding_rate", "funding_interval_hours"]]


def fetch_klines(symbol: str, interval: str, start_d: date, end_d: date) -> pd.DataFrame:
    """Concatenate monthly archives + trailing daily files."""
    frames: List[pd.DataFrame] = []
    today = datetime.now(timezone.utc).date()
    # Use monthly archives for any month strictly before the current month.
    # For the current month, fall back to daily files (which appear once UTC day closes).
    last_complete_month = (today.replace(day=1) - timedelta(days=1))
    monthly_end = min(end_d, last_complete_month)
    daily_start = max(start_d, today.replace(day=1))

    # Monthly chunks
    if monthly_end >= start_d:
        for y, m in month_iter(start_d.replace(day=1), monthly_end):
            url = f"{ARCHIVE_BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{y:04d}-{m:02d}.zip"
            df = fetch_zip_csv(url, optional=True)
            if df is None:
                print(f"  [miss] {y:04d}-{m:02d} monthly klines for {symbol}")
                continue
            frames.append(parse_klines_df(df))

    # Daily files for the current partial month
    if daily_start <= end_d:
        for d in day_iter(daily_start, end_d):
            url = f"{ARCHIVE_BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d:%Y-%m-%d}.zip"
            df = fetch_zip_csv(url, optional=True)
            if df is None:
                # daily file not yet published — silent
                continue
            frames.append(parse_klines_df(df))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts"], keep="first").sort_values("ts").reset_index(drop=True)
    out = out[(out["ts"] >= pd.Timestamp(start_d, tz="UTC")) & (out["ts"] <= pd.Timestamp(end_d, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59))]
    return out.reset_index(drop=True)


def fetch_funding(symbol: str, start_d: date, end_d: date) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    today = datetime.now(timezone.utc).date()
    last_complete_month = (today.replace(day=1) - timedelta(days=1))
    monthly_end = min(end_d, last_complete_month)
    daily_start = max(start_d, today.replace(day=1))

    if monthly_end >= start_d:
        for y, m in month_iter(start_d.replace(day=1), monthly_end):
            url = f"{ARCHIVE_BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y:04d}-{m:02d}.zip"
            df = fetch_zip_csv(url, optional=True)
            if df is None:
                print(f"  [miss] {y:04d}-{m:02d} monthly funding for {symbol}")
                continue
            frames.append(parse_funding_df(df))

    if daily_start <= end_d:
        for d in day_iter(daily_start, end_d):
            url = f"{ARCHIVE_BASE}/daily/fundingRate/{symbol}/{symbol}-fundingRate-{d:%Y-%m-%d}.zip"
            df = fetch_zip_csv(url, optional=True)
            if df is None:
                continue
            frames.append(parse_funding_df(df))

    if not frames:
        return pd.DataFrame(columns=["ts", "funding_rate", "funding_interval_hours"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts"], keep="first").sort_values("ts").reset_index(drop=True)
    return out.reset_index(drop=True)


def aggregate_funding_daily(funding_8h: pd.DataFrame) -> pd.DataFrame:
    if funding_8h.empty:
        return pd.DataFrame(columns=["ts", "daily_funding_rate", "n_payments"])
    df = funding_8h.copy()
    df["date"] = df["ts"].dt.tz_convert("UTC").dt.normalize()
    g = df.groupby("date").agg(daily_funding_rate=("funding_rate", "sum"),
                                n_payments=("funding_rate", "count")).reset_index()
    g = g.rename(columns={"date": "ts"})
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance USD-M perpetual data via the public archive.")
    parser.add_argument("--symbols", default="ETHUSDT", help="Comma-separated symbols")
    parser.add_argument("--start", default="2019-11-01", help="ISO start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="ISO end date YYYY-MM-DD (default: today)")
    parser.add_argument("--interval", default="1d", help="Kline interval (1m, 5m, 1h, 4h, 1d, ...)")
    parser.add_argument("--out_dir", default="data/binance_perp")
    parser.add_argument("--no_funding", action="store_true")
    parser.add_argument("--no_klines", action="store_true")
    args = parser.parse_args()

    start_d = datetime.fromisoformat(args.start).date()
    end_d = datetime.fromisoformat(args.end).date() if args.end else datetime.now(timezone.utc).date()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    for symbol in symbols:
        sym_lower = symbol.lower()
        print(f"\n=== {symbol} ===")
        print(f"  Window: {start_d} -> {end_d}")

        if not args.no_klines:
            print(f"  Fetching klines (interval={args.interval}) via archive...")
            t0 = time.time()
            klines = fetch_klines(symbol, args.interval, start_d, end_d)
            if klines.empty:
                print(f"  WARN: no klines for {symbol}")
            else:
                fname = out_dir / f"{sym_lower}_{args.interval}.parquet"
                klines.to_parquet(fname, index=False)
                print(f"  klines: {len(klines)} rows  range=[{klines['ts'].iloc[0]}, {klines['ts'].iloc[-1]}]  ({time.time()-t0:.1f}s)  -> {fname}")

        if not args.no_funding:
            print(f"  Fetching funding rates (8h)...")
            t0 = time.time()
            funding = fetch_funding(symbol, start_d, end_d)
            if funding.empty:
                print(f"  WARN: no funding for {symbol}")
            else:
                fname = out_dir / f"{sym_lower}_funding_8h.parquet"
                funding.to_parquet(fname, index=False)
                print(f"  funding_8h: {len(funding)} rows  range=[{funding['ts'].iloc[0]}, {funding['ts'].iloc[-1]}]  ({time.time()-t0:.1f}s)  -> {fname}")
                daily = aggregate_funding_daily(funding)
                fname_d = out_dir / f"{sym_lower}_funding_1d.parquet"
                daily.to_parquet(fname_d, index=False)
                avg_daily = float(daily["daily_funding_rate"].mean())
                print(f"  funding_1d: {len(daily)} rows  ->  {fname_d}")
                print(f"    avg daily funding rate: {avg_daily*100:+.4f}% /day   ({avg_daily*365*100:+.2f}% APR equiv)")
                print(f"    sign: + = longs pay shorts, - = shorts pay longs")


if __name__ == "__main__":
    main()
