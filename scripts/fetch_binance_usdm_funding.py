from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import polars as pl
import requests


def parse_dt(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def request_with_retry(url: str, params: dict, *, max_retries: int = 5) -> List[dict]:
    backoff = 1.0
    for attempt in range(max_retries + 1):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 10.0)
            continue
        resp.raise_for_status()
    resp.raise_for_status()
    return []


def fetch_symbol_funding(
    base_url: str,
    symbol: str,
    start: datetime,
    end: datetime,
    limit: int,
    sleep_seconds: float,
) -> pl.DataFrame:
    url = f"{base_url}/fapi/v1/fundingRate"
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    rows: List[dict] = []
    while True:
        params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit}
        data = request_with_retry(url, params)
        if not data:
            break
        for d in data:
            rows.append({"ts": datetime.fromtimestamp(d["fundingTime"] / 1000, tz=timezone.utc), "fundingRate": float(d["fundingRate"])})
        last_ts = data[-1]["fundingTime"]
        if last_ts >= end_ms:
            break
        start_ms = last_ts + 1
        time.sleep(sleep_seconds)
        if len(data) < limit:
            break
    if not rows:
        return pl.DataFrame({"ts": [], "funding_apr": []})
    df = pl.DataFrame(rows).sort("ts")
    df = df.with_columns((pl.col("fundingRate") * 3 * 365).alias("funding_apr"))
    return df.select(["ts", "funding_apr"])


def resample_hourly(df: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    if df.is_empty():
        ts_range = pl.datetime_range(start=start, end=end, interval="1h", time_zone="UTC")
        return pl.DataFrame({"ts": ts_range, "funding_apr": 0.0})
    df = df.sort("ts")
    ts_range = pl.datetime_range(start=start, end=end, interval="1h", time_zone="UTC")
    hourly = pl.DataFrame({"ts": ts_range})
    hourly = hourly.join(df, on="ts", how="left")
    hourly = hourly.with_columns(pl.col("funding_apr").forward_fill().fill_null(0.0))
    return hourly


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance USD-M funding and output hourly APR series.")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--start", required=True, help="ISO8601 UTC start, e.g., 2016-05-17T17:00:00Z")
    parser.add_argument("--end", required=True, help="ISO8601 UTC end")
    parser.add_argument("--out_csv", default="data/funding/binance_usdm_funding_hourly.csv")
    parser.add_argument("--base_url", default="https://fapi.binance.com")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep_seconds", type=float, default=0.25)
    args = parser.parse_args()

    start_dt = parse_dt(args.start)
    end_dt = parse_dt(args.end)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    out_cols: Dict[str, pl.Series] = {"ts": pl.datetime_range(start=start_dt, end=end_dt, interval="1h", time_zone="UTC")}
    summary = {}

    for sym in symbols:
        df_sym = fetch_symbol_funding(args.base_url, sym, start_dt, end_dt, args.limit, args.sleep_seconds)
        hourly = resample_hourly(df_sym, start_dt, end_dt)
        col_name = f"{sym}_funding_apr"
        out_cols[col_name] = hourly["funding_apr"]
        summary[sym] = {
            "funding_points": df_sym.height,
            "hourly_rows": hourly.height,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        }

    out_df = pl.DataFrame(out_cols)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(out_path)

    print(f"Wrote {out_path}")
    for sym, info in summary.items():
        print(f"{sym}: funding_points={info['funding_points']}, hourly_rows={info['hourly_rows']}, range={info['start']} -> {info['end']}")


if __name__ == "__main__":
    main()

