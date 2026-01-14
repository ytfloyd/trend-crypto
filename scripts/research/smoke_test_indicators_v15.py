#!/usr/bin/env python
from __future__ import annotations

"""
Smoke test for indicator primitives added for V1.5 Growth Sleeve.

Pulls a small BTC-USD sample (daily and optional 4H) from DuckDB and computes:
- ADX(14)
- Ichimoku (9/26/52)
- Keltner (20, 2.0 ATR)
- DEWMA(20)

Prints tail values and NaN counts; asserts ADX is bounded [0, 100] where defined.
"""

import duckdb
import pandas as pd

from alpha_utils import (
    calc_adx,
    calc_ichimoku,
    calc_keltner,
    calc_dewma,
)


DB_PATH = "../data/coinbase_daily_121025.duckdb"


def fetch_df(table: str, limit: int = 300) -> pd.DataFrame:
    con = duckdb.connect(database=DB_PATH, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT ts, symbol, open, high, low, close, volume
            FROM {table}
            WHERE symbol = 'BTC-USD'
            ORDER BY ts DESC
            LIMIT {limit}
            """
        ).fetch_df()
        df["ts"] = pd.to_datetime(df["ts"])
        return df.sort_values("ts").reset_index(drop=True)
    finally:
        con.close()


def maybe_table(names):
    con = duckdb.connect(database=DB_PATH, read_only=True)
    try:
        existing = set(con.execute("SHOW TABLES").fetch_df()["name"])
    finally:
        con.close()
    for n in names:
        if n in existing:
            return n
    return None


def main() -> None:
    daily_table = maybe_table(["bars_1d_usd_universe_clean_adv10m", "bars_1d_usd_universe_clean"])
    if not daily_table:
        raise SystemExit("No daily bars table found for smoke test.")
    intraday_table = maybe_table(["bars_4h_usd_universe_clean_adv10m", "bars_4h_usd_universe_clean"])

    print(f"Using daily table: {daily_table}")
    if intraday_table:
        print(f"Using intraday table: {intraday_table}")
    else:
        print("No intraday 4H table found; skipping fast DEWMA smoke on 4H.")

    df_day = fetch_df(daily_table)
    df_day = df_day.rename(columns=str.lower)

    adx = calc_adx(df_day, n=14)
    ichi = calc_ichimoku(df_day)
    kelt = calc_keltner(df_day)
    dewma = calc_dewma(df_day["close"], n=20)

    print("\nADX tail:")
    print(adx.tail(3))
    print("ADX NaNs:", adx.isna().sum(), "min:", adx.min(), "max:", adx.max())
    assert (adx.dropna() >= 0).all() and (adx.dropna() <= 100).all()

    print("\nIchimoku cloud tail (cloud_top/cloud_bottom/in_cloud):")
    print(ichi[["cloud_top", "cloud_bottom", "in_cloud"]].tail(3))
    print("Ichimoku NaNs:", ichi.isna().sum().to_dict())

    print("\nKeltner tail:")
    print(kelt.tail(3))
    print("Keltner NaNs:", kelt.isna().sum().to_dict())

    print("\nDEWMA tail:")
    print(dewma.tail(3))
    print("DEWMA NaNs:", dewma.isna().sum())

    if intraday_table:
        df_4h = fetch_df(intraday_table, limit=500)
        df_4h = df_4h.rename(columns=str.lower)
        dewma_4h = calc_dewma(df_4h["close"], n=20)
        print("\nDEWMA 4H tail:")
        print(dewma_4h.tail(3))
        print("DEWMA 4H NaNs:", dewma_4h.isna().sum())

    print("\nSmoke test OK.")


if __name__ == "__main__":
    main()
