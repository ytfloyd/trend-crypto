#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from typing import Optional

import duckdb


def detect_table(con: duckdb.DuckDBPyConnection, candidates: list[str]) -> Optional[str]:
    tables = set(con.execute("SHOW TABLES").fetch_df()["name"])
    for t in candidates:
        if t in tables:
            return t
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Create Top-50 (30D volume) ADV>10M daily+4H universe views for Growth Sleeve v1.5.")
    p.add_argument("--db", required=True, help="Path to DuckDB, e.g. ../data/coinbase_daily_121025.duckdb")
    p.add_argument("--source_table_1d", default=None, help="Daily bars source (default: bars_1d_usd_universe_clean)")
    p.add_argument("--source_table_4h", default=None, help="4H bars source (default: bars_4h_usd_universe_clean)")
    p.add_argument("--top_n", type=int, default=50, help="Top N by rolling dollar volume")
    p.add_argument("--lookback_days", type=int, default=30, help="Rolling window (days)")
    p.add_argument("--adv_min_usd", type=float, default=10_000_000, help="ADV threshold in USD")
    p.add_argument(
        "--out_table_1d",
        default="bars_1d_usd_universe_clean_top50_adv10m",
        help="Output daily table/view name",
    )
    p.add_argument(
        "--out_table_4h",
        default="bars_4h_usd_universe_clean_top50_adv10m",
        help="Output 4H table/view name",
    )
    args = p.parse_args()

    con = duckdb.connect(args.db)
    try:
        src_1d = args.source_table_1d or detect_table(con, ["bars_1d_usd_universe_clean_adv10m", "bars_1d_usd_universe_clean"])
        if not src_1d:
            raise SystemExit("Could not find a daily source table (bars_1d_usd_universe_clean[_adv10m]).")
        src_4h = args.source_table_4h or detect_table(con, ["bars_4h_usd_universe_clean_adv10m", "bars_4h_usd_universe_clean"])
        if not src_4h:
            print("[create_universe] No 4H table found; skipping 4H view.", file=sys.stderr)

        print(f"[create_universe] Using daily source: {src_1d}")
        if src_4h:
            print(f"[create_universe] Using 4H source: {src_4h}")

        # Build membership (time-varying) on daily bars
        con.execute(f"DROP TABLE IF EXISTS {args.out_table_1d}_membership")
        con.execute(
            f"""
            CREATE TABLE {args.out_table_1d}_membership AS
            WITH daily AS (
                SELECT ts, symbol, close, volume, (close * volume) AS dollar_vol
                FROM {src_1d}
            ),
            rolling AS (
                SELECT
                    ts,
                    symbol,
                    avg(dollar_vol) OVER (PARTITION BY symbol ORDER BY ts ROWS BETWEEN {args.lookback_days - 1} PRECEDING AND CURRENT ROW) AS adv_usd
                FROM daily
            ),
            ranked AS (
                SELECT
                    ts,
                    symbol,
                    adv_usd,
                    row_number() OVER (PARTITION BY ts ORDER BY adv_usd DESC NULLS LAST) AS rn
                FROM rolling
            )
            SELECT ts, symbol, adv_usd, rn
            FROM ranked
            WHERE rn <= {args.top_n} AND adv_usd >= {args.adv_min_usd}
            """
        )
        print(f"[create_universe] Created membership table {args.out_table_1d}_membership")

        # Daily filtered table
        con.execute(f"DROP TABLE IF EXISTS {args.out_table_1d}")
        con.execute(
            f"""
            CREATE TABLE {args.out_table_1d} AS
            SELECT b.*
            FROM {src_1d} b
            JOIN {args.out_table_1d}_membership m
              ON b.symbol = m.symbol AND b.ts = m.ts
            """
        )
        print(f"[create_universe] Created daily universe table {args.out_table_1d}")

        # 4H filtered table (if available)
        if src_4h:
            con.execute(f"DROP TABLE IF EXISTS {args.out_table_4h}")
            con.execute(
                f"""
                CREATE TABLE {args.out_table_4h} AS
                SELECT h.*
                FROM {src_4h} h
                JOIN {args.out_table_1d}_membership m
                  ON h.symbol = m.symbol AND CAST(h.ts AS DATE) = CAST(m.ts AS DATE)
                """
            )
            print(f"[create_universe] Created 4H universe table {args.out_table_4h}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
