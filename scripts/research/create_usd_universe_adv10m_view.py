#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create ADV>10M USD view for 101-alphas research.")
    p.add_argument("--db", required=True, help="Path to DuckDB file (e.g., ../data/coinbase_daily_121025.duckdb)")
    p.add_argument(
        "--source_view",
        default="bars_1d_usd_universe_clean",
        help="Source daily bars view/table (default: bars_1d_usd_universe_clean)",
    )
    p.add_argument(
        "--adv_window",
        type=int,
        default=20,
        help="Rolling window for ADV in days (default: 20)",
    )
    p.add_argument(
        "--adv_threshold_usd",
        type=float,
        default=10_000_000,
        help="ADV USD threshold (default: 10,000,000)",
    )
    p.add_argument(
        "--out_view",
        default="bars_1d_usd_universe_clean_adv10m",
        help="Name of the output view to create/replace.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    con = duckdb.connect(str(db_path))
    con.execute("SET TimeZone='UTC';")

    query = f"""
    CREATE OR REPLACE VIEW {args.out_view} AS
    WITH adv AS (
        SELECT
            symbol,
            ts,
            open,
            high,
            low,
            close,
            volume,
            vwap,
            AVG(close * volume) OVER (
                PARTITION BY symbol
                ORDER BY ts
                ROWS BETWEEN {args.adv_window - 1} PRECEDING AND CURRENT ROW
            ) AS adv20_usd
        FROM {args.source_view}
    )
    SELECT *
    FROM adv
    WHERE adv20_usd >= {args.adv_threshold_usd};
    """
    con.execute(query)

    summary = con.execute(
        f"""
        SELECT
            COUNT(DISTINCT symbol) AS n_symbols,
            MIN(ts) AS min_ts,
            MAX(ts) AS max_ts,
            MIN(adv20_usd) AS min_adv,
            MAX(adv20_usd) AS max_adv
        FROM {args.out_view};
        """
    ).fetch_df()
    con.close()

    print(
        f"[create_usd_universe_adv10m_view] Created view {args.out_view} from {args.source_view} "
        f"with ADV window={args.adv_window}, threshold={args.adv_threshold_usd:.0f}"
    )
    if not summary.empty:
        row = summary.iloc[0]
        print(
            f"[create_usd_universe_adv10m_view] Symbols={int(row['n_symbols'])}, "
            f"ts_range=[{row['min_ts']}, {row['max_ts']}], "
            f"adv20_usd min/max=({row['min_adv']:.2f}, {row['max_adv']:.2f})"
        )


if __name__ == "__main__":
    main()

