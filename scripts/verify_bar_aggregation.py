from __future__ import annotations

import argparse
import sys
import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify resampled bar aggregation consistency.")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--symbol", required=True, help="Symbol to validate (e.g., BTC-USD)")
    parser.add_argument("--samples", type=int, default=20, help="Number of random buckets to check")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="Tolerance for value comparisons")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("SET TimeZone='UTC';")

    # sample buckets
    sample = con.execute(
        """
        SELECT ts
        FROM bars_4h_clean
        WHERE symbol = ?
        ORDER BY random()
        LIMIT ?
        """,
        [args.symbol, args.samples],
    ).fetchall()

    for (ts_bucket,) in sample:
        # expected aggregated from hourly_bars_clean
        expected = con.execute(
            """
            SELECT
              arg_min(open, ts) AS open,
              max(high) AS high,
              min(low) AS low,
              arg_max(close, ts) AS close,
              sum(volume) AS volume
            FROM hourly_bars_clean
            WHERE symbol = ?
              AND ts >= ?
              AND ts < ? + INTERVAL '4 hours'
            """,
            [args.symbol, ts_bucket],
        ).fetchone()
        actual = con.execute(
            """
            SELECT open, high, low, close, volume
            FROM bars_4h_clean
            WHERE symbol = ? AND ts = ?
            """,
            [args.symbol, ts_bucket],
        ).fetchone()
        if actual is None:
            print(f"Missing 4h bar at {ts_bucket}")
            sys.exit(1)
        for exp_val, act_val, field in zip(expected, actual, ["open", "high", "low", "close", "volume"]):
            if exp_val is None or act_val is None or abs(exp_val - act_val) > args.tolerance:
                print(f"Mismatch at {ts_bucket} field {field}: expected {exp_val} got {act_val}")
                sys.exit(1)

    print(f"Verified {len(sample)} buckets for {args.symbol} with tolerance {args.tolerance}")
    con.close()


if __name__ == "__main__":
    main()

