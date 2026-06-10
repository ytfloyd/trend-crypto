#!/usr/bin/env python
"""Build an unadjusted front-month stitched 1-minute artifact for a symbol.

This is the symbol-agnostic generalization of
``scripts/research/si_front_month_continuous.py``. At every timestamp it keeps
the bars whose contract has not yet entered its delivery month
(``ts < first-of-expiry-month``) and picks the nearest such expiry. The result
is an *unadjusted* front-month stitch (no back-adjustment), identical in
methodology to the SI series already carried in the frozen snapshots.

This rule rolls on the first calendar day of the expiry month, which is a clean,
reproducible convention. For quarterly products (e.g. ES) this rolls slightly
ahead of the third-Friday CME roll; that is intentional and conservative.

The principled volume-crossover / back-adjusted builder
(``volbook.continuous.construct_continuous_series``) is currently CL-only
(it hard-codes CL last-trade-date conventions), so it cannot be used here.

Source: ``data/futures_market.duckdb`` table ``bars_1m`` for the given symbol.

Outputs (under ``--out_dir``):
    bars_1m.parquet   stitched 1m bars (ts, expiry, o, h, l, c, v)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from volbook.datalake import DEFAULT_LAKE_PATH  # noqa: E402

STITCH_SQL = """
WITH active AS (
    SELECT
        ts, expiry, o, h, l, c, v,
        strptime(expiry || '01', '%Y%m%d') AS roll_point
    FROM bars_1m
    WHERE symbol = ? AND expiry != 'continuous'
),
ranked AS (
    SELECT ts, expiry, o, h, l, c, v,
           row_number() OVER (PARTITION BY ts ORDER BY expiry ASC) AS rn
    FROM active
    WHERE ts < roll_point
)
SELECT ts, expiry, o, h, l, c, v
FROM ranked
WHERE rn = 1
ORDER BY ts
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True, help="Lake symbol, e.g. ES or VX.")
    p.add_argument("--lake_path", default=str(DEFAULT_LAKE_PATH))
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output dir (default artifacts/research/<symbol_lower>_continuous).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper()
    out_dir = Path(
        args.out_dir
        or (REPO_ROOT / f"artifacts/research/{symbol.lower()}_continuous")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.lake_path, read_only=True)
    try:
        df = con.execute(STITCH_SQL, [symbol]).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError(f"No dated rows for symbol={symbol!r} to stitch.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    dest = out_dir / "bars_1m.parquet"
    df.to_parquet(dest, index=False)

    print(
        f"{symbol} front-month: {len(df):,} rows  "
        f"{df['ts'].min()} -> {df['ts'].max()}  "
        f"({df['expiry'].nunique()} expiries)  -> {dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
