#!/usr/bin/env python
"""Rebuild the SI (silver) front-month stitched 1-minute artifact.

The series is an *unadjusted* front-month stitch: at every timestamp we keep
the bars whose contract has not yet entered its delivery month
(``ts < first-of-expiry-month``) and pick the nearest such expiry. This is the
exact rule used to originally produce
``artifacts/research/si_quicklook/si_front_month_1m.parquet`` (which had no
saved builder), so re-running this regenerates the historical series
byte-for-byte while extending it to the latest data in the lake.

Source: ``data/futures_market.duckdb`` table ``bars_1m`` where ``symbol='SI'``.

Outputs (under ``--out_dir``, default the quicklook artifact dir):
    si_front_month_1m.parquet   stitched 1m bars (ts, expiry, o, h, l, c, v)
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
    WHERE symbol='SI' AND expiry != 'continuous'
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
    p.add_argument("--lake_path", default=str(DEFAULT_LAKE_PATH))
    p.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "artifacts/research/si_quicklook"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.lake_path, read_only=True)
    try:
        df = con.execute(STITCH_SQL).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError("No SI dated rows were available to stitch.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    dest = out_dir / "si_front_month_1m.parquet"
    df.to_parquet(dest, index=False)

    print(
        f"SI front-month: {len(df):,} rows  "
        f"{df['ts'].min()} -> {df['ts'].max()}  "
        f"({df['expiry'].nunique()} expiries)  -> {dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
