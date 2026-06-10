#!/usr/bin/env python
"""Build the institutional back-adjusted CL continuous research artifact.

Reads every dated CL contract from the volbook minute lake and stitches a
front-month continuous series using the volume-crossover roll policy with a
calendar guard and additive back-adjustment (``volbook.continuous``).

The lake only supports a *valid* front-month continuous where the near/front
strip is present.  Sessions where the near contracts are missing (e.g. the
deep 2022-2024 history, which only contains far-dated December contracts) are
emitted with ``front_month_valid = False`` so downstream consumers can filter.

Outputs (under ``--out_dir``):
    bars_1m.parquet            stitched, back-adjusted 1m bars
    roll_schedule.csv          per-roll lineage (dates, gaps, triggers)
    metadata.json              construction parameters + coverage summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from volbook.continuous import (  # noqa: E402
    FrontMonthGuard,
    NymexObservedHolidayCalendar,
    RollPolicy,
    construct_continuous_series,
)
from volbook.datalake import DEFAULT_LAKE_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lake_path", default=str(DEFAULT_LAKE_PATH))
    p.add_argument("--out_dir", default="artifacts/research/cl_institutional_continuous")
    p.add_argument("--adjustment", default="additive", choices=["raw", "additive", "ratio"])
    p.add_argument(
        "--valid_only",
        action="store_true",
        help="Write only front_month_valid rows (tradable institutional series).",
    )
    return p.parse_args()


def load_dated_cl(lake_path: str) -> pd.DataFrame:
    con = duckdb.connect(lake_path, read_only=True)
    try:
        df = con.execute(
            """
            SELECT ts, expiry, o, h, l, c, v
            FROM bars_1m
            WHERE symbol = 'CL' AND expiry != 'continuous'
            ORDER BY ts, expiry
            """
        ).fetchdf()
    finally:
        con.close()
    return df


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dated = load_dated_cl(args.lake_path)
    if dated.empty:
        raise RuntimeError("No dated CL rows found in lake.")
    dated["symbol"] = "CL"

    result = construct_continuous_series(
        dated,
        symbol="CL",
        policy=RollPolicy(name="volume_crossover_with_calendar_guard"),
        adjustment=args.adjustment,
        calendar=NymexObservedHolidayCalendar(),
        front_month_guard=FrontMonthGuard(
            symbols=("CL",), max_curve_position=2, on_missing="mark"
        ),
    )

    bars = result.bars.copy()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars = bars.sort_values("ts").reset_index(drop=True)

    if "front_month_valid" in bars.columns:
        valid_mask = bars["front_month_valid"].astype(bool)
    else:
        valid_mask = pd.Series(True, index=bars.index)
    valid = bars[valid_mask]

    out_bars = valid if args.valid_only else bars
    out_bars.to_parquet(out_dir / "bars_1m.parquet", index=False)
    if not result.schedule.empty:
        result.schedule.to_csv(out_dir / "roll_schedule.csv", index=False)

    valid_window = {
        "first_valid_ts": valid["ts"].min().isoformat() if not valid.empty else None,
        "last_valid_ts": valid["ts"].max().isoformat() if not valid.empty else None,
        "valid_row_count": int(len(valid)),
    }
    metadata = {
        "symbol": "CL",
        "construction": "back_adjusted_front_month_continuous",
        "adjustment": args.adjustment,
        "roll_policy": "volume_crossover_with_calendar_guard",
        "calendar": "nymex_observed",
        "lake_path": str(args.lake_path),
        "valid_only_written": bool(args.valid_only),
        "total_row_count": int(len(bars)),
        "full_start_ts": bars["ts"].min().isoformat(),
        "full_end_ts": bars["ts"].max().isoformat(),
        "valid_front_month_window": valid_window,
        "source_expiries": sorted(dated["expiry"].astype(str).unique().tolist()),
        "roll_count": int(len(result.schedule)),
        "construction_metadata": result.metadata,
    }

    def _default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_default), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "construction_metadata"}, indent=2, default=_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
