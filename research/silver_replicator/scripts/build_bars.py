"""CLI: resample stitched silver front-month 1m bars into 1H/4H/8H/1D parquets.

Usage:
    python scripts/build_bars.py [--start 2025-10-01] [--end 2026-05-05]

Outputs:
    artifacts/si_front_month_1H.parquet
    artifacts/si_front_month_4H.parquet
    artifacts/si_front_month_8H.parquet
    artifacts/si_front_month_1D.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_ROOT))

from src.bars import TIMEFRAMES, multi_tf_bars  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=_ROOT / "artifacts",
        help="Output directory for parquet files.",
    )
    args = parser.parse_args(argv)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    bars = multi_tf_bars(start=args.start, end=args.end)

    print(f"Bar window: {args.start} -> {args.end}")
    print("-" * 60)
    for tf in TIMEFRAMES:
        df = bars[tf]
        path = args.artifacts / f"si_front_month_{tf}.parquet"
        df.reset_index().to_parquet(path, index=False)
        if df.empty:
            print(f"  {tf:>3s}: 0 bars (no data)")
            continue
        print(
            f"  {tf:>3s}: {len(df):>7,d} bars   "
            f"{df.index.min()} -> {df.index.max()}"
        )
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
