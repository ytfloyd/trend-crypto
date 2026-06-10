"""CLI: compute TA-Lib features per timeframe and persist them.

Usage:
    python scripts/build_features.py [--artifacts artifacts]

Reads:
    artifacts/si_front_month_{1H,4H,8H,1D}.parquet

Writes:
    artifacts/features_{1H,4H,8H,1D}.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_ROOT))

from src.bars import TIMEFRAMES  # noqa: E402
from src.talib_features import compute_features  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=_ROOT / "artifacts",
        help="Directory that holds bar parquets and will receive feature parquets.",
    )
    args = parser.parse_args(argv)

    print("Feature build")
    print("-" * 60)
    for tf in TIMEFRAMES:
        in_path = args.artifacts / f"si_front_month_{tf}.parquet"
        out_path = args.artifacts / f"features_{tf}.parquet"
        if not in_path.exists():
            print(f"  {tf:>3s}: MISSING {in_path}")
            continue
        bars = pd.read_parquet(in_path)
        if "ts" in bars.columns:
            bars = bars.set_index(pd.to_datetime(bars["ts"], utc=True)).drop(columns=["ts"])
        feat = compute_features(bars)
        feat.reset_index().to_parquet(out_path, index=False)

        all_nan = [c for c in feat.columns if feat[c].isna().all()]
        print(
            f"  {tf:>3s}: shape={feat.shape}   "
            f"all-NaN cols={len(all_nan)}   -> {out_path.name}"
        )
        if all_nan:
            print(f"        all-NaN: {all_nan}")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
