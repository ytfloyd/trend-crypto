#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from transtrend_crypto_metrics_v1 import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transtrend Crypto v2 multi-speed metrics")
    p.add_argument("--combined_equity", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--fast_equity", type=str, default=None)
    p.add_argument("--slow_equity", type=str, default=None)
    return p.parse_args()


def write_metrics(equity_path: Path, out_path: Path) -> None:
    df = pd.read_csv(equity_path)
    metrics = compute_metrics(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_out = out_dir / "metrics_transtrend_crypto_v2.csv"
    write_metrics(Path(args.combined_equity), combined_out)

    if args.fast_equity:
        write_metrics(Path(args.fast_equity), out_dir / "fast_metrics.csv")
    if args.slow_equity:
        write_metrics(Path(args.slow_equity), out_dir / "slow_metrics.csv")

    print(f"[transtrend_crypto_v2_multispeed] Wrote {combined_out}")


if __name__ == "__main__":
    main()
