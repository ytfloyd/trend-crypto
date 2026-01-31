#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from transtrend_crypto_metrics_v0 import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metrics for Transtrend Crypto simple baseline")
    p.add_argument("--equity", required=True, help="Path to equity.csv")
    p.add_argument("--out", required=True, help="Output metrics CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.equity)
    metrics = compute_metrics(df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)
    print(f"[transtrend_crypto_simple_baseline_metrics] Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
