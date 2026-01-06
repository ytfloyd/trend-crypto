#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime labels from alpha_c16/19/20 medians.")
    p.add_argument(
        "--alphas",
        required=True,
        help="Path to alphas parquet (e.g. artifacts/research/101_alphas/alphas_101_v0.parquet)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output CSV for regimes (e.g. artifacts/research/101_alphas/alphas101_regimes_v0.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alphas_path = Path(args.alphas)
    out_path = Path(args.out)

    df = pd.read_parquet(alphas_path)
    if "ts" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Alphas parquet must have 'ts' and 'symbol' columns.")

    needed = ["alpha_c16", "alpha_c19", "alpha_c20"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in alphas parquet.")

    df["ts"] = pd.to_datetime(df["ts"])

    agg = (
        df.groupby("ts")[needed]
        .median()
        .rename(columns={
            "alpha_c16": "c16_med",
            "alpha_c19": "c19_med",
            "alpha_c20": "c20_med",
        })
        .sort_index()
    )

    for col, zcol in [("c16_med", "z_c16"), ("c19_med", "z_c19"), ("c20_med", "z_c20")]:
        mu = agg[col].mean(skipna=True)
        sigma = agg[col].std(ddof=1, skipna=True)
        if sigma and sigma > 0:
            agg[zcol] = (agg[col] - mu) / sigma
        else:
            agg[zcol] = 0.0

    agg["regime"] = "mean_rev"
    danger_mask = (agg["z_c19"] > 1.0) | (agg["z_c20"] > 1.0)
    trend_mask = ~danger_mask & (agg["z_c16"] > 0.0)
    agg.loc[trend_mask, "regime"] = "trend"
    agg.loc[danger_mask, "regime"] = "danger"

    out_df = agg.reset_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[alphas101_regime_labels_v0] Wrote regime labels ({len(out_df)} rows) to {out_path}")


if __name__ == "__main__":
    main()

