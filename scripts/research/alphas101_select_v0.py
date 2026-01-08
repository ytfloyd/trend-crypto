#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select and orient alphas based on IC panel.")
    p.add_argument("--ic_panel", required=True, help="Path to IC panel CSV.")
    p.add_argument("--min_tstat", type=float, default=3.0, help="Min |tstat_ic| threshold.")
    p.add_argument("--min_mean_ic", type=float, default=0.01, help="Min |mean_ic_oriented| threshold.")
    p.add_argument("--min_n_days", type=int, default=400, help="Minimum IC observations.")
    p.add_argument("--max_alphas", type=int, default=40, help="Max number of alphas to keep.")
    p.add_argument(
        "--no_flip",
        action="store_true",
        help="Do not flip alphas with negative mean_ic; keep sign=+1 for all.",
    )
    p.add_argument("--out", required=True, help="Output selection CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.ic_panel)
    if "alpha" not in df.columns and "alpha_name" in df.columns:
        df["alpha"] = df["alpha_name"]

    required = {"alpha", "n_days", "mean_ic", "std_ic", "tstat_ic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"IC panel missing columns: {missing}")

    df = df[df["n_days"] >= args.min_n_days].copy()

    df["sign"] = 1.0
    if not args.no_flip:
        df.loc[df["mean_ic"] < 0, "sign"] = -1.0

    df["mean_ic_oriented"] = df["mean_ic"] * df["sign"]

    df = df[df["mean_ic_oriented"].abs() >= args.min_mean_ic]
    df = df[df["tstat_ic"].abs() >= args.min_tstat]

    df = df.sort_values("mean_ic_oriented", ascending=False)
    if args.max_alphas is not None and args.max_alphas > 0:
        df = df.head(args.max_alphas)

    out_cols = ["alpha", "sign", "n_days", "mean_ic", "tstat_ic", "mean_ic_oriented"]
    df_out = df[out_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[alphas101_select_v0] Selected {len(df_out)} alphas -> {out_path}")


if __name__ == "__main__":
    main()

