#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capacity sensitivity table for 101-alphas V1 remediation.")
    p.add_argument(
        "--metrics_dir",
        default="artifacts/research/101_alphas",
        help="Directory containing metrics CSVs.",
    )
    p.add_argument(
        "--base_metrics",
        default="metrics_101_ensemble_filtered_v1.csv",
        help="Base metrics CSV (gross, no explicit costs).",
    )
    p.add_argument(
        "--cost_metrics_glob",
        default="metrics_101_ensemble_filtered_v1_costs_bps*.csv",
        help="Glob pattern for cost metrics files.",
    )
    p.add_argument(
        "--out",
        default="capacity_sensitivity_v1.csv",
        help="Output CSV filename (written under metrics_dir).",
    )
    return p.parse_args()


def load_metrics_file(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Metrics file empty: {path}")
    return df.iloc[0]


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)

    base_path = metrics_dir / args.base_metrics
    if not base_path.exists():
        raise FileNotFoundError(f"Base metrics not found: {base_path}")
    rows = []

    base_row = load_metrics_file(base_path).copy()
    base_row["cost_bps"] = 0
    rows.append(base_row)

    cost_files = glob.glob(str(metrics_dir / args.cost_metrics_glob))
    for fn in cost_files:
        try:
            bps_part = os.path.basename(fn).split("bps", 1)[1].split(".csv", 1)[0]
            cost_bps = int(bps_part)
        except Exception:
            continue
        row = load_metrics_file(Path(fn)).copy()
        row["cost_bps"] = cost_bps
        rows.append(row)

    if not rows:
        raise RuntimeError("No metrics rows collected.")

    out_df = pd.DataFrame(rows)
    out_df = out_df[
        [
            "cost_bps",
            "sharpe",
            "cagr",
            "total_return",
            "vol",
            "max_dd",
        ]
    ]
    # Placeholder columns for Risk to extend
    out_df["assumed_aum_cap_musd"] = pd.NA
    out_df["assumed_participation_cap"] = pd.NA

    out_path = metrics_dir / args.out
    out_df = out_df.sort_values("cost_bps").reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(f"[alphas101_capacity_sensitivity_v1] Wrote capacity table to {out_path}")


if __name__ == "__main__":
    main()

