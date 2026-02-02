#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tearsheet_common_v0 import compute_stats, load_equity_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare simple long-only baseline vs MA(5/40) baseline."
    )
    p.add_argument(
        "--simple_dir",
        type=str,
        default="artifacts/research/transtrend_crypto_simple_baseline",
        help="Directory containing simple baseline equity.csv",
    )
    p.add_argument(
        "--ma_dir",
        type=str,
        required=True,
        help="Directory containing MA(5/40) equity.csv",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/research/compare_simple_vs_ma_5_40",
        help="Output directory for comparison artifacts",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plotting normalized equity.",
    )
    return p.parse_args()


def _load_equity(path: Path) -> pd.Series:
    return load_equity_csv(str(path))


def _normalize(eq: pd.Series) -> pd.Series:
    return eq / eq.iloc[0] if len(eq) > 0 else eq


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    simple_dir = Path(args.simple_dir)
    ma_dir = Path(args.ma_dir)

    simple_eq_path = simple_dir / "equity.csv"
    ma_eq_path = ma_dir / "equity.csv"
    if not simple_eq_path.exists():
        raise FileNotFoundError(f"Missing equity.csv at {simple_eq_path}")
    if not ma_eq_path.exists():
        raise FileNotFoundError(f"Missing equity.csv at {ma_eq_path}")

    simple_eq = _load_equity(simple_eq_path)
    ma_eq = _load_equity(ma_eq_path)

    common_idx = simple_eq.index.intersection(ma_eq.index)
    if common_idx.empty:
        raise ValueError("No overlapping timestamps between the two equity series.")

    simple_eq = simple_eq.reindex(common_idx).dropna()
    ma_eq = ma_eq.reindex(common_idx).dropna()

    # Align again after dropna to keep identical index
    common_idx = simple_eq.index.intersection(ma_eq.index)
    simple_eq = simple_eq.reindex(common_idx)
    ma_eq = ma_eq.reindex(common_idx)

    simple_stats = compute_stats(simple_eq)
    ma_stats = compute_stats(ma_eq)

    summary = pd.DataFrame(
        [
            {
                "strategy": "simple_baseline_long_only",
                **simple_stats,
                "start": simple_eq.index.min(),
                "end": simple_eq.index.max(),
            },
            {
                "strategy": "ma_5_40_baseline",
                **ma_stats,
                "start": ma_eq.index.min(),
                "end": ma_eq.index.max(),
            },
        ]
    )
    summary_path = out_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    compare_df = pd.DataFrame(
        {
            "ts": common_idx,
            "simple_equity": simple_eq.values,
            "ma_equity": ma_eq.values,
            "simple_equity_norm": _normalize(simple_eq).values,
            "ma_equity_norm": _normalize(ma_eq).values,
            "simple_ret": simple_eq.pct_change().fillna(0.0).values,
            "ma_ret": ma_eq.pct_change().fillna(0.0).values,
        }
    )
    compare_path = out_dir / "equity_comparison.csv"
    compare_df.to_csv(compare_path, index=False)

    if not args.no_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(compare_df["ts"], compare_df["simple_equity_norm"], label="Simple Baseline")
        plt.plot(compare_df["ts"], compare_df["ma_equity_norm"], label="MA(5/40) Baseline")
        plt.title("Normalized Equity Comparison")
        plt.xlabel("Timestamp (UTC)")
        plt.ylabel("Equity (normalized)")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "equity_comparison.png", dpi=150)
        plt.close()

    print(f"Wrote {summary_path}")
    print(f"Wrote {compare_path}")


if __name__ == "__main__":
    main()
