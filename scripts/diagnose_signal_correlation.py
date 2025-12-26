from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def load_positions(run_dir: Path) -> pl.DataFrame:
    path = run_dir / "positions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing positions.parquet in {run_dir}")
    return pl.read_parquet(path).select(["ts", "weight"]).sort("ts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose signal correlation between two runs.")
    parser.add_argument("--run_a", required=True, help="Path to first run dir (e.g., BTC)")
    parser.add_argument("--run_b", required=True, help="Path to second run dir (e.g., ETH)")
    parser.add_argument("--out_json", required=True, help="Path to write JSON summary")
    args = parser.parse_args()

    pos_a = load_positions(Path(args.run_a)).rename({"weight": "weight_a"})
    pos_b = load_positions(Path(args.run_b)).rename({"weight": "weight_b"})

    joined = pos_a.join(pos_b, on="ts", how="inner").drop_nulls().sort("ts")
    if joined.is_empty():
        raise ValueError("No overlapping timestamps between runs.")

    long_a = (joined["weight_a"] > 0).cast(pl.Int8)
    long_b = (joined["weight_b"] > 0).cast(pl.Int8)

    n = len(joined)
    n11 = int(((long_a == 1) & (long_b == 1)).sum())
    n10 = int(((long_a == 1) & (long_b == 0)).sum())
    n01 = int(((long_a == 0) & (long_b == 1)).sum())
    n00 = int(((long_a == 0) & (long_b == 0)).sum())

    overlap_frac = n11 / n if n > 0 else 0.0
    p_a = (n11 + n10) / n if n > 0 else 0.0
    p_b = (n11 + n01) / n if n > 0 else 0.0

    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    denom = denom ** 0.5
    phi = ((n11 * n00 - n10 * n01) / denom) if denom > 0 else 0.0
    jaccard = n11 / (n11 + n10 + n01) if (n11 + n10 + n01) > 0 else 0.0

    avg_weight_a_when_long = (
        joined.filter(pl.col("weight_a") > 0)["weight_a"].mean() if n11 + n10 > 0 else 0.0
    )
    avg_weight_b_when_long = (
        joined.filter(pl.col("weight_b") > 0)["weight_b"].mean() if n11 + n01 > 0 else 0.0
    )

    summary = {
        "samples": n,
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "overlap_frac": overlap_frac,
        "p_a_long": p_a,
        "p_b_long": p_b,
        "phi": phi,
        "jaccard": jaccard,
        "avg_weight_a_when_long": avg_weight_a_when_long,
        "avg_weight_b_when_long": avg_weight_b_when_long,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

