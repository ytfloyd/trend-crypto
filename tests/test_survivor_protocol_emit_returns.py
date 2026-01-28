import csv
from pathlib import Path

import polars as pl

from scripts.run_survivor_protocol_universe import verify_spread_returns_artifacts


def test_verify_spread_returns_artifacts(tmp_path: Path):
    survivors_csv = tmp_path / "gatekeeper_survivors.csv"
    rows = [
        {"alpha": "alpha_a", "verdict": "PASS"},
        {"alpha": "alpha_b", "verdict": "PASS"},
    ]
    with open(survivors_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "verdict"])
        writer.writeheader()
        writer.writerows(rows)

    survivor_dir = tmp_path / "survivors"
    for alpha in ["alpha_a", "alpha_b"]:
        alpha_dir = survivor_dir / alpha
        alpha_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame({"ts": ["2024-01-01"], "spread_ret": [0.01]})
        df.write_parquet(alpha_dir / f"{alpha}.spread_returns.parquet")

    verify_spread_returns_artifacts(survivors_csv, survivor_dir)
