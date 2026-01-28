#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full-universe Survivor Protocol")
    p.add_argument("--db", required=True, help="DuckDB path")
    p.add_argument("--price_table", default="bars_1d_clean", help="Price table/view")
    p.add_argument("--start", default="2023-01-01", help="Start date")
    p.add_argument("--end", default="2024-12-31", help="End date")
    p.add_argument(
        "--alphas_out",
        default="artifacts/research/101_alphas/alphas_101_v0.parquet",
        help="Output alpha panel parquet",
    )
    p.add_argument(
        "--out_dir",
        default="artifacts/research/101_alphas/tearsheets_v0",
        help="Output directory for survivor artifacts",
    )
    p.add_argument("--max_alphas", type=int, default=None, help="Optional alpha cap")
    p.add_argument("--emit-returns", action="store_true", help="Emit spread returns for survivors")
    return p.parse_args()


def _read_gatekeeper_counts(path: Path) -> tuple[int, int, int, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_total = len(rows)
    survivors = [r["alpha"] for r in rows if r.get("verdict") == "PASS"]
    n_pass = len(survivors)
    n_fail = n_total - n_pass
    return n_total, n_pass, n_fail, survivors


def verify_spread_returns_artifacts(survivors_csv: Path, survivor_dir: Path) -> None:
    if not survivors_csv.exists():
        raise FileNotFoundError(f"Missing gatekeeper_survivors.csv: {survivors_csv}")
    with open(survivors_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    survivors = [r["alpha"] for r in rows if r.get("verdict") == "PASS"]
    missing = []
    for alpha in survivors:
        spread_path = survivor_dir / alpha / f"{alpha}.spread_returns.parquet"
        if not spread_path.exists():
            missing.append(alpha)
    if missing:
        raise RuntimeError(
            "Missing spread return artifacts for: "
            + ", ".join(missing)
            + ". Re-run with --emit-returns."
        )


def main() -> None:
    args = parse_args()

    alphas_out = Path(args.alphas_out)
    out_dir = Path(args.out_dir)
    alphas_out.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: compute alpha panel
    cmd_compute = [
        sys.executable,
        "scripts/research/run_101_alphas_compute_v0.py",
        "--db",
        args.db,
        "--table",
        args.price_table,
        "--start",
        args.start,
        "--end",
        args.end,
        "--out",
        str(alphas_out),
    ]
    subprocess.run(cmd_compute, check=True)

    if not alphas_out.exists():
        raise SystemExit(f"Alpha panel not created: {alphas_out}")

    # Step 2: run survivor protocol
    cmd_batch = [
        sys.executable,
        "scripts/batch_alpha_tearsheets.py",
        "--alphas",
        str(alphas_out),
        "--db",
        args.db,
        "--price_table",
        args.price_table,
        "--out_dir",
        str(out_dir),
    ]
    if args.max_alphas is not None:
        cmd_batch += ["--max_alphas", str(args.max_alphas)]
    if args.emit_returns:
        cmd_batch += ["--emit-returns"]
    subprocess.run(cmd_batch, check=True)

    all_csv = out_dir / "gatekeeper_all.csv"
    survivors_csv = out_dir / "gatekeeper_survivors.csv"
    if not all_csv.exists():
        raise SystemExit(f"Missing gatekeeper_all.csv: {all_csv}")
    if not survivors_csv.exists():
        raise SystemExit(f"Missing gatekeeper_survivors.csv: {survivors_csv}")

    n_total, n_pass, n_fail, survivors = _read_gatekeeper_counts(all_csv)
    if args.emit_returns:
        verify_spread_returns_artifacts(survivors_csv, out_dir / "survivors")
        print("Verified spread return artifacts exist for all survivors.")

    print("=" * 72)
    print("Survivor Protocol (Full Universe)")
    print("=" * 72)
    print(f"alphas_out: {alphas_out}")
    print(f"out_dir: {out_dir}")
    print(f"n_total={n_total}")
    print(f"n_pass={n_pass}")
    print(f"n_fail={n_fail}")
    if survivors:
        print("passing alphas:")
        for name in survivors:
            print(f"  {name}")


if __name__ == "__main__":
    main()
