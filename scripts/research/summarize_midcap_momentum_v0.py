#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def infer_symbol_from_run_id(run_id: str, prefix: str) -> str:
    """
    run_id: e.g. "midcap_momentum_v0_sol_usd_20260101T190505Z"
    prefix: "midcap_momentum_v0_"
    -> "SOL-USD"
    """
    if not run_id.startswith(prefix):
        return "UNKNOWN"

    suffix = run_id[len(prefix) :]  # "sol_usd_20260101T190505Z"
    parts = suffix.split("_")
    if len(parts) < 2:
        return "UNKNOWN"

    # last token is timestamp; rest are symbol tokens
    sym_tokens = parts[:-1]
    return "-".join(t.upper() for t in sym_tokens)


def load_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    with summary_path.open("r") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise ValueError(f"summary.json in {run_dir} is not a dict")
    return summary


def collect_rows(run_root: Path, prefix: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        run_id = child.name
        if not run_id.startswith(prefix):
            continue

        try:
            summary = load_summary(child)
        except Exception as e:
            print(f"[WARN] Skipping {run_id}: {e}")
            continue

        symbol = infer_symbol_from_run_id(run_id, prefix)

        # take only scalar metrics from summary
        scalars: Dict[str, Any] = {}
        for k, v in summary.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                scalars[k] = v
            else:
                # non-scalar: store stringified version (optional)
                scalars[k] = json.dumps(v)

        row = {"run_id": run_id, "symbol": symbol}
        row.update(scalars)
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize midcap momentum v0 runs into a single CSV."
    )
    parser.add_argument(
        "--run_root",
        type=str,
        default="artifacts/runs",
        help="Root directory containing run_id subdirectories.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="midcap_momentum_v0_",
        help="Run_id prefix for midcap momentum v0 runs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/research/midcap_momentum_v0/summary.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()
    run_root = Path(args.run_root)
    out_path = Path(args.out)

    if not run_root.exists():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    rows = collect_rows(run_root, args.prefix)
    if not rows:
        raise RuntimeError(f"No runs found under {run_root} with prefix={args.prefix}")

    df = pd.DataFrame(rows).sort_values(["symbol", "run_id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[summarize_midcap_momentum_v0] Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

