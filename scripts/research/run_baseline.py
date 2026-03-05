#!/usr/bin/env python
"""Record a versioned baseline of all existing backtest run metrics.

Scans artifacts/runs/ for summary.json + manifest.json pairs, extracts
key metrics, and writes a single JSON snapshot. This baseline is diffed
after each implementation phase to detect regressions.

Usage
-----
    python scripts/research/run_baseline.py --save baseline_pre_phase1.json
    python scripts/research/run_baseline.py --save baseline_pre_phase2.json
    python scripts/research/run_baseline.py --diff baseline_pre_phase1.json baseline_pre_phase2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / ".." / "artifacts" / "runs"
BASELINES_DIR = Path(__file__).resolve().parents[1] / ".." / "artifacts" / "baselines"


def collect_run_metrics(artifacts_dir: Path) -> list[dict]:
    """Scan all run directories for summary.json + manifest.json pairs."""
    runs = []
    for run_dir in sorted(artifacts_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        if not summary_path.exists() or not manifest_path.exists():
            continue

        with summary_path.open() as f:
            summary = json.load(f)
        with manifest_path.open() as f:
            manifest = json.load(f)

        run_name = manifest.get("params", {}).get("run_name", run_dir.name)
        symbol = manifest.get("symbol", "unknown")
        config_hash = manifest.get("config_hash", "")

        runs.append({
            "run_dir": run_dir.name,
            "run_name": run_name,
            "symbol": symbol,
            "config_hash": config_hash,
            "total_return": summary.get("total_return"),
            "sharpe": summary.get("sharpe"),
            "max_drawdown": summary.get("max_drawdown"),
            "entry_exit_events": summary.get("entry_exit_events"),
            "generated_at": manifest.get("generated_at"),
        })
    return runs


def save_baseline(runs: list[dict], out_path: Path) -> None:
    """Write baseline snapshot to JSON."""
    snapshot = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"[baseline] Saved {len(runs)} runs -> {out_path}")


def diff_baselines(path_a: Path, path_b: Path) -> None:
    """Compare two baseline snapshots and report differences."""
    with path_a.open() as f:
        a = json.load(f)
    with path_b.open() as f:
        b = json.load(f)

    runs_a = {r["run_dir"]: r for r in a["runs"]}
    runs_b = {r["run_dir"]: r for r in b["runs"]}

    common = sorted(set(runs_a) & set(runs_b))
    only_a = sorted(set(runs_a) - set(runs_b))
    only_b = sorted(set(runs_b) - set(runs_a))

    if only_a:
        print(f"\n  Runs only in {path_a.name}: {len(only_a)}")
        for r in only_a[:5]:
            print(f"    {r}")
    if only_b:
        print(f"\n  Runs only in {path_b.name}: {len(only_b)}")
        for r in only_b[:5]:
            print(f"    {r}")

    diffs_found = False
    metrics_to_check = ["total_return", "sharpe", "max_drawdown"]
    print(f"\n  Comparing {len(common)} common runs...")
    print(f"  {'Run':<60s} {'Metric':<15s} {'Before':>12s} {'After':>12s} {'Delta':>12s}")
    print("  " + "-" * 111)

    for run_dir in common:
        ra, rb = runs_a[run_dir], runs_b[run_dir]
        for metric in metrics_to_check:
            va = ra.get(metric)
            vb = rb.get(metric)
            if va is None or vb is None:
                continue
            if abs(va - vb) > 1e-10:
                diffs_found = True
                delta = vb - va
                print(f"  {run_dir:<60s} {metric:<15s} {va:>12.6f} {vb:>12.6f} {delta:>+12.6f}")

    if not diffs_found:
        print("  No metric differences detected. Baseline is clean.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or diff backtest baselines.")
    parser.add_argument("--save", metavar="FILENAME", help="Save baseline to artifacts/baselines/FILENAME")
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"), help="Diff two baseline files")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR), help="Run artifacts directory")
    args = parser.parse_args()

    if args.diff:
        path_a = BASELINES_DIR / args.diff[0] if not Path(args.diff[0]).is_absolute() else Path(args.diff[0])
        path_b = BASELINES_DIR / args.diff[1] if not Path(args.diff[1]).is_absolute() else Path(args.diff[1])
        diff_baselines(path_a, path_b)
    elif args.save:
        artifacts_dir = Path(args.artifacts_dir)
        if not artifacts_dir.exists():
            print(f"[baseline] Artifacts directory not found: {artifacts_dir}", file=sys.stderr)
            sys.exit(1)
        runs = collect_run_metrics(artifacts_dir)
        out_path = BASELINES_DIR / args.save
        save_baseline(runs, out_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
