#!/usr/bin/env python
"""Regenerate all summary.json files from equity.parquet using the corrected engine.

This script:
1. Reads each equity.parquet
2. Recomputes summary stats using the corrected _summary_stats function
3. Independently verifies with numpy (no Polars, no engine code)
4. Writes corrected summary.json (preserving non-metric fields)
5. Produces a diff report showing what changed

Usage:
    python scripts/regenerate_all_summaries.py [--dry-run] [--dir artifacts/runs]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backtest.engine import _summary_stats  # noqa: E402


METRIC_KEYS = {"total_return", "total_return_decimal", "total_return_pct",
               "total_return_multiple", "sharpe", "max_drawdown"}


def verify_with_numpy(eq_path: Path) -> dict:
    """Independent verification using only numpy — no Polars, no engine code."""
    eq = pl.read_parquet(eq_path)
    nav = eq["nav"].to_numpy()
    ts = eq["ts"].to_list()

    total_return = nav[-1] / nav[0] - 1.0 if nav[0] != 0 else 0.0

    rets = np.diff(nav) / nav[:-1]

    if len(ts) > 1:
        dt_seconds = [(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts)-1)]
        dt = np.median(dt_seconds)
        ppy = 365 * 24 * 3600 / dt if dt > 0 else 365.0
    else:
        ppy = 365.0

    if len(rets) > 1:
        mean = np.mean(rets)
        std = np.std(rets, ddof=1)
        sharpe = (mean / std) * np.sqrt(ppy) if std > 1e-12 else 0.0
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(nav)
    drawdowns = nav / running_max - 1.0
    max_dd = float(np.min(drawdowns))

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "periods_per_year": ppy,
    }


def regenerate(runs_dir: Path, dry_run: bool = False) -> None:
    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())

    total = 0
    changed = 0
    verified = 0
    failures = []

    for run_dir in run_dirs:
        eq_path = run_dir / "equity.parquet"
        sum_path = run_dir / "summary.json"
        if not eq_path.exists() or not sum_path.exists():
            continue

        total += 1
        stored = json.loads(sum_path.read_text())

        # Recompute with corrected engine
        eq = pl.read_parquet(eq_path)
        new_stats = _summary_stats(eq)

        # Independent numpy verification
        np_stats = verify_with_numpy(eq_path)
        sharpe_match = abs(new_stats["sharpe"] - np_stats["sharpe"]) < 0.01
        ret_match = abs(new_stats["total_return"] - np_stats["total_return"]) < 1e-6
        dd_match = abs(new_stats["max_drawdown"] - np_stats["max_drawdown"]) < 1e-6

        if not (sharpe_match and ret_match and dd_match):
            failures.append({
                "run": run_dir.name,
                "engine_sharpe": new_stats["sharpe"],
                "numpy_sharpe": np_stats["sharpe"],
                "engine_ret": new_stats["total_return"],
                "numpy_ret": np_stats["total_return"],
            })
            continue

        verified += 1

        # Check for changes
        metrics_changed = False
        diffs = {}
        for key in METRIC_KEYS:
            old_val = stored.get(key)
            new_val = new_stats.get(key)
            if old_val is None:
                continue
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if abs(old_val - new_val) > 1e-10:
                    metrics_changed = True
                    diffs[key] = {"old": round(old_val, 8), "new": round(new_val, 8)}

        if metrics_changed:
            changed += 1
            print(f"CHANGED: {run_dir.name}")
            for k, v in diffs.items():
                ratio = v["old"] / v["new"] if abs(v["new"]) > 1e-12 else float("inf")
                print(f"  {k}: {v['old']} -> {v['new']} (ratio: {ratio:.4f})")

            if not dry_run:
                updated = dict(stored)
                for key in METRIC_KEYS:
                    if key in new_stats:
                        updated[key] = new_stats[key]
                updated["_regenerated_at"] = datetime.now(timezone.utc).isoformat()
                updated["_regenerated_reason"] = "sharpe_annualization_fix_and_t0_return_drop"
                sum_path.write_text(json.dumps(updated, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"Total runs: {total}")
    print(f"Verified (engine == numpy): {verified}")
    print(f"Changed: {changed}")
    print(f"Unchanged: {verified - changed}")
    print(f"Verification failures: {len(failures)}")
    if dry_run:
        print("DRY RUN — no files modified")
    if failures:
        print("\nFAILURES (engine != numpy):")
        for f in failures:
            print(f"  {f['run']}: engine_sharpe={f['engine_sharpe']:.6f}, numpy_sharpe={f['numpy_sharpe']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate all summary.json files")
    parser.add_argument("--dir", default="artifacts/runs", help="Runs directory")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()
    regenerate(Path(args.dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
