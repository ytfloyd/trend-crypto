#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_cagr(start: datetime, end: datetime, total_return: float) -> float:
    years = (end - start).days / 365.0
    if years <= 0:
        raise ValueError("Invalid time range for CAGR calculation.")
    growth_multiple = 1.0 + total_return
    return growth_multiple ** (1.0 / years) - 1.0


def build_row(manifest: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    start = parse_dt(manifest["time_range"]["start"])
    end = parse_dt(manifest["time_range"]["end"])
    total_return = float(summary["total_return"])
    cagr = compute_cagr(start, end, total_return)
    max_drawdown = float(summary["max_drawdown"])
    return {
        "symbol": manifest.get("symbol"),
        "start": manifest["time_range"]["start"],
        "end": manifest["time_range"]["end"],
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": summary.get("sharpe"),
        "max_dd": max_drawdown,
        "max_drawdown": max_drawdown,
        "entry_exit_events": summary.get("entry_exit_events"),
        "avg_cash_weight": summary.get("avg_cash_weight"),
        "cash_yield_annual": summary.get("cash_yield_annual"),
        "config_hash": manifest.get("config_hash"),
        "git_hash": manifest.get("git_hash"),
        "generated_at": manifest.get("generated_at"),
    }


def write_csv(path: Path, row: Dict[str, Any]) -> None:
    columns = [
        "symbol",
        "start",
        "end",
        "total_return",
        "cagr",
        "sharpe",
        "max_dd",
        "max_drawdown",
        "entry_exit_events",
        "avg_cash_weight",
        "cash_yield_annual",
        "config_hash",
        "git_hash",
        "generated_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate MA(5/40) baseline metrics CSV from manifest.json and summary.json.\n\n"
            "Examples:\n"
            "  python scripts/research/regenerate_ma_5_40_baseline_metrics_v0.py "
            "--run_dir artifacts/research/ma_5_40_btc_eth_baseline_v0/btc_usd\n"
            "  python scripts/research/regenerate_ma_5_40_baseline_metrics_v0.py "
            "--run_dir artifacts/research/ma_5_40_btc_eth_baseline_v0/eth_usd"
        )
    )
    parser.add_argument("--run_dir", required=True, help="Baseline folder containing manifest.json + summary.json.")
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path (default: <run_dir>/metrics_ma_5_40_baseline_v0.csv).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.json at {manifest_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json at {summary_path}")

    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "metrics_ma_5_40_baseline_v0.csv"

    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    row = build_row(manifest, summary)
    write_csv(out_csv, row)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
