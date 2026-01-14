#!/usr/bin/env python
from __future__ import annotations

"""
Placeholder runner for kuma_trend v0.

NOTE: The original backtest implementation is not present in this checkout.
This runner writes a run manifest for provenance and warns that no backtest
was executed. It is provided to keep registry recipes reproducible insofar as
tracking provenance; it should be replaced with the actual backtest logic if/when
restored.
"""

import argparse
import sys
from pathlib import Path

from run_manifest_v0 import build_base_manifest, fingerprint_file, write_run_manifest, hash_config_blob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="(Placeholder) kuma_trend backtest v0")
    p.add_argument("--db", required=True, help="Path to DuckDB database.")
    p.add_argument("--table", required=True, help="Table/view for prices.")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    p.add_argument("--out_dir", default="artifacts/research/kuma_trend", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_base_manifest(
        strategy_id="kuma_trend_v0",
        argv=sys.argv,
        repo_root=Path(__file__).resolve().parents[2],
    )
    manifest.update(
        {
            "config": vars(args),
            "config_hash": hash_config_blob(vars(args)),
            "data_sources": {
                "duckdb": fingerprint_file(args.db),
                "price_table": args.table,
            },
            "universe": args.table,
            "artifacts_written": {},
            "warning": "No backtest executed; runner is a placeholder pending restored kuma_trend logic.",
        }
    )
    manifest_path = out_dir / "run_manifest.json"
    write_run_manifest(manifest_path, manifest)
    print(f"[kuma_trend_runner] Wrote placeholder run manifest to {manifest_path}")
    print("[kuma_trend_runner] WARNING: No backtest executed; implement strategy logic to produce artifacts.")


if __name__ == "__main__":
    main()
