#!/usr/bin/env python
import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml


DEFAULT_BASE_CONFIG = "configs/runs/btc_daily_ma_5_40_v25_tv60_cash_yield.yaml"
DEFAULT_RUN_ROOT = Path("artifacts/runs")


def run_backtest_with_config(config_path: Path) -> None:
    cmd = [sys.executable, "scripts/run_backtest.py", "--config", str(config_path)]
    print(f"[run_btc_eth_daily_ma_5_40_v0] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BTC/ETH daily MA 5/40 baseline (research-only) using existing engine config."
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=DEFAULT_BASE_CONFIG,
        help="Base BTC daily MA 5/40 config YAML.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTC-USD", "ETH-USD"],
        help="Symbols to run (default: BTC-USD ETH-USD).",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(DEFAULT_RUN_ROOT),
        help="Root directory for run artifacts (must match engine's default).",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="bars_1d_midcap_clean",
        help="Daily bars table to use (default: bars_1d_midcap_clean).",
    )
    parser.add_argument(
        "--run_prefix",
        type=str,
        default="btc_eth_daily_ma_5_40_v0_",
        help="Prefix for run_id / run_name.",
    )

    args = parser.parse_args()
    base_config_path = Path(args.base_config)

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with base_config_path.open("r") as f:
        base_cfg = yaml.safe_load(f)

    for symbol in args.symbols:
        cfg = copy.deepcopy(base_cfg)

        # Override data source to use deduped daily view
        cfg["data"]["table"] = args.table
        cfg["data"]["symbol"] = symbol

        # Try to set a distinctive run_id / run_name if present in config
        run_stub = symbol.lower().replace("-", "_")
        run_id = f"{args.run_prefix}{run_stub}"
        if "run_id" in cfg:
            cfg["run_id"] = run_id
        elif "run_name" in cfg:
            cfg["run_name"] = run_id
        else:
            # fall back to adding a top-level field; engine may or may not use it
            cfg["run_id"] = run_id

        # Write temp config under artifacts to avoid cluttering configs/runs
        out_dir = Path(args.out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_cfg_path = out_dir / f"{run_id}.yaml"

        with tmp_cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f)

        print(
            f"[run_btc_eth_daily_ma_5_40_v0] Symbol={symbol} run_id={run_id} cfg={tmp_cfg_path}"
        )
        run_backtest_with_config(tmp_cfg_path)


if __name__ == "__main__":
    main()

