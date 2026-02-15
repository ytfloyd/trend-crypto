#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import copy
import subprocess
from typing import List

import yaml


DEFAULT_BASE_CONFIG = "configs/runs/btc_daily_ma_5_40_v25_tv60_cash_yield.yaml"
DEFAULT_RUN_ROOT = Path("artifacts/runs")


def run_backtest_with_config(config_path: Path) -> str | None:
    """Run the engine and return the run_id parsed from stdout (or None)."""
    cmd = [sys.executable, "scripts/run_backtest.py", "--config", str(config_path)]
    print(f"[run_btc_eth_daily_ma_5_40_v0] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    for line in result.stdout.splitlines():
        if line.startswith("Run ID:"):
            return line.split("Run ID:")[1].strip()
    return None


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
    parser.add_argument("--no_html", action="store_true", help="Skip HTML tearsheet generation.")

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
        engine_run_id = run_backtest_with_config(tmp_cfg_path)

        # --- HTML tearsheet ---
        if not args.no_html and engine_run_id:
            try:
                import pandas as _pd
                eq_path = Path("artifacts/runs") / engine_run_id / "equity.parquet"
                if eq_path.exists():
                    from tearsheet_common_v0 import build_standard_html_tearsheet
                    eq_df = _pd.read_parquet(eq_path)
                    nav_col = "nav" if "nav" in eq_df.columns else "equity"
                    strat_eq = (
                        eq_df.set_index("ts")[nav_col]
                        .rename("equity")
                        .pipe(lambda s: s.set_axis(_pd.to_datetime(s.index)))
                        .sort_index()
                    )
                    build_standard_html_tearsheet(
                        out_html=eq_path.parent / "tearsheet.html",
                        strategy_label=f"BTC/ETH Daily MA(5/40) – {symbol}",
                        strategy_equity=strat_eq,
                        equity_csv_path=str(eq_path),
                        subtitle=f"Engine-based MA(5/40) baseline for {symbol}",
                    )
            except Exception as exc:
                print(f"[run_btc_eth_daily_ma_5_40_v0] HTML tearsheet skipped: {exc}")


if __name__ == "__main__":
    main()

