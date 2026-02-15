from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import subprocess
from typing import List

import yaml

DEFAULT_SYMBOLS: List[str] = [
    "SOL-USD",
    "ETC-USD",
    "LINK-USD",
    "BCH-USD",
    "SUI-USD",
    "UNI-USD",
    "SUSHI-USD",
    "ALGO-USD",
]
TMP_DIR = Path("artifacts/tmp")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_temp_config(cfg: dict, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{cfg['run_name']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def run_backtest(cfg_path: Path) -> str:
    result = subprocess.run(
        ["python", "scripts/run_backtest.py", "--config", str(cfg_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[run_backtest] FAILED for {cfg_path}")
        print("---- STDOUT ----")
        print(result.stdout)
        print("---- STDERR ----")
        print(result.stderr)
        raise RuntimeError(f"run_backtest failed for {cfg_path}")

    run_id = None
    for line in result.stdout.splitlines():
        if line.startswith("Run ID:"):
            run_id = line.split("Run ID:")[1].strip()
            break
    if not run_id:
        print(result.stdout)
        raise RuntimeError(f"Could not parse Run ID from run_backtest output for {cfg_path}")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch midcap momentum (daily MA 5/40 long-only) backtests."
    )
    parser.add_argument(
        "--config",
        default="configs/research/midcap_daily_ma_5_40_v0.yaml",
        help="Base YAML config to clone per symbol.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=DEFAULT_SYMBOLS,
        help="Symbols to run. Defaults to the midcap research set.",
    )
    parser.add_argument(
        "--start",
        help="Override data.start in the config (ISO8601 UTC).",
    )
    parser.add_argument(
        "--end",
        help="Override data.end in the config (ISO8601 UTC).",
    )
    parser.add_argument("--no_html", action="store_true", help="Skip HTML tearsheet generation.")
    args = parser.parse_args()

    base_cfg = load_config(Path(args.config))
    symbols = args.symbols or DEFAULT_SYMBOLS

    for symbol in symbols:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["data"]["symbol"] = symbol
        if args.start:
            cfg["data"]["start"] = args.start
        if args.end:
            cfg["data"]["end"] = args.end
        cfg["run_name"] = f"midcap_momentum_v0_{symbol.replace('-', '_').lower()}"

        cfg_path = write_temp_config(cfg, TMP_DIR)
        run_id = run_backtest(cfg_path)
        print(f"{symbol}: run_id={run_id}")

        # --- HTML tearsheet ---
        if not args.no_html and run_id:
            try:
                import pandas as _pd
                eq_path = Path("artifacts/runs") / run_id / "equity.parquet"
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
                        strategy_label=f"Midcap Momentum – {symbol}",
                        strategy_equity=strat_eq,
                        equity_csv_path=str(eq_path),
                        subtitle=f"Daily MA(5/40) long-only momentum for {symbol}",
                    )
            except Exception as exc:
                print(f"[midcap_momentum] HTML tearsheet skipped for {symbol}: {exc}")


if __name__ == "__main__":
    main()

# Commands to test:
# python scripts/research/create_midcap_daily_clean_view.py
# python scripts/research/run_midcap_momentum_v0.py --config configs/research/midcap_daily_ma_5_40_v0.yaml

