from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import polars as pl
import subprocess
import yaml


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_temp_config(cfg: dict) -> Path:
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    Path(tmp_path).write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return Path(tmp_path)


def run_backtest(cfg_path: Path) -> Dict[str, float]:
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
        raise RuntimeError(f"run_backtest failed with code {result.returncode}")
    run_id = None
    for line in result.stdout.splitlines():
        if line.startswith("Run ID:"):
            run_id = line.split("Run ID:")[1].strip()
    if not run_id:
        raise RuntimeError("Could not parse Run ID")
    run_dir = Path("artifacts") / "runs" / run_id
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    equity_path = run_dir / "equity.parquet"
    turnover_events = 0
    if equity_path.exists():
        eq = pl.read_parquet(equity_path)
        if "turnover" in eq.columns:
            turnover_events = (eq["turnover"].fill_null(0) > 0).sum()
    return {
        "run_id": run_id,
        "total_return": summary.get("total_return"),
        "sharpe": summary.get("sharpe"),
        "max_drawdown": summary.get("max_drawdown"),
        "turnover_events": turnover_events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check look-ahead bias via execution lag sweep.")
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument("--lags", default="1,2", help="Comma-separated lags to test, e.g. 1,2")
    args = parser.parse_args()

    base_cfg = load_config(Path(args.config))
    lags = [int(x.strip()) for x in args.lags.split(",") if x.strip()]

    results: List[Dict[str, float]] = []
    for lag in lags:
        cfg = json.loads(json.dumps(base_cfg))
        cfg.setdefault("execution", {})
        cfg["execution"]["execution_lag_bars"] = lag
        cfg_path = write_temp_config(cfg)
        metrics = run_backtest(cfg_path)
        metrics["lag"] = lag
        results.append(metrics)

    print(f"{'Lag':<6}{'TotalRet':>12}{'Sharpe':>12}{'MaxDD':>12}{'TurnEvts':>10}")
    for r in results:
        print(
            f"{r['lag']:<6}{r['total_return']:>12.4f}{r['sharpe']:>12.4f}{r['max_drawdown']:>12.4f}{r['turnover_events']:>10}"
        )

    if len(results) >= 2:
        r0, r1 = results[0], results[1]
        if abs((r0.get("sharpe") or 0) - (r1.get("sharpe") or 0)) > 2.0:
            print("Warning: Possible look-ahead bias (Sharpe changed by > 2.0 between lags)")


if __name__ == "__main__":
    main()

