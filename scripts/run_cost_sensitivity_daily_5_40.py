from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import polars as pl
import yaml


def load_summary(run_id: str) -> Dict:
    path = Path("artifacts") / "runs" / run_id / "summary.json"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def load_turnover_events(run_id: str) -> int:
    equity_path = Path("artifacts") / "runs" / run_id / "equity.parquet"
    if not equity_path.exists():
        return 0
    eq = pl.read_parquet(equity_path)
    if "turnover" not in eq.columns:
        return 0
    return int((eq["turnover"].fill_null(0) > 0).sum())


def run_backtest(cfg_path: Path) -> str:
    result = subprocess.run(
        ["python", "scripts/run_backtest.py", "--config", str(cfg_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("run_backtest failed")
    run_id = None
    for line in result.stdout.splitlines():
        if line.startswith("Run ID:"):
            run_id = line.split("Run ID:")[1].strip()
            break
    if not run_id:
        raise RuntimeError("Could not parse run id")
    return run_id


def write_temp_config(base_cfg: Dict, fee_bps: float, slippage_bps: float, path: Path) -> Path:
    tmp = base_cfg.copy()
    tmp["execution"] = tmp.get("execution", {})
    tmp["execution"]["fee_bps"] = fee_bps
    tmp["execution"]["slippage_bps"] = slippage_bps
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(tmp, f, sort_keys=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily 5/40 cost sensitivity (long-only).")
    parser.add_argument(
        "--base_config_btc", default="configs/runs/btc_daily_ma_5_40_long_only.yaml"
    )
    parser.add_argument(
        "--base_config_eth", default="configs/runs/eth_daily_ma_5_40_long_only.yaml"
    )
    parser.add_argument("--cost_grid_bps", default="10,20,25")
    parser.add_argument(
        "--out_csv", default="artifacts/compare/daily_5_40_cost_sensitivity.csv"
    )
    args = parser.parse_args()

    cost_grid = [float(x.strip()) for x in args.cost_grid_bps.split(",") if x.strip()]
    cfg_btc_base = yaml.safe_load(Path(args.base_config_btc).read_text())
    cfg_eth_base = yaml.safe_load(Path(args.base_config_eth).read_text())

    rows: List[Dict] = []
    for cost in cost_grid:
        fee_bps = 0.0
        slippage_bps = cost

        for asset, base_cfg in [("BTC", cfg_btc_base), ("ETH", cfg_eth_base)]:
            tmp_path = Path(f"/tmp/{asset.lower()}_cost_{int(cost)}.yaml")
            write_temp_config(base_cfg, fee_bps, slippage_bps, tmp_path)
            run_id = run_backtest(tmp_path)
            summ = load_summary(run_id)
            rows.append(
                {
                    "asset": asset,
                    "total_cost_bps": cost,
                    "sharpe": summ.get("sharpe"),
                    "max_drawdown": summ.get("max_drawdown"),
                    "total_return": summ.get("total_return"),
                    "run_id": run_id,
                    "turnover_events": load_turnover_events(run_id),
                }
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

