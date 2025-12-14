from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.engine import BacktestEngine
from common.config import (
    RunConfig,
    load_config_from_yaml,
    make_run_id,
    write_manifest,
)
from data.portal import DataPortal
from execution.sim import ExecutionSim
from risk.risk_manager import RiskManager
from strategy.ma_cross_vol_hysteresis import MACrossVolHysteresis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC-USD hourly backtest.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: RunConfig = load_config_from_yaml(args.config)

    run_id = make_run_id(cfg.run_name)
    run_dir = Path("artifacts") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_portal = DataPortal(cfg.data)
    strategy = MACrossVolHysteresis(cfg.strategy)
    risk_manager = RiskManager(cfg.risk)
    exec_sim = ExecutionSim(
        fee_bps=cfg.execution.fee_bps,
        slippage_bps=cfg.execution.slippage_bps,
    )
    engine = BacktestEngine(cfg, strategy, risk_manager, data_portal, exec_sim)

    portfolio, summary = engine.run()
    frames = portfolio.to_frames()

    bars = data_portal.load_bars()
    manifest = write_manifest(
        run_dir,
        cfg,
        bars_start=bars[0, "ts"],
        bars_end=bars.item(bars.height - 1, "ts"),
    )

    frames["equity"].write_parquet(run_dir / "equity.parquet")
    frames["positions"].write_parquet(run_dir / "positions.parquet")
    frames["trades"].write_parquet(run_dir / "trades.parquet")
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Run ID: {run_id}")
    print(f"Manifest written: {manifest['config_hash']}")
    print(f"Total return: {summary['total_return']:.4f}")
    print(f"Sharpe: {summary['sharpe']:.4f}")
    print(f"Max drawdown: {summary['max_drawdown']:.4f}")


if __name__ == "__main__":
    main()

