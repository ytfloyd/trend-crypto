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
from common.timeframe import hours_per_bar, periods_per_year_from_timeframe
from data.portal import DataPortal
from risk.risk_manager import RiskManager
from strategy.ma_cross_vol_hysteresis import MACrossVolHysteresis
from strategy.buy_and_hold import BuyAndHoldStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC-USD hourly backtest.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: RunConfig = load_config_from_yaml(args.config)

    bar_hours = hours_per_bar(cfg.data.timeframe)
    ppy = periods_per_year_from_timeframe(cfg.data.timeframe)

    # Scale strategy windows if provided in hours
    if cfg.strategy.mode != "buy_and_hold" and cfg.strategy.window_units == "hours":
        orig_fast = cfg.strategy.fast
        orig_slow = cfg.strategy.slow
        orig_vol = cfg.strategy.vol_window
        cfg.strategy.fast = max(1, round(orig_fast / bar_hours))
        cfg.strategy.slow = max(cfg.strategy.fast + 1, round(orig_slow / bar_hours))
        cfg.strategy.vol_window = max(2, round(orig_vol / bar_hours))
    if cfg.risk.window_units == "hours":
        orig_risk_vol = cfg.risk.vol_window
        cfg.risk.vol_window = max(2, round(orig_risk_vol / bar_hours))

    print(
        f"tf={cfg.data.timeframe} bar_hours={bar_hours:.4f} ppy={ppy:.2f} "
        f"fast={cfg.strategy.fast} slow={cfg.strategy.slow} vol_window={cfg.strategy.vol_window} "
        f"risk_vol_window={cfg.risk.vol_window}"
    )

    run_id = make_run_id(cfg.run_name)
    run_dir = Path("artifacts") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_portal = DataPortal(cfg.data)
    if cfg.strategy.mode == "buy_and_hold":
        strategy = BuyAndHoldStrategy(cfg.strategy)
    else:
        strategy = MACrossVolHysteresis(cfg.strategy)
    risk_manager = RiskManager(cfg.risk, ppy)
    engine = BacktestEngine(cfg, strategy, risk_manager, data_portal)

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

