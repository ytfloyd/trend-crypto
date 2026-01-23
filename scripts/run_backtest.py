from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.engine import BacktestEngine
from common.config import (
    compile_config,
    load_config_from_yaml,
    make_run_id,
    write_manifest,
)
from data.portal import DataPortal
from risk.risk_manager import RiskManager
from strategy.ma_cross_vol_hysteresis import MACrossVolHysteresis
from strategy.buy_and_hold import BuyAndHoldStrategy
from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC-USD hourly backtest.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = load_config_from_yaml(args.config)
    cfg = compile_config(raw_cfg)

    print(
        f"tf={cfg.data.timeframe} bar_hours={cfg.bar_hours:.4f} "
        f"annualization_factor={cfg.annualization_factor:.2f} "
        f"fast={cfg.strategy.fast} slow={cfg.strategy.slow} vol_window={cfg.strategy.vol_window} "
        f"risk_vol_window={cfg.risk.vol_window} "
        f"config_hash={cfg.compute_hash()}"
    )

    run_id = make_run_id(cfg.run_name)
    run_dir = Path("artifacts") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_portal = DataPortal(cfg.data)
    if cfg.strategy.mode == "buy_and_hold":
        strategy = BuyAndHoldStrategy(cfg.strategy)
    elif cfg.strategy.mode == "ma_crossover_long_only":
        strategy = MACrossoverLongOnlyStrategy(
            fast=cfg.strategy.fast,
            slow=cfg.strategy.slow,
            weight_on=cfg.strategy.weight_on,
            target_vol_annual=cfg.strategy.target_vol_annual,
            vol_lookback=cfg.strategy.vol_lookback or 20,
            max_weight=cfg.strategy.max_weight,
            enable_adx_filter=cfg.strategy.enable_adx_filter,
            adx_window=cfg.strategy.adx_window,
            adx_threshold=cfg.strategy.adx_threshold,
            adx_entry_only=cfg.strategy.adx_entry_only,
        )
    else:
        strategy = MACrossVolHysteresis(cfg.strategy)
    risk_manager = RiskManager(cfg.risk, cfg.annualization_factor)
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

