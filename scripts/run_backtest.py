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
    parser.add_argument(
        "--include-provenance-in-summary",
        action="store_true",
        help="Include data_provenance in summary.json (default: false).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = load_config_from_yaml(args.config)
    cfg = compile_config(raw_cfg)

    raw_fast = cfg.raw.strategy.fast if cfg.raw.strategy.window_units == "hours" else None
    raw_slow = cfg.raw.strategy.slow if cfg.raw.strategy.window_units == "hours" else None
    raw_risk_vol = cfg.raw.risk.vol_window if cfg.raw.risk.window_units == "hours" else None

    print(
        f"tf={cfg.data.timeframe} bar_hours={cfg.bar_hours:.4f} "
        f"annualization_factor={cfg.annualization_factor:.2f} "
        f"config_hash={cfg.compute_hash()}"
    )
    print(
        f"strategy_windows: fast_hours={raw_fast} slow_hours={raw_slow} "
        f"fast_bars={cfg.strategy.fast} slow_bars={cfg.strategy.slow} "
        f"vol_window_bars={cfg.strategy.vol_window}"
    )
    print(
        f"risk_vol_window: hours={raw_risk_vol} bars={cfg.risk.vol_window}"
    )

    run_id = make_run_id(cfg.run_name)
    run_dir = Path("artifacts") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_portal = DataPortal(cfg.data, strict_validation=cfg.engine.strict_validation)
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
    provenance = data_portal.last_provenance
    if provenance:
        print("data_provenance:")
        print(
            f"  requested_timeframe={provenance.get('requested_timeframe')} "
            f"native_timeframe={provenance.get('native_timeframe')} "
            f"resample_rule={provenance.get('resampling_rule')} "
            f"bucket_alignment={provenance.get('bucket_alignment')}"
        )
        print(
            f"  drop_incomplete_bars={cfg.data.drop_incomplete_bars} "
            f"min_bucket_coverage_frac={cfg.data.min_bucket_coverage_frac}"
        )
        print(
            f"  first_bucket_coverage={provenance.get('first_bucket_coverage')} "
            f"last_bucket_coverage={provenance.get('last_bucket_coverage')} "
            f"dropped_first_bucket={provenance.get('dropped_first_bucket')} "
            f"dropped_last_bucket={provenance.get('dropped_last_bucket')}"
        )
        print(
            f"  resample_time_range_before_drop={provenance.get('resample_time_range_before_drop')} "
            f"resample_time_range_after_drop={provenance.get('resample_time_range_after_drop')}"
        )
    manifest = write_manifest(
        run_dir,
        cfg,
        bars_start=bars[0, "ts"],
        bars_end=bars.item(bars.height - 1, "ts"),
        data_provenance=data_portal.last_provenance,
    )

    frames["equity"].write_parquet(run_dir / "equity.parquet")
    frames["positions"].write_parquet(run_dir / "positions.parquet")
    frames["trades"].write_parquet(run_dir / "trades.parquet")
    if args.include_provenance_in_summary:
        summary["data_provenance"] = provenance
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Run ID: {run_id}")
    print(f"Manifest written: {manifest['config_hash']}")
    print(
        f"Total return: {summary['total_return_decimal']:.4f} (decimal), "
        f"{summary['total_return_pct']:.2f}% (pct), "
        f"{summary['total_return_multiple']:.4f}x (multiple)"
    )
    print(f"Sharpe: {summary['sharpe']:.4f}")
    print(f"Max drawdown: {summary['max_drawdown']:.4f}")


if __name__ == "__main__":
    main()

