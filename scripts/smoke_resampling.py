#!/usr/bin/env python
"""Frequency smoke test ladder for resampling validation."""
from __future__ import annotations

import argparse
from datetime import datetime
from typing import Optional

from backtest.engine import BacktestEngine
from common.config import (
    DataConfig,
    EngineConfig,
    ExecutionConfig,
    RiskConfigRaw,
    RunConfigRaw,
    StrategyConfigRaw,
    compile_config,
)
from data.portal import DataPortal
from risk.risk_manager import RiskManager
from strategy.buy_and_hold import BuyAndHoldStrategy


def parse_datetime(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test resampling across timeframes")
    p.add_argument("--db-path", required=True, help="DuckDB path")
    p.add_argument("--table", default=None, help="Table name (optional; auto-resolve)")
    p.add_argument("--symbol", required=True, help="Symbol")
    p.add_argument("--start", required=True, help="Start ISO8601")
    p.add_argument("--end", required=True, help="End ISO8601")
    p.add_argument("--native-timeframe", default=None, help="Native timeframe (optional; inferred)")
    p.add_argument("--timeframes", default="1m,1h,1d", help="CSV of requested timeframes")
    p.add_argument("--strict-validation", action="store_true", help="Enable strict validation")
    p.add_argument("--drop-incomplete-bars", action="store_true", help="Drop incomplete buckets")
    p.add_argument("--min-coverage-frac", type=float, default=0.8, help="Min bucket coverage fraction")
    p.add_argument(
        "--risk-vol-window-hours",
        type=int,
        default=720,
        help="Risk vol window in hours intent (prevents strict-validation failures at coarse timeframes)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_datetime(args.start)
    end = parse_datetime(args.end)
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    for tf in timeframes:
        raw_cfg = RunConfigRaw(
            run_name=f"smoke_resampling_{tf}",
            data=DataConfig(
                db_path=args.db_path,
                table=args.table,
                symbol=args.symbol,
                start=start,
                end=end,
                timeframe=tf,
                native_timeframe=args.native_timeframe,
                drop_incomplete_bars=args.drop_incomplete_bars,
                min_bucket_coverage_frac=args.min_coverage_frac,
            ),
            engine=EngineConfig(strict_validation=args.strict_validation, lookback=10, initial_cash=100000.0),
            strategy=StrategyConfigRaw(mode="buy_and_hold", weight_on=1.0, window_units="hours"),
            risk=RiskConfigRaw(
                vol_window=args.risk_vol_window_hours,
                target_vol_annual=None,
                max_weight=1.0,
                window_units="hours",
            ),
            execution=ExecutionConfig(fee_bps=0.0, slippage_bps=0.0, execution_lag_bars=1),
        )
        cfg = compile_config(raw_cfg)
        portal = DataPortal(cfg.data, strict_validation=cfg.engine.strict_validation)
        engine = BacktestEngine(
            cfg,
            BuyAndHoldStrategy(cfg.strategy),
            RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor),
            portal,
        )
        portfolio, summary = engine.run()
        bars = portal.load_bars()
        prov = portal.last_provenance
        print(
            f"tf={tf} hash={cfg.compute_hash()} bars={bars.height} "
            f"native={prov.get('native_timeframe')} requested={prov.get('requested_timeframe')} "
            f"drop_first={prov.get('dropped_first_bucket')} drop_last={prov.get('dropped_last_bucket')} "
            f"first_cov={prov.get('first_bucket_coverage')} last_cov={prov.get('last_bucket_coverage')} "
            f"return={summary.get('total_return_decimal'):.6f} "
            f"pct={summary.get('total_return_pct'):.2f}% "
            f"mult={summary.get('total_return_multiple'):.4f}x"
        )


if __name__ == "__main__":
    main()
