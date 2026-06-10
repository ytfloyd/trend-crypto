#!/usr/bin/env python3
"""Run the K2 CL research diagnostics pipeline against the local volbook lake."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from k2_systematic_macro.configs.cl import CLResearchConfig
from k2_systematic_macro.models.expansion import ExpansionModelConfig
from k2_systematic_macro.research.pipeline import (
    CLResearchPipelineConfig,
    run_cl_research_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="auto", choices=["auto", "duckdb", "parquet"])
    parser.add_argument(
        "--contract-source",
        default=None,
        choices=["dated_front", "institutional_continuous", "continuous", "expiry"],
    )
    parser.add_argument("--expiry", default=None)
    parser.add_argument("--continuous-adjustment", default=None, choices=["raw", "additive", "ratio"])
    parser.add_argument("--continuous-business-calendar", default=None, choices=["weekend", "nymex_observed"])
    parser.add_argument(
        "--continuous-roll-policy",
        default=None,
        choices=[
            "last_trade_minus_n_business_days",
            "volume_crossover",
            "volume_crossover_with_calendar_guard",
        ],
    )
    parser.add_argument("--continuous-roll-window-business-days", type=int, default=None)
    parser.add_argument("--continuous-forced-roll-business-days-before-last-trade", type=int, default=None)
    parser.add_argument("--continuous-volume-crossover-sessions", type=int, default=None)
    parser.add_argument("--lake-path", type=Path, default=None)
    parser.add_argument("--parquet-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--start-ts", default=None)
    parser.add_argument("--end-ts", default=None)
    parser.add_argument("--no-optional-boosters", action="store_true")
    parser.add_argument("--min-train-size", type=int, default=120)
    parser.add_argument("--test-size", type=int, default=40)
    args = parser.parse_args()

    base_config = CLResearchConfig(source=args.source)
    cl_config = CLResearchConfig(
        symbol=base_config.symbol,
        primary_timeframe=base_config.primary_timeframe,
        secondary_timeframes=base_config.secondary_timeframes,
        source=args.source,
        contract_source=args.contract_source or base_config.contract_source,
        expiry=args.expiry or base_config.expiry,
        roll_days_before_expiry=base_config.roll_days_before_expiry,
        continuous_adjustment=args.continuous_adjustment or base_config.continuous_adjustment,
        continuous_roll_policy=args.continuous_roll_policy or base_config.continuous_roll_policy,
        continuous_business_calendar=args.continuous_business_calendar or base_config.continuous_business_calendar,
        continuous_roll_window_business_days=(
            args.continuous_roll_window_business_days
            if args.continuous_roll_window_business_days is not None
            else base_config.continuous_roll_window_business_days
        ),
        continuous_forced_roll_business_days_before_last_trade=(
            args.continuous_forced_roll_business_days_before_last_trade
            if args.continuous_forced_roll_business_days_before_last_trade is not None
            else base_config.continuous_forced_roll_business_days_before_last_trade
        ),
        continuous_volume_crossover_sessions=(
            args.continuous_volume_crossover_sessions
            if args.continuous_volume_crossover_sessions is not None
            else base_config.continuous_volume_crossover_sessions
        ),
        front_month_guard=base_config.front_month_guard,
        front_month_guard_max_curve_position=base_config.front_month_guard_max_curve_position,
        front_month_guard_on_missing=base_config.front_month_guard_on_missing,
        start_ts=args.start_ts if args.start_ts is not None else base_config.start_ts,
        end_ts=args.end_ts if args.end_ts is not None else base_config.end_ts,
        parquet_root=args.parquet_root or base_config.parquet_root,
        lake_path=args.lake_path or base_config.lake_path,
        output_root=base_config.output_root,
        session_tz=base_config.session_tz,
        session_start_hour=base_config.session_start_hour,
        feature_timeframes=base_config.feature_timeframes,
    )
    pipeline_config = CLResearchPipelineConfig(
        cl_config=cl_config,
        output_root=args.output_root or CLResearchPipelineConfig().output_root,
        expansion_config=ExpansionModelConfig(
            min_train_size=args.min_train_size,
            test_size=args.test_size,
        ),
        include_optional_boosters=not args.no_optional_boosters,
    )
    result = run_cl_research_pipeline(pipeline_config)
    print(f"Wrote K2 CL research artifacts to {result.output_dir}")
    for name, path in sorted(result.artifact_paths.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
