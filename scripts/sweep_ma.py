#!/usr/bin/env python
"""
MA parameter sweep runner (20-200h fast, 40-400h slow).

Evaluates each parameter combination over:
- Full period
- Subperiods: 2021, 2022, 2023

Results are written to artifacts/sweeps/ma_sweep_{timestamp}/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtest.engine import BacktestEngine
from common.config import DataConfig, EngineConfig, ExecutionConfig, RiskConfigRaw, RunConfigRaw, RunConfigResolved, StrategyConfigRaw, compile_config
from data.portal import DataPortal
from risk.risk_manager import RiskManager
from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy
from utils.duckdb_inspect import (
    describe_table,
    infer_start_end,
    resolve_bars_table,
    resolve_ts_column,
    validate_funding_column,
    validate_required_columns,
)


def parse_datetime(s: str) -> datetime:
    """
    Parse ISO datetime string to timezone-aware UTC datetime.
    
    Accepts:
    - "2021-01-01" → 2021-01-01T00:00:00+00:00
    - "2021-01-01T00:00:00Z" → 2021-01-01T00:00:00+00:00
    - "2021-01-01T00:00:00+00:00" → 2021-01-01T00:00:00+00:00
    """
    # Try full ISO format first
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]:
        try:
            dt = datetime.strptime(s.replace("Z", "+00:00") if s.endswith("Z") else s, fmt)
            # Ensure timezone-aware UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    
    raise ValueError(f"Cannot parse datetime: {s}. Expected ISO format like '2021-01-01' or '2021-01-01T00:00:00Z'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MA parameter sweep (20-200h fast, 40-400h slow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  TREND_CRYPTO_DB    DuckDB path (fallback if --db not provided)

Examples:
  # Using --db flag
  python scripts/sweep_ma.py --db ../data/market.duckdb --symbol BTC-USD
  
  # Using environment variable
  export TREND_CRYPTO_DB=../data/market.duckdb
  python scripts/sweep_ma.py --symbol BTC-USD
  
  # With explicit date range
  python scripts/sweep_ma.py --symbol BTC-USD --start 2020-01-01 --end 2024-01-01
"""
    )
    p.add_argument("--db", type=str, default=None, help="DuckDB path (or set TREND_CRYPTO_DB env var)")
    p.add_argument("--table", type=str, default=None, help="Table/view name (auto-selected if omitted)")
    p.add_argument("--symbol", type=str, default="BTC-USD", help="Symbol to backtest")
    p.add_argument("--timeframe", type=str, default="1h", help="Timeframe (1h, 4h, 1d)")
    p.add_argument("--start", type=str, default=None, help="Start date (ISO format, e.g., 2021-01-01 or 2021-01-01T00:00:00Z); auto-discovered if omitted")
    p.add_argument("--end", type=str, default=None, help="End date (ISO format); auto-discovered if omitted")
    p.add_argument("--fast-start", type=int, default=20, help="Fast MA start (hours)")
    p.add_argument("--fast-end", type=int, default=200, help="Fast MA end (hours)")
    p.add_argument("--fast-step", type=int, default=10, help="Fast MA step")
    p.add_argument("--slow-start", type=int, default=40, help="Slow MA start (hours)")
    p.add_argument("--slow-end", type=int, default=400, help="Slow MA end (hours)")
    p.add_argument("--slow-step", type=int, default=20, help="Slow MA step")
    p.add_argument("--fee-bps", type=float, default=10.0, help="Fee (bps)")
    p.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage (bps)")
    p.add_argument("--funding-mode", type=str, default="none", choices=["none", "column"], help="Funding mode")
    p.add_argument("--funding-col", type=str, default="funding_rate", help="Funding rate column name")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: artifacts/sweeps/ma_sweep_<timestamp>)")
    
    args = p.parse_args()
    
    # Resolve db path: CLI flag > env var > error
    if args.db is None:
        args.db = os.getenv("TREND_CRYPTO_DB")
    
    if args.db is None:
        p.error(
            "DuckDB path required. Provide via --db flag or TREND_CRYPTO_DB env var.\n"
            "Examples:\n"
            "  python scripts/sweep_ma.py --db ../data/market.duckdb --symbol BTC-USD\n"
            "  export TREND_CRYPTO_DB=../data/market.duckdb && python scripts/sweep_ma.py --symbol BTC-USD"
        )
    
    return args


def get_git_sha() -> str | None:
    """Get current git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def run_single_backtest(
    cfg: RunConfigResolved,
    strategy: MACrossoverLongOnlyStrategy,
    risk_manager: RiskManager,
    data_portal: DataPortal,
) -> tuple[dict[str, Any], str]:
    """Run a single backtest and return (summary, skip_reason)."""
    try:
        bars = data_portal.load_bars()
        if bars.height < 2:
            return {}, "insufficient_bars"
        
        engine = BacktestEngine(cfg, strategy, risk_manager, data_portal)
        portfolio, summary = engine.run()
        return summary, ""
    except Exception as e:
        return {}, f"error_{type(e).__name__}"


def main() -> None:
    args = parse_args()
    
    print("=" * 70)
    print("RESOLVE & VALIDATE")
    print("=" * 70)
    
    # 1) Resolve table
    try:
        resolved_table = resolve_bars_table(args.db, args.table, args.timeframe)
        print(f"✓ Table: {resolved_table}" + (" (auto-selected)" if not args.table else ""))
    except ValueError as e:
        print(f"✗ Error resolving table: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 2) Describe and validate columns
    try:
        columns = describe_table(args.db, resolved_table)
        validate_required_columns(columns)
        ts_col = resolve_ts_column(columns)
        print(f"✓ Columns validated (ts column: {ts_col})")
    except ValueError as e:
        print(f"✗ Error validating columns: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 3) Check for timeframe column
    has_timeframe_col = "timeframe" in columns
    timeframe_filter = args.timeframe if has_timeframe_col else None
    if has_timeframe_col:
        print(f"✓ Timeframe column detected; will filter by timeframe={args.timeframe}")
    
    # 4) Validate funding if requested
    if args.funding_mode == "column":
        if not validate_funding_column(args.db, resolved_table, args.funding_col):
            print(
                f"✗ Error: funding_mode=column but column '{args.funding_col}' not found in {resolved_table}",
                file=sys.stderr,
            )
            print(f"Available columns: {sorted(columns)}", file=sys.stderr)
            print(f"Suggestion: Use --funding-mode none", file=sys.stderr)
            sys.exit(1)
        print(f"✓ Funding column '{args.funding_col}' validated")
    else:
        print(f"✓ Funding disabled (mode={args.funding_mode})")
    
    # 5) Resolve start/end dates: CLI args > auto-discovery
    if args.start and args.end:
        global_start = parse_datetime(args.start)
        global_end = parse_datetime(args.end)
        date_source = "CLI args"
    elif args.start or args.end:
        print("✗ Error: --start and --end must both be provided or both omitted", file=sys.stderr)
        sys.exit(1)
    else:
        try:
            global_start, global_end = infer_start_end(
                args.db,
                resolved_table,
                args.symbol,
                ts_col,
                "timeframe" if has_timeframe_col else None,
                timeframe_filter,
            )
            date_source = "auto-discovered"
        except ValueError as e:
            print(f"✗ Error discovering date range: {e}", file=sys.stderr)
            sys.exit(1)
    
    print(f"✓ Date range: {global_start.date()} to {global_end.date()} ({date_source})")
    
    # Print resolved config
    print()
    print("RESOLVED CONFIG:")
    print(f"  db: {args.db}")
    print(f"  table: {resolved_table}")
    print(f"  ts_col: {ts_col}")
    print(f"  symbol: {args.symbol}")
    print(f"  timeframe: {args.timeframe}")
    print(f"  start: {global_start.isoformat()}")
    print(f"  end: {global_end.isoformat()}")
    print(f"  funding: {args.funding_mode}")
    print("=" * 70)
    print()
    
    # Setup output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("artifacts") / "sweeps" / f"ma_sweep_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_csv = out_dir / "results.csv"
    
    # Write manifest
    git_sha = get_git_sha()
    manifest = {
        "sweep_type": "ma_parameter_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "config": vars(args),
        "resolved": {
            "table": resolved_table,
            "ts_col": ts_col,
            "start": global_start.isoformat(),
            "end": global_end.isoformat(),
            "date_source": date_source,
            "has_timeframe_col": has_timeframe_col,
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"MA Sweep: {args.symbol} @ {args.timeframe}")
    print(f"Date range: {global_start.date()} to {global_end.date()} ({date_source})")
    print(f"Fast: {args.fast_start}-{args.fast_end} (step {args.fast_step})")
    print(f"Slow: {args.slow_start}-{args.slow_end} (step {args.slow_step})")
    print(f"Output: {results_csv}")
    print()
    
    # Build parameter grid
    fast_params = list(range(args.fast_start, args.fast_end + 1, args.fast_step))
    slow_params = list(range(args.slow_start, args.slow_end + 1, args.slow_step))
    
    # Subperiods with datetime objects
    subperiods: list[tuple[str, datetime | None, datetime | None]] = [
        ("full", global_start, global_end),
        ("2021", parse_datetime("2021-01-01"), parse_datetime("2021-12-31T23:59:59Z")),
        ("2022", parse_datetime("2022-01-01"), parse_datetime("2022-12-31T23:59:59Z")),
        ("2023", parse_datetime("2023-01-01"), parse_datetime("2023-12-31T23:59:59Z")),
    ]
    
    # CSV writer
    fieldnames = [
        "run_idx", "fast_hours", "slow_hours", "fast_bars", "slow_bars",
        "fee_bps", "slippage_bps", "funding_mode",
        "subperiod_name", "start_ts", "end_ts", "bars",
        "final_equity", "total_return", "sharpe", "max_drawdown",
        "total_funding_cost", "avg_funding_cost_per_bar", "funding_cost_as_pct_of_gross",
        "used_close_to_close_fallback", "return_mode", "trade_log_mode",
        "entry_exit_events", "avg_cash_weight",
        "skipped_reason",
    ]
    
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        run_idx = 0
        total_runs = len(fast_params) * len(slow_params) * len(subperiods)
        
        for fast_h in fast_params:
            for slow_h in slow_params:
                if slow_h <= fast_h:
                    continue  # Enforce slow > fast
                
                # Use raw hours; compiler will convert to bars
                fast_bars = None
                slow_bars = None
                
                for subperiod_name, subperiod_start, subperiod_end in subperiods:
                    run_idx += 1
                    
                    # Clamp subperiod to available data range
                    if subperiod_start and subperiod_end:
                        # Check if subperiod overlaps with available data
                        if subperiod_end < global_start or subperiod_start > global_end:
                            # No overlap - skip this subperiod
                            writer.writerow({
                                "run_idx": run_idx,
                                "fast_hours": fast_h,
                                "slow_hours": slow_h,
                                "fast_bars": fast_bars,
                                "slow_bars": slow_bars,
                                "fee_bps": args.fee_bps,
                                "slippage_bps": args.slippage_bps,
                                "funding_mode": args.funding_mode,
                                "subperiod_name": subperiod_name,
                                "skipped_reason": "no_data_overlap",
                            })
                            continue
                        
                        # Clamp to available range
                        clamped_start = max(subperiod_start, global_start)
                        clamped_end = min(subperiod_end, global_end)
                    else:
                        # "full" period uses global range
                        clamped_start = subperiod_start
                        clamped_end = subperiod_end
                    
                    # Build config with datetime objects
                    raw_cfg = RunConfigRaw(
                        run_name=f"ma_sweep_{fast_h}_{slow_h}_{subperiod_name}",
                        data=DataConfig(
                            db_path=args.db,
                            table=resolved_table,
                            symbol=args.symbol,
                            timeframe=args.timeframe,
                            start=clamped_start,
                            end=clamped_end,
                        ),
                        engine=EngineConfig(
                            strict_validation=False,
                            lookback=None,
                            initial_cash=100000.0,
                        ),
                        strategy=StrategyConfigRaw(
                            mode="ma_crossover_long_only",
                            fast=fast_h,
                            slow=slow_h,
                            vol_window=20,
                            k=1.0,
                            min_band=0.0,
                            window_units="hours",
                            weight_on=1.0,
                            target_vol_annual=None,
                            max_weight=1.0,
                            enable_adx_filter=False,
                        ),
                        risk=RiskConfigRaw(
                            vol_window=20,
                            target_vol_annual=None,
                            max_weight=1.0,
                            window_units="hours",
                        ),
                        execution=ExecutionConfig(
                            execution_lag_bars=1,
                            fee_bps=args.fee_bps,
                            slippage_bps=args.slippage_bps,
                            cash_yield_annual=0.0,
                        ),
                    )
                    cfg = compile_config(raw_cfg)

                    # Resolved bars after compilation
                    fast_bars = cfg.strategy.fast
                    slow_bars = cfg.strategy.slow
                    
                    data_portal = DataPortal(cfg.data, strict_validation=cfg.engine.strict_validation)
                    strategy = MACrossoverLongOnlyStrategy(
                        fast=fast_bars,
                        slow=slow_bars,
                        weight_on=1.0,
                        target_vol_annual=None,
                        vol_lookback=20,
                        max_weight=1.0,
                        enable_adx_filter=False,
                    )
                    risk_manager = RiskManager(cfg.risk, cfg.annualization_factor)
                    
                    summary, skip_reason = run_single_backtest(cfg, strategy, risk_manager, data_portal)
                    
                    if skip_reason:
                        # Write skipped row
                        writer.writerow({
                            "run_idx": run_idx,
                            "fast_hours": fast_h,
                            "slow_hours": slow_h,
                            "fast_bars": fast_bars,
                            "slow_bars": slow_bars,
                            "fee_bps": args.fee_bps,
                            "slippage_bps": args.slippage_bps,
                            "funding_mode": args.funding_mode,
                            "subperiod_name": subperiod_name,
                            "skipped_reason": skip_reason,
                        })
                    else:
                        # Write successful row
                        writer.writerow({
                            "run_idx": run_idx,
                            "fast_hours": fast_h,
                            "slow_hours": slow_h,
                            "fast_bars": fast_bars,
                            "slow_bars": slow_bars,
                            "fee_bps": args.fee_bps,
                            "slippage_bps": args.slippage_bps,
                            "funding_mode": args.funding_mode,
                            "subperiod_name": subperiod_name,
                            "start_ts": clamped_start.isoformat() if clamped_start else "",
                            "end_ts": clamped_end.isoformat() if clamped_end else "",
                            "bars": summary.get("bars", 0),
                            "final_equity": summary.get("final_equity", 0.0),
                            "total_return": summary.get("total_return", 0.0),
                            "sharpe": summary.get("sharpe", 0.0),
                            "max_drawdown": summary.get("max_drawdown", 0.0),
                            "total_funding_cost": summary.get("total_funding_cost", 0.0),
                            "avg_funding_cost_per_bar": summary.get("avg_funding_cost_per_bar", 0.0),
                            "funding_cost_as_pct_of_gross": summary.get("funding_cost_as_pct_of_gross"),
                            "used_close_to_close_fallback": summary.get("used_close_to_close_fallback", False),
                            "return_mode": summary.get("return_mode", ""),
                            "trade_log_mode": summary.get("trade_log_mode", ""),
                            "entry_exit_events": summary.get("entry_exit_events", 0),
                            "avg_cash_weight": summary.get("avg_cash_weight", 0.0),
                            "skipped_reason": "",
                        })
                    
                    # Progress
                    if run_idx % 50 == 0 or run_idx == total_runs:
                        print(f"  [{run_idx}/{total_runs}] fast={fast_h}h slow={slow_h}h {subperiod_name}")
    
    print()
    print(f"✓ Sweep complete: {results_csv}")
    print(f"  Total runs: {run_idx}")
    print(f"  Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
