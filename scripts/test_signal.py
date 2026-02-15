#!/usr/bin/env python
"""CLI signal tester — define a signal expression, evaluate, and backtest.

Usage examples:

  # Quick test with synthetic data (no DuckDB needed):
  python scripts/test_signal.py --synthetic

  # Test MA crossover with custom params on DuckDB data:
  python scripts/test_signal.py --db ../data/market.duckdb --fast 8 --slow 40

  # Test a custom signal expression:
  python scripts/test_signal.py --synthetic --signal 'pl.col("close").rolling_mean(10) - pl.col("close").rolling_mean(50)'

  # Run with walk-forward validation:
  python scripts/test_signal.py --synthetic --walk-forward

  # Parameter sweep:
  python scripts/test_signal.py --synthetic --sweep --fast-range 3,5,8,12 --slow-range 20,40,60
"""
from __future__ import annotations

import argparse
import math
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "src")
sys.path.insert(0, _SRC_DIR)

# In a git worktree, some modules (e.g. data.portal) live only in the main
# repo's src/. Discover the main repo via git and add its src/ as fallback.
import subprocess as _sp
try:
    _git_common = _sp.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=_SCRIPT_DIR, text=True, stderr=_sp.DEVNULL,
    ).strip()
    _main_repo_src = os.path.join(os.path.dirname(_git_common), "src")
    if os.path.isdir(_main_repo_src) and _main_repo_src != os.path.normpath(_SRC_DIR):
        sys.path.append(_main_repo_src)
except Exception:
    pass

import polars as pl

from research.alpha_pipeline import evaluate_alpha


def _make_synthetic_bars(n: int = 2000) -> pl.DataFrame:
    """Generate synthetic OHLCV bars for testing without DuckDB."""
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    price = 10000.0
    rows: list[dict[str, Any]] = []
    for i in range(n):
        ts = start + timedelta(days=i)
        trend = 0.0003 * i
        cycle = 500 * math.sin(2 * math.pi * i / 365)
        noise = 100 * math.sin(i * 0.7) * math.cos(i * 0.3)
        c = price + trend * price + cycle + noise
        o = c - 50 * math.sin(i * 0.5)
        rows.append({
            "ts": ts, "symbol": "BTC-USD", "open": o,
            "high": max(o, c) + abs(noise) * 0.3,
            "low": min(o, c) - abs(noise) * 0.3,
            "close": c, "volume": 1e6 + i * 100,
        })
    return pl.DataFrame(rows)


def _load_duckdb_bars(
    db_path: str,
    symbol: str,
    table: str,
    start: str | None,
    end: str | None,
    timeframe: str,
) -> pl.DataFrame:
    """Load bars from DuckDB via DataPortal."""
    from common.config import DataConfig
    from data.portal import DataPortal

    start_dt = (
        datetime.fromisoformat(start) if start
        else datetime(2015, 7, 20, tzinfo=timezone.utc)
    )
    end_dt = (
        datetime.fromisoformat(end) if end
        else datetime(2025, 12, 31, tzinfo=timezone.utc)
    )
    data_cfg = DataConfig(
        db_path=db_path, table=table, symbol=symbol,
        start=start_dt, end=end_dt, timeframe=timeframe,
    )
    portal = DataPortal(data_cfg, strict_validation=False)
    return portal.load_bars()


def _build_signal(
    bars: pl.DataFrame,
    signal_expr: str | None,
    fast: int,
    slow: int,
) -> pl.DataFrame:
    """Build a signal DataFrame from bars.

    If signal_expr is provided, evaluates it as a Polars expression.
    Otherwise uses MA crossover (fast - slow).
    """
    if signal_expr:
        # User-provided expression — evaluate as Polars
        signal_raw = bars.with_columns(
            eval(signal_expr).alias("signal_raw"),  # noqa: S307
        )
    else:
        signal_raw = bars.with_columns([
            pl.col("close").rolling_mean(fast).alias("ma_fast"),
            pl.col("close").rolling_mean(slow).alias("ma_slow"),
        ]).with_columns(
            (pl.col("ma_fast") - pl.col("ma_slow")).alias("signal_raw"),
        )

    signal_df = signal_raw.with_columns(
        pl.when(pl.col("signal_raw") > 0).then(1.0).otherwise(0.0).alias("signal"),
        pl.col("close").pct_change().shift(-1).alias("fwd_ret"),
    ).drop_nulls(subset=["signal", "fwd_ret"])

    return signal_df


def _print_header(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _print_metric(label: str, value: Any, fmt: str = ".4f") -> None:
    if isinstance(value, float):
        print(f"  {label:<24s} {value:{fmt}}")
    else:
        print(f"  {label:<24s} {value}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a signal: IC, hit rate, Sharpe, plus optional backtest."""
    from research.api import quick_backtest

    bars = _load_bars(args)

    _print_header("Signal Evaluation")
    print(f"  Bars:   {bars.height}")
    print(f"  Range:  {bars['ts'].min()} to {bars['ts'].max()}")

    # Build signal
    signal_df = _build_signal(bars, args.signal, args.fast, args.slow)
    long_frac = signal_df.filter(pl.col("signal") > 0).height / max(1, signal_df.height)
    print(f"  Signal observations: {signal_df.height}")
    print(f"  Long fraction:      {long_frac:.1%}")

    # Alpha evaluation
    _print_header("Alpha Quality")
    name = args.name or f"ma_{args.fast}_{args.slow}"
    result = evaluate_alpha(
        name=name,
        signal=signal_df["signal"],
        forward_returns=signal_df["fwd_ret"],
    )
    _print_metric("IC (mean)", result.ic_mean)
    _print_metric("IC IR", result.ic_ir)
    _print_metric("Hit Rate", result.hit_rate, ".1%")
    _print_metric("Turnover", result.turnover)
    _print_metric("Sharpe (signal)", result.sharpe, ".2f")

    # Quick backtest
    _print_header("Backtest")
    strategy_mode = "buy_and_hold" if args.signal else "ma_crossover_long_only"
    equity_df, summary = quick_backtest(
        bars,
        strategy_mode=strategy_mode,
        fast=args.fast, slow=args.slow,
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
    )
    _print_metric("Total Return", summary.get("total_return", 0.0))
    _print_metric("Sharpe", summary.get("sharpe", 0.0))
    _print_metric("Max Drawdown", summary.get("max_drawdown", 0.0))
    _print_metric("Trade Count", summary.get("trade_count", "N/A"), "s")

    # Optionally save tearsheet
    if args.tearsheet:
        from monitoring.dashboard import generate_html_tearsheet

        path = generate_html_tearsheet(
            equity_df, dict(summary),
            output_path=args.tearsheet,
            title=f"{name} Backtest",
        )
        print(f"\n  Tearsheet: {path}")

    print()


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a parameter sweep over fast/slow ranges."""
    from research.api import quick_sweep

    bars = _load_bars(args)

    fast_vals = [int(x) for x in args.fast_range.split(",")]
    slow_vals = [int(x) for x in args.slow_range.split(",")]

    _print_header(f"Parameter Sweep ({len(fast_vals) * len(slow_vals)} combinations)")
    print(f"  Fast: {fast_vals}")
    print(f"  Slow: {slow_vals}")
    print(f"  Bars: {bars.height}")

    results = quick_sweep(
        bars,
        param_grid={"fast": fast_vals, "slow": slow_vals},
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
    )

    sorted_results = results.sort("sharpe", descending=True)
    _print_header("Top Results (by Sharpe)")

    # Print table header
    print(f"  {'Fast':>6s}  {'Slow':>6s}  {'Sharpe':>8s}  {'Return':>10s}  {'MaxDD':>8s}  {'Trades':>7s}")
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 7}")

    for row in sorted_results.head(min(15, sorted_results.height)).iter_rows(named=True):
        sharpe = row.get("sharpe")
        ret = row.get("total_return")
        dd = row.get("max_drawdown")
        trades = row.get("trade_count")
        print(
            f"  {row.get('fast', ''):>6}  {row.get('slow', ''):>6}  "
            f"{sharpe if sharpe is not None else 'N/A':>8.4f}  "
            f"{ret if ret is not None else 'N/A':>10.4f}  "
            f"{dd if dd is not None else 'N/A':>8.4f}  "
            f"{trades if trades is not None else 'N/A':>7}"
        )
    print()


def cmd_walk_forward(args: argparse.Namespace) -> None:
    """Run walk-forward optimization with deflated Sharpe correction."""
    from research.api import quick_backtest
    from research.optimizer import ParameterOptimizer

    bars = _load_bars(args)

    fast_vals = [int(x) for x in args.fast_range.split(",")]
    slow_vals = [int(x) for x in args.slow_range.split(",")]

    _print_header("Walk-Forward Optimization")
    print(f"  Fast: {fast_vals}")
    print(f"  Slow: {slow_vals}")
    print(f"  Splits: {args.n_splits}")
    print(f"  Bars: {bars.height}")

    def evaluate_fn(bars_slice: pl.DataFrame, params: dict[str, Any]) -> float:
        _, s = quick_backtest(
            bars_slice,
            fast=params["fast"], slow=params["slow"],
            fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        )
        sharpe = s.get("sharpe", 0.0)
        return float(sharpe) if sharpe is not None else 0.0

    optimizer = ParameterOptimizer(
        bars=bars,
        evaluate_fn=evaluate_fn,
        param_grid={"fast": fast_vals, "slow": slow_vals},
        n_splits=args.n_splits,
        train_frac=0.7,
        gap=20,
    )
    opt_result = optimizer.optimize()

    _print_header("Walk-Forward Results")
    _print_metric("Best params", str(opt_result.best_params), "s")
    _print_metric("Best train metric", opt_result.best_metric, ".4f")
    if opt_result.deflated_sharpe is not None:
        _print_metric("Deflated Sharpe", opt_result.deflated_sharpe, ".4f")
    _print_metric("Trials tested", opt_result.n_trials, "d")

    # Show all results table
    all_df = pl.DataFrame(opt_result.all_results)
    if all_df.height > 0 and "avg_test_metric" in all_df.columns:
        sorted_df = all_df.sort("avg_test_metric", descending=True)
        print(f"\n  {'Params':<30s}  {'Train':>8s}  {'Test':>8s}")
        print(f"  {'─' * 30}  {'─' * 8}  {'─' * 8}")
        for row in sorted_df.head(10).iter_rows(named=True):
            params_str = str({k: v for k, v in row.items() if k not in ("avg_train_metric", "avg_test_metric")})
            print(
                f"  {params_str:<30s}  "
                f"{row.get('avg_train_metric', 0.0):>8.4f}  "
                f"{row.get('avg_test_metric', 0.0):>8.4f}"
            )
    print()


def _load_bars(args: argparse.Namespace) -> pl.DataFrame:
    """Load bars based on CLI args."""
    if args.synthetic:
        return _make_synthetic_bars(args.n_bars)

    db_path = args.db or os.environ.get("TREND_CRYPTO_DB")
    if not db_path:
        print("Error: provide --db or set TREND_CRYPTO_DB, or use --synthetic")
        sys.exit(1)

    return _load_duckdb_bars(
        db_path=db_path,
        symbol=args.symbol,
        table=args.table,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI signal tester — evaluate signals and run backtests from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source
    data = parser.add_argument_group("data source")
    data.add_argument("--synthetic", action="store_true", help="Use synthetic data (no DuckDB needed)")
    data.add_argument("--n-bars", type=int, default=2000, help="Number of synthetic bars (default: 2000)")
    data.add_argument("--db", type=str, help="DuckDB path (or set TREND_CRYPTO_DB)")
    data.add_argument("--symbol", type=str, default="BTC-USD", help="Symbol (default: BTC-USD)")
    data.add_argument("--table", type=str, default="bars_1d_clean", help="DuckDB table (default: bars_1d_clean)")
    data.add_argument("--timeframe", type=str, default="1d", help="Timeframe (default: 1d)")
    data.add_argument("--start", type=str, default=None, help="Start date (ISO format)")
    data.add_argument("--end", type=str, default=None, help="End date (ISO format)")

    # Signal definition
    sig = parser.add_argument_group("signal")
    sig.add_argument("--signal", type=str, default=None,
                     help="Custom Polars expression for signal_raw (e.g. 'pl.col(\"close\").rolling_mean(10) - pl.col(\"close\").rolling_mean(50)')")
    sig.add_argument("--fast", type=int, default=5, help="Fast MA window (default: 5)")
    sig.add_argument("--slow", type=int, default=40, help="Slow MA window (default: 40)")
    sig.add_argument("--name", type=str, default=None, help="Signal name for logging")

    # Execution
    ex = parser.add_argument_group("execution")
    ex.add_argument("--fee-bps", type=float, default=10.0, help="Fee in bps (default: 10)")
    ex.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage in bps (default: 5)")

    # Mode
    mode = parser.add_argument_group("mode")
    mode.add_argument("--sweep", action="store_true", help="Run parameter sweep instead of single eval")
    mode.add_argument("--walk-forward", action="store_true", help="Run walk-forward optimization")
    mode.add_argument("--fast-range", type=str, default="3,5,8,12,20",
                      help="Fast values for sweep (comma-separated, default: 3,5,8,12,20)")
    mode.add_argument("--slow-range", type=str, default="20,40,60,100",
                      help="Slow values for sweep (comma-separated, default: 20,40,60,100)")
    mode.add_argument("--n-splits", type=int, default=5, help="Walk-forward splits (default: 5)")

    # Output
    out = parser.add_argument_group("output")
    out.add_argument("--tearsheet", type=str, default=None,
                     help="Path to save HTML tearsheet (e.g. artifacts/tearsheets/my_signal.html)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.walk_forward:
        cmd_walk_forward(args)
    elif args.sweep:
        cmd_sweep(args)
    else:
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
