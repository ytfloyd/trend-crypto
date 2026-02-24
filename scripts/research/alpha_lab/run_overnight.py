#!/usr/bin/env python3
"""
Alpha Lab — Overnight Signal Discovery Agent

Systematically tests every signal in the parameterized signal space,
backtests each with transaction costs, computes metrics + IC + regime
analysis, and catalogues all findings.

Designed to run unattended overnight. Features:
  - Resumable: skips previously tested signals on restart
  - Incremental: writes each result to disk immediately
  - Fault-tolerant: catches per-signal errors, continues to next
  - Progress: prints status every N signals and at completion
  - Summary: generates ranked markdown report at end

Usage:
    # Full run with default DuckDB (market.duckdb)
    python -m scripts.research.alpha_lab.run_overnight

    # Use pre-materialized daily table (faster)
    python -m scripts.research.alpha_lab.run_overnight \
        --db ../data/coinbase_daily_121025.duckdb \
        --table bars_1d_usd_universe_clean

    # Quick test: first 10 signals only
    python -m scripts.research.alpha_lab.run_overnight --limit 10

    # Custom date range and output directory
    python -m scripts.research.alpha_lab.run_overnight \
        --start 2018-01-01 --end 2025-12-15 \
        --test-start 2024-01-01 \
        --output artifacts/research/alpha_lab_run2
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure research common is importable
_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import load_daily_bars, filter_universe, ANN_FACTOR

from .signals import build_signal_space, SignalSpec
from .harness import run_signal_test
from .catalogue import Catalogue
from .onchain_data import fetch_all_onchain, compute_derived_onchain


def _load_from_table(db_path: str, table: str, start: str, end: str) -> pd.DataFrame:
    """Load data directly from a named DuckDB table (fast path)."""
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM {table}
            WHERE ts >= ? AND ts <= ?
              AND open > 0 AND close > 0
            ORDER BY ts, symbol
            """,
            [start, end],
        ).fetch_df()
    finally:
        con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    return df


def _prepare_data(
    db_path: str | None,
    table: str | None,
    start: str,
    end: str,
    min_adv_usd: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare wide-format data matrices."""
    print(f"[alpha-lab] Loading data ({start} to {end}) ...")

    if table and db_path:
        panel = _load_from_table(db_path, table, start, end)
        panel = filter_universe(panel, min_adv_usd=min_adv_usd, min_history_days=90)
    elif db_path:
        panel = load_daily_bars(db_path=db_path, start=start, end=end)
        panel = filter_universe(panel, min_adv_usd=min_adv_usd, min_history_days=90)
    else:
        panel = load_daily_bars(start=start, end=end)
        panel = filter_universe(panel, min_adv_usd=min_adv_usd, min_history_days=90)

    panel = panel.sort_values(["ts", "symbol"])

    close_wide = panel.pivot(index="ts", columns="symbol", values="close")
    volume_wide = panel.pivot(index="ts", columns="symbol", values="volume")
    returns_wide = close_wide.pct_change(fill_method=None)
    universe_wide = panel.pivot(index="ts", columns="symbol", values="in_universe").fillna(False).infer_objects(copy=False).astype(bool)

    n_assets = universe_wide.sum(axis=1).median()
    n_days = len(close_wide)
    print(f"[alpha-lab] Data ready: {n_days} days, ~{n_assets:.0f} assets in universe")

    return close_wide, volume_wide, returns_wide, universe_wide


def _log(log_path: Path, msg: str) -> None:
    """Append a timestamped message to the run log."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(log_path, "a") as f:
        f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Lab — Overnight Signal Discovery")
    parser.add_argument("--db", type=str, default=None, help="Path to DuckDB file")
    parser.add_argument("--table", type=str, default=None, help="DuckDB table name (fast path)")
    parser.add_argument("--start", type=str, default="2017-01-01", help="Data start date")
    parser.add_argument("--end", type=str, default="2025-12-15", help="Data end date")
    parser.add_argument("--test-start", type=str, default=None,
                        help="OOS test start date (if set, metrics computed only on test period)")
    parser.add_argument("--min-adv", type=float, default=500_000,
                        help="Minimum 20d ADV in USD for universe filter")
    parser.add_argument("--cost-bps", type=float, default=20.0, help="Transaction cost in bps")
    parser.add_argument("--output", type=str, default="artifacts/research/alpha_lab",
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of signals to test (for quick runs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    print("=" * 70)
    print("  ALPHA LAB — OVERNIGHT SIGNAL DISCOVERY AGENT")
    print("=" * 70)

    close_wide, volume_wide, returns_wide, universe_wide = _prepare_data(
        db_path=args.db,
        table=args.table,
        start=args.start,
        end=args.end,
        min_adv_usd=args.min_adv,
    )

    # Load on-chain data (BTC blockchain metrics)
    try:
        onchain_raw = fetch_all_onchain(start=args.start, end=args.end, use_cache=True)
        onchain_derived = compute_derived_onchain(onchain_raw)
        onchain_derived.index = pd.to_datetime(onchain_derived.index)
        # Align to trading calendar
        onchain_derived = onchain_derived.reindex(close_wide.index, method="ffill")
        _onchain_data = onchain_derived
        print(f"[alpha-lab] On-chain data: {len(onchain_derived.columns)} features loaded")
    except Exception as e:
        print(f"[alpha-lab] WARNING: On-chain data fetch failed: {e}")
        _onchain_data = None

    signal_space = build_signal_space()
    total = len(signal_space)
    if args.limit:
        signal_space = signal_space[:args.limit]
    print(f"[alpha-lab] Signal space: {len(signal_space)} signals ({total} total defined)")

    catalogue = Catalogue(output_dir)
    n_skip = sum(1 for s in signal_space if catalogue.already_tested(s.name))
    n_todo = len(signal_space) - n_skip
    print(f"[alpha-lab] Skipping {n_skip} already tested, {n_todo} remaining")

    _log(log_path, f"Alpha Lab started. Signals: {len(signal_space)}, Skip: {n_skip}, "
         f"Data: {args.start} to {args.end}, Test start: {args.test_start or 'full period'}")

    t_start = time.time()
    n_done = 0
    n_errors = 0

    for i, spec in enumerate(signal_space):
        if catalogue.already_tested(spec.name):
            continue

        # Inject on-chain data into params for signals that need it
        if _onchain_data is not None and spec.family in (
            "onchain", "onchain_network", "onchain_miner", "onchain_valuation",
            "onchain_activity", "onchain_composite",
        ):
            enriched_params = {**spec.params, "_onchain": _onchain_data}
            enriched_spec = SignalSpec(
                name=spec.name, family=spec.family,
                params=enriched_params, description=spec.description,
            )
        else:
            enriched_spec = spec

        result = run_signal_test(
            spec=enriched_spec,
            close_wide=close_wide,
            volume_wide=volume_wide,
            returns_wide=returns_wide,
            universe_wide=universe_wide,
            test_start=args.test_start,
            cost_bps=args.cost_bps,
        )

        catalogue.record(result)
        n_done += 1
        if result.error:
            n_errors += 1

        ls_sharpe = result.long_short.get("sharpe", np.nan)
        elapsed = result.meta.get("elapsed_sec", 0)
        status = "ERROR" if result.error else f"Sharpe={ls_sharpe:.2f}" if not np.isnan(ls_sharpe) else "no data"

        if n_done % 10 == 0 or n_done == 1 or n_done == n_todo:
            pct = n_done / max(n_todo, 1) * 100
            elapsed_total = time.time() - t_start
            rate = n_done / max(elapsed_total, 1) * 3600
            eta_h = (n_todo - n_done) / max(rate, 1) * 60
            print(
                f"[alpha-lab] [{n_done}/{n_todo}] {pct:5.1f}% | "
                f"{spec.name:<30s} | {status:<16s} | "
                f"{elapsed:.1f}s | rate: {rate:.0f}/hr | ETA: {eta_h:.0f}min"
            )

        _log(log_path, f"[{n_done}/{n_todo}] {spec.name}: {status} ({elapsed:.1f}s)")

    elapsed_total = time.time() - t_start
    elapsed_min = elapsed_total / 60

    print(f"\n{'=' * 70}")
    print(f"  ALPHA LAB COMPLETE")
    print(f"  Signals tested: {n_done} | Errors: {n_errors} | Time: {elapsed_min:.1f} min")
    print(f"{'=' * 70}")

    summary_path = catalogue.write_summary()
    print(f"[alpha-lab] Summary written to {summary_path}")

    _log(log_path, f"Alpha Lab complete. Tested: {n_done}, Errors: {n_errors}, "
         f"Elapsed: {elapsed_min:.1f} min")

    top5 = catalogue.get_ranked(metric="sharpe", mode="long_short")[:5]
    if top5:
        print(f"\n--- TOP 5 LONG-SHORT SIGNALS ---")
        for i, r in enumerate(top5, 1):
            m = r["long_short"]
            print(
                f"  {i}. {r['name']:<30s} "
                f"Sharpe={m.get('sharpe', np.nan):>6.2f}  "
                f"CAGR={m.get('cagr', np.nan):>7.1%}  "
                f"MaxDD={m.get('max_dd', np.nan):>7.1%}"
            )


if __name__ == "__main__":
    main()
