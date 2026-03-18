#!/usr/bin/env python3
"""
Medallion Lite — live signal runner.

Run this on a cron schedule (every hour) to compute target weights
and publish them for the execution engine.

Usage:
    # Single cycle (cron mode):
    python scripts/run_medallion_live.py

    # Continuous daemon (every hour):
    python scripts/run_medallion_live.py --daemon

    # Paper trading with existing LiveRunner:
    python scripts/run_medallion_live.py --paper

Outputs:
    signal_output.json   — target weights for execution engine
    medallion_state.json — persistent portfolio state
    live_signals table   — DuckDB signal history
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from live.medallion_signal import MedallionSignalService, SignalConfig


def main():
    parser = argparse.ArgumentParser(description="Medallion Lite Live Signal Service")
    parser.add_argument("--db", default=None,
                        help="Path to market.duckdb (default: auto-detect)")
    parser.add_argument("--state-dir", default=str(ROOT / "live_state"),
                        help="Directory for state + output files")
    parser.add_argument("--daemon", action="store_true",
                        help="Run continuously, cycling every hour")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Daemon interval in seconds (default: 3600)")
    parser.add_argument("--paper", action="store_true",
                        help="Run paper trading with existing LiveRunner")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    config = SignalConfig(
        state_path=str(state_dir / "medallion_state.json"),
        output_path=str(state_dir / "signal_output.json"),
    )
    if args.db:
        config.db_path = args.db

    svc = MedallionSignalService(config)

    if args.paper:
        _run_paper(svc, config)
        return

    if args.daemon:
        logging.info("Starting daemon mode (interval=%ds)", args.interval)
        while True:
            try:
                output = svc.run_cycle()
                _print_summary(output)
            except Exception:
                logging.exception("Cycle failed")
            time.sleep(args.interval)
    else:
        output = svc.run_cycle()
        _print_summary(output)


def _print_summary(output):
    tw = output.target_weights
    active = {k: v for k, v in tw.items() if v > 0}
    print(f"\n{'='*60}")
    print(f"  Cycle: {output.cycle_id}  |  {output.ts}")
    print(f"  Regime: {output.regime_score:.2f}  |  "
          f"Holdings: {output.diagnostics.get('n_holdings', 0)}  |  "
          f"Exposure: {sum(tw.values()):.1%}")
    if active:
        print(f"  {'─'*56}")
        for sym, w in sorted(active.items(), key=lambda x: -x[1]):
            print(f"    {sym:<15s} {w:>8.2%}")
    if output.actions:
        print(f"  {'─'*56}")
        for a in output.actions:
            print(f"    {a['action']:<15s} {a['symbol']:<15s}", end="")
            if a.get("hours_held"):
                print(f"  held={a['hours_held']}h  ret={a.get('cum_ret', 0):.1%}", end="")
            print()
    if output.stale:
        print(f"  ⚠ DATA STALE ({output.diagnostics.get('data_freshness_hours', '?')}h)")
    print(f"{'='*60}\n")


def _run_paper(svc, config):
    """Run paper trading using the existing LiveRunner infrastructure."""
    from data.live_feed import DuckDBLiveDataFeed
    from execution.paper_broker import PaperBroker
    from execution.oms import OrderManagementSystem
    from strategy.medallion_portfolio import MedallionEmbeddedAdapter

    logging.info("Starting paper trading mode")

    output = svc.run_cycle()
    symbols = [s for s, w in output.target_weights.items() if w > 0]
    if not symbols:
        symbols = list(output.target_weights.keys())[:10]

    feed = DuckDBLiveDataFeed(db_path=config.db_path, symbols=symbols)
    broker = PaperBroker(fee_bps=30.0, slippage_bps=5.0)
    oms = OrderManagementSystem(broker, deadband=0.01)
    adapter = MedallionEmbeddedAdapter(svc)

    logging.info("Paper trading with %d symbols", len(symbols))
    logging.info("Weights: %s", {k: f"{v:.2%}" for k, v in output.target_weights.items() if v > 0})

    # Single cycle demonstration
    contexts = {}
    from strategy.context import StrategyContext
    for sym in symbols:
        history = feed.get_history(sym, 200)
        if not history.is_empty():
            from strategy.context import make_strategy_context
            ctx = make_strategy_context(history, history.height - 1, 200)
            contexts[sym] = ctx

    weights = adapter.on_bar_close_portfolio(contexts)
    active = {k: v for k, v in weights.items() if v > 0}

    logging.info("Target weights computed:")
    for sym, w in sorted(active.items(), key=lambda x: -x[1]):
        logging.info("  %s: %.2f%%", sym, w * 100)

    nav = broker.nav()
    result = oms.rebalance_to_targets(weights, {s: 0.0 for s in symbols}, nav)
    logging.info(
        "Rebalance: %d submitted, %d filled",
        result.orders_submitted, result.orders_filled,
    )


if __name__ == "__main__":
    main()
