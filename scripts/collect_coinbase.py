#!/usr/bin/env python
"""CLI for Coinbase Advanced Trade data collection.

Collects ALL spot products by default (every quote currency).
ADV filtering and universe selection belong in database views, not here.

Subcommands:
  discover   List all available spot products
  backfill   Download full history for all or specific symbols
  update     Incremental update from last known timestamp (for cron)
  refresh    Materialize resampled clean tables (bars_1h_clean, etc.)
  status     Show DB summary (symbols, row counts, date ranges)

Examples:
  # Discover all available spot products
  python scripts/collect_coinbase.py discover

  # Discover only BTC-quoted products
  python scripts/collect_coinbase.py discover --quotes BTC

  # Backfill every spot product on Coinbase
  python scripts/collect_coinbase.py backfill --all --workers 8

  # Backfill only USD and BTC-quoted products
  python scripts/collect_coinbase.py backfill --all --quotes USD,BTC --workers 8

  # Backfill single symbol
  python scripts/collect_coinbase.py backfill --symbol ETH-BTC

  # Incremental update (add to crontab), parallel
  python scripts/collect_coinbase.py update --workers 8

  # Materialize clean tables for fast backtest
  python scripts/collect_coinbase.py refresh

  # Show database status
  python scripts/collect_coinbase.py status

Environment Variables:
  TREND_CRYPTO_DB        DuckDB path (fallback if --db not provided)
  COINBASE_API_KEY       Coinbase Advanced Trade API key
  COINBASE_API_SECRET    Coinbase Advanced Trade API secret
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "src"))


def _resolve_db(args: argparse.Namespace) -> str:
    db = getattr(args, "db", None) or os.environ.get("TREND_CRYPTO_DB")
    if not db:
        db = "data/market.duckdb"
    return db


def _parse_quotes(args: argparse.Namespace) -> set[str] | None:
    """Parse --quotes flag into a set, or None for all."""
    raw = getattr(args, "quotes", None)
    if not raw:
        return None
    return {q.strip().upper() for q in raw.split(",")}


def _make_collector(args: argparse.Namespace) -> "CoinbaseCollector":  # noqa: F821
    from data.collector import CoinbaseCollector

    return CoinbaseCollector(
        db_path=_resolve_db(args),
        api_key=getattr(args, "api_key", None) or os.environ.get("COINBASE_API_KEY"),
        api_secret=getattr(args, "api_secret", None) or os.environ.get("COINBASE_API_SECRET"),
        max_rps=getattr(args, "max_rps", 25.0),
    )


def cmd_discover(args: argparse.Namespace) -> None:
    collector = _make_collector(args)
    quote_filter = _parse_quotes(args)
    symbols = collector.discover_products(quote_currencies=quote_filter)

    by_quote: dict[str, list[str]] = defaultdict(list)
    for sym in symbols:
        parts = sym.rsplit("-", 1)
        quote = parts[1] if len(parts) == 2 else "?"
        by_quote[quote].append(sym)

    scope = "all" if quote_filter is None else ",".join(sorted(quote_filter))
    print(f"\nFound {len(symbols)} spot products (quotes: {scope}):\n")

    for quote in sorted(by_quote, key=lambda q: -len(by_quote[q])):
        syms = by_quote[quote]
        print(f"  ── {quote} ({len(syms)}) ──")
        for i, sym in enumerate(syms, 1):
            print(f"    {i:3d}. {sym}")
        print()

    collector.close()


def cmd_backfill(args: argparse.Namespace) -> None:
    collector = _make_collector(args)

    start = None
    if args.start:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))

    end = None
    if args.end:
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))

    workers = getattr(args, "workers", 1)

    if args.symbol:
        rows = collector.backfill_symbol(symbol=args.symbol, start=start, end=end)
        print(f"\n{args.symbol}: {rows} rows inserted")
    elif args.all:
        quote_filter = _parse_quotes(args)
        results = collector.backfill_all(
            min_history_days=args.min_history_days,
            workers=workers,
            start=start,
            end=end,
            quote_currencies=quote_filter,
        )
        print("\nBackfill complete:")
        for sym, rows in sorted(results.items()):
            if rows > 0:
                print(f"  {sym}: {rows:,} rows")
        total = sum(results.values())
        success = sum(1 for v in results.values() if v > 0)
        print(f"\n  Total: {total:,} rows across {success} symbols")
    else:
        print("Error: specify --symbol or --all")
        sys.exit(1)

    collector.close()


def cmd_update(args: argparse.Namespace) -> None:
    collector = _make_collector(args)
    workers = getattr(args, "workers", 1)

    if args.symbol:
        rows = collector.update_symbol(args.symbol)
        print(f"{args.symbol}: {rows} new rows")
    else:
        results = collector.update_all(workers=workers)
        total = sum(results.values())
        if total > 0:
            print(f"Updated {len(results)} symbols, {total:,} new rows")
        else:
            print("All symbols up to date")

    collector.close()


def cmd_refresh(args: argparse.Namespace) -> None:
    collector = _make_collector(args)
    collector.refresh_clean_tables()
    print("Clean tables refreshed: bars_1h_clean, bars_4h_clean, bars_1d_clean")
    collector.close()


def cmd_status(args: argparse.Namespace) -> None:
    collector = _make_collector(args)
    rows = collector.status()

    if not rows:
        print("Database is empty. Run 'backfill' first.")
        collector.close()
        return

    print(f"\n{'Symbol':<16s} {'Rows':>12s} {'First':>22s} {'Last':>22s} {'Behind':>10s}")
    print(f"{'─' * 16} {'─' * 12} {'─' * 22} {'─' * 22} {'─' * 10}")

    total_rows = 0
    for r in rows:
        first = r["first_ts"].strftime("%Y-%m-%d %H:%M") if r["first_ts"] else "N/A"
        last = r["last_ts"].strftime("%Y-%m-%d %H:%M") if r["last_ts"] else "N/A"
        behind = f"{r['hours_behind']:.1f}h" if r["hours_behind"] is not None else "N/A"
        print(f"{r['symbol']:<16s} {r['row_count']:>12,d} {first:>22s} {last:>22s} {behind:>10s}")
        total_rows += r["row_count"]

    print(f"\nTotal: {len(rows)} symbols, {total_rows:,} rows")
    print(f"Database: {_resolve_db(args)}\n")
    collector.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coinbase Advanced Trade data collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global options
    parser.add_argument("--db", help="DuckDB path (or set TREND_CRYPTO_DB)")
    parser.add_argument("--api-key", help="Coinbase API key (or set COINBASE_API_KEY)")
    parser.add_argument("--api-secret", help="Coinbase API secret (or set COINBASE_API_SECRET)")
    parser.add_argument("--max-rps", type=float, default=25.0, help="Max requests/sec (default: 25)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # discover
    disc = sub.add_parser("discover", help="List available spot products")
    disc.add_argument("--quotes", default=None,
                      help="Comma-separated quote currencies to filter (default: all)")

    # backfill
    bf = sub.add_parser("backfill", help="Download full history")
    bf.add_argument("--symbol", help="Single symbol to backfill (e.g. BTC-USD, ETH-BTC)")
    bf.add_argument("--all", action="store_true", help="Backfill all spot products")
    bf.add_argument("--quotes", default=None,
                    help="Comma-separated quote currencies to limit (default: all)")
    bf.add_argument("--start", help="Start date (ISO8601, default: auto-discover)")
    bf.add_argument("--end", help="End date (ISO8601, default: now)")
    bf.add_argument("--min-history-days", type=int, default=0,
                    help="Skip symbols with less history (default: 0 = keep all)")
    bf.add_argument("--workers", type=int, default=1,
                    help="Parallel workers for backfill (default: 1 = sequential)")

    # update
    up = sub.add_parser("update", help="Incremental update from last timestamp")
    up.add_argument("--symbol", help="Update single symbol (default: all)")
    up.add_argument("--workers", type=int, default=1,
                    help="Parallel workers for update (default: 1 = sequential)")

    # refresh
    sub.add_parser("refresh", help="Materialize resampled clean tables")

    # status
    sub.add_parser("status", help="Show DB summary")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    commands = {
        "discover": cmd_discover,
        "backfill": cmd_backfill,
        "update": cmd_update,
        "refresh": cmd_refresh,
        "status": cmd_status,
    }

    if not args.command:
        print("Error: specify a subcommand (discover, backfill, update, refresh, status)")
        sys.exit(1)

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
