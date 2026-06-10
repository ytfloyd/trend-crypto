#!/usr/bin/env python3
"""Incremental refresh for the 1-minute continuous-futures lake.

IBKR rejects ``endDateTime`` on ``ContFuture`` historical requests, so
the refresh issues a single trailing-window request per symbol. The
``--duration`` is sized to comfortably cover the gap between the lake's
``latest_ts`` and ``now`` (capped at the IBKR small-bar maximum), and
``upsert_bars`` will deduplicate any overlap with existing rows.

Run this once a day to keep the lake current; pair it with the dated
walk for deep-history backfills.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.logging import get_logger
from volbook.contracts import (
    CORE_MACRO_ALIASES,
    OPTIONS_UNDERLYING_ALIASES,
    is_continuous_eligible,
    resolve_futures_spec,
)
from volbook.datalake import CONTINUOUS_EXPIRY, DEFAULT_LAKE_PATH, MinuteLake
from volbook.ibkr_client import IBHistoricalClient

logger = get_logger("volbook.refresh_minute")

MAX_TRAILING_DAYS = 30


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volbook.refresh_futures_minute",
        description="Incrementally update the IBKR continuous-futures minute lake.",
    )
    p.add_argument(
        "--universe",
        choices=["options-underlyings", "core-macro"],
        default="options-underlyings",
    )
    p.add_argument("--aliases", nargs="+", default=None)
    p.add_argument(
        "--lake-path",
        default=str(DEFAULT_LAKE_PATH),
        help=f"DuckDB path for the minute lake (default: {DEFAULT_LAKE_PATH}).",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=2,
        help="Days back from `now` to request when no state exists (default: 2).",
    )
    p.add_argument(
        "--max-days",
        type=int,
        default=MAX_TRAILING_DAYS,
        help=(
            f"Hard cap on the trailing window in days (default: {MAX_TRAILING_DAYS}). "
            "IBKR's 1-min ContFuture limit; gaps larger than this are filled from "
            "the latest call only."
        ),
    )
    p.add_argument(
        "--pace-seconds",
        type=float,
        default=11.0,
        help="Sleep between IB requests (default: 11s for small-bar pacing).",
    )
    p.add_argument("--what-to-show", default="TRADES")
    p.add_argument("--use-rth", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7496)
    p.add_argument(
        "--client-id",
        type=int,
        default=29,
        help="Distinct IB clientId so the lake refresh won't collide with the dashboard refresh.",
    )
    p.add_argument(
        "--market-data-type",
        type=int,
        default=2,
        help="IB reqMarketDataType: 1=live, 2=frozen, 3=delayed, 4=delayed-frozen.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _resolve_aliases(args: argparse.Namespace) -> list[str]:
    if args.aliases:
        return [a.upper() for a in args.aliases]
    return list(
        CORE_MACRO_ALIASES
        if args.universe == "core-macro"
        else OPTIONS_UNDERLYING_ALIASES
    )


def _call_with_reconnect(client: IBHistoricalClient, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ConnectionError, OSError) as exc:
        logger.warning("IB call %s failed (%s); reconnecting", fn.__name__, exc)
        client.reconnect_if_needed()
        return fn(*args, **kwargs)


def _select_duration(*, latest_ts: datetime | None, now: datetime, args: argparse.Namespace) -> str:
    """Choose an IB durationStr that covers the gap, capped at args.max_days."""
    cap = max(1, int(args.max_days))
    if latest_ts is None:
        days = max(1, int(args.lookback_days))
    else:
        gap = (now - latest_ts).total_seconds() / 86400.0
        days = max(1, math.ceil(gap) + 1)
    days = min(days, cap)
    return f"{days} D"


def _refresh_symbol(
    *,
    alias: str,
    client: IBHistoricalClient,
    lake: MinuteLake,
    args: argparse.Namespace,
) -> int:
    base_spec = resolve_futures_spec(alias=alias)
    symbol = base_spec.label_symbol
    if not is_continuous_eligible(base_spec.symbol):
        logger.info(
            "%s: skipped (no IBKR ContFuture security definition; "
            "deep history comes from the dated walk)",
            symbol,
        )
        return 0
    state = lake.get_state(symbol, expiry=CONTINUOUS_EXPIRY)
    now = datetime.now(timezone.utc)
    latest_ts = state.latest_ts if state else None
    duration_str = _select_duration(latest_ts=latest_ts, now=now, args=args)

    if client.reconnect_if_needed():
        logger.warning("IB connection re-established")

    try:
        bars = _call_with_reconnect(
            client,
            client.fetch_continuous_minute_bars,
            base_spec,
            duration=duration_str,
            what_to_show=args.what_to_show,
            use_rth=args.use_rth,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s fetch failed: %s", symbol, exc)
        lake.record_notes(
            symbol, f"refresh_failed: {exc}", expiry=CONTINUOUS_EXPIRY
        )
        return 1

    if not bars:
        logger.info("%s: IB returned 0 bars (duration=%s)", symbol, duration_str)
        return 0
    written = lake.upsert_bars(symbol, bars, expiry=CONTINUOUS_EXPIRY)
    new_state = lake.get_state(symbol, expiry=CONTINUOUS_EXPIRY)
    new_latest = new_state.latest_ts.isoformat() if new_state and new_state.latest_ts else "-"
    logger.info(
        "%s: fetched=%d written=%d duration=%s latest=%s",
        symbol,
        len(bars),
        written,
        duration_str,
        new_latest,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    aliases = _resolve_aliases(args)
    logger.info("Refreshing %d aliases into %s", len(aliases), args.lake_path)

    lake = MinuteLake(args.lake_path)
    lake.connect()
    client = IBHistoricalClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        market_data_type=args.market_data_type,
    )
    total_failures = 0
    try:
        client.connect()
        try:
            for idx, alias in enumerate(aliases, start=1):
                logger.info("[%d/%d] %s", idx, len(aliases), alias)
                total_failures += _refresh_symbol(
                    alias=alias, client=client, lake=lake, args=args
                )
                if args.pace_seconds > 0 and idx < len(aliases):
                    time.sleep(args.pace_seconds)
        finally:
            client.disconnect()
    finally:
        lake.close()

    logger.info("Refresh done: %d aliases, %d failures", len(aliases), total_failures)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
