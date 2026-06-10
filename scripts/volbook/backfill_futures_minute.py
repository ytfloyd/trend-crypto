#!/usr/bin/env python3
"""Backfill 1-minute continuous-futures bars from IBKR into the lake.

IBKR rejects ``endDateTime`` on ``ContFuture`` historical requests
(error 10339, followed by a socket disconnect), so we cannot chunk
backward through history on the continuous contract. Instead this
script issues a single trailing-window request per symbol and writes
whatever IBKR returns.

The ``--duration`` flag controls how far back IB is asked to look.
Without ``endDateTime`` IB will return the most recent ``duration`` of
1-minute bars from its continuous front-month series. For 1-minute
bars on ``ContFuture`` the practical max is ``"30 D"``; longer values
are typically silently truncated by IB.

Each successful symbol updates the lake's ingest state with IB's head
timestamp so deeper history can be filled in later by the dated
contract walk.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.logging import get_logger
from volbook.contracts import (
    CORE_MACRO_ALIASES,
    OPTIONS_UNDERLYING_ALIASES,
    resolve_futures_spec,
)
from volbook.datalake import CONTINUOUS_EXPIRY, DEFAULT_LAKE_PATH, MinuteLake
from volbook.ibkr_client import IBHistoricalClient

logger = get_logger("volbook.backfill_minute")


@dataclass(frozen=True)
class FetchFailure:
    alias: str
    stage: str
    error: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volbook.backfill_futures_minute",
        description="Backfill 1-minute IBKR continuous-futures bars into DuckDB.",
    )
    p.add_argument(
        "--universe",
        choices=["options-underlyings", "core-macro"],
        default="options-underlyings",
        help="Alias universe to backfill when --aliases is not supplied.",
    )
    p.add_argument(
        "--aliases",
        nargs="+",
        default=None,
        help="Root aliases whose continuous front contracts should be backfilled.",
    )
    p.add_argument(
        "--lake-path",
        default=str(DEFAULT_LAKE_PATH),
        help=f"DuckDB path for the minute lake (default: {DEFAULT_LAKE_PATH}).",
    )
    p.add_argument(
        "--duration",
        default="30 D",
        help=(
            "IB durationStr for the trailing-window request. Max usable value "
            "for 1-minute ContFuture is around '30 D' (default)."
        ),
    )
    p.add_argument(
        "--pace-seconds",
        type=float,
        default=11.0,
        help="Sleep between IB requests to respect 60/10min small-bar pacing (default: 11s).",
    )
    p.add_argument(
        "--what-to-show",
        default="TRADES",
        help="IB whatToShow value (default: TRADES).",
    )
    p.add_argument(
        "--use-rth",
        action="store_true",
        help="Restrict to regular trading hours.",
    )
    p.add_argument(
        "--skip-head-timestamp",
        action="store_true",
        help="Skip the reqHeadTimeStamp call; useful when IB has already returned it.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7496)
    p.add_argument(
        "--client-id",
        type=int,
        default=29,
        help="Distinct IB clientId so the backfill won't collide with the refresh scripts.",
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


def _ensure_connection(client: IBHistoricalClient) -> None:
    if client.reconnect_if_needed():
        logger.warning("IB connection re-established")


def _call_with_reconnect(client: IBHistoricalClient, fn, *args, **kwargs):
    """Run an IB call, reconnect once on transient connection errors, then retry."""
    try:
        return fn(*args, **kwargs)
    except (ConnectionError, OSError) as exc:
        logger.warning("IB call %s failed (%s); reconnecting", fn.__name__, exc)
        client.reconnect_if_needed()
        return fn(*args, **kwargs)


def _backfill_symbol(
    *,
    alias: str,
    client: IBHistoricalClient,
    lake: MinuteLake,
    args: argparse.Namespace,
) -> list[FetchFailure]:
    failures: list[FetchFailure] = []
    base_spec = resolve_futures_spec(alias=alias)
    symbol = base_spec.label_symbol

    _ensure_connection(client)

    if not args.skip_head_timestamp:
        try:
            head_ts = _call_with_reconnect(
                client,
                client.head_timestamp_continuous,
                base_spec,
                what_to_show=args.what_to_show,
                use_rth=args.use_rth,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Head timestamp failed for %s: %s", alias, exc)
            lake.record_notes(
                symbol, f"head_failed: {exc}", expiry=CONTINUOUS_EXPIRY
            )
            failures.append(FetchFailure(alias=alias, stage="head", error=str(exc)))
            return failures

        if head_ts is not None:
            lake.set_head_timestamp(symbol, head_ts, expiry=CONTINUOUS_EXPIRY)
            logger.info("%s: head_timestamp=%s", symbol, head_ts.isoformat())
        else:
            logger.warning("%s: IB returned no head timestamp", symbol)
            lake.record_notes(
                symbol, "head_returned_none", expiry=CONTINUOUS_EXPIRY
            )

    try:
        bars = _call_with_reconnect(
            client,
            client.fetch_continuous_minute_bars,
            base_spec,
            duration=args.duration,
            what_to_show=args.what_to_show,
            use_rth=args.use_rth,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s trailing fetch failed: %s", symbol, exc)
        lake.record_notes(
            symbol, f"fetch_failed: {exc}", expiry=CONTINUOUS_EXPIRY
        )
        failures.append(FetchFailure(alias=alias, stage="fetch", error=str(exc)))
        return failures

    if not bars:
        logger.warning("%s: IB returned 0 bars", symbol)
        lake.record_notes(
            symbol, "fetch_empty", expiry=CONTINUOUS_EXPIRY
        )
        return failures

    written = lake.upsert_bars(symbol, bars, expiry=CONTINUOUS_EXPIRY)
    lake.record_notes(symbol, "", expiry=CONTINUOUS_EXPIRY)
    state = lake.get_state(symbol, expiry=CONTINUOUS_EXPIRY)
    earliest = state.earliest_ts.isoformat() if state and state.earliest_ts else "-"
    latest = state.latest_ts.isoformat() if state and state.latest_ts else "-"
    logger.info(
        "%s: fetched=%d written=%d earliest=%s latest=%s",
        symbol,
        len(bars),
        written,
        earliest,
        latest,
    )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    aliases = _resolve_aliases(args)
    logger.info(
        "Backfilling %d aliases into %s (duration=%s)",
        len(aliases),
        args.lake_path,
        args.duration,
    )

    lake = MinuteLake(args.lake_path)
    lake.connect()
    client = IBHistoricalClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        market_data_type=args.market_data_type,
    )
    failures: list[FetchFailure] = []
    try:
        client.connect()
        try:
            for idx, alias in enumerate(aliases, start=1):
                logger.info("[%d/%d] %s", idx, len(aliases), alias)
                failures.extend(
                    _backfill_symbol(
                        alias=alias, client=client, lake=lake, args=args
                    )
                )
                if args.pace_seconds > 0 and idx < len(aliases):
                    time.sleep(args.pace_seconds)
        finally:
            client.disconnect()
    finally:
        lake.close()

    failed_aliases = {f.alias for f in failures}
    succeeded = len(aliases) - len(failed_aliases)
    logger.info(
        "Backfill done: %d aliases OK, %d failed (%d failure events)",
        succeeded,
        len(failed_aliases),
        len(failures),
    )
    if failures:
        for f in failures:
            logger.warning("  %s [%s]: %s", f.alias, f.stage, f.error)
    return 0 if succeeded > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
