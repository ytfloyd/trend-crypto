#!/usr/bin/env python3
"""Forward-walking refresh for the dated futures minute lake.

Counterpart to :mod:`scripts.volbook.walk_dated_futures_minute`: instead
of walking *backward* from each contract's earliest stored bar toward
its data head, this CLI walks *forward* from each active contract's
``latest_ts`` toward ``now`` in 30-day chunks. It also optionally seeds
brand-new upcoming expiries that don't yet exist in the lake.

Three reasons we need a separate refresh script (rather than re-using
the existing continuous-only ``refresh_futures_minute.py``):

1. The continuous refresh updates ``expiry='continuous'`` — that path
   uses IBKR's ``ContFuture`` historical contract, which tracks the
   front month without an ``endDateTime`` parameter. Useful for the
   rolling minute view, but the dated lake is what feeds deep history.

2. Active dated contracts (current front-month, plus the next 1-2
   listed quarterlies) keep printing bars every trading day. Without a
   forward refresh, the lake's ``latest_ts`` for these contracts goes
   stale immediately after the deep walk.

3. New expiries get listed every quarter. The seeding step adds them
   to the lake with a trailing 30-day pull so they're caught the day
   they start trading; the deep walker isn't usually re-run at that
   cadence.

Run this once per trading day alongside ``refresh_futures_minute.py``
to keep both the continuous and dated front-month views current.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.logging import get_logger
from volbook.contracts import (
    CORE_MACRO_ALIASES,
    OPTIONS_UNDERLYING_ALIASES,
    FuturesSpec,
    enumerate_dated_specs,
    resolve_futures_spec,
)
from volbook.datalake import CONTINUOUS_EXPIRY, DEFAULT_LAKE_PATH, MinuteLake
from volbook.ibkr_client import HistoricalDataTimeout, IBHistoricalClient

logger = get_logger("volbook.refresh_dated_minute")

# Mirrored from walk_dated_futures_minute so the retry behavior stays
# identical between the deep walker and the daily refresh.
_TIMEOUT_RETRY_ATTEMPTS = 3
_TIMEOUT_BACKOFF_SECONDS = 15.0


@dataclass(frozen=True)
class RefreshFailure:
    alias: str
    expiry: str
    stage: str
    error: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volbook.refresh_dated_futures_minute",
        description=(
            "Forward-walk every active dated `(symbol, expiry)` in the "
            "volbook minute lake from its latest_ts up to `now`."
        ),
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
        "--keep-active-days",
        type=int,
        default=14,
        help=(
            "Refresh contracts whose expiry month ended within this many "
            "days. Default: 14. Older expiries are fixed history and skipped."
        ),
    )
    p.add_argument(
        "--max-new-expiries",
        type=int,
        default=0,
        help=(
            "Seed up to this many never-seen-before upcoming expiries per "
            "symbol with a trailing 30-day pull (default: 0 = disabled). "
            "Useful when a quarter rolls over and a new contract lists."
        ),
    )
    p.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="IB durationStr days per request (default: 30, IB's 1-min max).",
    )
    p.add_argument(
        "--max-chunks-per-contract",
        type=int,
        default=0,
        help="Optional cap on chunks per contract (0 = walk to now).",
    )
    p.add_argument(
        "--pace-seconds",
        type=float,
        default=11.0,
        help="Sleep between IB requests to respect 60/10min small-bar pacing.",
    )
    p.add_argument("--what-to-show", default="TRADES")
    p.add_argument("--use-rth", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7496)
    p.add_argument(
        "--client-id",
        type=int,
        default=31,
        help="Distinct IB clientId so the dated refresh won't collide with the walker (30) or continuous refresh (29).",
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


def _expiry_first_of_month(expiry: str) -> datetime:
    return datetime(int(expiry[:4]), int(expiry[4:6]), 1, tzinfo=timezone.utc)


def is_expiry_active(
    expiry: str, *, keep_active_days: int, today: datetime
) -> bool:
    """A dated expiry is "active" if its month + grace window is on/after today.

    We use first-of-next-month as a coarse proxy for the contract's last
    trade date (which varies by product but is always inside the expiry
    month). With ``keep_active_days=14``, a Jun 2025 contract stays
    refreshable through ~mid-July 2025 — long enough to backfill any
    final-day prints we missed, then it's left alone.
    """
    if len(expiry) < 6 or not expiry.isdigit():
        return False
    end_of_month = _expiry_first_of_month(expiry) + timedelta(days=32)
    end_of_month = end_of_month.replace(day=1)
    cutoff = end_of_month + timedelta(days=keep_active_days)
    return cutoff.date() >= today.date()


def _ensure_connection(client: IBHistoricalClient) -> None:
    if client.reconnect_if_needed():
        logger.warning("IB connection re-established")


def _forward_cursors(
    *, latest_ts: datetime | None, now: datetime, chunk_days: int
) -> list[datetime]:
    """End-cursor sequence for the forward walk.

    IB ``reqHistoricalData`` returns bars *ending* at ``end_datetime``,
    so to walk forward we step the end-cursor from ``latest_ts +
    chunk_days`` toward ``now`` in ``chunk_days`` increments. A trailing
    chunk anchored at ``now`` always closes the sequence so the front
    edge stays current.
    """
    if latest_ts is None:
        return [now]
    if latest_ts >= now:
        return []
    cursors: list[datetime] = []
    cursor = latest_ts + timedelta(days=chunk_days)
    while cursor < now:
        cursors.append(cursor)
        cursor = cursor + timedelta(days=chunk_days)
    cursors.append(now)
    return cursors


def _refresh_dated_contract(
    *,
    spec: FuturesSpec,
    contract: object,
    client: IBHistoricalClient,
    lake: MinuteLake,
    args: argparse.Namespace,
) -> list[RefreshFailure]:
    """Forward-walk one ``(symbol, expiry)`` from latest_ts to now."""
    failures: list[RefreshFailure] = []
    symbol = spec.label_symbol
    expiry = spec.expiry
    state = lake.get_state(symbol, expiry=expiry)
    latest_ts = state.latest_ts if state else None
    now = datetime.now(timezone.utc)
    cursors = _forward_cursors(
        latest_ts=latest_ts, now=now, chunk_days=args.chunk_days
    )
    if not cursors:
        logger.info(
            "%s: latest_ts=%s already at/after now, skipping",
            spec.label,
            latest_ts.isoformat() if latest_ts else "-",
        )
        return failures

    cap = args.max_chunks_per_contract or 0
    if cap:
        cursors = cursors[:cap]

    duration_str = f"{args.chunk_days} D"
    written_total = 0
    fetched_total = 0

    logger.info(
        "%s: forward-walking from %s to %s in %d chunks (duration=%s)",
        spec.label,
        latest_ts.isoformat() if latest_ts else "(seed)",
        now.isoformat(),
        len(cursors),
        duration_str,
    )

    for chunk_idx, end_cursor in enumerate(cursors, start=1):
        try:
            bars = client.fetch_dated_minute_bars_with_retry(
                contract,
                end_datetime=end_cursor,
                duration=duration_str,
                what_to_show=args.what_to_show,
                use_rth=args.use_rth,
                max_retries=_TIMEOUT_RETRY_ATTEMPTS,
                backoff_seconds=_TIMEOUT_BACKOFF_SECONDS,
                label=f"{spec.label} end={end_cursor.isoformat()}",
            )
        except HistoricalDataTimeout as exc:
            logger.warning(
                "%s: persistent timeout at end=%s; bailing out",
                spec.label,
                end_cursor.isoformat(),
            )
            failures.append(
                RefreshFailure(
                    alias=spec.label, expiry=expiry, stage="timeout", error=str(exc)
                )
            )
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s chunk end=%s failed: %s",
                spec.label,
                end_cursor.isoformat(),
                exc,
            )
            failures.append(
                RefreshFailure(
                    alias=spec.label, expiry=expiry, stage="fetch", error=str(exc)
                )
            )
            break

        if bars:
            written = lake.upsert_bars(symbol, bars, expiry=expiry)
            written_total += written
            fetched_total += len(bars)
        # 0 bars in the middle of a forward-walk just means a quiet
        # window (e.g. a weekend) — keep walking unless we're already
        # at the trailing cursor.
        if chunk_idx < len(cursors) and args.pace_seconds > 0:
            time.sleep(args.pace_seconds)

    if written_total > 0 or fetched_total > 0:
        new_state = lake.get_state(symbol, expiry=expiry)
        new_latest = (
            new_state.latest_ts.isoformat() if new_state and new_state.latest_ts else "-"
        )
        logger.info(
            "%s: %d chunks, fetched=%d written=%d latest=%s",
            spec.label,
            len(cursors),
            fetched_total,
            written_total,
            new_latest,
        )
    else:
        logger.info("%s: 0 new bars across %d chunks", spec.label, len(cursors))
    return failures


def _seed_new_expiries(
    *,
    base_spec: FuturesSpec,
    known_expiries: set[str],
    client: IBHistoricalClient,
    lake: MinuteLake,
    args: argparse.Namespace,
    today: datetime,
) -> list[RefreshFailure]:
    """Pull a single trailing-30D chunk for any never-seen-before upcoming expiry."""
    failures: list[RefreshFailure] = []
    if args.max_new_expiries <= 0:
        return failures
    floor = f"{today.year:04d}{today.month:02d}"
    horizon = today + timedelta(days=31 * args.max_new_expiries + 31)
    ceiling = f"{horizon.year:04d}{horizon.month:02d}"
    candidates = enumerate_dated_specs(
        base_spec, min_expiry=floor, max_expiry=ceiling
    )
    fresh = [s for s in candidates if s.expiry not in known_expiries]
    if not fresh:
        return failures
    fresh = fresh[: args.max_new_expiries]
    logger.info(
        "%s: probing %d new expiries: %s",
        base_spec.label_symbol,
        len(fresh),
        ", ".join(s.expiry for s in fresh),
    )
    try:
        qualified_pairs = client.qualify_dated_futures(fresh)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s: batch qualify failed during seeding: %s",
            base_spec.label_symbol,
            exc,
        )
        return [
            RefreshFailure(
                alias=base_spec.label_symbol,
                expiry="-",
                stage="seed_qualify",
                error=str(exc),
            )
        ]
    listed = [(s, c) for s, c in qualified_pairs if c is not None]
    if not listed:
        logger.info("%s: no new expiries listed at IB yet", base_spec.label_symbol)
        return failures

    duration_str = f"{args.chunk_days} D"
    for spec, contract in listed:
        try:
            bars = client.fetch_dated_minute_bars_with_retry(
                contract,
                end_datetime=None,
                duration=duration_str,
                what_to_show=args.what_to_show,
                use_rth=args.use_rth,
                max_retries=_TIMEOUT_RETRY_ATTEMPTS,
                backoff_seconds=_TIMEOUT_BACKOFF_SECONDS,
                label=f"{spec.label} (seed)",
            )
        except HistoricalDataTimeout as exc:
            logger.warning("%s seed timed out: %s", spec.label, exc)
            failures.append(
                RefreshFailure(
                    alias=spec.label, expiry=spec.expiry, stage="seed_timeout", error=str(exc)
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s seed failed: %s", spec.label, exc)
            failures.append(
                RefreshFailure(
                    alias=spec.label, expiry=spec.expiry, stage="seed_fetch", error=str(exc)
                )
            )
            continue
        if bars:
            written = lake.upsert_bars(spec.label_symbol, bars, expiry=spec.expiry)
            logger.info(
                "%s seed: fetched=%d written=%d",
                spec.label,
                len(bars),
                written,
            )
        else:
            lake.record_notes(
                spec.label_symbol, "seed_empty", expiry=spec.expiry
            )
            logger.info("%s seed: 0 bars (not yet trading)", spec.label)
        if args.pace_seconds > 0:
            time.sleep(args.pace_seconds)
    return failures


def _refresh_symbol(
    *,
    alias: str,
    client: IBHistoricalClient,
    lake: MinuteLake,
    args: argparse.Namespace,
) -> list[RefreshFailure]:
    failures: list[RefreshFailure] = []
    base_spec = resolve_futures_spec(alias=alias)
    symbol = base_spec.label_symbol
    today = datetime.now(timezone.utc)

    states = [
        s
        for s in lake.list_states()
        if s.symbol == symbol and s.expiry != CONTINUOUS_EXPIRY
    ]
    known_expiries = {s.expiry for s in states}
    active = [
        s
        for s in states
        if is_expiry_active(
            s.expiry, keep_active_days=args.keep_active_days, today=today
        )
    ]

    if not active and args.max_new_expiries <= 0:
        logger.info(
            "%s: no active dated expiries (known=%d), skipping",
            symbol,
            len(states),
        )
        return failures

    logger.info(
        "%s: %d active expiries (of %d known): %s",
        symbol,
        len(active),
        len(states),
        ", ".join(sorted(s.expiry for s in active)) or "-",
    )

    if active:
        specs = [base_spec.with_expiry(s.expiry) for s in active]
        try:
            qualified_pairs = client.qualify_dated_futures(specs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: batch qualify failed: %s", symbol, exc)
            return [RefreshFailure(alias=symbol, expiry="-", stage="qualify", error=str(exc))]
        listed = [(s, c) for s, c in qualified_pairs if c is not None]
        if len(listed) < len(specs):
            unlisted = [s.expiry for s, c in qualified_pairs if c is None]
            logger.warning(
                "%s: %d/%d expiries no longer listed at IB: %s",
                symbol,
                len(unlisted),
                len(specs),
                ", ".join(unlisted),
            )

        for idx, (spec, contract) in enumerate(listed, start=1):
            logger.info("[%d/%d] %s", idx, len(listed), spec.label)
            failures.extend(
                _refresh_dated_contract(
                    spec=spec, contract=contract, client=client, lake=lake, args=args
                )
            )
            if args.pace_seconds > 0 and idx < len(listed):
                time.sleep(args.pace_seconds)

    failures.extend(
        _seed_new_expiries(
            base_spec=base_spec,
            known_expiries=known_expiries,
            client=client,
            lake=lake,
            args=args,
            today=today,
        )
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
        "Refreshing %d dated aliases into %s (keep_active_days=%d, max_new_expiries=%d)",
        len(aliases),
        args.lake_path,
        args.keep_active_days,
        args.max_new_expiries,
    )

    lake = MinuteLake(args.lake_path)
    lake.connect()
    client = IBHistoricalClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        market_data_type=args.market_data_type,
    )
    failures: list[RefreshFailure] = []
    try:
        client.connect()
        try:
            for idx, alias in enumerate(aliases, start=1):
                logger.info("===== [%d/%d] %s =====", idx, len(aliases), alias)
                _ensure_connection(client)
                failures.extend(
                    _refresh_symbol(alias=alias, client=client, lake=lake, args=args)
                )
                if args.pace_seconds > 0 and idx < len(aliases):
                    time.sleep(args.pace_seconds)
        finally:
            client.disconnect()
    finally:
        lake.close()

    failed_aliases = {f.alias for f in failures}
    logger.info(
        "Dated refresh done: %d aliases, %d failures (across %d aliases)",
        len(aliases),
        len(failures),
        len(failed_aliases),
    )
    if failures:
        for f in failures:
            logger.warning("  %s [%s/%s]: %s", f.alias, f.expiry, f.stage, f.error)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
