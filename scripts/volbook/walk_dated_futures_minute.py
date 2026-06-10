#!/usr/bin/env python3
"""Walk dated futures contracts and backfill 1-minute history into the lake.

Unlike :mod:`scripts.volbook.backfill_futures_minute` (which is bound by
IBKR's continuous-future limitation that rejects ``endDateTime``), this
script walks each historical *dated* expiry of a product and chunks
backward through 1-minute bars in 30-day windows. Bars are stored under
``expiry='YYYYMM'`` so :meth:`MinuteLake.stitch_continuous_series` can
later produce a true historical front-month series.

Per ``(symbol, expiry)`` workflow:
1. Qualify the dated contract (with ``includeExpired=True``).
2. Cache IB's head timestamp.
3. Walk backward from the contract's last-trade date (or ``now`` for an
   active contract) toward the head, 30 days per IB call, sleeping
   ``--pace-seconds`` between requests.
4. Stop when the next end-cursor would precede the head, when IB returns
   zero bars (no more data), or when ``--max-chunks-per-contract`` is hit.

The script is resumable: each new run consults ``ingest_state`` and
restarts from the existing ``earliest_ts`` for partially-walked
contracts.
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

logger = get_logger("volbook.walk_dated_minute")

# Number of times to retry a chunk that comes back as a timeout (not "no
# data"). 3 attempts × ~5 min timeout = up to ~15 min on a single stuck
# request, after which we bail rather than mistake it for the data head.
_TIMEOUT_RETRY_ATTEMPTS = 3
_TIMEOUT_BACKOFF_SECONDS = 15.0


@dataclass(frozen=True)
class WalkFailure:
    alias: str
    expiry: str
    stage: str
    error: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volbook.walk_dated_futures_minute",
        description=(
            "Walk historical dated futures contracts and backfill 1-minute "
            "OHLCV into the volbook DuckDB lake."
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
        "--years",
        type=int,
        default=5,
        help=(
            "Walk back this many years from the current month (default: 5). "
            "Ignored if --min-expiry is supplied."
        ),
    )
    p.add_argument(
        "--min-expiry",
        default=None,
        help="Hard floor on contract expiry (YYYYMM). Overrides --years.",
    )
    p.add_argument(
        "--max-expiry",
        default=None,
        help="Hard ceiling on contract expiry (YYYYMM). Default: no cap.",
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
        help="Optional cap on chunks per dated contract (0 = walk to head).",
    )
    p.add_argument(
        "--max-expiries-per-symbol",
        type=int,
        default=0,
        help="Optional cap on contracts per symbol (0 = no cap).",
    )
    p.add_argument(
        "--oldest-first",
        action="store_true",
        help="Walk oldest expiries first (default: newest first).",
    )
    p.add_argument(
        "--skip-active",
        action="store_true",
        help="Skip contracts whose expiry month is in the future.",
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
        default=30,
        help="Distinct IB clientId so the dated walker won't collide with other refreshers.",
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


def _floor_min_expiry(args: argparse.Namespace) -> str:
    if args.min_expiry:
        return args.min_expiry
    today = datetime.now(timezone.utc)
    floor_year = today.year - max(0, args.years)
    return f"{floor_year:04d}{today.month:02d}"


def _expiry_first_of_month(expiry: str) -> datetime:
    return datetime(int(expiry[:4]), int(expiry[4:6]), 1, tzinfo=timezone.utc)


def _contract_history_upper_bound(contract: object, fallback_expiry: str) -> datetime:
    """Return an end cursor that IB will serve for a qualified dated contract."""
    last_trade = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "")
    if len(last_trade) >= 8 and last_trade[:8].isdigit():
        yyyymmdd = last_trade[:8]
        return datetime(
            int(yyyymmdd[:4]),
            int(yyyymmdd[4:6]),
            int(yyyymmdd[6:8]),
            tzinfo=timezone.utc,
        ) + timedelta(days=1)

    contract_eom = _expiry_first_of_month(fallback_expiry) + timedelta(days=32)
    return contract_eom.replace(day=1)  # first of *next* month


def _ensure_connection(client: IBHistoricalClient) -> None:
    if client.reconnect_if_needed():
        logger.warning("IB connection re-established")


def _call_with_reconnect(client: IBHistoricalClient, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ConnectionError, OSError) as exc:
        logger.warning("IB call %s failed (%s); reconnecting", fn.__name__, exc)
        client.reconnect_if_needed()
        return fn(*args, **kwargs)


def _fetch_chunk_with_retry(
    *,
    client: IBHistoricalClient,
    contract: object,
    end_cursor: datetime,
    duration: str,
    what_to_show: str,
    use_rth: bool,
    label: str,
):
    """Thin wrapper around :meth:`IBHistoricalClient.fetch_dated_minute_bars_with_retry`.

    Kept as a module-level helper so existing tests (which mock the
    walker's local fetch helper) continue to exercise the right code
    path; the refresh CLI calls the client method directly.
    """
    return client.fetch_dated_minute_bars_with_retry(
        contract,
        end_datetime=end_cursor,
        duration=duration,
        what_to_show=what_to_show,
        use_rth=use_rth,
        max_retries=_TIMEOUT_RETRY_ATTEMPTS,
        backoff_seconds=_TIMEOUT_BACKOFF_SECONDS,
        label=f"{label} end={end_cursor.isoformat()}",
    )


def _walk_dated_contract(
    *,
    spec: FuturesSpec,
    contract: object,
    client: IBHistoricalClient,
    lake: MinuteLake,
    args: argparse.Namespace,
) -> list[WalkFailure]:
    failures: list[WalkFailure] = []
    symbol = spec.label_symbol
    expiry = spec.expiry
    state = lake.get_state(symbol, expiry=expiry)
    earliest_ts = state.earliest_ts if state else None

    now = datetime.now(timezone.utc)
    upper_bound = min(_contract_history_upper_bound(contract, expiry), now)
    end_cursor = earliest_ts if earliest_ts is not None else upper_bound

    if end_cursor <= datetime(1970, 1, 2, tzinfo=timezone.utc):
        return failures

    duration_str = f"{args.chunk_days} D"
    cap = args.max_chunks_per_contract or 0
    written_total = 0
    fetched_total = 0
    chunk_idx = 0

    logger.info(
        "%s: walking back from %s in %s chunks",
        spec.label,
        end_cursor.isoformat(),
        duration_str,
    )

    while True:
        chunk_idx += 1
        if cap and chunk_idx > cap:
            logger.info("%s: hit max-chunks-per-contract=%d", spec.label, cap)
            break
        try:
            bars = _fetch_chunk_with_retry(
                client=client,
                contract=contract,
                end_cursor=end_cursor,
                duration=duration_str,
                what_to_show=args.what_to_show,
                use_rth=args.use_rth,
                label=spec.label,
            )
        except HistoricalDataTimeout as exc:
            logger.warning(
                "%s: persistent timeout at end=%s; NOT treating as head, bailing out",
                spec.label,
                end_cursor.isoformat(),
            )
            failures.append(
                WalkFailure(
                    alias=spec.label, expiry=expiry, stage="timeout", error=str(exc)
                )
            )
            if written_total == 0:
                lake.record_notes(symbol, "timeout", expiry=expiry)
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s chunk end=%s failed: %s", spec.label, end_cursor.isoformat(), exc
            )
            failures.append(
                WalkFailure(
                    alias=spec.label, expiry=expiry, stage="fetch", error=str(exc)
                )
            )
            break

        if not bars:
            logger.info(
                "%s: IB returned 0 bars at end=%s (reached data head)",
                spec.label,
                end_cursor.isoformat(),
            )
            if written_total == 0:
                lake.record_notes(symbol, "fetch_empty", expiry=expiry)
            break

        written = lake.upsert_bars(symbol, bars, expiry=expiry)
        written_total += written
        fetched_total += len(bars)

        new_state = lake.get_state(symbol, expiry=expiry)
        new_earliest = new_state.earliest_ts if new_state else None
        if new_earliest is None or new_earliest >= end_cursor:
            logger.info(
                "%s: earliest didn't advance past %s; stopping",
                spec.label,
                end_cursor.isoformat(),
            )
            break

        # Step the cursor backward to just before the new earliest.
        end_cursor = new_earliest - timedelta(seconds=1)
        if args.pace_seconds > 0:
            time.sleep(args.pace_seconds)

    if written_total > 0:
        lake.record_notes(symbol, "", expiry=expiry)
        new_state = lake.get_state(symbol, expiry=expiry)
        earliest = new_state.earliest_ts.isoformat() if new_state and new_state.earliest_ts else "-"
        latest = new_state.latest_ts.isoformat() if new_state and new_state.latest_ts else "-"
        logger.info(
            "%s: %d chunks, fetched=%d written=%d earliest=%s latest=%s",
            spec.label,
            chunk_idx - 1 if chunk_idx > 0 else 0,
            fetched_total,
            written_total,
            earliest,
            latest,
        )
    return failures


def _walk_symbol(
    *,
    alias: str,
    client: IBHistoricalClient,
    lake: MinuteLake,
    args: argparse.Namespace,
) -> list[WalkFailure]:
    failures: list[WalkFailure] = []
    base_spec = resolve_futures_spec(alias=alias)
    _ensure_connection(client)

    min_expiry = _floor_min_expiry(args)
    today = datetime.now(timezone.utc)
    # Default ceiling: 24 months out so we capture currently-listed contracts.
    max_expiry = args.max_expiry or f"{today.year + 2:04d}{today.month:02d}"
    candidates = enumerate_dated_specs(
        base_spec, min_expiry=min_expiry, max_expiry=max_expiry
    )
    if not candidates:
        logger.warning(
            "%s: no candidate expiries in [%s, %s]", alias, min_expiry, max_expiry
        )
        return failures

    current_month = f"{today.year:04d}{today.month:02d}"
    if args.skip_active:
        candidates = [s for s in candidates if s.expiry < current_month]

    ordered = sorted(candidates, key=lambda s: s.expiry, reverse=not args.oldest_first)
    if args.max_expiries_per_symbol > 0:
        ordered = ordered[: args.max_expiries_per_symbol]

    logger.info(
        "%s: %d candidate dated contracts (%s..%s, %s first); batch-qualifying...",
        alias,
        len(ordered),
        ordered[0].expiry if ordered else "-",
        ordered[-1].expiry if ordered else "-",
        "oldest" if args.oldest_first else "newest",
    )

    try:
        qualified_pairs = _call_with_reconnect(
            client, client.qualify_dated_futures, ordered
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: batch qualify failed: %s", alias, exc)
        return [WalkFailure(alias=alias, expiry="-", stage="qualify", error=str(exc))]

    listed = [(s, c) for s, c in qualified_pairs if c is not None]
    if not listed:
        logger.warning(
            "%s: no candidates qualified as listed contracts in IB", alias
        )
        return failures

    logger.info("%s: %d/%d candidates qualified; walking each", alias, len(listed), len(ordered))

    for idx, (spec, contract) in enumerate(listed, start=1):
        logger.info("[%d/%d] %s", idx, len(listed), spec.label)
        failures.extend(
            _walk_dated_contract(
                spec=spec, contract=contract, client=client, lake=lake, args=args
            )
        )
        if args.pace_seconds > 0 and idx < len(listed):
            time.sleep(args.pace_seconds)

    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    aliases = _resolve_aliases(args)
    logger.info(
        "Walking %d aliases into %s (years=%d, min_expiry=%s)",
        len(aliases),
        args.lake_path,
        args.years,
        _floor_min_expiry(args),
    )

    lake = MinuteLake(args.lake_path)
    lake.connect()
    client = IBHistoricalClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        market_data_type=args.market_data_type,
    )
    failures: list[WalkFailure] = []
    try:
        client.connect()
        try:
            for idx, alias in enumerate(aliases, start=1):
                logger.info("===== [%d/%d] %s =====", idx, len(aliases), alias)
                failures.extend(
                    _walk_symbol(alias=alias, client=client, lake=lake, args=args)
                )
                if args.pace_seconds > 0 and idx < len(aliases):
                    time.sleep(args.pace_seconds)
        finally:
            client.disconnect()
    finally:
        lake.close()

    failed_aliases = {f.alias for f in failures}
    succeeded = len(aliases) - len({f.alias.split()[0] for f in failures if f.stage == "discover"})
    logger.info(
        "Dated walk done: %d aliases attempted, %d failures (across %d aliases)",
        len(aliases),
        len(failures),
        len(failed_aliases),
    )
    if failures:
        for f in failures:
            logger.warning("  %s [%s/%s]: %s", f.alias, f.expiry, f.stage, f.error)
    return 0 if succeeded > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
