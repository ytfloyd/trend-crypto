#!/usr/bin/env python3
"""Refresh the volbook options-underlying futures universe.

Fetches daily 1Y and hourly 30D bars for the configured futures universe,
computes TA-Lib indicators + technical risk/reward setups, and rewrites
the bundle plus dashboards in one pass.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.logging import get_logger
from volbook.bundle import OhlcvBundle, OhlcvSeries
from volbook.canvas_writer import default_canvas_path, write_canvas
from volbook.cli import DEFAULT_BUNDLE_PATH
from volbook.contracts import CORE_MACRO_ALIASES, OPTIONS_UNDERLYING_ALIASES, resolve_futures_spec
from volbook.html_writer import DEFAULT_HTML_PATH, write_html
from volbook.ibkr_client import IBHistoricalClient

logger = get_logger("volbook.refresh_core_futures")

DEFAULT_TIMEFRAMES: tuple[tuple[str, str], ...] = (
    ("1 day", "1 Y"),
    ("1 hour", "30 D"),
)
DAILY_ONLY_TIMEFRAMES: tuple[tuple[str, str], ...] = (("1 day", "1 Y"),)


@dataclass(frozen=True)
class FetchFailure:
    alias: str
    bar_size: str
    duration: str
    error: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volbook.refresh_core_futures",
        description="Refresh the option-underlying futures universe for volbook.",
    )
    p.add_argument(
        "--universe",
        choices=["options-underlyings", "core-macro"],
        default="options-underlyings",
        help="Alias universe to refresh when --aliases is not supplied.",
    )
    p.add_argument(
        "--aliases",
        nargs="+",
        default=None,
        help="Root aliases whose futures curves should be refreshed (default: selected --universe).",
    )
    p.add_argument(
        "--curve-points",
        type=int,
        default=5,
        help="Number of active futures expiries to fetch per alias/root. Use 0 for all active curve points (default: 5).",
    )
    p.add_argument(
        "--fixed-alias-expiry",
        action="store_true",
        help="Fetch only the alias expiry instead of discovering the first curve points.",
    )
    p.add_argument(
        "--continuous-only",
        action="store_true",
        help="Fetch only IB continuous/front futures for each alias instead of dated curve points.",
    )
    p.add_argument(
        "--hourly-curve-points",
        type=int,
        default=5,
        help="Number of front curve points to fetch hourly history for. Use 0 for all hourly points (default: 5). Daily history is still fetched for every selected curve point.",
    )
    p.add_argument(
        "--daily-only",
        action="store_true",
        help="Fetch only daily bars, skipping the hourly timeframe for faster refreshes.",
    )
    p.add_argument(
        "--bundle-path",
        default=str(DEFAULT_BUNDLE_PATH),
        help=f"Bundle JSON destination (default: {DEFAULT_BUNDLE_PATH}).",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Start from an empty bundle instead of upserting into the existing one.",
    )
    p.add_argument(
        "--html-path",
        default=str(DEFAULT_HTML_PATH),
        help=f"Standalone HTML dashboard destination (default: {DEFAULT_HTML_PATH}).",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip standalone HTML dashboard regeneration.",
    )
    p.add_argument(
        "--canvas-path",
        default=None,
        help="Override the generated canvas path. Defaults to Cursor's canvases dir.",
    )
    p.add_argument(
        "--no-canvas",
        action="store_true",
        help="Skip canvas regeneration. Recommended if only the browser dashboard is needed.",
    )
    p.add_argument(
        "--no-indicators",
        action="store_true",
        help="Skip TA-Lib indicators and risk/reward setups.",
    )
    p.add_argument(
        "--indicator-tail",
        type=int,
        default=20,
        help="Bars of indicator history to persist per output (default: 20).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--port",
        type=int,
        default=7496,
        help="TWS/Gateway socket port (default: 7496; paper is often 7497).",
    )
    p.add_argument("--client-id", type=int, default=27)
    p.add_argument(
        "--market-data-type",
        type=int,
        default=2,
        help="IB reqMarketDataType: 1=live, 2=frozen, 3=delayed, 4=delayed-frozen.",
    )
    p.add_argument(
        "--what-to-show",
        default="TRADES",
        help="IB whatToShow for all requests (default: TRADES).",
    )
    p.add_argument(
        "--use-rth",
        action="store_true",
        help="Restrict all requests to regular trading hours.",
    )
    p.add_argument(
        "--end-datetime",
        default="",
        help="IB endDateTime for all requests. Blank = now.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _attach_analytics(series: OhlcvSeries, indicator_tail: int) -> None:
    from volbook.indicators import TALIB_AVAILABLE, compute_all_indicators
    from volbook.signals import TALIB_AVAILABLE as SIGNALS_TALIB
    from volbook.signals import build_setups

    if not TALIB_AVAILABLE:
        logger.warning("TA-Lib not installed; skipping indicators for %s", series.key)
        return
    series.indicators = compute_all_indicators(series.bars, tail=indicator_tail)
    if SIGNALS_TALIB:
        series.setups = build_setups(series.bars)


def _curve_specs(
    client: IBHistoricalClient,
    base_spec,
    *,
    curve_points: int,
    fixed_alias_expiry: bool,
):
    if fixed_alias_expiry:
        return [base_spec]
    return client.discover_futures_curve(base_spec, limit=None if curve_points <= 0 else curve_points)


def _selected_timeframes(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    return DAILY_ONLY_TIMEFRAMES if args.daily_only else DEFAULT_TIMEFRAMES


def _attempted_request_count(
    args: argparse.Namespace,
    timeframes: tuple[tuple[str, str], ...],
) -> int:
    if args.continuous_only:
        return len(args.aliases) * len(timeframes)
    curve_count = 1 if args.fixed_alias_expiry else max(args.curve_points, 1)
    attempts_per_alias = 0
    for bar_size, _duration in timeframes:
        if bar_size == "1 hour" and args.hourly_curve_points > 0:
            attempts_per_alias += min(curve_count, args.hourly_curve_points)
        else:
            attempts_per_alias += curve_count
    return len(args.aliases) * attempts_per_alias


def _fetch_all(args: argparse.Namespace, bundle: OhlcvBundle) -> list[FetchFailure]:
    failures: list[FetchFailure] = []
    timeframes = _selected_timeframes(args)
    client = IBHistoricalClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        market_data_type=args.market_data_type,
    )
    client.connect()
    try:
        for raw_alias in args.aliases:
            alias = raw_alias.upper()
            base_spec = resolve_futures_spec(alias=alias)
            if args.continuous_only:
                for bar_size, duration in timeframes:
                    try:
                        logger.info(
                            "Fetching %s continuous %s duration=%s",
                            base_spec.label_symbol,
                            bar_size,
                            duration,
                        )
                        series = client.fetch_continuous_futures_ohlcv(
                            base_spec,
                            bar_size=bar_size,
                            duration=duration,
                            what_to_show=args.what_to_show,
                            use_rth=args.use_rth,
                            end_datetime=args.end_datetime,
                        )
                        if not args.no_indicators:
                            _attach_analytics(series, args.indicator_tail)
                        bundle.upsert(series)
                        logger.info(
                            "Upserted %s: %d bars, %d indicator groups, %d setups",
                            series.key,
                            len(series.bars),
                            len(series.indicators),
                            len(series.setups),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Failed %s continuous %s", alias, bar_size)
                        failures.append(
                            FetchFailure(
                                alias=alias,
                                bar_size=bar_size,
                                duration=duration,
                                error=str(exc),
                            )
                        )
                continue
            try:
                specs = _curve_specs(
                    client,
                    base_spec,
                    curve_points=args.curve_points,
                    fixed_alias_expiry=args.fixed_alias_expiry,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed curve discovery for %s", alias)
                failures.append(
                    FetchFailure(
                        alias=alias,
                        bar_size="curve",
                        duration=f"{args.curve_points} points",
                        error=str(exc),
                    )
                )
                continue

            for spec_idx, spec in enumerate(specs):
                for bar_size, duration in timeframes:
                    if (
                        bar_size == "1 hour"
                        and args.hourly_curve_points > 0
                        and spec_idx >= args.hourly_curve_points
                    ):
                        logger.info(
                            "Skipping %s %s beyond hourly front-%d cap",
                            spec.label,
                            bar_size,
                            args.hourly_curve_points,
                        )
                        continue
                    try:
                        logger.info(
                            "Fetching %s %s duration=%s",
                            spec.label,
                            bar_size,
                            duration,
                        )
                        series = client.fetch_futures_ohlcv(
                            spec,
                            bar_size=bar_size,
                            duration=duration,
                            what_to_show=args.what_to_show,
                            use_rth=args.use_rth,
                            end_datetime=args.end_datetime,
                        )
                        if not args.no_indicators:
                            _attach_analytics(series, args.indicator_tail)
                        bundle.upsert(series)
                        logger.info(
                            "Upserted %s: %d bars, %d indicator groups, %d setups",
                            series.key,
                            len(series.bars),
                            len(series.indicators),
                            len(series.setups),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Failed %s %s %s", alias, spec.label, bar_size)
                        failures.append(
                            FetchFailure(
                                alias=f"{alias}:{spec.expiry}",
                                bar_size=bar_size,
                                duration=duration,
                                error=str(exc),
                            )
                        )
    finally:
        client.disconnect()
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.aliases is None:
        args.aliases = list(
            CORE_MACRO_ALIASES
            if args.universe == "core-macro"
            else OPTIONS_UNDERLYING_ALIASES
        )
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    bundle_path = Path(args.bundle_path)
    bundle = OhlcvBundle() if args.replace else OhlcvBundle.load(bundle_path)
    failures = _fetch_all(args, bundle)
    saved = bundle.save(bundle_path)
    logger.info("Wrote bundle -> %s (%d series)", saved, len(bundle.series))

    if args.no_html:
        logger.info("Skipping HTML dashboard regeneration (--no-html).")
    else:
        html_path = write_html(bundle, Path(args.html_path))
        logger.info("Wrote HTML dashboard -> %s", html_path)

    if args.no_canvas:
        logger.info("Skipping canvas regeneration (--no-canvas).")
    else:
        canvas_path = Path(args.canvas_path) if args.canvas_path else default_canvas_path()
        written = write_canvas(bundle, canvas_path)
        logger.info("Wrote canvas -> %s", written)

    attempted = _attempted_request_count(args, _selected_timeframes(args))
    succeeded = attempted - len(failures)
    logger.info("Refresh complete: %d succeeded, %d failed", succeeded, len(failures))
    if failures:
        logger.warning("Failures:")
        for f in failures:
            logger.warning("  %s %s (%s): %s", f.alias, f.bar_size, f.duration, f.error)
    return 0 if succeeded > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
