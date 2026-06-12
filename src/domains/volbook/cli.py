"""CLI for the volbook IBKR OHLCV tool.

Typical run (CL Jun'26, daily, 1 year)::

    python -m scripts.volbook.fetch_futures_ohlcv

Append an hourly view without losing the daily::

    python -m scripts.volbook.fetch_futures_ohlcv \\
        --alias CL_JUN26 --bar-size "1 hour" --duration "30 D" --append
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from common.logging import get_logger

from .bundle import OhlcvBundle
from .canvas_writer import default_canvas_path, write_canvas
from .contracts import FuturesSpec, resolve_futures_spec
from .html_writer import DEFAULT_HTML_PATH, write_html

logger = get_logger("volbook.cli")

DEFAULT_BUNDLE_PATH = Path("data/volbook/bundle.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volbook.fetch_futures_ohlcv",
        description="Fetch historical OHLCV from IB for a futures contract and "
        "regenerate the volbook dashboards.",
    )
    p.add_argument(
        "--alias",
        default=None,
        help="Named alias from contracts.KNOWN_FUTURES (e.g. CL_JUN26).",
    )
    p.add_argument("--symbol", default="CL", help="Futures root symbol (default: CL).")
    p.add_argument(
        "--expiry",
        default="202606",
        help="IB lastTradeDateOrContractMonth, YYYYMM or YYYYMMDD (default: 202606).",
    )
    p.add_argument(
        "--exchange",
        default=None,
        help="Exchange. Defaulted per symbol when omitted (CL→NYMEX, ES→CME, …).",
    )
    p.add_argument("--currency", default="USD")
    p.add_argument(
        "--bar-size",
        default="1 day",
        help="IB bar size string, e.g. '1 day', '1 hour', '5 mins' (default: '1 day').",
    )
    p.add_argument(
        "--duration",
        default="1 Y",
        help="IB duration string, e.g. '1 Y', '6 M', '30 D' (default: '1 Y').",
    )
    p.add_argument(
        "--what-to-show",
        default="TRADES",
        help="IB whatToShow: TRADES, MIDPOINT, BID_ASK, … (default: TRADES).",
    )
    p.add_argument(
        "--use-rth",
        action="store_true",
        help="Restrict to regular trading hours.",
    )
    p.add_argument(
        "--end-datetime",
        default="",
        help="IB endDateTime (YYYYMMDD HH:MM:SS, UTC). Blank = now.",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Upsert into the existing bundle instead of starting fresh.",
    )
    p.add_argument(
        "--bundle-path",
        default=str(DEFAULT_BUNDLE_PATH),
        help=f"Bundle JSON destination (default: {DEFAULT_BUNDLE_PATH}).",
    )
    p.add_argument(
        "--canvas-path",
        default=None,
        help="Override the generated canvas path. Defaults to the workspace "
        "canvases dir under ~/.cursor/projects/.",
    )
    p.add_argument(
        "--no-canvas",
        action="store_true",
        help="Skip regenerating the canvas (bundle JSON only).",
    )
    p.add_argument(
        "--html-path",
        default=str(DEFAULT_HTML_PATH),
        help=f"Standalone HTML dashboard destination (default: {DEFAULT_HTML_PATH}).",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip regenerating the standalone HTML dashboard.",
    )
    p.add_argument(
        "--no-indicators",
        action="store_true",
        help="Skip TA-Lib indicator computation (keeps bundle lean).",
    )
    p.add_argument(
        "--indicator-tail",
        type=int,
        default=20,
        help="Bars of indicator history to persist per output (default: 20).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497, help="TWS=7497 paper / 7496 live.")
    p.add_argument("--client-id", type=int, default=17)
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


def _fetch_series(args: argparse.Namespace, spec: FuturesSpec):
    """Open an IB connection, pull one series, and disconnect."""
    # Local import so ``--help`` works without ib_insync installed.
    from .ibkr_client import IBHistoricalClient

    client = IBHistoricalClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        market_data_type=args.market_data_type,
    )
    with client:
        return client.fetch_futures_ohlcv(
            spec,
            bar_size=args.bar_size,
            duration=args.duration,
            what_to_show=args.what_to_show,
            use_rth=args.use_rth,
            end_datetime=args.end_datetime,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    spec = resolve_futures_spec(
        alias=args.alias,
        symbol=args.symbol,
        expiry=args.expiry,
        exchange=args.exchange,
        currency=args.currency,
    )
    logger.info(
        "Fetching %s (%s) %s bars, duration=%s, what=%s",
        spec.label, spec.exchange, args.bar_size, args.duration, args.what_to_show,
    )

    bundle_path = Path(args.bundle_path)
    bundle = OhlcvBundle.load(bundle_path) if args.append else OhlcvBundle()

    series = _fetch_series(args, spec)

    if not args.no_indicators:
        from .indicators import TALIB_AVAILABLE, compute_all_indicators

        if TALIB_AVAILABLE:
            logger.info("Computing TA-Lib indicators (tail=%d)", args.indicator_tail)
            series.indicators = compute_all_indicators(
                series.bars, tail=args.indicator_tail
            )
            n_rows = sum(len(v) for v in series.indicators.values())
            logger.info(
                "Attached %d indicators across %d categories",
                n_rows,
                len(series.indicators),
            )
        else:
            logger.warning(
                "TA-Lib not installed; skipping indicators. "
                "Install with `pip install -e .[ta]`."
            )

        # Setups only make sense with TA-Lib; keeps the import guarded.
        from .signals import TALIB_AVAILABLE as SIG_TALIB, build_setups

        if SIG_TALIB:
            series.setups = build_setups(series.bars)
            logger.info(
                "Found %d actionable risk:reward setups", len(series.setups)
            )

    bundle.upsert(series)
    saved = bundle.save(bundle_path)
    logger.info("Wrote bundle → %s (%d series)", saved, len(bundle.series))

    if args.no_canvas:
        logger.info("Skipping canvas regeneration (--no-canvas).")
    else:
        canvas_path = Path(args.canvas_path) if args.canvas_path else default_canvas_path()
        written = write_canvas(bundle, canvas_path)
        logger.info("Wrote canvas → %s", written)

    if args.no_html:
        logger.info("Skipping HTML dashboard regeneration (--no-html).")
    else:
        html_path = write_html(bundle, Path(args.html_path))
        logger.info("Wrote HTML dashboard → %s", html_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
