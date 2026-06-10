"""Volatility book trading tools.

This package houses the trading-desk tooling that sits alongside the vol
research code. The first tool connects to Interactive Brokers, pulls
historical OHLCV for a user-selected futures contract, and regenerates
an interactive Cursor canvas so the desk can eyeball the tape.

Entry points
------------
``python -m scripts.volbook.fetch_futures_ohlcv``
    Default run: pull CL Jun'26 daily bars for the last year, upsert into
    ``data/volbook/bundle.json``, regenerate the canvas.

Python API
----------
``from volbook.bundle import OhlcvBundle, OhlcvSeries``
``from volbook.ibkr_client import IBHistoricalClient``
``from volbook.canvas_writer import write_canvas``
"""
from __future__ import annotations

from .bundle import Bar, OhlcvBundle, OhlcvSeries
from .continuous import (
    NymexObservedHolidayCalendar,
    RollPolicy,
    WeekendHolidayCalendar,
    calendar_from_name,
    cl_last_trade_date,
    construct_continuous_series,
)
from .contracts import CORE_MACRO_ALIASES, FuturesSpec, resolve_futures_spec

__all__ = [
    "Bar",
    "OhlcvBundle",
    "OhlcvSeries",
    "RollPolicy",
    "WeekendHolidayCalendar",
    "NymexObservedHolidayCalendar",
    "calendar_from_name",
    "cl_last_trade_date",
    "construct_continuous_series",
    "CORE_MACRO_ALIASES",
    "FuturesSpec",
    "resolve_futures_spec",
]
