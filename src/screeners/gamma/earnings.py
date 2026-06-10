"""Earnings calendar lookup for the gamma screener.

Uses Finnhub's free earnings-calendar endpoint when a FINNHUB_API_KEY
environment variable is present; otherwise returns an empty set (the
screener still runs, just without the earnings penalty).

Finnhub free tier: 60 calls/min, no cost. A single
``/calendar/earnings?from=...&to=...`` call returns the whole universe
over a date range, so one call per run is enough.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Optional

from common.logging import get_logger

logger = get_logger("gamma_screener_earnings")

FINNHUB_BASE = "https://finnhub.io/api/v1"


def fetch_earnings_in_window(
    symbols: list[str],
    as_of: Optional[date] = None,
    lookahead_days: int = 45,
    api_key: Optional[str] = None,
) -> set[str]:
    """Return the subset of ``symbols`` with earnings in the next N days.

    Returns an empty set (and logs a warning) if the API key is missing
    or the HTTP call fails. The screener treats an empty set as "no
    earnings known" and skips the penalty.
    """
    api_key = api_key or os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logger.info("FINNHUB_API_KEY not set; skipping earnings lookup")
        return set()

    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; skipping earnings lookup")
        return set()

    as_of = as_of or datetime.utcnow().date()
    end = as_of + timedelta(days=lookahead_days)
    url = f"{FINNHUB_BASE}/calendar/earnings"
    params = {
        "from": as_of.isoformat(),
        "to": end.isoformat(),
        "token": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Finnhub earnings fetch failed: %s", exc)
        return set()

    events = payload.get("earningsCalendar", []) or []
    symbol_set = set(symbols)
    hits: set[str] = set()
    for evt in events:
        sym = str(evt.get("symbol", "")).upper()
        if sym in symbol_set:
            hits.add(sym)
    logger.info(
        "Earnings in next %d days: %d of %d symbols",
        lookahead_days, len(hits), len(symbols),
    )
    return hits
