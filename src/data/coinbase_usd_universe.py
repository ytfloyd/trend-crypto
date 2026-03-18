from __future__ import annotations

import json
import time
from typing import List, Set

import requests


BASE_URL = "https://api.coinbase.com"
EXCHANGE_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"

STABLE_BASES: Set[str] = {
    "USDC",
    "USDT",
    "DAI",
    "PAX",
    "TUSD",
    "GUSD",
    "BUSD",
    "USDP",
    "PYUSD",
    "FDUSD",
    "USDS",
}


def _request_with_retry(
    session: requests.Session,
    url: str,
    params: dict,
    *,
    timeout: float,
    max_retries: int,
) -> requests.Response:
    for attempt in range(max_retries + 1):
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(float(retry_after))
                except ValueError:
                    pass
            else:
                backoff = min(60, 2**attempt)
                time.sleep(backoff)
            continue
        resp.raise_for_status()
    resp.raise_for_status()
    return resp


def fetch_products(session: requests.Session, timeout: float = 10.0, max_retries: int = 3):
    """
    Fetch the full Coinbase Exchange product list via the public market data API.

    Returns:
        - Typically: list[dict], one per product (Exchange /products JSON)
        - Fallback: whatever _request_with_retry returns if it's already decoded
    """
    resp = _request_with_retry(
        session,
        EXCHANGE_PRODUCTS_URL,
        params=None,
        timeout=timeout,
        max_retries=max_retries,
    )

    if isinstance(resp, requests.Response):
        return resp.json()

    return resp


def get_spot_universe(
    session: requests.Session,
    quote_currencies: Set[str] | None = None,
    exclude_stable_bases: bool = False,
) -> List[str]:
    """
    Return a sorted list of spot product symbols from Coinbase Exchange.

    Args:
        session: requests session.
        quote_currencies: If provided, only return products with these quote
            currencies (e.g. ``{"USD", "BTC"}``).  ``None`` returns all.
        exclude_stable_bases: If True, exclude products whose base currency
            is a stablecoin (e.g. USDC-BTC).

    Handles both Exchange (list[dict]) and Brokerage-style ({"products": [...]})
    payloads, and decodes bytes/str JSON from fetch_products().
    """
    raw = fetch_products(session)

    if isinstance(raw, (bytes, str)):
        try:
            raw = json.loads(raw)
        except Exception as e:
            raise ValueError(f"Unexpected products payload (cannot JSON-decode): {type(raw)}") from e

    if isinstance(raw, dict):
        products = raw.get("products", [])
    elif isinstance(raw, list):
        products = raw
    else:
        raise ValueError(f"Unexpected products payload type: {type(raw)}")

    symbols: List[str] = []

    for p in products:
        if isinstance(p, (bytes, str)):
            try:
                p = json.loads(p)
            except Exception:
                continue

        if not isinstance(p, dict):
            continue

        pid = p.get("id")
        base = p.get("base_currency")
        quote = p.get("quote_currency")

        if not pid or not base or not quote:
            continue

        if quote_currencies is not None and quote not in quote_currencies:
            continue

        if exclude_stable_bases and base.upper() in STABLE_BASES:
            continue

        symbols.append(pid)

    return sorted(set(symbols))


def get_usd_spot_universe_ex_stables(session: requests.Session) -> List[str]:
    """Backward-compatible wrapper: USD products excluding stablecoin bases."""
    return get_spot_universe(session, quote_currencies={"USD"}, exclude_stable_bases=True)

