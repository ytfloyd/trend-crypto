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


def get_usd_spot_universe_ex_stables(session: requests.Session) -> List[str]:
    """
    Return a sorted list of USD spot product symbols (e.g. 'SOL-USD')
    from Coinbase Exchange, excluding stablecoin bases.

    Handles both:
    - Exchange: list[dict]
    - Brokerage-style: {"products": [...]}
    and also decodes bytes/str JSON payloads from fetch_products().
    """
    raw = fetch_products(session)

    # If fetch_products returns bytes or str, JSON-decode it.
    if isinstance(raw, (bytes, str)):
        try:
            raw = json.loads(raw)
        except Exception as e:
            raise ValueError(f"Unexpected products payload (cannot JSON-decode): {type(raw)}") from e

    # Normalize into a list of product dicts.
    if isinstance(raw, dict):
        # Advanced Trade style: {"products": [...]}
        products = raw.get("products", [])
    elif isinstance(raw, list):
        # Exchange style: list[product_dict]
        products = raw
    else:
        raise ValueError(f"Unexpected products payload type: {type(raw)}")

    symbols: List[str] = []

    for p in products:
        # In case individual entries are serialized as bytes/str JSON, decode them too.
        if isinstance(p, (bytes, str)):
            try:
                p = json.loads(p)
            except Exception:
                continue  # skip malformed item

        if not isinstance(p, dict):
            continue

        pid = p.get("id")  # e.g. "SOL-USD"
        base = p.get("base_currency")  # e.g. "SOL"
        quote = p.get("quote_currency")  # e.g. "USD"

        if not pid or not base or not quote:
            continue

        # USD spot only
        if quote != "USD":
            continue

        # Exclude stablecoin bases
        if base.upper() in STABLE_BASES:
            continue

        symbols.append(pid)

    return sorted(set(symbols))

