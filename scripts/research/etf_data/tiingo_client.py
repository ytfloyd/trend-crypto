"""
Tiingo REST API wrapper for ETF daily data.

Handles:
- API key management (env var ``TIINGO_API_KEY``)
- Rate limiting (respects Tiingo free-tier: 50 requests/hour, 500/day)
- Retry with exponential backoff
- Converts responses to pandas DataFrames with standardised column names

Usage::

    from scripts.research.etf_data.tiingo_client import TiingoDaily

    client = TiingoDaily()                           # reads TIINGO_API_KEY
    df = client.fetch("SPY", "2006-01-01", "2026-01-01")
    # → DataFrame(symbol, ts, open, high, low, close, volume,
    #             adj_open, adj_high, adj_low, adj_close, adj_volume,
    #             dividend, split_factor)
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import requests


# Tiingo free-tier limits
_MAX_REQUESTS_PER_HOUR = 50
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = 2.0  # seconds, doubles each retry


class TiingoDaily:
    """Thin wrapper around the Tiingo daily price endpoint."""

    BASE_URL = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Tiingo API key required.  Set TIINGO_API_KEY env var or pass api_key=."
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        })
        self._request_count = 0
        self._hour_start = time.monotonic()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        """Sleep if approaching the hourly request cap."""
        elapsed = time.monotonic() - self._hour_start
        if elapsed > 3600:
            self._request_count = 0
            self._hour_start = time.monotonic()
        if self._request_count >= _MAX_REQUESTS_PER_HOUR:
            wait = 3600 - elapsed + 1
            print(f"[tiingo] Rate limit reached, sleeping {wait:.0f}s ...")
            time.sleep(wait)
            self._request_count = 0
            self._hour_start = time.monotonic()

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------
    def fetch(
        self,
        ticker: str,
        start: str = "2005-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily adjusted OHLCV for a single ticker.

        Parameters
        ----------
        ticker : str
            ETF ticker symbol (e.g. ``"SPY"``).
        start : str
            Start date (ISO-8601).
        end : str | None
            End date.  Defaults to today.

        Returns
        -------
        pd.DataFrame
            Columns: symbol, ts, open, high, low, close, volume,
            adj_open, adj_high, adj_low, adj_close, adj_volume,
            dividend, split_factor.
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/{ticker}/prices"
        params = {
            "startDate": start,
            "endDate": end,
            "format": "json",
        }

        self._rate_limit()
        data = self._get_with_retry(url, params)
        self._request_count += 1

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = self._normalize(df, ticker)
        return df

    def fetch_meta(self, ticker: str) -> dict:
        """Fetch metadata (description, exchange, start/end dates) for a ticker."""
        url = f"{self.BASE_URL}/{ticker}"
        self._rate_limit()
        data = self._get_with_retry(url, {})
        self._request_count += 1
        return data if data else {}

    # ------------------------------------------------------------------
    # HTTP with retry
    # ------------------------------------------------------------------
    def _get_with_retry(self, url: str, params: dict) -> list | dict:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = self._session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    wait = _RETRY_BACKOFF * (2 ** attempt)
                    print(f"[tiingo] 429 rate-limited, retrying in {wait:.0f}s ...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    print(f"[tiingo] 404 for {url} — ticker not found")
                    return []
                print(f"[tiingo] HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_BACKOFF * (2 ** attempt))
            except requests.RequestException as e:
                print(f"[tiingo] Request error: {e}")
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_BACKOFF * (2 ** attempt))
        return []

    # ------------------------------------------------------------------
    # Normalise Tiingo response → standard DataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        rename = {
            "date": "ts",
            "adjOpen": "adj_open",
            "adjHigh": "adj_high",
            "adjLow": "adj_low",
            "adjClose": "adj_close",
            "adjVolume": "adj_volume",
            "divCash": "dividend",
            "splitFactor": "split_factor",
        }
        df = df.rename(columns=rename)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        df["symbol"] = ticker

        keep = [
            "symbol", "ts", "open", "high", "low", "close", "volume",
            "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
            "dividend", "split_factor",
        ]
        keep = [c for c in keep if c in df.columns]
        return df[keep].sort_values("ts").reset_index(drop=True)
