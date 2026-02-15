"""Tests for the Coinbase data collector.

Tests use a temporary DuckDB database and mock the Coinbase API to avoid
real network calls. Run with: python -m pytest tests/test_collector.py -v
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import duckdb
import polars as pl
import pytest

from data.collector import CoinbaseCollector, GRANULARITY_1M


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Create a temporary DuckDB path."""
    return str(tmp_path / "test_market.duckdb")


@pytest.fixture
def collector(tmp_db: str) -> CoinbaseCollector:
    """Create a collector with a temp DB (no API client needed for schema tests)."""
    return CoinbaseCollector(
        db_path=tmp_db,
        api_key="test_key",
        api_secret="test_secret",
        max_rps=100.0,  # No real rate limiting in tests
    )


def _make_fake_candle(ts: datetime, price: float = 100.0, volume: float = 1000.0) -> Any:
    """Create a mock Candle object matching SDK response format."""
    candle = MagicMock()
    candle.start = str(int(ts.timestamp()))
    candle.open = str(price)
    candle.high = str(price * 1.01)
    candle.low = str(price * 0.99)
    candle.close = str(price * 1.005)
    candle.volume = str(volume)
    return candle


def _make_fake_product(product_id: str, base: str, quote: str) -> Any:
    """Create a mock Product object matching SDK response format."""
    product = MagicMock()
    product.product_id = product_id
    product.base_currency_id = base
    product.quote_currency_id = quote
    return product


# ─── Schema Tests ───────────────────────────────────────────────────

class TestSchema:
    def test_ensure_schema_creates_table(self, collector: CoinbaseCollector) -> None:
        """candles_1m table should exist after initialization."""
        result = collector._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'candles_1m'"
        ).fetchone()
        assert result is not None
        assert result[0] == "candles_1m"

    def test_ensure_schema_creates_views(self, collector: CoinbaseCollector) -> None:
        """Resampled views should exist after initialization."""
        views = collector._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
        ).fetchall()
        view_names = {v[0] for v in views}
        assert "bars_1h" in view_names
        assert "bars_4h" in view_names
        assert "bars_1d" in view_names

    def test_candles_1m_columns(self, collector: CoinbaseCollector) -> None:
        """candles_1m should have the expected OHLCV columns."""
        cols = collector._conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'candles_1m'"
        ).fetchall()
        col_names = {c[0] for c in cols}
        expected = {"symbol", "ts", "open", "high", "low", "close", "volume"}
        assert expected == col_names


# ─── Upsert Tests ───────────────────────────────────────────────────

class TestUpsert:
    def test_upsert_inserts_rows(self, collector: CoinbaseCollector) -> None:
        """Inserting candles should add rows to the table."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = pl.DataFrame([
            {"ts": base_ts, "symbol": "BTC-USD", "open": 42000.0, "high": 42100.0,
             "low": 41900.0, "close": 42050.0, "volume": 100.0},
            {"ts": base_ts + timedelta(minutes=1), "symbol": "BTC-USD", "open": 42050.0,
             "high": 42150.0, "low": 41950.0, "close": 42100.0, "volume": 150.0},
        ])

        collector._upsert_candles(df)

        count = collector._conn.execute(
            "SELECT COUNT(*) FROM candles_1m WHERE symbol = 'BTC-USD'"
        ).fetchone()
        assert count is not None
        assert count[0] == 2

    def test_upsert_deduplicates(self, collector: CoinbaseCollector) -> None:
        """Re-inserting the same window should not create duplicates."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = pl.DataFrame([
            {"ts": base_ts, "symbol": "BTC-USD", "open": 42000.0, "high": 42100.0,
             "low": 41900.0, "close": 42050.0, "volume": 100.0},
        ])

        collector._upsert_candles(df)
        collector._upsert_candles(df)  # Insert same data again

        count = collector._conn.execute(
            "SELECT COUNT(*) FROM candles_1m WHERE symbol = 'BTC-USD'"
        ).fetchone()
        assert count is not None
        assert count[0] == 1  # Should still be 1, not 2

    def test_upsert_updates_values(self, collector: CoinbaseCollector) -> None:
        """Re-inserting with different values should update."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        df1 = pl.DataFrame([
            {"ts": base_ts, "symbol": "BTC-USD", "open": 42000.0, "high": 42100.0,
             "low": 41900.0, "close": 42050.0, "volume": 100.0},
        ])
        collector._upsert_candles(df1)

        df2 = pl.DataFrame([
            {"ts": base_ts, "symbol": "BTC-USD", "open": 43000.0, "high": 43100.0,
             "low": 42900.0, "close": 43050.0, "volume": 200.0},
        ])
        collector._upsert_candles(df2)

        row = collector._conn.execute(
            "SELECT open, volume FROM candles_1m WHERE symbol = 'BTC-USD'"
        ).fetchone()
        assert row is not None
        assert row[0] == 43000.0  # Updated value
        assert row[1] == 200.0


# ─── Resume Tests ───────────────────────────────────────────────────

class TestResume:
    def test_get_last_timestamp(self, collector: CoinbaseCollector) -> None:
        """Should return the max timestamp for a symbol."""
        ts1 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc)
        df = pl.DataFrame([
            {"ts": ts1, "symbol": "BTC-USD", "open": 42000.0, "high": 42100.0,
             "low": 41900.0, "close": 42050.0, "volume": 100.0},
            {"ts": ts2, "symbol": "BTC-USD", "open": 42050.0, "high": 42150.0,
             "low": 41950.0, "close": 42100.0, "volume": 150.0},
        ])
        collector._upsert_candles(df)

        last = collector._get_last_timestamp("BTC-USD")
        assert last is not None
        # Compare UTC timestamps (DuckDB may return local TZ, we normalize to UTC)
        last_utc = last.astimezone(timezone.utc)
        assert last_utc == ts2

    def test_get_last_timestamp_empty(self, collector: CoinbaseCollector) -> None:
        """Should return None if no data for symbol."""
        assert collector._get_last_timestamp("NONEXIST-USD") is None

    def test_get_symbols_in_db(self, collector: CoinbaseCollector) -> None:
        """Should return all distinct symbols."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for sym in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            df = pl.DataFrame([
                {"ts": ts, "symbol": sym, "open": 100.0, "high": 101.0,
                 "low": 99.0, "close": 100.5, "volume": 50.0},
            ])
            collector._upsert_candles(df)

        symbols = collector._get_symbols_in_db()
        assert symbols == ["BTC-USD", "ETH-USD", "SOL-USD"]


# ─── Resample Tests ─────────────────────────────────────────────────

class TestResample:
    def test_bars_1h_view(self, collector: CoinbaseCollector) -> None:
        """1-hour view should aggregate 60 1-minute candles correctly."""
        base_ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(60):
            ts = base_ts + timedelta(minutes=i)
            price = 42000.0 + i * 10  # Increasing price
            rows.append({
                "ts": ts, "symbol": "BTC-USD",
                "open": price, "high": price + 5.0,
                "low": price - 5.0, "close": price + 2.0,
                "volume": 10.0,
            })
        df = pl.DataFrame(rows)
        collector._upsert_candles(df)

        result = collector._conn.execute(
            "SELECT * FROM bars_1h WHERE symbol = 'BTC-USD'"
        ).fetchall()

        assert len(result) == 1
        row = result[0]
        symbol, ts, open_p, high_p, low_p, close_p, volume = row

        assert symbol == "BTC-USD"
        assert open_p == 42000.0      # First candle's open
        assert high_p == 42595.0      # Max of all highs (42590 + 5)
        assert low_p == 41995.0       # Min of all lows (42000 - 5)
        assert close_p == 42592.0     # Last candle's close (42590 + 2)
        assert volume == 600.0        # Sum of 60 * 10

    def test_refresh_clean_tables(self, collector: CoinbaseCollector) -> None:
        """refresh_clean_tables should create materialized tables."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = pl.DataFrame([
            {"ts": ts, "symbol": "BTC-USD", "open": 42000.0, "high": 42100.0,
             "low": 41900.0, "close": 42050.0, "volume": 100.0},
        ])
        collector._upsert_candles(df)
        collector.refresh_clean_tables()

        for table in ["bars_1h_clean", "bars_4h_clean", "bars_1d_clean"]:
            result = collector._conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()
            assert result is not None
            assert result[0] >= 1


# ─── API Mock Tests ─────────────────────────────────────────────────

class TestAPIParsing:
    def test_fetch_candles_parsing(self, collector: CoinbaseCollector) -> None:
        """Should correctly parse SDK candle response into Polars DataFrame."""
        ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

        mock_response = MagicMock()
        mock_response.candles = [
            _make_fake_candle(ts, price=42000.0, volume=500.0),
            _make_fake_candle(ts + timedelta(minutes=1), price=42100.0, volume=600.0),
        ]

        with patch.object(collector, "_get_client") as mock_client:
            mock_client.return_value.get_candles.return_value = mock_response

            df = collector._fetch_candles(
                "BTC-USD",
                ts - timedelta(minutes=1),
                ts + timedelta(minutes=5),
                GRANULARITY_1M,
            )

        assert df.height == 2
        assert set(df.columns) == {"ts", "symbol", "open", "high", "low", "close", "volume"}
        assert df["symbol"][0] == "BTC-USD"
        assert df["open"][0] == 42000.0

    def test_fetch_candles_empty(self, collector: CoinbaseCollector) -> None:
        """Should return empty DataFrame when API returns no candles."""
        mock_response = MagicMock()
        mock_response.candles = []

        with patch.object(collector, "_get_client") as mock_client:
            mock_client.return_value.get_candles.return_value = mock_response

            df = collector._fetch_candles(
                "BTC-USD",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc),
                GRANULARITY_1M,
            )

        assert df.is_empty()

    def test_discover_products_filters_stables(self, collector: CoinbaseCollector) -> None:
        """Should exclude stablecoins from product list."""
        mock_response = MagicMock()
        mock_response.products = [
            _make_fake_product("BTC-USD", "BTC", "USD"),
            _make_fake_product("ETH-USD", "ETH", "USD"),
            _make_fake_product("USDC-USD", "USDC", "USD"),  # Stablecoin
            _make_fake_product("USDT-USD", "USDT", "USD"),  # Stablecoin
            _make_fake_product("BTC-EUR", "BTC", "EUR"),    # Not USD
        ]

        with patch.object(collector, "_get_client") as mock_client:
            mock_client.return_value.get_products.return_value = mock_response

            symbols = collector.discover_products()

        assert "BTC-USD" in symbols
        assert "ETH-USD" in symbols
        assert "USDC-USD" not in symbols
        assert "USDT-USD" not in symbols
        assert "BTC-EUR" not in symbols


# ─── Status Tests ───────────────────────────────────────────────────

class TestStatus:
    def test_status_empty(self, collector: CoinbaseCollector) -> None:
        """Should return empty list for empty DB."""
        assert collector.status() == []

    def test_status_with_data(self, collector: CoinbaseCollector) -> None:
        """Should return correct summary for each symbol."""
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc)

        for ts in [ts1, ts2]:
            df = pl.DataFrame([
                {"ts": ts, "symbol": "BTC-USD", "open": 42000.0, "high": 42100.0,
                 "low": 41900.0, "close": 42050.0, "volume": 100.0},
            ])
            collector._upsert_candles(df)

        rows = collector.status()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "BTC-USD"
        assert rows[0]["row_count"] == 2
