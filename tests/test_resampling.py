"""Tests for DataPortal resampling from native 1m bars."""
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("polars")

from datetime import datetime, timedelta, timezone
import tempfile
from pathlib import Path

import duckdb
import polars as pl

from common.config import DataConfig
from data.portal import DataPortal


def _make_1m_bars(n: int = 120, start: datetime | None = None) -> pl.DataFrame:
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = start + timedelta(minutes=i)
        rows.append(
            {
                "ts": ts,
                "symbol": "BTC-USD",
                "open": float(i),
                "high": float(i) + 0.5,
                "low": float(i) - 0.5,
                "close": float(i) + 0.2,
                "volume": 1.0,
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def temp_db_with_1m_bars():
    fd, db_path = tempfile.mkstemp(suffix=".duckdb")
    Path(db_path).unlink(missing_ok=True)

    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE bars_1m (
            ts TIMESTAMP,
            symbol VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)
    bars = _make_1m_bars(180)
    conn.register("bars", bars)
    conn.execute("INSERT INTO bars_1m SELECT * FROM bars")

    # Also create a 1h table for invalid-downsample tests
    conn.execute("""
        CREATE TABLE bars_1h (
            ts TIMESTAMP,
            symbol VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)
    # Aggregate to hourly bars for the same period
    conn.execute("""
        INSERT INTO bars_1h
        SELECT
            date_trunc('hour', ts) AS ts,
            symbol,
            arg_min(open, ts) AS open,
            max(high) AS high,
            min(low) AS low,
            arg_max(close, ts) AS close,
            sum(volume) AS volume
        FROM bars_1m
        GROUP BY 1,2
        ORDER BY ts
    """)
    conn.close()

    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_db_with_1m_bars_long():
    fd, db_path = tempfile.mkstemp(suffix=".duckdb")
    Path(db_path).unlink(missing_ok=True)

    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE bars_1m (
            ts TIMESTAMP,
            symbol VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)
    bars = _make_1m_bars(60 * 24 * 3)
    conn.register("bars", bars)
    conn.execute("INSERT INTO bars_1m SELECT * FROM bars")

    # Also create a 1h table for invalid-downsample tests
    conn.execute("""
        CREATE TABLE bars_1h (
            ts TIMESTAMP,
            symbol VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO bars_1h
        SELECT
            date_trunc('hour', ts) AS ts,
            symbol,
            arg_min(open, ts) AS open,
            max(high) AS high,
            min(low) AS low,
            arg_max(close, ts) AS close,
            sum(volume) AS volume
        FROM bars_1m
        GROUP BY 1,2
        ORDER BY ts
    """)
    conn.close()

    yield db_path
    Path(db_path).unlink(missing_ok=True)


def test_resample_1m_to_1h(temp_db_with_1m_bars):
    cfg = DataConfig(
        db_path=temp_db_with_1m_bars,
        table="bars_1m",
        symbol="BTC-USD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 1, 59, tzinfo=timezone.utc),
        timeframe="1h",
        native_timeframe="1m",
    )
    portal = DataPortal(cfg, strict_validation=True)
    bars = portal.load_bars()

    assert bars.height == 2

    # First hour bucket (0-59)
    row0 = bars.row(0)
    assert row0[2] == pytest.approx(0.0)      # open
    assert row0[3] == pytest.approx(59.5)     # high
    assert row0[4] == pytest.approx(-0.5)     # low
    assert row0[5] == pytest.approx(59.2)     # close
    assert row0[6] == pytest.approx(60.0)     # volume

    # Second hour bucket (60-119)
    row1 = bars.row(1)
    assert row1[2] == pytest.approx(60.0)
    assert row1[3] == pytest.approx(119.5)
    assert row1[4] == pytest.approx(59.5)
    assert row1[5] == pytest.approx(119.2)
    assert row1[6] == pytest.approx(60.0)


def test_resample_1m_to_1d(temp_db_with_1m_bars):
    cfg = DataConfig(
        db_path=temp_db_with_1m_bars,
        table="bars_1m",
        symbol="BTC-USD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 1, 59, tzinfo=timezone.utc),
        timeframe="1d",
        native_timeframe="1m",
        drop_incomplete_bars=False,
    )
    portal = DataPortal(cfg, strict_validation=True)
    bars = portal.load_bars()

    assert bars.height == 1
    row = bars.row(0)
    assert row[2] == pytest.approx(0.0)       # open
    assert row[3] == pytest.approx(119.5)     # high
    assert row[4] == pytest.approx(-0.5)      # low
    assert row[5] == pytest.approx(119.2)     # close
    assert row[6] == pytest.approx(120.0)     # volume


def test_bucket_alignment_hour_and_day(temp_db_with_1m_bars):
    cfg_h = DataConfig(
        db_path=temp_db_with_1m_bars,
        table="bars_1m",
        symbol="BTC-USD",
        start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 2, 59, tzinfo=timezone.utc),
        timeframe="1h",
        native_timeframe="1m",
        drop_incomplete_bars=False,
    )
    bars_h = DataPortal(cfg_h, strict_validation=True).load_bars()
    assert all(ts.minute == 0 and ts.second == 0 for ts in bars_h["ts"].to_list())

    cfg_d = DataConfig(
        db_path=temp_db_with_1m_bars,
        table="bars_1m",
        symbol="BTC-USD",
        start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 2, 59, tzinfo=timezone.utc),
        timeframe="1d",
        native_timeframe="1m",
        drop_incomplete_bars=False,
    )
    bars_d = DataPortal(cfg_d, strict_validation=True).load_bars()
    assert all(ts.hour == 0 and ts.minute == 0 and ts.second == 0 for ts in bars_d["ts"].to_list())


def test_partial_bucket_drop_policy(temp_db_with_1m_bars):
    cfg = DataConfig(
        db_path=temp_db_with_1m_bars,
        table="bars_1m",
        symbol="BTC-USD",
        start=datetime(2024, 1, 1, 0, 50, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 2, 19, tzinfo=timezone.utc),
        timeframe="1h",
        native_timeframe="1m",
        drop_incomplete_bars=True,
        min_bucket_coverage_frac=0.8,
    )
    portal = DataPortal(cfg, strict_validation=True)
    bars = portal.load_bars()
    assert bars.height == 1
    assert portal.last_provenance["dropped_first_bucket"] is True
    assert portal.last_provenance["dropped_last_bucket"] is True
    assert portal.last_provenance["first_bucket_coverage"] < 0.8
    assert portal.last_provenance["last_bucket_coverage"] < 0.8

    cfg2 = cfg.model_copy(update={"drop_incomplete_bars": False})
    portal2 = DataPortal(cfg2, strict_validation=True)
    bars2 = portal2.load_bars()
    assert bars2.height == 3
    assert portal2.last_provenance["dropped_first_bucket"] is False
    assert portal2.last_provenance["dropped_last_bucket"] is False


def test_invalid_request_finer_than_native(temp_db_with_1m_bars):
    cfg = DataConfig(
        db_path=temp_db_with_1m_bars,
        table="bars_1h",
        symbol="BTC-USD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 1, 59, tzinfo=timezone.utc),
        timeframe="1m",
        native_timeframe="1h",
    )
    portal = DataPortal(cfg, strict_validation=True)
    with pytest.raises(ValueError, match="finer than native"):
        portal.load_bars()


def test_invalid_request_non_multiple(temp_db_with_1m_bars):
    cfg = DataConfig(
        db_path=temp_db_with_1m_bars,
        table="bars_1h",
        symbol="BTC-USD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 1, 59, tzinfo=timezone.utc),
        timeframe="90m",
        native_timeframe="1h",
    )
    portal = DataPortal(cfg, strict_validation=True)
    with pytest.raises(ValueError, match="not an integer multiple"):
        portal.load_bars()


def test_resample_1h_to_4h_and_1d(temp_db_with_1m_bars_long):
    cfg = DataConfig(
        db_path=temp_db_with_1m_bars_long,
        table="bars_1h",
        symbol="BTC-USD",
        start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 3, 23, 0, tzinfo=timezone.utc),
        timeframe="4h",
        native_timeframe="1h",
        drop_incomplete_bars=False,
    )
    portal = DataPortal(cfg, strict_validation=True)
    bars_4h = portal.load_bars()
    assert bars_4h.height < 72
    assert portal.last_provenance["median_delta_hours"] == pytest.approx(4.0, rel=1e-6)
    ts_dtype = bars_4h.schema.get("ts")
    assert getattr(ts_dtype, "time_zone", None) in (None, "UTC")

    cfg2 = cfg.model_copy(update={"timeframe": "1d"})
    portal2 = DataPortal(cfg2, strict_validation=True)
    bars_1d = portal2.load_bars()
    assert bars_1d.height in (3, 4)
    assert portal2.last_provenance["median_delta_hours"] == pytest.approx(24.0, rel=1e-6)
