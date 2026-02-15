"""Tests for DuckDB inspection and validation utilities."""
import pytest

pytest.importorskip("duckdb")

from datetime import datetime
import tempfile
from pathlib import Path

import duckdb

from utils.duckdb_inspect import (
    describe_table,
    infer_start_end,
    list_tables,
    normalize_timeframe,
    resolve_bars_table,
    resolve_ts_column,
    validate_funding_column,
    validate_required_columns,
)


@pytest.fixture
def temp_db_with_bars_1d():
    """Create a temporary DuckDB with bars_1d table and sample data."""
    import os
    
    # Create temp file path but delete the file so DuckDB can create a fresh database
    fd, db_path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.unlink(db_path)
    
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE bars_1d (
            ts TIMESTAMP,
            symbol VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)
    
    # Insert sample data
    conn.execute("""
        INSERT INTO bars_1d VALUES
        ('2021-01-01 00:00:00', 'BTC-USD', 29000.0, 30000.0, 28500.0, 29500.0, 1000.0),
        ('2021-01-02 00:00:00', 'BTC-USD', 29500.0, 31000.0, 29000.0, 30500.0, 1100.0),
        ('2021-01-03 00:00:00', 'BTC-USD', 30500.0, 32000.0, 30000.0, 31500.0, 1200.0),
        ('2022-01-01 00:00:00', 'BTC-USD', 47000.0, 48000.0, 46500.0, 47500.0, 2000.0),
        ('2023-01-01 00:00:00', 'BTC-USD', 16500.0, 17000.0, 16000.0, 16800.0, 3000.0)
    """)
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink()


@pytest.fixture
def temp_db_with_multiple_tables():
    """Create a temporary DuckDB with multiple bars_* tables."""
    import os
    
    # Create temp file path but delete the file so DuckDB can create a fresh database
    fd, db_path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.unlink(db_path)
    
    conn = duckdb.connect(db_path)
    
    # Create bars_1h
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
    
    # Create bars_1d
    conn.execute("""
        CREATE TABLE bars_1d (
            ts TIMESTAMP,
            symbol VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)
    
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink()


def test_normalize_timeframe():
    """Test timeframe normalization."""
    assert normalize_timeframe("1h") == "1h"
    assert normalize_timeframe("1H") == "1h"
    assert normalize_timeframe("60m") == "1h"
    assert normalize_timeframe("60min") == "1h"
    assert normalize_timeframe("4h") == "4h"
    assert normalize_timeframe("4H") == "4h"
    assert normalize_timeframe("240m") == "4h"
    assert normalize_timeframe("1d") == "1d"
    assert normalize_timeframe("1D") == "1d"
    assert normalize_timeframe("24h") == "1d"
    assert normalize_timeframe("1440m") == "1d"
    
    with pytest.raises(ValueError, match="Cannot normalize timeframe"):
        normalize_timeframe("3h")


def test_list_tables(temp_db_with_bars_1d):
    """Test listing tables."""
    tables = list_tables(temp_db_with_bars_1d)
    assert "bars_1d" in tables


def test_describe_table(temp_db_with_bars_1d):
    """Test describing table columns."""
    columns = describe_table(temp_db_with_bars_1d, "bars_1d")
    assert "ts" in columns
    assert "symbol" in columns
    assert "open" in columns
    assert "high" in columns
    assert "low" in columns
    assert "close" in columns
    assert "volume" in columns


def test_resolve_ts_column():
    """Test resolving timestamp column name."""
    assert resolve_ts_column({"ts", "symbol", "open"}) == "ts"
    assert resolve_ts_column({"time", "symbol", "open"}) == "time"
    
    # Prefer "ts" over "time"
    assert resolve_ts_column({"ts", "time", "symbol"}) == "ts"
    
    with pytest.raises(ValueError, match="No timestamp column found"):
        resolve_ts_column({"symbol", "open", "close"})


def test_validate_required_columns():
    """Test validating required OHLCV columns."""
    # Valid column set
    valid_cols = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    validate_required_columns(valid_cols)  # Should not raise
    
    # Missing open
    with pytest.raises(ValueError, match="missing required columns.*open"):
        validate_required_columns({"ts", "symbol", "high", "low", "close", "volume"})
    
    # Missing timestamp
    with pytest.raises(ValueError, match="missing timestamp column"):
        validate_required_columns({"symbol", "open", "high", "low", "close", "volume"})


def test_resolve_bars_table_single_table(temp_db_with_bars_1d):
    """Test resolve_bars_table with single bars_* table."""
    # Should auto-select bars_1d when timeframe=1d and no table specified
    table = resolve_bars_table(temp_db_with_bars_1d, None, "1d")
    assert table == "bars_1d"
    
    # Should also work with "1D" or "24h"
    table = resolve_bars_table(temp_db_with_bars_1d, None, "1D")
    assert table == "bars_1d"
    
    table = resolve_bars_table(temp_db_with_bars_1d, None, "24h")
    assert table == "bars_1d"


def test_resolve_bars_table_explicit(temp_db_with_bars_1d):
    """Test resolve_bars_table with explicit table name."""
    table = resolve_bars_table(temp_db_with_bars_1d, "bars_1d", "1h")
    assert table == "bars_1d"
    
    # Non-existent table should error
    with pytest.raises(ValueError, match="not found"):
        resolve_bars_table(temp_db_with_bars_1d, "bars_4h", "4h")


def test_resolve_bars_table_multiple_tables(temp_db_with_multiple_tables):
    """Test resolve_bars_table with multiple bars_* tables."""
    # Should auto-select bars_1h when timeframe=1h
    table = resolve_bars_table(temp_db_with_multiple_tables, None, "1h")
    assert table == "bars_1h"
    
    # Should auto-select bars_1d when timeframe=1d
    table = resolve_bars_table(temp_db_with_multiple_tables, None, "1d")
    assert table == "bars_1d"
    
    # Should error when requesting unsupported timeframe
    with pytest.raises(ValueError, match="Multiple bars_\\* tables found"):
        resolve_bars_table(temp_db_with_multiple_tables, None, "4h")


def test_infer_start_end(temp_db_with_bars_1d):
    """Test inferring start/end dates from table."""
    start, end = infer_start_end(temp_db_with_bars_1d, "bars_1d", "BTC-USD")
    
    # Should return timezone-aware UTC datetimes
    assert isinstance(start, datetime)
    assert isinstance(end, datetime)
    assert start.tzinfo is not None
    assert end.tzinfo is not None
    
    # Check values
    assert start.year == 2021
    assert start.month == 1
    assert start.day == 1
    assert end.year == 2023
    assert end.month == 1
    assert end.day == 1
    
    # Non-existent symbol should error
    with pytest.raises(ValueError, match="No data found"):
        infer_start_end(temp_db_with_bars_1d, "bars_1d", "ETH-USD")


def test_validate_funding_column(temp_db_with_bars_1d):
    """Test validating funding column existence."""
    # funding_rate column does not exist
    assert not validate_funding_column(temp_db_with_bars_1d, "bars_1d", "funding_rate")
    
    # ts column exists
    assert validate_funding_column(temp_db_with_bars_1d, "bars_1d", "ts")
    
    # symbol column exists
    assert validate_funding_column(temp_db_with_bars_1d, "bars_1d", "symbol")
