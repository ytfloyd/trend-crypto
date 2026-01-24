"""DuckDB inspection and validation utilities for zero-footgun research workflows."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import duckdb

from common.timeframe import timeframe_to_seconds

def normalize_timeframe(timeframe: str) -> str:
    """
    Normalize timeframe string to canonical token.
    
    Supported inputs: 1h, 1H, 60m, 60min, 1d, 1D, 24h, 4h, 4H, etc.
    Returns: canonical tokens like "1h", "4h", "1d"
    
    Examples:
        "1h" -> "1h"
        "1H" -> "1h"
        "60m" -> "1h"
        "60min" -> "1h"
        "1d" -> "1d"
        "24h" -> "1d"
        "4h" -> "4h"
    """
    tf = timeframe.lower().strip()
    
    # Direct matches
    if tf in ["1h", "4h", "1d"]:
        return tf
    
    # Minute conversions
    if tf in ["60m", "60min"]:
        return "1h"
    if tf in ["240m", "240min"]:
        return "4h"
    if tf in ["1440m", "1440min"]:
        return "1d"
    
    # Hour conversions
    if tf == "24h":
        return "1d"
    
    # Try to parse patterns like "1H", "4H", "1D"
    match = re.match(r"^(\d+)([hdm])$", tf)
    if match:
        num, unit = match.groups()
        num = int(num)
        if unit == "h":
            if num == 1:
                return "1h"
            elif num == 4:
                return "4h"
            elif num == 24:
                return "1d"
        elif unit == "d" and num == 1:
            return "1d"
        elif unit == "m":
            if num == 60:
                return "1h"
            elif num == 240:
                return "4h"
            elif num == 1440:
                return "1d"
    
    raise ValueError(
        f"Cannot normalize timeframe '{timeframe}'. "
        f"Supported: 1h, 4h, 1d, 60m, 240m, 1440m, 24h"
    )


def list_tables(db_path: str) -> list[str]:
    """List all tables in the DuckDB database."""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        result = conn.execute("SHOW TABLES").fetchall()
        return [row[0] for row in result]
    finally:
        conn.close()


def describe_table(db_path: str, table: str) -> set[str]:
    """Return set of column names in the table."""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        result = conn.execute(f"DESCRIBE {table}").fetchall()
        return {row[0] for row in result}
    finally:
        conn.close()


def resolve_ts_column(columns: set[str]) -> str:
    """
    Resolve timestamp column name.
    
    Prefers "ts", else "time", else raises error.
    """
    if "ts" in columns:
        return "ts"
    if "time" in columns:
        return "time"
    
    raise ValueError(
        f"No timestamp column found. Expected 'ts' or 'time'. "
        f"Available columns: {sorted(columns)}"
    )


def validate_required_columns(columns: set[str]) -> None:
    """
    Validate that table has required OHLCV columns.
    
    Required: symbol, open, high, low, close, volume, (ts or time)
    """
    required = {"symbol", "open", "high", "low", "close", "volume"}
    missing = required - columns
    
    if missing:
        raise ValueError(
            f"Table missing required columns: {sorted(missing)}. "
            f"Available: {sorted(columns)}"
        )
    
    # Check for timestamp column
    if "ts" not in columns and "time" not in columns:
        raise ValueError(
            f"Table missing timestamp column (expected 'ts' or 'time'). "
            f"Available: {sorted(columns)}"
        )


def resolve_bars_table(
    db_path: str,
    requested_table: Optional[str],
    timeframe: str,
) -> str:
    """
    Resolve bars table name, with auto-selection when not specified.
    
    Logic:
    - If requested_table provided: verify it exists; else error with table list.
    - If not provided:
      - Normalize timeframe to canonical form
      - Prefer table named f"bars_{canonical_timeframe}"
      - Else if exactly one table matches r"^bars_": choose it
      - Else raise error listing candidates and suggesting --table
    
    Returns:
        table name
    
    Raises:
        ValueError with actionable error message
    """
    tables = list_tables(db_path)
    
    if requested_table:
        if requested_table in tables:
            return requested_table
        else:
            bars_tables = [t for t in tables if t.startswith("bars_")]
            if bars_tables:
                raise ValueError(
                    f"Table '{requested_table}' not found. "
                    f"Available bars_* tables: {bars_tables}. "
                    f"Use --table to specify."
                )
            else:
                raise ValueError(
                    f"Table '{requested_table}' not found. "
                    f"Available tables: {tables}. "
                    f"Use --table to specify."
                )
    
    # Auto-select based on timeframe
    canonical_tf = normalize_timeframe(timeframe)
    preferred_table = f"bars_{canonical_tf}"
    
    if preferred_table in tables:
        return preferred_table
    
    # Fallback: if exactly one bars_* table exists, use it
    bars_tables = [t for t in tables if t.startswith("bars_")]
    if len(bars_tables) == 1:
        return bars_tables[0]
    
    # Multiple bars_* tables or none found
    if len(bars_tables) > 1:
        raise ValueError(
            f"Multiple bars_* tables found: {bars_tables}. "
            f"Expected timeframe={canonical_tf} → table=bars_{canonical_tf}, but it does not exist. "
            f"Use --table to specify which table to use."
        )
    else:
        raise ValueError(
            f"No bars_* tables found in database. "
            f"Available tables: {tables}. "
            f"Use --table to specify a table with OHLCV data."
        )


def infer_start_end(
    db_path: str,
    table: str,
    symbol: str,
    ts_col: str = "ts",
    timeframe_col: Optional[str] = None,
    timeframe_val: Optional[str] = None,
) -> tuple[datetime, datetime]:
    """
    Infer start and end datetimes from table via MIN/MAX query.
    
    Args:
        db_path: Path to DuckDB
        table: Table name
        symbol: Symbol to filter by
        ts_col: Timestamp column name (default "ts")
        timeframe_col: Optional timeframe column name
        timeframe_val: Optional timeframe value to filter by
    
    Returns:
        (start, end) as timezone-aware UTC datetimes
    
    Raises:
        ValueError if no data found
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        # Build query
        if timeframe_col and timeframe_val:
            query = f"SELECT MIN({ts_col}) AS start_ts, MAX({ts_col}) AS end_ts FROM {table} WHERE symbol = ? AND {timeframe_col} = ?"
            params = [symbol, timeframe_val]
        else:
            query = f"SELECT MIN({ts_col}) AS start_ts, MAX({ts_col}) AS end_ts FROM {table} WHERE symbol = ?"
            params = [symbol]
        
        result = conn.execute(query, params).fetchone()
        
        if result is None or result[0] is None or result[1] is None:
            filter_desc = f"symbol={symbol}"
            if timeframe_col and timeframe_val:
                filter_desc += f", {timeframe_col}={timeframe_val}"
            raise ValueError(
                f"No data found in {table} for {filter_desc}"
            )
        
        start_ts, end_ts = result
        
        # Ensure timezone-aware UTC
        if isinstance(start_ts, datetime):
            if start_ts.tzinfo is None:
                start_ts = start_ts.replace(tzinfo=timezone.utc)
        else:
            start_ts = datetime.fromisoformat(str(start_ts)).replace(tzinfo=timezone.utc)
        
        if isinstance(end_ts, datetime):
            if end_ts.tzinfo is None:
                end_ts = end_ts.replace(tzinfo=timezone.utc)
        else:
            end_ts = datetime.fromisoformat(str(end_ts)).replace(tzinfo=timezone.utc)
        
        return start_ts, end_ts
    finally:
        conn.close()


def infer_native_timeframe(
    db_path: str,
    table: str,
    symbol: str,
    ts_col: str = "ts",
    timeframe_col: Optional[str] = None,
    timeframe_val: Optional[str] = None,
    sample_limit: int = 200,
) -> str:
    """
    Infer native timeframe by sampling timestamp deltas.

    Returns a canonical timeframe string (e.g., "1m", "5m", "1h", "1d").
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        if timeframe_col and timeframe_val:
            query = f"""
                SELECT {ts_col}
                FROM {table}
                WHERE symbol = ? AND {timeframe_col} = ?
                ORDER BY {ts_col} ASC
                LIMIT {sample_limit}
            """
            params = [symbol, timeframe_val]
        else:
            query = f"""
                SELECT {ts_col}
                FROM {table}
                WHERE symbol = ?
                ORDER BY {ts_col} ASC
                LIMIT {sample_limit}
            """
            params = [symbol]

        rows = conn.execute(query, params).fetchall()
        if len(rows) < 2:
            raise ValueError(f"Insufficient rows to infer native timeframe for {symbol} in {table}")

        # Compute median delta in seconds
        ts_list = [r[0] for r in rows]
        deltas = []
        for i in range(1, len(ts_list)):
            dt = ts_list[i] - ts_list[i - 1]
            deltas.append(int(dt.total_seconds()))
        deltas.sort()
        median = deltas[len(deltas) // 2]

        if median <= 0:
            raise ValueError(f"Invalid timestamp deltas for {symbol} in {table}")

        # Convert to canonical timeframe string
        if median % 60 != 0:
            raise ValueError(f"Non-minute native interval ({median}s) for {symbol} in {table}")
        minutes = median // 60
        if minutes < 60:
            return f"{minutes}m"
        if minutes % 60 == 0:
            hours = minutes // 60
            if hours < 24:
                return f"{hours}h"
            if hours % 24 == 0:
                days = hours // 24
                return f"{days}d"
        return f"{minutes}m"
    finally:
        conn.close()


def validate_funding_column(
    db_path: str,
    table: str,
    funding_col: str,
) -> bool:
    """
    Check if funding column exists in table.
    
    Returns:
        True if column exists, False otherwise
    """
    columns = describe_table(db_path, table)
    return funding_col in columns
