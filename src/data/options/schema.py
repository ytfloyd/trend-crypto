"""DuckDB schema for options data storage.

Tables:
    option_chains      — static chain metadata (strikes, expiries, multiplier)
    option_ticks       — point-in-time option quotes (bid/ask/last, greeks, IV)
    vol_surface_snaps  — aggregated vol surface snapshots for research

Design mirrors the existing candles_1m pattern in collector.py:
create tables idempotently, upsert on (symbol, expiry, strike, ts).
"""
from __future__ import annotations

import duckdb

from common.logging import get_logger

logger = get_logger("options_schema")


class OptionsSchema:
    """Create and manage DuckDB tables for options data."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def ensure_tables(self) -> None:
        """Create all options tables if they don't exist."""
        self._create_option_chains()
        self._create_option_ticks()
        self._create_vol_surface_snaps()
        logger.info("Options schema ready")

    def _create_option_chains(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS option_chains (
                underlying        VARCHAR NOT NULL,
                exchange          VARCHAR NOT NULL,
                currency          VARCHAR NOT NULL DEFAULT 'USD',
                expiry            DATE NOT NULL,
                strike            DOUBLE NOT NULL,
                right             VARCHAR NOT NULL,  -- 'C' or 'P'
                multiplier        DOUBLE NOT NULL DEFAULT 100.0,
                con_id            BIGINT,             -- IB contract ID
                last_updated      TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (underlying, expiry, strike, right)
            );
        """)

    def _create_option_ticks(self) -> None:
        """Tick-level option quotes with Greeks from IB's model."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS option_ticks (
                ts                TIMESTAMPTZ NOT NULL,
                underlying        VARCHAR NOT NULL,
                expiry            DATE NOT NULL,
                strike            DOUBLE NOT NULL,
                right             VARCHAR NOT NULL,
                underlying_price  DOUBLE,
                bid               DOUBLE,
                ask               DOUBLE,
                last              DOUBLE,
                volume            BIGINT,
                open_interest     BIGINT,
                iv                DOUBLE,           -- IB model implied vol
                delta             DOUBLE,
                gamma             DOUBLE,
                theta             DOUBLE,
                vega              DOUBLE
            );
        """)
        try:
            self._conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_option_ticks_pk
                ON option_ticks (underlying, expiry, strike, right, ts);
            """)
        except duckdb.CatalogException:
            pass

    def _create_vol_surface_snaps(self) -> None:
        """Aggregated vol surface snapshots for research."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS vol_surface_snaps (
                snap_ts           TIMESTAMPTZ NOT NULL,
                underlying        VARCHAR NOT NULL,
                expiry            DATE NOT NULL,
                tte_years         DOUBLE NOT NULL,
                strike            DOUBLE NOT NULL,
                moneyness         DOUBLE,           -- ln(K/F)
                right             VARCHAR NOT NULL,
                underlying_price  DOUBLE,
                forward           DOUBLE,
                mid_price         DOUBLE,
                bid_iv            DOUBLE,
                ask_iv            DOUBLE,
                mid_iv            DOUBLE,
                delta             DOUBLE,
                gamma             DOUBLE,
                vega              DOUBLE,
                open_interest     BIGINT,
                volume            BIGINT
            );
        """)
        try:
            self._conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_vol_surface_pk
                ON vol_surface_snaps (underlying, expiry, strike, right, snap_ts);
            """)
        except duckdb.CatalogException:
            pass

    def table_stats(self) -> dict[str, int]:
        """Row counts for all options tables."""
        stats = {}
        for table in ("option_chains", "option_ticks", "vol_surface_snaps"):
            try:
                row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[table] = row[0] if row else 0
            except duckdb.CatalogException:
                stats[table] = -1
        return stats
