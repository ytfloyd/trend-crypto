"""DuckDB schema for the gamma screener output table.

Stores one row per (as_of_date, symbol) with all raw features, sub-scores,
and the combined rank. Append-only; backtests / time-series analysis of
the screener come from querying this table.
"""
from __future__ import annotations

import duckdb

from common.logging import get_logger

logger = get_logger("gamma_screener_schema")


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS gamma_screener_daily (
    as_of_date       DATE NOT NULL,
    symbol           VARCHAR NOT NULL,
    spot             DOUBLE,
    iv7              DOUBLE,
    iv30             DOUBLE,
    iv60             DOUBLE,
    iv90             DOUBLE,
    rv_cc10          DOUBLE,
    rv_cc20          DOUBLE,
    rv_yz20          DOUBLE,
    rv_yz60          DOUBLE,
    iv30_rv20_ratio  DOUBLE,
    iv7_rv10_ratio   DOUBLE,
    term_30_90       DOUBLE,
    term_7_30        DOUBLE,
    skew_25d_30      DOUBLE,
    butterfly_25d_30 DOUBLE,
    iv_rank_252      DOUBLE,
    options_adv_usd  DOUBLE,
    stock_adv_usd    DOUBLE,
    bid_ask_pct      DOUBLE,
    earnings_in_window BOOLEAN,
    score_short      DOUBLE,
    score_thirty     DOUBLE,
    score_term       DOUBLE,
    score_combined   DOUBLE,
    rank_combined    INTEGER,
    PRIMARY KEY (as_of_date, symbol)
);
"""


class GammaScreenerSchema:
    """Create and manage the gamma_screener_daily table."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def ensure_tables(self) -> None:
        self._conn.execute(CREATE_TABLE_SQL)
        logger.info("gamma_screener_daily table ready")

    def upsert_rows(self, rows: list[dict[str, object]]) -> int:
        """Insert or replace rows for a given as_of_date.

        Caller is expected to pass rows that all share the same as_of_date.
        We delete-then-insert scoped to (as_of_date, symbol) for upsert
        semantics without DuckDB's ON CONFLICT. Crucially, this preserves
        rows for symbols NOT in ``rows`` (other chunks from the same day),
        so repeated chunked runs on a single trading day accumulate cleanly
        in gamma_screener_daily.
        """
        if not rows:
            return 0
        import polars as pl
        df = pl.DataFrame(rows)
        as_of = rows[0]["as_of_date"]
        symbols = [r["symbol"] for r in rows]
        placeholders = ",".join(["?"] * len(symbols))
        self._conn.execute(
            f"DELETE FROM gamma_screener_daily "
            f"WHERE as_of_date = ? AND symbol IN ({placeholders})",
            [as_of, *symbols],
        )
        self._conn.register("_tmp_gamma", df)
        self._conn.execute("""
            INSERT INTO gamma_screener_daily
            SELECT * FROM _tmp_gamma;
        """)
        self._conn.unregister("_tmp_gamma")
        return len(rows)
