from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path("data/market.duckdb")


def resolve_db_path(args: argparse.Namespace) -> Path:
    # priority: CLI -> ENV -> default
    if getattr(args, "db", None):
        db_path = Path(args.db)
    else:
        env_path = os.getenv("TREND_CRYPTO_DUCKDB_PATH")
        if env_path:
            db_path = Path(env_path)
        else:
            db_path = DEFAULT_DB_PATH

    db_path = db_path.expanduser().resolve()

    if not db_path.exists():
        candidates = list(Path("data").glob("*.duckdb")) if Path("data").exists() else []
        msg = [
            f"[create_midcap_daily_clean_view] ERROR: DuckDB file not found at: {db_path}",
            "",
            "Hints:",
            "- If you already have a market DuckDB, either:",
            "  * set TREND_CRYPTO_DUCKDB_PATH=/path/to/your.duckdb, or",
            "  * pass --db /path/to/your.duckdb, or",
            "  * symlink/copy it to data/market.duckdb",
        ]
        if candidates:
            msg.append("")
            msg.append("Discovered .duckdb files under ./data:")
            for c in candidates:
                msg.append(f"  - {c.resolve()}")
        raise FileNotFoundError("\n".join(msg))

    return db_path


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return (
        con.execute(
            """
            SELECT COUNT(*) AS n
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [name],
        ).fetchone()[0]
        > 0
    )


def _create_from_daily(con: duckdb.DuckDBPyConnection, source_view: str) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW bars_1d_midcap_clean AS
        WITH dedup AS (
            SELECT
                ts,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                ROW_NUMBER() OVER (
                    PARTITION BY symbol, ts
                    ORDER BY volume DESC, close DESC
                ) AS rn
            FROM {source_view}
        )
        SELECT ts, symbol, open, high, low, close, volume
        FROM dedup
        WHERE rn = 1
        ORDER BY symbol, ts;
        """
    )


def _create_from_hourly(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE VIEW bars_1d_midcap_clean AS
        WITH base AS (
            SELECT
                symbol,
                CAST(date_trunc('day', ts) AS TIMESTAMP) AS ts,
                ts AS ts_raw,
                open,
                high,
                low,
                close,
                volume
            FROM hourly_bars
        ),
        agg AS (
            SELECT
                ts,
                symbol,
                arg_min(open, ts_raw) AS open,
                max(high) AS high,
                min(low) AS low,
                arg_max(close, ts_raw) AS close,
                sum(volume) AS volume
            FROM base
            GROUP BY ts, symbol
        )
        SELECT ts, symbol, open, high, low, close, volume
        FROM agg
        ORDER BY symbol, ts;
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deduplicated midcap daily view for research backtests."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to DuckDB file (overrides TREND_CRYPTO_DUCKDB_PATH and default data/market.duckdb).",
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args)
    print(f"[create_midcap_daily_clean_view] Using DuckDB: {db_path}")

    con = duckdb.connect(str(db_path))
    con.execute("SET TimeZone='UTC';")

    built_from = None
    if _table_exists(con, "bars_1d_clean"):
        _create_from_daily(con, "bars_1d_clean")
        built_from = "bars_1d_clean"
    else:
        if not _table_exists(con, "hourly_bars"):
            con.close()
            raise RuntimeError(
                "Missing source data: neither bars_1d_clean view nor hourly_bars table exists."
            )
        _create_from_hourly(con)
        built_from = "hourly_bars"

    print(
        f"[create_midcap_daily_clean_view] Created bars_1d_midcap_clean from {built_from}."
    )

    dup_total = con.execute(
        """
        SELECT COUNT(*) AS n
        FROM (
            SELECT symbol, ts
            FROM bars_1d_midcap_clean
            GROUP BY symbol, ts
            HAVING COUNT(*) > 1
        ) t;
        """
    ).fetchone()[0]
    if dup_total > 0:
        dup_rows = con.execute(
            """
            SELECT symbol, ts, COUNT(*) AS n
            FROM bars_1d_midcap_clean
            GROUP BY symbol, ts
            HAVING COUNT(*) > 1
            ORDER BY n DESC, symbol, ts
            LIMIT 5;
            """
        ).fetchall()
        offenders = ", ".join(f"{sym} {ts} (n={n})" for sym, ts, n in dup_rows)
        con.close()
        raise RuntimeError(
            f"Found {dup_total} duplicate (symbol, ts) groups in bars_1d_midcap_clean; top offenders: {offenders}"
        )

    print("OK: no duplicate (symbol, ts) rows in bars_1d_midcap_clean")
    con.close()


if __name__ == "__main__":
    main()

