#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Set

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
            f"[create_usd_universe_daily_clean_view] ERROR: DuckDB file not found at: {db_path}",
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
}


def fetch_usd_symbols(con: duckdb.DuckDBPyConnection) -> List[str]:
    rows = con.execute(
        """
        SELECT DISTINCT symbol
        FROM hourly_bars
        WHERE symbol LIKE '%-USD'
        ORDER BY symbol;
        """
    ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a USD spot universe view (daily) excluding stablecoin bases."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to DuckDB file (overrides TREND_CRYPTO_DUCKDB_PATH and default data/market.duckdb).",
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args)
    print(f"[create_usd_universe_daily_clean_view] Using DuckDB: {db_path}")

    con = duckdb.connect(str(db_path))
    con.execute("SET TimeZone='UTC';")

    symbols = fetch_usd_symbols(con)
    bases = [(s, s.split("-")[0]) for s in symbols]
    filtered = [sym for sym, base in bases if base not in STABLE_BASES]

    print(
        f"[create_usd_universe_daily_clean_view] Found {len(symbols)} USD symbols, "
        f"{len(filtered)} after excluding stablecoin bases."
    )

    if not filtered:
        con.close()
        raise RuntimeError(
            "No USD symbols remaining after excluding stablecoin bases; aborting view creation."
        )

    con.execute("CREATE OR REPLACE TEMP TABLE usd_universe(symbol TEXT);")
    con.executemany("INSERT INTO usd_universe VALUES (?)", [(s,) for s in filtered])

    con.execute(
        """
        CREATE OR REPLACE VIEW bars_1d_usd_universe_clean AS
        SELECT *
        FROM bars_1d_clean
        WHERE symbol IN (SELECT symbol FROM usd_universe);
        """
    )

    print(
        "[create_usd_universe_daily_clean_view] Created bars_1d_usd_universe_clean from bars_1d_clean."
    )

    dup_total = con.execute(
        """
        SELECT COUNT(*) AS n
        FROM (
            SELECT symbol, ts
            FROM bars_1d_usd_universe_clean
            GROUP BY symbol, ts
            HAVING COUNT(*) > 1
        ) t;
        """
    ).fetchone()[0]
    if dup_total > 0:
        dup_rows = con.execute(
            """
            SELECT symbol, ts, COUNT(*) AS n
            FROM bars_1d_usd_universe_clean
            GROUP BY symbol, ts
            HAVING COUNT(*) > 1
            ORDER BY n DESC, symbol, ts
            LIMIT 5;
            """
        ).fetchall()
        offenders = ", ".join(f"{sym} {ts} (n={n})" for sym, ts, n in dup_rows)
        con.close()
        raise RuntimeError(
            f"Found {dup_total} duplicate (symbol, ts) groups in bars_1d_usd_universe_clean; top offenders: {offenders}"
        )

    print("OK: no duplicate (symbol, ts) rows in bars_1d_usd_universe_clean")
    con.close()


if __name__ == "__main__":
    main()

