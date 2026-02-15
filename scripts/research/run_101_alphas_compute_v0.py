#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os
from typing import Optional

import duckdb
import pandas as pd

from alphas101_lib_v0 import compute_all_alphas_v0

DEFAULT_DB_PATH = Path("data/market.duckdb")


def resolve_db_path(args: argparse.Namespace) -> Path:
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
            f"[run_101_alphas_compute_v0] ERROR: DuckDB file not found at: {db_path}",
            "",
            "Hints:",
            "- set TREND_CRYPTO_DUCKDB_PATH=/path/to/your.duckdb, or",
            "  pass --db /path/to/your.duckdb, or",
            "  symlink/copy it to data/market.duckdb",
        ]
        if candidates:
            msg.append("")
            msg.append("Discovered .duckdb files under ./data:")
            for c in candidates:
                msg.append(f"  - {c.resolve()}")
        raise FileNotFoundError("\n".join(msg))
    return db_path


def load_prices(
    db_path: Path,
    table: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    con.execute("SET TimeZone='UTC';")

    cols = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [table],
    ).fetchall()
    has_vwap = any(row[0].lower() == "vwap" for row in cols)

    where = []
    params = []
    if start:
        where.append("ts >= ?")
        params.append(start)
    if end:
        where.append("ts <= ?")
        params.append(end)
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    vwap_select = "vwap" if has_vwap else "CAST(NULL AS DOUBLE) AS vwap"
    query = f"""
        SELECT ts, symbol, open, high, low, close, volume, {vwap_select}
        FROM {table}
        {where_clause}
        ORDER BY ts, symbol;
    """
    df = con.execute(query, params).fetch_df()
    con.close()

    if "vwap" not in df.columns:
        df["vwap"] = pd.NA
    df["vwap"] = df["vwap"].fillna(df["close"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a subset of 101 Alphas over the USD universe."
    )
    parser.add_argument("--db", type=str, default=None, help="Path to DuckDB file.")
    parser.add_argument(
        "--table",
        type=str,
        default="bars_1d_usd_universe_clean",
        help="Daily bars table/view to read.",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (inclusive)")
    parser.add_argument("--end", type=str, default=None, help="End date (inclusive)")
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/research/101_alphas/alphas_101_v0.parquet",
        help="Output parquet path for alpha panel.",
    )
    parser.add_argument(
        "--use_adv10m_view",
        action="store_true",
        help="Use bars_1d_usd_universe_clean_adv10m instead of --table (requires the view to exist).",
    )

    args = parser.parse_args()
    db_path = resolve_db_path(args)
    table = args.table
    if args.use_adv10m_view:
        table = "bars_1d_usd_universe_clean_adv10m"
        con = duckdb.connect(str(db_path))
        exists = con.execute(
            """
            SELECT COUNT(*) AS n
            FROM information_schema.tables
            WHERE table_name = 'bars_1d_usd_universe_clean_adv10m'
            """
        ).fetch_df().iloc[0]["n"]
        con.close()
        if exists == 0:
            raise RuntimeError(
                "[run_101_alphas_compute_v0] bars_1d_usd_universe_clean_adv10m not found. "
                "Run scripts/research/create_usd_universe_adv10m_view.py first."
            )

    df = load_prices(db_path, table, args.start, args.end)

    if df.empty:
        raise RuntimeError("No data returned for the specified range/table.")

    symbols = df["symbol"].nunique()
    print(
        f"[run_101_alphas_compute_v0] Loaded {len(df)} rows for {symbols} symbols from {table}"
    )

    alphas = compute_all_alphas_v0(df)
    alphas = alphas.reset_index().rename(columns={"level_0": "symbol", "level_1": "ts"})
    alphas = alphas.sort_values(["ts", "symbol"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alphas.to_parquet(out_path, index=False)

    cols = [c for c in alphas.columns if c.startswith("alpha_")]
    print(
        f"[run_101_alphas_compute_v0] Computed {len(cols)} alpha-series: {', '.join(cols)}"
    )
    print(f"[run_101_alphas_compute_v0] Wrote alphas to {out_path}")


if __name__ == "__main__":
    main()

