"""
High-frequency (hourly) data loading for LPPLS research.

Queries bars_1h from market.duckdb for intraday bubble detection.
Uses SQL-side pre-filtering to avoid loading millions of irrelevant rows.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_DB = str(
    Path(__file__).resolve().parents[3] / ".." / "data" / "market.duckdb"
)
ANN_FACTOR_HOURLY = 365.0 * 24  # 8,760 hours/year


def load_hourly_bars(
    db_path: str = DEFAULT_DB,
    start: str = "2023-01-01",
    end: str = "2026-12-31",
    min_adv_usd: float = 5_000_000,
    max_symbols: int = 50,
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load hourly OHLCV for top liquid USD crypto symbols.

    Pre-filters in SQL to only load symbols with sufficient daily
    volume, dramatically reducing the data volume.
    """
    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parent / "_cache")
    cache_path = Path(cache_dir) / f"bars_1h_{start}_{end}_top{max_symbols}.parquet"

    if cache_path.exists():
        print(f"[data_hf] Loading cached hourly bars from {cache_path}")
        df = pd.read_parquet(cache_path)
        df["ts"] = pd.to_datetime(df["ts"])
        return df

    print(f"[data_hf] Querying bars_1h from {db_path} ({start} to {end}) ...")
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Step 1: find top symbols by median daily dollar volume
        top_syms = con.execute(
            f"""
            WITH daily_dv AS (
                SELECT symbol,
                       date_trunc('day', ts) AS day,
                       SUM(close * volume) AS dv
                FROM bars_1h
                WHERE ts >= ? AND ts <= ?
                  AND close > 0 AND volume > 0
                GROUP BY symbol, date_trunc('day', ts)
            )
            SELECT symbol, MEDIAN(dv) AS med_dv
            FROM daily_dv
            GROUP BY symbol
            HAVING MEDIAN(dv) >= {min_adv_usd}
            ORDER BY med_dv DESC
            LIMIT {max_symbols}
            """,
            [start, end],
        ).fetchall()
        sym_list = [r[0] for r in top_syms]
        print(f"[data_hf] {len(sym_list)} symbols pass ADV filter (>= ${min_adv_usd/1e6:.0f}M)")

        if not sym_list:
            con.close()
            return pd.DataFrame()

        # Step 2: load only those symbols
        placeholders = ", ".join(["?"] * len(sym_list))
        df = con.execute(
            f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM bars_1h
            WHERE ts >= ? AND ts <= ?
              AND symbol IN ({placeholders})
              AND open > 0 AND close > 0
              AND high >= low
            ORDER BY symbol, ts
            """,
            [start, end] + sym_list,
        ).fetch_df()
    finally:
        con.close()

    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["open", "close"])

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[data_hf] Loaded {len(df):,} rows ({df['symbol'].nunique()} symbols) -> cached")
    return df


def filter_universe_hourly(
    panel: pd.DataFrame,
    min_history_hours: int = 24 * 30,
) -> pd.DataFrame:
    """Lightweight universe filter — symbols already pre-filtered in SQL."""
    df = panel.copy()
    counts = df.groupby("symbol")["ts"].transform("cumcount") + 1
    df["in_universe"] = counts >= min_history_hours
    return df
