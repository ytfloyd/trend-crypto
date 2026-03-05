"""
Data loading for Sornette LPPL research.

Loads daily OHLCV from market.duckdb (crypto) via the bars_1d view.
Same pattern as the momentum research, kept self-contained here.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_DB = str(
    Path(__file__).resolve().parents[3] / ".." / "data" / "market.duckdb"
)
ANN_FACTOR = 365.0


def load_daily_bars(
    db_path: str = DEFAULT_DB,
    start: str = "2017-01-01",
    end: str = "2026-12-31",
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load daily OHLCV for all USD crypto symbols."""
    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parent / "_cache")
    cache_path = Path(cache_dir) / f"bars_1d_{start}_{end}.parquet"

    if cache_path.exists():
        print(f"[data] Loading cached daily bars from {cache_path}")
        df = pd.read_parquet(cache_path)
        df["ts"] = pd.to_datetime(df["ts"])
        return df

    print(f"[data] Querying bars_1d from {db_path} ({start} to {end}) ...")
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            """
            SELECT symbol, ts, open, high, low, close, volume
            FROM bars_1d
            WHERE ts >= ? AND ts <= ?
              AND open > 0 AND close > 0
              AND high >= low
            ORDER BY ts, symbol
            """,
            [start, end],
        ).fetch_df()
    finally:
        con.close()

    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["open", "close"])

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[data] Cached {len(df):,} rows -> {cache_path}")
    return df


def filter_universe(
    panel: pd.DataFrame,
    min_adv_usd: float = 1_000_000,
    min_history_days: int = 90,
    adv_window: int = 20,
) -> pd.DataFrame:
    """Dynamic universe filter: rolling ADV + minimum listing age."""
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        dollar_vol = g["close"] * g["volume"]
        adv = dollar_vol.rolling(adv_window, min_periods=adv_window).mean()
        day_count = np.arange(1, len(g) + 1)
        g["in_universe"] = (adv >= min_adv_usd) & (day_count >= min_history_days)
        return g

    out = df.groupby("symbol", group_keys=False).apply(_per_symbol)
    out["in_universe"] = out["in_universe"].fillna(False)
    return out
