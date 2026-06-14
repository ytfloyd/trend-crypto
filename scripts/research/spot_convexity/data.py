"""Point-in-time daily panel for the spot-convexity sleeve (Coinbase USD spot).

Reuses the medallion-sleeve convention: top-N universe by 20-day trailing dollar-ADV,
survivorship-free (membership known as-of each date). Daily bars from bars_1d_usd_universe_clean.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LAKE = str(ROOT.parent / "data" / "coinbase_crypto_ohlcv_lake.duckdb")


def load_daily_panel(start: str = "2021-01-01", end: str = "2026-06-01", top_n: int = 100) -> pd.DataFrame:
    """Long DataFrame [symbol, ts, open, high, low, close, volume, adv20, rank, in_universe]."""
    con = duckdb.connect(LAKE, read_only=True)
    df = con.execute(
        """SELECT symbol, ts, open, high, low, close, volume
           FROM bars_1d_usd_universe_clean
           WHERE ts>=? AND ts<=? AND close>0 AND open>0 AND high>=low
           ORDER BY symbol, ts""", [start, end]).fetch_df()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df["dv"] = df["close"] * df["volume"]
    df["adv20"] = df.groupby("symbol")["dv"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["rank"] = df.groupby("ts")["adv20"].rank(ascending=False, method="first")
    df["in_universe"] = (df["adv20"] > 0) & (df["rank"] <= top_n)
    return df
