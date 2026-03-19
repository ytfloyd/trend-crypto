"""
V2 data loader — full universe with deduplication and liquidity tiering.

Key changes from V1:
  - No hard ADV floor at load time (filter at portfolio construction)
  - Deduplicates USD/USDC/USDT pairs (prefer USD, fallback USDC/USDT)
  - Excludes stablecoins, fiat pairs, wrapped assets
  - Computes rolling ADV per symbol for dynamic liquidity-aware sizing
  - Returns ADV alongside OHLCV for tiered position capping
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_DB = str(
    Path(__file__).resolve().parents[3] / ".." / "data" / "market.duckdb"
)

_STABLECOINS = {
    "USDC", "USDT", "DAI", "PAX", "TUSD", "GUSD",
    "BUSD", "USDP", "PYUSD", "FDUSD", "USDS", "UST",
    "EURC",
}

_FIAT = {"EUR", "GBP", "CAD", "AUD", "JPY"}

_EXCLUDED_BASES = _STABLECOINS | {"WBTC", "WETH", "CBETH", "STETH", "CBBTC"}


def load_full_universe(
    db_path: str = DEFAULT_DB,
    start: str = "2021-01-01",
    end: str = "2026-12-31",
    min_adv_usd: float = 50_000,
    min_history_days: int = 60,
    *,
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the full hourly universe with deduplication and ADV.

    Returns
    -------
    panel : pd.DataFrame
        Long-format OHLCV with columns [symbol, ts, open, high, low, close, volume]
        where symbol is the canonical (deduplicated) name.
    adv_daily : pd.DataFrame
        Daily ADV per symbol (ts × symbol wide-format) for liquidity-aware sizing.
    """
    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parent / "_cache")
    tag = f"v2_{start}_{end}_adv{int(min_adv_usd)}"
    cache_panel = Path(cache_dir) / f"panel_{tag}.parquet"
    cache_adv = Path(cache_dir) / f"adv_{tag}.parquet"

    if cache_panel.exists() and cache_adv.exists():
        print(f"[data_v2] Loading cached data from {cache_dir}")
        panel = pd.read_parquet(cache_panel)
        panel["ts"] = pd.to_datetime(panel["ts"])
        adv = pd.read_parquet(cache_adv)
        adv.index = pd.to_datetime(adv.index)
        return panel, adv

    print(f"[data_v2] Querying full universe from {db_path} ({start} to {end}) ...")
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute("""
            SELECT symbol, ts, open, high, low, close, volume
            FROM bars_1h
            WHERE ts >= $1 AND ts <= $2
              AND open > 0 AND close > 0 AND high >= low AND volume > 0
            ORDER BY symbol, ts
        """, [start, end]).fetch_df()
    finally:
        con.close()

    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    print(f"[data_v2] Raw: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # ── Parse symbol components ──────────────────────────────────
    parts = df["symbol"].str.split("-", expand=True)
    df["base"] = parts[0]
    df["quote"] = parts[1]

    # ── Filter: exclude stablecoins, fiat, wrapped ───────────────
    mask = (
        ~df["base"].isin(_EXCLUDED_BASES) &
        ~df["quote"].isin(_FIAT) &
        ~df["base"].isin(_FIAT)
    )
    df = df[mask].copy()
    print(f"[data_v2] After exclusions: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # ── Deduplicate: prefer USD > USDC > USDT ────────────────────
    quote_priority = {"USD": 0, "USDC": 1, "USDT": 2}
    df["quote_rank"] = df["quote"].map(quote_priority).fillna(3)
    df = df.sort_values(["base", "ts", "quote_rank"])
    df = df.drop_duplicates(subset=["base", "ts"], keep="first")
    df["symbol"] = df["base"] + "-USD"
    df = df.drop(columns=["base", "quote", "quote_rank"])
    print(f"[data_v2] After dedup: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # ── Compute daily ADV ────────────────────────────────────────
    df["dollar_vol"] = df["close"] * df["volume"]
    daily_dv = (
        df.groupby([df["symbol"], df["ts"].dt.date])["dollar_vol"]
        .sum()
        .reset_index()
    )
    daily_dv.columns = ["symbol", "date", "dv"]
    daily_dv["date"] = pd.to_datetime(daily_dv["date"])

    # Rolling 20-day median ADV
    adv_wide = daily_dv.pivot(index="date", columns="symbol", values="dv")
    adv_rolling = adv_wide.rolling(20, min_periods=5).median()

    # ── Filter by minimum ADV and history ────────────────────────
    median_adv = adv_wide.median()
    passing = median_adv[median_adv >= min_adv_usd].index.tolist()

    sym_counts = df.groupby("symbol")["ts"].nunique()
    min_hours = min_history_days * 24
    enough_hist = sym_counts[sym_counts >= min_hours].index.tolist()

    keep = sorted(set(passing) & set(enough_hist))
    df = df[df["symbol"].isin(keep)].copy()
    adv_rolling = adv_rolling[keep]
    print(f"[data_v2] Final: {len(df):,} rows, {df['symbol'].nunique()} symbols "
          f"(ADV >= ${min_adv_usd/1e3:.0f}K, history >= {min_history_days}d)")

    df = df.drop(columns=["dollar_vol"], errors="ignore")

    # ── Cache ────────────────────────────────────────────────────
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_panel, index=False)
    adv_rolling.to_parquet(cache_adv)
    print(f"[data_v2] Cached to {cache_dir}")

    return df, adv_rolling
