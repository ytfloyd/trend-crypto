"""
Shared data loading and universe filtering utilities.

Used by all paper-recreation research packages.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DB = str(
    Path(__file__).resolve().parents[3] / ".." / "data" / "market.duckdb"
)
ANN_FACTOR = 365.0  # crypto trades every day


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_daily_bars(
    db_path: str = DEFAULT_DB,
    start: str = "2017-01-01",
    end: str = "2026-12-31",
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load daily OHLCV bars for ALL USD symbols from market.duckdb.

    Uses the ``bars_1d`` view (resamples candles_1m on-the-fly).
    Results are cached to parquet for fast reload on subsequent calls.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, ts, open, high, low, close, volume
        ts is tz-naive UTC datetime.
    """
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


# Frequency label -> DuckDB time_bucket interval
FREQ_INTERVALS: dict[str, str] = {
    "5m":  "5 minutes",
    "30m": "30 minutes",
    "1h":  "1 hour",
    "4h":  "4 hours",
    "8h":  "8 hours",
    "1d":  "1 day",
}

# Approximate bars per calendar day (crypto = 24h markets)
BARS_PER_DAY: dict[str, float] = {
    "5m":  288.0,
    "30m":  48.0,
    "1h":   24.0,
    "4h":    6.0,
    "8h":    3.0,
    "1d":    1.0,
}


def load_bars(
    freq: str,
    db_path: str = DEFAULT_DB,
    start: str = "2017-01-01",
    end: str = "2026-12-31",
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV bars at *any* supported frequency from market.duckdb.

    Resamples ``candles_1m`` on-the-fly via DuckDB ``time_bucket``.
    Results are cached to parquet for fast reload on subsequent calls.

    Parameters
    ----------
    freq : str
        One of ``5m``, ``30m``, ``1h``, ``4h``, ``8h``, ``1d``.
    db_path : str
        Path to DuckDB file.
    start / end : str
        ISO-8601 date bounds.
    cache_dir : str | None
        Directory for parquet cache.  Defaults to ``common/_cache``.

    Returns
    -------
    pd.DataFrame  (symbol, ts, open, high, low, close, volume)
    """
    if freq not in FREQ_INTERVALS:
        raise ValueError(f"Unsupported freq {freq!r}. Choose from {list(FREQ_INTERVALS)}")

    if freq == "1d":
        return load_daily_bars(db_path=db_path, start=start, end=end, cache_dir=cache_dir)

    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parent / "_cache")
    cache_path = Path(cache_dir) / f"bars_{freq}_{start}_{end}.parquet"

    if cache_path.exists():
        print(f"[data] Loading cached {freq} bars from {cache_path}")
        df = pd.read_parquet(cache_path)
        df["ts"] = pd.to_datetime(df["ts"])
        return df

    interval = FREQ_INTERVALS[freq]
    print(f"[data] Querying candles_1m -> {freq} ({interval}) from {db_path} ({start} to {end}) ...")
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT
                symbol,
                time_bucket(INTERVAL '{interval}', ts) AS ts,
                FIRST(open ORDER BY ts)  AS open,
                MAX(high)                AS high,
                MIN(low)                 AS low,
                LAST(close ORDER BY ts)  AS close,
                SUM(volume)              AS volume
            FROM candles_1m
            WHERE ts >= ? AND ts <= ?
            GROUP BY symbol, time_bucket(INTERVAL '{interval}', ts)
            HAVING open > 0 AND close > 0 AND MAX(high) >= MIN(low)
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


# ---------------------------------------------------------------------------
# Universe filtering
# ---------------------------------------------------------------------------
def filter_universe(
    panel: pd.DataFrame,
    min_adv_usd: float = 1_000_000,
    min_history_days: int = 90,
    adv_window: int = 20,
) -> pd.DataFrame:
    """Apply dynamic universe filter: rolling ADV + minimum listing age.

    Adds boolean column ``in_universe`` to the panel.
    Symbols enter once they have ``min_history_days`` of data AND
    their rolling ``adv_window``-day average daily volume (in USD) exceeds
    ``min_adv_usd``.

    Returns a copy with the ``in_universe`` column.
    """
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        dollar_vol = g["close"] * g["volume"]
        adv = dollar_vol.rolling(adv_window, min_periods=adv_window).mean()
        day_count = np.arange(1, len(g) + 1)
        g["in_universe"] = (adv >= min_adv_usd) & (day_count >= min_history_days)
        return g

    out = df.groupby("symbol", group_keys=False).apply(_per_symbol)
    out["in_universe"] = out["in_universe"].fillna(False)
    return out


# ---------------------------------------------------------------------------
# BTC benchmark helper
# ---------------------------------------------------------------------------
def compute_btc_benchmark(panel: pd.DataFrame) -> pd.Series:
    """Compute BTC buy-and-hold equity curve from the panel data.

    Returns a Series indexed by ts, normalized to start at 1.0.
    """
    btc = panel.loc[panel["symbol"] == "BTC-USD", ["ts", "close"]].copy()
    btc = btc.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    btc_equity = btc["close"] / btc["close"].iloc[0]
    btc_equity.name = "btc_equity"
    return btc_equity
