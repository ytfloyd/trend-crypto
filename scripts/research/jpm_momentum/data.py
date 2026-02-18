"""
Data loading for JPM Momentum research.

Supports two markets:
- **Crypto**: queries ``market.duckdb`` via the ``bars_1d`` view
  (resamples ``candles_1m`` on-the-fly).
- **ETF**: queries ``etf_market.duckdb`` ``bars_1d`` table
  (ingested from Tiingo via ``scripts.research.etf_data.ingest``).

Both produce the same long-format DataFrame:
``(symbol, ts, open, high, low, close, volume)``.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# Default paths, relative to this file's location
DEFAULT_CRYPTO_DB = str(
    Path(__file__).resolve().parents[3] / ".." / "data" / "market.duckdb"
)
DEFAULT_ETF_DB = str(
    Path(__file__).resolve().parents[3] / ".." / "data" / "etf_market.duckdb"
)

# Annualisation factors
ANN_FACTOR_CRYPTO = 365.0   # crypto trades every calendar day
ANN_FACTOR_ETF = 252.0      # ~252 trading days per year
ANN_FACTOR = ANN_FACTOR_CRYPTO  # backward-compatible default


def ann_factor_for_market(market: str) -> float:
    """Return the annualisation factor for the given market."""
    if market == "etf":
        return ANN_FACTOR_ETF
    return ANN_FACTOR_CRYPTO


# ---------------------------------------------------------------------------
# Crypto data loading
# ---------------------------------------------------------------------------
def load_daily_bars(
    db_path: str = DEFAULT_CRYPTO_DB,
    start: str = "2017-01-01",
    end: str = "2026-12-31",
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load daily OHLCV bars for ALL USD crypto symbols from market.duckdb.

    Uses the ``bars_1d`` view (resamples candles_1m on-the-fly).
    Results are cached to parquet for fast reload on subsequent calls.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, ts, open, high, low, close, volume.
        ``ts`` is tz-naive UTC datetime.
    """
    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parent / "_cache")
    cache_path = Path(cache_dir) / f"bars_1d_crypto_{start}_{end}.parquet"

    if cache_path.exists():
        print(f"[data] Loading cached crypto daily bars from {cache_path}")
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


# ---------------------------------------------------------------------------
# ETF data loading
# ---------------------------------------------------------------------------
def load_etf_daily_bars(
    db_path: str = DEFAULT_ETF_DB,
    start: str = "2005-01-01",
    end: str = "2026-12-31",
    *,
    tickers: list[str] | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load daily OHLCV bars for ETFs from etf_market.duckdb.

    Uses pre-ingested adjusted prices (from Tiingo via the ingest script).
    Results are cached to parquet for fast reload on subsequent calls.

    Parameters
    ----------
    db_path : str
        Path to the ETF DuckDB file.
    start / end : str
        ISO-8601 date bounds.
    tickers : list[str] | None
        If provided, filter to these tickers only.  Otherwise load all.
    cache_dir : str | None
        Directory for parquet cache.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, ts, open, high, low, close, volume.
        ``ts`` is tz-naive UTC datetime.
    """
    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parent / "_cache")

    ticker_tag = "all" if tickers is None else f"{len(tickers)}etfs"
    cache_path = Path(cache_dir) / f"bars_1d_etf_{ticker_tag}_{start}_{end}.parquet"

    if cache_path.exists():
        print(f"[data] Loading cached ETF daily bars from {cache_path}")
        df = pd.read_parquet(cache_path)
        df["ts"] = pd.to_datetime(df["ts"])
        if tickers is not None:
            df = df[df["symbol"].isin(tickers)]
        return df

    print(f"[data] Querying ETF bars_1d from {db_path} ({start} to {end}) ...")
    con = duckdb.connect(db_path, read_only=True)
    try:
        if tickers is not None:
            placeholders = ",".join(["?"] * len(tickers))
            df = con.execute(
                f"""
                SELECT symbol, ts, open, high, low, close, volume
                FROM bars_1d
                WHERE ts >= ? AND ts <= ?
                  AND symbol IN ({placeholders})
                  AND open > 0 AND close > 0
                  AND high >= low
                ORDER BY ts, symbol
                """,
                [start, end] + tickers,
            ).fetch_df()
        else:
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

    df["ts"] = pd.to_datetime(df["ts"])
    # ETF timestamps from Tiingo are already tz-naive dates
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df = df.dropna(subset=["open", "close"])

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[data] Cached {len(df):,} rows -> {cache_path}")

    return df


# ---------------------------------------------------------------------------
# Shared helpers (market-agnostic)
# ---------------------------------------------------------------------------
def compute_returns_wide(
    panel: pd.DataFrame,
    method: str = "open_to_close",
) -> pd.DataFrame:
    """Pivot panel to wide-format returns.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel with symbol, ts, open, close columns.
    method : str
        ``"open_to_close"`` — intra-day return (close / open - 1).
            Use for crypto (24h markets, no overnight gap).
        ``"close_to_close"`` — full daily return (close[t] / close[t-1] - 1).
            Use for ETFs (captures overnight gaps and dividends).

    Returns
    -------
    pd.DataFrame
        Index = ts (datetime), columns = symbols, values = daily return.
    """
    df = panel.copy()
    if method == "close_to_close":
        df = df.sort_values(["symbol", "ts"])
        df["ret"] = df.groupby("symbol")["close"].pct_change()
    else:
        df["ret"] = df["close"] / df["open"] - 1.0
    return df.pivot(index="ts", columns="symbol", values="ret").sort_index()


def compute_close_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot panel to wide-format close prices.

    Returns
    -------
    pd.DataFrame
        Index = ts, columns = symbols, values = close price.
    """
    return panel.pivot(index="ts", columns="symbol", values="close").sort_index()


def compute_benchmark(panel: pd.DataFrame, ticker: str = "BTC-USD") -> pd.Series:
    """Compute buy-and-hold equity curve for a benchmark ticker.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel.
    ticker : str
        Benchmark symbol.  ``"BTC-USD"`` for crypto, ``"SPY"`` for ETFs.

    Returns a Series indexed by ts, normalised to start at 1.0.
    """
    bench = panel.loc[panel["symbol"] == ticker, ["ts", "close"]].copy()
    if bench.empty:
        raise ValueError(f"Benchmark ticker {ticker!r} not found in panel")
    bench = bench.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    eq = bench["close"] / bench["close"].iloc[0]
    eq.name = f"{ticker}_equity"
    return eq


# Backward-compatible alias
def compute_btc_benchmark(panel: pd.DataFrame) -> pd.Series:
    """Compute BTC buy-and-hold equity curve (backward-compatible alias)."""
    return compute_benchmark(panel, ticker="BTC-USD")
