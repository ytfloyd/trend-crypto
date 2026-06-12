"""(symbol, bar_frequency) data provider over the existing DuckDB lakes.

Three back-ends, routed by symbol:

* crypto  - Coinbase OHLCV lake (`coinbase_crypto_ohlcv_lake.duckdb`).
            Symbols contain a dash, e.g. ``BTC-USD``. Frequencies: 1d, 4h, 1h/60min,
            1m.
* etf     - daily ETF lake (`etf_market.duckdb`). Bare alpha tickers, e.g. ``SPY``.
            Daily only.
* futures - continuous-futures parquet artifacts (CL / NG / SI). 1-minute base
            bars resampled to the requested frequency.

All series are returned tz-naive (UTC) indexed by ``ts`` with float OHLCV columns.
The provider is read-only and opens short-lived DuckDB connections so it never
contends with the live ingestion processes (it also degrades gracefully if a
lake is locked, raising a clear error only for the symbol that needs it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import duckdb
import pandas as pd

# Shared data lakes live one level above the repo (same dir as core.data's
# DEFAULT_DB). Override with TREND_DATA_ROOT; the fallback is repo-relative so it
# works on any checkout (and in CI) rather than a machine-specific path.
DEFAULT_DATA_ROOT = os.environ.get(
    "TREND_DATA_ROOT",
    str(Path(__file__).resolve().parents[4] / ".." / "data"),
)

# Continuous-futures parquet artifacts (1-minute base bars).
DEFAULT_FUTURES_PARQUET: Dict[str, str] = {
    "CL": "artifacts/research/cl_institutional_continuous/bars_1m.parquet",
    "NG": "artifacts/research/ng_institutional_continuous/bars_1m.parquet",
    "SI": "artifacts/research/si_quicklook/si_front_month_1m.parquet",
}

# Crypto frequency -> table name in the Coinbase lake.
_CRYPTO_TABLE_BY_FREQ: Dict[str, str] = {
    "1d": "bars_1d_clean",
    "4h": "bars_4h_clean",
    "1h": "bars_1h_clean",
    "60min": "bars_1h_clean",
    "1m": "candles_1m",
    "1min": "candles_1m",
}

# Normalize a free-text frequency to a canonical token + pandas resample rule.
_FREQ_CANON: Dict[str, str] = {
    "1d": "1d", "1day": "1d", "daily": "1d", "d": "1d",
    "1w": "1w", "weekly": "1w", "w": "1w",
    "4h": "4h", "240min": "4h",
    "1h": "1h", "60min": "1h", "60m": "1h", "hourly": "1h",
    "30min": "30min", "30m": "30min",
    "5min": "5min", "5m": "5min",
    "1m": "1m", "1min": "1m",
    "intraday": "1h",   # registry shorthand; treat as hourly for research
}

_RESAMPLE_RULE: Dict[str, str] = {
    "1w": "1W", "1d": "1D", "4h": "240min", "1h": "60min",
    "30min": "30min", "5min": "5min", "1m": "1min",
}


def canonical_freq(bar_frequency: str) -> str:
    """Map a free-text frequency to a canonical token. Raises on unknown input."""
    key = str(bar_frequency).strip().lower()
    if key not in _FREQ_CANON:
        raise ValueError(f"Unsupported bar_frequency: {bar_frequency!r}")
    return _FREQ_CANON[key]


@dataclass
class LakeDataProvider:
    """Loads OHLCV bars for ``(symbol, bar_frequency)`` with a small in-process cache."""

    data_root: str = DEFAULT_DATA_ROOT
    futures_parquet: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FUTURES_PARQUET))
    repo_root: Optional[str] = None
    _cache: Dict[tuple, pd.DataFrame] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_bars(self, symbol: str, bar_frequency: str) -> pd.DataFrame:
        """Return tz-naive UTC OHLCV bars indexed by ts for (symbol, frequency)."""
        freq = canonical_freq(bar_frequency)
        cache_key = (symbol, freq)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        if symbol.upper() in self.futures_parquet:
            df = self._load_futures(symbol.upper(), freq)
        elif "-" in symbol:
            df = self._load_crypto(symbol, freq)
        else:
            df = self._load_etf(symbol, freq)

        df = _finalize(df)
        self._cache[cache_key] = df
        return df.copy()

    # ------------------------------------------------------------------
    # Back-ends
    # ------------------------------------------------------------------
    def _load_crypto(self, symbol: str, freq: str) -> pd.DataFrame:
        table = _CRYPTO_TABLE_BY_FREQ.get(freq)
        if table is None:
            raise ValueError(f"Crypto lake has no table for frequency {freq!r}")
        path = Path(self.data_root) / "coinbase_crypto_ohlcv_lake.duckdb"
        con = duckdb.connect(str(path), read_only=True)
        try:
            df = con.execute(
                f'SELECT ts, open, high, low, close, volume FROM "{table}" '
                "WHERE symbol = ? ORDER BY ts",
                [symbol],
            ).fetchdf()
        finally:
            con.close()
        if df.empty:
            raise ValueError(f"No crypto bars for {symbol!r} @ {freq}")
        return df

    def _load_etf(self, symbol: str, freq: str) -> pd.DataFrame:
        if freq != "1d":
            raise ValueError(f"ETF lake only supports daily bars (got {freq!r} for {symbol!r})")
        path = Path(self.data_root) / "etf_market.duckdb"
        con = duckdb.connect(str(path), read_only=True)
        try:
            df = con.execute(
                "SELECT ts, open, high, low, close, volume FROM bars_1d "
                "WHERE symbol = ? ORDER BY ts",
                [symbol],
            ).fetchdf()
        finally:
            con.close()
        if df.empty:
            raise ValueError(f"No ETF bars for {symbol!r}")
        return df

    def _load_futures(self, symbol: str, freq: str) -> pd.DataFrame:
        rel = self.futures_parquet[symbol]
        path = Path(rel)
        if not path.is_absolute():
            base = Path(self.repo_root) if self.repo_root else Path.cwd()
            path = base / rel
        raw = pd.read_parquet(path)
        # Continuous parquet uses single-letter OHLCV columns (o/h/l/c/v).
        rename = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        raw = raw.rename(columns=rename)
        cols = ["ts", "open", "high", "low", "close", "volume"]
        raw = raw[[c for c in cols if c in raw.columns]].copy()
        base_1m = _to_utc_naive_index(raw)
        return _resample_ohlcv(base_1m, _RESAMPLE_RULE[freq])


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _to_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``ts`` column to a tz-naive UTC DatetimeIndex, sorted & deduped."""
    out = df.copy()
    ts = pd.to_datetime(out["ts"], utc=True)
    out = out.drop(columns=["ts"])
    out.index = ts.dt.tz_localize(None)
    out.index.name = "ts"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-minute bars to ``rule`` using standard OHLCV aggregation."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out = df.resample(rule, label="right", closed="right").agg(agg)
    return out.dropna(subset=["close"])


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a clean tz-naive UTC index for every back-end."""
    if "ts" in df.columns:
        df = _to_utc_naive_index(df)
    else:
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        df = df.copy()
        df.index = idx
        df.index.name = "ts"
        df = df[~df.index.duplicated(keep="last")].sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].dropna(subset=["close"])
