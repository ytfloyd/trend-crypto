"""Silver front-month bar loaders and multi-timeframe resamplers.

Source of truth for prices is the pre-stitched parquet:
    trend_crypto/artifacts/research/si_quicklook/si_front_month_1m.parquet

The underlying 1-minute bars come from data/futures_market.duckdb
(table bars_1m, symbol='SI'). The stitched series ends 2026-05-05.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

# Project paths. We resolve to whichever mount root actually exists so the
# module works under the user's macOS layout AND inside the workspace mount.
_CANDIDATE_ROOTS = (
    Path("/Users/russellfloyd/Dropbox/NRT/nrt_dev"),
    Path("/sessions/optimistic-funny-volta/mnt/nrt_dev"),
)


def _resolve_default_front_month_parquet() -> Path:
    for root in _CANDIDATE_ROOTS:
        p = root / "trend_crypto" / "artifacts" / "research" / "si_quicklook" / "si_front_month_1m.parquet"
        if p.exists():
            return p
    # Fall back to the macOS path; the caller will see a clear FileNotFoundError.
    return _CANDIDATE_ROOTS[0] / "trend_crypto" / "artifacts" / "research" / "si_quicklook" / "si_front_month_1m.parquet"


DEFAULT_FRONT_MONTH_PARQUET = _resolve_default_front_month_parquet()

TIMEFRAMES = ("1H", "4H", "8H", "1D")


def load_si_front_month(
    start: str = "2025-10-01",
    end: str = "2026-05-05",
    parquet_path: Path | None = None,
) -> pd.DataFrame:
    """Load the stitched silver front-month 1-minute bars.

    Returns a DataFrame indexed by tz-aware UTC ``ts`` with columns
    ``o, h, l, c, v, expiry``.
    """
    path = Path(parquet_path) if parquet_path else DEFAULT_FRONT_MONTH_PARQUET
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["ts"], utc=True)
    df = df.assign(ts=ts).set_index("ts").sort_index()

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    # Use inclusive end-of-day to keep the named end date in the window.
    end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    keep = ["o", "h", "l", "c", "v", "expiry"]
    return df[[c for c in keep if c in df.columns]]


def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars to ``rule`` using OHLC aggregation.

    Drops bars with no observations (so we don't fabricate empty candles
    in the overnight market closure).
    """
    if df_1m.empty:
        return df_1m.copy()

    # Normalise legacy uppercase frequency codes (1H/4H/8H) to the pandas
    # 2.x lowercase ('h') form to avoid FutureWarnings while keeping the
    # public API call-compatible with either case.
    norm_rule = rule
    if isinstance(rule, str) and len(rule) >= 2 and rule[-1] == "H":
        norm_rule = rule[:-1] + "h"

    agg = {
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }
    cols = [c for c in agg if c in df_1m.columns]
    resampled = (
        df_1m[cols]
        .resample(norm_rule, label="left", closed="left")
        .agg({c: agg[c] for c in cols})
    )

    if "expiry" in df_1m.columns:
        expiry = df_1m["expiry"].resample(norm_rule, label="left", closed="left").last()
        resampled["expiry"] = expiry

    # Drop bars with no close (i.e. no underlying observations).
    if "c" in resampled.columns:
        resampled = resampled.dropna(subset=["c"])
    return resampled


def multi_tf_bars(
    start: str = "2025-10-01",
    end: str = "2026-05-05",
    parquet_path: Path | None = None,
) -> Dict[str, pd.DataFrame]:
    """Return a dict of resampled bars keyed by timeframe (1H/4H/8H/1D)."""
    base = load_si_front_month(start=start, end=end, parquet_path=parquet_path)
    return {tf: resample_ohlcv(base, tf) for tf in TIMEFRAMES}
