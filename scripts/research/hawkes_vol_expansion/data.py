"""Load continuous futures bars for CL / ZL / ZW research."""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAKE = Path(
    os.environ.get(
        "VOLBOOK_LAKE_PATH",
        REPO_ROOT.parent / "data" / "futures_market.duckdb",
    )
)
DEFAULT_LAKE_SNAPSHOT = Path("/tmp/futures_market_readonly.duckdb")

FRONT_MONTH_STITCH_SQL = """
WITH active AS (
    SELECT
        ts, expiry, o, h, l, c, v,
        strptime(expiry || '01', '%Y%m%d') AS roll_point
    FROM bars_1m
    WHERE symbol = ? AND expiry != 'continuous'
),
ranked AS (
    SELECT ts, expiry, o, h, l, c, v,
           row_number() OVER (PARTITION BY ts ORDER BY expiry ASC) AS rn
    FROM active
    WHERE ts < roll_point
)
SELECT ts, expiry, o, h, l, c, v
FROM ranked
WHERE rn = 1
ORDER BY ts
"""

ROLL_CONVENTIONS = {
    "CL": (
        "institutional_continuous: volume_crossover_with_calendar_guard, "
        "additive back-adjust, NYMEX observed calendar (volbook.continuous)"
    ),
    "ZL": "front_month_unadjusted: ts < first-of-expiry-month, nearest expiry",
    "ZW": "front_month_unadjusted: ts < first-of-expiry-month, nearest expiry",
}


def resolve_lake_path(lake_path: Path | None = None) -> Path:
    path = lake_path or DEFAULT_LAKE
    if path.exists():
        return path
    if DEFAULT_LAKE_SNAPSHOT.exists():
        return DEFAULT_LAKE_SNAPSHOT
    return path


def load_continuous_minute_bars(
    symbol: str,
    *,
    lake_path: Path | None = None,
) -> pd.DataFrame:
    """Return 1-minute OHLCV for the symbol's research continuous series."""
    sym = symbol.upper()
    lake = resolve_lake_path(lake_path)
    if not lake.exists():
        raise FileNotFoundError(f"Futures lake not found: {lake}")

    if sym == "CL":
        try:
            frame = _load_cl_institutional_from_lake(lake)
        except Exception:
            artifact = REPO_ROOT / "artifacts/research/cl_institutional_continuous/bars_1m.parquet"
            frame = pd.read_parquet(artifact)
            frame["symbol"] = sym
    else:
        con = duckdb.connect(str(lake), read_only=True)
        try:
            frame = con.execute(FRONT_MONTH_STITCH_SQL, [sym]).fetchdf()
        finally:
            con.close()
        frame["symbol"] = sym

    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    for col in ("o", "h", "l", "c", "v"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("ts").reset_index(drop=True)


def _load_cl_institutional_from_lake(lake: Path) -> pd.DataFrame:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from k2_systematic_macro.data.volbook_adapter import (  # noqa: WPS433
        VolbookLoadSpec,
        VolbookResearchDataAdapter,
    )

    adapter = VolbookResearchDataAdapter(
        VolbookLoadSpec(
            symbol="CL",
            contract_source="institutional_continuous",
            lake_path=lake,
            front_month_guard_on_missing="mark",
        )
    )
    return adapter.load_minute_bars()


def resample_bars(minute_bars: pd.DataFrame, bar_size: str = "1D") -> pd.DataFrame:
    """Resample 1-minute bars to research frequency (1D or 4h)."""
    if minute_bars.empty:
        return minute_bars.copy()
    agg = (
        minute_bars.set_index("ts")
        .resample(bar_size)
        .agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum", "symbol": "last"})
        .dropna(subset=["c"])
        .reset_index()
    )
    agg["timeframe"] = "1d" if bar_size.upper() in {"1D", "1d", "D"} else bar_size
    return agg


def resample_daily_bars(minute_bars: pd.DataFrame) -> pd.DataFrame:
    return resample_bars(minute_bars, "1D")
