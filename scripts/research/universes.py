#!/usr/bin/env python
from __future__ import annotations

from typing import List

import duckdb
import pandas as pd

KUMA_LIVE_UNIVERSE_CANDIDATES = [
    "BTC-USD",
    "ETH-USD",
    "LTC-USD",
    "BCH-USD",
    "EOS-USD",
    "OXT-USD",
    "XLM-USD",
    "XTZ-USD",
    "ETC-USD",
    "LINK-USD",
    "REP-USD",
    "ZRX-USD",
    "KNC-USD",
    "DASH-USD",
    "MKR-USD",
    "ATOM-USD",
    "OMG-USD",
    "ALGO-USD",
    "COMP-USD",
    "BAND-USD",
    "NMR-USD",
    "CGLD-USD",
    "UMA-USD",
    "LRC-USD",
    "YFI-USD",
    "UNI-USD",
    "REN-USD",
]


def resolve_universe_from_db(
    db_path: str,
    table: str,
    candidates: List[str],
    *,
    lookback_days: int = 7,
) -> List[str]:
    if not candidates:
        raise ValueError("Candidates list is empty.")

    con = duckdb.connect(db_path, read_only=True)
    try:
        con.execute("SET TimeZone='UTC'")
        max_ts = con.execute(f"SELECT MAX(ts) AS max_ts FROM {table}").fetchone()[0]
        if max_ts is None:
            raise ValueError(f"No timestamps found in table {table}.")

        cutoff = pd.to_datetime(max_ts, utc=True) - pd.Timedelta(days=lookback_days)
        placeholders = ",".join(["?"] * len(candidates))
        rows = con.execute(
            f"""
            SELECT DISTINCT symbol
            FROM {table}
            WHERE symbol IN ({placeholders})
              AND ts >= ?
              AND ts <= ?
            """,
            candidates + [cutoff.to_pydatetime(), max_ts],
        ).fetchall()
    finally:
        con.close()

    resolved = sorted([r[0] for r in rows])
    if not resolved:
        raise ValueError(
            "No symbols resolved from DB. Check db/table, candidates, and lookback window."
        )
    return resolved


def get_universe(
    name: str,
    *,
    db_path: str | None = None,
    table: str | None = None,
    lookback_days: int = 7,
) -> List[str]:
    if name == "kuma_live_universe":
        if not db_path or not table:
            raise ValueError("db_path and table are required for kuma_live_universe.")
        return resolve_universe_from_db(
            db_path,
            table,
            KUMA_LIVE_UNIVERSE_CANDIDATES,
            lookback_days=lookback_days,
        )
    if name == "kuma_live_universe_candidates":
        return KUMA_LIVE_UNIVERSE_CANDIDATES

    available = ["kuma_live_universe", "kuma_live_universe_candidates"]
    raise ValueError(f"Unknown universe '{name}'. Available: {available}")
