"""Options data collection via Interactive Brokers API.

Provides infrastructure for:
    - Connecting to IB TWS / Gateway
    - Fetching option chains (strikes, expiries, multipliers)
    - Snapshotting implied vol surfaces
    - Storing option data in DuckDB for backtesting and research
    - Building VolSurface objects from stored snapshots

Requires: ib_insync (pip install ib_insync)
"""
from __future__ import annotations

from .schema import OptionsSchema
from .chains import IBOptionChainFetcher
from .snapshot import IBVolSurfaceCollector

__all__ = [
    "OptionsSchema",
    "IBOptionChainFetcher",
    "IBVolSurfaceCollector",
]
