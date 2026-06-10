"""Adapters that connect the convexity pipeline to existing infrastructure.

`data_provider`  - (symbol, bar_frequency) -> OHLCV DataFrame over the DuckDB
                   lakes (crypto / ETF) and the continuous-futures parquet
                   artifacts.
`existing_engine_adapter` - wraps `scripts/research/common/backtest.py::simple_backtest`
                   and produces fully-populated `BacktestResult` objects for
                   every variant the runner emits.
"""
from .data_provider import LakeDataProvider
from .existing_engine_adapter import ExistingEngineAdapter

__all__ = ["LakeDataProvider", "ExistingEngineAdapter"]
