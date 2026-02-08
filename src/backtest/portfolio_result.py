"""Multi-asset portfolio backtest result container."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class PortfolioResult:
    """Container for multi-asset portfolio backtest output.

    Attributes:
        equity_df: Per-bar NAV, gross/net returns, drawdown.
            Columns: ts, nav, gross_ret, net_ret, turnover, cost_ret, dd
        weights_df: Per-bar target and held weights per symbol.
            Columns: ts, symbol, target_weight, held_weight
        contributions_df: Per-bar return contribution per symbol.
            Columns: ts, symbol, contribution
        trades_df: Per-bar turnover per symbol.
            Columns: ts, symbol, turnover, cost_ret
        summary: Aggregate performance statistics.
    """

    equity_df: pl.DataFrame
    weights_df: pl.DataFrame
    contributions_df: pl.DataFrame
    trades_df: pl.DataFrame
    summary: dict[str, object]
