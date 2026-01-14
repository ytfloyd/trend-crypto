import pytest
pytest.importorskip("polars")

from datetime import datetime, timedelta, timezone

import polars as pl

from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy
from strategy.context import StrategyContext


def _ctx(prices):
    rows = []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i, p in enumerate(prices):
        ts = start + timedelta(days=i)
        rows.append({"ts": ts, "symbol": "X", "open": p, "high": p, "low": p, "close": p, "volume": 1.0})
    hist = pl.DataFrame(rows)
    return StrategyContext(history=hist, decision_ts=hist[-1, "ts"])


def test_vol_target_changes_weight():
    prices = [100, 110, 90, 120, 80, 130, 70, 140, 60, 150, 160, 170]
    ctx = _ctx(prices)
    strat_low = MACrossoverLongOnlyStrategy(
        fast=2,
        slow=3,
        weight_on=1.0,
        target_vol_annual=0.40,
        vol_lookback=5,
        max_weight=1.0,
        enable_adx_filter=False,
    )
    strat_high = MACrossoverLongOnlyStrategy(
        fast=2,
        slow=3,
        weight_on=1.0,
        target_vol_annual=0.50,
        vol_lookback=5,
        max_weight=1.0,
        enable_adx_filter=False,
    )
    w_low = strat_low.on_bar_close(ctx)
    w_high = strat_high.on_bar_close(ctx)
    assert w_low >= 0.0 and w_low <= 1.0
    assert w_high >= 0.0 and w_high <= 1.0
    assert w_high > w_low

