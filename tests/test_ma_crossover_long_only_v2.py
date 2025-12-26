from datetime import datetime, timedelta, timezone

import polars as pl

from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy
from strategy.context import StrategyContext


def _ctx_from_prices(prices):
    rows = []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i, p in enumerate(prices):
        ts = start + timedelta(days=i)
        rows.append({"ts": ts, "symbol": "X", "open": p, "high": p, "low": p, "close": p, "volume": 1.0})
    hist = pl.DataFrame(rows)
    return StrategyContext(history=hist, decision_ts=hist[-1, "ts"])


def test_vol_scalar_clipped():
    strat = MACrossoverLongOnlyStrategy(fast=2, slow=3, target_vol_annual=0.5, vol_lookback=2, max_weight=1.0, enable_adx_filter=False)
    ctx = _ctx_from_prices([100, 101, 102, 103, 104])
    w = strat.on_bar_close(ctx)
    assert 0.0 <= w <= 1.0


def test_adx_filter_blocks_when_low():
    strat = MACrossoverLongOnlyStrategy(fast=2, slow=3, target_vol_annual=None, vol_lookback=2, max_weight=1.0, enable_adx_filter=True, adx_window=3, adx_threshold=50.0, adx_entry_only=True)
    ctx = _ctx_from_prices([100, 100, 100, 100, 100, 100])
    w = strat.on_bar_close(ctx)
    assert w == 0.0


def test_long_only():
    strat = MACrossoverLongOnlyStrategy(fast=2, slow=3, target_vol_annual=None, vol_lookback=2, max_weight=1.0, enable_adx_filter=False)
    ctx = _ctx_from_prices([1, 2, 3, 4, 5, 6])
    w = strat.on_bar_close(ctx)
    assert w >= 0.0


def test_adx_entry_only_persists_when_in_pos():
    strat = MACrossoverLongOnlyStrategy(
        fast=2,
        slow=3,
        target_vol_annual=None,
        vol_lookback=2,
        max_weight=1.0,
        enable_adx_filter=True,
        adx_window=3,
        adx_threshold=50.0,
        adx_entry_only=True,
    )
    # first context with flat prices -> adx low -> should block entry
    ctx1 = _ctx_from_prices([100, 100, 100, 100, 100, 100])
    w1 = strat.on_bar_close(ctx1)
    assert w1 == 0.0
    # now rising prices boost ADX and allow entry
    ctx2 = _ctx_from_prices([100, 101, 102, 103, 104, 105, 106, 107, 108])
    w2 = strat.on_bar_close(ctx2)
    assert w2 > 0.0
    # drop ADX again but MA still long; should stay in position (no forced exit)
    ctx3 = _ctx_from_prices([108, 108, 108, 108, 108, 109, 109, 109, 109, 110])
    w3 = strat.on_bar_close(ctx3)
    assert w3 > 0.0

