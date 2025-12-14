from backtest.rebalance import rebalance_to_target_weight
from common.config import ExecutionConfig


def test_min_rebalance_notional_fraction_blocks_small_trade():
    nav = 1000.0
    price = 100.0
    cfg = ExecutionConfig(min_rebalance_notional_frac=0.05, min_trade_notional=0.0)
    order = rebalance_to_target_weight(
        target_weight=0.01,  # $10
        cash=nav,
        units=0.0,
        price=price,
        nav=nav,
        cfg=cfg,
        bars_since_last_trade=None,
    )
    assert order is None

