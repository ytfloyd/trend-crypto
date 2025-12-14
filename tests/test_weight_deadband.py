from backtest.rebalance import rebalance_to_target_weight
from common.config import ExecutionConfig


def test_weight_deadband_blocks_small_changes():
    cfg = ExecutionConfig(weight_deadband=0.05, min_trade_notional=0.0)
    nav = 1000.0
    price = 100.0
    units = 0.1  # current weight 0.01
    order = rebalance_to_target_weight(
        target_weight=0.02,
        cash=nav - units * price,
        units=units,
        price=price,
        nav=nav,
        cfg=cfg,
        bars_since_last_trade=None,
    )
    assert order is None

