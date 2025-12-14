from backtest.rebalance import rebalance_to_target_weight
from common.config import ExecutionConfig


def test_sell_does_not_exceed_position():
    cfg = ExecutionConfig(min_trade_notional=0.0)
    units = 1.5
    order = rebalance_to_target_weight(
        target_weight=0.0,
        cash=0.0,
        units=units,
        price=200.0,
        nav=units * 200.0,
        cfg=cfg,
        bars_since_last_trade=None,
    )
    assert order is not None
    assert order.qty <= units

