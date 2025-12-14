from datetime import datetime, timezone

from backtest.rebalance import rebalance_to_target_weight
from common.config import ExecutionConfig
from execution.sim import ExecutionSim


def test_cash_buffer_prevents_negative_cash():
    cfg = ExecutionConfig(fee_bps=100, slippage_bps=500, min_trade_notional=0.0)
    price = 10.0
    cash = 100.0
    nav = cash

    order = rebalance_to_target_weight(
        target_weight=1.0,
        cash=cash,
        units=0.0,
        price=price,
        nav=nav,
        cfg=cfg,
        bars_since_last_trade=None,
    )
    assert order is not None

    fill = ExecutionSim(fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps).fill_order(
        order, ts=datetime.now(timezone.utc)
    )
    total_cost = fill.notional + fill.fee
    assert total_cost <= cash + 1e-9

