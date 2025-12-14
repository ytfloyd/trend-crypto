from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .types import Fill, Order, Side


@dataclass
class ExecutionSim:
    fee_bps: float = 0.0
    slippage_bps: float = 0.0

    def fill_order(self, order: Order, ts: datetime) -> Fill:
        """
        Simulate a single market order fill at the provided timestamp.
        """
        slip = self.slippage_bps / 10_000
        fee_rate = self.fee_bps / 10_000
        if order.side == Side.BUY:
            fill_price = order.ref_price * (1 + slip)
        else:
            fill_price = order.ref_price * (1 - slip)
        notional = fill_price * order.qty
        fee = notional * fee_rate
        return Fill(
            ts=ts,
            side=order.side,
            qty=order.qty,
            ref_price=order.ref_price,
            fill_price=fill_price,
            notional=notional,
            fee=fee,
        )

