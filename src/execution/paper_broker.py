"""Paper (simulated) broker for testing live trading workflows.

Fills all orders immediately at the given price with configurable
fees and slippage. No real exchange interaction.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from common.logging import get_logger
from .broker import (
    BrokerFill,
    BrokerInterface,
    BrokerOrder,
    OrderSide,
    OrderStatus,
    Position,
)

logger = get_logger("paper_broker")


@dataclass
class PaperBroker(BrokerInterface):
    """Simulated broker for paper trading.

    All orders are filled instantly at the reference price with
    configurable slippage and fees.

    Attributes:
        fee_bps: Fee rate in basis points.
        slippage_bps: Slippage in basis points.
        initial_cash: Starting USD balance.
    """

    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_cash: float = 100_000.0
    _prices: dict[str, float] = field(default_factory=dict)
    _positions: dict[str, Position] = field(default_factory=dict)
    _cash: float = 0.0
    _orders: dict[str, BrokerOrder] = field(default_factory=dict)
    _fills: dict[str, list[BrokerFill]] = field(default_factory=dict)
    _order_status: dict[str, OrderStatus] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._cash = self.initial_cash

    def set_price(self, symbol: str, price: float) -> None:
        """Set the current price for a symbol (used by data feed)."""
        self._prices[symbol] = price

    def submit_order(self, order: BrokerOrder) -> str:
        """Immediately fill the order at current price with slippage."""
        order_id = order.order_id or str(uuid.uuid4())
        self._orders[order_id] = order

        price = self._prices.get(order.symbol, 0.0)
        if price <= 0:
            self._order_status[order_id] = OrderStatus.REJECTED
            logger.warning("Order rejected: no price for %s", order.symbol)
            return order_id

        # Apply slippage
        slip = self.slippage_bps / 10_000
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + slip)
        else:
            fill_price = price * (1 - slip)

        notional = fill_price * order.qty
        fee = notional * (self.fee_bps / 10_000)

        # Update cash and position
        if order.side == OrderSide.BUY:
            total_cost = notional + fee
            self._cash -= total_cost
            pos = self._positions.get(order.symbol, Position(symbol=order.symbol))
            old_cost = pos.avg_cost * pos.qty
            pos.qty += order.qty
            pos.avg_cost = (old_cost + notional) / pos.qty if pos.qty > 0 else 0.0
            self._positions[order.symbol] = pos
        else:
            proceeds = notional - fee
            self._cash += proceeds
            pos = self._positions.get(order.symbol, Position(symbol=order.symbol))
            pos.qty -= order.qty
            if pos.qty <= 0:
                pos.qty = 0.0
                pos.avg_cost = 0.0
            self._positions[order.symbol] = pos

        fill = BrokerFill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            fill_price=fill_price,
            fee=fee,
            ts=datetime.now(timezone.utc),
        )
        self._fills.setdefault(order_id, []).append(fill)
        self._order_status[order_id] = OrderStatus.FILLED
        logger.info(
            "Paper fill: %s %s %.6f @ %.2f (fee=%.4f)",
            order.side.value, order.symbol, order.qty, fill_price, fee,
        )
        return order_id

    def get_order_status(self, order_id: str) -> OrderStatus:
        return self._order_status.get(order_id, OrderStatus.PENDING)

    def get_fills(self, order_id: str) -> list[BrokerFill]:
        return self._fills.get(order_id, [])

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_balances(self) -> dict[str, float]:
        balances: dict[str, float] = {"USD": self._cash}
        for sym, pos in self._positions.items():
            if pos.qty > 0:
                balances[sym] = pos.qty
        return balances

    def get_latest_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    def nav(self) -> float:
        """Compute current net asset value."""
        total = self._cash
        for sym, pos in self._positions.items():
            price = self._prices.get(sym, 0.0)
            total += pos.qty * price
        return total
