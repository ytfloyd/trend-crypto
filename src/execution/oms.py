"""Order Management System for live trading.

Translates target weights into orders, submits to a broker, and tracks
fill status. Handles reconciliation between expected and actual positions.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from common.logging import get_logger
from .broker import BrokerInterface, BrokerOrder, OrderSide, OrderStatus

logger = get_logger("oms")


@dataclass
class RebalanceResult:
    """Result of a rebalance operation.

    Attributes:
        orders_submitted: Number of orders submitted.
        orders_filled: Number of orders confirmed filled.
        orders_rejected: Number of orders rejected.
        target_weights: The target weights that were sent.
        delta_weights: The weight deltas computed.
    """

    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    target_weights: dict[str, float] = field(default_factory=dict)
    delta_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class ReconciliationReport:
    """Report from position reconciliation.

    Attributes:
        expected: Symbol → expected weight.
        actual: Symbol → actual weight (from broker).
        drifts: Symbol → drift (actual - expected).
        max_drift: Maximum absolute drift.
        is_reconciled: True if all drifts are within tolerance.
    """

    expected: dict[str, float]
    actual: dict[str, float]
    drifts: dict[str, float]
    max_drift: float
    is_reconciled: bool


class OrderManagementSystem:
    """Manages the lifecycle of orders for portfolio rebalancing.

    Args:
        broker: Broker interface for order execution.
        min_trade_notional: Minimum trade size in USD.
        deadband: Weight change threshold below which no trade is generated.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        min_trade_notional: float = 0.0,
        deadband: float = 0.01,
    ) -> None:
        self.broker = broker
        self.min_trade_notional = min_trade_notional
        self.deadband = deadband
        self._pending_orders: list[str] = []

    def rebalance_to_targets(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        nav: float,
    ) -> RebalanceResult:
        """Generate and submit orders to move from current to target weights.

        Args:
            target_weights: Symbol → target weight.
            current_weights: Symbol → current weight.
            nav: Current portfolio NAV for notional computation.

        Returns:
            RebalanceResult with order counts and diagnostics.
        """
        result = RebalanceResult(target_weights=dict(target_weights))
        all_symbols = set(target_weights) | set(current_weights)

        for symbol in sorted(all_symbols):
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            delta = target - current
            result.delta_weights[symbol] = delta

            if abs(delta) < self.deadband:
                continue

            notional = abs(delta) * nav
            if notional < self.min_trade_notional:
                continue

            price = self.broker.get_latest_price(symbol)
            if price <= 0:
                logger.warning("No price for %s, skipping order", symbol)
                result.orders_rejected += 1
                continue

            qty = notional / price
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            order = BrokerOrder(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                qty=qty,
            )
            order_id = self.broker.submit_order(order)
            result.orders_submitted += 1
            self._pending_orders.append(order_id)

            status = self.broker.get_order_status(order_id)
            if status == OrderStatus.FILLED:
                result.orders_filled += 1
            elif status == OrderStatus.REJECTED:
                result.orders_rejected += 1

        logger.info(
            "Rebalance: %d submitted, %d filled, %d rejected",
            result.orders_submitted, result.orders_filled, result.orders_rejected,
        )
        return result

    def reconcile(
        self,
        expected_weights: dict[str, float],
        nav: float,
        tolerance: float = 0.02,
    ) -> ReconciliationReport:
        """Compare broker positions against expected weights.

        Args:
            expected_weights: Symbol → expected weight.
            nav: Current NAV for weight computation.
            tolerance: Maximum acceptable drift before flagging.

        Returns:
            ReconciliationReport with drift analysis.
        """
        positions = self.broker.get_positions()
        actual_weights: dict[str, float] = {}
        if nav > 0:
            for sym, pos in positions.items():
                price = self.broker.get_latest_price(sym)
                actual_weights[sym] = (pos.qty * price) / nav

        all_symbols = set(expected_weights) | set(actual_weights)
        drifts: dict[str, float] = {}
        for sym in all_symbols:
            expected = expected_weights.get(sym, 0.0)
            actual = actual_weights.get(sym, 0.0)
            drifts[sym] = actual - expected

        max_drift = max((abs(d) for d in drifts.values()), default=0.0)
        is_reconciled = max_drift <= tolerance

        if not is_reconciled:
            logger.warning(
                "Position drift detected: max_drift=%.4f (tolerance=%.4f)",
                max_drift, tolerance,
            )

        return ReconciliationReport(
            expected=dict(expected_weights),
            actual=actual_weights,
            drifts=drifts,
            max_drift=max_drift,
            is_reconciled=is_reconciled,
        )
