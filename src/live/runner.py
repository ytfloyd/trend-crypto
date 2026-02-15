"""Live trading runner.

Orchestrates the trading loop: fetch data → compute signals → apply risk →
submit orders → reconcile. Supports both single-cycle and continuous modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from common.logging import get_logger
from data.feed import DataFeed
from execution.broker import BrokerInterface
from execution.oms import OrderManagementSystem, RebalanceResult, ReconciliationReport
from risk.risk_manager import RiskManager
from strategy.base import TargetWeightStrategy
from strategy.context import make_strategy_context

logger = get_logger("live_runner")


@dataclass
class CycleResult:
    """Result of a single trading cycle.

    Attributes:
        ts: Timestamp of the cycle.
        target_weights: Computed target weights.
        rebalance: Rebalance result (orders submitted/filled).
        reconciliation: Post-trade reconciliation report.
        nav: Portfolio NAV at cycle time.
    """

    ts: datetime
    target_weights: dict[str, float]
    rebalance: Optional[RebalanceResult] = None
    reconciliation: Optional[ReconciliationReport] = None
    nav: float = 0.0


class LiveRunner:
    """Orchestrates live/paper trading cycles.

    Args:
        symbols: Symbols to trade.
        strategies: Per-symbol strategy instances.
        risk_manager: Risk manager for position sizing.
        broker: Broker for order execution.
        feed: Data feed for market data.
        oms: Order management system.
        lookback: History lookback in bars.
    """

    def __init__(
        self,
        symbols: list[str],
        strategies: dict[str, TargetWeightStrategy],
        risk_manager: RiskManager,
        broker: BrokerInterface,
        feed: DataFeed,
        oms: OrderManagementSystem,
        lookback: Optional[int] = None,
    ) -> None:
        self.symbols = symbols
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.broker = broker
        self.feed = feed
        self.oms = oms
        self.lookback = lookback
        self._current_weights: dict[str, float] = {s: 0.0 for s in symbols}
        self._cycle_history: list[CycleResult] = []

    def run_once(self) -> CycleResult:
        """Execute a single trading cycle.

        1. Fetch latest data for all symbols.
        2. Compute strategy signals.
        3. Apply risk management.
        4. Rebalance portfolio via OMS.
        5. Reconcile positions.
        """
        ts = datetime.now(timezone.utc)
        target_weights: dict[str, float] = {}

        # Step 1-3: Compute target weights for each symbol
        for symbol in self.symbols:
            strategy = self.strategies.get(symbol)
            if strategy is None:
                target_weights[symbol] = 0.0
                continue

            history = self.feed.get_history(symbol, self.lookback or 200)
            if history.is_empty() or history.height < 2:
                target_weights[symbol] = 0.0
                continue

            # Build context from the latest bar
            idx = history.height - 1
            ctx = make_strategy_context(history, idx, self.lookback)
            raw_weight = strategy.on_bar_close(ctx)
            scaled_weight = self.risk_manager.apply(raw_weight, history)
            target_weights[symbol] = scaled_weight

        # Step 4: Compute NAV and rebalance
        nav = _compute_nav(self.broker, self.symbols)
        rebalance_result = self.oms.rebalance_to_targets(
            target_weights, self._current_weights, nav,
        )

        # Step 5: Reconcile
        reconciliation = self.oms.reconcile(target_weights, nav)
        self._current_weights = dict(target_weights)

        cycle = CycleResult(
            ts=ts,
            target_weights=target_weights,
            rebalance=rebalance_result,
            reconciliation=reconciliation,
            nav=nav,
        )
        self._cycle_history.append(cycle)
        logger.info(
            "Cycle complete: NAV=%.2f, %d orders, reconciled=%s",
            nav, rebalance_result.orders_submitted, reconciliation.is_reconciled,
        )
        return cycle

    def run_replay(self, max_cycles: Optional[int] = None) -> list[CycleResult]:
        """Run continuous cycles on a replay data feed until exhausted.

        Args:
            max_cycles: Maximum number of cycles to run (safety limit).

        Returns:
            List of CycleResult from each cycle.
        """

        results: list[CycleResult] = []
        count = 0
        while True:
            if max_cycles is not None and count >= max_cycles:
                break
            if hasattr(self.feed, "is_exhausted") and self.feed.is_exhausted:
                break

            # Update broker prices from feed
            for sym in self.symbols:
                price = self.feed.get_latest_price(sym)
                if hasattr(self.broker, "set_price"):
                    self.broker.set_price(sym, price)

            result = self.run_once()
            results.append(result)
            count += 1

            # Advance the replay feed
            if hasattr(self.feed, "advance_all"):
                self.feed.advance_all()

        logger.info("Replay completed: %d cycles", len(results))
        return results

    @property
    def cycle_history(self) -> list[CycleResult]:
        return list(self._cycle_history)


def _compute_nav(broker: BrokerInterface, symbols: list[str]) -> float:
    """Compute current NAV from broker balances and positions."""
    balances = broker.get_balances()
    nav = balances.get("USD", 0.0)
    positions = broker.get_positions()
    for sym in symbols:
        pos = positions.get(sym)
        if pos is not None and pos.qty > 0:
            price = broker.get_latest_price(sym)
            nav += pos.qty * price
    return nav
