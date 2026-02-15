"""Tests for Phase 4: Live Trading Bridge.

Covers:
- PaperBroker: order fills, positions, balances
- OMS: rebalancing, reconciliation
- ReplayDataFeed: bar advancement, history
- LiveRunner: end-to-end paper trading replay
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from common.config import RiskConfigResolved
from data.feed import BarData, ReplayDataFeed
from execution.broker import BrokerOrder, OrderSide, OrderStatus
from execution.oms import OrderManagementSystem
from execution.paper_broker import PaperBroker
from live.runner import LiveRunner
from risk.risk_manager import RiskManager
from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy


def _make_bars(
    n: int, symbol: str = "BTC-USD", base_price: float = 1000.0
) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = base_price
    for i in range(n):
        ts = start + timedelta(hours=i)
        delta = 0.5 * (1 if (i * 7 + 3) % 5 < 3 else -1)
        o = price
        c = price + delta
        rows.append({
            "ts": ts, "symbol": symbol,
            "open": o, "high": max(o, c) + 0.1,
            "low": min(o, c) - 0.1, "close": c,
            "volume": 1000.0 + i * 10.0,
        })
        price = c
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------


class TestPaperBroker:
    def test_buy_order_fills(self) -> None:
        broker = PaperBroker(fee_bps=10.0, slippage_bps=5.0, initial_cash=100_000.0)
        broker.set_price("BTC-USD", 50_000.0)
        order = BrokerOrder(
            order_id="test-1", symbol="BTC-USD", side=OrderSide.BUY, qty=0.1,
        )
        order_id = broker.submit_order(order)
        assert broker.get_order_status(order_id) == OrderStatus.FILLED
        fills = broker.get_fills(order_id)
        assert len(fills) == 1
        assert fills[0].qty == 0.1
        assert fills[0].fill_price > 50_000.0  # Slippage applied

    def test_sell_order_fills(self) -> None:
        broker = PaperBroker(fee_bps=0.0, slippage_bps=0.0, initial_cash=100_000.0)
        broker.set_price("BTC-USD", 50_000.0)
        # Buy first
        broker.submit_order(BrokerOrder(
            order_id="buy-1", symbol="BTC-USD", side=OrderSide.BUY, qty=1.0,
        ))
        # Then sell
        broker.submit_order(BrokerOrder(
            order_id="sell-1", symbol="BTC-USD", side=OrderSide.SELL, qty=0.5,
        ))
        positions = broker.get_positions()
        assert abs(positions["BTC-USD"].qty - 0.5) < 1e-10

    def test_nav_computation(self) -> None:
        broker = PaperBroker(fee_bps=0.0, slippage_bps=0.0, initial_cash=100_000.0)
        broker.set_price("BTC-USD", 50_000.0)
        broker.submit_order(BrokerOrder(
            order_id="buy-1", symbol="BTC-USD", side=OrderSide.BUY, qty=1.0,
        ))
        # NAV = cash (100k - 50k) + position (1 * 50k) = 100k
        assert abs(broker.nav() - 100_000.0) < 0.01

    def test_rejected_order_no_price(self) -> None:
        broker = PaperBroker()
        order = BrokerOrder(
            order_id="test-1", symbol="UNKNOWN", side=OrderSide.BUY, qty=1.0,
        )
        order_id = broker.submit_order(order)
        assert broker.get_order_status(order_id) == OrderStatus.REJECTED

    def test_balances(self) -> None:
        broker = PaperBroker(initial_cash=50_000.0, fee_bps=0.0, slippage_bps=0.0)
        broker.set_price("BTC-USD", 10_000.0)
        broker.submit_order(BrokerOrder(
            order_id="b1", symbol="BTC-USD", side=OrderSide.BUY, qty=2.0,
        ))
        balances = broker.get_balances()
        assert abs(balances["USD"] - 30_000.0) < 0.01
        assert abs(balances["BTC-USD"] - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# OMS
# ---------------------------------------------------------------------------


class TestOMS:
    def test_rebalance_generates_orders(self) -> None:
        broker = PaperBroker(fee_bps=0.0, slippage_bps=0.0, initial_cash=100_000.0)
        broker.set_price("BTC-USD", 50_000.0)
        oms = OrderManagementSystem(broker, deadband=0.01)
        result = oms.rebalance_to_targets(
            target_weights={"BTC-USD": 0.5},
            current_weights={"BTC-USD": 0.0},
            nav=100_000.0,
        )
        assert result.orders_submitted == 1
        assert result.orders_filled == 1

    def test_deadband_prevents_small_trades(self) -> None:
        broker = PaperBroker(initial_cash=100_000.0)
        broker.set_price("BTC-USD", 50_000.0)
        oms = OrderManagementSystem(broker, deadband=0.05)
        result = oms.rebalance_to_targets(
            target_weights={"BTC-USD": 0.03},
            current_weights={"BTC-USD": 0.0},
            nav=100_000.0,
        )
        assert result.orders_submitted == 0

    def test_reconciliation_clean(self) -> None:
        broker = PaperBroker(fee_bps=0.0, slippage_bps=0.0, initial_cash=100_000.0)
        broker.set_price("BTC-USD", 50_000.0)
        # Buy to reach ~50% weight
        broker.submit_order(BrokerOrder(
            order_id="b1", symbol="BTC-USD", side=OrderSide.BUY, qty=1.0,
        ))
        oms = OrderManagementSystem(broker)
        report = oms.reconcile(
            expected_weights={"BTC-USD": 0.5},
            nav=100_000.0,
            tolerance=0.05,
        )
        assert report.is_reconciled

    def test_reconciliation_drift(self) -> None:
        broker = PaperBroker(fee_bps=0.0, slippage_bps=0.0, initial_cash=100_000.0)
        broker.set_price("BTC-USD", 50_000.0)
        oms = OrderManagementSystem(broker)
        # Expect 50% weight but have 0% (no position)
        report = oms.reconcile(
            expected_weights={"BTC-USD": 0.5},
            nav=100_000.0,
            tolerance=0.05,
        )
        assert not report.is_reconciled
        assert report.max_drift >= 0.5


# ---------------------------------------------------------------------------
# ReplayDataFeed
# ---------------------------------------------------------------------------


class TestReplayDataFeed:
    def test_get_latest_bar(self) -> None:
        bars = _make_bars(10, "BTC-USD")
        feed = ReplayDataFeed({"BTC-USD": bars})
        bar = feed.get_latest_bar("BTC-USD")
        assert bar is not None
        assert isinstance(bar, BarData)
        assert bar.symbol == "BTC-USD"

    def test_advance_moves_cursor(self) -> None:
        bars = _make_bars(10, "BTC-USD")
        feed = ReplayDataFeed({"BTC-USD": bars})
        price_0 = feed.get_latest_price("BTC-USD")
        feed.advance("BTC-USD")
        price_1 = feed.get_latest_price("BTC-USD")
        # Prices should differ (synthetic bars have a trend)
        assert price_0 != price_1

    def test_history_returns_correct_length(self) -> None:
        bars = _make_bars(20, "BTC-USD")
        feed = ReplayDataFeed({"BTC-USD": bars})
        # Advance to bar 10
        for _ in range(10):
            feed.advance("BTC-USD")
        history = feed.get_history("BTC-USD", 5)
        assert history.height == 5

    def test_exhaustion(self) -> None:
        bars = _make_bars(5, "BTC-USD")
        feed = ReplayDataFeed({"BTC-USD": bars})
        assert not feed.is_exhausted
        for _ in range(4):
            feed.advance_all()
        assert feed.is_exhausted


# ---------------------------------------------------------------------------
# LiveRunner end-to-end
# ---------------------------------------------------------------------------


class TestLiveRunner:
    def test_paper_trading_replay(self) -> None:
        """End-to-end paper trading with replay feed."""
        n = 60
        bars = _make_bars(n, "BTC-USD", base_price=1000.0)
        feed = ReplayDataFeed({"BTC-USD": bars})
        broker = PaperBroker(
            fee_bps=10.0, slippage_bps=5.0, initial_cash=100_000.0,
        )
        strategy = MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0)
        risk_cfg = RiskConfigResolved(
            vol_window=10, target_vol_annual=None, max_weight=1.0, window_units="bars",
        )
        rm = RiskManager(cfg=risk_cfg, periods_per_year=8760.0)
        oms = OrderManagementSystem(broker, deadband=0.01)

        runner = LiveRunner(
            symbols=["BTC-USD"],
            strategies={"BTC-USD": strategy},
            risk_manager=rm,
            broker=broker,
            feed=feed,
            oms=oms,
            lookback=50,
        )

        # Run replay with 30 cycles (needs warmup for MA)
        results = runner.run_replay(max_cycles=30)
        assert len(results) == 30
        for r in results:
            assert r.nav > 0
            assert "BTC-USD" in r.target_weights
            assert r.rebalance is not None

    def test_single_cycle(self) -> None:
        bars = _make_bars(30, "BTC-USD", base_price=500.0)
        feed = ReplayDataFeed({"BTC-USD": bars})
        # Advance past warmup
        for _ in range(25):
            feed.advance_all()
        broker = PaperBroker(initial_cash=50_000.0)
        broker.set_price("BTC-USD", feed.get_latest_price("BTC-USD"))
        strategy = MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0)
        risk_cfg = RiskConfigResolved(
            vol_window=10, target_vol_annual=None, max_weight=1.0, window_units="bars",
        )
        rm = RiskManager(cfg=risk_cfg, periods_per_year=8760.0)
        oms = OrderManagementSystem(broker, deadband=0.01)

        runner = LiveRunner(
            symbols=["BTC-USD"],
            strategies={"BTC-USD": strategy},
            risk_manager=rm,
            broker=broker,
            feed=feed,
            oms=oms,
        )
        result = runner.run_once()
        assert result.nav > 0
        assert len(runner.cycle_history) == 1
