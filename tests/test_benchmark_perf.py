"""Performance benchmark and vectorized-vs-loop equivalence tests.

These tests verify:
1. Engine processes N bars within a wall-clock budget.
2. Vectorized signal computation produces identical results to the per-bar loop.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from backtest.engine import BacktestEngine
from common.config import (
    DataConfig,
    EngineConfig,
    ExecutionConfig,
    RiskConfigRaw,
    RunConfigRaw,
    StrategyConfigRaw,
    compile_config,
)
from data.portal import DataPortal
from risk.risk_manager import RiskManager
from strategy.buy_and_hold import BuyAndHoldStrategy
from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy
from strategy.base import StrategySignals
from strategy.context import make_strategy_context


class DummyPortal(DataPortal):
    def __init__(self, bars: pl.DataFrame) -> None:
        self._bars = bars

    def load_bars(self) -> pl.DataFrame:
        return self._bars


def _synthetic_bars(n: int, base_price: float = 100.0) -> pl.DataFrame:
    """Create synthetic bars with realistic price continuity."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = base_price
    for i in range(n):
        ts = start + timedelta(hours=i)
        # Small random-ish walk using a deterministic pattern
        delta = 0.5 * (1 if (i * 7 + 3) % 5 < 3 else -1)
        o = price
        c = price + delta
        rows.append(
            {
                "ts": ts,
                "symbol": "BTC-USD",
                "open": o,
                "high": max(o, c) + 0.1,
                "low": min(o, c) - 0.1,
                "close": c,
                "volume": 1000.0 + i * 10.0,
            }
        )
        price = c
    return pl.DataFrame(rows)


def _make_engine(
    bars: pl.DataFrame,
    strategy_mode: str = "buy_and_hold",
    fast: int = 5,
    slow: int = 20,
) -> BacktestEngine:
    """Build a BacktestEngine with the given strategy and synthetic bars."""
    raw_cfg = RunConfigRaw(
        run_name="perf_bench",
        data=DataConfig(
            db_path=":memory:",
            table="bars",
            symbol="BTC-USD",
            start=bars[0, "ts"],
            end=bars[bars.height - 1, "ts"],
            timeframe="1h",
        ),
        engine=EngineConfig(strict_validation=False, lookback=None, initial_cash=100_000.0),
        strategy=StrategyConfigRaw(
            mode=strategy_mode,
            fast=fast if strategy_mode != "buy_and_hold" else None,
            slow=slow if strategy_mode != "buy_and_hold" else None,
            weight_on=1.0,
            window_units="bars",
        ),
        risk=RiskConfigRaw(vol_window=10, target_vol_annual=None, max_weight=1.0, window_units="bars"),
        execution=ExecutionConfig(
            fee_bps=10.0,
            slippage_bps=5.0,
            execution_lag_bars=1,
            rebalance_deadband=0.01,
        ),
    )
    cfg = compile_config(raw_cfg)

    if strategy_mode == "buy_and_hold":
        strategy = BuyAndHoldStrategy(cfg.strategy)
    else:
        strategy = MACrossoverLongOnlyStrategy(
            fast=cfg.strategy.fast,
            slow=cfg.strategy.slow,
            max_weight=cfg.strategy.max_weight,
        )

    rm = RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor)
    return BacktestEngine(cfg, strategy, rm, DummyPortal(bars))


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------


def test_engine_500_bars_buy_and_hold():
    """Buy-and-hold engine should process 500 bars quickly."""
    bars = _synthetic_bars(500)
    engine = _make_engine(bars, strategy_mode="buy_and_hold")
    start = time.perf_counter()
    portfolio, summary = engine.run()
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"Engine took {elapsed:.3f}s for 500 bars (buy_and_hold)"
    assert summary["total_return"] != 0.0


def test_engine_500_bars_ma_crossover():
    """MA crossover engine should process 500 bars within budget."""
    bars = _synthetic_bars(500, base_price=1000.0)
    engine = _make_engine(bars, strategy_mode="ma_crossover_long_only", fast=5, slow=20)
    start = time.perf_counter()
    portfolio, summary = engine.run()
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"Engine took {elapsed:.3f}s for 500 bars (ma_crossover)"


# ---------------------------------------------------------------------------
# Vectorized-vs-loop equivalence
# ---------------------------------------------------------------------------


def test_buy_and_hold_vectorized_equivalence():
    """BuyAndHoldStrategy vectorized path must match loop path."""
    bars = _synthetic_bars(100)
    raw_cfg = RunConfigRaw(
        run_name="vec_test",
        data=DataConfig(
            db_path=":memory:", table="bars", symbol="BTC-USD",
            start=bars[0, "ts"], end=bars[bars.height - 1, "ts"], timeframe="1h",
        ),
        engine=EngineConfig(strict_validation=False, lookback=None, initial_cash=100_000.0),
        strategy=StrategyConfigRaw(mode="buy_and_hold", weight_on=1.0, window_units="bars"),
        risk=RiskConfigRaw(vol_window=10, target_vol_annual=None, max_weight=1.0, window_units="bars"),
        execution=ExecutionConfig(fee_bps=0.0, slippage_bps=0.0, execution_lag_bars=1),
    )
    cfg = compile_config(raw_cfg)
    strategy = BuyAndHoldStrategy(cfg.strategy)

    # Loop path
    loop_weights = []
    for i in range(bars.height):
        ctx = make_strategy_context(bars, i, None)
        w = strategy.on_bar_close(ctx)
        loop_weights.append(w)

    # Vectorized path
    vec_df = strategy.compute_signals_vectorized(bars, None)
    vec_weights = vec_df["target_weight"].to_list()

    assert len(loop_weights) == len(vec_weights)
    for i in range(len(loop_weights)):
        assert abs(loop_weights[i] - vec_weights[i]) < 1e-12, (
            f"Mismatch at bar {i}: loop={loop_weights[i]}, vec={vec_weights[i]}"
        )


def test_ma_crossover_vectorized_equivalence():
    """MACrossoverLongOnlyStrategy vectorized path must match loop path.

    The default compute_signals_vectorized falls back to the per-bar loop,
    so results must be identical. This test serves as a regression guard
    for when a native vectorized override is added later.
    """
    bars = _synthetic_bars(200, base_price=1000.0)
    strategy = MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0)

    # Loop path
    loop_weights = []
    loop_signals = []
    for i in range(bars.height):
        ctx = make_strategy_context(bars, i, None)
        w = strategy.on_bar_close(ctx)
        loop_weights.append(w)
        loop_signals.append(strategy.get_last_signals())

    # Reset strategy state for vectorized run
    strategy2 = MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0)
    vec_df = strategy2.compute_signals_vectorized(bars, None)
    vec_weights = vec_df["target_weight"].to_list()

    assert len(loop_weights) == len(vec_weights)
    for i in range(len(loop_weights)):
        assert abs(loop_weights[i] - vec_weights[i]) < 1e-12, (
            f"Weight mismatch at bar {i}: loop={loop_weights[i]}, vec={vec_weights[i]}"
        )


def test_strategy_signals_typed():
    """StrategySignals dataclass should be properly typed and frozen."""
    sig = StrategySignals(target_weight=0.75, vol_scalar=0.8, ma_signal=True, in_pos=True)
    assert sig.target_weight == 0.75
    assert sig.vol_scalar == 0.8
    assert sig.ma_signal is True
    assert sig.in_pos is True

    # Frozen: should not allow mutation
    with pytest.raises(AttributeError):
        sig.target_weight = 0.5  # type: ignore[misc]


def test_get_last_signals_ma_crossover():
    """MACrossoverLongOnlyStrategy.get_last_signals returns typed signals."""
    bars = _synthetic_bars(50, base_price=1000.0)
    strategy = MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0)

    ctx = make_strategy_context(bars, 49, None)
    w = strategy.on_bar_close(ctx)
    signals = strategy.get_last_signals()

    assert isinstance(signals, StrategySignals)
    assert isinstance(signals.vol_scalar, (float, type(None)))
    assert isinstance(signals.ma_signal, bool)
    assert isinstance(signals.adx_pass, bool)
    assert isinstance(signals.in_pos, bool)
