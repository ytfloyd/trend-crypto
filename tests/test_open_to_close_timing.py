"""Test that open-to-close returns (Model B) are correctly applied.

This test locks in the correct timing model:
- Decision at close[t-1]
- Execution at open[t]
- Earn return from open[t] to close[t]

The old "double shift" bug would have the strategy earn close[t-1] to close[t],
which incorrectly assumes the strategy can execute at close[t-1] instantly.
"""
import pytest
pytest.importorskip("polars")

from datetime import datetime, timezone
import polars as pl

from backtest.engine import BacktestEngine, compute_asset_returns
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
from strategy.base import TargetWeightStrategy


class DummyPortal(DataPortal):
    def __init__(self, bars: pl.DataFrame) -> None:
        self._bars = bars

    def load_bars(self) -> pl.DataFrame:
        return self._bars


class AlwaysLong(TargetWeightStrategy):
    """Strategy that immediately goes long and holds."""
    def on_bar_close(self, ctx):  # type: ignore[override]
        return 1.0


def test_compute_asset_returns_open_to_close():
    """Test open-to-close return calculation."""
    bars = pl.DataFrame({
        "ts": [datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc) for i in range(4)],
        "symbol": ["BTC-USD"] * 4,
        "open": [100.0, 105.0, 110.0, 108.0],
        "close": [105.0, 110.0, 108.0, 112.0],
        "high": [106.0, 111.0, 111.0, 113.0],
        "low": [99.0, 104.0, 107.0, 107.0],
        "volume": [1000.0] * 4,
    })
    
    asset_rets, used_fallback = compute_asset_returns(bars, mode="open_to_close")
    
    assert not used_fallback, "Should use open-to-close when open column exists"
    assert asset_rets[0] == 0.0, "First bar should have 0 return"
    assert asset_rets[1] == pytest.approx(110.0 / 105.0 - 1.0), "t=1: close/open - 1"
    assert asset_rets[2] == pytest.approx(108.0 / 110.0 - 1.0), "t=2: close/open - 1"
    assert asset_rets[3] == pytest.approx(112.0 / 108.0 - 1.0), "t=3: close/open - 1"


def test_compute_asset_returns_fallback_to_close_to_close():
    """Test fallback to close-to-close when open is missing."""
    bars = pl.DataFrame({
        "ts": [datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc) for i in range(4)],
        "symbol": ["BTC-USD"] * 4,
        "close": [100.0, 105.0, 110.0, 108.0],
        "high": [106.0, 111.0, 111.0, 113.0],
        "low": [99.0, 104.0, 107.0, 107.0],
        "volume": [1000.0] * 4,
    })
    
    asset_rets, used_fallback = compute_asset_returns(bars, mode="open_to_close")
    
    assert used_fallback, "Should fall back to close-to-close when open missing"
    assert asset_rets[0] == 0.0
    assert asset_rets[1] == pytest.approx(105.0 / 100.0 - 1.0), "close[t]/close[t-1] - 1"
    assert asset_rets[2] == pytest.approx(110.0 / 105.0 - 1.0)
    assert asset_rets[3] == pytest.approx(108.0 / 110.0 - 1.0)


def test_open_to_close_timing_model_b():
    """
    Test that strategy earns open-to-close returns (Model B).
    
    Setup:
    - Bar 0: open=100, close=105 (strategy decides to go long at close)
    - Bar 1: open=105, close=110 (position executes at open=105, earns 105->110)
    
    The old bug would compute asset_ret[1] = 110/105 - 1 but position would be
    shifted such that it earns nothing on bar 1.
    
    Correct behavior: position decided at close[0] executes at open[1] and earns
    the open[1]->close[1] return.
    """
    bars = pl.DataFrame({
        "ts": [datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc) for i in range(3)],
        "symbol": ["BTC-USD"] * 3,
        "open": [100.0, 105.0, 110.0],
        "close": [105.0, 110.0, 108.0],
        "high": [106.0, 111.0, 111.0],
        "low": [99.0, 104.0, 107.0],
        "volume": [1000.0] * 3,
    })
    
    raw_cfg = RunConfigRaw(
        run_name="test_open_to_close",
        data=DataConfig(
            db_path=":memory:",
            table="bars",
            symbol="BTC-USD",
            start=bars[0, "ts"],
            end=bars[2, "ts"],
        ),
        engine=EngineConfig(
            strict_validation=False,
            lookback=10,
            initial_cash=100000.0,
        ),
        strategy=StrategyConfigRaw(
            fast=2,
            slow=3,
            vol_window=2,
            k=1.0,
            min_band=0.0,
        ),
        risk=RiskConfigRaw(
            vol_window=2,
            target_vol_annual=None,  # No vol targeting
            max_weight=1.0,
        ),
        execution=ExecutionConfig(execution_lag_bars=1),
    )
    cfg = compile_config(raw_cfg)
    
    engine = BacktestEngine(
        cfg,
        AlwaysLong(),  # Decides to go 100% long immediately
        RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor),
        DummyPortal(bars),
    )
    
    portfolio, summary = engine.run()
    frames = portfolio.to_frames()
    equity = frames["equity"].sort("ts")
    
    # Verify we used open-to-close returns
    assert not summary.get("used_close_to_close_fallback"), \
        "Should use open-to-close when open column exists"
    
    # At t=0: strategy decides to go long at close
    # At t=1: position executes at open=105, earns 105->110 = +4.76% on 100% position
    # gross_ret[1] should be approximately (110/105 - 1) = 0.0476
    gross_ret_1 = equity.filter(pl.col("ts") == bars[1, "ts"])["gross_ret"].item()
    expected_ret_1 = 110.0 / 105.0 - 1.0  # open[1] to close[1]
    
    assert gross_ret_1 == pytest.approx(expected_ret_1, rel=1e-6), \
        f"Expected gross_ret[1] = {expected_ret_1:.6f}, got {gross_ret_1:.6f}. " \
        f"Strategy should earn open[1]->close[1] return after deciding at close[0]."
    
    # At t=2: position remains 100%, earns 110->108 = -1.82%
    gross_ret_2 = equity.filter(pl.col("ts") == bars[2, "ts"])["gross_ret"].item()
    expected_ret_2 = 108.0 / 110.0 - 1.0  # open[2] to close[2]
    
    assert gross_ret_2 == pytest.approx(expected_ret_2, rel=1e-6), \
        f"Expected gross_ret[2] = {expected_ret_2:.6f}, got {gross_ret_2:.6f}"


def test_funding_diagnostics_present():
    """Test that funding diagnostics are included in output."""
    bars = pl.DataFrame({
        "ts": [datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc) for i in range(3)],
        "symbol": ["BTC-USD"] * 3,
        "open": [100.0, 105.0, 110.0],
        "close": [105.0, 110.0, 108.0],
        "high": [106.0, 111.0, 111.0],
        "low": [99.0, 104.0, 107.0],
        "volume": [1000.0] * 3,
    })
    
    raw_cfg = RunConfigRaw(
        run_name="test_funding",
        data=DataConfig(
            db_path=":memory:",
            table="bars",
            symbol="BTC-USD",
            start=bars[0, "ts"],
            end=bars[2, "ts"],
        ),
        engine=EngineConfig(strict_validation=False, lookback=10, initial_cash=100000.0),
        strategy=StrategyConfigRaw(fast=2, slow=3, vol_window=2, k=1.0, min_band=0.0),
        risk=RiskConfigRaw(vol_window=2, target_vol_annual=None, max_weight=1.0),
        execution=ExecutionConfig(),
    )
    cfg = compile_config(raw_cfg)
    
    engine = BacktestEngine(cfg, AlwaysLong(), RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor), DummyPortal(bars))
    portfolio, summary = engine.run()
    frames = portfolio.to_frames()
    equity = frames["equity"]
    
    # Verify funding columns exist
    assert "funding_costs" in equity.columns, "funding_costs column must exist in equity_df"
    assert "cum_funding_costs" in equity.columns, "cum_funding_costs column must exist in equity_df"
    
    # Verify summary fields
    assert "total_funding_cost" in summary, "total_funding_cost must be in summary"
    assert "trade_log_mode" in summary, "trade_log_mode must be in summary"
    assert "trade_log_note" in summary, "trade_log_note must be in summary"
    
    assert summary["trade_log_mode"] == "binary_entries_exits_only"
    assert "partial resizes" in summary["trade_log_note"]
    
    # Funding costs should be 0.0 (placeholder until funding rates are provided)
    assert summary["total_funding_cost"] == 0.0
