"""Minimal regression test for engine accounting/timing behavior."""
import pytest

pytest.importorskip("polars")

from datetime import datetime, timedelta, timezone
import polars as pl

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


class DummyPortal(DataPortal):
    def __init__(self, bars: pl.DataFrame) -> None:
        self._bars = bars

    def load_bars(self) -> pl.DataFrame:
        return self._bars


def _synthetic_bars(n: int = 12, open_price: float = 100.0, close_price: float = 101.0) -> pl.DataFrame:
    """Create a synthetic monotonic bars series with fixed open->close returns."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = start + timedelta(hours=i)
        rows.append(
            {
                "ts": ts,
                "symbol": "BTC-USD",
                "open": open_price,
                "high": close_price,
                "low": open_price,
                "close": close_price,
                "volume": 1000.0,
            }
        )
    return pl.DataFrame(rows)


def test_engine_regression_minimal_buy_and_hold():
    """Lock in engine timing/PnL behavior with deterministic synthetic bars."""
    bars = _synthetic_bars(n=12, open_price=100.0, close_price=101.0)  # +1% per bar

    raw_cfg = RunConfigRaw(
        run_name="test_engine_regression",
        data=DataConfig(
            db_path=":memory:",
            table="bars",
            symbol="BTC-USD",
            start=bars[0, "ts"],
            end=bars[bars.height - 1, "ts"],
            timeframe="1h",
        ),
        engine=EngineConfig(strict_validation=True, lookback=5, initial_cash=1000.0),
        strategy=StrategyConfigRaw(mode="buy_and_hold", weight_on=1.0, window_units="hours"),
        risk=RiskConfigRaw(vol_window=2, target_vol_annual=None, max_weight=1.0, window_units="hours"),
        execution=ExecutionConfig(fee_bps=0.0, slippage_bps=0.0, execution_lag_bars=1),
    )

    cfg = compile_config(raw_cfg)
    
    engine = BacktestEngine(
        cfg,
        BuyAndHoldStrategy(cfg.strategy),
        RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor),
        DummyPortal(bars),
    )

    portfolio, summary = engine.run()
    frames = portfolio.to_frames()
    equity = frames["equity"].sort("ts")

    # Expected: position starts at t=1 due to execution_lag_bars=1
    # Per-bar return is 1% (open->close) for bars 1..n-1
    expected_total_return = (1.01 ** (bars.height - 1)) - 1

    assert summary["total_return"] == pytest.approx(expected_total_return, rel=1e-6)

    # Equity curve: strictly increasing, no NaNs
    nav = equity["nav"].to_list()
    assert all(v is not None for v in nav)
    assert all(nav[i] < nav[i + 1] for i in range(len(nav) - 1))

    # Entry/exit events: single entry from 0 -> 1 exposure
    assert summary["entry_exit_events"] == 1

    # Check gross return for first invested bar (t=1)
    gross_ret_t1 = equity["gross_ret"][1]
    assert gross_ret_t1 == pytest.approx(0.01, rel=1e-6)
