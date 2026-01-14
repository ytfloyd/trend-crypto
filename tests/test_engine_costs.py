import pytest
pytest.importorskip("polars")

from datetime import datetime, timedelta, timezone

import polars as pl

from backtest.engine import BacktestEngine
from common.config import (
    DataConfig,
    EngineConfig,
    ExecutionConfig,
    RiskConfig,
    RunConfig,
    StrategyConfig,
)
from risk.risk_manager import RiskManager
from strategy.base import TargetWeightStrategy
from data.portal import DataPortal


class DummyPortal(DataPortal):
    def __init__(self, bars: pl.DataFrame) -> None:
        self._bars = bars

    def load_bars(self) -> pl.DataFrame:
        return self._bars


class AlwaysOne(TargetWeightStrategy):
    def on_bar_close(self, ctx):  # type: ignore[override]
        return 1.0


def _bars(close_prices) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i, c in enumerate(close_prices):
        ts = start + timedelta(hours=i)
        rows.append(
            {
                "ts": ts,
                "symbol": "BTC-USD",
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 1_000,
            }
        )
    return pl.DataFrame(rows)


def test_no_lookahead_and_costs():
    bars = _bars([100, 110, 121])
    cfg = RunConfig(
        run_name="test",
        data=DataConfig(db_path=":memory:", table="bars", symbol="BTC-USD", start=bars[0, "ts"], end=bars[bars.height - 1, "ts"], timeframe="1h"),
        engine=EngineConfig(strict_validation=True, lookback=5, initial_cash=1000.0),
        strategy=StrategyConfig(fast=2, slow=3, vol_window=2, k=2.0, min_band=0.0, window_units="bars"),
        risk=RiskConfig(vol_window=2, target_vol_annual=None, max_weight=1.0, window_units="bars"),
        execution=ExecutionConfig(fee_bps=10.0, slippage_bps=0.0, execution_lag_bars=1),
    )

    engine = BacktestEngine(cfg, AlwaysOne(), RiskManager(cfg.risk, periods_per_year=8760), DummyPortal(bars))
    portfolio, _ = engine.run()
    eq = portfolio.to_frames()["equity"].sort("ts")

    # lagged: first bar no exposure
    assert abs(eq["gross_ret"][0]) < 1e-12
    assert abs(eq["net_ret"][0]) < 1e-12

    # second bar: asset ret 10%, held weight 1, turnover 1, cost 10bps
    assert abs(eq["gross_ret"][1] - 0.10) < 1e-6
    assert abs(eq["turnover"][1] - 1.0) < 1e-6
    assert abs(eq["cost_ret"][1] - 0.001) < 1e-6
    assert abs(eq["net_ret"][1] - (0.10 - 0.001)) < 1e-6

    # third bar: turnover 0, cost 0, net == gross
    assert abs(eq["turnover"][2]) < 1e-12
    assert abs(eq["cost_ret"][2]) < 1e-12
    assert abs(eq["net_ret"][2] - eq["gross_ret"][2]) < 1e-12


def test_execution_lag_shift():
    bars = _bars([100, 110, 121, 133.1])  # three returns of 10%
    cfg = RunConfig(
        run_name="test_lag",
        data=DataConfig(db_path=":memory:", table="bars", symbol="BTC-USD", start=bars[0, "ts"], end=bars[bars.height - 1, "ts"], timeframe="1h"),
        engine=EngineConfig(strict_validation=True, lookback=5, initial_cash=1000.0),
        strategy=StrategyConfig(fast=2, slow=3, vol_window=2, k=2.0, min_band=0.0, window_units="bars"),
        risk=RiskConfig(vol_window=2, target_vol_annual=None, max_weight=1.0, window_units="bars"),
        execution=ExecutionConfig(fee_bps=0.0, slippage_bps=0.0, execution_lag_bars=2),
    )

    engine = BacktestEngine(cfg, AlwaysOne(), RiskManager(cfg.risk, periods_per_year=8760), DummyPortal(bars))
    portfolio, _ = engine.run()
    eq = portfolio.to_frames()["equity"].sort("ts")

    # lag=2 means exposure starts at bar index 2
    assert abs(eq["gross_ret"][1]) < 1e-12
    assert abs(eq["gross_ret"][2] - 0.10) < 1e-6
    assert abs(eq["gross_ret"][3] - 0.10) < 1e-6

