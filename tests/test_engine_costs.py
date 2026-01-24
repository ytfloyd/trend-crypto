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
    """
    Create bars with realistic open prices.
    Open[t] is set to close[t-1] (for t > 0) to match real market behavior.
    This ensures open-to-close returns work correctly.
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i, c in enumerate(close_prices):
        ts = start + timedelta(hours=i)
        # Open at close of previous bar (except first bar)
        o = close_prices[i - 1] if i > 0 else c
        rows.append(
            {
                "ts": ts,
                "symbol": "BTC-USD",
                "open": o,
                "high": max(o, c),
                "low": min(o, c),
                "close": c,
                "volume": 1_000,
            }
        )
    return pl.DataFrame(rows)


def test_no_lookahead_and_costs():
    bars = _bars([100, 110, 121])
    raw_cfg = RunConfigRaw(
        run_name="test",
        data=DataConfig(db_path=":memory:", table="bars", symbol="BTC-USD", start=bars[0, "ts"], end=bars[bars.height - 1, "ts"], timeframe="1h"),
        engine=EngineConfig(strict_validation=True, lookback=5, initial_cash=1000.0),
        strategy=StrategyConfigRaw(fast=2, slow=3, vol_window=2, k=2.0, min_band=0.0, window_units="bars"),
        risk=RiskConfigRaw(vol_window=5, target_vol_annual=None, max_weight=1.0, window_units="bars"),
        execution=ExecutionConfig(fee_bps=10.0, slippage_bps=0.0, execution_lag_bars=1),
    )
    cfg = compile_config(raw_cfg)

    engine = BacktestEngine(cfg, AlwaysOne(), RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor), DummyPortal(bars))
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
    raw_cfg = RunConfigRaw(
        run_name="test_lag",
        data=DataConfig(db_path=":memory:", table="bars", symbol="BTC-USD", start=bars[0, "ts"], end=bars[bars.height - 1, "ts"], timeframe="1h"),
        engine=EngineConfig(strict_validation=True, lookback=5, initial_cash=1000.0),
        strategy=StrategyConfigRaw(fast=2, slow=3, vol_window=2, k=2.0, min_band=0.0, window_units="bars"),
        risk=RiskConfigRaw(vol_window=5, target_vol_annual=None, max_weight=1.0, window_units="bars"),
        execution=ExecutionConfig(fee_bps=0.0, slippage_bps=0.0, execution_lag_bars=2),
    )
    cfg = compile_config(raw_cfg)

    engine = BacktestEngine(cfg, AlwaysOne(), RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor), DummyPortal(bars))
    portfolio, _ = engine.run()
    eq = portfolio.to_frames()["equity"].sort("ts")

    # lag=2 means exposure starts at bar index 2
    assert abs(eq["gross_ret"][1]) < 1e-12
    assert abs(eq["gross_ret"][2] - 0.10) < 1e-6
    assert abs(eq["gross_ret"][3] - 0.10) < 1e-6

