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
from data.portal import DataPortal
from execution.sim import ExecutionSim
from risk.risk_manager import RiskManager
from strategy.base import TargetWeightStrategy
from strategy.context import make_strategy_context


class DummyPortal(DataPortal):
    def __init__(self, bars: pl.DataFrame) -> None:
        self._bars = bars

    def load_bars(self) -> pl.DataFrame:
        return self._bars


class AlwaysLong(TargetWeightStrategy):
    def on_bar_close(self, ctx):  # type: ignore[override]
        return 1.0


def _sample_bars() -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for i in range(3):
        ts = start + timedelta(hours=i)
        rows.append(
            {
                "ts": ts,
                "symbol": "BTC-USD",
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.5,
                "volume": 1_000,
            }
        )
        price += 5
    return pl.DataFrame(rows)


def test_temporal_and_fill_timing():
    bars = _sample_bars()
    cfg = RunConfig(
        run_name="test",
        data=DataConfig(
            db_path=":memory:", table="bars", symbol="BTC-USD", start=bars[0, "ts"], end=bars[2, "ts"]
        ),
        engine=EngineConfig(strict_validation=True, lookback=10, initial_cash=1000.0),
        strategy=StrategyConfig(fast=2, slow=3, vol_window=2, k=2.0, min_band=0.0),
        risk=RiskConfig(vol_window=2, target_vol_annual=1.0, max_weight=1.0),
        execution=ExecutionConfig(),
    )
    engine = BacktestEngine(
        cfg, AlwaysLong(), RiskManager(cfg.risk), DummyPortal(bars), ExecutionSim()
    )
    portfolio, _ = engine.run()
    trades = portfolio.to_frames()["trades"]

    assert trades.height == 1
    assert trades[0, "ts"] == bars[1, "ts"]
    assert trades[0, "ref_price"] == bars[1, "open"]

    ctx = make_strategy_context(bars, 1, lookback=2)
    assert ctx.history.select(pl.col("ts").max()).item() == ctx.decision_ts
    assert ctx.history.filter(pl.col("ts") > ctx.decision_ts).is_empty()

