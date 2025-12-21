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
        cfg,
        AlwaysLong(),
        RiskManager(cfg.risk, periods_per_year=8760),
        DummyPortal(bars),
    )
    portfolio, _ = engine.run()
    frames = portfolio.to_frames()
    eq = frames["equity"].sort("ts")
    pos = frames["positions"].sort("ts")

    if "position_units" in pos.columns:
        in_pos = (pos["position_units"].abs() > 1e-12)
        assert in_pos.sum() >= 1
        assert (in_pos.cast(int).diff().fill_null(0) == 1).sum() == 1

    if "turnover" in eq.columns:
        turn = eq["turnover"].fill_null(0)
        assert (turn > 0).sum() == 1

    assert abs(eq["nav"][0] - cfg.engine.initial_cash) < 1e-9

    ctx = make_strategy_context(bars, 1, lookback=2)
    assert ctx.history.select(pl.col("ts").max()).item() == ctx.decision_ts
    assert ctx.history.filter(pl.col("ts") > ctx.decision_ts).is_empty()

