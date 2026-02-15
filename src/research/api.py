"""Notebook-friendly API for quick backtesting and parameter sweeps.

Provides one-liner functions for common research workflows:
- ``quick_backtest()``: single strategy backtest from minimal config
- ``quick_sweep()``: grid search over parameter combinations
"""
from __future__ import annotations

from itertools import product
from typing import Any, Optional

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
from common.logging import get_logger
from data.portal import DataPortal
from risk.risk_manager import RiskManager
from strategy.base import TargetWeightStrategy
from strategy.buy_and_hold import BuyAndHoldStrategy
from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy

logger = get_logger("research_api")


class _InMemoryPortal(DataPortal):  # type: ignore[misc]
    """DataPortal backed by an in-memory DataFrame."""

    def __init__(self, bars: pl.DataFrame) -> None:
        self._bars = bars

    def load_bars(self) -> pl.DataFrame:
        return self._bars


def quick_backtest(
    bars: pl.DataFrame,
    strategy_mode: str = "ma_crossover_long_only",
    fast: int = 5,
    slow: int = 20,
    initial_cash: float = 100_000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    target_vol_annual: Optional[float] = None,
    max_weight: float = 1.0,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Run a single backtest from a DataFrame of bars.

    Args:
        bars: OHLCV DataFrame with columns: ts, open, high, low, close, volume.
        strategy_mode: "buy_and_hold" or "ma_crossover_long_only".
        fast: Fast MA window (bars).
        slow: Slow MA window (bars).
        initial_cash: Starting capital.
        fee_bps: Transaction fee in bps.
        slippage_bps: Slippage in bps.
        target_vol_annual: If set, apply vol targeting.
        max_weight: Maximum position weight.

    Returns:
        (equity_df, summary_dict)
    """
    raw = RunConfigRaw(
        run_name="quick_bt",
        data=DataConfig(
            db_path=":memory:", table="bars",
            symbol=bars[0, "symbol"] if "symbol" in bars.columns else "ASSET",
            start=bars[0, "ts"], end=bars[bars.height - 1, "ts"],
            timeframe="1h",
        ),
        engine=EngineConfig(strict_validation=False, lookback=None, initial_cash=initial_cash),
        strategy=StrategyConfigRaw(
            mode=strategy_mode,
            fast=fast if strategy_mode != "buy_and_hold" else None,
            slow=slow if strategy_mode != "buy_and_hold" else None,
            weight_on=1.0, window_units="bars",
            target_vol_annual=target_vol_annual,
            max_weight=max_weight,
        ),
        risk=RiskConfigRaw(
            vol_window=20, target_vol_annual=target_vol_annual,
            max_weight=max_weight, window_units="bars",
        ),
        execution=ExecutionConfig(
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            execution_lag_bars=1, rebalance_deadband=0.01,
        ),
    )
    cfg = compile_config(raw)

    strategy: TargetWeightStrategy
    if strategy_mode == "buy_and_hold":
        strategy = BuyAndHoldStrategy(cfg.strategy)
    else:
        strategy = MACrossoverLongOnlyStrategy(
            fast=cfg.strategy.fast or fast,
            slow=cfg.strategy.slow or slow,
            max_weight=cfg.strategy.max_weight,
        )

    rm = RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor)
    portal = _InMemoryPortal(bars)
    engine = BacktestEngine(cfg, strategy, rm, portal)
    portfolio, summary = engine.run()

    equity_df = pl.DataFrame(portfolio.equity_history) if portfolio.equity_history else pl.DataFrame()
    return equity_df, summary


def quick_sweep(
    bars: pl.DataFrame,
    param_grid: dict[str, list[Any]],
    strategy_mode: str = "ma_crossover_long_only",
    initial_cash: float = 100_000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> pl.DataFrame:
    """Grid search over parameter combinations.

    Args:
        bars: OHLCV DataFrame.
        param_grid: Dict mapping parameter name → list of values.
            Supported keys: "fast", "slow", "max_weight", "target_vol_annual".
        strategy_mode: Strategy to test.
        initial_cash: Starting capital.
        fee_bps: Transaction fee in bps.
        slippage_bps: Slippage in bps.

    Returns:
        DataFrame with one row per combination, including all params and metrics.
    """
    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(product(*values))

    results: list[dict[str, Any]] = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        try:
            _, summary = quick_backtest(
                bars,
                strategy_mode=strategy_mode,
                fast=params.get("fast", 5),
                slow=params.get("slow", 20),
                initial_cash=initial_cash,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                target_vol_annual=params.get("target_vol_annual"),
                max_weight=params.get("max_weight", 1.0),
            )
            row: dict[str, Any] = dict(params)
            row["total_return"] = summary.get("total_return", 0.0)
            row["sharpe"] = summary.get("sharpe", 0.0)
            row["max_drawdown"] = summary.get("max_drawdown", 0.0)
            row["trade_count"] = summary.get("trade_count", 0)
            results.append(row)
        except Exception as e:
            logger.warning("Sweep failed for params %s: %s", params, e)
            row = dict(params)
            row["total_return"] = None
            row["sharpe"] = None
            row["max_drawdown"] = None
            row["trade_count"] = None
            results.append(row)

    return pl.DataFrame(results)
