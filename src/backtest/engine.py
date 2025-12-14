from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import polars as pl

from backtest.portfolio import Portfolio
from backtest.rebalance import rebalance_to_target_weight
from backtest.validators import validate_bars, validate_context_bounds, validate_fill_timing
from common.config import RunConfig
from data.portal import DataPortal
from execution.sim import ExecutionSim
from strategy.context import make_strategy_context
from strategy.base import TargetWeightStrategy
from risk.risk_manager import RiskManager


@dataclass
class PendingTarget:
    ts_decided: datetime
    target_weight: float


class BacktestEngine:
    """
    One-bar pending target engine: decide at close[t], execute at open[t+1].
    """

    def __init__(
        self,
        cfg: RunConfig,
        strategy: TargetWeightStrategy,
        risk_manager: RiskManager,
        data_portal: DataPortal,
        exec_sim: ExecutionSim,
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.data_portal = data_portal
        self.exec_sim = exec_sim

    def run(self) -> tuple[Portfolio, dict]:
        bars = self.data_portal.load_bars()
        validate_bars(bars)

        portfolio = Portfolio(cash=self.cfg.engine.initial_cash)
        pending: Optional[PendingTarget] = None
        bars_since_last_trade: int = 0

        n = bars.height
        if n < 2:
            raise ValueError("Need at least two bars to simulate execution at next open.")

        for i in range(n):
            bar_ts = bars[i, "ts"]

            # Execute pending target at the open of current bar (decision made at prior close)
            if pending is not None:
                price = bars[i, "open"]
                nav_at_open = portfolio.nav(price)
                order = rebalance_to_target_weight(
                    target_weight=pending.target_weight,
                    cash=portfolio.cash,
                    units=portfolio.position_units,
                    price=price,
                    nav=nav_at_open,
                    cfg=self.cfg.execution,
                    bars_since_last_trade=bars_since_last_trade,
                )
                if order:
                    fill = self.exec_sim.fill_order(order, ts=bar_ts)
                    portfolio.apply_fill(fill, reason="rebalance")
                    bars_since_last_trade = 0
                else:
                    bars_since_last_trade += 1
            else:
                bars_since_last_trade += 1

            # Mark-to-market at the close of the current bar
            close_price = bars[i, "close"]
            portfolio.mark_to_market(bar_ts, close_price)

            # Last bar: nothing further to decide
            if i == n - 1:
                break

            # Build strategy context using data up to and including this bar
            ctx = make_strategy_context(bars, i, self.cfg.engine.lookback)
            if self.cfg.engine.strict_validation:
                validate_context_bounds(ctx.history, ctx.decision_ts)

            base_weight = self.strategy.on_bar_close(ctx)
            final_weight = self.risk_manager.apply(base_weight, ctx.history)
            pending = PendingTarget(ts_decided=bar_ts, target_weight=final_weight)

        frames = portfolio.to_frames()
        trades_df = frames["trades"]
        if self.cfg.engine.strict_validation:
            validate_fill_timing(trades_df, bars)

        summary = _summary_stats(frames["equity"])
        return portfolio, summary


def _summary_stats(equity: pl.DataFrame) -> dict:
    if equity.is_empty():
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    equity = equity.sort("ts")
    nav = equity["nav"]
    returns = nav.pct_change().fill_null(0.0)
    start_nav = nav.item(0)
    end_nav = nav.item(nav.len() - 1)
    total_return = (end_nav / start_nav) - 1 if start_nav else 0.0
    if returns.len() > 1:
        mean = returns.mean()
        std = returns.std(ddof=1)
        sharpe = (mean / std) * (8760**0.5) if std and std > 0 else 0.0
    else:
        sharpe = 0.0
    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_drawdown = drawdowns.min()
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }

