from __future__ import annotations

from dataclasses import dataclass
import polars as pl

from backtest.portfolio import Portfolio
from backtest.validators import validate_bars, validate_context_bounds
from common.config import RunConfig
from data.portal import DataPortal
from strategy.context import make_strategy_context
from strategy.base import TargetWeightStrategy
from risk.risk_manager import RiskManager


@dataclass
class PendingTarget:
    ts_decided: int
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
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.data_portal = data_portal

    def run(self) -> tuple[Portfolio, dict]:
        bars = self.data_portal.load_bars()
        validate_bars(bars)

        n = bars.height
        if n < 2:
            raise ValueError("Need at least two bars to simulate execution at next open.")

        # compute weights at close
        w_signal: list[float] = []
        for i in range(n):
            ctx = make_strategy_context(bars, i, self.cfg.engine.lookback)
            if self.cfg.engine.strict_validation:
                validate_context_bounds(ctx.history, ctx.decision_ts)
            base_weight = self.strategy.on_bar_close(ctx)
            final_weight = self.risk_manager.apply(base_weight, ctx.history)
            w_signal.append(final_weight)

        lag = max(1, self.cfg.execution.execution_lag_bars)
        fee_slip_bps = (self.cfg.execution.fee_bps or 0.0) + (self.cfg.execution.slippage_bps or 0.0)

        asset_close = bars["close"].to_list()
        ts_list = bars["ts"].to_list()
        nav_list = []
        gross_ret_list = []
        net_ret_list = []
        turnover_list = []
        cost_ret_list = []
        w_held_list = []

        nav_prev = self.cfg.engine.initial_cash
        nav_list.append(nav_prev)
        gross_ret_list.append(0.0)
        net_ret_list.append(0.0)
        turnover_list.append(0.0)
        cost_ret_list.append(0.0)
        w_held_list.append(0.0)

        for t in range(1, n):
            asset_ret = asset_close[t] / asset_close[t - 1] - 1
            w_held = w_signal[t - lag] if t - lag >= 0 else 0.0
            w_prev = w_held_list[-1]
            turnover = abs(w_held - w_prev)
            cost_ret = turnover * fee_slip_bps / 10000.0
            gross_ret = w_held * asset_ret
            net_ret = gross_ret - cost_ret
            nav_curr = nav_prev * (1 + net_ret)

            w_held_list.append(w_held)
            turnover_list.append(turnover)
            cost_ret_list.append(cost_ret)
            gross_ret_list.append(gross_ret)
            net_ret_list.append(net_ret)
            nav_list.append(nav_curr)
            nav_prev = nav_curr

        equity_df = pl.DataFrame(
            {
                "ts": ts_list,
                "nav": nav_list,
                "gross_ret": gross_ret_list,
                "net_ret": net_ret_list,
                "turnover": turnover_list,
                "cost_ret": cost_ret_list,
                "w_held": w_held_list,
            }
        )

        portfolio = Portfolio(cash=self.cfg.engine.initial_cash)
        portfolio.equity_history = equity_df.to_dicts()
        portfolio.position_history = [{"ts": ts_list[i], "position_units": w_held_list[i]} for i in range(n)]

        summary = _summary_stats(equity_df)
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
    diffs = equity.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
    dt_seconds = diffs.median() if diffs.len() > 0 else 0
    periods_per_year = (365 * 24 * 3600 / dt_seconds) if dt_seconds and dt_seconds > 0 else 8760
    if returns.len() > 1:
        mean = returns.mean()
        std = returns.std(ddof=1)
        sharpe = (mean / std) * (periods_per_year**0.5) if std and std > 0 else 0.0
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

