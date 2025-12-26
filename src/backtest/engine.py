from __future__ import annotations

from dataclasses import dataclass
import math
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
        vol_signal: list[float] = []
        adx_signal: list[float] = []
        ma_signal: list[bool] = []
        adx_pass_signal: list[bool] = []
        in_pos_signal: list[bool] = []
        for i in range(n):
            ctx = make_strategy_context(bars, i, self.cfg.engine.lookback)
            if self.cfg.engine.strict_validation:
                validate_context_bounds(ctx.history, ctx.decision_ts)
            base_weight = self.strategy.on_bar_close(ctx)
            final_weight = self.risk_manager.apply(base_weight, ctx.history)
            w_signal.append(final_weight)
            vol_signal.append(getattr(self.strategy, "last_vol_scalar", None))
            adx_signal.append(getattr(self.strategy, "last_adx", None))
            ma_signal.append(getattr(self.strategy, "last_ma_signal", False))
            adx_pass_signal.append(getattr(self.strategy, "last_adx_pass", False))
            in_pos_signal.append(getattr(self.strategy, "_in_pos", False))

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
        entry_exit_events = 0
        dd_list = []
        dd_scaler_list = []
        cash_weight_list = []
        cash_ret_list = []
        rf_bar_list = []

        nav_prev = self.cfg.engine.initial_cash
        nav_list.append(nav_prev)
        gross_ret_list.append(0.0)
        net_ret_list.append(0.0)
        turnover_list.append(0.0)
        cost_ret_list.append(0.0)
        w_held_list.append(0.0)
        peak_nav = nav_prev
        dd_list.append(0.0)
        dd_scaler_list.append(1.0)
        # cash yield per bar
        diffs = bars.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
        dt_seconds = diffs.median() if diffs.len() > 0 else 0
        periods_per_year = (365 * 24 * 3600 / dt_seconds) if dt_seconds and dt_seconds > 0 else 8760
        rf_bar = (1 + self.cfg.execution.cash_yield_annual) ** (1 / periods_per_year) - 1 if periods_per_year > 0 else 0.0
        rf_bar_list.append(rf_bar)
        cash_weight_list.append(1.0)
        cash_ret_list.append(rf_bar)

        for t in range(1, n):
            asset_ret = asset_close[t] / asset_close[t - 1] - 1
            raw_target = w_signal[t - lag] if t - lag >= 0 else 0.0
            w_prev = w_held_list[-1]
            # drawdown throttle on target
            if nav_prev > peak_nav:
                peak_nav = nav_prev
            current_dd = 1 - (nav_prev / peak_nav) if peak_nav > 0 else 0.0
            dd_list.append(current_dd)
            if self.cfg.execution.enable_dd_throttle:
                dd_scaler = max(
                    self.cfg.execution.dd_throttle_floor,
                    min(1.0, 1 - current_dd / self.cfg.execution.max_allowed_drawdown),
                )
            else:
                dd_scaler = 1.0
            dd_scaler_list.append(dd_scaler)
            effective_max_weight = dd_scaler
            if abs(raw_target) > effective_max_weight:
                raw_target = math.copysign(effective_max_weight, raw_target)
            delta = raw_target - w_prev
            db = self.cfg.execution.rebalance_deadband or 0.0
            if abs(delta) < db:
                w_held = w_prev
            else:
                step = self.cfg.execution.max_weight_step
                if step is not None:
                    delta = max(min(delta, step), -step)
                w_held = w_prev + delta
            turnover = abs(w_held - w_prev)
            cost_ret = turnover * fee_slip_bps / 10000.0
            gross_ret = w_held * asset_ret
            cash_weight = max(0.0, 1.0 - abs(w_prev))
            cash_ret = cash_weight * rf_bar
            net_ret = gross_ret + cash_ret - cost_ret
            nav_curr = nav_prev * (1 + net_ret)

            prev_long = w_prev > 1e-12
            curr_long = w_held > 1e-12
            if prev_long != curr_long:
                entry_exit_events += 1

            w_held_list.append(w_held)
            turnover_list.append(turnover)
            cost_ret_list.append(cost_ret)
            gross_ret_list.append(gross_ret)
            net_ret_list.append(net_ret)
            nav_list.append(nav_curr)
            nav_prev = nav_curr
            cash_weight_list.append(cash_weight)
            cash_ret_list.append(cash_ret)
            rf_bar_list.append(rf_bar)

        equity_df = pl.DataFrame(
            {
                "ts": ts_list,
                "nav": nav_list,
                "gross_ret": gross_ret_list,
                "net_ret": net_ret_list,
                "turnover": turnover_list,
                "cost_ret": cost_ret_list,
                "w_held": w_held_list,
                "dd": dd_list,
                "dd_scaler": dd_scaler_list,
                "cash_weight": cash_weight_list,
                "cash_ret": cash_ret_list,
                "rf_bar": rf_bar_list,
            }
        )

        portfolio = Portfolio(cash=self.cfg.engine.initial_cash)
        portfolio.equity_history = equity_df.to_dicts()
        position_rows = []
        for i in range(n):
            position_rows.append(
                {
                    "ts": ts_list[i],
                    "weight": w_held_list[i],
                    "signal_weight": w_signal[i] if i < len(w_signal) else None,
                    "vol_scalar": vol_signal[i] if i < len(vol_signal) else None,
                    "adx": adx_signal[i] if i < len(adx_signal) else None,
                    "ma_signal": ma_signal[i] if i < len(ma_signal) else None,
                    "adx_pass": adx_pass_signal[i] if i < len(adx_pass_signal) else None,
                    "in_pos": in_pos_signal[i] if i < len(in_pos_signal) else None,
                    "dd": dd_list[i] if i < len(dd_list) else None,
                    "dd_scaler": dd_scaler_list[i] if i < len(dd_scaler_list) else None,
                    "cash_weight": cash_weight_list[i] if i < len(cash_weight_list) else None,
                }
            )
        portfolio.position_history = position_rows

        summary = _summary_stats(equity_df)
        summary["entry_exit_events"] = entry_exit_events
        summary["cash_yield_annual"] = self.cfg.execution.cash_yield_annual
        summary["avg_cash_weight"] = float(sum(cash_weight_list)) / float(len(cash_weight_list)) if cash_weight_list else 0.0
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

