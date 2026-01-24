from __future__ import annotations

from dataclasses import dataclass
import math
import polars as pl

from backtest.portfolio import Portfolio
from backtest.validators import validate_bars, validate_context_bounds
from common.config import RunConfigResolved
from data.portal import DataPortal
from strategy.context import make_strategy_context
from strategy.base import TargetWeightStrategy
from risk.risk_manager import RiskManager


# Epsilon for near-zero gross PnL checks (funding cost as % of gross)
_GROSS_PNL_EPSILON = 1e-9


def compute_asset_returns(bars: pl.DataFrame, mode: str = "open_to_close") -> tuple[list[float], bool]:
    """
    Compute asset returns using open-to-close (Model B) or close-to-close fallback.
    
    Args:
        bars: DataFrame with columns: ts, open, close
        mode: "open_to_close" (default) or "close_to_close"
    
    Returns:
        (asset_returns, used_close_to_close_fallback)
        - asset_returns[0] = 0.0 (no return for first bar)
        - asset_returns[t] = close[t]/open[t] - 1 for t>=1 (if open exists)
        - used_close_to_close_fallback = True if fell back to close.pct_change()
    """
    n = bars.height
    asset_ret_list = [0.0]  # First bar has no return
    used_fallback = False
    
    if mode == "open_to_close" and "open" in bars.columns:
        # Model B: Open-to-close returns (decision at close[t-1], execute at open[t], earn open[t]->close[t])
        open_prices = bars["open"].to_list()
        close_prices = bars["close"].to_list()
        for t in range(1, n):
            asset_ret_list.append(close_prices[t] / open_prices[t] - 1.0)
    else:
        # Fallback: Close-to-close returns
        used_fallback = True
        close_prices = bars["close"].to_list()
        for t in range(1, n):
            asset_ret_list.append(close_prices[t] / close_prices[t - 1] - 1.0)
    
    return asset_ret_list, used_fallback


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
        cfg: RunConfigResolved,
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

        # Compute asset returns (open-to-close default, Model B)
        asset_ret_list, used_close_to_close_fallback = compute_asset_returns(bars, mode="open_to_close")
        
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
        funding_cost_list = []

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
        funding_cost_list.append(0.0)  # No funding cost at t=0
        # cash yield per bar
        diffs = bars.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
        dt_seconds = diffs.median() if diffs.len() > 0 else 0
        periods_per_year = (365 * 24 * 3600 / dt_seconds) if dt_seconds and dt_seconds > 0 else 8760
        rf_bar = (1 + self.cfg.execution.cash_yield_annual) ** (1 / periods_per_year) - 1 if periods_per_year > 0 else 0.0
        rf_bar_list.append(rf_bar)
        cash_weight_list.append(1.0)
        cash_ret_list.append(rf_bar)

        for t in range(1, n):
            asset_ret = asset_ret_list[t]
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
            
            # Funding costs (placeholder: 0.0 until funding rates are provided)
            # Convention: positive means longs pay shorts
            funding_cost = 0.0
            
            cash_weight = max(0.0, 1.0 - abs(w_prev))
            cash_ret = cash_weight * rf_bar
            net_ret = gross_ret + cash_ret - cost_ret - funding_cost
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
            funding_cost_list.append(funding_cost)

        # Defensive check: all lists must have same length as bars (gated by strict_validation)
        if self.cfg.engine.strict_validation:
            assert len(funding_cost_list) == n, f"funding_cost_list has length {len(funding_cost_list)}, expected {n}"
            assert len(nav_list) == n, f"nav_list has length {len(nav_list)}, expected {n}"
        
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
                "funding_costs": funding_cost_list,
            }
        ).with_columns(
            cum_funding_costs=pl.col("funding_costs").fill_null(0.0).cum_sum()
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
        summary["used_close_to_close_fallback"] = used_close_to_close_fallback
        summary["return_mode"] = "close_to_close_fallback" if used_close_to_close_fallback else "open_to_close"
        
        # Funding diagnostics
        total_funding = float(sum(funding_cost_list))
        summary["total_funding_cost"] = total_funding
        summary["avg_funding_cost_per_bar"] = total_funding / n if n > 0 else 0.0
        summary["funding_convention"] = "positive_means_longs_pay"
        
        # Funding as % of gross PnL (with epsilon guard for zero gross)
        gross_pnl_abs = abs(sum(gross_ret_list))
        if gross_pnl_abs > _GROSS_PNL_EPSILON:
            summary["funding_cost_as_pct_of_gross"] = total_funding / gross_pnl_abs
        else:
            summary["funding_cost_as_pct_of_gross"] = None
        
        summary["trade_log_mode"] = "binary_entries_exits_only"
        summary["trade_log_note"] = "Valid for 0/1 exposure strategies; ignores partial resizes (e.g., vol targeting)."
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
    total_return_pct = total_return * 100.0
    total_return_multiple = 1.0 + total_return
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
        "total_return_decimal": total_return,
        "total_return_pct": total_return_pct,
        "total_return_multiple": total_return_multiple,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }

