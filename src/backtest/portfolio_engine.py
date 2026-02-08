"""Multi-asset portfolio backtesting engine.

Supports both:
- ``dict[str, TargetWeightStrategy]`` (per-asset independent strategies)
- ``PortfolioStrategy`` (joint multi-asset strategies)

Uses the same Model B execution timing as :class:`BacktestEngine`:
decide at close[t], execute at open[t+1].
"""
from __future__ import annotations

import math
from typing import Optional, Union

import polars as pl

from backtest.portfolio_result import PortfolioResult
from backtest.validators import validate_bars
from common.config import PortfolioConfig, RunConfigResolved
from common.logging import get_logger
from risk.risk_manager import RiskManager
from strategy.base import PortfolioStrategy, SingleAssetAdapter, TargetWeightStrategy
from strategy.context import StrategyContext, make_strategy_context

logger = get_logger("portfolio_engine")


def _align_multi_symbol_bars(
    bars_by_symbol: dict[str, pl.DataFrame],
) -> tuple[list[object], dict[str, pl.DataFrame]]:
    """Align bars across symbols to a common timestamp index.

    Only timestamps present in *all* symbols are kept (inner join).

    Returns:
        (common_timestamps, aligned_bars_by_symbol)
        where each DataFrame is sorted by ts and filtered to common timestamps.
    """
    ts_sets = []
    for symbol, bars in bars_by_symbol.items():
        ts_sets.append(set(bars["ts"].to_list()))
    if not ts_sets:
        return [], {}
    common_ts = ts_sets[0]
    for ts_set in ts_sets[1:]:
        common_ts = common_ts & ts_set
    common_ts_sorted = sorted(common_ts)
    if not common_ts_sorted:
        return [], {}

    aligned: dict[str, pl.DataFrame] = {}
    for symbol, bars in bars_by_symbol.items():
        filtered = bars.filter(pl.col("ts").is_in(common_ts_sorted)).sort("ts")
        aligned[symbol] = filtered

    return common_ts_sorted, aligned


class PortfolioEngine:
    """Multi-asset portfolio backtesting engine.

    Args:
        cfg: Resolved run configuration (must include portfolio config).
        strategy: Either a ``PortfolioStrategy`` for joint decisions, or a
            ``dict[str, TargetWeightStrategy]`` for per-asset independent strategies.
        risk_manager: Per-asset risk manager (applied to each symbol independently).
        bars_by_symbol: Pre-loaded bars per symbol.
        portfolio_cfg: Portfolio-level constraints (leverage, concentration limits).
    """

    def __init__(
        self,
        cfg: RunConfigResolved,
        strategy: Union[PortfolioStrategy, dict[str, TargetWeightStrategy]],
        risk_manager: RiskManager,
        bars_by_symbol: dict[str, pl.DataFrame],
        portfolio_cfg: Optional[PortfolioConfig] = None,
    ) -> None:
        self.cfg = cfg
        if isinstance(strategy, dict):
            self._portfolio_strategy: PortfolioStrategy = SingleAssetAdapter(strategy)
        else:
            self._portfolio_strategy = strategy
        self.risk_manager = risk_manager
        self._raw_bars = bars_by_symbol
        self.portfolio_cfg = portfolio_cfg or (cfg.raw.portfolio if cfg.raw.portfolio else None)
        if self.portfolio_cfg is None:
            raise ValueError("PortfolioEngine requires a PortfolioConfig")

    def run(self) -> PortfolioResult:
        """Execute multi-asset backtest and return portfolio result."""
        # Validate and align bars
        for symbol, bars in self._raw_bars.items():
            validate_bars(bars)
        common_ts, aligned = _align_multi_symbol_bars(self._raw_bars)
        n = len(common_ts)
        symbols = sorted(aligned.keys())
        if n < 2:
            raise ValueError("Need at least two aligned bars across symbols.")
        logger.info("PortfolioEngine: %d bars, %d symbols: %s", n, len(symbols), symbols)

        assert self.portfolio_cfg is not None  # enforced in __init__
        lag = max(1, self.cfg.execution.execution_lag_bars)
        fee_slip_bps = (self.cfg.execution.fee_bps or 0.0) + (self.cfg.execution.slippage_bps or 0.0)
        max_gross = self.portfolio_cfg.max_gross_leverage
        max_single = self.portfolio_cfg.max_single_name_weight

        # Pre-extract price lists for each symbol
        close_prices: dict[str, list[float]] = {}
        open_prices: dict[str, list[float]] = {}
        for sym in symbols:
            close_prices[sym] = aligned[sym]["close"].to_list()
            open_prices[sym] = aligned[sym]["open"].to_list()

        # Compute asset returns (Model B: open-to-close)
        asset_returns: dict[str, list[float]] = {}
        for sym in symbols:
            rets = [0.0]
            for t in range(1, n):
                rets.append(close_prices[sym][t] / open_prices[sym][t] - 1.0)
            asset_returns[sym] = rets

        # Run strategy + risk per bar
        raw_targets: list[dict[str, float]] = []
        risk_targets: list[dict[str, float]] = []
        for i in range(n):
            contexts: dict[str, StrategyContext] = {}
            for sym in symbols:
                contexts[sym] = make_strategy_context(
                    aligned[sym], i, self.cfg.engine.lookback
                )
            raw_w = self._portfolio_strategy.on_bar_close_portfolio(contexts)
            raw_targets.append(raw_w)

            # Apply per-asset risk manager
            risk_w: dict[str, float] = {}
            for sym in symbols:
                base = raw_w.get(sym, 0.0)
                scaled = self.risk_manager.apply(base, contexts[sym].history)
                risk_w[sym] = scaled
            risk_targets.append(risk_w)

        # Execution simulation
        nav_prev = self.cfg.engine.initial_cash
        peak_nav = nav_prev
        db = self.cfg.execution.rebalance_deadband or 0.0
        w_held: dict[str, float] = {sym: 0.0 for sym in symbols}

        # Output accumulators
        nav_list = [nav_prev]
        gross_ret_list = [0.0]
        net_ret_list = [0.0]
        turnover_list = [0.0]
        cost_ret_list = [0.0]
        dd_list = [0.0]

        # Per-symbol records
        weight_records: list[dict[str, object]] = []
        contribution_records: list[dict[str, object]] = []
        trade_records: list[dict[str, object]] = []
        trade_count = 0

        # Record initial weights
        for sym in symbols:
            weight_records.append({
                "ts": common_ts[0], "symbol": sym,
                "target_weight": risk_targets[0].get(sym, 0.0),
                "held_weight": 0.0,
            })
            contribution_records.append({
                "ts": common_ts[0], "symbol": sym, "contribution": 0.0,
            })
            trade_records.append({
                "ts": common_ts[0], "symbol": sym, "turnover": 0.0, "cost_ret": 0.0,
            })

        for t in range(1, n):
            # Target weights with execution lag
            target_idx = t - lag if t - lag >= 0 else 0
            target_w = risk_targets[target_idx]

            # Drawdown throttle
            if nav_prev > peak_nav:
                peak_nav = nav_prev
            current_dd = 1 - (nav_prev / peak_nav) if peak_nav > 0 else 0.0
            dd_list.append(current_dd)

            dd_scaler = 1.0
            if self.cfg.execution.enable_dd_throttle:
                dd_scaler = max(
                    self.cfg.execution.dd_throttle_floor,
                    min(1.0, 1 - current_dd / self.cfg.execution.max_allowed_drawdown),
                )

            # Apply portfolio constraints: gross leverage + single name limits
            constrained_w: dict[str, float] = {}
            for sym in symbols:
                w = target_w.get(sym, 0.0) * dd_scaler
                w = max(-max_single, min(w, max_single))
                constrained_w[sym] = w
            gross = sum(abs(v) for v in constrained_w.values())
            if gross > max_gross and gross > 0:
                scale = max_gross / gross
                constrained_w = {s: v * scale for s, v in constrained_w.items()}

            # Deadband + step limit
            bar_turnover = 0.0
            bar_cost_ret = 0.0
            for sym in symbols:
                prev = w_held[sym]
                tgt = constrained_w.get(sym, 0.0)
                delta = tgt - prev
                if abs(delta) < db:
                    continue
                step = self.cfg.execution.max_weight_step
                if step is not None:
                    delta = max(min(delta, step), -step)
                w_held[sym] = prev + delta
                sym_turnover = abs(delta)
                sym_cost = sym_turnover * (fee_slip_bps / 10000.0)
                bar_turnover += sym_turnover
                bar_cost_ret += sym_cost
                if sym_turnover > 0:
                    trade_count += 1
                trade_records.append({
                    "ts": common_ts[t], "symbol": sym,
                    "turnover": sym_turnover, "cost_ret": sym_cost,
                })

            # Compute portfolio gross return from held weights
            bar_gross_ret = 0.0
            for sym in symbols:
                sym_ret = asset_returns[sym][t]
                contribution = w_held[sym] * sym_ret
                bar_gross_ret += contribution
                contribution_records.append({
                    "ts": common_ts[t], "symbol": sym, "contribution": contribution,
                })
                weight_records.append({
                    "ts": common_ts[t], "symbol": sym,
                    "target_weight": constrained_w.get(sym, 0.0),
                    "held_weight": w_held[sym],
                })

            net_ret = bar_gross_ret - bar_cost_ret
            nav_curr = nav_prev * (1 + net_ret)

            turnover_list.append(bar_turnover)
            cost_ret_list.append(bar_cost_ret)
            gross_ret_list.append(bar_gross_ret)
            net_ret_list.append(net_ret)
            nav_list.append(nav_curr)
            nav_prev = nav_curr

        # Build output DataFrames
        equity_df = pl.DataFrame({
            "ts": common_ts,
            "nav": nav_list,
            "gross_ret": gross_ret_list,
            "net_ret": net_ret_list,
            "turnover": turnover_list,
            "cost_ret": cost_ret_list,
            "dd": dd_list,
        })

        weights_df = pl.DataFrame(weight_records)
        contributions_df = pl.DataFrame(contribution_records)
        trades_df = pl.DataFrame(trade_records)

        summary = _portfolio_summary(equity_df, trade_count, symbols)

        return PortfolioResult(
            equity_df=equity_df,
            weights_df=weights_df,
            contributions_df=contributions_df,
            trades_df=trades_df,
            summary=summary,
        )


def _portfolio_summary(
    equity: pl.DataFrame, trade_count: int, symbols: list[str]
) -> dict[str, object]:
    """Compute aggregate portfolio performance statistics."""
    if equity.is_empty():
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    equity = equity.sort("ts")
    nav = equity["nav"]
    returns = nav.pct_change().fill_null(0.0)
    start_nav = float(nav.item(0))
    end_nav = float(nav.item(nav.len() - 1))
    total_return = (end_nav / start_nav) - 1 if start_nav else 0.0

    diffs = equity.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
    dt_seconds: float = float(diffs.median()) if diffs.len() > 0 else 0.0  # type: ignore[arg-type]
    periods_per_year: float = (365 * 24 * 3600 / dt_seconds) if dt_seconds > 0 else 8760.0

    sharpe: float = 0.0
    sortino: float = 0.0
    if returns.len() > 1:
        mean = float(returns.mean() or 0.0)  # type: ignore[arg-type]
        std = float(returns.std(ddof=1) or 0.0)  # type: ignore[arg-type]
        sharpe = (mean / std) * (periods_per_year ** 0.5) if std > 0 else 0.0
        downside = returns.filter(returns < 0)
        down_std = float(downside.std(ddof=1) or 0.0) if downside.len() > 1 else 0.0  # type: ignore[arg-type]
        sortino = (mean / down_std) * (periods_per_year ** 0.5) if down_std > 0 else 0.0

    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_drawdown = float(drawdowns.min() or 0.0)  # type: ignore[arg-type]

    total_turnover = float(equity["turnover"].sum() or 0.0)
    total_cost = float(equity["cost_ret"].sum() or 0.0)

    return {
        "total_return": total_return,
        "total_return_pct": total_return * 100.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "trade_count": trade_count,
        "total_turnover": total_turnover,
        "total_cost": total_cost,
        "n_symbols": len(symbols),
        "symbols": symbols,
        "n_bars": equity.height,
    }
