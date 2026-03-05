"""
Portfolio simulation engine for JPM Momentum research.

Follows the existing codebase pattern:
- Model-B timing: signal at close(t), execute at open(t+1)
- Apply trading costs (configurable, default 20 bps round-trip)
- Track: equity curve, weights held, turnover, gross exposure
- Cash earns risk-free rate (configurable, default 4%)

Supports both crypto (ann_factor=365) and ETF (ann_factor=252) markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ANN_FACTOR_CRYPTO

DEFAULT_COST_BPS = 20.0  # 10 fee + 10 slippage


def simple_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
    execution_lag: int = 1,
    cash_yield: float = 0.04,
    ann_factor: float = ANN_FACTOR_CRYPTO,
) -> pd.DataFrame:
    """Run a simple portfolio backtest from weight and return matrices.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format: index = ts (datetime), columns = symbols, values = target weights.
        Weights are signal weights decided at close of day t.
    returns : pd.DataFrame
        Wide-format: index = ts, columns = symbols, values = open-to-close returns.
    cost_bps : float
        Round-trip transaction cost in basis points.
    execution_lag : int
        Number of bars between signal and execution (Model-B: signal at close t,
        execute at open t+1, earn close t+1 return).
    cash_yield : float
        Annualised risk-free rate earned on idle cash.
    ann_factor : float
        Annualisation factor (365 for crypto, 252 for ETFs).

    Returns
    -------
    pd.DataFrame
        Columns: ts, portfolio_ret, portfolio_equity, gross_exposure,
        turnover, cost_ret, cash_ret.
    """
    common_ts = weights.index.intersection(returns.index).sort_values()
    w = weights.reindex(common_ts).fillna(0.0)
    r = returns.reindex(common_ts).fillna(0.0)

    w_held = w.shift(execution_lag).fillna(0.0)

    gross = w_held.abs().sum(axis=1)
    cash_weight = (1.0 - gross).clip(lower=0.0)
    turnover = (w_held - w_held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_ret = turnover * (cost_bps / 10_000)

    daily_cash = cash_yield / ann_factor
    cash_return = cash_weight * daily_cash

    port_ret = (w_held * r).sum(axis=1) + cash_return - cost_ret
    port_equity = (1 + port_ret).cumprod()

    return pd.DataFrame({
        "ts": common_ts,
        "portfolio_ret": port_ret.values,
        "portfolio_equity": port_equity.values,
        "gross_exposure": gross.values,
        "turnover": turnover.values,
        "cost_ret": cost_ret.values,
        "cash_ret": cash_return.values,
    })
