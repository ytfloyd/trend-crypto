"""Vol-targeted portfolio construction for time-series trend signals.

This module converts per-asset continuous forecasts into a portfolio of
daily returns, following the methodology of AHL/Man Group, Carver, and
other institutional trend followers:

  weight_i(t) = forecast_i(t) / FORECAST_SCALE * (vol_target / vol_i(t))

The result is a single daily return series for the whole portfolio,
which can then be evaluated by CPCV, PBO, and DSR.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FORECAST_SCALE = 10.0  # Carver convention: average |forecast| ≈ 10


@dataclass
class PortfolioResult:
    """Output of a vol-targeted portfolio backtest."""

    daily_returns: pd.Series  # net of costs
    gross_returns: pd.Series
    cost_series: pd.Series
    turnover_series: pd.Series
    weights: pd.DataFrame  # (ts x symbol)
    net_sharpe: float
    net_sortino: float
    cagr: float
    max_drawdown: float
    annual_turnover: float
    total_cost_drag: float


def vol_targeted_backtest(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    vol_target: float = 0.15,
    vol_lookback: int = 60,
    max_weight: float = 0.40,
    max_gross_leverage: float = 2.0,
    cost_bps: float = 20.0,
    ann_factor: float = 365.0,
) -> PortfolioResult:
    """Run a vol-targeted portfolio backtest from forecasts.

    Parameters
    ----------
    forecasts : (ts x symbol) DataFrame of continuous forecasts.
        Positive = long, negative = short.  Carver convention:
        average |forecast| ≈ 10 maps to full risk allocation.
    returns : (ts x symbol) DataFrame of daily asset returns.
    vol_target : annualised volatility target per instrument.
    vol_lookback : rolling window for vol estimation.
    max_weight : per-asset weight cap.
    max_gross_leverage : portfolio-level gross leverage limit.
    cost_bps : round-trip transaction cost in basis points.
    ann_factor : trading days per year (365 for crypto).

    Returns
    -------
    PortfolioResult with daily return series and summary metrics.
    """
    common_dates = sorted(forecasts.index.intersection(returns.index))
    common_syms = sorted(forecasts.columns.intersection(returns.columns))

    if len(common_dates) < 60 or len(common_syms) < 3:
        return _empty_result()

    fc = forecasts.loc[common_dates, common_syms]
    ret = returns.loc[common_dates, common_syms]

    # Rolling volatility (annualised)
    vol = ret.rolling(vol_lookback, min_periods=max(vol_lookback // 2, 10)).std()
    vol_ann = vol * np.sqrt(ann_factor)
    vol_ann = vol_ann.replace(0, np.nan)

    # Per-asset target weights: forecast / scale * (vol_target / vol)
    raw_weights = (fc / FORECAST_SCALE) * (vol_target / vol_ann)
    raw_weights = raw_weights.clip(-max_weight, max_weight)

    # Gross leverage constraint
    gross = raw_weights.abs().sum(axis=1)
    scale_factor = (max_gross_leverage / gross).clip(upper=1.0)
    weights = raw_weights.mul(scale_factor, axis=0)
    weights = weights.fillna(0.0)

    # Turnover
    weight_diff = weights.diff().abs().fillna(0.0)
    daily_turnover = weight_diff.sum(axis=1)

    # Cost: turnover * cost_bps / 10_000
    daily_cost = daily_turnover * (cost_bps / 10_000.0)

    # Portfolio gross return
    gross_ret = (weights.shift(1) * ret).sum(axis=1)
    gross_ret.iloc[0] = 0.0

    # Net return
    net_ret = gross_ret - daily_cost

    # Summary metrics
    valid_net = net_ret.iloc[vol_lookback:]  # skip warm-up
    if len(valid_net) < 30:
        return _empty_result()

    mean_ret = float(valid_net.mean())
    std_ret = float(valid_net.std())
    n_years = len(valid_net) / ann_factor

    net_sharpe = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 1e-12 else 0.0
    downside = valid_net[valid_net < 0]
    down_std = float(downside.std()) if len(downside) > 1 else 1e-12
    net_sortino = (mean_ret / down_std * np.sqrt(ann_factor)) if down_std > 1e-12 else 0.0

    cum = (1 + net_ret).cumprod()
    running_max = cum.cummax()
    drawdowns = cum / running_max - 1
    max_dd = float(drawdowns.min())

    end_val = float(cum.iloc[-1])
    cagr = (end_val ** (1.0 / n_years) - 1) if n_years > 0 and end_val > 0 else 0.0

    annual_turnover = float(daily_turnover.mean()) * ann_factor
    total_cost_drag = float(daily_cost.sum())

    return PortfolioResult(
        daily_returns=net_ret,
        gross_returns=gross_ret,
        cost_series=daily_cost,
        turnover_series=daily_turnover,
        weights=weights,
        net_sharpe=net_sharpe,
        net_sortino=net_sortino,
        cagr=cagr,
        max_drawdown=max_dd,
        annual_turnover=annual_turnover,
        total_cost_drag=total_cost_drag,
    )


def _empty_result() -> PortfolioResult:
    """Return a zero-filled result for insufficient data."""
    empty_s = pd.Series(dtype=float)
    empty_df = pd.DataFrame()
    return PortfolioResult(
        daily_returns=empty_s,
        gross_returns=empty_s,
        cost_series=empty_s,
        turnover_series=empty_s,
        weights=empty_df,
        net_sharpe=0.0,
        net_sortino=0.0,
        cagr=0.0,
        max_drawdown=0.0,
        annual_turnover=0.0,
        total_cost_drag=0.0,
    )
