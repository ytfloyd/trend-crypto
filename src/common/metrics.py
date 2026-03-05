"""Canonical metrics computation for equity curves.

Every script that computes Sharpe, CAGR, max drawdown, etc. should import
from this module to guarantee consistency. The engine's _summary_stats
already uses equivalent logic for its own summaries.
"""
from __future__ import annotations

from math import sqrt

import polars as pl

_STD_FLOOR = 1e-12


def infer_periods_per_year(equity: pl.DataFrame) -> float:
    """Infer annualization factor from timestamp column spacing."""
    diffs = equity.select(
        pl.col("ts").diff().dt.total_seconds()
    ).to_series().drop_nulls()
    if diffs.len() == 0:
        return 365.0
    dt_seconds = float(diffs.median())  # type: ignore[arg-type]
    if dt_seconds <= 0:
        return 365.0
    return 365 * 24 * 3600 / dt_seconds


def compute_sharpe(returns: pl.Series, periods_per_year: float) -> float:
    """Compute annualized Sharpe ratio from a return series.

    The return series should NOT include a padding zero for t=0.
    If your data comes from pct_change(), drop_nulls() first.
    If your data comes from equity.parquet net_ret column, slice(1) first.
    """
    if returns.len() < 2:
        return 0.0
    mean = float(returns.mean() or 0.0)  # type: ignore[arg-type]
    std = float(returns.std(ddof=1) or 0.0)  # type: ignore[arg-type]
    if std <= _STD_FLOOR:
        return 0.0
    return (mean / std) * sqrt(periods_per_year)


def compute_cagr(start_nav: float, end_nav: float,
                 n_periods: int, periods_per_year: float) -> float:
    """Compound annual growth rate."""
    if start_nav <= 0 or n_periods <= 0:
        return 0.0
    return float((end_nav / start_nav) ** (periods_per_year / n_periods) - 1.0)


def compute_max_drawdown(nav: pl.Series) -> float:
    """Maximum drawdown (negative number, e.g. -0.25 for 25% DD)."""
    running_max = nav.cum_max()
    drawdowns = nav / running_max - 1.0
    return float(drawdowns.min() or 0.0)  # type: ignore[arg-type]


def compute_vol(returns: pl.Series, periods_per_year: float) -> float:
    """Annualized volatility."""
    std = float(returns.std(ddof=1) or 0.0)  # type: ignore[arg-type]
    return std * sqrt(periods_per_year)


def equity_metrics(equity: pl.DataFrame,
                   return_col: str | None = None) -> dict[str, float]:
    """Compute standard metrics from an equity DataFrame.

    Parameters
    ----------
    equity : pl.DataFrame
        Must have columns: ts, nav.
        Optionally has a return column (net_ret, r_port, etc.)
    return_col : str | None
        Name of the return column. If None, returns are derived via
        pct_change on nav (dropping the first null). If a column name is
        given, the first row is skipped (assumed to be engine padding).
    """
    if equity.is_empty() or equity.height < 2:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "cagr": 0.0, "vol_annual": 0.0}

    equity = equity.sort("ts")
    nav = equity["nav"]
    start_nav = float(nav.item(0))
    end_nav = float(nav.item(nav.len() - 1))
    total_return = (end_nav / start_nav) - 1.0 if start_nav > 0 else 0.0

    if return_col and return_col in equity.columns:
        returns = equity[return_col].slice(1)
    else:
        returns = nav.pct_change().drop_nulls()

    ppy = infer_periods_per_year(equity)
    sharpe = compute_sharpe(returns, ppy)
    cagr = compute_cagr(start_nav, end_nav, returns.len(), ppy)
    max_dd = compute_max_drawdown(nav)
    vol_ann = compute_vol(returns, ppy)
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "vol_annual": vol_ann,
        "calmar": calmar,
        "periods_per_year": ppy,
    }
