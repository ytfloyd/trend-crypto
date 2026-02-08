"""Centralized portfolio performance metrics.

Provides standalone metric functions that operate on equity/weight DataFrames
produced by either BacktestEngine or PortfolioEngine.
"""
from __future__ import annotations

import math
from typing import Optional

import polars as pl


def sharpe_ratio(
    returns: pl.Series, periods_per_year: float = 8760.0, risk_free: float = 0.0
) -> float:
    """Annualized Sharpe ratio.

    Args:
        returns: Per-bar return series.
        periods_per_year: Annualization factor (default: hourly bars, 8760).
        risk_free: Per-bar risk-free rate (default: 0).
    """
    if returns.len() < 2:
        return 0.0
    excess = returns - risk_free
    mean = float(excess.mean() or 0.0)  # type: ignore[arg-type]
    std = float(excess.std(ddof=1) or 0.0)  # type: ignore[arg-type]
    if std <= 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: pl.Series, periods_per_year: float = 8760.0, risk_free: float = 0.0
) -> float:
    """Annualized Sortino ratio (downside deviation denominator).

    Args:
        returns: Per-bar return series.
        periods_per_year: Annualization factor.
        risk_free: Per-bar risk-free rate.
    """
    if returns.len() < 2:
        return 0.0
    excess = returns - risk_free
    mean = float(excess.mean() or 0.0)  # type: ignore[arg-type]
    downside = excess.filter(excess < 0)
    if downside.len() < 2:
        return 0.0
    down_std = float(downside.std(ddof=1) or 0.0)  # type: ignore[arg-type]
    if down_std <= 0:
        return 0.0
    return (mean / down_std) * math.sqrt(periods_per_year)


def calmar_ratio(
    returns: pl.Series, nav: pl.Series, periods_per_year: float = 8760.0
) -> float:
    """Annualized Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Per-bar return series.
        nav: NAV series aligned with returns.
        periods_per_year: Annualization factor.
    """
    if nav.len() < 2:
        return 0.0
    total_bars = returns.len()
    total_return = float(nav.item(nav.len() - 1)) / float(nav.item(0)) - 1.0
    years = total_bars / periods_per_year
    if years <= 0:
        return 0.0
    ann_return = (1 + total_return) ** (1 / years) - 1
    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_dd = abs(float(drawdowns.min() or 0.0))  # type: ignore[arg-type]
    if max_dd <= 0:
        return 0.0
    return float(ann_return / max_dd)


def max_drawdown(nav: pl.Series) -> float:
    """Maximum drawdown (negative number)."""
    if nav.len() < 2:
        return 0.0
    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    return float(drawdowns.min() or 0.0)  # type: ignore[arg-type]


def diversification_ratio(weights: dict[str, float], cov_matrix: dict[str, dict[str, float]]) -> float:
    """Diversification ratio = weighted sum of vols / portfolio vol.

    A ratio > 1 indicates diversification benefit.

    Args:
        weights: Symbol → weight mapping.
        cov_matrix: Nested dict of covariances, cov_matrix[i][j].
    """
    symbols = sorted(weights.keys())
    if not symbols:
        return 1.0
    w_sum_vol = 0.0
    port_var = 0.0
    for i, si in enumerate(symbols):
        wi = weights.get(si, 0.0)
        var_i = cov_matrix.get(si, {}).get(si, 0.0)
        w_sum_vol += abs(wi) * math.sqrt(max(0.0, var_i))
        for j, sj in enumerate(symbols):
            wj = weights.get(sj, 0.0)
            cov_ij = cov_matrix.get(si, {}).get(sj, 0.0)
            port_var += wi * wj * cov_ij
    port_vol = math.sqrt(max(0.0, port_var))
    if port_vol <= 0:
        return 1.0
    return w_sum_vol / port_vol


def hhi_concentration(weights: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index of weight concentration.

    HHI ranges from 1/N (equal weight) to 1.0 (single asset).
    Computed on absolute weights.
    """
    total = sum(abs(w) for w in weights.values())
    if total <= 0:
        return 0.0
    return sum((abs(w) / total) ** 2 for w in weights.values())


def return_contribution(
    weights_df: pl.DataFrame,
    contributions_df: pl.DataFrame,
    symbols: list[str],
) -> dict[str, float]:
    """Total return contribution per symbol over the backtest.

    Args:
        weights_df: DataFrame with columns: ts, symbol, held_weight.
        contributions_df: DataFrame with columns: ts, symbol, contribution.
        symbols: List of symbols.

    Returns:
        Dict mapping symbol → total return contribution.
    """
    result: dict[str, float] = {}
    for sym in symbols:
        sym_contrib = contributions_df.filter(pl.col("symbol") == sym)
        total = float(sym_contrib["contribution"].sum() or 0.0)
        result[sym] = total
    return result


def risk_contribution(
    weights_df: pl.DataFrame,
    returns_by_symbol: dict[str, pl.Series],
    periods_per_year: float = 8760.0,
) -> dict[str, float]:
    """Marginal risk contribution per symbol.

    Uses a simple approximation: RC_i = w_i * sigma_i * rho(i, portfolio).
    Returns annualized marginal risk contribution per symbol.

    Args:
        weights_df: DataFrame with columns: ts, symbol, held_weight.
        returns_by_symbol: Per-symbol return series aligned to the same timestamps.
        periods_per_year: Annualization factor.
    """
    symbols = sorted(returns_by_symbol.keys())
    if not symbols:
        return {}

    # Get last-known weights as representative
    last_ts = weights_df["ts"].max()
    last_weights = weights_df.filter(pl.col("ts") == last_ts)
    w: dict[str, float] = {}
    for row in last_weights.iter_rows(named=True):
        w[str(row["symbol"])] = float(row["held_weight"])

    # Compute portfolio return series
    port_returns_list: list[float] = [0.0] * max(
        (r.len() for r in returns_by_symbol.values()), default=0
    )
    for sym in symbols:
        rets = returns_by_symbol[sym].to_list()
        wi = w.get(sym, 0.0)
        for t in range(len(rets)):
            port_returns_list[t] += wi * rets[t]

    port_series = pl.Series("port", port_returns_list)
    port_std = float(port_series.std(ddof=1) or 0.0)  # type: ignore[arg-type]

    result: dict[str, float] = {}
    for sym in symbols:
        sym_series = returns_by_symbol[sym]
        sym_std = float(sym_series.std(ddof=1) or 0.0)  # type: ignore[arg-type]
        wi = w.get(sym, 0.0)
        # Correlation via covariance
        if sym_series.len() > 1 and port_series.len() > 1:
            cov = float(sym_series.to_frame("a").with_columns(
                pl.Series("b", port_returns_list[:sym_series.len()])
            ).select(pl.corr("a", "b")).item() or 0.0)
        else:
            cov = 0.0
        rc = abs(wi) * sym_std * cov * math.sqrt(periods_per_year)
        result[sym] = rc

    return result
