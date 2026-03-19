"""Shared execution helpers used by both BacktestEngine and PortfolioEngine.

Centralises logic that was previously duplicated across both engines,
eliminating the class of bugs that arise from independent implementations
drifting out of sync.
"""
from __future__ import annotations

import polars as pl


def apply_deadband(
    target_w: float,
    held_w: float,
    deadband: float,
    max_step: float | None = None,
) -> float:
    """Apply rebalance deadband and optional max weight step limit.

    Returns the new held weight after applying the deadband filter and
    clamping the delta to ``max_step`` (if provided).
    """
    delta = target_w - held_w
    if abs(delta) < deadband:
        return held_w
    if max_step is not None:
        delta = max(min(delta, max_step), -max_step)
    return held_w + delta


def apply_dd_throttle(
    nav: float,
    peak_nav: float,
    max_allowed_dd: float,
    floor: float,
    enabled: bool = True,
) -> tuple[float, float]:
    """Compute drawdown scaler for position sizing.

    Returns:
        (dd_scaler, current_dd) where dd_scaler is in [floor, 1.0].
    """
    current_dd = 1 - (nav / peak_nav) if peak_nav > 0 else 0.0
    if not enabled:
        return 1.0, current_dd
    dd_scaler = max(floor, min(1.0, 1 - current_dd / max_allowed_dd))
    return dd_scaler, current_dd


def compute_summary_stats(equity: pl.DataFrame) -> dict[str, object]:
    """Compute standard performance statistics from an equity DataFrame.

    Expects columns: ``ts``, ``nav``.  Computes total return, Sharpe,
    Sortino, max drawdown, and CAGR.
    """
    if equity.is_empty():
        return {"total_return": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}
    equity = equity.sort("ts")
    nav = equity["nav"]
    start_nav = float(nav.item(0))
    end_nav = float(nav.item(nav.len() - 1))
    total_return = (end_nav / start_nav) - 1 if start_nav else 0.0

    returns = nav.pct_change().drop_nulls()

    diffs = equity.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
    dt_seconds: float = float(diffs.median()) if diffs.len() > 0 else 0.0  # type: ignore[arg-type]
    periods_per_year: float = (365 * 24 * 3600 / dt_seconds) if dt_seconds > 0 else 8760.0

    _STD_FLOOR = 1e-12
    sharpe: float = 0.0
    sortino: float = 0.0
    if returns.len() > 1:
        mean = float(returns.mean() or 0.0)  # type: ignore[arg-type]
        std = float(returns.std(ddof=1) or 0.0)  # type: ignore[arg-type]
        sharpe = (mean / std) * (periods_per_year ** 0.5) if std > _STD_FLOOR else 0.0
        downside = returns.filter(returns < 0)
        down_std = float(downside.std(ddof=1) or 0.0) if downside.len() > 1 else 0.0  # type: ignore[arg-type]
        sortino = (mean / down_std) * (periods_per_year ** 0.5) if down_std > _STD_FLOOR else 0.0

    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_drawdown = float(drawdowns.min() or 0.0)  # type: ignore[arg-type]

    n_bars = equity.height
    n_years = n_bars / periods_per_year if periods_per_year > 0 else 0.0
    cagr = (end_nav / start_nav) ** (1 / n_years) - 1 if n_years > 0 and start_nav > 0 else 0.0

    return {
        "total_return": total_return,
        "total_return_decimal": total_return,
        "total_return_pct": total_return * 100.0,
        "total_return_multiple": 1.0 + total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "cagr": cagr,
        "n_bars": n_bars,
        "periods_per_year": periods_per_year,
    }
