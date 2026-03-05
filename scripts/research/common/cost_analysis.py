"""Cost-aware rule selection following Carver's speed-limit framework.

A trading rule is worth including only if its pre-cost Sharpe improvement
exceeds the cost drag from its turnover.  This module provides utilities
to evaluate individual rules and prune a candidate set.

Reference: Robert Carver, *Systematic Trading*, Chapter 12 (Appendix B).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import ANN_FACTOR
from .backtest import simple_backtest


@dataclass
class RuleCostReport:
    """Performance and cost analysis for a single trading rule."""

    name: str
    gross_sharpe: float
    annual_turnover: float
    cost_drag: float
    net_sharpe: float
    avg_holding_period_days: float
    viable: bool

    def __repr__(self) -> str:
        flag = "OK" if self.viable else "REJECT"
        return (
            f"[{flag}] {self.name:20s}  "
            f"gross_SR={self.gross_sharpe:+.3f}  "
            f"turnover={self.annual_turnover:.1f}x  "
            f"cost_drag={self.cost_drag:.4f}  "
            f"net_SR={self.net_sharpe:+.3f}  "
            f"hold={self.avg_holding_period_days:.0f}d"
        )


def analyse_rule(
    name: str,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 20.0,
    execution_lag: int = 1,
    min_net_sharpe: float = 0.0,
) -> RuleCostReport:
    """Evaluate a single rule's gross/net Sharpe and cost drag.

    Parameters
    ----------
    name : human-readable rule label.
    weights : wide-format weight matrix (from the rule's forecast → sizing).
    returns : wide-format returns matrix.
    cost_bps : round-trip transaction cost in basis points.
    execution_lag : bars between signal and execution.
    min_net_sharpe : minimum net Sharpe to be considered viable.
    """
    bt_gross = simple_backtest(weights, returns, cost_bps=0.0, execution_lag=execution_lag)
    bt_net = simple_backtest(weights, returns, cost_bps=cost_bps, execution_lag=execution_lag)

    eq_gross = bt_gross.set_index("ts")["portfolio_equity"]
    eq_net = bt_net.set_index("ts")["portfolio_equity"]

    gross_ret = eq_gross.pct_change().dropna()
    net_ret = eq_net.pct_change().dropna()

    gross_sharpe = _sharpe(gross_ret)
    net_sharpe = _sharpe(net_ret)

    annual_turnover = float(bt_net["turnover"].sum()) / max(1.0, len(bt_net) / ANN_FACTOR)
    cost_drag = annual_turnover * (cost_bps / 10_000)

    daily_turnover = bt_net.set_index("ts")["turnover"]
    avg_turnover = float(daily_turnover.mean())
    avg_hold = 1.0 / avg_turnover if avg_turnover > 1e-8 else float("inf")

    return RuleCostReport(
        name=name,
        gross_sharpe=gross_sharpe,
        annual_turnover=annual_turnover,
        cost_drag=cost_drag,
        net_sharpe=net_sharpe,
        avg_holding_period_days=avg_hold,
        viable=net_sharpe >= min_net_sharpe,
    )


def analyse_rule_set(
    candidate_rules: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    cost_bps: float = 20.0,
    execution_lag: int = 1,
    min_net_sharpe: float = 0.0,
) -> pd.DataFrame:
    """Evaluate a set of candidate rules and rank by net Sharpe.

    Parameters
    ----------
    candidate_rules : dict mapping rule name to its wide-format weight matrix.
    returns : wide-format returns matrix (shared across all rules).
    cost_bps : round-trip cost in bps.
    execution_lag : bars between signal and execution.
    min_net_sharpe : floor for viability.

    Returns
    -------
    DataFrame with one row per rule, sorted by net_sharpe descending.
    """
    reports: list[dict] = []
    for name, weights in candidate_rules.items():
        rpt = analyse_rule(
            name=name,
            weights=weights,
            returns=returns,
            cost_bps=cost_bps,
            execution_lag=execution_lag,
            min_net_sharpe=min_net_sharpe,
        )
        reports.append({
            "name": rpt.name,
            "gross_sharpe": rpt.gross_sharpe,
            "annual_turnover": rpt.annual_turnover,
            "cost_drag": rpt.cost_drag,
            "net_sharpe": rpt.net_sharpe,
            "avg_hold_days": rpt.avg_holding_period_days,
            "viable": rpt.viable,
        })
    df = pd.DataFrame(reports).sort_values("net_sharpe", ascending=False)
    return df.reset_index(drop=True)


def select_viable_rules(
    candidate_rules: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    cost_bps: float = 20.0,
    execution_lag: int = 1,
    min_net_sharpe: float = 0.0,
) -> list[str]:
    """Return names of rules that pass the cost viability filter."""
    df = analyse_rule_set(
        candidate_rules=candidate_rules,
        returns=returns,
        cost_bps=cost_bps,
        execution_lag=execution_lag,
        min_net_sharpe=min_net_sharpe,
    )
    return df.loc[df["viable"], "name"].tolist()


def marginal_value(
    existing_blend_weights: pd.DataFrame,
    candidate_weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 20.0,
    execution_lag: int = 1,
    candidate_allocation: float = 0.10,
) -> dict[str, float]:
    """Measure the marginal Sharpe improvement of adding a candidate rule.

    Blends (1 - candidate_allocation) * existing + candidate_allocation * candidate
    and compares net Sharpe to the existing blend alone.

    Returns
    -------
    dict with keys: base_sharpe, blended_sharpe, marginal_sharpe.
    """
    base_bt = simple_backtest(existing_blend_weights, returns, cost_bps=cost_bps, execution_lag=execution_lag)
    base_eq = base_bt.set_index("ts")["portfolio_equity"]
    base_sharpe = _sharpe(base_eq.pct_change().dropna())

    common_idx = existing_blend_weights.index.intersection(candidate_weights.index)
    common_cols = existing_blend_weights.columns.intersection(candidate_weights.columns)

    blended = (
        (1.0 - candidate_allocation) * existing_blend_weights.reindex(index=common_idx, columns=common_cols).fillna(0.0)
        + candidate_allocation * candidate_weights.reindex(index=common_idx, columns=common_cols).fillna(0.0)
    )
    blend_bt = simple_backtest(blended, returns, cost_bps=cost_bps, execution_lag=execution_lag)
    blend_eq = blend_bt.set_index("ts")["portfolio_equity"]
    blended_sharpe = _sharpe(blend_eq.pct_change().dropna())

    return {
        "base_sharpe": base_sharpe,
        "blended_sharpe": blended_sharpe,
        "marginal_sharpe": blended_sharpe - base_sharpe,
    }


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe from daily returns."""
    if len(returns) < 20:
        return 0.0
    mu = float(returns.mean())
    sigma = float(returns.std())
    if sigma < 1e-12:
        return 0.0
    return mu / sigma * np.sqrt(ANN_FACTOR)
