"""
Thin metrics wrapper for JPM Momentum research.

Wraps the existing ``compute_metrics()`` pattern and adds
momentum-specific summary helpers (turnover, skewness tables, etc.).

All functions accept ``ann_factor`` to support both crypto (365) and ETF (252).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ANN_FACTOR_CRYPTO, ANN_FACTOR_ETF


def compute_metrics(equity: pd.Series, ann_factor: float = ANN_FACTOR_CRYPTO) -> dict:
    """Compute standard performance metrics from an equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity curve indexed by datetime, starting at 1.0.
    ann_factor : float
        Annualisation factor (365 for crypto, 252 for ETFs).

    Returns
    -------
    dict
        Keys: cagr, vol, sharpe, sortino, calmar, max_dd, hit_rate,
              skewness, kurtosis, n_days, total_return.
    """
    ret = equity.pct_change().dropna()
    n = len(equity)
    if n < 2:
        return {k: np.nan for k in [
            "cagr", "vol", "sharpe", "sortino", "calmar",
            "max_dd", "hit_rate", "skewness", "kurtosis",
            "n_days", "total_return",
        ]}

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = (1 + total_return) ** (ann_factor / n) - 1.0
    vol = float(ret.std() * np.sqrt(ann_factor))
    sharpe = float((ret.mean() / ret.std()) * np.sqrt(ann_factor)) if ret.std() > 0 else np.nan

    neg = ret[ret < 0]
    sortino = (
        float((ret.mean() / neg.std()) * np.sqrt(ann_factor))
        if len(neg) > 1 and neg.std() > 0 else np.nan
    )

    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else np.nan

    hit_rate = float((ret > 0).mean())
    skewness = float(ret.skew())
    kurtosis = float(ret.kurtosis())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "n_days": n,
    }


def format_metrics_table(results: dict | list[dict], label_key: str = "label") -> str:
    """Pretty-print a metrics dict (or list of dicts) as an aligned table."""
    if isinstance(results, dict):
        results = [results]

    header = (
        f"{'Strategy':<30s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} "
        f"{'Sortino':>8s} {'MaxDD':>8s} {'Hit%':>7s} {'Skew':>7s} {'Kurt':>7s}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lbl = r.get(label_key, r.get("label", ""))
        lines.append(
            f"{lbl:<30s} "
            f"{r.get('cagr', np.nan):>7.1%} "
            f"{r.get('vol', np.nan):>7.1%} "
            f"{r.get('sharpe', np.nan):>8.2f} "
            f"{r.get('sortino', np.nan):>8.2f} "
            f"{r.get('max_dd', np.nan):>7.1%} "
            f"{r.get('hit_rate', np.nan):>6.1%} "
            f"{r.get('skewness', np.nan):>7.2f} "
            f"{r.get('kurtosis', np.nan):>7.2f}"
        )
    return "\n".join(lines)


def compute_turnover_stats(
    backtest_df: pd.DataFrame,
    ann_factor: float = ANN_FACTOR_CRYPTO,
) -> dict:
    """Compute average turnover statistics from backtest output.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output from :func:`backtest.simple_backtest`.
    ann_factor : float
        Annualisation factor.

    Returns
    -------
    dict
        Keys: avg_turnover, median_turnover, avg_gross_exposure,
              avg_cost_drag (annualised).
    """
    return {
        "avg_turnover": float(backtest_df["turnover"].mean()),
        "median_turnover": float(backtest_df["turnover"].median()),
        "avg_gross_exposure": float(backtest_df["gross_exposure"].mean()),
        "avg_cost_drag": float(backtest_df["cost_ret"].mean() * ann_factor),
    }
