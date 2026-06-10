"""Diagnostics for latent regimes before any trading rules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeEvaluation:
    """Regime diagnostics ready to serialize as research artifacts."""

    transition_matrix: pd.DataFrame
    summary: pd.DataFrame
    persistence: pd.DataFrame


def evaluate_regimes(
    frame: pd.DataFrame,
    *,
    regime_col: str = "regime",
    return_col: str = "log_return_1",
    tail_col: str = "target_tail_event_1d",
) -> RegimeEvaluation:
    """Evaluate regime behavior without converting it into trading rules."""
    if frame.empty or regime_col not in frame:
        empty = pd.DataFrame()
        return RegimeEvaluation(empty, empty, empty)

    work = frame.copy()
    work = work[work[regime_col].notna()].sort_values("ts").reset_index(drop=True)
    if work.empty:
        empty = pd.DataFrame()
        return RegimeEvaluation(empty, empty, empty)

    work[regime_col] = work[regime_col].astype(int)
    transition_matrix = _transition_matrix(work[regime_col])
    persistence = _persistence(work[regime_col])
    summary = _summary(work, regime_col, return_col, tail_col)
    return RegimeEvaluation(transition_matrix, summary, persistence)


def _transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    current = regimes.iloc[:-1].reset_index(drop=True)
    nxt = regimes.iloc[1:].reset_index(drop=True)
    counts = pd.crosstab(current, nxt, rownames=["from_regime"], colnames=["to_regime"])
    states = sorted(set(regimes.dropna().astype(int)))
    counts = counts.reindex(index=states, columns=states, fill_value=0)
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def _persistence(regimes: pd.Series) -> pd.DataFrame:
    rows = []
    for regime, group in regimes.groupby((regimes != regimes.shift()).cumsum()):
        del regime
        rows.append({"regime": int(group.iloc[0]), "run_length": int(group.shape[0])})
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs
    return (
        runs.groupby("regime")["run_length"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .rename(columns={"count": "runs"})
    )


def _summary(
    frame: pd.DataFrame,
    regime_col: str,
    return_col: str,
    tail_col: str,
) -> pd.DataFrame:
    if return_col not in frame:
        frame = frame.assign(**{return_col: np.nan})
    if tail_col not in frame:
        frame = frame.assign(**{tail_col: np.nan})

    rows = []
    for regime, group in frame.groupby(regime_col):
        returns = pd.to_numeric(group[return_col], errors="coerce").dropna()
        tails = pd.to_numeric(group[tail_col], errors="coerce")
        rows.append(
            {
                "regime": int(regime),
                "bars": int(group.shape[0]),
                "share": float(group.shape[0] / frame.shape[0]),
                "mean_return": _safe(returns.mean()),
                "median_return": _safe(returns.median()),
                "volatility": _safe(returns.std()),
                "skew": _safe(returns.skew()),
                "max_drawdown": _safe(_max_drawdown(returns)),
                "tail_participation": _safe(tails.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = np.exp(returns.cumsum())
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _safe(value: float) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
