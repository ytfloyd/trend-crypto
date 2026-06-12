"""Convexity-first metrics.

Pure functions. No state. Handles edge cases gracefully (empty series,
all-zero, NaN, divide-by-zero). All returns are floats; None where the
metric is undefined.

See docs/research/convexity_alpha_pipeline_spec.md, section 4 for the
formal definitions. Each function below documents its formula.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Basic return-distribution metrics
# ---------------------------------------------------------------------------

def skew(returns: pd.Series) -> Optional[float]:
    """Sample skewness of returns. Standard third-moment definition.

    Returns None if the series is too short to compute (< 3 non-NaN values).
    """
    r = pd.Series(returns).dropna()
    if len(r) < 3:
        return None
    return float(stats.skew(r.values, bias=False))


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> Optional[float]:
    """Annualized Sharpe. Reported but not used as gate.

    Sharpe = mean / std * sqrt(periods_per_year).
    Returns None if std is zero or series too short.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return None
    s = r.std(ddof=1)
    # Tolerance check: FP-equality on identical inputs leaves tiny std
    if s < 1e-12 or not np.isfinite(s):
        return None
    return float(r.mean() / s * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.25 = -25% DD).

    equity is a cumulative equity curve (1.0 = flat from start).
    Returns 0.0 if no drawdown observed.
    """
    e = pd.Series(equity).dropna()
    if len(e) < 2:
        return 0.0
    peak = e.cummax()
    dd = (e - peak) / peak
    return float(-dd.min())


def calmar(returns: pd.Series, periods_per_year: int = 252,
           min_dd_floor: float = 0.05) -> Optional[float]:
    """Annualized return / |Max Drawdown|.

    Uses min_dd_floor to prevent explosion when DD is tiny. Defaults to 5% floor.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return None
    ann_return = (1 + r.mean()) ** periods_per_year - 1
    equity = (1 + r).cumprod()
    dd = max_drawdown(equity)
    denom = max(dd, min_dd_floor)
    if denom == 0:
        return None
    return float(ann_return / denom)


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------

def payoff_ratio(trade_pnls: pd.Series) -> Optional[float]:
    """mean(win PnL) / |mean(loss PnL)|. None if no losses (would be inf)."""
    p = pd.Series(trade_pnls).dropna()
    wins = p[p > 0]
    losses = p[p < 0]
    if len(wins) == 0 or len(losses) == 0:
        return None
    return float(wins.mean() / abs(losses.mean()))


def profit_factor(trade_pnls: pd.Series) -> Optional[float]:
    """sum(wins) / |sum(losses)|. None if no losses."""
    p = pd.Series(trade_pnls).dropna()
    wins_sum = p[p > 0].sum()
    losses_sum = p[p < 0].sum()
    if losses_sum == 0:
        return None
    return float(wins_sum / abs(losses_sum))


def hit_rate(trade_pnls: pd.Series) -> Optional[float]:
    """Fraction of trades with positive PnL. None if no trades."""
    p = pd.Series(trade_pnls).dropna()
    if len(p) == 0:
        return None
    return float((p > 0).mean())


def max_consecutive_losses(trade_pnls: pd.Series) -> int:
    """Longest run of consecutive negative-PnL trades."""
    p = pd.Series(trade_pnls).dropna()
    if len(p) == 0:
        return 0
    losing = (p < 0).astype(int)
    if losing.sum() == 0:
        return 0
    # Run-length encoding via groupby on differences
    groups = (losing != losing.shift()).cumsum()
    runs = losing.groupby(groups).sum()
    return int(runs.max())


def trade_duration_stats(durations: pd.Series) -> Dict[str, float]:
    """Median, mean, IQR of trade durations in bars."""
    d = pd.Series(durations).dropna()
    if len(d) == 0:
        return {"median": 0.0, "mean": 0.0, "iqr": 0.0, "n_trades": 0}
    return {
        "median": float(d.median()),
        "mean": float(d.mean()),
        "iqr": float(d.quantile(0.75) - d.quantile(0.25)),
        "n_trades": int(len(d)),
    }


# ---------------------------------------------------------------------------
# Convexity-specific metrics
# ---------------------------------------------------------------------------

def tail_capture(
    alpha_returns: pd.Series,
    underlying_returns: pd.Series,
    decile: float = 0.10,
) -> Optional[float]:
    """Fraction of underlying's top-decile |move| that alpha captured in correct direction.

    Definition:
        top_moves = days where |r_underlying| is in top `decile` of |r_underlying|.
        capture = sum(alpha_pnl during top_moves with correct sign) /
                  sum(|r_underlying| during top_moves)

    Returns a value in roughly [0, 1] though it can exceed 1 if alpha overshoots.
    Returns None if no top-decile observations.
    """
    a = pd.Series(alpha_returns).dropna()
    u = pd.Series(underlying_returns).dropna()
    common = a.index.intersection(u.index)
    if len(common) < 20:
        return None
    a = a.reindex(common)
    u = u.reindex(common)

    abs_u = u.abs()
    threshold = abs_u.quantile(1.0 - decile)
    top_mask = abs_u >= threshold
    if top_mask.sum() == 0:
        return None

    # On top-decile days, alpha PnL aligned with underlying direction
    aligned = np.where(u[top_mask] > 0, a[top_mask], -a[top_mask])
    aligned_sum = float(np.sum(aligned[aligned > 0]))
    underlying_abs_sum = float(abs_u[top_mask].sum())
    if underlying_abs_sum == 0:
        return None
    return aligned_sum / underlying_abs_sum


def convexity_beta(
    alpha_returns: pd.Series,
    underlying_returns: pd.Series,
) -> Tuple[Optional[float], Optional[float]]:
    """Regress alpha returns on |underlying| + signed underlying.

    Model: r_alpha = a + b * |r_underlying| + c * r_underlying + e

    Returns (b, p_value_of_b). b > 0 = long-vol payoff (convex);
    b < 0 = concave; b ~ 0 = neutral.
    Returns (None, None) if regression cannot be fit.
    """
    a = pd.Series(alpha_returns).dropna()
    u = pd.Series(underlying_returns).dropna()
    common = a.index.intersection(u.index)
    if len(common) < 30:
        return (None, None)
    a = a.reindex(common).values
    u = u.reindex(common).values
    abs_u = np.abs(u)

    X = np.column_stack([np.ones(len(u)), abs_u, u])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, a, rcond=None)
        residuals = a - X @ beta
        n = len(a)
        k = X.shape[1]
        if n - k <= 0:
            return (float(beta[1]), None)
        sigma2 = (residuals @ residuals) / (n - k)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        b_se = float(np.sqrt(cov[1, 1]))
        if b_se == 0 or not np.isfinite(b_se):
            return (float(beta[1]), None)
        t_stat = float(beta[1]) / b_se
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - k)))
        return (float(beta[1]), p_val)
    except (np.linalg.LinAlgError, ValueError):
        return (None, None)


def tail_sharpe_asymmetry(returns: pd.Series) -> Optional[float]:
    """Sharpe of positive-return regime / |Sharpe of negative-return regime|.

    Convex alphas have TSA > 1. Returns None if either regime has < 5 obs.
    """
    r = pd.Series(returns).dropna()
    pos = r[r > 0]
    neg = r[r < 0]
    if len(pos) < 5 or len(neg) < 5:
        return None
    pos_sharpe = pos.mean() / pos.std(ddof=1) if pos.std(ddof=1) > 0 else None
    neg_sharpe = neg.mean() / neg.std(ddof=1) if neg.std(ddof=1) > 0 else None
    if pos_sharpe is None or neg_sharpe is None or neg_sharpe == 0:
        return None
    return float(pos_sharpe / abs(neg_sharpe))


def pain_ratio(equity: pd.Series) -> Optional[float]:
    """avg drawdown depth / avg underwater duration (in bars).

    Lower is better - captures pain-to-hold during DD.
    """
    e = pd.Series(equity).dropna()
    if len(e) < 2:
        return None
    peak = e.cummax()
    dd = (e - peak) / peak
    under_water = dd < 0
    if under_water.sum() == 0:
        return 0.0
    groups = (under_water != under_water.shift()).cumsum()
    spell_lens = under_water.groupby(groups).sum()
    spell_lens = spell_lens[spell_lens > 0]
    spell_depths = dd.groupby(groups).min().abs()
    spell_depths = spell_depths.reindex(spell_lens.index)
    if len(spell_lens) == 0:
        return 0.0
    avg_dur = float(spell_lens.mean())
    avg_dep = float(spell_depths.mean())
    if avg_dur == 0:
        return None
    return avg_dep / avg_dur


# ---------------------------------------------------------------------------
# Composite Convexity Score (CCS)
# ---------------------------------------------------------------------------

def _skew_factor(s: Optional[float]) -> float:
    """skew_factor(s) = 1 + 0.5 * clip(s, -1, 2)"""
    if s is None:
        return 1.0
    return float(1.0 + 0.5 * np.clip(s, -1.0, 2.0))


def _payoff_factor(p: Optional[float]) -> float:
    """payoff_factor(p) = sqrt(max(p, 0.5))"""
    if p is None:
        return 1.0
    return float(np.sqrt(max(p, 0.5)))


def composite_convexity_score(
    returns: pd.Series,
    trade_pnls: pd.Series,
    alpha_returns_for_capture: Optional[pd.Series] = None,
    underlying_returns_for_capture: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    min_dd_floor: float = 0.05,
    tail_decile: float = 0.10,
) -> Optional[float]:
    """Composite Convexity Score (CCS).

    CCS = Calmar * skew_factor(skew) * payoff_factor(payoff_ratio) * tail_capture

    The primary screening metric at each stage of the pipeline. See spec section 4.2.

    Returns None if Calmar cannot be computed. Tail capture is set to 1.0
    (no penalty/boost) if not enough underlying data is provided.
    """
    cal = calmar(returns, periods_per_year=periods_per_year, min_dd_floor=min_dd_floor)
    if cal is None:
        return None
    sk = skew(returns)
    pf_r = payoff_ratio(trade_pnls)

    if alpha_returns_for_capture is not None and underlying_returns_for_capture is not None:
        tc = tail_capture(
            alpha_returns_for_capture,
            underlying_returns_for_capture,
            decile=tail_decile,
        )
        if tc is None:
            tc = 1.0
    else:
        tc = 1.0

    return float(cal * _skew_factor(sk) * _payoff_factor(pf_r) * max(tc, 0.0))


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def calculate_all(
    alpha_returns: pd.Series,
    underlying_returns: pd.Series,
    equity: pd.Series,
    trade_pnls: pd.Series,
    trade_durations: pd.Series,
    periods_per_year: int = 252,
    min_dd_floor: float = 0.05,
    tail_decile: float = 0.10,
) -> Dict[str, Any]:
    """Compute every metric. Returns a dict suitable for logging."""
    b, b_p = convexity_beta(alpha_returns, underlying_returns)
    duration = trade_duration_stats(trade_durations)
    out: Dict[str, Any] = {
        "skew": skew(alpha_returns),
        "sharpe": sharpe(alpha_returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(
            alpha_returns,
            periods_per_year=periods_per_year,
            min_dd_floor=min_dd_floor,
        ),
        "payoff_ratio": payoff_ratio(trade_pnls),
        "profit_factor": profit_factor(trade_pnls),
        "hit_rate": hit_rate(trade_pnls),
        "max_consecutive_losses": max_consecutive_losses(trade_pnls),
        "tail_capture": tail_capture(
            alpha_returns, underlying_returns, decile=tail_decile
        ),
        "convexity_beta_b": b,
        "convexity_beta_p": b_p,
        "tail_sharpe_asymmetry": tail_sharpe_asymmetry(alpha_returns),
        "pain_ratio": pain_ratio(equity),
        "n_trades": duration["n_trades"],
        "median_trade_duration": duration["median"],
        "mean_trade_duration": duration["mean"],
        "iqr_trade_duration": duration["iqr"],
    }
    out["ccs"] = composite_convexity_score(
        returns=alpha_returns,
        trade_pnls=trade_pnls,
        alpha_returns_for_capture=alpha_returns,
        underlying_returns_for_capture=underlying_returns,
        periods_per_year=periods_per_year,
        min_dd_floor=min_dd_floor,
        tail_decile=tail_decile,
    )
    return out
