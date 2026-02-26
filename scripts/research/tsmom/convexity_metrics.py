"""
Convexity-specific metrics for the TSMOM long convexity engine.

Extends the standard metrics (Sharpe, CAGR, MaxDD) with diagnostics
that measure the *shape* of the return distribution — the property
we are actually trying to engineer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import ANN_FACTOR
from common.metrics import compute_metrics, compute_regime


# Re-export under the name used throughout this module
classify_regime = compute_regime


# -----------------------------------------------------------------------
# Core convexity diagnostics
# -----------------------------------------------------------------------

def compute_convexity_metrics(
    equity: pd.Series,
    weights: pd.DataFrame | None = None,
) -> dict:
    """Compute standard metrics plus convexity-specific diagnostics.

    Returns dict with keys from compute_metrics() plus:
      skewness, kurtosis (already in compute_metrics)
      avg_win, avg_loss, win_loss_ratio,
      time_in_market (if weights provided)
    """
    base = compute_metrics(equity)
    ret = equity.pct_change().dropna()

    wins = ret[ret > 0]
    losses = ret[ret < 0]

    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss_ratio = (
        abs(avg_win / avg_loss) if abs(avg_loss) > 1e-12 else np.nan
    )

    base["avg_win"] = avg_win
    base["avg_loss"] = avg_loss
    base["win_loss_ratio"] = win_loss_ratio

    if weights is not None:
        gross = weights.abs().sum(axis=1)
        invested = (gross > 1e-6).astype(float)
        base["time_in_market"] = float(invested.mean())
    else:
        base["time_in_market"] = np.nan

    return base


# -----------------------------------------------------------------------
# Regime analysis
# -----------------------------------------------------------------------

def conditional_correlation(
    strategy_ret: pd.Series,
    btc_ret: pd.Series,
    regime: pd.Series,
) -> dict[str, float]:
    """Compute correlation of strategy to BTC separately per regime."""
    common = strategy_ret.index.intersection(btc_ret.index).intersection(regime.index)
    s = strategy_ret.reindex(common)
    b = btc_ret.reindex(common)
    r = regime.reindex(common)

    result = {}
    for label in ["BULL", "BEAR", "CHOP"]:
        mask = r == label
        s_r = s[mask]
        b_r = b[mask]
        if len(s_r) < 10:
            result[label] = np.nan
            continue
        result[label] = float(s_r.corr(b_r))
    return result


def time_in_market_by_regime(
    weights: pd.DataFrame,
    regime: pd.Series,
) -> dict[str, float]:
    """Fraction of days invested during each regime."""
    common = weights.index.intersection(regime.index)
    gross = weights.reindex(common).abs().sum(axis=1)
    invested = gross > 1e-6
    r = regime.reindex(common)

    result = {}
    for label in ["BULL", "BEAR", "CHOP"]:
        mask = r == label
        n = mask.sum()
        if n < 1:
            result[label] = np.nan
            continue
        result[label] = float(invested[mask].mean())
    return result


def regime_sharpe_skew(
    strategy_ret: pd.Series,
    regime: pd.Series,
) -> dict[str, dict]:
    """Per-regime Sharpe and skewness."""
    common = strategy_ret.index.intersection(regime.index)
    s = strategy_ret.reindex(common)
    r = regime.reindex(common)

    result = {}
    for label in ["BULL", "BEAR", "CHOP"]:
        mask = r == label
        sr = s[mask]
        n = len(sr)
        if n < 10:
            result[label] = {"sharpe": np.nan, "skewness": np.nan, "n_days": n}
            continue
        std = float(sr.std())
        sharpe = float((sr.mean() / std) * np.sqrt(ANN_FACTOR)) if std > 1e-12 else np.nan
        skew = float(sr.skew())
        result[label] = {"sharpe": round(sharpe, 3), "skewness": round(skew, 3), "n_days": n}
    return result


# -----------------------------------------------------------------------
# Participation ratio
# -----------------------------------------------------------------------

def participation_ratio_portfolio(
    strategy_equity: pd.Series,
    btc_equity: pd.Series,
    decile_threshold: float = 0.90,
) -> float:
    """Portfolio-level participation: fraction of top-decile BTC monthly
    returns where the strategy also had a positive monthly return.
    """
    strat_monthly = strategy_equity.resample("ME").last().pct_change().dropna()
    btc_monthly = btc_equity.resample("ME").last().pct_change().dropna()

    common = strat_monthly.index.intersection(btc_monthly.index)
    if len(common) < 5:
        return np.nan
    strat_m = strat_monthly.reindex(common)
    btc_m = btc_monthly.reindex(common)

    threshold = btc_m.quantile(decile_threshold)
    top_months = btc_m >= threshold
    if top_months.sum() == 0:
        return np.nan

    return float((strat_m[top_months] > 0).mean())


def participation_ratio_per_asset(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    universe: pd.DataFrame,
    decile_threshold: float = 0.90,
) -> float:
    """Per-asset participation: for each asset-month in the top decile of
    monthly returns (within the tradeable universe at that date), was the
    strategy long that asset during that month?
    """
    monthly_ret = returns_wide.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_univ = universe.resample("ME").last().fillna(False)

    hits = 0
    total = 0

    monthly_w = weights.resample("ME").mean()

    for dt in monthly_ret.index:
        if dt not in monthly_univ.index:
            continue
        univ_mask = monthly_univ.loc[dt]
        rets = monthly_ret.loc[dt][univ_mask]
        rets = rets.dropna()
        if len(rets) < 5:
            continue

        threshold = rets.quantile(decile_threshold)
        top_assets = rets[rets >= threshold].index

        if dt in monthly_w.index:
            w_row = monthly_w.loc[dt]
            for asset in top_assets:
                total += 1
                if asset in w_row.index and w_row[asset] > 1e-6:
                    hits += 1
        else:
            total += len(top_assets)

    return hits / total if total > 0 else np.nan


# -----------------------------------------------------------------------
# Bootstrap confidence intervals
# -----------------------------------------------------------------------

def bootstrap_ci(
    returns: pd.Series,
    stat_fn: callable,
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for an arbitrary statistic.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    vals = returns.values
    n = len(vals)
    point = stat_fn(vals)

    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = vals[rng.randint(0, n, size=n)]
        boot_stats[b] = stat_fn(sample)

    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(boot_stats, 100 * alpha))
    hi = float(np.nanpercentile(boot_stats, 100 * (1 - alpha)))
    return float(point), lo, hi


def bootstrap_sharpe(returns: pd.Series, **kwargs) -> tuple[float, float, float]:
    def _sharpe(vals):
        std = np.std(vals, ddof=1)
        return (np.mean(vals) / std) * np.sqrt(ANN_FACTOR) if std > 1e-12 else np.nan
    return bootstrap_ci(returns, _sharpe, **kwargs)


def bootstrap_skewness(returns: pd.Series, **kwargs) -> tuple[float, float, float]:
    def _skew(vals):
        from scipy.stats import skew
        return skew(vals, nan_policy="omit")
    return bootstrap_ci(returns, _skew, **kwargs)


# -----------------------------------------------------------------------
# Crisis timeline data
# -----------------------------------------------------------------------

CRISIS_EPISODES = {
    "2018 Bear": ("2018-01-01", "2018-12-31"),
    "Mar 2020": ("2020-02-15", "2020-05-15"),
    "May 2021": ("2021-04-15", "2021-07-31"),
    "Nov 2022 (FTX)": ("2022-10-01", "2023-01-31"),
}


def extract_crisis_timeline(
    btc_close: pd.Series,
    portfolio_weights: pd.DataFrame,
    portfolio_ret: pd.Series,
    episode_name: str,
) -> pd.DataFrame | None:
    """Extract a narrow time series for a single crisis episode.

    Returns DataFrame with columns: btc_price, total_weight, daily_pnl
    """
    if episode_name not in CRISIS_EPISODES:
        return None
    start, end = CRISIS_EPISODES[episode_name]

    mask = (btc_close.index >= pd.Timestamp(start)) & (btc_close.index <= pd.Timestamp(end))
    btc_slice = btc_close[mask]
    if btc_slice.empty:
        return None

    common_idx = btc_slice.index
    total_wt = portfolio_weights.reindex(common_idx).abs().sum(axis=1).fillna(0.0)
    pnl = portfolio_ret.reindex(common_idx).fillna(0.0)

    return pd.DataFrame({
        "btc_price": btc_slice,
        "total_weight": total_wt,
        "daily_pnl": pnl,
    })
