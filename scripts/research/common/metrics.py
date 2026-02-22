"""
Shared performance metrics and formatting utilities.

Used by all paper-recreation research packages.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .data import ANN_FACTOR


def compute_metrics(equity: pd.Series) -> dict:
    """Compute standard performance metrics from an equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity curve indexed by datetime, starting at 1.0.

    Returns
    -------
    dict with keys: cagr, vol, sharpe, sortino, calmar, max_dd, hit_rate,
                    skewness, kurtosis, n_days, total_return
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
    cagr = (1 + total_return) ** (ANN_FACTOR / n) - 1.0
    _std = float(ret.std())
    vol = _std * np.sqrt(ANN_FACTOR)
    sharpe = float((ret.mean() / _std) * np.sqrt(ANN_FACTOR)) if _std > 1e-12 else np.nan

    neg = ret[ret < 0]
    neg_std = float(neg.std()) if len(neg) > 1 else 0.0
    sortino = (
        float((ret.mean() / neg_std) * np.sqrt(ANN_FACTOR))
        if neg_std > 1e-12 else np.nan
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


# ---------------------------------------------------------------------------
# Information Horizon Analysis (Phase 2)
# ---------------------------------------------------------------------------

def information_horizon(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: Sequence[int] = (1, 4, 8, 24, 48, 120, 240, 504),
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute IC at multiple forward horizons for signal evaluation.

    For each horizon h, computes the cross-sectional rank correlation between
    signal(t) and the cumulative forward return from t+1 to t+h at each
    timestep, then averages across time to produce the mean IC.

    Parameters
    ----------
    signal : pd.DataFrame
        Wide-format: index = ts (datetime), columns = symbols, values = signal.
    returns : pd.DataFrame
        Wide-format: index = ts, columns = symbols, values = single-period
        arithmetic returns (e.g. close-to-close or open-to-close).
    horizons : sequence of int
        Forward horizons in bars to evaluate.
    method : str
        Rank correlation method: 'spearman' (default) or 'pearson'.

    Returns
    -------
    pd.DataFrame with columns: horizon, ic_mean, ic_std, ic_tstat, ic_pval,
        n_periods, hit_rate (fraction of periods with positive IC).
    """
    common_idx = signal.index.intersection(returns.index).sort_values()
    common_cols = signal.columns.intersection(returns.columns)
    if len(common_cols) < 2:
        raise ValueError(
            f"Need >= 2 common symbols between signal and returns, got {len(common_cols)}"
        )

    sig = signal.reindex(index=common_idx, columns=common_cols)
    ret = returns.reindex(index=common_idx, columns=common_cols)

    results = []
    for h in horizons:
        fwd_ret = ret.rolling(window=h).sum().shift(-h)

        ic_series = _cross_sectional_ic(sig, fwd_ret, method=method)
        ic_series = ic_series.dropna()
        n = len(ic_series)

        if n < 3:
            results.append({
                "horizon": h,
                "ic_mean": np.nan, "ic_std": np.nan,
                "ic_tstat": np.nan, "ic_pval": np.nan,
                "n_periods": n, "hit_rate": np.nan,
            })
            continue

        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std(ddof=1))
        ic_tstat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else np.nan
        ic_pval = float(2 * (1 - stats.t.cdf(abs(ic_tstat), df=n - 1))) if not np.isnan(ic_tstat) else np.nan
        hit_rate = float((ic_series > 0).mean())

        results.append({
            "horizon": h,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_tstat": ic_tstat,
            "ic_pval": ic_pval,
            "n_periods": n,
            "hit_rate": hit_rate,
        })

    return pd.DataFrame(results)


def _cross_sectional_ic(
    signal: pd.DataFrame,
    forward_ret: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Compute per-timestep cross-sectional rank IC between signal and returns."""
    ic_values = {}
    for ts in signal.index:
        s = signal.loc[ts].dropna()
        r = forward_ret.loc[ts].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 3:
            continue
        s_vals = s.loc[common].values
        r_vals = r.loc[common].values
        if np.std(s_vals) == 0 or np.std(r_vals) == 0:
            continue
        if method == "spearman":
            corr, _ = stats.spearmanr(s_vals, r_vals)
        else:
            corr, _ = stats.pearsonr(s_vals, r_vals)
        ic_values[ts] = corr
    return pd.Series(ic_values, dtype=float)


def compute_ic_decay(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10, 21, 42, 63, 126, 252),
    method: str = "spearman",
) -> pd.DataFrame:
    """Measure how a signal's predictive power attenuates over forward horizons.

    Wraps ``information_horizon`` and adds a normalised decay curve: IC at
    each horizon divided by the IC at the shortest horizon.  This lets you
    compare decay rates across signals with different absolute IC levels.

    Following Rahimikia et al. (2025), who find that TSFMs exhibit slower
    IC decay than tree-based ensembles — a useful diagnostic for gauging
    signal longevity.

    Parameters
    ----------
    signal, returns : pd.DataFrame
        Wide-format, index=datetime, columns=symbols.
    horizons : sequence of int
        Forward horizons in bars to evaluate.
    method : str
        'spearman' or 'pearson'.

    Returns
    -------
    pd.DataFrame with columns from ``information_horizon`` plus:
        ic_normalised (IC / IC_at_min_horizon), decay_rate (linear slope of
        normalised IC vs log-horizon).
    """
    ih = information_horizon(signal, returns, horizons=horizons, method=method)
    base_ic = ih.loc[ih["horizon"] == min(horizons), "ic_mean"]
    base_val = float(base_ic.iloc[0]) if len(base_ic) > 0 and not np.isnan(base_ic.iloc[0]) else 0.0

    if abs(base_val) > 1e-8:
        ih["ic_normalised"] = ih["ic_mean"] / base_val
    else:
        ih["ic_normalised"] = np.nan

    valid = ih.dropna(subset=["ic_normalised"])
    if len(valid) >= 3:
        log_h = np.log(valid["horizon"].values.astype(float))
        ic_n = valid["ic_normalised"].values
        slope, _, _, _, _ = stats.linregress(log_h, ic_n)
        ih["decay_rate"] = float(slope)
    else:
        ih["decay_rate"] = np.nan

    return ih


def compute_yearly_sharpe_trend(
    equity: pd.Series,
) -> pd.DataFrame:
    """Track year-over-year Sharpe to detect strengthening or degradation.

    Following the expanding-window evaluation design of Rahimikia et al.
    (2025), computes annualised Sharpe for each calendar year and fits a
    linear trend.

    Parameters
    ----------
    equity : pd.Series
        Equity curve indexed by datetime, starting at 1.0.

    Returns
    -------
    pd.DataFrame with columns: year, sharpe, cumulative_sharpe,
        trend_slope, trend_label.
    """
    ret = equity.pct_change().dropna()
    if len(ret) < 2:
        return pd.DataFrame(columns=["year", "sharpe", "cumulative_sharpe",
                                      "trend_slope", "trend_label"])

    ret.index = pd.to_datetime(ret.index)
    years = sorted(ret.index.year.unique())

    rows = []
    for yr in years:
        yr_ret = ret[ret.index.year == yr]
        if len(yr_ret) < 20:
            continue
        mu = yr_ret.mean()
        sigma = yr_ret.std()
        sr = float((mu / sigma) * np.sqrt(ANN_FACTOR)) if sigma > 1e-12 else np.nan

        cum_ret = ret[ret.index.year <= yr]
        cum_mu = cum_ret.mean()
        cum_sigma = cum_ret.std()
        cum_sr = float((cum_mu / cum_sigma) * np.sqrt(ANN_FACTOR)) if cum_sigma > 1e-12 else np.nan

        rows.append({"year": yr, "sharpe": sr, "cumulative_sharpe": cum_sr})

    df = pd.DataFrame(rows)
    if len(df) < 3:
        df["trend_slope"] = np.nan
        df["trend_label"] = "INSUFFICIENT_DATA"
        return df

    x = np.arange(len(df), dtype=float)
    y = df["sharpe"].values
    mask = ~np.isnan(y)
    if mask.sum() >= 3:
        slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
        df["trend_slope"] = float(slope)
        if slope < -0.05:
            df["trend_label"] = "DEGRADING"
        elif slope > 0.05:
            df["trend_label"] = "STRENGTHENING"
        else:
            df["trend_label"] = "STABLE"
    else:
        df["trend_slope"] = np.nan
        df["trend_label"] = "INSUFFICIENT_DATA"

    return df


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
