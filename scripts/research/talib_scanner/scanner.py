"""
Phase 1: IC-based scan across all TA-Lib features.

For each feature, computes cross-sectional Spearman IC with forward returns
at multiple horizons. Produces a ranked table of features by predictive power.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _cross_sectional_ic_series(
    signal: pd.Series,
    forward_ret: pd.Series,
    ts_col: pd.Series,
    min_assets: int = 10,
) -> pd.Series:
    """Compute per-date cross-sectional Spearman IC from long-format data."""
    df = pd.DataFrame({
        "ts": ts_col.values,
        "sig": signal.values,
        "fwd": forward_ret.values,
    }).dropna()

    ic_vals = {}
    for ts, grp in df.groupby("ts"):
        if len(grp) < min_assets:
            continue
        s = grp["sig"].values
        r = grp["fwd"].values
        if np.std(s) == 0 or np.std(r) == 0:
            continue
        corr, _ = stats.spearmanr(s, r)
        if not np.isnan(corr):
            ic_vals[ts] = corr

    return pd.Series(ic_vals, dtype=float)


def run_ic_scan(
    panel: pd.DataFrame,
    feature_cols: list[str],
    forward_horizons: list[int] = (1, 3, 5, 7, 10, 14, 21),
    min_assets: int = 10,
    min_history: int = 365,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run IC scan across all features and horizons.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel with columns: symbol, ts, close, + feature columns.
    feature_cols : list[str]
        Feature column names to evaluate.
    forward_horizons : list[int]
        Forward return horizons in days.
    min_assets : int
        Minimum assets per date for cross-sectional IC.
    min_history : int
        Minimum bars per asset to include.

    Returns
    -------
    pd.DataFrame with columns: feature, horizon, ic_mean, ic_std, ic_tstat,
        ic_pval, n_periods, hit_rate, abs_ic_mean, sign
    """
    panel = panel.copy()
    panel["ts"] = pd.to_datetime(panel["ts"])
    panel = panel.sort_values(["symbol", "ts"])

    # filter assets with enough history
    counts = panel.groupby("symbol")["ts"].count()
    valid_syms = counts[counts >= min_history].index
    panel = panel[panel["symbol"].isin(valid_syms)]

    if verbose:
        print(f"[scanner] {len(valid_syms)} assets, {len(feature_cols)} features, "
              f"{len(forward_horizons)} horizons")

    # pre-compute forward returns
    fwd_ret_cols = {}
    for h in forward_horizons:
        col = f"_fwd_{h}d"
        panel[col] = panel.groupby("symbol")["close"].pct_change(h).shift(-h)
        fwd_ret_cols[h] = col

    rows = []
    n_total = len(feature_cols) * len(forward_horizons)
    done = 0

    for feat in feature_cols:
        if feat not in panel.columns:
            continue

        for h in forward_horizons:
            fwd_col = fwd_ret_cols[h]

            mask = panel[feat].notna() & panel[fwd_col].notna()
            sub = panel.loc[mask]

            if len(sub) < 100:
                done += 1
                continue

            ic_series = _cross_sectional_ic_series(
                sub[feat], sub[fwd_col], sub["ts"], min_assets=min_assets
            )

            n = len(ic_series)
            if n < 10:
                done += 1
                continue

            ic_mean = ic_series.mean()
            ic_std = ic_series.std(ddof=1)
            ic_tstat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0
            ic_pval = 2 * (1 - stats.t.cdf(abs(ic_tstat), df=n - 1)) if ic_std > 0 else 1.0
            hit_rate = (ic_series > 0).mean()

            rows.append({
                "feature": feat,
                "horizon": h,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_tstat": ic_tstat,
                "ic_pval": ic_pval,
                "n_periods": n,
                "hit_rate": hit_rate,
                "abs_ic_mean": abs(ic_mean),
                "sign": "+" if ic_mean > 0 else "-",
            })

            done += 1
            if verbose and done % 100 == 0:
                print(f"  [{done}/{n_total}] completed...")

    result = pd.DataFrame(rows)
    if verbose:
        print(f"[scanner] Done. {len(result)} valid (feature, horizon) pairs.")
    return result


def rank_features(
    scan_results: pd.DataFrame,
    horizon: int | None = None,
    min_periods: int = 100,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Rank features by absolute IC, with multiple-testing correction.

    Parameters
    ----------
    scan_results : pd.DataFrame
        Output of run_ic_scan.
    horizon : int or None
        If specified, rank only at this horizon. Otherwise use best horizon per feature.
    min_periods : int
        Minimum number of IC periods to include.
    fdr_alpha : float
        FDR significance threshold.

    Returns
    -------
    pd.DataFrame ranked by abs_ic_mean descending.
    """
    df = scan_results[scan_results["n_periods"] >= min_periods].copy()

    if horizon is not None:
        df = df[df["horizon"] == horizon]
        ranked = df.sort_values("abs_ic_mean", ascending=False).reset_index(drop=True)
    else:
        # best horizon per feature
        idx = df.groupby("feature")["abs_ic_mean"].idxmax()
        ranked = df.loc[idx].sort_values("abs_ic_mean", ascending=False).reset_index(drop=True)
        ranked = ranked.rename(columns={"horizon": "best_horizon"})

    # Benjamini-Hochberg FDR correction
    n_tests = len(ranked)
    if n_tests > 0:
        ranked = ranked.sort_values("ic_pval")
        ranked["bh_rank"] = range(1, n_tests + 1)
        ranked["bh_threshold"] = ranked["bh_rank"] / n_tests * fdr_alpha
        ranked["significant_bh"] = ranked["ic_pval"] <= ranked["bh_threshold"]
        ranked = ranked.sort_values("abs_ic_mean", ascending=False).reset_index(drop=True)
        ranked["rank"] = range(1, n_tests + 1)

    return ranked


def compute_feature_correlation(
    panel: pd.DataFrame,
    feature_cols: list[str],
    method: str = "spearman",
    sample_frac: float = 0.1,
) -> pd.DataFrame:
    """Compute pairwise correlation matrix between features.

    Uses a random sample of rows for speed.
    """
    sub = panel[feature_cols].dropna()
    if len(sub) > 10000:
        sub = sub.sample(n=int(len(sub) * sample_frac), random_state=42)
    return sub.corr(method=method)
