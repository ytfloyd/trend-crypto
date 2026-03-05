"""
Phase 2: Conditional-return edge analysis for top indicators.

For each indicator, sweeps quantile-based thresholds and measures:
- Conditional vs unconditional forward returns
- Bootstrap significance
- Temporal stability (first/second half, year-by-year)
- Market-cap tier splits
- Regime conditioning (BULL/BEAR/CHOP)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dedupe_signals(sig: pd.Series, gap: int = 5) -> pd.DatetimeIndex:
    dates = sig[sig].index
    clean, last = [], None
    for t in dates:
        if last is None or (t - last).days >= gap:
            clean.append(t)
            last = t
    return pd.DatetimeIndex(clean)


def run_edge_analysis(
    panel: pd.DataFrame,
    feature_col: str,
    forward_days: list[int] = (1, 3, 5, 7, 10, 14, 21),
    quantile_thresholds: list[float] = (0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90),
    min_events: int = 15,
    n_bootstrap: int = 10000,
    dedup_gap: int = 5,
    ic_sign: str = "+",
) -> dict:
    """Run full edge analysis for a single indicator.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format with symbol, ts, close, and the feature column.
    feature_col : str
        Name of the indicator column.
    forward_days : list[int]
        Forward return horizons.
    quantile_thresholds : list[float]
        Quantile thresholds to sweep (relative to per-asset distribution).
    min_events : int
        Minimum trigger events per asset.
    n_bootstrap : int
        Number of bootstrap iterations for significance test.
    dedup_gap : int
        Minimum days between trigger events.
    ic_sign : str
        "+" means high values are bullish; "-" means low values are bullish.

    Returns
    -------
    dict with keys: summary, threshold_sweep, per_asset, temporal,
        regime, bootstrap, tier_analysis
    """
    panel = panel.copy()
    panel["ts"] = pd.to_datetime(panel["ts"])

    results = {"feature": feature_col, "ic_sign": ic_sign}

    # ── Threshold sweep across all assets ─────────────────────────────
    sweep_rows = []

    for sym, gdf in panel.groupby("symbol"):
        gdf = gdf.sort_values("ts").set_index("ts")
        if gdf[feature_col].notna().sum() < 200:
            continue

        fwd_rets = {}
        for fd in forward_days:
            fwd_rets[fd] = gdf["close"].pct_change(fd).shift(-fd)

        feat_vals = gdf[feature_col].dropna()

        for q in quantile_thresholds:
            thresh_val = feat_vals.quantile(q)

            if ic_sign == "+":
                cross = (feat_vals > thresh_val) & (feat_vals.shift(1) <= thresh_val)
            else:
                cross = (feat_vals < thresh_val) & (feat_vals.shift(1) >= thresh_val)

            trigs = _dedupe_signals(cross, gap=dedup_gap)

            for fd in forward_days:
                cond = fwd_rets[fd].loc[fwd_rets[fd].index.isin(trigs)].dropna()
                unc = fwd_rets[fd].dropna()

                if len(cond) < 5:
                    continue

                sweep_rows.append({
                    "symbol": sym,
                    "quantile": q,
                    "fwd_days": fd,
                    "n": len(cond),
                    "mean": cond.mean(),
                    "median": cond.median(),
                    "hit": (cond > 0).mean(),
                    "std": cond.std(),
                    "unc_mean": unc.mean(),
                    "edge": cond.mean() - unc.mean(),
                })

    sweep_df = pd.DataFrame(sweep_rows)
    results["threshold_sweep"] = sweep_df

    if sweep_df.empty:
        results["summary"] = "No valid events found."
        return results

    # ── Pooled summary by quantile and horizon ────────────────────────
    pooled = (
        sweep_df.groupby(["quantile", "fwd_days"])
        .agg(
            n_assets=("symbol", "nunique"),
            total_events=("n", "sum"),
            avg_mean=("mean", "mean"),
            avg_hit=("hit", "mean"),
            avg_edge=("edge", "mean"),
            pct_positive_edge=("edge", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )
    results["pooled_sweep"] = pooled

    # ── Best quantile (by avg_edge at 7d) ─────────────────────────────
    p7 = pooled[pooled["fwd_days"] == 7]
    if not p7.empty:
        best_q = p7.loc[p7["avg_edge"].idxmax(), "quantile"]
    else:
        best_q = 0.80
    results["best_quantile"] = best_q

    # ── Per-asset results at best quantile, 7d ────────────────────────
    per_asset = sweep_df[
        (sweep_df["quantile"] == best_q) & (sweep_df["fwd_days"] == 7)
    ].sort_values("edge", ascending=False)
    results["per_asset"] = per_asset

    # ── Temporal stability (pooled, best quantile, 7d) ────────────────
    best_events = sweep_df[
        (sweep_df["quantile"] == best_q) & (sweep_df["fwd_days"] == 7)
    ]

    # collect actual event-level returns for temporal + bootstrap
    all_cond_vals = []
    all_unc_vals = []

    for sym, gdf in panel.groupby("symbol"):
        gdf = gdf.sort_values("ts").set_index("ts")
        if gdf[feature_col].notna().sum() < 200:
            continue

        feat_vals = gdf[feature_col].dropna()
        thresh_val = feat_vals.quantile(best_q)

        if ic_sign == "+":
            cross = (feat_vals > thresh_val) & (feat_vals.shift(1) <= thresh_val)
        else:
            cross = (feat_vals < thresh_val) & (feat_vals.shift(1) >= thresh_val)

        trigs = _dedupe_signals(cross, gap=dedup_gap)

        fwd7 = gdf["close"].pct_change(7).shift(-7)
        cond = fwd7.loc[fwd7.index.isin(trigs)].dropna()
        unc = fwd7.dropna()

        all_cond_vals.append(cond)
        all_unc_vals.append(unc)

    if all_cond_vals:
        pooled_cond = pd.concat(all_cond_vals)
        pooled_unc = pd.concat(all_unc_vals)
        pooled_cond.index = pd.to_datetime(pooled_cond.index)

        # year-by-year
        yearly = {}
        for year in sorted(pooled_cond.index.year.unique()):
            yr = pooled_cond[pooled_cond.index.year == year]
            if len(yr) >= 5:
                yearly[year] = {
                    "n": len(yr),
                    "mean": yr.mean(),
                    "hit": (yr > 0).mean(),
                }
        results["temporal_yearly"] = yearly

        # first vs second half
        sorted_c = pooled_cond.sort_index()
        mid = len(sorted_c) // 2
        h1, h2 = sorted_c.iloc[:mid], sorted_c.iloc[mid:]
        results["temporal_halves"] = {
            "first": {"n": len(h1), "mean": h1.mean(), "hit": (h1 > 0).mean(),
                      "start": str(h1.index[0].date()), "end": str(h1.index[-1].date())},
            "second": {"n": len(h2), "mean": h2.mean(), "hit": (h2 > 0).mean(),
                       "start": str(h2.index[0].date()), "end": str(h2.index[-1].date())},
        }

        # bootstrap significance
        rng = np.random.default_rng(42)
        obs_mean = pooled_cond.mean()
        obs_hit = (pooled_cond > 0).mean()
        n_cond = len(pooled_cond)
        unc_arr = pooled_unc.values

        boot_means = np.array([
            rng.choice(unc_arr, size=n_cond, replace=True).mean()
            for _ in range(n_bootstrap)
        ])

        if ic_sign == "+":
            p_val = (boot_means >= obs_mean).mean()
        else:
            p_val = (boot_means <= obs_mean).mean()

        z_score = ((obs_mean - boot_means.mean()) / boot_means.std()
                   if boot_means.std() > 0 else 0)

        results["bootstrap"] = {
            "obs_mean": obs_mean,
            "obs_hit": obs_hit,
            "boot_mean": boot_means.mean(),
            "boot_std": boot_means.std(),
            "p_value": p_val,
            "z_score": z_score,
            "n_events": n_cond,
        }

    # ── Market-cap tier analysis ──────────────────────────────────────
    vol_rank = (
        panel.groupby("symbol")
        .apply(lambda g: (g["close"] * g["volume"]).mean(), include_groups=False)
        .sort_values(ascending=False)
    )
    tiers = {
        "large": vol_rank.head(20).index.tolist(),
        "mid": vol_rank.iloc[20:60].index.tolist(),
        "small": vol_rank.iloc[60:].index.tolist(),
    }

    tier_results = {}
    for tier_name, tier_syms in tiers.items():
        tier_data = per_asset[per_asset["symbol"].isin(tier_syms)]
        if len(tier_data) < 3:
            continue
        tier_results[tier_name] = {
            "n_assets": len(tier_data),
            "total_events": int(tier_data["n"].sum()),
            "avg_edge": tier_data["edge"].mean(),
            "avg_hit": tier_data["hit"].mean(),
            "pct_positive": (tier_data["edge"] > 0).mean(),
        }
    results["tier_analysis"] = tier_results

    return results
