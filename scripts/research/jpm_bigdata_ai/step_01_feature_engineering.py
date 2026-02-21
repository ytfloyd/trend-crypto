"""
Step 1 — Feature Engineering & Exploratory Analysis
=====================================================
JPM Big Data & AI Strategies: Crypto Recreation
Kolanovic & Krishnamachari (2017)

This script:
  1. Loads daily OHLCV data and applies the dynamic universe filter.
  2. Computes 54 technical features via TA-Lib.
  3. Defines prediction targets (forward returns at 1d, 5d, 21d horizons).
  4. Measures univariate predictive power (Spearman IC) per feature.
  5. Explores feature distributions, correlations, and stability.
  6. Produces diagnostic plots and summary tables.

All features are lagged by 1 day to prevent lookahead bias.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from scripts.research.jpm_bigdata_ai.helpers import (
    FEATURE_COLS,
    PAPER_REF,
    add_cross_sectional_ranks,
    compute_features,
    filter_universe,
    load_daily_bars,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "jpm_bigdata_ai" / "step_01"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

FORWARD_HORIZONS = {"fwd_1d": 1, "fwd_5d": 5, "fwd_21d": 21}
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ===================================================================
# 1. Load & prepare data
# ===================================================================
print("=" * 70)
print("STEP 1: Feature Engineering & Exploratory Analysis")
print(f"Reference: {PAPER_REF}")
print("=" * 70)

panel = load_daily_bars()
panel = filter_universe(panel)
panel = compute_features(panel)
feat_cols = list(FEATURE_COLS)  # snapshot after compute_features populates it

print(f"\nPanel shape: {panel.shape}")
print(f"Symbols: {panel['symbol'].nunique()}")
print(f"Date range: {panel['ts'].min().date()} to {panel['ts'].max().date()}")
print(f"Feature columns: {len(feat_cols)}")

# ===================================================================
# 2. Define forward-return targets
# ===================================================================
print("\n--- Computing forward returns ---")


def _add_forward_returns(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    for name, days in FORWARD_HORIZONS.items():
        g[name] = g["close"].shift(-days) / g["close"] - 1.0
    return g


panel = panel.groupby("symbol", group_keys=False).apply(_add_forward_returns)
target_cols = list(FORWARD_HORIZONS.keys())

univ = panel[panel["in_universe"]].copy()
print(f"In-universe rows: {len(univ):,}")
for tc in target_cols:
    valid = univ[tc].notna().sum()
    print(f"  {tc}: {valid:,} valid obs, mean={univ[tc].mean():.4f}, std={univ[tc].std():.4f}")

# ===================================================================
# 3. Add cross-sectional ranks
# ===================================================================
print("\n--- Adding cross-sectional ranks ---")
univ = add_cross_sectional_ranks(univ, feat_cols)
rank_cols = [f"{c}_xsrank" for c in feat_cols]
all_feat_cols = feat_cols + rank_cols
print(f"Total features (raw + xsrank): {len(all_feat_cols)}")

# ===================================================================
# 4. Feature summary statistics
# ===================================================================
print("\n--- Feature Summary Statistics ---")
feat_stats = univ[feat_cols].describe().T
feat_stats["nan_pct"] = univ[feat_cols].isna().mean().values * 100
feat_stats = feat_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "nan_pct"]]
feat_stats.to_csv(ARTIFACT_DIR / "feature_summary_stats.csv", float_format="%.4f")
print(feat_stats.round(3).to_string())

# ===================================================================
# 5. Univariate Information Coefficients (Spearman rank corr)
# ===================================================================
print("\n--- Univariate IC (Spearman rank correlation with forward returns) ---")

ic_records = []
for target in target_cols:
    mask = univ[target].notna()
    sub = univ.loc[mask]
    for feat in feat_cols:
        fmask = sub[feat].notna()
        if fmask.sum() < 100:
            continue
        ic_val, pval = sp_stats.spearmanr(sub.loc[fmask, feat], sub.loc[fmask, target])
        ic_records.append({
            "feature": feat,
            "target": target,
            "ic": ic_val,
            "pval": pval,
            "n_obs": int(fmask.sum()),
        })

ic_df = pd.DataFrame(ic_records)
ic_df.to_csv(ARTIFACT_DIR / "univariate_ic.csv", index=False, float_format="%.6f")

# Print top features per horizon
for target in target_cols:
    sub = ic_df[ic_df["target"] == target].sort_values("ic", ascending=False, key=abs)
    print(f"\n  Top 10 features for {target}:")
    for _, row in sub.head(10).iterrows():
        sig = "***" if row["pval"] < 0.001 else "**" if row["pval"] < 0.01 else "*" if row["pval"] < 0.05 else ""
        print(f"    {row['feature']:<28s} IC={row['ic']:+.4f} {sig}")

# ===================================================================
# 6. Time-varying IC (rolling Spearman, monthly)
# ===================================================================
print("\n--- Computing rolling IC over time ---")

univ["year_month"] = univ["ts"].dt.to_period("M")
top_feats = (
    ic_df[ic_df["target"] == "fwd_5d"]
    .sort_values("ic", ascending=False, key=abs)
    .head(8)["feature"]
    .tolist()
)

rolling_ic = []
for ym, grp in univ.groupby("year_month"):
    if len(grp) < 30:
        continue
    for feat in top_feats:
        mask = grp["fwd_5d"].notna() & grp[feat].notna()
        if mask.sum() < 20:
            continue
        ic_val, _ = sp_stats.spearmanr(grp.loc[mask, feat], grp.loc[mask, "fwd_5d"])
        rolling_ic.append({"date": ym.to_timestamp(), "feature": feat, "ic": ic_val})

rolling_ic_df = pd.DataFrame(rolling_ic)
rolling_ic_df.to_csv(ARTIFACT_DIR / "rolling_ic_monthly.csv", index=False, float_format="%.6f")

# ===================================================================
# 7. PLOTS
# ===================================================================
print("\n--- Generating plots ---")

# --- 7a. Feature correlation heatmap ---
print("  [1/6] Feature correlation heatmap")
corr = univ[feat_cols].corr()
fig, ax = plt.subplots(figsize=(18, 15))
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, mask=mask_upper, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.3, ax=ax,
    cbar_kws={"shrink": 0.6, "label": "Pearson correlation"},
)
ax.set_title("Feature Correlation Matrix (54 TA-Lib Features)", fontsize=14, pad=15)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "feature_correlation_heatmap.png")
plt.close(fig)

# --- 7b. IC bar chart (5-day horizon) ---
print("  [2/6] IC bar chart (fwd_5d)")
ic_5d = ic_df[ic_df["target"] == "fwd_5d"].sort_values("ic")
fig, ax = plt.subplots(figsize=(10, 14))
colors = ["#2196F3" if v > 0 else "#F44336" for v in ic_5d["ic"]]
ax.barh(ic_5d["feature"], ic_5d["ic"], color=colors, edgecolor="white", linewidth=0.3)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Spearman IC (vs 5-day forward return)", fontsize=11)
ax.set_title("Univariate Feature IC — 5-Day Horizon", fontsize=13, pad=10)
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "ic_bar_chart_5d.png")
plt.close(fig)

# --- 7c. IC bar chart (1-day and 21-day) ---
print("  [3/6] IC bar charts (fwd_1d, fwd_21d)")
fig, axes = plt.subplots(1, 2, figsize=(18, 12))
for ax, target, label in zip(
    axes, ["fwd_1d", "fwd_21d"], ["1-Day Forward Return", "21-Day Forward Return"]
):
    sub = ic_df[ic_df["target"] == target].sort_values("ic")
    colors = ["#2196F3" if v > 0 else "#F44336" for v in sub["ic"]]
    ax.barh(sub["feature"], sub["ic"], color=colors, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"Spearman IC (vs {label})", fontsize=10)
    ax.set_title(f"Univariate IC — {label}", fontsize=12)
    ax.tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "ic_bar_chart_1d_21d.png")
plt.close(fig)

# --- 7d. Rolling IC over time (top 8 features, 5d horizon) ---
print("  [4/6] Rolling IC time series")
fig, ax = plt.subplots(figsize=(14, 6))
for feat in top_feats:
    sub = rolling_ic_df[rolling_ic_df["feature"] == feat]
    ax.plot(sub["date"], sub["ic"], label=feat, alpha=0.7, linewidth=1.0)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Monthly Spearman IC", fontsize=11)
ax.set_title("Rolling IC Over Time — Top 8 Features (5-Day Horizon)", fontsize=13, pad=10)
ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.8)
ax.set_xlabel("")
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "rolling_ic_timeseries.png")
plt.close(fig)

# --- 7e. Feature distribution grid (top 12 by absolute IC) ---
print("  [5/6] Feature distribution grid")
top12 = (
    ic_df[ic_df["target"] == "fwd_5d"]
    .sort_values("ic", ascending=False, key=abs)
    .head(12)["feature"]
    .tolist()
)
fig, axes = plt.subplots(3, 4, figsize=(18, 10))
for ax, feat in zip(axes.flat, top12):
    vals = univ[feat].dropna()
    # Clip extreme outliers for display
    lo_clip, hi_clip = vals.quantile(0.01), vals.quantile(0.99)
    clipped = vals.clip(lo_clip, hi_clip)
    ax.hist(clipped, bins=60, color="#1976D2", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_title(feat, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_ylabel("")
fig.suptitle("Feature Distributions (Top 12 by |IC|, 1%–99% clipped)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "feature_distributions_top12.png")
plt.close(fig)

# --- 7f. IC decay across horizons ---
print("  [6/6] IC decay across horizons")
decay_data = []
for target in target_cols:
    sub = ic_df[ic_df["target"] == target]
    mean_abs_ic = sub["ic"].abs().mean()
    median_abs_ic = sub["ic"].abs().median()
    decay_data.append({
        "horizon": target,
        "horizon_days": FORWARD_HORIZONS[target],
        "mean_abs_ic": mean_abs_ic,
        "median_abs_ic": median_abs_ic,
    })

decay_df = pd.DataFrame(decay_data)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    decay_df["horizon"], decay_df["mean_abs_ic"],
    color="#1976D2", alpha=0.8, label="Mean |IC|", width=0.35,
)
ax.bar(
    [x + 0.35 for x in range(len(decay_df))], decay_df["median_abs_ic"],
    color="#FF9800", alpha=0.8, label="Median |IC|", width=0.35,
    tick_label=decay_df["horizon"],
)
ax.set_ylabel("Absolute Spearman IC", fontsize=11)
ax.set_title("Predictive Power Decay Across Horizons", fontsize=13, pad=10)
ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "ic_decay_across_horizons.png")
plt.close(fig)

# ===================================================================
# 8. Summary report
# ===================================================================
print("\n" + "=" * 70)
print("STEP 1 RESULTS SUMMARY")
print("=" * 70)

print(f"\nUniverse: {univ['symbol'].nunique()} symbols, "
      f"{univ['ts'].nunique()} trading days")
print(f"Date range: {univ['ts'].min().date()} — {univ['ts'].max().date()}")
print(f"Features: {len(feat_cols)} raw + {len(rank_cols)} cross-sectional ranks = {len(all_feat_cols)} total")
print(f"NaN rate (in-universe, raw features): {univ[feat_cols].isna().mean().mean()*100:.1f}%")

print("\n--- Average |IC| by horizon ---")
for _, row in decay_df.iterrows():
    print(f"  {row['horizon']:<10s}  Mean |IC|={row['mean_abs_ic']:.4f}  "
          f"Median |IC|={row['median_abs_ic']:.4f}")

print(f"\n--- Top 5 predictive features (5d horizon) ---")
top5 = (
    ic_df[ic_df["target"] == "fwd_5d"]
    .sort_values("ic", ascending=False, key=abs)
    .head(5)
)
for _, row in top5.iterrows():
    print(f"  {row['feature']:<28s}  IC={row['ic']:+.4f}  p={row['pval']:.2e}")

print(f"\n--- Least predictive features (5d horizon) ---")
bottom5 = (
    ic_df[ic_df["target"] == "fwd_5d"]
    .sort_values("ic", ascending=False, key=abs)
    .tail(5)
)
for _, row in bottom5.iterrows():
    print(f"  {row['feature']:<28s}  IC={row['ic']:+.4f}  p={row['pval']:.2e}")

# Feature group mean IC
print("\n--- Mean |IC| by feature group (5d) ---")
groups = {
    "Returns": [c for c in feat_cols if c.startswith("ret_")],
    "Volatility (realized)": [c for c in feat_cols if c.startswith("vol_") and "ratio" not in c],
    "Volume ratios": [c for c in feat_cols if "vol_ratio" in c],
    "Trend": ["adx_14", "adx_28", "macd", "macd_signal", "macd_hist",
              "aroon_osc", "linearreg_slope_14", "linearreg_slope_42"],
    "Momentum": ["rsi_14", "rsi_28", "stoch_k", "stoch_d", "cci_14", "cci_28",
                 "willr_14", "mfi_14", "roc_10", "roc_21", "ultosc"],
    "Volatility (indicator)": ["atr_14", "natr_14", "bb_width", "bb_pctb", "hl_range"],
    "Volume (indicator)": ["obv_slope_14", "ad_slope_14"],
    "Price structure": ["channel_pos_14", "channel_pos_42"],
    "Open-price": ["overnight_gap", "body_ratio", "upper_shadow", "lower_shadow"],
    "Candlestick": [c for c in feat_cols if c.startswith("cdl_")],
}

ic_5d_lookup = ic_df[ic_df["target"] == "fwd_5d"].set_index("feature")["ic"]
for grp_name, cols in groups.items():
    existing = [c for c in cols if c in ic_5d_lookup.index]
    if existing:
        mean_ic = ic_5d_lookup[existing].abs().mean()
        print(f"  {grp_name:<25s}  Mean |IC|={mean_ic:.4f}  (n={len(existing)} features)")

print(f"\nArtifacts saved to: {ARTIFACT_DIR}")
print("Done.")
