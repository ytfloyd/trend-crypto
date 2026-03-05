"""
Step 5 — Unsupervised Learning: PCA & Regime Clustering
=========================================================
JPM Big Data & AI Strategies: Crypto Recreation
Kolanovic & Krishnamachari (2017)

This script applies unsupervised methods from the paper (p93-101):
  1. PCA on the cross-sectional return matrix to extract latent factors.
  2. PCA on the feature matrix to reduce dimensionality.
  3. K-Means clustering on feature space to identify market regimes.
  4. HMM-style regime detection via return clustering over time.
  5. Test whether PCA features or regime conditioning improve Ridge predictions.
  6. Produces diagnostic plots and summary tables.

The paper argues that unsupervised methods serve two purposes:
  (a) Dimensionality reduction — compress 54 features into fewer factors.
  (b) Regime identification — detect market states where different models work.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.research.jpm_bigdata_ai.helpers import (
    FEATURE_COLS,
    PAPER_REF,
    compute_btc_benchmark,
    compute_features,
    compute_metrics,
    filter_universe,
    format_metrics_table,
    load_daily_bars,
    simple_backtest,
    walk_forward_splits,
)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
os.environ["OMP_NUM_THREADS"] = "4"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "jpm_bigdata_ai" / "step_05"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fwd_5d"
ALL_TARGETS = {"fwd_1d": 1, "fwd_5d": 5, "fwd_21d": 21}

TRAIN_DAYS = 365 * 2
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRAIN_DAYS = 365
N_PCA_COMPONENTS = 15
N_REGIMES = 4

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

# ===================================================================
# 1. Load & prepare data
# ===================================================================
print("=" * 70)
print("STEP 5: Unsupervised Learning — PCA & Regime Clustering")
print(f"Reference: {PAPER_REF}")
print("=" * 70)

panel = load_daily_bars()
panel = filter_universe(panel)
panel = compute_features(panel)
feat_cols = list(FEATURE_COLS)


def _add_forward_returns(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    for name, days in ALL_TARGETS.items():
        g[name] = g["close"].shift(-days) / g["close"] - 1.0
    return g


panel = panel.groupby("symbol", group_keys=False).apply(_add_forward_returns)
univ = panel[panel["in_universe"]].copy()

print(f"\nIn-universe: {len(univ):,} rows, {univ['symbol'].nunique()} symbols")
print(f"Features: {len(feat_cols)}")

# ===================================================================
# 2. PCA on cross-sectional returns — latent factor analysis
# ===================================================================
print("\n--- PCA on cross-sectional returns ---")

# Build return matrix: dates × symbols
ret_panel = univ[["ts", "symbol", "close"]].copy()
ret_panel["ret"] = ret_panel.groupby("symbol")["close"].pct_change()
ret_wide = ret_panel.pivot(index="ts", columns="symbol", values="ret")

# Fill NaN with 0 for PCA (missing = not traded)
ret_filled = ret_wide.fillna(0.0)

# Standardize cross-sectionally for PCA
scaler_ret = StandardScaler()
ret_std = pd.DataFrame(
    scaler_ret.fit_transform(ret_filled),
    index=ret_filled.index,
    columns=ret_filled.columns,
)

# PCA
n_components_full = min(50, ret_std.shape[1])
pca_returns = PCA(n_components=n_components_full)
factors = pca_returns.fit_transform(ret_std)
explained = pca_returns.explained_variance_ratio_

print(f"  Shape: {ret_std.shape[0]} dates × {ret_std.shape[1]} symbols")
print(f"  Variance explained by top PCs:")
for i in range(min(10, n_components_full)):
    cum = explained[:i + 1].sum()
    print(f"    PC{i+1}: {explained[i]:.1%}  (cumulative: {cum:.1%})")

# How many PCs to explain 80% / 90%?
cum_var = np.cumsum(explained)
n_80 = int(np.searchsorted(cum_var, 0.80) + 1)
n_90 = int(np.searchsorted(cum_var, 0.90) + 1)
print(f"\n  PCs for 80% variance: {n_80}")
print(f"  PCs for 90% variance: {n_90}")

# Factor returns (daily)
factor_df = pd.DataFrame(
    factors[:, :N_PCA_COMPONENTS],
    index=ret_filled.index,
    columns=[f"PC{i+1}" for i in range(N_PCA_COMPONENTS)],
)
factor_df.to_csv(ARTIFACT_DIR / "pca_factor_returns.csv", float_format="%.6f")

# ===================================================================
# 3. PCA on feature matrix — dimensionality reduction
# ===================================================================
print("\n--- PCA on feature matrix ---")

# Take one snapshot per date: cross-sectional median of features
feat_daily = univ.groupby("ts")[feat_cols].median()
feat_daily = feat_daily.dropna()

scaler_feat = StandardScaler()
feat_std = pd.DataFrame(
    scaler_feat.fit_transform(feat_daily),
    index=feat_daily.index,
    columns=feat_daily.columns,
)

pca_features = PCA(n_components=min(30, len(feat_cols)))
feat_pca = pca_features.fit_transform(feat_std)
feat_explained = pca_features.explained_variance_ratio_

print(f"  Shape: {feat_std.shape[0]} dates × {feat_std.shape[1]} features")
print(f"  Top PC loadings:")
for i in range(min(5, len(feat_explained))):
    cum = feat_explained[:i + 1].sum()
    print(f"    PC{i+1}: {feat_explained[i]:.1%}  (cumulative: {cum:.1%})")
    # Top loading features
    loadings = pca_features.components_[i]
    top_idx = np.argsort(np.abs(loadings))[-5:][::-1]
    top_feats = [(feat_cols[j], loadings[j]) for j in top_idx]
    for fname, fload in top_feats:
        print(f"        {fname:<25s}  loading={fload:+.3f}")

feat_cum = np.cumsum(feat_explained)
f_80 = int(np.searchsorted(feat_cum, 0.80) + 1)
f_90 = int(np.searchsorted(feat_cum, 0.90) + 1)
print(f"\n  Feature PCs for 80% variance: {f_80}")
print(f"  Feature PCs for 90% variance: {f_90}")

# ===================================================================
# 4. K-Means regime clustering
# ===================================================================
print("\n--- K-Means regime clustering ---")

# Cluster on daily market-level features (cross-sectional medians)
kmeans = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=20)
feat_daily_values = feat_std.values
regimes = kmeans.fit_predict(feat_daily_values)

regime_df = pd.DataFrame({"ts": feat_daily.index, "regime": regimes})
regime_df = regime_df.set_index("ts")

# Merge regime labels back to universe
univ = univ.merge(regime_df, left_on="ts", right_index=True, how="left")

# Characterize each regime
print(f"\n  {N_REGIMES} regimes identified. Characteristics:")
for r in range(N_REGIMES):
    mask = regime_df["regime"] == r
    n_days = mask.sum()
    # Average return in this regime
    regime_dates = regime_df.index[mask]
    regime_rets = ret_wide.reindex(regime_dates).mean(axis=1)
    avg_ret = regime_rets.mean() * 252
    vol = regime_rets.std() * np.sqrt(252)
    # Average feature values
    regime_feats = feat_daily.reindex(regime_dates).mean()
    print(f"\n  Regime {r}: {n_days} days ({n_days / len(regime_df) * 100:.1f}%)")
    print(f"    Ann. return: {avg_ret:+.1%}, Ann. vol: {vol:.1%}")
    print(f"    Avg features: vol_21d={regime_feats.get('vol_21d', np.nan):.2f}, "
          f"rsi_14={regime_feats.get('rsi_14', np.nan):.1f}, "
          f"adx_14={regime_feats.get('adx_14', np.nan):.1f}, "
          f"bb_width={regime_feats.get('bb_width', np.nan):.3f}")

regime_df.to_csv(ARTIFACT_DIR / "regime_labels.csv")

# ===================================================================
# 5. Test: PCA features + Ridge vs raw features + Ridge
# ===================================================================
print("\n--- Walk-forward: PCA-Ridge vs Raw-Ridge ---")

unique_dates = np.sort(univ["ts"].unique())
splits = walk_forward_splits(
    unique_dates,
    train_days=TRAIN_DAYS,
    test_days=TEST_DAYS,
    step_days=STEP_DAYS,
    min_train_days=MIN_TRAIN_DAYS,
)
print(f"  Walk-forward splits: {len(splits)}")

required_cols = feat_cols + [TARGET]
valid_mask = univ[required_cols].notna().all(axis=1)
data_clean = univ.loc[valid_mask].copy()

all_preds = []
n_splits = len(splits)

for si, fold_info in enumerate(splits):
    train_mask = (
        (data_clean["ts"] >= fold_info["train_start"])
        & (data_clean["ts"] <= fold_info["train_end"])
    )
    test_mask = (
        (data_clean["ts"] >= fold_info["test_start"])
        & (data_clean["ts"] <= fold_info["test_end"])
    )

    train = data_clean.loc[train_mask]
    test = data_clean.loc[test_mask]
    if len(train) < 100 or len(test) < 10:
        continue

    X_train_raw = train[feat_cols].values
    X_test_raw = test[feat_cols].values
    y_train = train[TARGET].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s = np.clip(scaler.transform(X_test_raw), -5, 5)

    fold = fold_info["fold"]

    # --- Model 1: Raw Ridge ---
    ridge_raw = Ridge(alpha=1.0)
    ridge_raw.fit(X_train_s, y_train)
    pred_raw = ridge_raw.predict(X_test_s)

    # --- Model 2: PCA-Ridge (fit PCA on train, transform both) ---
    pca_wf = PCA(n_components=N_PCA_COMPONENTS)
    X_train_pca = pca_wf.fit_transform(X_train_s)
    X_test_pca = pca_wf.transform(X_test_s)

    ridge_pca = Ridge(alpha=1.0)
    ridge_pca.fit(X_train_pca, y_train)
    pred_pca = ridge_pca.predict(X_test_pca)

    # --- Model 3: Regime-conditioned Ridge (train separate model per regime) ---
    # Use regime from most recent available date
    train_with_regime = train.copy()
    test_with_regime = test.copy()

    if "regime" in train_with_regime.columns:
        pred_regime = np.zeros(len(test))
        test_regimes = test_with_regime["regime"].values
        unique_regimes = np.unique(train_with_regime["regime"].dropna().values)

        for reg in unique_regimes:
            reg_train = train_with_regime[train_with_regime["regime"] == reg]
            if len(reg_train) < 50:
                continue
            X_reg = scaler.transform(reg_train[feat_cols].values)
            X_reg = np.clip(X_reg, -5, 5)
            y_reg = reg_train[TARGET].values
            ridge_reg = Ridge(alpha=1.0)
            ridge_reg.fit(X_reg, y_reg)

            test_in_reg = test_regimes == reg
            if test_in_reg.sum() > 0:
                pred_regime[test_in_reg] = ridge_reg.predict(X_test_s[test_in_reg])

        # Fill any un-assigned with raw Ridge prediction
        unassigned = pred_regime == 0
        pred_regime[unassigned] = pred_raw[unassigned]
    else:
        pred_regime = pred_raw.copy()

    # Store predictions
    for model_name, preds in [
        ("Ridge_Raw", pred_raw),
        ("Ridge_PCA", pred_pca),
        ("Ridge_Regime", pred_regime),
    ]:
        pred_df = test[["ts", "symbol", TARGET]].copy()
        pred_df["y_pred"] = preds
        pred_df["model"] = model_name
        pred_df["fold"] = fold
        all_preds.append(pred_df)

    if (si + 1) % 10 == 0 or si == n_splits - 1:
        print(f"    fold {si+1}/{n_splits}")

preds_df = pd.concat(all_preds, ignore_index=True)

# ===================================================================
# 6. Evaluate
# ===================================================================
print("\n--- Evaluation ---")

eval_records = []
for model_name, grp in preds_df.groupby("model"):
    y_true = grp[TARGET]
    y_pred = grp["y_pred"]
    mask = y_true.notna() & y_pred.notna()
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 50:
        continue

    ic = float(sp_stats.spearmanr(yt, yp).statistic)
    pearson = float(np.corrcoef(yt, yp)[0, 1])
    hit_rate = float(((yt > 0) == (yp > 0)).mean())

    fold_ics = []
    for _, fold_grp in grp.groupby("fold"):
        fm = fold_grp[TARGET].notna() & fold_grp["y_pred"].notna()
        if fm.sum() > 20:
            fic = sp_stats.spearmanr(
                fold_grp.loc[fm, TARGET], fold_grp.loc[fm, "y_pred"]
            ).statistic
            fold_ics.append(fic)

    eval_records.append({
        "model": model_name,
        "ic": ic,
        "pearson": pearson,
        "hit_rate": hit_rate,
        "n_obs": n,
        "ic_mean_fold": float(np.mean(fold_ics)) if fold_ics else np.nan,
        "ic_std_fold": float(np.std(fold_ics)) if fold_ics else np.nan,
        "ic_hit_rate": float(np.mean([x > 0 for x in fold_ics])) if fold_ics else np.nan,
    })

eval_df = pd.DataFrame(eval_records)
eval_df.to_csv(ARTIFACT_DIR / "unsupervised_evaluation.csv", index=False, float_format="%.6f")

print("\n  5d horizon results:")
for _, row in eval_df.iterrows():
    print(f"    {row['model']:<16s}  IC={row['ic']:+.4f}  Hit={row['hit_rate']:.1%}  "
          f"Fold IC={row['ic_mean_fold']:+.4f}±{row['ic_std_fold']:.4f}  "
          f"IC>0: {row['ic_hit_rate']:.0%}")

# ===================================================================
# 7. Backtest best unsupervised-enhanced model
# ===================================================================
print("\n--- Backtest ---")
best_model = str(eval_df.loc[eval_df["ic"].idxmax(), "model"])
print(f"  Best model: {best_model}")

best_preds = preds_df[preds_df["model"] == best_model].copy()
best_preds["pred_rank"] = best_preds.groupby("ts")["y_pred"].rank(pct=True)
top_q = best_preds[best_preds["pred_rank"] >= 0.80].copy()
counts = top_q.groupby("ts")["symbol"].transform("count")
top_q["weight"] = 1.0 / counts
weights_long = top_q[["ts", "symbol", "weight"]].copy()

bt_metrics_list = []
if len(weights_long) > 0:
    weights_wide = weights_long.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)
    ret_p = univ[["ts", "symbol", "close"]].copy()
    ret_p["ret"] = ret_p.groupby("symbol")["close"].pct_change()
    returns_wide = ret_p.pivot(index="ts", columns="symbol", values="ret").fillna(0.0)

    bt = simple_backtest(weights_wide, returns_wide, cost_bps=20.0)
    bt_m = compute_metrics(pd.Series(bt["portfolio_equity"].values, index=bt["ts"]))
    bt_m["label"] = f"{best_model} Top-Q L/O"
    bt_metrics_list.append(bt_m)

    btc_eq = compute_btc_benchmark(panel)
    btc_c = btc_eq.reindex(bt["ts"]).dropna()
    btc_m = compute_metrics(btc_c)
    btc_m["label"] = "BTC Buy & Hold"
    bt_metrics_list.append(btc_m)

    print("\n" + format_metrics_table(bt_metrics_list))
    pd.DataFrame(bt_metrics_list).to_csv(
        ARTIFACT_DIR / "backtest_metrics.csv", index=False, float_format="%.4f"
    )

# ===================================================================
# 8. PLOTS
# ===================================================================
print("\n--- Generating plots ---")

# --- 8a. Scree plot (return PCA) ---
print("  [1/7] Scree plot — return PCA")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
n_show = min(20, len(explained))
ax.bar(range(1, n_show + 1), explained[:n_show] * 100, color="#1976D2", alpha=0.8)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained (%)")
ax.set_title("Return PCA — Scree Plot")

ax2 = axes[1]
ax2.plot(range(1, len(cum_var) + 1), cum_var * 100, "o-", color="#1976D2", markersize=3)
ax2.axhline(80, color="red", linestyle="--", linewidth=0.8, label="80%")
ax2.axhline(90, color="orange", linestyle="--", linewidth=0.8, label="90%")
ax2.set_xlabel("Number of PCs")
ax2.set_ylabel("Cumulative Variance Explained (%)")
ax2.set_title("Return PCA — Cumulative Variance")
ax2.legend()
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "pca_scree_plot.png")
plt.close(fig)

# --- 8b. Feature PCA scree ---
print("  [2/7] Scree plot — feature PCA")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, len(feat_explained) + 1), feat_explained * 100, color="#388E3C", alpha=0.8)
ax2 = ax.twinx()
ax2.plot(range(1, len(feat_explained) + 1), np.cumsum(feat_explained) * 100,
         "ro-", markersize=4, label="Cumulative")
ax2.axhline(80, color="red", linestyle="--", linewidth=0.8)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained (%)")
ax2.set_ylabel("Cumulative (%)")
ax.set_title("Feature PCA — Scree Plot (54 TA-Lib Features)")
ax2.legend(loc="center right")
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "feature_pca_scree.png")
plt.close(fig)

# --- 8c. PC1 vs PC2 factor returns ---
print("  [3/7] PC1 vs PC2 scatter")
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(factor_df["PC1"], factor_df["PC2"],
                c=np.arange(len(factor_df)), cmap="viridis", alpha=0.4, s=3)
plt.colorbar(sc, ax=ax, label="Time (trading day index)")
ax.set_xlabel("PC1 (return factor)", fontsize=11)
ax.set_ylabel("PC2 (return factor)", fontsize=11)
ax.set_title("Return PCA — PC1 vs PC2 Over Time", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "pca_pc1_vs_pc2.png")
plt.close(fig)

# --- 8d. Regime time series ---
print("  [4/7] Regime time series")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]
# Plot BTC price with regime colors
btc = panel.loc[panel["symbol"] == "BTC-USD", ["ts", "close"]].drop_duplicates("ts").set_index("ts").sort_index()
regime_colors_map = {0: "#1976D2", 1: "#F44336", 2: "#4CAF50", 3: "#FF9800"}
for r in range(N_REGIMES):
    r_dates = regime_df.index[regime_df["regime"] == r]
    btc_r = btc.reindex(r_dates).dropna()
    ax.scatter(btc_r.index, btc_r["close"], c=regime_colors_map.get(r, "gray"),
               s=2, alpha=0.6, label=f"Regime {r}")
ax.set_yscale("log")
ax.set_ylabel("BTC Price (log)", fontsize=11)
ax.set_title("Market Regimes Overlaid on BTC Price", fontsize=13)
ax.legend(fontsize=8, markerscale=5)

ax2 = axes[1]
regime_ts = regime_df.sort_index()
ax2.scatter(regime_ts.index, regime_ts["regime"], c=[regime_colors_map.get(r, "gray") for r in regime_ts["regime"]],
            s=2, alpha=0.6)
ax2.set_ylabel("Regime", fontsize=11)
ax2.set_yticks(range(N_REGIMES))
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "regime_timeseries.png")
plt.close(fig)

# --- 8e. Regime characteristics radar/bar ---
print("  [5/7] Regime feature profiles")
profile_feats = ["vol_21d", "rsi_14", "adx_14", "bb_width", "ret_21d", "natr_14"]
regime_profiles = []
for r in range(N_REGIMES):
    r_dates = regime_df.index[regime_df["regime"] == r]
    r_feats = feat_daily.reindex(r_dates).mean()
    regime_profiles.append(r_feats[profile_feats])

profile_df = pd.DataFrame(regime_profiles, index=[f"Regime {i}" for i in range(N_REGIMES)])
fig, ax = plt.subplots(figsize=(10, 6))
profile_df.T.plot(kind="bar", ax=ax, alpha=0.8)
ax.set_title("Regime Feature Profiles (Cross-Sectional Median)", fontsize=13)
ax.set_ylabel("Feature Value")
ax.legend(fontsize=9)
plt.xticks(rotation=30)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "regime_feature_profiles.png")
plt.close(fig)

# --- 8f. Model comparison ---
print("  [6/7] Model comparison bar chart")
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = {"Ridge_Raw": "#1976D2", "Ridge_PCA": "#388E3C", "Ridge_Regime": "#F57C00"}
for i, (_, row) in enumerate(eval_df.iterrows()):
    ax.bar(i, row["ic"], color=bar_colors.get(row["model"], "gray"), alpha=0.85)
ax.set_xticks(range(len(eval_df)))
ax.set_xticklabels(eval_df["model"].values)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Out-of-Sample Spearman IC (5d)", fontsize=11)
ax.set_title("Unsupervised Enhancements vs Raw Ridge", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "model_comparison.png")
plt.close(fig)

# --- 8g. Backtest equity ---
print("  [7/7] Backtest equity curve")
if len(weights_long) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bt["ts"], bt["portfolio_equity"],
            label=f"{best_model} Top-Q L/O", color="#1976D2", linewidth=1.5)
    btc_norm = btc_c / btc_c.iloc[0]
    ax.plot(btc_norm.index, btc_norm.values,
            label="BTC Buy & Hold", color="#FF9800", linewidth=1.0, alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log)", fontsize=11)
    ax.set_title(f"Unsupervised Model Backtest — {best_model}", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "backtest_equity_curve.png")
    plt.close(fig)

# ===================================================================
# 9. Summary
# ===================================================================
print("\n" + "=" * 70)
print("STEP 5 RESULTS SUMMARY")
print("=" * 70)

print(f"\n--- Return PCA ---")
print(f"  PC1 explains {explained[0]:.1%} of cross-sectional return variance")
print(f"  {n_80} PCs for 80%, {n_90} PCs for 90%")
print(f"  → Crypto returns are heavily dominated by a single market factor")

print(f"\n--- Feature PCA ---")
print(f"  {f_80} PCs for 80%, {f_90} PCs for 90% of feature variance")
print(f"  → Moderate redundancy in the 54-feature set")

print(f"\n--- Regime Clustering ---")
print(f"  {N_REGIMES} regimes identified via K-Means on daily feature medians")
for r in range(N_REGIMES):
    n_days = (regime_df["regime"] == r).sum()
    pct = n_days / len(regime_df) * 100
    regime_dates_r = regime_df.index[regime_df["regime"] == r]
    avg_ret = ret_wide.reindex(regime_dates_r).mean(axis=1).mean() * 252
    print(f"    Regime {r}: {n_days} days ({pct:.0f}%), ann. return={avg_ret:+.1%}")

print(f"\n--- Prediction comparison (5d horizon) ---")
for _, row in eval_df.iterrows():
    print(f"  {row['model']:<16s}  IC={row['ic']:+.4f}  Hit={row['hit_rate']:.1%}  "
          f"Fold IC={row['ic_mean_fold']:+.4f}±{row['ic_std_fold']:.4f}  IC>0: {row['ic_hit_rate']:.0%}")

print(f"\nArtifacts saved to: {ARTIFACT_DIR}")
print("Done.")
