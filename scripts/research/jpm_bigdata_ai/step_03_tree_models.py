"""
Step 3 — Tree-Based Models
============================
JPM Big Data & AI Strategies: Crypto Recreation
Kolanovic & Krishnamachari (2017)

This script:
  1. Loads data, computes 54 TA-Lib features (raw only — trees don't need ranks).
  2. Runs walk-forward validation with 3 tree-based models:
       Random Forest, XGBoost, LightGBM
  3. Compares OOS IC, hit rate, and stability against Ridge baseline (Step 2).
  4. Examines feature importance (built-in, averaged across folds).
  5. Converts the best model's predictions into a simple long-only backtest.
  6. Produces diagnostic plots and summary tables.

The paper (Part III, Chapter 7) argues that tree-based models should outperform
linear models by capturing non-linear interactions — e.g., "high RSI combined
with declining volume" may be more predictive than either feature alone.

Note: We use raw features only (54, not 108) because tree-based models:
  - Handle non-linear relationships natively
  - Don't benefit from rank transforms (invariant to monotonic transforms)
  - Train faster with fewer features
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

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

# Fix OpenMP threading conflict on macOS (XGBoost + LightGBM)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "jpm_bigdata_ai" / "step_03"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fwd_5d"
ALL_TARGETS = {"fwd_1d": 1, "fwd_5d": 5, "fwd_21d": 21}

TRAIN_DAYS = 365 * 2
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRAIN_DAYS = 365

# Model hyperparameters — conservative to avoid overfitting
MODELS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=50,
        max_features=0.5,
        n_jobs=4,
        random_state=42,
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=50,
        n_jobs=4,
        random_state=42,
        verbosity=0,
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=50,
        n_jobs=4,
        random_state=42,
        verbose=-1,
    ),
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

# ===================================================================
# 1. Load & prepare data
# ===================================================================
print("=" * 70)
print("STEP 3: Tree-Based Models")
print(f"Reference: {PAPER_REF}")
print("=" * 70)

panel = load_daily_bars()
panel = filter_universe(panel)
panel = compute_features(panel)
feat_cols = list(FEATURE_COLS)  # 54 raw features only

print(f"\nFeatures: {len(feat_cols)} (raw only — trees don't need rank transforms)")


def _add_forward_returns(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    for name, days in ALL_TARGETS.items():
        g[name] = g["close"].shift(-days) / g["close"] - 1.0
    return g


panel = panel.groupby("symbol", group_keys=False).apply(_add_forward_returns)
univ = panel[panel["in_universe"]].copy()

print(f"In-universe: {len(univ):,} rows, {univ['symbol'].nunique()} symbols")
print(f"Date range: {univ['ts'].min().date()} — {univ['ts'].max().date()}")

# ===================================================================
# 2. Walk-forward splits
# ===================================================================
unique_dates = np.sort(univ["ts"].unique())
splits = walk_forward_splits(
    unique_dates,
    train_days=TRAIN_DAYS,
    test_days=TEST_DAYS,
    step_days=STEP_DAYS,
    min_train_days=MIN_TRAIN_DAYS,
)
print(f"\nWalk-forward splits: {len(splits)}")
if splits:
    print(f"  First: train {splits[0]['train_start'].date()}–{splits[0]['train_end'].date()}, "
          f"test {splits[0]['test_start'].date()}–{splits[0]['test_end'].date()}")
    print(f"  Last:  train {splits[-1]['train_start'].date()}–{splits[-1]['train_end'].date()}, "
          f"test {splits[-1]['test_start'].date()}–{splits[-1]['test_end'].date()}")


# ===================================================================
# 3. Walk-forward evaluation
# ===================================================================
print("\n--- Running walk-forward evaluation ---")


def run_walk_forward(
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    models: dict,
    splits: list[dict],
    label: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward for tree models. Returns (predictions_df, importance_history)."""
    all_preds = []
    importance_history: dict[str, list] = {name: [] for name in models}

    required_cols = features + [target_col]
    valid_mask = data[required_cols].notna().all(axis=1)
    data_clean = data.loc[valid_mask]

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

        X_train = train[features].values
        y_train = train[target_col].values
        X_test = test[features].values

        fold = fold_info["fold"]
        for model_name, model_template in models.items():
            model = clone(model_template)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            pred_df = test[["ts", "symbol", target_col]].copy()
            pred_df["y_pred"] = y_pred
            pred_df["model"] = model_name
            pred_df["fold"] = fold
            all_preds.append(pred_df)

            # Feature importance
            if hasattr(model, "feature_importances_"):
                importance_history[model_name].append({
                    "fold": fold,
                    "importances": model.feature_importances_.copy(),
                })

        if (si + 1) % 5 == 0 or si == n_splits - 1:
            print(f"    [{label}] fold {si+1}/{n_splits} "
                  f"(train={len(train):,}, test={len(test):,})")

    preds = pd.concat(all_preds, ignore_index=True)
    return preds, importance_history


# Primary target: 5d
preds_5d, imp_hist_5d = run_walk_forward(univ, feat_cols, TARGET, MODELS, splits, label="5d")
print(f"  5d predictions: {len(preds_5d):,} rows")

# Other horizons
preds_1d, _ = run_walk_forward(univ, feat_cols, "fwd_1d", MODELS, splits, label="1d")
print(f"  1d predictions: {len(preds_1d):,} rows")

preds_21d, _ = run_walk_forward(univ, feat_cols, "fwd_21d", MODELS, splits, label="21d")
print(f"  21d predictions: {len(preds_21d):,} rows")


# ===================================================================
# 4. Evaluate predictions
# ===================================================================
print("\n--- Out-of-sample evaluation ---")


def evaluate_model_preds(preds: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Compute aggregate metrics per model."""
    records = []
    for model_name, grp in preds.groupby("model"):
        y_true = grp[target_col]
        y_pred = grp["y_pred"]
        mask = y_true.notna() & y_pred.notna()
        yt, yp = y_true[mask], y_pred[mask]
        n = len(yt)
        if n < 50:
            continue

        ic = float(sp_stats.spearmanr(yt, yp).statistic)
        pearson = float(np.corrcoef(yt, yp)[0, 1])
        rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
        mae = float((yt - yp).abs().mean())
        hit_rate = float(((yt > 0) == (yp > 0)).mean())

        fold_ics = []
        for _, fold_grp in grp.groupby("fold"):
            fm = fold_grp[target_col].notna() & fold_grp["y_pred"].notna()
            if fm.sum() > 20:
                fic = sp_stats.spearmanr(
                    fold_grp.loc[fm, target_col], fold_grp.loc[fm, "y_pred"]
                ).statistic
                fold_ics.append(fic)

        records.append({
            "model": model_name,
            "ic": ic,
            "pearson": pearson,
            "rmse": rmse,
            "mae": mae,
            "hit_rate": hit_rate,
            "n_obs": n,
            "n_folds": len(fold_ics),
            "ic_mean_fold": float(np.mean(fold_ics)) if fold_ics else np.nan,
            "ic_std_fold": float(np.std(fold_ics)) if fold_ics else np.nan,
            "ic_hit_rate": float(np.mean([x > 0 for x in fold_ics])) if fold_ics else np.nan,
        })
    return pd.DataFrame(records)


eval_results = {}
for label, preds_df, target_col in [
    ("1d", preds_1d, "fwd_1d"),
    ("5d", preds_5d, "fwd_5d"),
    ("21d", preds_21d, "fwd_21d"),
]:
    ev = evaluate_model_preds(preds_df, target_col)
    ev["horizon"] = label
    eval_results[label] = ev
    print(f"\n  === {label} horizon ===")
    for _, row in ev.iterrows():
        print(
            f"    {row['model']:<14s}  IC={row['ic']:+.4f}  "
            f"Pearson={row['pearson']:+.4f}  Hit={row['hit_rate']:.1%}  "
            f"RMSE={row['rmse']:.4f}  "
            f"IC_fold={row['ic_mean_fold']:+.4f}±{row['ic_std_fold']:.4f}  "
            f"IC>0: {row['ic_hit_rate']:.0%}"
        )

all_eval = pd.concat(eval_results.values(), ignore_index=True)
all_eval.to_csv(ARTIFACT_DIR / "tree_models_evaluation.csv", index=False, float_format="%.6f")

# ===================================================================
# 5. Comparison with Step 2 linear models
# ===================================================================
print("\n--- Comparison with Step 2 (Linear Models) ---")
step2_path = ARTIFACT_DIR.parent / "step_02" / "linear_models_evaluation.csv"
if step2_path.exists():
    linear_eval = pd.read_csv(step2_path)
    linear_5d = linear_eval[linear_eval["horizon"] == "5d"][["model", "ic", "hit_rate", "ic_mean_fold", "ic_std_fold"]]
    tree_5d = eval_results["5d"][["model", "ic", "hit_rate", "ic_mean_fold", "ic_std_fold"]]
    combined = pd.concat([linear_5d, tree_5d], ignore_index=True).sort_values("ic", ascending=False)
    print("\n  All models ranked by IC (5d horizon):")
    for _, row in combined.iterrows():
        print(f"    {row['model']:<14s}  IC={row['ic']:+.4f}  Hit={row['hit_rate']:.1%}  "
              f"Fold IC={row['ic_mean_fold']:+.4f}±{row['ic_std_fold']:.4f}")
    combined.to_csv(ARTIFACT_DIR / "linear_vs_tree_comparison.csv", index=False, float_format="%.6f")
else:
    print("  (Step 2 results not found — skipping comparison)")

# ===================================================================
# 6. Feature importance analysis
# ===================================================================
print("\n--- Feature importance analysis ---")

importance_dfs = {}
for model_name in MODELS:
    if imp_hist_5d.get(model_name):
        imp_matrix = np.array([h["importances"] for h in imp_hist_5d[model_name]])
        mean_imp = imp_matrix.mean(axis=0)
        std_imp = imp_matrix.std(axis=0)
        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "mean_importance": mean_imp,
            "std_importance": std_imp,
        }).sort_values("mean_importance", ascending=False)
        imp_df["rank"] = range(1, len(imp_df) + 1)
        importance_dfs[model_name] = imp_df
        imp_df.to_csv(ARTIFACT_DIR / f"feature_importance_{model_name.lower()}.csv",
                       index=False, float_format="%.6f")

        print(f"\n  {model_name} — Top 15 features:")
        for _, row in imp_df.head(15).iterrows():
            print(f"    {row['feature']:<28s}  imp={row['mean_importance']:.4f} ± {row['std_importance']:.4f}")

# Cross-model consensus: features that are top-10 in all models
if len(importance_dfs) >= 2:
    top10_sets = {}
    for mname, idf in importance_dfs.items():
        top10_sets[mname] = set(idf.head(10)["feature"].tolist())
    consensus = set.intersection(*top10_sets.values())
    print(f"\n  Consensus top-10 features (in all {len(importance_dfs)} models): {sorted(consensus)}")

# ===================================================================
# 7. Per-fold IC time series
# ===================================================================
print("\n--- Per-fold IC time series ---")

fold_ic_records = []
for model_name in MODELS:
    model_preds = preds_5d[preds_5d["model"] == model_name]
    for fold_id, fold_grp in model_preds.groupby("fold"):
        fm = fold_grp[TARGET].notna() & fold_grp["y_pred"].notna()
        if fm.sum() < 20:
            continue
        fic = sp_stats.spearmanr(
            fold_grp.loc[fm, TARGET], fold_grp.loc[fm, "y_pred"]
        ).statistic
        mid_date = fold_grp["ts"].median()
        fold_ic_records.append({
            "model": model_name,
            "fold": fold_id,
            "date": mid_date,
            "ic": fic,
            "n_obs": int(fm.sum()),
        })

fold_ic_df = pd.DataFrame(fold_ic_records)
fold_ic_df.to_csv(ARTIFACT_DIR / "fold_ic_timeseries.csv", index=False, float_format="%.6f")

# ===================================================================
# 8. Backtest from best model
# ===================================================================
print("\n--- Backtest: converting predictions to portfolio ---")

# Pick best model by aggregate IC
ev5 = eval_results["5d"]
best_model = str(ev5.loc[ev5["ic"].idxmax(), "model"])
print(f"  Best model by IC: {best_model}")

best_preds = preds_5d[preds_5d["model"] == best_model].copy()
best_preds["pred_rank"] = best_preds.groupby("ts")["y_pred"].rank(pct=True)

# Vectorized weight construction
top_q = best_preds[best_preds["pred_rank"] >= 0.80].copy()
counts = top_q.groupby("ts")["symbol"].transform("count")
top_q["weight"] = 1.0 / counts
weights_long = top_q[["ts", "symbol", "weight"]].copy()

bt_metrics_dict = {}
if len(weights_long) > 0:
    weights_wide = weights_long.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)

    ret_panel = univ[["ts", "symbol", "close"]].copy()
    ret_panel["ret"] = ret_panel.groupby("symbol")["close"].pct_change()
    returns_wide = ret_panel.pivot(index="ts", columns="symbol", values="ret").fillna(0.0)

    bt = simple_backtest(weights_wide, returns_wide, cost_bps=20.0)
    bt_metrics = compute_metrics(pd.Series(bt["portfolio_equity"].values, index=bt["ts"]))
    bt_metrics["label"] = f"{best_model} Top-Quintile L/O"
    bt_metrics_dict["tree"] = bt_metrics

    # BTC benchmark
    btc_eq = compute_btc_benchmark(panel)
    btc_common = btc_eq.reindex(bt["ts"]).dropna()
    btc_metrics = compute_metrics(btc_common)
    btc_metrics["label"] = "BTC Buy & Hold"
    bt_metrics_dict["btc"] = btc_metrics

    # EW universe
    ew_panel = univ[univ["in_universe"]][["ts", "symbol", "close"]].copy()
    ew_panel["ret"] = ew_panel.groupby("symbol")["close"].pct_change()
    ew_daily = ew_panel.groupby("ts")["ret"].mean()
    ew_equity = (1 + ew_daily).cumprod()
    ew_equity = ew_equity.reindex(bt["ts"].values).dropna()
    ew_metrics = compute_metrics(ew_equity)
    ew_metrics["label"] = "Equal-Weight Universe"
    bt_metrics_dict["ew"] = ew_metrics

    print("\n" + format_metrics_table(list(bt_metrics_dict.values())))
    pd.DataFrame(list(bt_metrics_dict.values())).to_csv(
        ARTIFACT_DIR / "backtest_metrics.csv", index=False, float_format="%.4f"
    )

# ===================================================================
# 9. PLOTS
# ===================================================================
print("\n--- Generating plots ---")

model_colors = {
    "RandomForest": "#1976D2",
    "XGBoost": "#388E3C",
    "LightGBM": "#F57C00",
}

# --- 9a. Model comparison bar chart (all methods including linear) ---
print("  [1/7] Model comparison (IC — trees vs linear)")
fig, ax = plt.subplots(figsize=(12, 6))
horizons = ["1d", "5d", "21d"]
if step2_path.exists():
    all_models_5d = combined.sort_values("ic", ascending=True)
else:
    all_models_5d = eval_results["5d"].sort_values("ic", ascending=True)
colors = []
for m in all_models_5d["model"]:
    if m in model_colors:
        colors.append(model_colors[m])
    elif m in ("Ridge", "OLS"):
        colors.append("#9E9E9E")
    elif m == "LASSO":
        colors.append("#BDBDBD")
    else:
        colors.append("#E0E0E0")
ax.barh(all_models_5d["model"], all_models_5d["ic"], color=colors, alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Out-of-Sample Spearman IC (5-Day Horizon)", fontsize=11)
ax.set_title("Tree vs Linear Models — OOS Prediction IC", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "tree_vs_linear_ic.png")
plt.close(fig)

# --- 9b. IC by horizon for tree models ---
print("  [2/7] Tree model IC by horizon")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(MODELS))
width = 0.22
for i, hz in enumerate(horizons):
    ev = eval_results[hz].set_index("model")
    vals = [ev.loc[m, "ic"] if m in ev.index else 0 for m in MODELS]
    ax.bar(x + i * width, vals, width, label=f"{hz} horizon", alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(MODELS.keys())
ax.set_ylabel("Out-of-Sample Spearman IC", fontsize=11)
ax.set_title("Tree-Based Models — OOS IC by Horizon", fontsize=13)
ax.legend()
ax.axhline(0, color="black", linewidth=0.5)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "tree_ic_by_horizon.png")
plt.close(fig)

# --- 9c. Fold IC time series ---
print("  [3/7] Fold IC time series")
fig, ax = plt.subplots(figsize=(14, 6))
for model_name in MODELS:
    sub = fold_ic_df[fold_ic_df["model"] == model_name].sort_values("date")
    ax.plot(sub["date"], sub["ic"], marker="o", markersize=3,
            label=model_name, color=model_colors.get(model_name, "gray"), alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Spearman IC (per fold)", fontsize=11)
ax.set_title("Walk-Forward IC Over Time — Tree Models (5d Horizon)", fontsize=13)
ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "fold_ic_timeseries.png")
plt.close(fig)

# --- 9d. Feature importance comparison ---
print("  [4/7] Feature importance comparison")
if len(importance_dfs) >= 2:
    fig, axes = plt.subplots(1, len(importance_dfs), figsize=(6 * len(importance_dfs), 12))
    if len(importance_dfs) == 1:
        axes = [axes]
    for ax, (mname, idf) in zip(axes, importance_dfs.items()):
        top20 = idf.head(20)
        ax.barh(range(len(top20)), top20["mean_importance"].values,
                xerr=top20["std_importance"].values,
                color=model_colors.get(mname, "#1976D2"), alpha=0.8, capsize=2)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20["feature"].values, fontsize=7)
        ax.set_title(f"{mname} — Top 20", fontsize=11)
        ax.set_xlabel("Mean Feature Importance", fontsize=9)
        ax.invert_yaxis()
    fig.suptitle("Feature Importance Comparison — Tree Models (5d Horizon)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "feature_importance_comparison.png")
    plt.close(fig)

# --- 9e. Feature importance rank correlation ---
print("  [5/7] Feature importance rank correlation")
if len(importance_dfs) >= 2:
    model_names_list = list(importance_dfs.keys())
    rank_corr = pd.DataFrame(index=model_names_list, columns=model_names_list, dtype=float)
    for m1 in model_names_list:
        for m2 in model_names_list:
            imp1 = importance_dfs[m1].set_index("feature")["mean_importance"]
            imp2 = importance_dfs[m2].set_index("feature")["mean_importance"]
            common = imp1.index.intersection(imp2.index)
            rank_corr.loc[m1, m2] = sp_stats.spearmanr(imp1[common], imp2[common]).statistic
    fig, ax = plt.subplots(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(rank_corr.astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_title("Feature Importance Rank Correlation", fontsize=12)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "importance_rank_correlation.png")
    plt.close(fig)

# --- 9f. Backtest equity curve ---
print("  [6/7] Backtest equity curve")
if len(weights_long) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(bt["ts"], bt["portfolio_equity"], label=f"{best_model} Top-Quintile L/O",
            color=model_colors.get(best_model, "#1976D2"), linewidth=1.5)
    btc_aligned = btc_eq.reindex(bt["ts"]).dropna()
    btc_aligned_norm = btc_aligned / btc_aligned.iloc[0]
    ax.plot(btc_aligned_norm.index, btc_aligned_norm.values,
            label="BTC Buy & Hold", color="#FF9800", linewidth=1.0, alpha=0.7)
    if len(ew_equity) > 0:
        ew_norm = ew_equity / ew_equity.iloc[0]
        ax.plot(ew_norm.index, ew_norm.values,
                label="EW Universe", color="#9E9E9E", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("Equity (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"Tree Model Backtest — {best_model} Top-Quintile Long-Only", fontsize=13)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    eq_series = pd.Series(bt["portfolio_equity"].values, index=bt["ts"])
    dd = eq_series / eq_series.cummax() - 1.0
    ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.4)
    ax2.set_ylabel("Drawdown", fontsize=10)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "backtest_equity_curve.png")
    plt.close(fig)

# --- 9g. Prediction scatter ---
print("  [7/7] Prediction scatter plot")
best_pred_data = preds_5d[preds_5d["model"] == best_model].copy()
sample = best_pred_data.dropna(subset=[TARGET, "y_pred"]).sample(
    min(5000, len(best_pred_data)), random_state=42
)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(sample[TARGET], sample["y_pred"], alpha=0.08, s=5,
           color=model_colors.get(best_model, "#1976D2"))
lims = [-0.5, 0.5]
ax.plot(lims, lims, "r--", linewidth=0.8, alpha=0.5, label="Perfect prediction")
ax.set_xlim(lims)
ax.set_ylim([sample["y_pred"].quantile(0.01), sample["y_pred"].quantile(0.99)])
ax.set_xlabel("Actual 5-Day Return", fontsize=11)
ax.set_ylabel("Predicted 5-Day Return", fontsize=11)
ic_val = eval_results["5d"].set_index("model").loc[best_model, "ic"]
ax.set_title(f"{best_model} Predictions vs Actuals (IC={ic_val:+.4f})", fontsize=13)
ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "prediction_scatter.png")
plt.close(fig)

# ===================================================================
# 10. Summary
# ===================================================================
print("\n" + "=" * 70)
print("STEP 3 RESULTS SUMMARY")
print("=" * 70)

print(f"\nWalk-forward: {len(splits)} folds, "
      f"{TRAIN_DAYS//365}yr train / {TEST_DAYS}d test / {STEP_DAYS}d step")
print(f"Features: {len(feat_cols)} (raw TA-Lib features)")

print("\n--- OOS IC summary (5d horizon) ---")
ev5 = eval_results["5d"]
for _, row in ev5.iterrows():
    print(f"  {row['model']:<14s}  IC={row['ic']:+.4f}  Hit={row['hit_rate']:.1%}  "
          f"Fold IC: {row['ic_mean_fold']:+.4f} ± {row['ic_std_fold']:.4f} "
          f"(positive in {row['ic_hit_rate']:.0%} of folds)")

print("\n--- IC across horizons ---")
for hz in horizons:
    ev = eval_results[hz]
    best = ev.loc[ev["ic"].idxmax()]
    print(f"  {hz:<5s}  Best: {best['model']:<14s} IC={best['ic']:+.4f}")

if step2_path.exists():
    print("\n--- Trees vs Linear (5d horizon) ---")
    tree_best_ic = ev5["ic"].max()
    linear_best_ic = linear_5d["ic"].max()
    improvement = tree_best_ic - linear_best_ic
    print(f"  Best tree IC:   {tree_best_ic:+.4f} ({ev5.loc[ev5['ic'].idxmax(), 'model']})")
    print(f"  Best linear IC: {linear_best_ic:+.4f}")
    print(f"  Improvement:    {improvement:+.4f} ({improvement/abs(linear_best_ic)*100:+.1f}%)")

if bt_metrics_dict:
    print(f"\n--- {best_model} top-quintile backtest ---")
    print(format_metrics_table(list(bt_metrics_dict.values())))
    avg_holdings = weights_wide.gt(0).sum(axis=1).mean()
    print(f"\n  Avg holdings: {avg_holdings:.1f} symbols")
    print(f"  Avg turnover: {bt['turnover'].mean():.1%} per day")

if consensus:
    print(f"\n  Cross-model consensus top features: {sorted(consensus)}")

print(f"\nArtifacts saved to: {ARTIFACT_DIR}")
print("Done.")
