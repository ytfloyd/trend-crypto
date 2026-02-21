"""
Step 4 — SVM and Classification Framing
=========================================
JPM Big Data & AI Strategies: Crypto Recreation
Kolanovic & Krishnamachari (2017)

This script reframes the prediction task from regression to classification:
  - Target: will the 5-day forward return be positive or negative?
  - Models: Logistic Regression (L2), Linear SVM, RBF SVM, XGBoost Classifier
  - Evaluate: Accuracy, Precision, Recall, F1, AUC, Spearman IC (on probabilities)
  - Compare classification hit rate vs regression hit rate from Steps 2-3

The paper (Part III, Chapter 8) argues that predicting direction is more
natural for trading decisions and that SVMs' maximum-margin principle is
well-suited for noisy financial data.

Note: RBF SVM is O(n^2)–O(n^3), so we subsample training to 8K rows max
for the RBF kernel while using the full dataset for linear models.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

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
ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "jpm_bigdata_ai" / "step_04"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REG = "fwd_5d"
ALL_TARGETS = {"fwd_1d": 1, "fwd_5d": 5, "fwd_21d": 21}
RBF_MAX_TRAIN = 8000  # subsample limit for RBF SVM

TRAIN_DAYS = 365 * 2
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRAIN_DAYS = 365

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

# ===================================================================
# 1. Load & prepare data
# ===================================================================
print("=" * 70)
print("STEP 4: SVM and Classification Framing")
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

# Binary classification targets
for tc in ALL_TARGETS:
    univ[f"{tc}_cls"] = (univ[tc] > 0).astype(int)

print(f"\nIn-universe: {len(univ):,} rows, {univ['symbol'].nunique()} symbols")
print(f"Features: {len(feat_cols)}")
print(f"Class balance (5d): {univ['fwd_5d_cls'].mean():.1%} positive")

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

# ===================================================================
# 3. Define models
# ===================================================================
MODELS = {
    "LogisticReg": LogisticRegression(
        C=1.0, penalty="l2", max_iter=1000, solver="lbfgs", random_state=42,
    ),
    "LinearSVM": CalibratedClassifierCV(
        LinearSVC(C=0.1, max_iter=2000, random_state=42, dual=True),
        cv=3, method="sigmoid",
    ),
    "RBF_SVM": CalibratedClassifierCV(
        SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42),
        cv=3, method="sigmoid",
    ),
    "XGB_Clf": None,  # Created per-fold to avoid serialization issues
}


def _make_xgb_clf():
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=50,
        n_jobs=4,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )


# ===================================================================
# 4. Walk-forward evaluation
# ===================================================================
print("\n--- Running walk-forward evaluation ---")


def run_classification_wf(
    data: pd.DataFrame,
    features: list[str],
    target_cls: str,
    target_reg: str,
    models: dict,
    splits: list[dict],
    label: str = "",
) -> pd.DataFrame:
    """Walk-forward classification. Returns predictions with probabilities."""
    required_cols = features + [target_cls, target_reg]
    valid_mask = data[required_cols].notna().all(axis=1)
    data_clean = data.loc[valid_mask]

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

        X_train_raw = train[features].values
        y_train = train[target_cls].values
        X_test_raw = test[features].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_raw)
        X_test_s = scaler.transform(X_test_raw)
        X_test_s = np.clip(X_test_s, -5, 5)

        fold = fold_info["fold"]

        for model_name, model_template in models.items():
            if model_name == "XGB_Clf":
                model = _make_xgb_clf()
                # XGBoost doesn't need standardization but we use it anyway for consistency
            elif model_name == "RBF_SVM" and len(X_train_s) > RBF_MAX_TRAIN:
                # Subsample for RBF SVM
                rng = np.random.RandomState(42 + fold)
                idx = rng.choice(len(X_train_s), RBF_MAX_TRAIN, replace=False)
                model = clone(model_template)
                model.fit(X_train_s[idx], y_train[idx])
            else:
                model = clone(model_template)

            if model_name == "RBF_SVM" and len(X_train_s) > RBF_MAX_TRAIN:
                pass  # already fit above
            else:
                model.fit(X_train_s, y_train)

            # Get predicted probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_s)[:, 1]
            else:
                y_prob = model.decision_function(X_test_s)
            y_pred_cls = (y_prob >= 0.5).astype(int)

            pred_df = test[["ts", "symbol", target_cls, target_reg]].copy()
            pred_df["y_prob"] = y_prob
            pred_df["y_pred_cls"] = y_pred_cls
            pred_df["model"] = model_name
            pred_df["fold"] = fold
            all_preds.append(pred_df)

        if (si + 1) % 5 == 0 or si == n_splits - 1:
            print(f"    [{label}] fold {si+1}/{n_splits} "
                  f"(train={len(train):,}, test={len(test):,})")

    return pd.concat(all_preds, ignore_index=True)


# Run for all horizons
preds_all = {}
for target_name, horizon in ALL_TARGETS.items():
    lbl = f"{horizon}d"
    preds = run_classification_wf(
        univ, feat_cols, f"{target_name}_cls", target_name,
        MODELS, splits, label=lbl,
    )
    preds_all[lbl] = preds
    print(f"  {lbl} predictions: {len(preds):,} rows")

# ===================================================================
# 5. Evaluate predictions
# ===================================================================
print("\n--- Out-of-sample evaluation ---")


def evaluate_classification(preds: pd.DataFrame, target_cls: str, target_reg: str) -> pd.DataFrame:
    """Compute classification and ranking metrics."""
    records = []
    for model_name, grp in preds.groupby("model"):
        y_true_cls = grp[target_cls].values
        y_pred_cls = grp["y_pred_cls"].values
        y_prob = grp["y_prob"].values
        y_true_reg = grp[target_reg].values

        mask = ~np.isnan(y_prob) & ~np.isnan(y_true_reg)
        n = mask.sum()
        if n < 50:
            continue

        acc = accuracy_score(y_true_cls[mask], y_pred_cls[mask])
        prec = precision_score(y_true_cls[mask], y_pred_cls[mask], zero_division=0)
        rec = recall_score(y_true_cls[mask], y_pred_cls[mask], zero_division=0)
        f1 = f1_score(y_true_cls[mask], y_pred_cls[mask], zero_division=0)

        try:
            auc = roc_auc_score(y_true_cls[mask], y_prob[mask])
        except ValueError:
            auc = np.nan

        # Spearman IC: rank correlation between predicted probability and actual return
        ic = float(sp_stats.spearmanr(y_true_reg[mask], y_prob[mask]).statistic)

        # Per-fold IC stability
        fold_ics = []
        for _, fold_grp in grp.groupby("fold"):
            fm = fold_grp["y_prob"].notna() & fold_grp[target_reg].notna()
            if fm.sum() > 20:
                fic = sp_stats.spearmanr(
                    fold_grp.loc[fm, target_reg], fold_grp.loc[fm, "y_prob"]
                ).statistic
                fold_ics.append(fic)

        records.append({
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "ic": ic,
            "n_obs": int(n),
            "ic_mean_fold": float(np.mean(fold_ics)) if fold_ics else np.nan,
            "ic_std_fold": float(np.std(fold_ics)) if fold_ics else np.nan,
            "ic_hit_rate": float(np.mean([x > 0 for x in fold_ics])) if fold_ics else np.nan,
        })
    return pd.DataFrame(records)


eval_results = {}
for lbl in ["1d", "5d", "21d"]:
    target_cls = f"fwd_{lbl}_cls"
    target_reg = f"fwd_{lbl}"
    ev = evaluate_classification(preds_all[lbl], target_cls, target_reg)
    ev["horizon"] = lbl
    eval_results[lbl] = ev
    print(f"\n  === {lbl} horizon ===")
    for _, row in ev.iterrows():
        print(
            f"    {row['model']:<14s}  Acc={row['accuracy']:.1%}  "
            f"AUC={row['auc']:.4f}  IC={row['ic']:+.4f}  "
            f"F1={row['f1']:.3f}  "
            f"Fold IC={row['ic_mean_fold']:+.4f}±{row['ic_std_fold']:.4f}  "
            f"IC>0: {row['ic_hit_rate']:.0%}"
        )

all_eval = pd.concat(eval_results.values(), ignore_index=True)
all_eval.to_csv(ARTIFACT_DIR / "classification_evaluation.csv", index=False, float_format="%.6f")

# ===================================================================
# 6. Cross-method comparison (regression vs classification)
# ===================================================================
print("\n--- Classification vs Regression comparison (5d) ---")

# Load Step 2 and 3 results for comparison
step2_path = ARTIFACT_DIR.parent / "step_02" / "linear_models_evaluation.csv"
step3_path = ARTIFACT_DIR.parent / "step_03" / "tree_models_evaluation.csv"

comparison_rows = []
if step2_path.exists():
    s2 = pd.read_csv(step2_path)
    s2_5d = s2[s2["horizon"] == "5d"]
    for _, row in s2_5d.iterrows():
        comparison_rows.append({
            "model": row["model"],
            "type": "Regression",
            "ic": row["ic"],
            "hit_rate": row["hit_rate"],
        })

if step3_path.exists():
    s3 = pd.read_csv(step3_path)
    s3_5d = s3[s3["horizon"] == "5d"]
    for _, row in s3_5d.iterrows():
        comparison_rows.append({
            "model": row["model"],
            "type": "Regression (Tree)",
            "ic": row["ic"],
            "hit_rate": row["hit_rate"],
        })

ev5 = eval_results["5d"]
for _, row in ev5.iterrows():
    comparison_rows.append({
        "model": row["model"],
        "type": "Classification",
        "ic": row["ic"],
        "hit_rate": row["accuracy"],
    })

comparison_df = pd.DataFrame(comparison_rows).sort_values("ic", ascending=False)
print("\n  All models ranked by IC (5d horizon):")
for _, row in comparison_df.iterrows():
    print(f"    {row['type']:<20s} {row['model']:<14s}  IC={row['ic']:+.4f}  "
          f"Hit/Acc={row['hit_rate']:.1%}")
comparison_df.to_csv(ARTIFACT_DIR / "all_methods_comparison.csv", index=False, float_format="%.6f")

# ===================================================================
# 7. Per-fold IC time series
# ===================================================================
fold_ic_records = []
for model_name in MODELS:
    model_preds = preds_all["5d"][preds_all["5d"]["model"] == model_name]
    for fold_id, fold_grp in model_preds.groupby("fold"):
        fm = fold_grp["y_prob"].notna() & fold_grp[TARGET_REG].notna()
        if fm.sum() < 20:
            continue
        fic = sp_stats.spearmanr(
            fold_grp.loc[fm, TARGET_REG], fold_grp.loc[fm, "y_prob"]
        ).statistic
        mid_date = fold_grp["ts"].median()
        fold_ic_records.append({
            "model": model_name,
            "fold": fold_id,
            "date": mid_date,
            "ic": fic,
        })
fold_ic_df = pd.DataFrame(fold_ic_records)
fold_ic_df.to_csv(ARTIFACT_DIR / "fold_ic_timeseries.csv", index=False, float_format="%.6f")

# ===================================================================
# 8. Backtest from best classifier
# ===================================================================
print("\n--- Backtest: probability-weighted portfolio ---")

best_model = str(ev5.loc[ev5["ic"].idxmax(), "model"])
print(f"  Best classifier by IC: {best_model}")

best_preds = preds_all["5d"][preds_all["5d"]["model"] == best_model].copy()

# Strategy: rank by predicted probability, long top quintile
best_preds["prob_rank"] = best_preds.groupby("ts")["y_prob"].rank(pct=True)
top_q = best_preds[best_preds["prob_rank"] >= 0.80].copy()
counts = top_q.groupby("ts")["symbol"].transform("count")
top_q["weight"] = 1.0 / counts
weights_long = top_q[["ts", "symbol", "weight"]].copy()

bt_metrics_list = []
if len(weights_long) > 0:
    weights_wide = weights_long.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)

    ret_panel = univ[["ts", "symbol", "close"]].copy()
    ret_panel["ret"] = ret_panel.groupby("symbol")["close"].pct_change()
    returns_wide = ret_panel.pivot(index="ts", columns="symbol", values="ret").fillna(0.0)

    bt = simple_backtest(weights_wide, returns_wide, cost_bps=20.0)
    bt_metrics = compute_metrics(pd.Series(bt["portfolio_equity"].values, index=bt["ts"]))
    bt_metrics["label"] = f"{best_model} Top-Q Clf L/O"
    bt_metrics_list.append(bt_metrics)

    btc_eq = compute_btc_benchmark(panel)
    btc_common = btc_eq.reindex(bt["ts"]).dropna()
    btc_metrics = compute_metrics(btc_common)
    btc_metrics["label"] = "BTC Buy & Hold"
    bt_metrics_list.append(btc_metrics)

    ew_panel = univ[univ["in_universe"]][["ts", "symbol", "close"]].copy()
    ew_panel["ret"] = ew_panel.groupby("symbol")["close"].pct_change()
    ew_daily = ew_panel.groupby("ts")["ret"].mean()
    ew_equity = (1 + ew_daily).cumprod()
    ew_equity = ew_equity.reindex(bt["ts"].values).dropna()
    ew_metrics = compute_metrics(ew_equity)
    ew_metrics["label"] = "Equal-Weight Universe"
    bt_metrics_list.append(ew_metrics)

    print("\n" + format_metrics_table(bt_metrics_list))
    pd.DataFrame(bt_metrics_list).to_csv(
        ARTIFACT_DIR / "backtest_metrics.csv", index=False, float_format="%.4f"
    )

# ===================================================================
# 9. PLOTS
# ===================================================================
print("\n--- Generating plots ---")

model_colors = {
    "LogisticReg": "#1976D2",
    "LinearSVM": "#388E3C",
    "RBF_SVM": "#F57C00",
    "XGB_Clf": "#7B1FA2",
}

# --- 9a. Classification metrics comparison ---
print("  [1/6] Classification metrics comparison")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics_to_plot = ["accuracy", "auc", "ic"]
titles = ["Accuracy", "AUC-ROC", "Spearman IC"]
for ax, metric, title in zip(axes, metrics_to_plot, titles):
    for i, hz in enumerate(["1d", "5d", "21d"]):
        ev = eval_results[hz].set_index("model")
        x = np.arange(len(MODELS))
        vals = [ev.loc[m, metric] if m in ev.index else 0 for m in MODELS]
        ax.bar(x + i * 0.22, vals, 0.22, label=f"{hz}", alpha=0.85)
    ax.set_xticks(np.arange(len(MODELS)) + 0.22)
    ax.set_xticklabels(MODELS.keys(), rotation=20, fontsize=8)
    ax.set_title(title, fontsize=11)
    if metric == "accuracy":
        ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--")
    elif metric == "ic":
        ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=7)
fig.suptitle("Classification Models — OOS Metrics by Horizon", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "classification_metrics.png")
plt.close(fig)

# --- 9b. All methods comparison (IC) ---
print("  [2/6] All methods IC comparison")
fig, ax = plt.subplots(figsize=(12, 7))
comp_sorted = comparison_df.sort_values("ic", ascending=True)
colors = []
for _, row in comp_sorted.iterrows():
    if row["type"] == "Classification":
        colors.append(model_colors.get(row["model"], "#9C27B0"))
    elif row["type"] == "Regression (Tree)":
        colors.append("#A5D6A7")
    else:
        colors.append("#90CAF9")
ax.barh(range(len(comp_sorted)),
        comp_sorted["ic"].values, color=colors, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(comp_sorted)))
labels = [f"{row['model']} ({row['type'][:4]})" for _, row in comp_sorted.iterrows()]
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Out-of-Sample Spearman IC (5d Horizon)", fontsize=11)
ax.set_title("All ML Methods — OOS IC Comparison", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "all_methods_ic_comparison.png")
plt.close(fig)

# --- 9c. Fold IC time series ---
print("  [3/6] Fold IC time series")
fig, ax = plt.subplots(figsize=(14, 6))
for model_name in MODELS:
    sub = fold_ic_df[fold_ic_df["model"] == model_name].sort_values("date")
    if len(sub) > 0:
        ax.plot(sub["date"], sub["ic"], marker="o", markersize=3,
                label=model_name, color=model_colors.get(model_name, "gray"), alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Spearman IC (per fold)", fontsize=11)
ax.set_title("Walk-Forward IC — Classification Models (5d)", fontsize=13)
ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "fold_ic_timeseries.png")
plt.close(fig)

# --- 9d. ROC curve (aggregate, 5d) ---
print("  [4/6] ROC curves")
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(8, 8))
for model_name in MODELS:
    sub = preds_all["5d"][preds_all["5d"]["model"] == model_name]
    mask = sub["y_prob"].notna() & sub["fwd_5d_cls"].notna()
    if mask.sum() > 100:
        fpr, tpr, _ = roc_curve(sub.loc[mask, "fwd_5d_cls"], sub.loc[mask, "y_prob"])
        auc_val = roc_auc_score(sub.loc[mask, "fwd_5d_cls"], sub.loc[mask, "y_prob"])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})",
                color=model_colors.get(model_name, "gray"), linewidth=1.5)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — Classification Models (5d Horizon)", fontsize=13)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "roc_curves.png")
plt.close(fig)

# --- 9e. Backtest equity curve ---
print("  [5/6] Backtest equity curve")
if len(weights_long) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(bt["ts"], bt["portfolio_equity"],
            label=f"{best_model} Top-Q Clf L/O",
            color=model_colors.get(best_model, "#1976D2"), linewidth=1.5)
    btc_aligned = btc_eq.reindex(bt["ts"]).dropna()
    btc_norm = btc_aligned / btc_aligned.iloc[0]
    ax.plot(btc_norm.index, btc_norm.values,
            label="BTC Buy & Hold", color="#FF9800", linewidth=1.0, alpha=0.7)
    if len(ew_equity) > 0:
        ew_norm = ew_equity / ew_equity.iloc[0]
        ax.plot(ew_norm.index, ew_norm.values,
                label="EW Universe", color="#9E9E9E", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("Equity (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"Classification Backtest — {best_model} Top-Quintile Long-Only", fontsize=13)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    eq_s = pd.Series(bt["portfolio_equity"].values, index=bt["ts"])
    dd = eq_s / eq_s.cummax() - 1.0
    ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.4)
    ax2.set_ylabel("Drawdown", fontsize=10)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "backtest_equity_curve.png")
    plt.close(fig)

# --- 9f. Probability calibration ---
print("  [6/6] Probability calibration plot")
fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 4))
if len(MODELS) == 1:
    axes = [axes]
for ax, model_name in zip(axes, MODELS):
    sub = preds_all["5d"][preds_all["5d"]["model"] == model_name]
    mask = sub["y_prob"].notna() & sub["fwd_5d_cls"].notna()
    if mask.sum() < 100:
        continue
    probs = sub.loc[mask, "y_prob"].values
    actual = sub.loc[mask, "fwd_5d_cls"].values
    # Bin probabilities
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_idx = np.digitize(probs, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)
    bin_means = [actual[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan
                 for i in range(len(bin_centers))]
    ax.plot(bin_centers, bin_means, "o-", color=model_colors.get(model_name, "gray"))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("Predicted Prob", fontsize=8)
    ax.set_ylabel("Actual Frequency", fontsize=8)
    ax.set_title(model_name, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.suptitle("Probability Calibration — 5d Horizon", fontsize=12)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "probability_calibration.png")
plt.close(fig)

# ===================================================================
# 10. Summary
# ===================================================================
print("\n" + "=" * 70)
print("STEP 4 RESULTS SUMMARY")
print("=" * 70)

print(f"\nWalk-forward: {len(splits)} folds")
print(f"Features: {len(feat_cols)} (raw)")
print(f"Class balance: {univ['fwd_5d_cls'].mean():.1%} positive (5d)")

print("\n--- Classification metrics (5d horizon) ---")
ev5 = eval_results["5d"]
for _, row in ev5.iterrows():
    print(f"  {row['model']:<14s}  Acc={row['accuracy']:.1%}  AUC={row['auc']:.4f}  "
          f"IC={row['ic']:+.4f}  F1={row['f1']:.3f}  "
          f"IC>0: {row['ic_hit_rate']:.0%} of folds")

print("\n--- Classification vs Regression (5d, top models) ---")
best_clf_ic = ev5["ic"].max()
best_clf_name = str(ev5.loc[ev5["ic"].idxmax(), "model"])
print(f"  Best classifier:   {best_clf_name:<14s} IC={best_clf_ic:+.4f}")
if step2_path.exists():
    s2_best = s2[s2["horizon"] == "5d"]["ic"].max()
    print(f"  Best linear (reg): Ridge          IC={s2_best:+.4f}")
if step3_path.exists():
    s3_best = s3[s3["horizon"] == "5d"]["ic"].max()
    print(f"  Best tree (reg):   XGBoost        IC={s3_best:+.4f}")

if bt_metrics_list:
    print(f"\n--- {best_model} top-quintile backtest ---")
    print(format_metrics_table(bt_metrics_list))
    avg_h = weights_wide.gt(0).sum(axis=1).mean()
    print(f"\n  Avg holdings: {avg_h:.1f} symbols")
    print(f"  Avg turnover: {bt['turnover'].mean():.1%} per day")

print(f"\nArtifacts saved to: {ARTIFACT_DIR}")
print("Done.")
