"""
Step 2 — Linear Models Baseline
=================================
JPM Big Data & AI Strategies: Crypto Recreation
Kolanovic & Krishnamachari (2017)

This script:
  1. Loads data, computes 54 TA-Lib features + cross-sectional ranks.
  2. Defines prediction targets (forward 1d, 5d, 21d returns).
  3. Runs walk-forward validation with 4 linear models:
       OLS, Ridge, LASSO, Elastic Net
  4. Features are standardized within each training window.
  5. Evaluates out-of-sample predictions: IC, hit rate, RMSE, Pearson corr.
  6. Examines LASSO feature selection (non-zero coefficients).
  7. Converts the best model's predictions into a simple long-only backtest.
  8. Produces diagnostic plots and summary tables.

The paper (Part III, Chapters 5-6) emphasizes that linear models are the
natural starting point — interpretable, fast, and provide a baseline that
tree-based and deep learning methods must beat.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

# Force unbuffered stdout so progress prints appear immediately
sys.stdout.reconfigure(line_buffering=True)

from scripts.research.jpm_bigdata_ai.helpers import (
    FEATURE_COLS,
    PAPER_REF,
    add_cross_sectional_ranks,
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "jpm_bigdata_ai" / "step_02"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fwd_5d"
TARGET_HORIZON = 5
ALL_TARGETS = {"fwd_1d": 1, "fwd_5d": 5, "fwd_21d": 21}

TRAIN_DAYS = 365 * 2
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRAIN_DAYS = 365

MODELS = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "LASSO": Lasso(alpha=0.001, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ===================================================================
# 1. Load & prepare data
# ===================================================================
print("=" * 70)
print("STEP 2: Linear Models Baseline")
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

# Add cross-sectional ranks
univ = add_cross_sectional_ranks(univ, feat_cols)
rank_cols = [f"{c}_xsrank" for c in feat_cols]
all_feat_cols = feat_cols + rank_cols

print(f"\nIn-universe: {len(univ):,} rows, {univ['symbol'].nunique()} symbols")
print(f"Features: {len(all_feat_cols)} (raw + xsrank)")
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
# 3. Walk-forward evaluation — all models, all horizons
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
    """Run walk-forward for all models. Returns (predictions_df, coef_history)."""
    all_preds = []
    coef_history: dict[str, list] = {name: [] for name in models}

    # Pre-filter rows with any NaN in features or target (avoids per-fold dropna)
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

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_test_s = np.clip(X_test_s, -5, 5)

        fold = fold_info["fold"]
        for model_name, model_template in models.items():
            model = clone(model_template)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            pred_df = test[["ts", "symbol", target_col]].copy()
            pred_df["y_pred"] = y_pred
            pred_df["model"] = model_name
            pred_df["fold"] = fold
            all_preds.append(pred_df)

            if hasattr(model, "coef_"):
                coef_history[model_name].append({
                    "fold": fold,
                    "coefs": model.coef_.copy(),
                })

        if (si + 1) % 5 == 0 or si == n_splits - 1:
            print(f"    [{label}] fold {si+1}/{n_splits} "
                  f"(train={len(train):,}, test={len(test):,})")

    preds = pd.concat(all_preds, ignore_index=True)
    return preds, coef_history


# Run for primary target (5d)
preds_5d, coef_hist_5d = run_walk_forward(univ, all_feat_cols, TARGET, MODELS, splits, label="5d")
print(f"  5d predictions: {len(preds_5d):,} rows")

# Run for 1d and 21d
preds_1d, _ = run_walk_forward(univ, all_feat_cols, "fwd_1d", MODELS, splits, label="1d")
print(f"  1d predictions: {len(preds_1d):,} rows")

preds_21d, _ = run_walk_forward(univ, all_feat_cols, "fwd_21d", MODELS, splits, label="21d")
print(f"  21d predictions: {len(preds_21d):,} rows")

# ===================================================================
# 4. Evaluate predictions
# ===================================================================
print("\n--- Out-of-sample evaluation ---")


def evaluate_model_preds(preds: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Compute aggregate metrics per model from walk-forward predictions."""
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

        # Per-fold IC for stability
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


# Evaluate all horizons
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
            f"    {row['model']:<12s}  IC={row['ic']:+.4f}  "
            f"Pearson={row['pearson']:+.4f}  Hit={row['hit_rate']:.1%}  "
            f"RMSE={row['rmse']:.4f}  "
            f"IC_fold_mean={row['ic_mean_fold']:+.4f} ± {row['ic_std_fold']:.4f}  "
            f"IC>0: {row['ic_hit_rate']:.0%}"
        )

all_eval = pd.concat(eval_results.values(), ignore_index=True)
all_eval.to_csv(ARTIFACT_DIR / "linear_models_evaluation.csv", index=False, float_format="%.6f")

# ===================================================================
# 5. LASSO feature selection analysis
# ===================================================================
print("\n--- LASSO feature selection (5d horizon) ---")

if coef_hist_5d.get("LASSO"):
    coef_matrix = np.array([h["coefs"] for h in coef_hist_5d["LASSO"]])
    mean_coefs = coef_matrix.mean(axis=0)
    nonzero_pct = (np.abs(coef_matrix) > 1e-10).mean(axis=0)

    lasso_importance = pd.DataFrame({
        "feature": all_feat_cols,
        "mean_coef": mean_coefs,
        "abs_mean_coef": np.abs(mean_coefs),
        "nonzero_pct": nonzero_pct,
    }).sort_values("abs_mean_coef", ascending=False)

    lasso_importance.to_csv(ARTIFACT_DIR / "lasso_feature_importance.csv", index=False, float_format="%.6f")

    print(f"  Features with non-zero coef in >50% of folds:")
    selected = lasso_importance[lasso_importance["nonzero_pct"] > 0.5]
    for _, row in selected.head(20).iterrows():
        print(f"    {row['feature']:<30s}  coef={row['mean_coef']:+.6f}  "
              f"selected {row['nonzero_pct']:.0%} of folds")

    n_selected = (lasso_importance["nonzero_pct"] > 0.5).sum()
    n_always = (lasso_importance["nonzero_pct"] > 0.95).sum()
    n_never = (lasso_importance["nonzero_pct"] < 0.05).sum()
    print(f"\n  Summary: {n_selected} features selected >50%, "
          f"{n_always} always selected, {n_never} never selected "
          f"(out of {len(all_feat_cols)})")

# Ridge coefficient analysis
print("\n--- Ridge coefficient magnitudes (5d horizon) ---")
if coef_hist_5d.get("Ridge"):
    ridge_coefs = np.array([h["coefs"] for h in coef_hist_5d["Ridge"]])
    ridge_mean = np.abs(ridge_coefs).mean(axis=0)
    ridge_imp = pd.DataFrame({
        "feature": all_feat_cols,
        "mean_abs_coef": ridge_mean,
    }).sort_values("mean_abs_coef", ascending=False)
    ridge_imp.to_csv(ARTIFACT_DIR / "ridge_feature_importance.csv", index=False, float_format="%.6f")
    print("  Top 15 by mean |coefficient|:")
    for _, row in ridge_imp.head(15).iterrows():
        print(f"    {row['feature']:<30s}  |coef|={row['mean_abs_coef']:.6f}")

# ===================================================================
# 6. Per-fold IC time series
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
# 7. Simple backtest from best model predictions
# ===================================================================
print("\n--- Backtest: converting predictions to portfolio ---")

# Use Ridge (typically best bias-variance tradeoff for linear models)
best_model = "Ridge"
best_preds = preds_5d[preds_5d["model"] == best_model].copy()

# Strategy: rank predictions cross-sectionally, go long top quintile
best_preds["pred_rank"] = best_preds.groupby("ts")["y_pred"].rank(pct=True)

# Build weight matrix: equal-weight top 20% predicted (vectorized)
top_q = best_preds[best_preds["pred_rank"] >= 0.80].copy()
counts = top_q.groupby("ts")["symbol"].transform("count")
top_q["weight"] = 1.0 / counts
weights_long = top_q[["ts", "symbol", "weight"]].copy()
if len(weights_long) > 0:
    weights_wide = weights_long.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)

    # Build return matrix from panel
    ret_panel = univ[["ts", "symbol", "close"]].copy()
    ret_panel["ret"] = ret_panel.groupby("symbol")["close"].pct_change()
    returns_wide = ret_panel.pivot(index="ts", columns="symbol", values="ret").fillna(0.0)

    bt = simple_backtest(weights_wide, returns_wide, cost_bps=20.0)
    bt_metrics = compute_metrics(pd.Series(bt["portfolio_equity"].values, index=bt["ts"]))
    bt_metrics["label"] = f"{best_model} Top-Quintile L/O"

    # BTC benchmark
    btc_eq = compute_btc_benchmark(panel)
    btc_common = btc_eq.reindex(bt["ts"]).dropna()
    btc_metrics = compute_metrics(btc_common)
    btc_metrics["label"] = "BTC Buy & Hold"

    # Equal-weight universe benchmark
    ew_panel = univ[univ["in_universe"]][["ts", "symbol", "close"]].copy()
    ew_panel["ret"] = ew_panel.groupby("symbol")["close"].pct_change()
    ew_daily = ew_panel.groupby("ts")["ret"].mean()
    ew_equity = (1 + ew_daily).cumprod()
    ew_equity = ew_equity.reindex(bt["ts"].values).dropna()
    ew_metrics = compute_metrics(ew_equity)
    ew_metrics["label"] = "Equal-Weight Universe"

    print("\n" + format_metrics_table([bt_metrics, ew_metrics, btc_metrics]))
    pd.DataFrame([bt_metrics, ew_metrics, btc_metrics]).to_csv(
        ARTIFACT_DIR / "backtest_metrics.csv", index=False, float_format="%.4f"
    )

# ===================================================================
# 8. PLOTS
# ===================================================================
print("\n--- Generating plots ---")

# --- 8a. Model comparison bar chart ---
print("  [1/6] Model comparison (IC by horizon)")
fig, ax = plt.subplots(figsize=(10, 6))
horizons = ["1d", "5d", "21d"]
x = np.arange(len(MODELS))
width = 0.22
for i, hz in enumerate(horizons):
    ev = eval_results[hz].set_index("model")
    vals = [ev.loc[m, "ic"] if m in ev.index else 0 for m in MODELS]
    ax.bar(x + i * width, vals, width, label=f"{hz} horizon", alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(MODELS.keys())
ax.set_ylabel("Out-of-Sample Spearman IC", fontsize=11)
ax.set_title("Linear Models — OOS Prediction IC by Horizon", fontsize=13)
ax.legend()
ax.axhline(0, color="black", linewidth=0.5)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "model_ic_comparison.png")
plt.close(fig)

# --- 8b. Hit rate comparison ---
print("  [2/6] Hit rate comparison")
fig, ax = plt.subplots(figsize=(10, 6))
for i, hz in enumerate(horizons):
    ev = eval_results[hz].set_index("model")
    vals = [ev.loc[m, "hit_rate"] if m in ev.index else 0.5 for m in MODELS]
    ax.bar(x + i * width, vals, width, label=f"{hz} horizon", alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(MODELS.keys())
ax.set_ylabel("Directional Hit Rate", fontsize=11)
ax.set_title("Linear Models — OOS Directional Accuracy", fontsize=13)
ax.axhline(0.5, color="red", linewidth=1, linestyle="--", label="Random (50%)")
ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "model_hitrate_comparison.png")
plt.close(fig)

# --- 8c. Fold IC time series ---
print("  [3/6] Fold IC time series")
fig, ax = plt.subplots(figsize=(14, 6))
colors = {"OLS": "#1976D2", "Ridge": "#388E3C", "LASSO": "#F57C00", "ElasticNet": "#7B1FA2"}
for model_name in MODELS:
    sub = fold_ic_df[fold_ic_df["model"] == model_name].sort_values("date")
    ax.plot(sub["date"], sub["ic"], marker="o", markersize=3,
            label=model_name, color=colors.get(model_name, "gray"), alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Spearman IC (per fold)", fontsize=11)
ax.set_title("Walk-Forward IC Over Time — Linear Models (5d Horizon)", fontsize=13)
ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "fold_ic_timeseries.png")
plt.close(fig)

# --- 8d. LASSO feature selection ---
print("  [4/6] LASSO feature selection")
if coef_hist_5d.get("LASSO"):
    top30 = lasso_importance.head(30)
    fig, ax = plt.subplots(figsize=(10, 10))
    colors_bar = ["#2196F3" if v > 0 else "#F44336" for v in top30["mean_coef"]]
    ax.barh(range(len(top30)), top30["mean_coef"].values, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["feature"].values, fontsize=7)
    ax.set_xlabel("Mean LASSO Coefficient (across folds)", fontsize=10)
    ax.set_title("LASSO Feature Selection — Top 30 by |Coefficient|", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "lasso_feature_selection.png")
    plt.close(fig)

# --- 8e. Backtest equity curve ---
print("  [5/6] Backtest equity curve")
if len(weights_long) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(bt["ts"], bt["portfolio_equity"], label=f"{best_model} Top-Quintile L/O",
            color="#1976D2", linewidth=1.5)
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
    ax.set_title(f"Linear Model Backtest — {best_model} Top-Quintile Long-Only", fontsize=13)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    eq_series = pd.Series(bt["portfolio_equity"].values, index=bt["ts"])
    dd = eq_series / eq_series.cummax() - 1.0
    ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.4)
    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.set_xlabel("")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "backtest_equity_curve.png")
    plt.close(fig)

# --- 8f. Prediction scatter (5d, Ridge) ---
print("  [6/6] Prediction scatter plot")
ridge_preds = preds_5d[preds_5d["model"] == best_model].copy()
sample = ridge_preds.dropna(subset=[TARGET, "y_pred"]).sample(
    min(5000, len(ridge_preds)), random_state=42
)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(sample[TARGET], sample["y_pred"], alpha=0.08, s=5, color="#1976D2")
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
fig.savefig(ARTIFACT_DIR / "prediction_scatter_ridge.png")
plt.close(fig)

# ===================================================================
# 9. Summary
# ===================================================================
print("\n" + "=" * 70)
print("STEP 2 RESULTS SUMMARY")
print("=" * 70)

print(f"\nWalk-forward: {len(splits)} folds, "
      f"{TRAIN_DAYS//365}yr train / {TEST_DAYS}d test / {STEP_DAYS}d step")
print(f"Features: {len(all_feat_cols)} (54 raw + 54 xsrank)")

print("\n--- OOS IC summary (5d horizon) ---")
ev5 = eval_results["5d"]
for _, row in ev5.iterrows():
    print(f"  {row['model']:<12s}  IC={row['ic']:+.4f}  Hit={row['hit_rate']:.1%}  "
          f"Fold IC: {row['ic_mean_fold']:+.4f} ± {row['ic_std_fold']:.4f} "
          f"(positive in {row['ic_hit_rate']:.0%} of folds)")

print("\n--- IC across horizons ---")
for hz in horizons:
    ev = eval_results[hz]
    best = ev.loc[ev["ic"].abs().idxmax()]
    print(f"  {hz:<5s}  Best: {best['model']:<12s} IC={best['ic']:+.4f}")

if coef_hist_5d.get("LASSO"):
    n_sel = (lasso_importance["nonzero_pct"] > 0.5).sum()
    print(f"\nLASSO selects {n_sel}/{len(all_feat_cols)} features consistently")

if len(weights_long) > 0:
    print(f"\n--- {best_model} top-quintile backtest ---")
    print(format_metrics_table([bt_metrics, ew_metrics, btc_metrics]))
    avg_holdings = weights_wide.gt(0).sum(axis=1).mean()
    print(f"\n  Avg holdings: {avg_holdings:.1f} symbols")
    print(f"  Avg turnover: {bt['turnover'].mean():.1%} per day")

print(f"\nArtifacts saved to: {ARTIFACT_DIR}")
print("Done.")
