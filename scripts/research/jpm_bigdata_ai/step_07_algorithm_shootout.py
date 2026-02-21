"""
Step 7 — Algorithm Shootout: Unified Comparison
=================================================
JPM Big Data & AI Strategies: Crypto Recreation
Kolanovic & Krishnamachari (2017)

This script runs EVERY model from Steps 2-6 on the exact same data pipeline:
  - Same 54 raw features (no cross-sectional ranks — common denominator)
  - Same walk-forward splits (45 folds, 2-year rolling, 63-day test)
  - Same 5d forward return target
  - Same StandardScaler preprocessing

Models tested (9 total):
  Regression:     Ridge, LASSO, ElasticNet
  Tree:           XGBoost, LightGBM, RandomForest
  Classification: LogisticReg, XGB_Clf (predicting P(ret>0) as continuous signal)
  Deep Learning:  MLP (feedforward net)

We skip:
  - OLS: numerically unstable on this dataset (RMSE > 1e9)
  - LSTM: destructive IC (-0.053), very slow, needs per-symbol sequencing

The shootout produces:
  1. Unified leaderboard (IC, hit rate, RMSE)
  2. Per-fold IC distributions with error bars
  3. Pairwise statistical significance tests (paired t-test on fold ICs)
  4. IC time series across folds
  5. Top-quintile backtest for the winner
  6. Ensemble (average of top-3 model predictions) vs best single model
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

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

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# Use CPU for MLP to avoid MPS + XGBoost/LightGBM threading conflicts
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "jpm_bigdata_ai" / "step_07"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fwd_5d"
ALL_TARGETS = {"fwd_1d": 1, "fwd_5d": 5, "fwd_21d": 21}

TRAIN_DAYS = 365 * 2
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRAIN_DAYS = 365

BATCH_SIZE = 256
MAX_EPOCHS = 80
PATIENCE = 8
LR = 1e-3

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ===================================================================
# Model definitions
# ===================================================================
class MLP(nn.Module):
    """2-layer feedforward network with LayerNorm and dropout."""
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    n_features: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Train MLP with early stopping and return test predictions."""
    model = MLP(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Split train into train/val
    val_sz = max(int(len(X_train) * 0.2), 50)
    X_tr = torch.FloatTensor(X_train[:-val_sz]).to(DEVICE)
    y_tr = torch.FloatTensor(y_train[:-val_sz]).to(DEVICE)
    X_val = torch.FloatTensor(X_train[-val_sz:]).to(DEVICE)
    y_val = torch.FloatTensor(y_train[-val_sz:]).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    n = len(X_tr)

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            pred = model(X_tr[idx])
            loss = criterion(pred, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(DEVICE)
        preds = []
        for b in range(0, len(X_t), BATCH_SIZE):
            preds.append(model(X_t[b:b + BATCH_SIZE]).cpu().numpy())
    return np.concatenate(preds)


# ===================================================================
# Model registry
# ===================================================================
def build_model_registry(n_features: int) -> dict:
    """Return dict: model_name -> (model_object, model_type).

    model_type: "regression" | "classification" | "neural"
    """
    models = {}

    # --- Regression ---
    models["Ridge"] = (Ridge(alpha=1.0), "regression")
    models["LASSO"] = (Lasso(alpha=0.001, max_iter=2000), "regression")
    models["ElasticNet"] = (ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000), "regression")

    # --- Trees ---
    if HAS_XGB:
        models["XGBoost"] = (
            xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=4,
                verbosity=0,
                tree_method="hist",
            ),
            "regression",
        )
    if HAS_LGB:
        models["LightGBM"] = (
            lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=1,
                num_threads=1,
                verbose=-1,
            ),
            "regression",
        )
    models["RandomForest"] = (
        RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            n_jobs=4,
            random_state=42,
        ),
        "regression",
    )

    # --- Classification ---
    models["LogisticReg"] = (
        LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
        "classification",
    )
    if HAS_XGB:
        models["XGB_Clf"] = (
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=4,
                verbosity=0,
                tree_method="hist",
            ),
            "classification",
        )

    # MLP handled separately via train_mlp
    models["MLP"] = (None, "neural")

    return models


# ===================================================================
# 1. Load & prepare
# ===================================================================
print("=" * 70)
print("STEP 7: Algorithm Shootout — Unified Comparison")
print(f"Reference: {PAPER_REF}")
print("=" * 70)

panel = load_daily_bars()
panel = filter_universe(panel)
panel = compute_features(panel)
feat_cols = list(FEATURE_COLS)
n_feat = len(feat_cols)


def _add_fwd(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    for name, days in ALL_TARGETS.items():
        g[name] = g["close"].shift(-days) / g["close"] - 1.0
    return g


panel = panel.groupby("symbol", group_keys=False).apply(_add_fwd)
univ = panel[panel["in_universe"]].copy()

print(f"\nIn-universe: {len(univ):,} rows, {univ['symbol'].nunique()} symbols")
print(f"Features: {n_feat} (raw, no cross-sectional ranks)")

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
print(f"Walk-forward splits: {len(splits)}")

# Clean data
required_cols = feat_cols + [TARGET]
valid_mask = univ[required_cols].notna().all(axis=1)
data = univ.loc[valid_mask].copy()
print(f"Clean observations: {len(data):,}")

# ===================================================================
# 3. Run all models through walk-forward
# ===================================================================
model_registry = build_model_registry(n_feat)
model_names = list(model_registry.keys())
print(f"\nModels: {', '.join(model_names)} ({len(model_names)} total)")
print("\n--- Running unified walk-forward ---")

all_preds = {name: [] for name in model_names}
n_splits = len(splits)

from sklearn.base import clone

print("Starting walk-forward loop...", flush=True)

for si, fold_info in enumerate(splits):
    fold = fold_info["fold"]

    train_mask = (data["ts"] >= fold_info["train_start"]) & (data["ts"] <= fold_info["train_end"])
    test_mask = (data["ts"] >= fold_info["test_start"]) & (data["ts"] <= fold_info["test_end"])
    train = data.loc[train_mask]
    test = data.loc[test_mask]

    if len(train) < 200 or len(test) < 10:
        continue

    X_train_raw = train[feat_cols].values
    y_train = train[TARGET].values
    y_train_cls = (y_train > 0).astype(int)
    X_test_raw = test[feat_cols].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s = np.clip(scaler.transform(X_test_raw), -5, 5)

    base_df = test[["ts", "symbol", TARGET]].copy()
    base_df["fold"] = fold

    for name, (model_obj, mtype) in model_registry.items():
        if si == 0:
            print(f"      {name} ({mtype}) ...", end=" ", flush=True)
        try:
            if mtype == "regression":
                m = clone(model_obj)
                m.fit(X_train_s, y_train)
                pred = m.predict(X_test_s)

            elif mtype == "classification":
                m = clone(model_obj)
                m.fit(X_train_s, y_train_cls)
                if hasattr(m, "predict_proba"):
                    pred = m.predict_proba(X_test_s)[:, 1]
                else:
                    pred = m.decision_function(X_test_s)

            elif mtype == "neural":
                pred = train_mlp(n_feat, X_train_s, y_train, X_test_s)

            else:
                continue

            pred_df = base_df.copy()
            pred_df["y_pred"] = pred
            pred_df["model"] = name
            all_preds[name].append(pred_df)
            if si == 0:
                print("ok", flush=True)

        except Exception as e:
            print(f"    [WARN] {name} fold {fold}: {e}", flush=True)

    if (si + 1) % 5 == 0 or si == n_splits - 1:
        print(f"    fold {si+1}/{n_splits} complete", flush=True)

# Combine
preds_list = []
for name in model_names:
    if all_preds[name]:
        preds_list.append(pd.concat(all_preds[name], ignore_index=True))
preds_df = pd.concat(preds_list, ignore_index=True)

print(f"\nTotal predictions: {len(preds_df):,}")
for m in model_names:
    n_m = (preds_df["model"] == m).sum()
    print(f"  {m}: {n_m:,}")

# ===================================================================
# 4. Evaluate — overall & per-fold
# ===================================================================
print("\n--- Evaluation ---")

eval_records = []
fold_ic_records = []

for name in model_names:
    m_preds = preds_df[preds_df["model"] == name]
    mask = m_preds[TARGET].notna() & m_preds["y_pred"].notna()
    yt = m_preds.loc[mask, TARGET]
    yp = m_preds.loc[mask, "y_pred"]
    n = len(yt)
    if n < 50:
        continue

    ic = float(sp_stats.spearmanr(yt, yp).statistic)
    pearson = float(np.corrcoef(yt, yp)[0, 1])
    hit = float(((yt > 0) == (yp > 0)).mean()) if name not in ("LogisticReg", "XGB_Clf") else \
          float(((yt > 0) == (yp > 0.5)).mean())
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))

    # Per-fold ICs
    fold_ics = []
    for fid, fgrp in m_preds.groupby("fold"):
        fm = fgrp[TARGET].notna() & fgrp["y_pred"].notna()
        if fm.sum() < 20:
            continue
        fic = float(sp_stats.spearmanr(fgrp.loc[fm, TARGET], fgrp.loc[fm, "y_pred"]).statistic)
        fold_ics.append(fic)
        fold_ic_records.append({"model": name, "fold": fid, "ic": fic,
                                "date": fgrp["ts"].median()})

    eval_records.append({
        "model": name,
        "ic": ic,
        "pearson": pearson,
        "hit_rate": hit,
        "rmse": rmse,
        "n_obs": n,
        "ic_mean": float(np.mean(fold_ics)) if fold_ics else np.nan,
        "ic_std": float(np.std(fold_ics)) if fold_ics else np.nan,
        "ic_se": float(np.std(fold_ics) / np.sqrt(len(fold_ics))) if fold_ics else np.nan,
        "ic_t": float(np.mean(fold_ics) / (np.std(fold_ics) / np.sqrt(len(fold_ics))))
               if fold_ics and np.std(fold_ics) > 0 else np.nan,
        "ic_hit_pct": float(np.mean([x > 0 for x in fold_ics])) if fold_ics else np.nan,
        "n_folds": len(fold_ics),
    })

eval_df = pd.DataFrame(eval_records).sort_values("ic", ascending=False).reset_index(drop=True)
fold_ic_df = pd.DataFrame(fold_ic_records)

eval_df.to_csv(ARTIFACT_DIR / "shootout_evaluation.csv", index=False, float_format="%.6f")
fold_ic_df.to_csv(ARTIFACT_DIR / "shootout_fold_ics.csv", index=False, float_format="%.6f")

print("\n  Unified Leaderboard (5d, 54 features, same splits):\n")
print(f"  {'Rank':<5s} {'Model':<14s} {'IC':>8s} {'IC(mean)':>10s} {'IC(std)':>9s} "
      f"{'t-stat':>8s} {'IC>0%':>7s} {'Hit':>7s} {'RMSE':>8s}")
print("  " + "-" * 85)
for rank, (_, row) in enumerate(eval_df.iterrows(), 1):
    print(f"  {rank:<5d} {row['model']:<14s} {row['ic']:+8.4f} {row['ic_mean']:+10.4f} "
          f"{row['ic_std']:9.4f} {row['ic_t']:8.2f} {row['ic_hit_pct']:6.0%} "
          f"{row['hit_rate']:6.1%} {row['rmse']:8.4f}")

# ===================================================================
# 5. Pairwise significance (paired t-test on fold ICs)
# ===================================================================
print("\n--- Pairwise significance (paired t-test on fold ICs) ---")

# Build fold IC matrix: rows=folds, cols=models
model_order = eval_df["model"].tolist()
fold_ids = sorted(fold_ic_df["fold"].unique())
ic_matrix = pd.DataFrame(index=fold_ids, columns=model_order, dtype=float)
for _, row in fold_ic_df.iterrows():
    ic_matrix.loc[row["fold"], row["model"]] = row["ic"]

# Pairwise t-test
n_models = len(model_order)
sig_matrix = pd.DataFrame(
    np.ones((n_models, n_models)),
    index=model_order,
    columns=model_order,
)
t_matrix = pd.DataFrame(
    np.zeros((n_models, n_models)),
    index=model_order,
    columns=model_order,
)

for i, m1 in enumerate(model_order):
    for j, m2 in enumerate(model_order):
        if i >= j:
            continue
        v1 = ic_matrix[m1].dropna()
        v2 = ic_matrix[m2].dropna()
        common = v1.index.intersection(v2.index)
        if len(common) < 5:
            continue
        t_stat, p_val = sp_stats.ttest_rel(v1.loc[common], v2.loc[common])
        sig_matrix.loc[m1, m2] = p_val
        sig_matrix.loc[m2, m1] = p_val
        t_matrix.loc[m1, m2] = t_stat
        t_matrix.loc[m2, m1] = -t_stat

sig_matrix.to_csv(ARTIFACT_DIR / "pairwise_pvalues.csv", float_format="%.4f")
t_matrix.to_csv(ARTIFACT_DIR / "pairwise_tstat.csv", float_format="%.4f")

# Print key comparisons
best_model = model_order[0]
print(f"\n  Best model: {best_model}")
print(f"  Pairwise p-values vs {best_model}:")
for m in model_order[1:]:
    pv = sig_matrix.loc[best_model, m]
    sig_str = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
    print(f"    {best_model} vs {m:<14s}  p={pv:.4f} {sig_str}")

# ===================================================================
# 6. Ensemble (average of top-3)
# ===================================================================
print("\n--- Ensemble: average of top-3 model predictions ---")
top3 = model_order[:3]
print(f"  Top-3: {', '.join(top3)}")

ens_dfs = []
for name in top3:
    sub = preds_df[preds_df["model"] == name][["ts", "symbol", "fold", TARGET, "y_pred"]].copy()
    sub = sub.rename(columns={"y_pred": f"pred_{name}"})
    ens_dfs.append(sub)

ens = ens_dfs[0]
for df in ens_dfs[1:]:
    ens = ens.merge(df[["ts", "symbol", "fold", f"pred_{df.columns[-1].replace('pred_', '')}"
                         if False else df.columns[-1]]],
                    on=["ts", "symbol", "fold"], how="inner")

pred_cols = [c for c in ens.columns if c.startswith("pred_")]
ens["y_pred_ensemble"] = ens[pred_cols].mean(axis=1)

# Evaluate ensemble
mask_e = ens[TARGET].notna() & ens["y_pred_ensemble"].notna()
yt_e = ens.loc[mask_e, TARGET]
yp_e = ens.loc[mask_e, "y_pred_ensemble"]
ic_ens = float(sp_stats.spearmanr(yt_e, yp_e).statistic)
hit_ens = float(((yt_e > 0) == (yp_e > 0)).mean())

# Fold-level ensemble IC
ens_fold_ics = []
for fid, fgrp in ens.groupby("fold"):
    fm = fgrp[TARGET].notna() & fgrp["y_pred_ensemble"].notna()
    if fm.sum() < 20:
        continue
    ens_fold_ics.append(
        float(sp_stats.spearmanr(fgrp.loc[fm, TARGET], fgrp.loc[fm, "y_pred_ensemble"]).statistic)
    )
    fold_ic_records.append({"model": "Ensemble", "fold": fid, "ic": ens_fold_ics[-1],
                            "date": fgrp["ts"].median()})

ic_ens_mean = float(np.mean(ens_fold_ics))
ic_ens_std = float(np.std(ens_fold_ics))
ic_ens_t = ic_ens_mean / (ic_ens_std / np.sqrt(len(ens_fold_ics))) if ic_ens_std > 0 else np.nan

print(f"  Ensemble IC={ic_ens:+.4f}  IC(mean)={ic_ens_mean:+.4f}  "
      f"IC(std)={ic_ens_std:.4f}  t={ic_ens_t:.2f}  Hit={hit_ens:.1%}")
print(f"  Best single ({best_model}) IC={eval_df.iloc[0]['ic']:+.4f}  "
      f"IC(mean)={eval_df.iloc[0]['ic_mean']:+.4f}")

# Add ensemble to eval
ens_row = {
    "model": "Ensemble(top3)",
    "ic": ic_ens,
    "pearson": float(np.corrcoef(yt_e, yp_e)[0, 1]),
    "hit_rate": hit_ens,
    "rmse": float(np.sqrt(((yt_e - yp_e) ** 2).mean())),
    "n_obs": len(yt_e),
    "ic_mean": ic_ens_mean,
    "ic_std": ic_ens_std,
    "ic_se": ic_ens_std / np.sqrt(len(ens_fold_ics)),
    "ic_t": ic_ens_t,
    "ic_hit_pct": float(np.mean([x > 0 for x in ens_fold_ics])),
    "n_folds": len(ens_fold_ics),
}
eval_df = pd.concat([eval_df, pd.DataFrame([ens_row])], ignore_index=True)
eval_df = eval_df.sort_values("ic", ascending=False).reset_index(drop=True)
eval_df.to_csv(ARTIFACT_DIR / "shootout_evaluation.csv", index=False, float_format="%.6f")

# Update fold_ic_df
fold_ic_df = pd.DataFrame(fold_ic_records)

# ===================================================================
# 7. Backtest — best single + ensemble
# ===================================================================
print("\n--- Backtest ---")


def _run_backtest(pred_source: pd.DataFrame, pred_col: str, label: str) -> dict | None:
    """Top-quintile long-only backtest."""
    sub = pred_source.copy()
    sub["pred_rank"] = sub.groupby("ts")[pred_col].rank(pct=True)
    top_q = sub[sub["pred_rank"] >= 0.80].copy()
    counts = top_q.groupby("ts")["symbol"].transform("count")
    top_q["weight"] = 1.0 / counts
    wts = top_q[["ts", "symbol", "weight"]].copy()

    if len(wts) == 0:
        return None

    wts_wide = wts.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)
    ret_p = univ[["ts", "symbol", "close"]].copy()
    ret_p["ret"] = ret_p.groupby("symbol")["close"].pct_change()
    ret_wide = ret_p.pivot(index="ts", columns="symbol", values="ret").fillna(0.0)
    bt = simple_backtest(wts_wide, ret_wide, cost_bps=20.0)
    m = compute_metrics(pd.Series(bt["portfolio_equity"].values, index=bt["ts"]))
    m["label"] = label
    return m, bt


bt_metrics = []
bt_curves = {}

# Best single model
best_preds_single = preds_df[preds_df["model"] == best_model].copy()
r = _run_backtest(best_preds_single, "y_pred", f"{best_model} Top-Q L/O")
if r:
    bt_metrics.append(r[0])
    bt_curves[best_model] = r[1]

# Ensemble
ens_bt = ens[["ts", "symbol", "fold", TARGET, "y_pred_ensemble"]].copy()
r = _run_backtest(ens_bt, "y_pred_ensemble", "Ensemble Top-Q L/O")
if r:
    bt_metrics.append(r[0])
    bt_curves["Ensemble"] = r[1]

# BTC benchmark
btc_eq = compute_btc_benchmark(panel)
if bt_curves:
    first_bt = list(bt_curves.values())[0]
    btc_c = btc_eq.reindex(first_bt["ts"]).dropna()
    btc_m = compute_metrics(btc_c)
    btc_m["label"] = "BTC Buy & Hold"
    bt_metrics.append(btc_m)

if bt_metrics:
    print("\n" + format_metrics_table(bt_metrics))
    pd.DataFrame(bt_metrics).to_csv(ARTIFACT_DIR / "backtest_metrics.csv", index=False, float_format="%.4f")

# ===================================================================
# 8. PLOTS
# ===================================================================
print("\n--- Generating plots ---")

TYPE_COLORS = {
    "Ridge": "#42A5F5", "LASSO": "#66BB6A", "ElasticNet": "#26A69A",
    "XGBoost": "#FFA726", "LightGBM": "#FFCA28", "RandomForest": "#8D6E63",
    "LogisticReg": "#AB47BC", "XGB_Clf": "#EC407A",
    "MLP": "#EF5350", "Ensemble(top3)": "#212121",
}

# --- 8a. IC bar chart with error bars ---
print("  [1/7] IC leaderboard bar chart")
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(eval_df))
bars = ax.bar(
    x,
    eval_df["ic_mean"],
    yerr=eval_df["ic_se"],
    capsize=3,
    color=[TYPE_COLORS.get(m, "#9E9E9E") for m in eval_df["model"]],
    alpha=0.85,
    edgecolor="white",
)
ax.set_xticks(list(x))
ax.set_xticklabels(eval_df["model"], rotation=35, ha="right", fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Mean OOS Spearman IC (±SE)", fontsize=11)
ax.set_title("Algorithm Shootout — 5d Forward Return (54 features, 45 folds)", fontsize=13)
for i, row in eval_df.iterrows():
    ax.text(i, row["ic_mean"] + row["ic_se"] + 0.002,
            f"t={row['ic_t']:.1f}", ha="center", fontsize=7, color="dimgray")
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "ic_leaderboard.png")
plt.close(fig)

# --- 8b. IC time series ---
print("  [2/7] IC time series")
fig, ax = plt.subplots(figsize=(14, 6))
for name in list(eval_df["model"].head(5)):
    sub = fold_ic_df[fold_ic_df["model"] == name].sort_values("date")
    if len(sub) > 0:
        ax.plot(sub["date"], sub["ic"], marker="o", markersize=3,
                label=name, color=TYPE_COLORS.get(name, "gray"), alpha=0.8, linewidth=1.2)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Spearman IC (per fold)", fontsize=11)
ax.set_title("Walk-Forward IC — Top 5 Models (5d)", fontsize=13)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "ic_timeseries_top5.png")
plt.close(fig)

# --- 8c. IC box plot ---
print("  [3/7] IC box plot")
box_data = []
box_labels = []
for name in eval_df["model"]:
    sub = fold_ic_df[fold_ic_df["model"] == name]["ic"].dropna().values
    if len(sub) > 0:
        box_data.append(sub)
        box_labels.append(name)

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="black", markersize=5))
for patch, name in zip(bp["boxes"], box_labels):
    patch.set_facecolor(TYPE_COLORS.get(name, "#BDBDBD"))
    patch.set_alpha(0.7)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Spearman IC per fold", fontsize=11)
ax.set_title("IC Distribution Across Folds", fontsize=13)
plt.xticks(rotation=35, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "ic_boxplot.png")
plt.close(fig)

# --- 8d. Significance heatmap ---
print("  [4/7] Significance heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
pv_arr = sig_matrix.loc[model_order, model_order].values.astype(float)
im = ax.imshow(pv_arr, cmap="RdYlGn", vmin=0, vmax=0.2, aspect="auto")
ax.set_xticks(range(n_models))
ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n_models))
ax.set_yticklabels(model_order, fontsize=8)
for i in range(n_models):
    for j in range(n_models):
        if i != j:
            ax.text(j, i, f"{pv_arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="p-value (paired t-test)")
ax.set_title("Pairwise Significance of IC Differences", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "significance_heatmap.png")
plt.close(fig)

# --- 8e. Cumulative IC ---
print("  [5/7] Cumulative IC")
fig, ax = plt.subplots(figsize=(14, 6))
for name in list(eval_df["model"].head(5)):
    sub = fold_ic_df[fold_ic_df["model"] == name].sort_values("date")
    if len(sub) > 1:
        ax.plot(sub["date"], sub["ic"].cumsum(), label=name,
                color=TYPE_COLORS.get(name, "gray"), linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Cumulative IC", fontsize=11)
ax.set_title("Cumulative IC Over Walk-Forward Folds (5d)", fontsize=13)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "cumulative_ic.png")
plt.close(fig)

# --- 8f. Backtest equity ---
print("  [6/7] Backtest equity")
if bt_curves:
    fig, ax = plt.subplots(figsize=(14, 6))
    for label_name, bt in bt_curves.items():
        ax.plot(bt["ts"], bt["portfolio_equity"],
                label=f"{label_name} Top-Q L/O",
                color=TYPE_COLORS.get(label_name, "#1976D2"), linewidth=1.5)
    btc_norm = btc_c / btc_c.iloc[0]
    ax.plot(btc_norm.index, btc_norm.values,
            label="BTC Buy & Hold", color="#FF9800", linewidth=1.0, alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log)", fontsize=11)
    ax.set_title("Backtest — Best Model vs Ensemble vs BTC", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "backtest_equity.png")
    plt.close(fig)

# --- 8g. Model category summary ---
print("  [7/7] Category summary")
cat_map = {
    "Ridge": "Linear", "LASSO": "Linear", "ElasticNet": "Linear",
    "XGBoost": "Tree", "LightGBM": "Tree", "RandomForest": "Tree",
    "LogisticReg": "Classification", "XGB_Clf": "Classification",
    "MLP": "Deep Learning", "Ensemble(top3)": "Ensemble",
}
eval_df["category"] = eval_df["model"].map(cat_map)
cat_df = eval_df.groupby("category").agg(
    best_ic=("ic", "max"),
    mean_ic=("ic", "mean"),
    best_model=("model", "first"),
).sort_values("best_ic", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
cat_colors = {
    "Linear": "#42A5F5", "Tree": "#FFA726", "Classification": "#AB47BC",
    "Deep Learning": "#EF5350", "Ensemble": "#212121",
}
cats = cat_df.index.tolist()
ax.barh(range(len(cats)), cat_df["best_ic"].values,
        color=[cat_colors.get(c, "#9E9E9E") for c in cats], alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(cats)))
ax.set_yticklabels([f"{c}\n({cat_df.loc[c, 'best_model']})" for c in cats], fontsize=9)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Best OOS Spearman IC (5d)", fontsize=11)
ax.set_title("ML Category Comparison", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "category_summary.png")
plt.close(fig)

# ===================================================================
# 9. Summary
# ===================================================================
print("\n" + "=" * 70)
print("STEP 7 RESULTS SUMMARY — ALGORITHM SHOOTOUT")
print("=" * 70)

print(f"\nSetup: {n_feat} raw features, {len(splits)} folds, 5d forward return")
print(f"Models tested: {len(model_names)} + Ensemble")

print("\n--- Final Leaderboard ---\n")
print(f"  {'Rank':<5s} {'Model':<16s} {'IC':>8s} {'IC(mean)':>10s} {'t-stat':>8s} "
      f"{'IC>0%':>7s} {'Hit':>7s}")
print("  " + "-" * 65)
for rank, (_, row) in enumerate(eval_df.iterrows(), 1):
    print(f"  {rank:<5d} {row['model']:<16s} {row['ic']:+8.4f} {row['ic_mean']:+10.4f} "
          f"{row['ic_t']:8.2f} {row['ic_hit_pct']:6.0%} {row['hit_rate']:6.1%}")

if bt_metrics:
    print("\n--- Backtest ---")
    print(format_metrics_table(bt_metrics))

print(f"\n--- Key findings ---")
print(f"  1. Best single model: {eval_df.iloc[0]['model']} (IC={eval_df.iloc[0]['ic']:+.4f})")
ens_idx = eval_df[eval_df["model"] == "Ensemble(top3)"].index
if len(ens_idx) > 0:
    ens_r = eval_df.iloc[ens_idx[0]]
    print(f"  2. Ensemble: IC={ens_r['ic']:+.4f} "
          f"({'beats' if ens_r['ic'] > eval_df.iloc[0]['ic'] else 'trails'} best single)")
print(f"  3. Statistical significance: see pairwise_pvalues.csv")
print(f"  4. Category ranking: {' > '.join(cat_df.index.tolist())}")

print(f"\nArtifacts saved to: {ARTIFACT_DIR}")
print("Done.")
