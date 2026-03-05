"""
Multi-Frequency ML Shootout
============================
Runs the core ML models from the JPM Big Data & AI study at multiple
intraday frequencies: 5m, 30m, 1h, 4h, 8h (plus daily baseline).

For each frequency:
  - Resample 1-min candles via DuckDB time_bucket
  - Compute the same 54 TA-Lib features (same bar-count parameters)
  - 5-day-equivalent forward return target
  - Walk-forward: 2yr train / 63d test / 63d step (calendar days)
  - Key models: Ridge, XGBoost, XGB_Clf, LightGBM, MLP
  - Evaluate IC, hit rate, t-stat

Higher frequencies capture faster dynamics but have more noise.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# Project imports
from scripts.research.common.data import (
    ANN_FACTOR,
    BARS_PER_DAY,
    load_bars,
)
from scripts.research.common.backtest import simple_backtest, DEFAULT_COST_BPS
from scripts.research.common.metrics import compute_metrics, format_metrics_table
from scripts.research.jpm_bigdata_ai.helpers import (
    FEATURE_COLS,
    compute_features,
    filter_universe,
    walk_forward_splits,
    compute_btc_benchmark,
)

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

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FREQUENCIES = ["8h", "4h", "1h", "30m", "5m"]
FWD_DAYS = 5
TRAIN_DAYS = 365 * 2
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRAIN_DAYS = 365

BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 6
LR = 1e-3

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "multifreq" / "ml"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ===================================================================
# MLP definition (same as step_07)
# ===================================================================
class MLP(nn.Module):
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


def train_mlp(n_features, X_train, y_train, X_test):
    model = MLP(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    val_sz = max(int(len(X_train) * 0.15), 50)
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
            idx = perm[i : i + BATCH_SIZE]
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
            preds.append(model(X_t[b : b + BATCH_SIZE]).cpu().numpy())
    return np.concatenate(preds)


# ===================================================================
# Model registry
# ===================================================================
def build_model_registry(n_features: int) -> dict:
    models = {}
    models["Ridge"] = (Ridge(alpha=1.0), "regression")
    if HAS_XGB:
        models["XGBoost"] = (
            xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, n_jobs=4, verbosity=0, tree_method="hist",
            ),
            "regression",
        )
        models["XGB_Clf"] = (
            xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                eval_metric="logloss", n_jobs=4, verbosity=0, tree_method="hist",
            ),
            "classification",
        )
    if HAS_LGB:
        models["LightGBM"] = (
            lgb.LGBMRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, n_jobs=1, num_threads=1, verbose=-1,
            ),
            "regression",
        )
    models["MLP"] = (None, "neural")
    return models


# ===================================================================
# Per-frequency runner
# ===================================================================
def run_frequency(
    freq: str,
    max_train_rows: int = 2_000_000,
) -> dict:
    """Run ML shootout at a single frequency. Returns summary dict."""
    t0 = time.time()
    bpd = BARS_PER_DAY[freq]
    fwd_bars = int(FWD_DAYS * bpd)

    print(f"\n{'='*70}")
    print(f"FREQUENCY: {freq}  (bars/day={bpd:.0f}, fwd_bars={fwd_bars})")
    print(f"{'='*70}")

    # 1. Load data
    panel = load_bars(freq)
    print(f"  Raw panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols")

    # 2. Universe filter (scale ADV window to ~20 calendar days of bars)
    adv_window = max(20, int(20 * bpd))
    panel = filter_universe(panel, min_adv_usd=1_000_000, min_history_days=int(90 * bpd), adv_window=adv_window)

    # 3. Features
    panel = compute_features(panel)
    feat_cols = list(FEATURE_COLS)
    n_feat = len(feat_cols)
    print(f"  Features: {n_feat}")

    # 4. Forward return (N bars forward, equivalent to ~FWD_DAYS days)
    def _add_fwd(g):
        g = g.copy()
        g["fwd_ret"] = g["close"].shift(-fwd_bars) / g["close"] - 1.0
        return g

    panel = panel.groupby("symbol", group_keys=False).apply(_add_fwd)
    univ = panel[panel["in_universe"]].copy()
    print(f"  In-universe: {len(univ):,} rows, {univ['symbol'].nunique()} symbols")

    # 5. Walk-forward splits (on calendar dates derived from bar timestamps)
    univ["_cal_date"] = univ["ts"].dt.date
    cal_dates = np.sort(univ["_cal_date"].unique())

    # Build date-level splits using the existing function
    splits = walk_forward_splits(
        pd.DatetimeIndex(pd.to_datetime(cal_dates)),
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        min_train_days=MIN_TRAIN_DAYS,
    )
    print(f"  Walk-forward splits: {len(splits)}")

    # 6. Clean data
    required_cols = feat_cols + ["fwd_ret"]
    valid_mask = univ[required_cols].notna().all(axis=1)
    data = univ.loc[valid_mask].copy()
    print(f"  Clean observations: {len(data):,}")

    if len(data) < 5000:
        print(f"  [SKIP] Not enough data for {freq}")
        return {"freq": freq, "status": "skipped", "reason": "insufficient data"}

    # 7. Build models and run walk-forward
    model_registry = build_model_registry(n_feat)
    model_names = list(model_registry.keys())
    all_preds = {name: [] for name in model_names}

    print(f"  Models: {', '.join(model_names)}")
    print(f"  Running walk-forward ...", flush=True)

    for si, fold_info in enumerate(splits):
        fold = fold_info["fold"]
        train_start = fold_info["train_start"]
        train_end = fold_info["train_end"]
        test_start = fold_info["test_start"]
        test_end = fold_info["test_end"]

        # Map calendar dates to bar timestamps
        train_mask = (data["_cal_date"] >= train_start.date()) & (data["_cal_date"] <= train_end.date())
        test_mask = (data["_cal_date"] >= test_start.date()) & (data["_cal_date"] <= test_end.date())
        train = data.loc[train_mask]
        test = data.loc[test_mask]

        if len(train) < 500 or len(test) < 50:
            continue

        # Subsample training if too large (for computational feasibility)
        if len(train) > max_train_rows:
            step = len(train) // max_train_rows + 1
            train = train.iloc[::step]

        X_train_raw = train[feat_cols].values
        y_train = train["fwd_ret"].values
        y_train_cls = (y_train > 0).astype(int)
        X_test_raw = test[feat_cols].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_raw)
        X_test_s = np.clip(scaler.transform(X_test_raw), -5, 5)

        base_df = test[["ts", "symbol", "fwd_ret"]].copy()
        base_df["fold"] = fold

        for name, (model_obj, mtype) in model_registry.items():
            try:
                if mtype == "regression":
                    m = clone(model_obj)
                    m.fit(X_train_s, y_train)
                    pred = m.predict(X_test_s)
                elif mtype == "classification":
                    m = clone(model_obj)
                    m.fit(X_train_s, y_train_cls)
                    pred = m.predict_proba(X_test_s)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X_test_s)
                elif mtype == "neural":
                    pred = train_mlp(n_feat, X_train_s, y_train, X_test_s)
                else:
                    continue

                pred_df = base_df.copy()
                pred_df["y_pred"] = pred
                pred_df["model"] = name
                all_preds[name].append(pred_df)
            except Exception as e:
                if si == 0:
                    print(f"      [WARN] {name} fold {fold}: {e}", flush=True)

        if (si + 1) % 10 == 0 or si == len(splits) - 1:
            print(f"    fold {si+1}/{len(splits)} done  (train={len(train):,}, test={len(test):,})", flush=True)

    # 8. Combine and evaluate
    preds_list = []
    for name in model_names:
        if all_preds[name]:
            preds_list.append(pd.concat(all_preds[name], ignore_index=True))

    if not preds_list:
        print(f"  [SKIP] No predictions produced for {freq}")
        return {"freq": freq, "status": "no_predictions"}

    preds_df = pd.concat(preds_list, ignore_index=True)

    eval_records = []
    for name in model_names:
        m_preds = preds_df[preds_df["model"] == name]
        mask = m_preds["fwd_ret"].notna() & m_preds["y_pred"].notna()
        yt = m_preds.loc[mask, "fwd_ret"]
        yp = m_preds.loc[mask, "y_pred"]
        n = len(yt)
        if n < 50:
            continue

        ic = float(sp_stats.spearmanr(yt, yp).statistic)
        hit = float(((yt > 0) == (yp > 0)).mean()) if name not in ("XGB_Clf",) else float(((yt > 0) == (yp > 0.5)).mean())

        # Per-fold IC
        fold_ics = []
        for fid, fgrp in m_preds.groupby("fold"):
            fm = fgrp["fwd_ret"].notna() & fgrp["y_pred"].notna()
            if fm.sum() < 20:
                continue
            fic = float(sp_stats.spearmanr(fgrp.loc[fm, "fwd_ret"], fgrp.loc[fm, "y_pred"]).statistic)
            fold_ics.append(fic)

        ic_mean = float(np.mean(fold_ics)) if fold_ics else np.nan
        ic_std = float(np.std(fold_ics)) if fold_ics else np.nan
        ic_t = ic_mean / (ic_std / np.sqrt(len(fold_ics))) if fold_ics and ic_std > 0 else np.nan

        eval_records.append({
            "freq": freq,
            "model": name,
            "ic": ic,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_t": ic_t,
            "ic_hit_pct": float(np.mean([x > 0 for x in fold_ics])) if fold_ics else np.nan,
            "hit_rate": hit,
            "n_obs": n,
            "n_folds": len(fold_ics),
        })

    elapsed = time.time() - t0
    eval_df = pd.DataFrame(eval_records).sort_values("ic", ascending=False).reset_index(drop=True)

    # Save per-frequency results
    eval_df.to_csv(ARTIFACT_DIR / f"eval_{freq}.csv", index=False, float_format="%.6f")

    print(f"\n  --- {freq} Leaderboard ---")
    print(f"  {'Model':<14s} {'IC':>8s} {'IC(mean)':>10s} {'t-stat':>8s} {'IC>0%':>7s} {'Hit':>7s}")
    print(f"  {'-'*60}")
    for _, row in eval_df.iterrows():
        print(f"  {row['model']:<14s} {row['ic']:+8.4f} {row['ic_mean']:+10.4f} "
              f"{row['ic_t']:8.2f} {row['ic_hit_pct']:6.0%} {row['hit_rate']:6.1%}")
    print(f"\n  Elapsed: {elapsed:.0f}s")

    return {
        "freq": freq,
        "status": "ok",
        "eval_df": eval_df,
        "elapsed": elapsed,
        "n_obs": len(data),
        "n_symbols": data["symbol"].nunique(),
        "n_splits": len(splits),
    }


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("MULTI-FREQUENCY ML SHOOTOUT")
    print("=" * 70)
    print(f"Frequencies: {FREQUENCIES}")
    print(f"Forward return: {FWD_DAYS}d equivalent")
    print(f"Walk-forward: {TRAIN_DAYS}d train / {TEST_DAYS}d test / {STEP_DAYS}d step")
    print()

    all_results = []

    # Also run daily as baseline
    for freq in FREQUENCIES + ["1d"]:
        try:
            result = run_frequency(freq)
            all_results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] {freq}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"freq": freq, "status": "error", "reason": str(e)})

    # ===================================================================
    # Cross-frequency comparison
    # ===================================================================
    print("\n" + "=" * 70)
    print("CROSS-FREQUENCY COMPARISON")
    print("=" * 70)

    cross_records = []
    for r in all_results:
        if r.get("status") != "ok":
            continue
        for _, row in r["eval_df"].iterrows():
            cross_records.append(row.to_dict())

    if not cross_records:
        print("No results to compare.")
        return

    cross_df = pd.DataFrame(cross_records)
    cross_df.to_csv(ARTIFACT_DIR / "cross_frequency_results.csv", index=False, float_format="%.6f")

    # Best model per frequency
    print("\n  Best model per frequency:")
    print(f"  {'Freq':<8s} {'Model':<14s} {'IC':>8s} {'IC(mean)':>10s} {'t-stat':>8s} {'Elapsed':>10s}")
    print(f"  {'-'*60}")
    for r in all_results:
        if r.get("status") != "ok":
            print(f"  {r['freq']:<8s} {'---':<14s} {'---':>8s} {'---':>10s} {'---':>8s} {'---':>10s}")
            continue
        best = r["eval_df"].iloc[0]
        print(f"  {r['freq']:<8s} {best['model']:<14s} {best['ic']:+8.4f} "
              f"{best['ic_mean']:+10.4f} {best['ic_t']:8.2f} {r['elapsed']:>9.0f}s")

    # ===================================================================
    # Plots
    # ===================================================================
    print("\n--- Generating plots ---")

    model_colors = {
        "Ridge": "#42A5F5", "XGBoost": "#FFA726", "XGB_Clf": "#EC407A",
        "LightGBM": "#FFCA28", "MLP": "#EF5350",
    }

    # 1. IC by frequency and model (grouped bar chart)
    ok_freqs = [r["freq"] for r in all_results if r.get("status") == "ok"]
    models_tested = cross_df["model"].unique()

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(ok_freqs))
    width = 0.15
    for i, model in enumerate(models_tested):
        ics = []
        for freq in ok_freqs:
            sub = cross_df[(cross_df["freq"] == freq) & (cross_df["model"] == model)]
            ics.append(sub["ic_mean"].values[0] if len(sub) > 0 else 0)
        offset = (i - len(models_tested) / 2 + 0.5) * width
        ax.bar(x + offset, ics, width, label=model,
               color=model_colors.get(model, "#9E9E9E"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(ok_freqs, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean OOS Spearman IC", fontsize=11)
    ax.set_xlabel("Bar Frequency", fontsize=11)
    ax.set_title(f"ML Signal Quality Across Frequencies ({FWD_DAYS}d Forward Return)", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "ic_by_frequency.png")
    plt.close(fig)
    print("  [1/3] IC by frequency")

    # 2. Best IC per frequency (line chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    best_ics = []
    best_models = []
    for freq in ok_freqs:
        sub = cross_df[cross_df["freq"] == freq].sort_values("ic_mean", ascending=False)
        best_ics.append(sub.iloc[0]["ic_mean"])
        best_models.append(sub.iloc[0]["model"])

    bars = ax.bar(range(len(ok_freqs)), best_ics,
                  color=[model_colors.get(m, "#9E9E9E") for m in best_models],
                  alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(ok_freqs)))
    ax.set_xticklabels(ok_freqs, fontsize=11)
    for i, (ic, model) in enumerate(zip(best_ics, best_models)):
        ax.text(i, ic + 0.002, f"{model}\n{ic:+.4f}", ha="center", fontsize=8, va="bottom")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Best Mean OOS IC", fontsize=11)
    ax.set_xlabel("Bar Frequency", fontsize=11)
    ax.set_title("Best ML Signal by Frequency", fontsize=13)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "best_ic_by_frequency.png")
    plt.close(fig)
    print("  [2/3] Best IC by frequency")

    # 3. t-stat heatmap (model x frequency)
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = np.full((len(models_tested), len(ok_freqs)), np.nan)
    for i, model in enumerate(models_tested):
        for j, freq in enumerate(ok_freqs):
            sub = cross_df[(cross_df["freq"] == freq) & (cross_df["model"] == model)]
            if len(sub) > 0:
                heatmap[i, j] = sub.iloc[0]["ic_t"]

    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=3)
    ax.set_xticks(range(len(ok_freqs)))
    ax.set_xticklabels(ok_freqs, fontsize=10)
    ax.set_yticks(range(len(models_tested)))
    ax.set_yticklabels(models_tested, fontsize=10)
    for i in range(len(models_tested)):
        for j in range(len(ok_freqs)):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                        fontweight="bold" if abs(val) >= 2 else "normal")
    plt.colorbar(im, ax=ax, label="t-statistic (IC)")
    ax.set_title("IC t-Statistic: Model x Frequency", fontsize=13)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "tstat_heatmap.png")
    plt.close(fig)
    print("  [3/3] t-stat heatmap")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("MULTI-FREQUENCY ML SHOOTOUT — SUMMARY")
    print("=" * 70)
    for r in all_results:
        if r.get("status") != "ok":
            print(f"  {r['freq']}: {r.get('status', 'unknown')} ({r.get('reason', '')})")
            continue
        best = r["eval_df"].iloc[0]
        print(f"  {r['freq']}: best={best['model']} IC={best['ic']:+.4f} "
              f"t={best['ic_t']:.1f} ({r['n_obs']:,} obs, {r['elapsed']:.0f}s)")

    total_time = sum(r.get("elapsed", 0) for r in all_results if r.get("status") == "ok")
    print(f"\nTotal elapsed: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
