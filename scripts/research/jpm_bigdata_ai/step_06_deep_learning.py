"""
Step 6 — Deep Learning: LSTM & Neural Networks
=================================================
JPM Big Data & AI Strategies: Crypto Recreation
Kolanovic & Krishnamachari (2017)

This script applies deep learning methods from the paper (p102-116):
  1. Feedforward Neural Network (MLP) — the simplest deep model.
  2. LSTM — exploits sequential/temporal structure in features.
  3. Both trained with walk-forward validation, early stopping.
  4. Compares against Ridge baseline on the same data/splits.
  5. Produces diagnostic plots and summary tables.

The paper argues deep learning can capture complex non-linear temporal
patterns that simpler models miss. We test this claim on crypto.

Architecture choices (conservative to avoid overfitting):
  - MLP: 2 hidden layers (64, 32), dropout 0.3, batch norm
  - LSTM: 1 LSTM layer (64 hidden), 1 linear head, dropout 0.2
  - Both: Adam optimizer, lr=1e-3, early stopping patience=10
  - Sequence length for LSTM: 21 days (1 month of history)
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sp_stats
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

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "jpm_bigdata_ai" / "step_06"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fwd_5d"
ALL_TARGETS = {"fwd_1d": 1, "fwd_5d": 5, "fwd_21d": 21}

TRAIN_DAYS = 365 * 2
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRAIN_DAYS = 365

SEQ_LEN = 21       # LSTM lookback window
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ===================================================================
# Model definitions
# ===================================================================
class MLP(nn.Module):
    """Simple 2-layer feedforward network with dropout and layer norm."""
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


class LSTMModel(nn.Module):
    """LSTM with a single recurrent layer and linear head."""
    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, n_features)
        output, (h_n, _) = self.lstm(x)
        last_hidden = h_n.squeeze(0)  # (batch, hidden_size)
        return self.head(self.dropout(last_hidden)).squeeze(-1)


# ===================================================================
# Training utilities
# ===================================================================
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> tuple[nn.Module, list[float], list[float]]:
    """Train with early stopping on validation loss."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.FloatTensor(y_val).to(DEVICE)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    n = len(X_t)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_losses


def build_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Build overlapping sequences for LSTM from a 2D array.

    Input: (n_samples, n_features)
    Output: (n_samples - seq_len + 1, seq_len, n_features)
    """
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]))
    seqs = np.array([X[i:i + seq_len] for i in range(n - seq_len + 1)])
    return seqs


# ===================================================================
# 1. Load & prepare data
# ===================================================================
print("=" * 70)
print("STEP 6: Deep Learning — LSTM & Neural Networks")
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

# ===================================================================
# 3. Walk-forward evaluation
# ===================================================================
print("\n--- Running walk-forward evaluation ---")

required_cols = feat_cols + [TARGET]
valid_mask = univ[required_cols].notna().all(axis=1)
data_clean = univ.loc[valid_mask].copy()

all_preds = []
loss_histories = {"MLP": [], "LSTM": []}
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
    if len(train) < 200 or len(test) < 10:
        continue

    X_train_raw = train[feat_cols].values
    y_train_all = train[TARGET].values
    X_test_raw = test[feat_cols].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s = np.clip(scaler.transform(X_test_raw), -5, 5)

    # Split train into train/val (last 20% for validation)
    val_size = max(int(len(X_train_s) * 0.2), 50)
    X_tr, X_val = X_train_s[:-val_size], X_train_s[-val_size:]
    y_tr, y_val = y_train_all[:-val_size], y_train_all[-val_size:]

    fold = fold_info["fold"]

    # --- Ridge baseline ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train_all)
    pred_ridge = ridge.predict(X_test_s)

    pred_df_ridge = test[["ts", "symbol", TARGET]].copy()
    pred_df_ridge["y_pred"] = pred_ridge
    pred_df_ridge["model"] = "Ridge"
    pred_df_ridge["fold"] = fold
    all_preds.append(pred_df_ridge)

    # --- MLP ---
    mlp = MLP(n_features=len(feat_cols))
    mlp, tl, vl = train_model(mlp, X_tr, y_tr, X_val, y_val)
    loss_histories["MLP"].append({"fold": fold, "train": tl, "val": vl})

    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_s).to(DEVICE)
        pred_mlp = mlp(X_test_t).cpu().numpy()

    pred_df_mlp = test[["ts", "symbol", TARGET]].copy()
    pred_df_mlp["y_pred"] = pred_mlp
    pred_df_mlp["model"] = "MLP"
    pred_df_mlp["fold"] = fold
    all_preds.append(pred_df_mlp)

    # --- LSTM ---
    # For LSTM, we need to build sequences per symbol within the fold
    # Simpler approach: treat the pooled panel as sequences sorted by (symbol, ts)
    # and build sequences per symbol
    lstm_test_preds = np.full(len(test), np.nan)

    # Get unique symbols in test
    test_symbols = test["symbol"].unique()
    lstm_model = LSTMModel(n_features=len(feat_cols))

    # Build training sequences across all symbols
    train_seqs_X = []
    train_seqs_y = []
    for sym in train["symbol"].unique():
        sym_train = train[train["symbol"] == sym].sort_values("ts")
        if len(sym_train) < SEQ_LEN + 1:
            continue
        X_sym = scaler.transform(sym_train[feat_cols].values)
        X_sym = np.clip(X_sym, -5, 5)
        y_sym = sym_train[TARGET].values

        seqs = build_sequences(X_sym, SEQ_LEN)
        targets = y_sym[SEQ_LEN - 1:]
        valid = ~np.isnan(targets)
        train_seqs_X.append(seqs[valid])
        train_seqs_y.append(targets[valid])

    if train_seqs_X:
        X_lstm_train = np.concatenate(train_seqs_X, axis=0)
        y_lstm_train = np.concatenate(train_seqs_y, axis=0)

        # Split for validation
        n_lstm = len(X_lstm_train)
        val_n = max(int(n_lstm * 0.2), 50)
        X_lstm_tr = X_lstm_train[:-val_n]
        y_lstm_tr = y_lstm_train[:-val_n]
        X_lstm_val = X_lstm_train[-val_n:]
        y_lstm_val = y_lstm_train[-val_n:]

        if len(X_lstm_tr) > 100:
            lstm_model, tl_l, vl_l = train_model(
                lstm_model, X_lstm_tr, y_lstm_tr, X_lstm_val, y_lstm_val,
            )
            loss_histories["LSTM"].append({"fold": fold, "train": tl_l, "val": vl_l})

            # Predict on test: build sequences per symbol
            test_idx_list = []
            test_seq_list = []

            for sym in test_symbols:
                # Need SEQ_LEN-1 days of history before test starts
                sym_all = data_clean[data_clean["symbol"] == sym].sort_values("ts")
                sym_test_mask = (
                    (sym_all["ts"] >= fold_info["test_start"])
                    & (sym_all["ts"] <= fold_info["test_end"])
                )
                sym_test = sym_all.loc[sym_test_mask]
                if len(sym_test) == 0:
                    continue

                # Get history for context
                first_test_ts = sym_test["ts"].iloc[0]
                sym_before = sym_all[sym_all["ts"] < first_test_ts].tail(SEQ_LEN - 1)
                sym_context = pd.concat([sym_before, sym_test])

                X_ctx = scaler.transform(sym_context[feat_cols].values)
                X_ctx = np.clip(X_ctx, -5, 5)
                seqs = build_sequences(X_ctx, SEQ_LEN)

                # Align: sequences correspond to the last day of each window
                n_before = len(sym_before)
                n_test_seqs = min(len(seqs) - n_before + SEQ_LEN - 1, len(sym_test))
                if n_test_seqs <= 0:
                    continue

                # The first test prediction uses seq starting from sym_before
                test_seqs = seqs[n_before:][:len(sym_test)]
                if len(test_seqs) > 0:
                    test_seq_list.append(test_seqs)
                    # Map back to test dataframe indices
                    sym_test_idx = sym_test.index[:len(test_seqs)]
                    test_idx_list.extend(sym_test_idx.tolist())

            if test_seq_list:
                X_lstm_test = np.concatenate(test_seq_list, axis=0)
                with torch.no_grad():
                    X_lt = torch.FloatTensor(X_lstm_test).to(DEVICE)
                    # Process in batches to avoid OOM
                    preds_lstm = []
                    for b_start in range(0, len(X_lt), BATCH_SIZE):
                        batch = X_lt[b_start:b_start + BATCH_SIZE]
                        preds_lstm.append(lstm_model(batch).cpu().numpy())
                    lstm_pred_arr = np.concatenate(preds_lstm)

                # Map predictions back
                pred_df_lstm = test.loc[test_idx_list, ["ts", "symbol", TARGET]].copy()
                pred_df_lstm["y_pred"] = lstm_pred_arr[:len(test_idx_list)]
                pred_df_lstm["model"] = "LSTM"
                pred_df_lstm["fold"] = fold
                all_preds.append(pred_df_lstm)

    if (si + 1) % 5 == 0 or si == n_splits - 1:
        n_mlp = len(pred_df_mlp)
        n_lstm = len(pred_df_lstm) if "pred_df_lstm" in dir() and pred_df_lstm is not None else 0
        print(f"    fold {si+1}/{n_splits} (MLP={n_mlp}, LSTM={n_lstm})")

preds_df = pd.concat(all_preds, ignore_index=True)
print(f"\nTotal predictions: {len(preds_df):,}")
for m in preds_df["model"].unique():
    print(f"  {m}: {(preds_df['model'] == m).sum():,}")

# ===================================================================
# 4. Evaluate predictions
# ===================================================================
print("\n--- Out-of-sample evaluation ---")

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
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))

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
        "rmse": rmse,
        "n_obs": n,
        "ic_mean_fold": float(np.mean(fold_ics)) if fold_ics else np.nan,
        "ic_std_fold": float(np.std(fold_ics)) if fold_ics else np.nan,
        "ic_hit_rate": float(np.mean([x > 0 for x in fold_ics])) if fold_ics else np.nan,
    })

eval_df = pd.DataFrame(eval_records)
eval_df.to_csv(ARTIFACT_DIR / "deep_learning_evaluation.csv", index=False, float_format="%.6f")

print("\n  5d horizon results:")
for _, row in eval_df.iterrows():
    print(f"    {row['model']:<8s}  IC={row['ic']:+.4f}  Hit={row['hit_rate']:.1%}  "
          f"RMSE={row['rmse']:.4f}  "
          f"Fold IC={row['ic_mean_fold']:+.4f}±{row['ic_std_fold']:.4f}  "
          f"IC>0: {row['ic_hit_rate']:.0%}")

# ===================================================================
# 5. Cross-step comparison
# ===================================================================
print("\n--- Cross-step comparison (5d) ---")
comparison_rows = []

# Load prior results
for step_file, step_type in [
    ("step_02/linear_models_evaluation.csv", "Linear"),
    ("step_03/tree_models_evaluation.csv", "Tree"),
    ("step_04/classification_evaluation.csv", "Classification"),
]:
    fpath = ARTIFACT_DIR.parent / step_file
    if fpath.exists():
        df = pd.read_csv(fpath)
        df_5d = df[df["horizon"] == "5d"] if "horizon" in df.columns else df
        for _, row in df_5d.iterrows():
            comparison_rows.append({
                "model": row["model"],
                "type": step_type,
                "ic": row["ic"],
            })

for _, row in eval_df.iterrows():
    comparison_rows.append({
        "model": row["model"],
        "type": "Deep Learning" if row["model"] != "Ridge" else "Linear (baseline)",
        "ic": row["ic"],
    })

comp_df = pd.DataFrame(comparison_rows).sort_values("ic", ascending=False)
print("\n  Full leaderboard (5d):")
for _, row in comp_df.head(15).iterrows():
    print(f"    {row['type']:<20s} {row['model']:<14s}  IC={row['ic']:+.4f}")
comp_df.to_csv(ARTIFACT_DIR / "full_leaderboard.csv", index=False, float_format="%.6f")

# ===================================================================
# 6. Per-fold IC time series
# ===================================================================
fold_ic_records = []
for model_name in preds_df["model"].unique():
    model_preds = preds_df[preds_df["model"] == model_name]
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
        })
fold_ic_df = pd.DataFrame(fold_ic_records)
fold_ic_df.to_csv(ARTIFACT_DIR / "fold_ic_timeseries.csv", index=False, float_format="%.6f")

# ===================================================================
# 7. Backtest
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

model_colors = {"Ridge": "#9E9E9E", "MLP": "#1976D2", "LSTM": "#F57C00"}

# --- 8a. Model comparison ---
print("  [1/5] Model comparison")
fig, ax = plt.subplots(figsize=(8, 5))
for i, (_, row) in enumerate(eval_df.iterrows()):
    ax.bar(i, row["ic"], color=model_colors.get(row["model"], "gray"), alpha=0.85)
ax.set_xticks(range(len(eval_df)))
ax.set_xticklabels(eval_df["model"].values)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("OOS Spearman IC (5d)", fontsize=11)
ax.set_title("Deep Learning vs Ridge Baseline", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "model_comparison.png")
plt.close(fig)

# --- 8b. Fold IC time series ---
print("  [2/5] Fold IC time series")
fig, ax = plt.subplots(figsize=(14, 6))
for model_name in eval_df["model"]:
    sub = fold_ic_df[fold_ic_df["model"] == model_name].sort_values("date")
    if len(sub) > 0:
        ax.plot(sub["date"], sub["ic"], marker="o", markersize=3,
                label=model_name, color=model_colors.get(model_name, "gray"), alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Spearman IC (per fold)", fontsize=11)
ax.set_title("Walk-Forward IC — Deep Learning (5d)", fontsize=13)
ax.legend()
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "fold_ic_timeseries.png")
plt.close(fig)

# --- 8c. Training loss curves (sample folds) ---
print("  [3/5] Training loss curves")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, model_name in zip(axes, ["MLP", "LSTM"]):
    histories = loss_histories.get(model_name, [])
    if not histories:
        continue
    # Plot last 3 folds
    for h in histories[-3:]:
        ax.plot(h["train"], alpha=0.5, label=f"fold {h['fold']} train")
        ax.plot(h["val"], alpha=0.8, linestyle="--", label=f"fold {h['fold']} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name} — Training Curves (last 3 folds)")
    ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "training_loss_curves.png")
plt.close(fig)

# --- 8d. Full leaderboard ---
print("  [4/5] Full leaderboard chart")
fig, ax = plt.subplots(figsize=(12, 8))
top15 = comp_df.head(15).sort_values("ic", ascending=True)
type_colors = {
    "Linear": "#90CAF9", "Linear (baseline)": "#90CAF9",
    "Tree": "#A5D6A7", "Classification": "#CE93D8",
    "Deep Learning": "#FF8A65",
}
colors = [type_colors.get(row["type"], "#E0E0E0") for _, row in top15.iterrows()]
ax.barh(range(len(top15)),
        top15["ic"].values, color=colors, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top15)))
labels = [f"{row['model']} ({row['type'][:5]})" for _, row in top15.iterrows()]
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("OOS Spearman IC (5d)", fontsize=11)
ax.set_title("Full ML Leaderboard After Step 6", fontsize=13)
fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "full_leaderboard.png")
plt.close(fig)

# --- 8e. Backtest equity ---
print("  [5/5] Backtest equity curve")
if bt_metrics_list:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bt["ts"], bt["portfolio_equity"],
            label=f"{best_model} Top-Q L/O",
            color=model_colors.get(best_model, "#1976D2"), linewidth=1.5)
    btc_norm = btc_c / btc_c.iloc[0]
    ax.plot(btc_norm.index, btc_norm.values,
            label="BTC Buy & Hold", color="#FF9800", linewidth=1.0, alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log)", fontsize=11)
    ax.set_title(f"Deep Learning Backtest — {best_model}", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "backtest_equity_curve.png")
    plt.close(fig)

# ===================================================================
# 9. Summary
# ===================================================================
print("\n" + "=" * 70)
print("STEP 6 RESULTS SUMMARY")
print("=" * 70)

print(f"\nModels: MLP (2-layer, 64→32), LSTM ({SEQ_LEN}-day lookback, 64 hidden)")
print(f"Training: Adam lr={LR}, early stopping patience={PATIENCE}, max {MAX_EPOCHS} epochs")
print(f"Walk-forward: {len(splits)} folds")

print("\n--- Deep learning vs Ridge (5d horizon) ---")
for _, row in eval_df.iterrows():
    print(f"  {row['model']:<8s}  IC={row['ic']:+.4f}  Hit={row['hit_rate']:.1%}  "
          f"RMSE={row['rmse']:.4f}  IC>0: {row['ic_hit_rate']:.0%} of folds")

print(f"\n--- Top 5 overall (5d) ---")
for _, row in comp_df.head(5).iterrows():
    print(f"  {row['type']:<20s} {row['model']:<14s}  IC={row['ic']:+.4f}")

if bt_metrics_list:
    print(f"\n--- {best_model} backtest ---")
    print(format_metrics_table(bt_metrics_list))

print(f"\nArtifacts saved to: {ARTIFACT_DIR}")
print("Done.")
