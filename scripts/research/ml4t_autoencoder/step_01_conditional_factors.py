"""
Study 3 — Autoencoder Conditional Risk Factors for Crypto
==========================================================
Replicates & extends Chapter 20 of Jansen (2020)
"ML for Algorithmic Trading, 2nd Ed."
Based on Gu, Kelly & Xiu (2019) "Autoencoder Asset Pricing Models"

Objective: Replace hand-crafted TA-Lib features with learned latent
factors from an autoencoder. Test whether learned representations
improve return prediction IC and portfolio Sharpe.

Architecture:
  - Conditional Autoencoder (CA): characteristics → latent factors
  - Encoder: maps N asset characteristics to K latent factors
  - Decoder: maps K latent factors to predicted returns
  - The latent factors are *conditional* on asset characteristics

Steps:
  1. Load daily panel with TA-Lib features (existing compute_features)
  2. Train conditional autoencoder via walk-forward
  3. Extract latent factors and evaluate IC
  4. Compare CA predictions vs XGBoost baseline
  5. Portfolio construction from CA signals

Reference: Jansen (2020) Ch. 20 — Autoencoders for Conditional Risk Factors
Also: Gu, Kelly & Xiu (2019), Kelly, Pruitt & Su (2019) "IPCA"
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

from scripts.research.common.data import (
    ANN_FACTOR,
    compute_btc_benchmark,
    filter_universe,
    load_daily_bars,
)
from scripts.research.common.backtest import simple_backtest
from scripts.research.common.metrics import compute_metrics, format_metrics_table
from scripts.research.common.risk_overlays import (
    apply_position_limit_wide,
    apply_vol_targeting,
)
from scripts.research.jpm_bigdata_ai.helpers import (
    FEATURE_COLS,
    compute_features,
    walk_forward_splits,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START = "2018-01-01"
END = "2025-12-31"
DATA_START = "2016-06-01"
MIN_ADV = 1_000_000
MIN_HISTORY = 90

# AE architecture
LATENT_DIM = 8          # number of learned latent factors
HIDDEN_DIM = 64         # hidden layer width
DROPOUT = 0.3
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 512

# Walk-forward
TARGET = "fwd_5d"
TRAIN_DAYS = 365 * 2
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRAIN_DAYS = 365

# Portfolio
VOL_TARGET = 0.20
MAX_WEIGHT = 0.15
COST_BPS = 20.0
REBAL_FREQ = 5

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "ml4t_autoencoder"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ===================================================================
# 1. Conditional Autoencoder
# ===================================================================
class ConditionalAutoencoder(nn.Module):
    """Conditional Autoencoder for asset pricing.

    Encoder: characteristics (N features) → latent factors (K)
    Decoder: latent factors (K) → predicted return (1)

    The encoder maps each asset's characteristics to a set of
    latent factor loadings. The decoder uses these loadings
    with shared latent factor returns to predict asset returns.
    """

    def __init__(self, n_features: int, latent_dim: int = LATENT_DIM,
                 hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()

        # Encoder: characteristics → factor loadings
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        # Decoder: factor loadings → predicted return
        # Shared factor returns (learned parameters)
        self.factor_returns = nn.Linear(latent_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_return, latent_factors)."""
        latent = self.encoder(x)
        pred_ret = self.factor_returns(latent).squeeze(-1)
        return pred_ret, latent

    def get_factors(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent factors without prediction."""
        with torch.no_grad():
            return self.encoder(x)


def train_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> ConditionalAutoencoder:
    """Train the conditional autoencoder."""
    model = ConditionalAutoencoder(n_features).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred, latent = model(xb)
            # MSE loss on return prediction
            loss = nn.functional.mse_loss(pred, yb)
            # L2 regularization on latent factors (encourage sparsity)
            loss += 1e-4 * latent.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

    model.eval()
    return model


def predict_autoencoder(
    model: ConditionalAutoencoder,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and latent factors from trained model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        pred, latent = model(X_t)
    return pred.cpu().numpy(), latent.cpu().numpy()


# ===================================================================
# 2. Walk-forward evaluation
# ===================================================================
def run_walk_forward(
    data: pd.DataFrame,
    feat_cols: list[str],
    target_col: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward evaluation for both CA and XGBoost baseline.

    Returns (ca_preds, xgb_preds) DataFrames.
    """
    unique_dates = np.sort(data["ts"].unique())
    splits = walk_forward_splits(
        unique_dates, train_days=TRAIN_DAYS, test_days=TEST_DAYS,
        step_days=STEP_DAYS, min_train_days=MIN_TRAIN_DAYS,
    )
    print(f"  Walk-forward splits: {len(splits)}")

    ca_all = []
    xgb_all = []

    for si, fold_info in enumerate(splits):
        train_mask = (data["ts"] >= fold_info["train_start"]) & (data["ts"] <= fold_info["train_end"])
        test_mask = (data["ts"] >= fold_info["test_start"]) & (data["ts"] <= fold_info["test_end"])
        train = data.loc[train_mask]
        test = data.loc[test_mask]

        if len(train) < 500 or len(test) < 50:
            continue

        # Scale features
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[feat_cols].values)
        X_te = np.clip(scaler.transform(test[feat_cols].values), -5, 5)
        y_tr = train[target_col].values
        y_te = test[target_col].values

        # --- Conditional Autoencoder ---
        try:
            model = train_autoencoder(X_tr, y_tr, n_features=len(feat_cols))
            ca_pred, ca_latent = predict_autoencoder(model, X_te)

            pred_df = test[["ts", "symbol", target_col]].copy()
            pred_df["ca_pred"] = ca_pred
            pred_df["ca_signal"] = ca_pred  # raw prediction is the signal
            for k in range(LATENT_DIM):
                pred_df[f"factor_{k}"] = ca_latent[:, k]
            pred_df["fold"] = fold_info["fold"]
            ca_all.append(pred_df)
        except Exception as e:
            print(f"    fold {si}: CA error — {e}")

        # --- XGBoost baseline ---
        if xgb is not None:
            try:
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="logloss",
                    n_jobs=1, verbosity=0, tree_method="hist",
                )
                y_cls = (y_tr > 0).astype(int)
                xgb_clf.fit(X_tr, y_cls)
                prob = xgb_clf.predict_proba(X_te)[:, 1]

                xgb_df = test[["ts", "symbol", target_col]].copy()
                xgb_df["xgb_pred"] = prob
                xgb_df["xgb_signal"] = prob - 0.5
                xgb_df["fold"] = fold_info["fold"]
                xgb_all.append(xgb_df)
            except Exception:
                pass

        if (si + 1) % 5 == 0 or si == len(splits) - 1:
            print(f"    fold {si+1}/{len(splits)}", flush=True)

    ca_preds = pd.concat(ca_all, ignore_index=True) if ca_all else pd.DataFrame()
    xgb_preds = pd.concat(xgb_all, ignore_index=True) if xgb_all else pd.DataFrame()

    return ca_preds, xgb_preds


# ===================================================================
# 3. Portfolio construction
# ===================================================================
def build_portfolio_from_signal(
    preds: pd.DataFrame,
    returns_wide: pd.DataFrame,
    signal_col: str,
    label: str,
) -> dict:
    """Build signal-proportional IVW portfolio."""
    sub = preds[preds[signal_col].notna()].copy()

    # Top-quintile selection
    sub["rank_pct"] = sub.groupby("ts")[signal_col].rank(pct=True)
    top = sub[sub["rank_pct"] >= 0.80].copy()
    if len(top) == 0:
        return {"label": label, "sharpe": np.nan}

    counts = top.groupby("ts")["symbol"].transform("count")
    top["weight"] = 1.0 / counts
    wts = top.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)

    # Position limits + vol targeting
    wts = apply_position_limit_wide(wts, MAX_WEIGHT)
    wts = apply_vol_targeting(wts, returns_wide, vol_target=VOL_TARGET)

    bt = simple_backtest(wts, returns_wide, cost_bps=COST_BPS)
    eq = pd.Series(bt["portfolio_equity"].values, index=bt["ts"])
    m = compute_metrics(eq)
    m["label"] = label
    m["avg_turnover"] = float(bt["turnover"].mean())
    m["avg_exposure"] = float(bt["gross_exposure"].mean())
    m["equity"] = eq
    return m


# ===================================================================
# Main
# ===================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("STUDY 3: AUTOENCODER CONDITIONAL RISK FACTORS")
    print("Replicating Jansen (2020) Ch. 20 — Gu, Kelly & Xiu (2019)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Latent dim: {LATENT_DIM}, Hidden: {HIDDEN_DIM}, Epochs: {EPOCHS}")

    # ------------------------------------------------------------------
    # Part 1: Load and prepare data
    # ------------------------------------------------------------------
    print("\n--- Part 1: Data Preparation ---")
    panel = load_daily_bars(start=DATA_START, end=END)
    panel = filter_universe(panel, min_adv_usd=MIN_ADV, min_history_days=MIN_HISTORY)
    panel = compute_features(panel)
    feat_cols = list(FEATURE_COLS)
    print(f"  Features: {len(feat_cols)}")

    # Add forward return target
    def _add_fwd(g):
        g = g.copy()
        g[TARGET] = g["close"].shift(-5) / g["close"] - 1.0
        return g

    panel = panel.groupby("symbol", group_keys=False).apply(_add_fwd)

    # Add realized vol for IVW
    panel["realized_vol"] = panel.groupby("symbol")["close"].transform(
        lambda x: x.pct_change().rolling(42, min_periods=21).std() * np.sqrt(ANN_FACTOR)
    )

    univ = panel[panel["in_universe"]].copy()
    required_cols = feat_cols + [TARGET]
    valid_mask = univ[required_cols].notna().all(axis=1)
    data = univ.loc[valid_mask].copy()

    print(f"  In-universe: {len(data):,} rows, {data['symbol'].nunique()} symbols")

    # Returns matrix for backtest
    ret_panel = univ[["ts", "symbol", "close"]].copy()
    ret_panel["ret"] = ret_panel.groupby("symbol")["close"].pct_change()
    returns_wide = ret_panel.pivot(index="ts", columns="symbol", values="ret").fillna(0.0)

    # ------------------------------------------------------------------
    # Part 2: Walk-forward evaluation
    # ------------------------------------------------------------------
    print("\n--- Part 2: Walk-Forward Training ---")
    print(f"  Models: Conditional Autoencoder (K={LATENT_DIM}) vs XGBoost Classifier")

    ca_preds, xgb_preds = run_walk_forward(data, feat_cols)

    # Save predictions
    if len(ca_preds) > 0:
        ca_preds.to_parquet(ARTIFACT_DIR / "ca_predictions.parquet", index=False)
    if len(xgb_preds) > 0:
        xgb_preds.to_parquet(ARTIFACT_DIR / "xgb_predictions.parquet", index=False)

    # ------------------------------------------------------------------
    # Part 3: IC evaluation
    # ------------------------------------------------------------------
    print("\n--- Part 3: Information Coefficient ---")

    ic_results = []

    if len(ca_preds) > 0:
        mask = ca_preds[TARGET].notna()
        ca_ic = float(sp_stats.spearmanr(
            ca_preds.loc[mask, TARGET], ca_preds.loc[mask, "ca_pred"]
        ).statistic)

        # IC by fold
        ca_ics = ca_preds.groupby("fold").apply(
            lambda g: sp_stats.spearmanr(
                g[TARGET].dropna(),
                g.loc[g[TARGET].notna(), "ca_pred"]
            ).statistic if len(g[TARGET].dropna()) > 10 else np.nan
        )
        ic_results.append({
            "model": "Conditional AE",
            "ic_mean": ca_ic,
            "ic_std": float(ca_ics.std()),
            "ic_t": float(ca_ic / (ca_ics.std() / np.sqrt(len(ca_ics)))) if ca_ics.std() > 0 else 0,
            "n_obs": int(mask.sum()),
        })
        print(f"  CA:  IC={ca_ic:+.4f} ± {ca_ics.std():.4f} (t={ic_results[-1]['ic_t']:.1f})")

        # IC per latent factor
        print(f"\n  Latent factor ICs:")
        for k in range(LATENT_DIM):
            fk = f"factor_{k}"
            if fk in ca_preds.columns:
                fic = float(sp_stats.spearmanr(
                    ca_preds.loc[mask, TARGET], ca_preds.loc[mask, fk]
                ).statistic)
                print(f"    Factor {k}: IC={fic:+.4f}")

    if len(xgb_preds) > 0:
        mask = xgb_preds[TARGET].notna()
        xgb_ic = float(sp_stats.spearmanr(
            xgb_preds.loc[mask, TARGET], xgb_preds.loc[mask, "xgb_pred"]
        ).statistic)

        xgb_ics = xgb_preds.groupby("fold").apply(
            lambda g: sp_stats.spearmanr(
                g[TARGET].dropna(),
                g.loc[g[TARGET].notna(), "xgb_pred"]
            ).statistic if len(g[TARGET].dropna()) > 10 else np.nan
        )
        ic_results.append({
            "model": "XGBoost Clf",
            "ic_mean": xgb_ic,
            "ic_std": float(xgb_ics.std()),
            "ic_t": float(xgb_ic / (xgb_ics.std() / np.sqrt(len(xgb_ics)))) if xgb_ics.std() > 0 else 0,
            "n_obs": int(mask.sum()),
        })
        print(f"  XGB: IC={xgb_ic:+.4f} ± {xgb_ics.std():.4f} (t={ic_results[-1]['ic_t']:.1f})")

    if ic_results:
        ic_df = pd.DataFrame(ic_results)
        ic_df.to_csv(ARTIFACT_DIR / "ic_comparison.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------------------
    # Part 4: Portfolio construction
    # ------------------------------------------------------------------
    print("\n--- Part 4: Portfolio Construction ---")

    portfolio_metrics = []

    if len(ca_preds) > 0:
        print("  Building CA portfolio ...")
        m_ca = build_portfolio_from_signal(ca_preds, returns_wide, "ca_signal", "CA Top-Q + VT")
        portfolio_metrics.append(m_ca)
        print(f"    CA: Sharpe={m_ca.get('sharpe',0):.2f} CAGR={m_ca.get('cagr',0):.1%}")

    if len(xgb_preds) > 0:
        print("  Building XGBoost portfolio ...")
        m_xgb = build_portfolio_from_signal(xgb_preds, returns_wide, "xgb_signal", "XGB Top-Q + VT")
        portfolio_metrics.append(m_xgb)
        print(f"    XGB: Sharpe={m_xgb.get('sharpe',0):.2f} CAGR={m_xgb.get('cagr',0):.1%}")

    # Ensemble: average CA + XGB signals
    if len(ca_preds) > 0 and len(xgb_preds) > 0:
        print("  Building Ensemble (CA + XGB) ...")
        merged = ca_preds[["ts", "symbol", TARGET, "ca_signal"]].merge(
            xgb_preds[["ts", "symbol", "xgb_signal"]],
            on=["ts", "symbol"], how="inner"
        )
        # Normalize both signals to [0,1] before averaging
        for col in ["ca_signal", "xgb_signal"]:
            mn = merged.groupby("ts")[col].transform("min")
            mx = merged.groupby("ts")[col].transform("max")
            rng = (mx - mn).clip(lower=1e-8)
            merged[f"{col}_norm"] = (merged[col] - mn) / rng

        merged["ensemble_signal"] = 0.5 * merged["ca_signal_norm"] + 0.5 * merged["xgb_signal_norm"]

        m_ens = build_portfolio_from_signal(merged, returns_wide, "ensemble_signal", "Ensemble (CA+XGB) + VT")
        portfolio_metrics.append(m_ens)
        print(f"    Ensemble: Sharpe={m_ens.get('sharpe',0):.2f} CAGR={m_ens.get('cagr',0):.1%}")

    # BTC benchmark
    btc_eq = compute_btc_benchmark(panel)
    ref_eq = portfolio_metrics[0]["equity"] if portfolio_metrics and "equity" in portfolio_metrics[0] else None
    if ref_eq is not None and len(btc_eq) > 0:
        btc_c = btc_eq.reindex(ref_eq.index).ffill().bfill()
        btc_c = btc_c / btc_c.iloc[0]
        m_btc = compute_metrics(btc_c)
        m_btc["label"] = "BTC Buy & Hold"
        portfolio_metrics.append(m_btc)

    # ------------------------------------------------------------------
    # Part 5: Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: AUTOENCODER vs XGBOOST")
    print("=" * 70)

    clean_metrics = [{k: v for k, v in m.items() if k != "equity"} for m in portfolio_metrics]
    if clean_metrics:
        print("\n" + format_metrics_table(clean_metrics))

    metrics_df = pd.DataFrame(clean_metrics)
    metrics_df.to_csv(ARTIFACT_DIR / "portfolio_metrics.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------------------
    # Part 6: Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")

    COLORS = ["#22c55e", "#3b82f6", "#8b5cf6", "#FFA726", "#ef4444"]

    # Plot 1: Equity curves
    fig, ax = plt.subplots(figsize=(16, 7))
    for i, m in enumerate(portfolio_metrics):
        if "equity" in m:
            ax.plot(m["equity"].index, m["equity"].values,
                    label=m["label"], color=COLORS[i % len(COLORS)],
                    linewidth=2.0 if "Ensemble" in m["label"] else 1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log)", fontsize=11)
    ax.set_title("Autoencoder vs XGBoost — Portfolio Equity", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "equity_curves.png")
    plt.close(fig)
    print("  [1/4] Equity curves")

    # Plot 2: IC comparison
    if ic_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        models = [r["model"] for r in ic_results]
        ics = [r["ic_mean"] for r in ic_results]
        stds = [r["ic_std"] for r in ic_results]
        ax.bar(range(len(models)), ics, yerr=stds,
               color=["#22c55e", "#3b82f6"][:len(models)],
               alpha=0.85, edgecolor="white", capsize=5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=11)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Information Coefficient (Spearman)", fontsize=11)
        ax.set_title("Signal Quality: Conditional AE vs XGBoost", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "ic_comparison.png")
        plt.close(fig)
        print("  [2/4] IC comparison")

    # Plot 3: Latent factor analysis
    if len(ca_preds) > 0:
        factor_cols = [c for c in ca_preds.columns if c.startswith("factor_")]
        if factor_cols:
            fig, axes = plt.subplots(2, 4, figsize=(18, 8))
            axes = axes.flatten()
            mask = ca_preds[TARGET].notna()
            for k, col in enumerate(factor_cols[:8]):
                ax = axes[k]
                ax.scatter(ca_preds.loc[mask, col], ca_preds.loc[mask, TARGET],
                           alpha=0.01, s=1, color="#3b82f6")
                ic = sp_stats.spearmanr(
                    ca_preds.loc[mask, col], ca_preds.loc[mask, TARGET]
                ).statistic
                ax.set_title(f"Factor {k} (IC={ic:+.3f})", fontsize=10)
                ax.set_xlabel("Factor loading", fontsize=8)
                ax.set_ylabel("5d return", fontsize=8)
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.axvline(0, color="gray", linewidth=0.5)
            fig.suptitle("Latent Factor Loadings vs Returns", fontsize=13, fontweight="bold")
            fig.tight_layout()
            fig.savefig(ARTIFACT_DIR / "latent_factors.png")
            plt.close(fig)
            print("  [3/4] Latent factors")

    # Plot 4: Rolling IC comparison
    if len(ca_preds) > 0 and len(xgb_preds) > 0:
        fig, ax = plt.subplots(figsize=(16, 5))

        for label, preds_df, sig_col, color in [
            ("Conditional AE", ca_preds, "ca_pred", "#22c55e"),
            ("XGBoost", xgb_preds, "xgb_pred", "#3b82f6"),
        ]:
            mask = preds_df[TARGET].notna()
            daily_ic = preds_df.loc[mask].groupby("ts").apply(
                lambda g: sp_stats.spearmanr(g[TARGET], g[sig_col]).statistic
                if len(g) > 5 else np.nan
            )
            rolling_ic = daily_ic.rolling(63, min_periods=21).mean()
            ax.plot(rolling_ic.index, rolling_ic.values, label=label,
                    color=color, linewidth=1.0)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Rolling 63d IC", fontsize=11)
        ax.set_title("Rolling IC: Conditional Autoencoder vs XGBoost", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "rolling_ic.png")
        plt.close(fig)
        print("  [4/4] Rolling IC")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("STUDY 3 SUMMARY — AUTOENCODER CONDITIONAL RISK FACTORS")
    print(f"{'='*70}")

    if ic_results:
        print("\n  Signal quality:")
        for r in ic_results:
            print(f"    {r['model']:<25s} IC={r['ic_mean']:+.4f} (t={r['ic_t']:.1f})")

    print("\n  Portfolio performance:")
    for m in portfolio_metrics:
        print(f"    {m.get('label',''):<30s} Sharpe={m.get('sharpe',0):.2f} "
              f"CAGR={m.get('cagr',0):.1%} MaxDD={m.get('max_dd',0):.1%}")

    if len(ic_results) >= 2:
        ca_ic = ic_results[0]["ic_mean"]
        xgb_ic = ic_results[1]["ic_mean"]
        winner = "Conditional AE" if ca_ic > xgb_ic else "XGBoost"
        print(f"\n  IC winner: {winner} ({max(ca_ic, xgb_ic):+.4f} vs {min(ca_ic, xgb_ic):+.4f})")

    print(f"\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
