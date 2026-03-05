"""
Logistic regression model training and walk-forward prediction.

Supports L2, L1, and Elastic-Net regularisation with optional Platt/isotonic
calibration.  Stores per-fold coefficients for interpretability and
stability diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from scripts.research.jpm_bigdata_ai.helpers import walk_forward_splits
except ModuleNotFoundError:
    from jpm_bigdata_ai.helpers import walk_forward_splits


@dataclass(frozen=True)
class ModelConfig:
    penalty: Literal["l2", "l1", "elasticnet"] = "l2"
    C: float = 1.0
    l1_ratio: float = 0.5  # only used for elasticnet
    class_weight: str | None = None  # None or "balanced"
    calibration: str | None = None  # None, "sigmoid" (Platt), or "isotonic"
    max_iter: int = 1000
    seed: int = 42


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 365 * 2
    test_days: int = 63
    step_days: int = 63
    min_train_days: int = 365


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    auc: float
    log_loss_val: float
    brier: float
    base_rate: float
    coefficients: dict[str, float]
    intercept: float


@dataclass
class ModelOutput:
    predictions: pd.DataFrame  # columns: ts, symbol, p_success
    fold_results: list[FoldResult]
    feature_names: list[str]

    def coefficient_summary(self) -> pd.DataFrame:
        """Mean and std of coefficients across folds."""
        rows = []
        for feat in self.feature_names:
            vals = [fr.coefficients.get(feat, np.nan) for fr in self.fold_results]
            rows.append({
                "feature": feat,
                "coef_mean": np.nanmean(vals),
                "coef_std": np.nanstd(vals),
                "coef_abs_mean": np.nanmean(np.abs(vals)),
                "sign_stability": np.nanmean(np.sign(vals)),
            })
        return pd.DataFrame(rows).sort_values("coef_abs_mean", ascending=False)

    def fold_metrics_df(self) -> pd.DataFrame:
        """Fold-level classification metrics."""
        return pd.DataFrame([
            {
                "fold": fr.fold,
                "train_start": fr.train_start,
                "train_end": fr.train_end,
                "test_start": fr.test_start,
                "test_end": fr.test_end,
                "n_train": fr.n_train,
                "n_test": fr.n_test,
                "auc": fr.auc,
                "log_loss": fr.log_loss_val,
                "brier": fr.brier,
                "base_rate": fr.base_rate,
            }
            for fr in self.fold_results
        ])


def _build_estimator(cfg: ModelConfig) -> LogisticRegression:
    solver = "saga" if cfg.penalty == "elasticnet" else "lbfgs"
    if cfg.penalty == "l1":
        solver = "saga"

    return LogisticRegression(
        penalty=cfg.penalty,
        C=cfg.C,
        l1_ratio=cfg.l1_ratio if cfg.penalty == "elasticnet" else None,
        solver=solver,
        class_weight=cfg.class_weight,
        max_iter=cfg.max_iter,
        random_state=cfg.seed,
        n_jobs=-1,
    )


def train_walk_forward(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    model_cfg: ModelConfig | None = None,
    wf_cfg: WalkForwardConfig | None = None,
) -> ModelOutput:
    """Train logistic regression in walk-forward fashion and produce OOS predictions.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format with columns: ts, symbol, <feature_cols>, <label_col>.
        Must be sorted by ts.
    feature_cols : list[str]
        Columns to use as features.
    label_col : str
        Column containing binary labels.
    model_cfg : ModelConfig
        Model hyperparameters.
    wf_cfg : WalkForwardConfig
        Walk-forward split parameters.

    Returns
    -------
    ModelOutput with OOS predictions and per-fold diagnostics.
    """
    if model_cfg is None:
        model_cfg = ModelConfig()
    if wf_cfg is None:
        wf_cfg = WalkForwardConfig()

    panel = panel.dropna(subset=[label_col])
    unique_dates = np.sort(panel["ts"].unique())

    splits = walk_forward_splits(
        unique_dates,
        train_days=wf_cfg.train_days,
        test_days=wf_cfg.test_days,
        step_days=wf_cfg.step_days,
        min_train_days=wf_cfg.min_train_days,
    )
    if not splits:
        raise ValueError(
            f"No walk-forward splits generated. Data has {len(unique_dates)} dates, "
            f"need at least {wf_cfg.min_train_days + wf_cfg.test_days}."
        )

    all_preds = []
    fold_results = []

    for sp in splits:
        train_mask = (panel["ts"] >= sp["train_start"]) & (panel["ts"] <= sp["train_end"])
        test_mask = (panel["ts"] >= sp["test_start"]) & (panel["ts"] <= sp["test_end"])
        train_data = panel.loc[train_mask].copy()
        test_data = panel.loc[test_mask].copy()

        train_valid = train_data.dropna(subset=feature_cols + [label_col])
        test_valid = test_data.dropna(subset=feature_cols)

        if len(train_valid) < 50 or len(test_valid) < 10:
            continue

        X_train = train_valid[feature_cols].values
        y_train = train_valid[label_col].values.astype(int)
        X_test = test_valid[feature_cols].values

        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        estimator = _build_estimator(model_cfg)
        estimator.fit(X_train_s, y_train)

        if model_cfg.calibration is not None:
            cal = CalibratedClassifierCV(
                estimator, method=model_cfg.calibration, cv=3,
            )
            cal.fit(X_train_s, y_train)
            probs = cal.predict_proba(X_test_s)[:, 1]
        else:
            probs = estimator.predict_proba(X_test_s)[:, 1]

        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)

        pred_df = test_valid[["ts", "symbol"]].copy()
        pred_df["p_success"] = probs
        all_preds.append(pred_df)

        y_test = test_valid[label_col].values.astype(int)
        has_both_classes = len(np.unique(y_test)) >= 2

        coefficients = dict(zip(feature_cols, estimator.coef_[0]))

        fold_results.append(FoldResult(
            fold=sp["fold"],
            train_start=sp["train_start"],
            train_end=sp["train_end"],
            test_start=sp["test_start"],
            test_end=sp["test_end"],
            n_train=len(train_valid),
            n_test=len(test_valid),
            auc=roc_auc_score(y_test, probs) if has_both_classes else np.nan,
            log_loss_val=log_loss(y_test, probs) if has_both_classes else np.nan,
            brier=brier_score_loss(y_test, probs) if has_both_classes else np.nan,
            base_rate=float(y_test.mean()),
            coefficients=coefficients,
            intercept=float(estimator.intercept_[0]),
        ))

    if not all_preds:
        raise ValueError("No valid walk-forward folds produced predictions.")

    predictions = pd.concat(all_preds, ignore_index=True)
    predictions = predictions.drop_duplicates(subset=["ts", "symbol"], keep="last")

    return ModelOutput(
        predictions=predictions,
        fold_results=fold_results,
        feature_names=feature_cols,
    )


def train_regime_model(
    btc_features: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "regime_label",
    model_cfg: ModelConfig | None = None,
    wf_cfg: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    """Train a BTC regime logistic model in walk-forward fashion.

    Parameters
    ----------
    btc_features : pd.DataFrame
        BTC-only features with columns: ts, <feature_cols>, <label_col>.
        label_col should be binary: 1 = trending/bull, 0 = not.
    feature_cols : list[str]
        Feature columns.
    model_cfg / wf_cfg : configs.

    Returns
    -------
    pd.DataFrame with columns: ts, p_regime
    """
    if model_cfg is None:
        model_cfg = ModelConfig()
    if wf_cfg is None:
        wf_cfg = WalkForwardConfig()

    btc_features = btc_features.dropna(subset=[label_col] + feature_cols)
    unique_dates = np.sort(btc_features["ts"].unique())

    splits = walk_forward_splits(
        unique_dates,
        train_days=wf_cfg.train_days,
        test_days=wf_cfg.test_days,
        step_days=wf_cfg.step_days,
        min_train_days=wf_cfg.min_train_days,
    )

    all_preds = []
    for sp in splits:
        train_mask = (
            (btc_features["ts"] >= sp["train_start"])
            & (btc_features["ts"] <= sp["train_end"])
        )
        test_mask = (
            (btc_features["ts"] >= sp["test_start"])
            & (btc_features["ts"] <= sp["test_end"])
        )
        train = btc_features.loc[train_mask]
        test = btc_features.loc[test_mask]

        if len(train) < 50 or len(test) < 5:
            continue

        X_train = train[feature_cols].values
        y_train = train[label_col].values.astype(int)
        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(test[feature_cols].values)

        est = _build_estimator(model_cfg)
        est.fit(X_tr, y_train)
        probs = np.clip(est.predict_proba(X_te)[:, 1], 1e-7, 1 - 1e-7)

        pred = test[["ts"]].copy()
        pred["p_regime"] = probs
        all_preds.append(pred)

    if not all_preds:
        return pd.DataFrame(columns=["ts", "p_regime"])

    return pd.concat(all_preds, ignore_index=True).drop_duplicates("ts", keep="last")
