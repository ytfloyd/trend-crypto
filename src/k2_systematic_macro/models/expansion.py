"""Walk-forward probabilistic volatility-expansion baselines."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpansionModelConfig:
    """Model and validation settings for expansion probability research."""

    horizon: str = "1d"
    target_col: str | None = None
    model_kinds: tuple[str, ...] = ("logistic",)
    min_train_size: int = 120
    test_size: int = 40
    step_size: int | None = None
    calibration_bins: int = 5
    feature_columns: tuple[str, ...] = (
        "realized_vol_12",
        "realized_vol_24",
        "vol_of_vol_24",
        "atr_compression",
        "range_compression",
        "trend_persistence",
        "trend_return_atr",
        "breakout_channel_width_atr",
        "regime",
        "regime_probability",
    )


@dataclass(frozen=True)
class ExpansionModelResult:
    """Walk-forward outputs for one model kind."""

    model_kind: str
    predictions: pd.DataFrame
    metrics: dict[str, float | int | None]
    fold_metrics: pd.DataFrame
    calibration: pd.DataFrame


def walk_forward_expansion_models(
    frame: pd.DataFrame,
    config: ExpansionModelConfig | None = None,
) -> dict[str, ExpansionModelResult]:
    """Run one or more walk-forward expansion classifiers."""
    cfg = config or ExpansionModelConfig()
    results: dict[str, ExpansionModelResult] = {}
    for model_kind in cfg.model_kinds:
        data = _model_frame(frame, cfg)
        if data.empty:
            results[model_kind] = _empty_result(model_kind)
            continue
        splits = list(_walk_forward_splits(data.shape[0], cfg))
        if not splits:
            results[model_kind] = _empty_result(model_kind)
            continue
        results[model_kind] = _fit_walk_forward(data, cfg, model_kind, splits)
    return results


def available_boosters() -> dict[str, bool]:
    """Report optional tree-boosting dependencies without importing them."""
    return {
        "xgboost": find_spec("xgboost") is not None,
        "lightgbm": find_spec("lightgbm") is not None,
    }


def _model_frame(frame: pd.DataFrame, config: ExpansionModelConfig) -> pd.DataFrame:
    target_col = config.target_col or f"target_vol_expansion_event_{_suffix(config.horizon)}"
    feature_columns = [col for col in config.feature_columns if col in frame]
    if target_col not in frame or not feature_columns:
        return pd.DataFrame()
    keep = ["ts", target_col, *feature_columns]
    out = frame[keep].copy()
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    for col in feature_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out[target_col].isin([0.0, 1.0])]
    return out.sort_values("ts").reset_index(drop=True)


def _walk_forward_splits(
    n_rows: int,
    config: ExpansionModelConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    step = config.step_size or config.test_size
    splits = []
    train_end = config.min_train_size
    while train_end < n_rows:
        test_end = min(train_end + config.test_size, n_rows)
        if test_end <= train_end:
            break
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
        train_end += step
    return splits


def _fit_walk_forward(
    data: pd.DataFrame,
    config: ExpansionModelConfig,
    model_kind: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> ExpansionModelResult:
    target_col = config.target_col or f"target_vol_expansion_event_{_suffix(config.horizon)}"
    feature_columns = [col for col in config.feature_columns if col in data]
    predictions = []
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        if train[target_col].nunique() < 2:
            continue
        estimator = _make_estimator(model_kind)
        estimator.fit(train[feature_columns], train[target_col].astype(int))
        prob = estimator.predict_proba(test[feature_columns])[:, 1]
        fold_pred = pd.DataFrame(
            {
                "ts": test["ts"].to_numpy(),
                "fold": fold,
                "y_true": test[target_col].astype(float).to_numpy(),
                "y_prob": prob,
                "model_kind": model_kind,
            }
        )
        predictions.append(fold_pred)
        fold_rows.append(_score_predictions(fold_pred, fold=fold))

    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metrics = _score_predictions(pred) if not pred.empty else _empty_metrics()
    fold_metrics = pd.DataFrame(fold_rows)
    calibration = _calibration(pred, config.calibration_bins)
    return ExpansionModelResult(model_kind, pred, metrics, fold_metrics, calibration)


def _make_estimator(model_kind: str) -> Any:
    if model_kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if model_kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    if model_kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    raise ValueError(f"unsupported expansion model kind: {model_kind}")


def _score_predictions(predictions: pd.DataFrame, *, fold: int | None = None) -> dict[str, float | int | None]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    prefix: dict[str, float | int | None] = {}
    if fold is not None:
        prefix["fold"] = fold
    if predictions.empty:
        return {**prefix, **_empty_metrics()}
    y_true = predictions["y_true"].astype(float)
    y_prob = predictions["y_prob"].astype(float).clip(1e-6, 1 - 1e-6)
    metrics: dict[str, float | int | None] = {
        "n": int(predictions.shape[0]),
        "event_rate": float(y_true.mean()),
        "mean_probability": float(y_prob.mean()),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "roc_auc": None,
    }
    if y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return {**prefix, **metrics}


def _calibration(predictions: pd.DataFrame, bins: int) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_probability", "event_rate"])
    out = predictions.copy()
    out["bin"] = pd.qcut(out["y_prob"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    return (
        out.groupby("bin", dropna=True)
        .agg(n=("y_true", "size"), mean_probability=("y_prob", "mean"), event_rate=("y_true", "mean"))
        .reset_index()
    )


def _empty_result(model_kind: str) -> ExpansionModelResult:
    return ExpansionModelResult(model_kind, pd.DataFrame(), _empty_metrics(), pd.DataFrame(), pd.DataFrame())


def _empty_metrics() -> dict[str, float | int | None]:
    return {
        "n": 0,
        "event_rate": None,
        "mean_probability": None,
        "brier": None,
        "log_loss": None,
        "roc_auc": None,
    }


def _suffix(horizon: str) -> str:
    return horizon.lower().replace(" ", "_")
