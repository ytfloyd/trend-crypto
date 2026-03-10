"""
Feature importance methods for financial ML.

Implements three complementary approaches from AFML Chapter 8:
  - MDI (Mean Decrease Impurity) — from tree structure
  - MDA (Mean Decrease Accuracy) — permutation-based
  - SFI (Single Feature Importance) — individual feature CV scores

Standard sklearn feature_importances_ (MDI) is biased towards high-cardinality
and noisy features.  MDA and SFI provide unbiased alternatives when used
with purged cross-validation and sample weights.

Reference:
    López de Prado, M. (2018) *Advances in Financial Machine Learning*,
    Chapter 8: Feature Importance (pp. 115–131).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# MDI — Mean Decrease Impurity  (AFML Snippet 8.1)
# -----------------------------------------------------------------------

def mean_decrease_impurity(
    estimator: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract MDI feature importance from a fitted tree ensemble.

    MDI measures how much each feature reduces impurity (Gini/entropy)
    across all trees.  Fast but biased towards features with many
    distinct values and correlated features.

    Parameters
    ----------
    estimator : fitted sklearn ensemble (RandomForest, ExtraTrees, etc.)
        Must have ``feature_importances_`` and ``estimators_`` attributes.
    feature_names : list[str]
        Feature column names.

    Returns
    -------
    pd.DataFrame with columns: feature, mdi_mean, mdi_std.
        Sorted by mdi_mean descending.
    """
    importances = np.array([
        tree.feature_importances_ for tree in estimator.estimators_
    ])

    result = pd.DataFrame({
        "feature": feature_names,
        "mdi_mean": importances.mean(axis=0),
        "mdi_std": importances.std(axis=0),
    })
    result["mdi_mean"] /= result["mdi_mean"].sum()
    return result.sort_values("mdi_mean", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------
# MDA — Mean Decrease Accuracy  (AFML Snippet 8.2)
# -----------------------------------------------------------------------

def mean_decrease_accuracy(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    feature_names: list[str],
    *,
    sample_weight: np.ndarray | None = None,
    n_repeats: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute MDA (permutation importance) with purged CV.

    For each fold and each feature, shuffle the feature values and
    measure the drop in accuracy.  Features that cause large drops
    when shuffled are important.

    Unlike MDI, MDA is unbiased and works with any estimator.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    cv : splitter (PurgedKFold, CPCV, etc.)
    feature_names : list[str]
    sample_weight : np.ndarray | None
    n_repeats : int
        Number of times to permute each feature per fold.
    random_state : int

    Returns
    -------
    pd.DataFrame with columns: feature, mda_mean, mda_std.
        mda_mean = average accuracy drop when feature is shuffled.
        Sorted by mda_mean descending.
    """
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    importance_per_fold: list[np.ndarray] = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[train_idx]

        clone = _clone(estimator)
        clone.fit(X_train, y_train, **fit_kwargs)

        w_test = sample_weight[test_idx] if sample_weight is not None else None
        baseline = accuracy_score(y_test, clone.predict(X_test), sample_weight=w_test)

        fold_imp = np.zeros(n_features)
        for j in range(n_features):
            drops = []
            for _ in range(n_repeats):
                X_perm = X_test.copy()
                rng.shuffle(X_perm[:, j])
                score_perm = accuracy_score(y_test, clone.predict(X_perm), sample_weight=w_test)
                drops.append(baseline - score_perm)
            fold_imp[j] = np.mean(drops)

        importance_per_fold.append(fold_imp)

    imp = np.array(importance_per_fold)
    result = pd.DataFrame({
        "feature": feature_names,
        "mda_mean": imp.mean(axis=0),
        "mda_std": imp.std(axis=0),
    })
    return result.sort_values("mda_mean", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------
# SFI — Single Feature Importance  (AFML Snippet 8.3)
# -----------------------------------------------------------------------

def single_feature_importance(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    feature_names: list[str],
    *,
    sample_weight: np.ndarray | None = None,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Compute SFI by training on each feature individually.

    For each feature, trains the model using only that feature and
    evaluates via purged CV.  Avoids the substitution effects that
    plague MDI with correlated features.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    cv : splitter
    feature_names : list[str]
    sample_weight : np.ndarray | None
    scoring : str

    Returns
    -------
    pd.DataFrame with columns: feature, sfi_mean, sfi_std.
        Sorted by sfi_mean descending.
    """
    from afml.cross_validation import cv_score

    results = []
    for j, feat in enumerate(feature_names):
        X_single = X[:, j : j + 1]
        scores = cv_score(
            estimator, X_single, y, cv,
            sample_weight=sample_weight,
            scoring=scoring,
        )
        results.append({
            "feature": feat,
            "sfi_mean": np.mean(scores),
            "sfi_std": np.std(scores),
        })

    result = pd.DataFrame(results)
    return result.sort_values("sfi_mean", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _clone(estimator: Any) -> Any:
    try:
        from sklearn.base import clone
        return clone(estimator)
    except ImportError:
        return estimator.__class__(**estimator.get_params())
