"""
Combinatorial Purged Cross-Validation (CPCV).

Extends the PurgedKFold from src/validation/cv.py with the combinatorial
approach from AFML Chapter 12.  Standard purged k-fold yields a single
backtest path.  CPCV generates C(N, k) paths by testing on every
combination of k groups out of N, enabling the Probability of
Backtest Overfitting (PBO) test.

Also provides a cv_score helper that runs sklearn-compatible estimators
through purged CV with proper sample weights.

Reference:
    López de Prado, M. (2018) *Advances in Financial Machine Learning*,
    Chapter 7: Cross-Validation in Finance (pp. 99–113),
    Chapter 12: Backtesting through Cross-Validation (pp. 171–183).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Generator

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# CPCV  (AFML Chapter 12)
# -----------------------------------------------------------------------

class CombinatorialPurgedKFold:
    """Combinatorial Purged Cross-Validation.

    Partitions data into *n_groups* contiguous blocks, then generates
    all C(n_groups, n_test_groups) train/test splits where *n_test_groups*
    blocks form the test set and the remaining blocks form the training
    set (with purging and embargo applied).

    Each combination produces one backtest "path" covering the test
    blocks.  Collecting all paths enables the PBO test: what fraction
    of paths overfit?

    Parameters
    ----------
    n_groups : int
        Number of contiguous blocks to partition the data into.
    n_test_groups : int
        Number of blocks in each test split (typically 2).
    t1 : pd.Series
        Index = prediction time, value = label end time.
    pct_embargo : float
        Fraction of samples to embargo after each test block.

    Reference
    ---------
    AFML Section 12.3, pp. 174–178.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.01,
    ):
        if t1 is None:
            raise ValueError("t1 (label end times) must be provided.")
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be < n_groups.")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of splits = C(n_groups, n_test_groups)."""
        from math import comb
        return comb(self.n_groups, self.n_test_groups)

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_indices, test_indices) for each combination.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Used only for its length.

        Yields
        ------
        train_idx : np.ndarray
        test_idx : np.ndarray
        """
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        indices = np.arange(n_samples)
        embargo = int(n_samples * self.pct_embargo)

        # Partition into contiguous blocks
        blocks = np.array_split(indices, self.n_groups)
        block_ranges = [(b[0], b[-1] + 1) for b in blocks]

        for test_combo in combinations(range(self.n_groups), self.n_test_groups):
            test_indices = np.concatenate([blocks[g] for g in test_combo])

            # Build the purged+embargoed training set
            train_indices = self._purge_train(
                indices, test_indices, block_ranges, test_combo, embargo
            )
            yield train_indices, test_indices

    def _purge_train(
        self,
        all_indices: np.ndarray,
        test_indices: np.ndarray,
        block_ranges: list[tuple[int, int]],
        test_combo: tuple[int, ...],
        embargo: int,
    ) -> np.ndarray:
        """Remove test indices, purge overlapping labels, apply embargo."""
        test_set = set(test_indices)

        # Determine test time boundaries for purging
        test_starts = [block_ranges[g][0] for g in test_combo]
        test_ends = [block_ranges[g][1] for g in test_combo]

        train = []
        for i in all_indices:
            if i in test_set:
                continue

            # Check purging: does this training sample's label overlap any test block?
            purged = False
            pred_time = self.t1.index[i]
            label_end = self.t1.iloc[i]

            for t_start_idx, t_end_idx in zip(test_starts, test_ends):
                test_start_time = self.t1.index[t_start_idx]
                test_end_time = self.t1.iloc[t_start_idx:t_end_idx].max()

                # Purge: train label overlaps test window
                if i < t_start_idx and label_end >= test_start_time:
                    purged = True
                    break

                # Embargo: train sample too close after test block
                if i >= t_end_idx:
                    if i < t_end_idx + embargo:
                        purged = True
                        break
                    if pred_time <= test_end_time:
                        purged = True
                        break

            if not purged:
                train.append(i)

        return np.array(train)


# -----------------------------------------------------------------------
# cv_score: run a model through purged CV with sample weights
# -----------------------------------------------------------------------

def cv_score(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    *,
    sample_weight: np.ndarray | None = None,
    scoring: str = "accuracy",
) -> list[float]:
    """Evaluate an estimator using a purged CV splitter.

    Supports sample_weight passed to both fit() and score().
    Compatible with PurgedKFold and CombinatorialPurgedKFold.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Must implement fit() and predict() / score().
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    cv : splitter
        PurgedKFold, CombinatorialPurgedKFold, or similar.
    sample_weight : np.ndarray | None
        Per-sample weights.
    scoring : str
        ``"accuracy"`` or ``"neg_log_loss"``.

    Returns
    -------
    list of float scores, one per fold.
    """
    from sklearn.metrics import accuracy_score, log_loss

    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[train_idx]

        clone = _clone_estimator(estimator)
        clone.fit(X_train, y_train, **fit_kwargs)

        if scoring == "accuracy":
            y_pred = clone.predict(X_test)
            w_test = sample_weight[test_idx] if sample_weight is not None else None
            scores.append(accuracy_score(y_test, y_pred, sample_weight=w_test))
        elif scoring == "neg_log_loss":
            y_prob = clone.predict_proba(X_test)
            w_test = sample_weight[test_idx] if sample_weight is not None else None
            scores.append(-log_loss(y_test, y_prob, sample_weight=w_test))
        else:
            scores.append(clone.score(X_test, y_test))

    return scores


def _clone_estimator(estimator: Any) -> Any:
    """Clone an sklearn estimator."""
    try:
        from sklearn.base import clone
        return clone(estimator)
    except ImportError:
        return estimator.__class__(**estimator.get_params())
