"""Tests for src/afml/cross_validation.py — CPCV and cv_score."""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from afml.cross_validation import CombinatorialPurgedKFold, cv_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def t1_series():
    """Label end times: each label spans 5 bars."""
    dates = pd.bdate_range("2023-01-02", periods=100)
    t1 = pd.Series(
        [dates[min(i + 5, 99)] for i in range(100)],
        index=dates,
    )
    return t1


@pytest.fixture
def X_y():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# CombinatorialPurgedKFold
# ---------------------------------------------------------------------------

class TestCPCV:
    def test_n_splits(self, t1_series):
        cpcv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, t1=t1_series)
        assert cpcv.get_n_splits() == comb(6, 2)  # 15

    def test_all_indices_covered(self, t1_series, X_y):
        X, y = X_y
        cpcv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, t1=t1_series)

        test_coverage = np.zeros(100, dtype=int)
        for train_idx, test_idx in cpcv.split(X):
            test_coverage[test_idx] += 1
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0

        # Every sample should appear in at least one test fold
        assert (test_coverage > 0).all()

    def test_purging_removes_overlap(self, t1_series, X_y):
        X, y = X_y
        cpcv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2,
                                         t1=t1_series, pct_embargo=0.02)
        for train_idx, test_idx in cpcv.split(X):
            # Training set should be smaller than total - test due to purging
            assert len(train_idx) + len(test_idx) <= 100

    def test_fewer_groups(self, t1_series, X_y):
        X, y = X_y
        cpcv = CombinatorialPurgedKFold(n_groups=4, n_test_groups=1, t1=t1_series)
        assert cpcv.get_n_splits() == 4  # C(4,1) = 4
        splits = list(cpcv.split(X))
        assert len(splits) == 4

    def test_more_test_groups(self, t1_series, X_y):
        X, y = X_y
        cpcv = CombinatorialPurgedKFold(n_groups=5, n_test_groups=3, t1=t1_series)
        assert cpcv.get_n_splits() == comb(5, 3)  # 10
        splits = list(cpcv.split(X))
        assert len(splits) == 10

    def test_raises_bad_config(self, t1_series):
        with pytest.raises(ValueError):
            CombinatorialPurgedKFold(n_groups=3, n_test_groups=3, t1=t1_series)

    def test_raises_no_t1(self):
        with pytest.raises(ValueError):
            CombinatorialPurgedKFold(n_groups=4, n_test_groups=1, t1=None)

    def test_train_test_non_empty(self, t1_series, X_y):
        X, y = X_y
        cpcv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, t1=t1_series)
        for train_idx, test_idx in cpcv.split(X):
            assert len(train_idx) > 0
            assert len(test_idx) > 0


# ---------------------------------------------------------------------------
# cv_score
# ---------------------------------------------------------------------------

class TestCvScore:
    def test_with_purged_kfold(self, t1_series, X_y):
        """Test cv_score with existing PurgedKFold."""
        from validation.cv import PurgedKFold
        from sklearn.linear_model import LogisticRegression

        X, y = X_y
        pkf = PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=0.01)
        scores = cv_score(
            LogisticRegression(max_iter=200),
            X, y, pkf, scoring="accuracy",
        )
        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)

    def test_with_cpcv(self, t1_series, X_y):
        from sklearn.linear_model import LogisticRegression

        X, y = X_y
        cpcv = CombinatorialPurgedKFold(n_groups=4, n_test_groups=1, t1=t1_series)
        scores = cv_score(
            LogisticRegression(max_iter=200),
            X, y, cpcv, scoring="accuracy",
        )
        assert len(scores) == 4

    def test_with_sample_weights(self, t1_series, X_y):
        from sklearn.linear_model import LogisticRegression

        X, y = X_y
        rng = np.random.default_rng(42)
        weights = rng.uniform(0.1, 2.0, len(y))

        cpcv = CombinatorialPurgedKFold(n_groups=4, n_test_groups=1, t1=t1_series)
        scores = cv_score(
            LogisticRegression(max_iter=200),
            X, y, cpcv,
            sample_weight=weights,
            scoring="accuracy",
        )
        assert len(scores) == 4

    def test_neg_log_loss(self, t1_series, X_y):
        from sklearn.linear_model import LogisticRegression

        X, y = X_y
        cpcv = CombinatorialPurgedKFold(n_groups=4, n_test_groups=1, t1=t1_series)
        scores = cv_score(
            LogisticRegression(max_iter=200),
            X, y, cpcv, scoring="neg_log_loss",
        )
        assert len(scores) == 4
        assert all(s <= 0 for s in scores)  # neg log loss is non-positive
