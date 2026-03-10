"""Tests for src/afml/feature_importance.py — MDI, MDA, SFI."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier

from afml.feature_importance import (
    mean_decrease_accuracy,
    mean_decrease_impurity,
    single_feature_importance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_with_signal():
    """Feature matrix where feature 0 is informative, 1-2 are noise."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int)  # only feature 0 matters
    return X, y


@pytest.fixture
def feature_names():
    return ["signal", "noise_1", "noise_2"]


@pytest.fixture
def t1_series():
    dates = pd.bdate_range("2023-01-02", periods=200)
    return pd.Series([dates[min(i + 3, 199)] for i in range(200)], index=dates)


@pytest.fixture
def cv(t1_series):
    from validation.cv import PurgedKFold
    return PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=0.01)


# ---------------------------------------------------------------------------
# MDI
# ---------------------------------------------------------------------------

class TestMDI:
    def test_basic(self, data_with_signal, feature_names):
        X, y = data_with_signal
        rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        rf.fit(X, y)

        result = mean_decrease_impurity(rf, feature_names)
        assert len(result) == 3
        assert "mdi_mean" in result.columns
        # Signal feature should be most important
        assert result.iloc[0]["feature"] == "signal"

    def test_sums_to_one(self, data_with_signal, feature_names):
        X, y = data_with_signal
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        result = mean_decrease_impurity(rf, feature_names)
        assert np.isclose(result["mdi_mean"].sum(), 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# MDA
# ---------------------------------------------------------------------------

class TestMDA:
    def test_basic(self, data_with_signal, feature_names, cv):
        X, y = data_with_signal
        rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)

        result = mean_decrease_accuracy(rf, X, y, cv, feature_names)
        assert len(result) == 3
        assert "mda_mean" in result.columns
        # Signal feature should have largest accuracy drop
        assert result.iloc[0]["feature"] == "signal"

    def test_noise_features_near_zero(self, data_with_signal, feature_names, cv):
        X, y = data_with_signal
        rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)

        result = mean_decrease_accuracy(rf, X, y, cv, feature_names)
        noise = result[result["feature"].isin(["noise_1", "noise_2"])]
        # Noise features should have MDA near 0
        assert all(abs(noise["mda_mean"]) < 0.15)


# ---------------------------------------------------------------------------
# SFI
# ---------------------------------------------------------------------------

class TestSFI:
    def test_basic(self, data_with_signal, feature_names, cv):
        X, y = data_with_signal
        rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)

        result = single_feature_importance(rf, X, y, cv, feature_names)
        assert len(result) == 3
        assert "sfi_mean" in result.columns
        # Signal feature should score highest
        assert result.iloc[0]["feature"] == "signal"

    def test_noise_near_random(self, data_with_signal, feature_names, cv):
        X, y = data_with_signal
        rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)

        result = single_feature_importance(rf, X, y, cv, feature_names)
        noise = result[result["feature"].isin(["noise_1", "noise_2"])]
        # Noise features trained alone should be ~50% accuracy
        assert all(noise["sfi_mean"] < 0.6)
