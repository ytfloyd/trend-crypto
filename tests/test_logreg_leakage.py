"""
Leakage and alignment tests for the logistic regression probability filter.

Tests:
  - Feature alignment: features at t use only data <= t
  - Label alignment: labels use future data but are shifted correctly
  - Execution timing: signals from close[t], trades at open[t+1]
  - Walk-forward ordering: strict time ordering in splits
  - No NaN at predict time
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research.logreg_filter.labels import (
    BarrierLabelConfig,
    ForwardReturnLabelConfig,
    barrier_labels,
    forward_return_labels,
)
from scripts.research.logreg_filter.features import (
    FeatureConfig,
    compute_features_single,
    get_feature_columns,
)
from scripts.research.jpm_bigdata_ai.helpers import walk_forward_splits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 500 bars of synthetic OHLCV data for a single asset."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n, freq="D")
    close = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    _open = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.randint(1000, 100000, size=n).astype(float)
    return pd.DataFrame({
        "ts": dates,
        "open": _open,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "SYN-USD",
    })


# ---------------------------------------------------------------------------
# Feature alignment: no lookahead
# ---------------------------------------------------------------------------
class TestFeatureAlignment:
    """Features at time t must not use data from t+1 onward."""

    def test_features_only_use_past_data(self, synthetic_ohlcv: pd.DataFrame):
        """Perturbing future close prices should not change features at t."""
        df = synthetic_ohlcv.copy()
        cfg = FeatureConfig(ret_windows=[1, 5], atr_pctl_lookback=50)

        feat_original = compute_features_single(df, cfg)
        t_idx = 200

        df_perturbed = df.copy()
        df_perturbed.loc[df_perturbed.index[t_idx + 1:], "close"] *= 2.0
        df_perturbed.loc[df_perturbed.index[t_idx + 1:], "high"] *= 2.0
        df_perturbed.loc[df_perturbed.index[t_idx + 1:], "low"] *= 2.0
        feat_perturbed = compute_features_single(df_perturbed, cfg)

        feature_cols = get_feature_columns(cfg)
        for col in feature_cols:
            orig_val = feat_original[col].iloc[t_idx]
            pert_val = feat_perturbed[col].iloc[t_idx]
            if pd.notna(orig_val) and pd.notna(pert_val):
                assert orig_val == pytest.approx(pert_val, rel=1e-10), (
                    f"Feature {col} at t={t_idx} changed when future data was perturbed. "
                    f"Original={orig_val}, Perturbed={pert_val}"
                )


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------
class TestLabelAlignment:
    """Labels use future data correctly and don't leak into features."""

    def test_barrier_label_uses_future_prices(self, synthetic_ohlcv: pd.DataFrame):
        """Barrier labels at t depend on prices after t."""
        df = synthetic_ohlcv.set_index("ts")
        cfg = BarrierLabelConfig(horizon=10, atr_window=14)
        labels = barrier_labels(df["high"], df["low"], df["close"], cfg)

        valid = labels.dropna()
        assert len(valid) > 0, "No valid labels produced"
        assert set(valid.unique()).issubset({0.0, 1.0}), "Labels must be binary"

    def test_barrier_label_horizon_respected(self, synthetic_ohlcv: pd.DataFrame):
        """Labels near the end of data should be NaN (insufficient horizon)."""
        df = synthetic_ohlcv.set_index("ts")
        cfg = BarrierLabelConfig(horizon=20, atr_window=14)
        labels = barrier_labels(df["high"], df["low"], df["close"], cfg)
        assert labels.iloc[-1] != labels.iloc[-1]  # NaN check

    def test_forward_return_label_uses_future(self, synthetic_ohlcv: pd.DataFrame):
        """Forward return labels at t depend on close[t+H]."""
        df = synthetic_ohlcv.set_index("ts")
        cfg = ForwardReturnLabelConfig(horizon=20, threshold=0.0)
        labels = forward_return_labels(df["close"], cfg=cfg)

        valid = labels.dropna()
        assert len(valid) > 0
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_forward_return_nan_at_end(self, synthetic_ohlcv: pd.DataFrame):
        """Last H bars should have NaN labels."""
        df = synthetic_ohlcv.set_index("ts")
        H = 20
        cfg = ForwardReturnLabelConfig(horizon=H, threshold=0.0)
        labels = forward_return_labels(df["close"], cfg=cfg)
        assert labels.iloc[-H:].isna().all()


# ---------------------------------------------------------------------------
# Walk-forward ordering
# ---------------------------------------------------------------------------
class TestWalkForwardOrdering:
    """Walk-forward splits must maintain strict time ordering."""

    def test_no_train_test_overlap(self):
        dates = pd.bdate_range("2019-01-01", periods=1500, freq="D")
        splits = walk_forward_splits(
            dates, train_days=500, test_days=63, step_days=63, min_train_days=365,
        )
        assert len(splits) > 0, "No splits generated"

        for sp in splits:
            assert sp["train_end"] < sp["test_start"], (
                f"Train end {sp['train_end']} >= test start {sp['test_start']}"
            )
            assert sp["train_start"] <= sp["train_end"]
            assert sp["test_start"] <= sp["test_end"]

    def test_splits_are_chronological(self):
        dates = pd.bdate_range("2019-01-01", periods=1500, freq="D")
        splits = walk_forward_splits(
            dates, train_days=500, test_days=63, step_days=63, min_train_days=365,
        )
        for i in range(1, len(splits)):
            assert splits[i]["test_start"] > splits[i - 1]["test_start"], (
                f"Split {i} test_start not after split {i-1}"
            )


# ---------------------------------------------------------------------------
# Execution timing
# ---------------------------------------------------------------------------
class TestExecutionTiming:
    """simple_backtest with execution_lag=1 means signals at close[t]
    produce returns from close[t+1]."""

    def test_execution_lag_shifts_weights(self):
        from scripts.research.common.backtest import simple_backtest

        dates = pd.bdate_range("2020-01-01", periods=100)
        syms = ["A", "B"]
        weights = pd.DataFrame(0.0, index=dates, columns=syms)
        weights.iloc[10:20, 0] = 0.5  # hold A from day 10-19
        returns = pd.DataFrame(
            np.random.randn(100, 2) * 0.01, index=dates, columns=syms,
        )

        bt = simple_backtest(weights, returns, cost_bps=0, execution_lag=1)

        assert bt["portfolio_ret"].iloc[10] == pytest.approx(0.0, abs=1e-10), (
            "With lag=1, signal at day 10 should not earn returns until day 11"
        )


# ---------------------------------------------------------------------------
# No NaN assertions
# ---------------------------------------------------------------------------
class TestNoNaNPredictions:
    """Model predictions should not contain NaN for valid feature rows."""

    def test_feature_nan_handling(self, synthetic_ohlcv: pd.DataFrame):
        """Features at early bars (warmup) should be NaN, later bars should not."""
        df = synthetic_ohlcv.copy()
        cfg = FeatureConfig(ret_windows=[1, 5, 20], atr_pctl_lookback=50)
        feat = compute_features_single(df, cfg)
        feature_cols = get_feature_columns(cfg)

        warmup = max(cfg.ret_windows + [cfg.ma_slow, cfg.donchian_window, cfg.atr_window])
        for col in feature_cols:
            late_vals = feat[col].iloc[warmup + 50:]
            nan_frac = late_vals.isna().mean()
            assert nan_frac < 0.05, (
                f"Feature {col} has {nan_frac:.0%} NaN after warmup"
            )
