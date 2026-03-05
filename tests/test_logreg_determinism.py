"""
Determinism test: same seed + config must produce identical results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy", reason="scipy required for logreg research tests")
pytest.importorskip("sklearn", reason="sklearn required for logreg research tests")

from scripts.research.logreg_filter.labels import (
    BarrierLabelConfig,
    barrier_labels,
    forward_return_labels,
    ForwardReturnLabelConfig,
)
from scripts.research.logreg_filter.features import (
    FeatureConfig,
    compute_features_single,
    get_feature_columns,
)
from scripts.research.logreg_filter.overlay import (
    apply_entry_filter,
    apply_conviction_sizing,
)


@pytest.fixture
def deterministic_data() -> pd.DataFrame:
    np.random.seed(123)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n, freq="D")
    close = 50.0 * np.exp(np.cumsum(np.random.randn(n) * 0.015))
    high = close * (1 + np.abs(np.random.randn(n) * 0.008))
    low = close * (1 - np.abs(np.random.randn(n) * 0.008))
    _open = close * (1 + np.random.randn(n) * 0.003)
    vol = np.random.randint(500, 50000, size=n).astype(float)
    return pd.DataFrame({
        "ts": dates, "open": _open, "high": high,
        "low": low, "close": close, "volume": vol, "symbol": "DET-USD",
    })


class TestDeterminism:
    def test_features_deterministic(self, deterministic_data: pd.DataFrame):
        cfg = FeatureConfig(ret_windows=[1, 5], atr_pctl_lookback=50)
        f1 = compute_features_single(deterministic_data.copy(), cfg)
        f2 = compute_features_single(deterministic_data.copy(), cfg)
        cols = get_feature_columns(cfg)
        for col in cols:
            pd.testing.assert_series_equal(f1[col], f2[col], check_names=False)

    def test_barrier_labels_deterministic(self, deterministic_data: pd.DataFrame):
        df = deterministic_data.set_index("ts")
        cfg = BarrierLabelConfig(horizon=10, atr_window=14)
        l1 = barrier_labels(df["high"], df["low"], df["close"], cfg)
        l2 = barrier_labels(df["high"], df["low"], df["close"], cfg)
        pd.testing.assert_series_equal(l1, l2)

    def test_fwd_labels_deterministic(self, deterministic_data: pd.DataFrame):
        df = deterministic_data.set_index("ts")
        cfg = ForwardReturnLabelConfig(horizon=10)
        l1 = forward_return_labels(df["close"], cfg=cfg)
        l2 = forward_return_labels(df["close"], cfg=cfg)
        pd.testing.assert_series_equal(l1, l2)

    def test_overlay_deterministic(self):
        np.random.seed(99)
        dates = pd.bdate_range("2020-01-01", periods=50)
        syms = ["A", "B", "C"]
        w = pd.DataFrame(
            np.random.rand(50, 3) * 0.1, index=dates, columns=syms,
        )
        p = pd.DataFrame(
            np.random.rand(50, 3), index=dates, columns=syms,
        )

        f1 = apply_entry_filter(w.copy(), p.copy(), p_enter=0.6)
        f2 = apply_entry_filter(w.copy(), p.copy(), p_enter=0.6)
        pd.testing.assert_frame_equal(f1, f2)

        s1 = apply_conviction_sizing(f1.copy(), p.copy())
        s2 = apply_conviction_sizing(f2.copy(), p.copy())
        pd.testing.assert_frame_equal(s1, s2)
