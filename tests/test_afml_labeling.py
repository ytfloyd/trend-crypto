"""Tests for src/afml/labeling.py — triple-barrier and meta-labeling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.labeling import (
    daily_volatility,
    fixed_horizon_labels,
    make_events,
    meta_labels,
    triple_barrier_labels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_price_series(n: int = 300, seed: int = 42) -> pd.Series:
    """Synthetic daily close prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    log_ret = rng.normal(0, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(close, index=dates, name="close")


@pytest.fixture
def close():
    return _make_price_series()


# ---------------------------------------------------------------------------
# Daily volatility
# ---------------------------------------------------------------------------

class TestDailyVolatility:
    def test_basic(self, close):
        vol = daily_volatility(close, span=20)
        assert len(vol) == len(close)
        # First `span` values should be NaN
        assert vol.iloc[:19].isna().all()
        assert vol.iloc[20:].notna().all()
        # Volatility should be positive
        assert (vol.dropna() > 0).all()

    def test_higher_vol_period(self):
        """Inject a high-vol period and check detection."""
        dates = pd.bdate_range("2023-01-01", periods=200)
        rng = np.random.default_rng(7)
        ret = np.concatenate([rng.normal(0, 0.01, 100), rng.normal(0, 0.05, 100)])
        close = pd.Series(100.0 * np.exp(np.cumsum(ret)), index=dates)
        vol = daily_volatility(close, span=10)
        # Vol in second half should be higher than first
        assert vol.iloc[150:].mean() > vol.iloc[50:100].mean()


# ---------------------------------------------------------------------------
# Triple-barrier labels
# ---------------------------------------------------------------------------

class TestTripleBarrierLabels:
    def test_profit_take(self):
        """Price goes straight up -> should hit upper barrier."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        close = pd.Series(100.0 + np.arange(20) * 2.0, index=dates)
        events = pd.DataFrame({
            "t1": [dates[10]],
            "trgt": [0.05],
            "side": [1.0],
        }, index=[dates[0]])

        out = triple_barrier_labels(close, events, pt_sl=(1.0, 1.0))
        assert len(out) == 1
        assert out["label"].iloc[0] == 1  # profit take

    def test_stop_loss(self):
        """Price goes straight down -> should hit lower barrier."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        close = pd.Series(100.0 - np.arange(20) * 2.0, index=dates)
        events = pd.DataFrame({
            "t1": [dates[10]],
            "trgt": [0.05],
            "side": [1.0],
        }, index=[dates[0]])

        out = triple_barrier_labels(close, events, pt_sl=(1.0, 1.0))
        assert len(out) == 1
        assert out["label"].iloc[0] == -1  # stop loss

    def test_vertical_barrier(self):
        """Price stays flat -> should hit time expiry."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        close = pd.Series(100.0, index=dates)  # perfectly flat
        events = pd.DataFrame({
            "t1": [dates[5]],
            "trgt": [0.10],
            "side": [1.0],
        }, index=[dates[0]])

        out = triple_barrier_labels(close, events, pt_sl=(1.0, 1.0))
        assert len(out) == 1
        assert out["label"].iloc[0] == 0  # time expiry
        assert np.isclose(out["ret"].iloc[0], 0.0)

    def test_disable_stop_loss(self):
        """With sl=0, only profit-take and vertical barriers are active."""
        dates = pd.bdate_range("2023-01-01", periods=30)
        close = pd.Series(100.0 - np.arange(30) * 0.5, index=dates)  # downtrend
        events = pd.DataFrame({
            "t1": [dates[10]],
            "trgt": [0.02],
            "side": [1.0],
        }, index=[dates[0]])

        out = triple_barrier_labels(close, events, pt_sl=(1.0, 0.0))
        assert len(out) == 1
        # Should expire (vertical) since no stop-loss and price went down
        assert out["label"].iloc[0] == 0

    def test_short_side(self):
        """With side=-1, a price drop should be profit-take."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        close = pd.Series(100.0 - np.arange(20) * 2.0, index=dates)
        events = pd.DataFrame({
            "t1": [dates[10]],
            "trgt": [0.05],
            "side": [-1.0],
        }, index=[dates[0]])

        out = triple_barrier_labels(close, events, pt_sl=(1.0, 1.0))
        assert out["label"].iloc[0] == 1  # profit take on short

    def test_multiple_events(self, close):
        vol = daily_volatility(close, span=20)
        events = make_events(close, vol, holding_periods=10)
        out = triple_barrier_labels(close, events, pt_sl=(2.0, 2.0))
        assert len(out) > 10
        assert set(out["label"].unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# make_events
# ---------------------------------------------------------------------------

class TestMakeEvents:
    def test_basic(self, close):
        vol = daily_volatility(close, span=20)
        events = make_events(close, vol, holding_periods=10)
        assert "t1" in events.columns
        assert "trgt" in events.columns
        assert "side" in events.columns
        assert (events["trgt"] > 0).all()
        assert len(events) > 0

    def test_with_side(self, close):
        vol = daily_volatility(close, span=20)
        side = pd.Series(np.where(np.arange(len(close)) % 2 == 0, 1, -1),
                         index=close.index)
        events = make_events(close, vol, holding_periods=5, side=side)
        assert set(events["side"].unique()).issubset({-1.0, 1.0})


# ---------------------------------------------------------------------------
# Meta-labeling
# ---------------------------------------------------------------------------

class TestMetaLabels:
    def test_basic(self):
        dates = pd.bdate_range("2023-01-01", periods=5)
        primary = pd.Series([1, 1, -1, -1, 1], index=dates, dtype=float)
        tb_out = pd.DataFrame({
            "t1": dates + pd.Timedelta(days=5),
            "ret": [0.05, -0.03, 0.04, -0.02, 0.01],
            "label": [1, -1, 1, -1, 1],
            "trgt": [0.02] * 5,
        }, index=dates)

        ml = meta_labels(primary, tb_out)
        assert len(ml) == 5
        # primary=+1, ret=+0.05 -> signed_ret=+0.05 -> meta=1
        assert ml["meta_label"].iloc[0] == 1
        # primary=+1, ret=-0.03 -> signed_ret=-0.03 -> meta=0
        assert ml["meta_label"].iloc[1] == 0
        # primary=-1, ret=+0.04 -> signed_ret=-0.04 -> meta=0
        assert ml["meta_label"].iloc[2] == 0
        # primary=-1, ret=-0.02 -> signed_ret=+0.02 -> meta=1
        assert ml["meta_label"].iloc[3] == 1

    def test_empty_overlap(self):
        dates_a = pd.bdate_range("2023-01-01", periods=3)
        dates_b = pd.bdate_range("2023-06-01", periods=3)
        primary = pd.Series([1, 1, 1], index=dates_a, dtype=float)
        tb_out = pd.DataFrame({
            "t1": dates_b, "ret": [0.01] * 3,
            "label": [1] * 3, "trgt": [0.02] * 3,
        }, index=dates_b)
        ml = meta_labels(primary, tb_out)
        assert len(ml) == 0


# ---------------------------------------------------------------------------
# Fixed-horizon labels
# ---------------------------------------------------------------------------

class TestFixedHorizonLabels:
    def test_sign_method(self, close):
        labels = fixed_horizon_labels(close, horizon=1, method="sign")
        assert set(labels.dropna().unique()).issubset({-1, 0, 1})
        # Last value should be NaN (no forward return)
        assert labels.iloc[-1] == 0 or np.isnan(close.iloc[-1] / close.iloc[-1] - 1)

    def test_threshold_method(self, close):
        labels = fixed_horizon_labels(close, horizon=5, method="threshold", threshold=0.03)
        assert set(labels.unique()).issubset({-1, 0, 1})
        # With 3% threshold, should have some 0 labels
        assert (labels == 0).sum() > 0
