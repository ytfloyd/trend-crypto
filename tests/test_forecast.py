"""Tests for src/strategy/forecast.py — Carver-style EWMAC and breakout forecasts."""
from __future__ import annotations

import numpy as np

from strategy.forecast import (
    FORECAST_CAP,
    _ewma,
    _rolling_max,
    _rolling_min,
    _rolling_std,
    _sma,
    breakout_forecast,
    breakout_raw,
    breakout_suite,
    cap_forecast,
    estimate_forecast_scalar,
    ewmac_forecast,
    ewmac_raw,
    ewmac_suite,
    long_only_forecast,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _trending_up(n: int = 500, start: float = 100.0, drift: float = 0.002) -> np.ndarray:
    """Deterministic uptrend with small noise."""
    rng = np.random.default_rng(42)
    returns = drift + rng.normal(0, 0.005, size=n)
    prices = start * np.cumprod(1 + returns)
    return prices


def _trending_down(n: int = 500, start: float = 100.0, drift: float = -0.002) -> np.ndarray:
    rng = np.random.default_rng(99)
    returns = drift + rng.normal(0, 0.005, size=n)
    prices = start * np.cumprod(1 + returns)
    return prices


def _flat_market(n: int = 500, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.005, size=n)
    prices = start * np.cumprod(1 + returns)
    return prices


# ── Rolling helpers ───────────────────────────────────────────────────

class TestRollingHelpers:
    def test_ewma_length(self):
        x = np.arange(50, dtype=float)
        result = _ewma(x, span=10)
        assert len(result) == 50

    def test_sma_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _sma(x, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_almost_equal(result[2], 2.0)
        np.testing.assert_almost_equal(result[3], 3.0)
        np.testing.assert_almost_equal(result[4], 4.0)

    def test_rolling_std_positive(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, size=100)
        result = _rolling_std(x, window=20)
        valid = result[~np.isnan(result)]
        assert (valid > 0).all()

    def test_rolling_max_min(self):
        x = np.array([5.0, 3.0, 8.0, 1.0, 6.0])
        assert _rolling_max(x, window=3)[4] == 8.0
        assert _rolling_min(x, window=3)[4] == 1.0


# ── EWMAC ─────────────────────────────────────────────────────────────

class TestEWMACRaw:
    def test_output_length(self):
        close = _trending_up(200)
        raw = ewmac_raw(close, fast_span=8, slow_span=32)
        assert len(raw) == 200

    def test_uptrend_positive(self):
        close = _trending_up(500)
        raw = ewmac_raw(close, fast_span=8, slow_span=32)
        valid = raw[~np.isnan(raw)]
        assert valid[-100:].mean() > 0, "EWMAC should be positive in an uptrend"

    def test_downtrend_negative(self):
        close = _trending_down(500)
        raw = ewmac_raw(close, fast_span=8, slow_span=32)
        valid = raw[~np.isnan(raw)]
        assert valid[-100:].mean() < 0, "EWMAC should be negative in a downtrend"

    def test_vol_lookback_override(self):
        close = _trending_up(300)
        raw_default = ewmac_raw(close, fast_span=8, slow_span=32)
        raw_custom = ewmac_raw(close, fast_span=8, slow_span=32, vol_lookback=60)
        valid_d = raw_default[~np.isnan(raw_default)]
        valid_c = raw_custom[~np.isnan(raw_custom)]
        assert not np.allclose(valid_d[-50:], valid_c[-50:])


class TestEWMACForecast:
    def test_capped(self):
        close = _trending_up(500)
        fc = ewmac_forecast(close, fast_span=8, slow_span=32)
        valid = fc[np.isfinite(fc)]
        assert valid.max() <= FORECAST_CAP
        assert valid.min() >= -FORECAST_CAP

    def test_explicit_scalar(self):
        close = _trending_up(300)
        fc = ewmac_forecast(close, fast_span=8, slow_span=32, scalar=5.0)
        raw = ewmac_raw(close, fast_span=8, slow_span=32)
        expected = np.clip(raw * 5.0, -FORECAST_CAP, FORECAST_CAP)
        np.testing.assert_array_almost_equal(
            fc[np.isfinite(fc)], expected[np.isfinite(expected)]
        )

    def test_custom_cap(self):
        close = _trending_up(500)
        fc = ewmac_forecast(close, fast_span=8, slow_span=32, cap=5.0)
        valid = fc[np.isfinite(fc)]
        assert valid.max() <= 5.0
        assert valid.min() >= -5.0


# ── Breakout ──────────────────────────────────────────────────────────

class TestBreakoutRaw:
    def test_output_range(self):
        close = _trending_up(300)
        high = close * 1.005
        low = close * 0.995
        raw = breakout_raw(close, high, low, lookback=20)
        valid = raw[np.isfinite(raw)]
        assert valid.min() >= -1.0 - 0.5  # some overshoot is possible
        assert valid.max() <= 1.0 + 0.5

    def test_uptrend_positive(self):
        close = _trending_up(500)
        high = close * 1.003
        low = close * 0.997
        raw = breakout_raw(close, high, low, lookback=40)
        valid = raw[np.isfinite(raw)]
        assert valid[-100:].mean() > 0, "Breakout should be positive near channel top"

    def test_no_lookahead(self):
        """Shifted window means index 0 should be NaN."""
        close = _trending_up(100)
        high = close * 1.01
        low = close * 0.99
        raw = breakout_raw(close, high, low, lookback=20)
        assert np.isnan(raw[0])


class TestBreakoutForecast:
    def test_capped(self):
        close = _trending_up(500)
        high = close * 1.005
        low = close * 0.995
        fc = breakout_forecast(close, high, low, lookback=40)
        valid = fc[np.isfinite(fc)]
        assert valid.max() <= FORECAST_CAP
        assert valid.min() >= -FORECAST_CAP


# ── Forecast scaling ──────────────────────────────────────────────────

class TestForecastScalar:
    def test_target_abs(self):
        rng = np.random.default_rng(5)
        raw = rng.normal(0, 2.0, size=500)
        scalar = estimate_forecast_scalar(raw, target_abs=10.0)
        scaled = raw * scalar
        np.testing.assert_allclose(
            np.median(np.abs(scaled)), 10.0, rtol=0.01
        )

    def test_insufficient_data(self):
        raw = np.array([1.0, 2.0])
        assert estimate_forecast_scalar(raw) == 1.0

    def test_zero_variance(self):
        raw = np.zeros(100)
        assert estimate_forecast_scalar(raw) == 1.0

    def test_nan_handling(self):
        raw = np.full(50, np.nan)
        raw[10:] = np.random.default_rng(3).normal(0, 1, size=40)
        scalar = estimate_forecast_scalar(raw)
        assert np.isfinite(scalar) and scalar > 0


class TestCapForecast:
    def test_clips(self):
        fc = np.array([-30.0, -10.0, 0.0, 10.0, 30.0])
        result = cap_forecast(fc, cap=20.0)
        np.testing.assert_array_equal(result, [-20.0, -10.0, 0.0, 10.0, 20.0])


class TestLongOnlyForecast:
    def test_no_negatives(self):
        fc = np.array([-15.0, -5.0, 0.0, 5.0, 15.0, 25.0])
        result = long_only_forecast(fc, cap=20.0)
        assert result.min() >= 0.0
        assert result.max() <= 20.0
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 5.0, 15.0, 20.0])


# ── Suite generators ──────────────────────────────────────────────────

class TestEWMACSuite:
    def test_default_speeds(self):
        close = _trending_up(300)
        suite = ewmac_suite(close, long_only=True)
        assert "ewmac_8_32" in suite
        assert "ewmac_64_256" in suite
        assert len(suite) == 4

    def test_long_only_no_negatives(self):
        close = _flat_market(300)
        suite = ewmac_suite(close, long_only=True)
        for name, fc in suite.items():
            valid = fc[np.isfinite(fc)]
            assert valid.min() >= 0.0, f"{name} has negative values in long-only mode"

    def test_long_short(self):
        close = _trending_down(300)
        suite = ewmac_suite(close, long_only=False)
        has_neg = False
        for fc in suite.values():
            valid = fc[np.isfinite(fc)]
            if valid.min() < 0:
                has_neg = True
        assert has_neg, "Long/short EWMAC should have negatives in downtrend"

    def test_custom_pairs(self):
        close = _trending_up(500)
        suite = ewmac_suite(close, pairs=[(2, 8), (4, 16)])
        assert len(suite) == 2
        assert "ewmac_2_8" in suite


class TestBreakoutSuite:
    def test_default_lookbacks(self):
        close = _trending_up(300)
        high = close * 1.005
        low = close * 0.995
        suite = breakout_suite(close, high, low, long_only=True)
        assert "breakout_20" in suite
        assert "breakout_160" in suite
        assert len(suite) == 4

    def test_custom_lookbacks(self):
        close = _trending_up(500)
        high = close * 1.005
        low = close * 0.995
        suite = breakout_suite(close, high, low, lookbacks=[10, 50], long_only=False)
        assert len(suite) == 2
        assert "breakout_10" in suite
