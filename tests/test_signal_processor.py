"""Tests for SignalProcessor — Phase 1 acceptance criteria.

CTO spec acceptance criteria:
✓ Unit test: SignalProcessor with all transforms disabled returns input unchanged
  to floating point precision.
✓ Unit test: Normalized output has mean ≈ 0 and std ≈ 1 cross-sectionally
  at each timestep.
✓ Unit test: EMA output is always between prior EMA and new input (no overshoot).
✓ Unit test: Winsorized output has no values beyond ±threshold std devs.
✓ No change to backtest results for any alpha with process_signals=False.
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from alphas.signal_processor import (
    SignalProcessor,
    SignalProcessorConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_wide_signal(n_dates: int = 50, n_symbols: int = 8, seed: int = 42) -> pl.DataFrame:
    """Create a wide-format signal DataFrame with known properties."""
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=n_dates - 1), eager=True)
    symbols = [f"SYM_{i:02d}" for i in range(n_symbols)]
    data = {"ts": dates}
    for sym in symbols:
        data[sym] = rng.standard_normal(n_dates) * 2.0 + rng.uniform(-1, 1)
    return pl.DataFrame(data)


def _make_extreme_signal(n_dates: int = 50, n_symbols: int = 8) -> pl.DataFrame:
    """Create a signal with extreme outliers."""
    import numpy as np
    rng = np.random.default_rng(99)
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=n_dates - 1), eager=True)
    symbols = [f"SYM_{i:02d}" for i in range(n_symbols)]
    data = {"ts": dates}
    for i, sym in enumerate(symbols):
        vals = rng.standard_normal(n_dates)
        vals[0] = 20.0 if i % 2 == 0 else -15.0
        vals[-1] = -25.0 if i % 2 == 0 else 18.0
        data[sym] = vals
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# AC1: Passthrough — disabled processor returns input unchanged
# ---------------------------------------------------------------------------

class TestPassthrough:

    def test_all_disabled_returns_identical_output(self):
        """With all transforms off, output must be bit-for-bit identical."""
        df = _make_wide_signal()
        proc = SignalProcessor(SignalProcessorConfig(
            normalize=False, ema_halflife=None, winsor_threshold=None,
        ))
        result = proc.process_wide(df)
        assert result.equals(df), "Passthrough should return identical DataFrame"

    def test_default_config_is_passthrough(self):
        """Default config should be passthrough."""
        cfg = SignalProcessorConfig()
        assert cfg.is_passthrough

    def test_passthrough_flag_false_when_normalize_on(self):
        cfg = SignalProcessorConfig(normalize=True)
        assert not cfg.is_passthrough

    def test_passthrough_flag_false_when_ema_on(self):
        cfg = SignalProcessorConfig(ema_halflife=4.0)
        assert not cfg.is_passthrough

    def test_passthrough_flag_false_when_winsor_on(self):
        cfg = SignalProcessorConfig(winsor_threshold=3.0)
        assert not cfg.is_passthrough

    def test_empty_dataframe_passthrough(self):
        df = pl.DataFrame({"ts": pl.Series([], dtype=pl.Date)})
        proc = SignalProcessor(SignalProcessorConfig(normalize=True))
        result = proc.process_wide(df)
        assert result.shape == df.shape


# ---------------------------------------------------------------------------
# AC2: Normalization — cross-sectional mean ≈ 0, std ≈ 1
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_normalized_mean_near_zero(self):
        df = _make_wide_signal(n_dates=100, n_symbols=10)
        proc = SignalProcessor(SignalProcessorConfig(normalize=True))
        result = proc.process_wide(df)

        signal_cols = [c for c in result.columns if c != "ts"]
        for row_idx in range(result.height):
            row_vals = [result[c][row_idx] for c in signal_cols]
            row_vals = [v for v in row_vals if v is not None]
            if len(row_vals) < 2:
                continue
            mean = sum(row_vals) / len(row_vals)
            assert abs(mean) < 1e-6, f"Row {row_idx}: mean={mean}, expected ~0"

    def test_normalized_std_near_one(self):
        df = _make_wide_signal(n_dates=100, n_symbols=10)
        proc = SignalProcessor(SignalProcessorConfig(normalize=True))
        result = proc.process_wide(df)

        signal_cols = [c for c in result.columns if c != "ts"]
        for row_idx in range(result.height):
            row_vals = [result[c][row_idx] for c in signal_cols]
            row_vals = [v for v in row_vals if v is not None]
            if len(row_vals) < 3:
                continue
            mean = sum(row_vals) / len(row_vals)
            var = sum((v - mean) ** 2 for v in row_vals) / (len(row_vals) - 1)
            std = var ** 0.5
            assert abs(std - 1.0) < 0.05, f"Row {row_idx}: std={std:.4f}, expected ~1.0"

    def test_single_symbol_normalization_no_crash(self):
        """Single symbol should not crash; std is zero so output should be safe."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), eager=True)
        df = pl.DataFrame({"ts": dates, "A": [1.0] * len(dates)})
        proc = SignalProcessor(SignalProcessorConfig(normalize=True))
        result = proc.process_wide(df)
        assert result.height == df.height


# ---------------------------------------------------------------------------
# AC3: EMA — output always between prior EMA and new input (no overshoot)
# ---------------------------------------------------------------------------

class TestEMASmoothing:

    def test_ema_no_overshoot(self):
        """EMA output should be between its previous value and the raw input."""
        df = _make_wide_signal(n_dates=100, n_symbols=5, seed=7)
        proc = SignalProcessor(SignalProcessorConfig(ema_halflife=4.0))
        result = proc.process_wide(df)

        signal_cols = [c for c in df.columns if c != "ts"]
        for c in signal_cols:
            raw = df[c].to_list()
            smoothed = result[c].to_list()
            for i in range(1, len(raw)):
                prev_ema = smoothed[i - 1]
                new_val = raw[i]
                cur_ema = smoothed[i]
                lo = min(prev_ema, new_val)
                hi = max(prev_ema, new_val)
                assert lo - 1e-10 <= cur_ema <= hi + 1e-10, (
                    f"EMA overshoot at {c}[{i}]: prev_ema={prev_ema:.4f}, "
                    f"raw={new_val:.4f}, ema={cur_ema:.4f}"
                )

    def test_ema_reduces_variance(self):
        """EMA-smoothed signal should have lower variance than raw."""
        import numpy as np
        df = _make_wide_signal(n_dates=200, n_symbols=5)
        proc = SignalProcessor(SignalProcessorConfig(ema_halflife=8.0))
        result = proc.process_wide(df)

        for c in [col for col in df.columns if col != "ts"]:
            raw_var = np.var(df[c].to_numpy())
            smooth_var = np.var(result[c].to_numpy())
            assert smooth_var < raw_var, f"EMA should reduce variance for {c}"

    def test_ema_first_value_equals_raw(self):
        """With min_periods=1, the first EMA value should equal the raw value."""
        df = _make_wide_signal(n_dates=20, n_symbols=3)
        proc = SignalProcessor(SignalProcessorConfig(ema_halflife=4.0))
        result = proc.process_wide(df)
        for c in [col for col in df.columns if col != "ts"]:
            assert abs(result[c][0] - df[c][0]) < 1e-10


# ---------------------------------------------------------------------------
# AC4: Winsorization — no values beyond ±threshold
# ---------------------------------------------------------------------------

class TestWinsorization:

    def test_no_values_beyond_threshold(self):
        threshold = 3.0
        df = _make_extreme_signal()
        proc = SignalProcessor(SignalProcessorConfig(winsor_threshold=threshold))
        result = proc.process_wide(df)

        signal_cols = [c for c in result.columns if c != "ts"]
        for c in signal_cols:
            vals = result[c].to_list()
            for i, v in enumerate(vals):
                if v is not None:
                    assert abs(v) <= threshold + 1e-10, (
                        f"Winsorization failed: {c}[{i}]={v}, threshold={threshold}"
                    )

    def test_values_within_threshold_unchanged(self):
        """Values already within bounds should not change."""
        import numpy as np
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), eager=True)
        vals = [0.5, -0.3, 1.2, -1.1, 0.0, 0.8, -0.5, 0.1, 0.9, -0.2]
        df = pl.DataFrame({"ts": dates, "A": vals})
        proc = SignalProcessor(SignalProcessorConfig(winsor_threshold=4.0))
        result = proc.process_wide(df)
        for i in range(len(vals)):
            assert abs(result["A"][i] - vals[i]) < 1e-10


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

class TestCombinedPipeline:

    def test_all_transforms_run_without_error(self):
        df = _make_extreme_signal(n_dates=100, n_symbols=10)
        proc = SignalProcessor(SignalProcessorConfig(
            normalize=True, ema_halflife=4.0, winsor_threshold=4.0,
        ))
        result = proc.process_wide(df)
        assert result.height == df.height
        assert result.width == df.width

    def test_long_format_roundtrip(self):
        """Long format processing should produce valid output."""
        wide = _make_wide_signal(n_dates=30, n_symbols=5)
        signal_cols = [c for c in wide.columns if c != "ts"]
        long_df = wide.unpivot(
            index="ts", on=signal_cols,
            variable_name="symbol", value_name="signal",
        )
        proc = SignalProcessor(SignalProcessorConfig(normalize=True))
        result = proc.process_long(long_df)
        assert "ts" in result.columns
        assert "symbol" in result.columns
        assert "signal" in result.columns
        assert result.height > 0

    def test_order_is_normalize_then_winsor_then_ema(self):
        """Normalization happens first, then winsorization clips, then EMA smooths.
        After normalization + winsorization at threshold=2, no raw normalized
        values exceed ±2. EMA should further reduce range."""
        df = _make_extreme_signal(n_dates=100, n_symbols=8)
        proc = SignalProcessor(SignalProcessorConfig(
            normalize=True, ema_halflife=4.0, winsor_threshold=2.0,
        ))
        result = proc.process_wide(df)
        signal_cols = [c for c in result.columns if c != "ts"]
        for c in signal_cols:
            max_abs = result[c].abs().max()
            assert max_abs <= 2.0 + 1e-6, (
                f"After norm+winsor(2)+ema, max abs should be <= 2.0, got {max_abs}"
            )
