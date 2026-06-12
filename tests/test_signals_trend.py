"""Tests for signals.trend.ma_crossover (pure signal function)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.trend import ma_crossover


def _bars(n: int = 30) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", periods=n, freq="D")
    up = pd.DataFrame({"symbol": "AAA-USD", "ts": ts, "close": np.linspace(100, 200, n)})
    flat = pd.DataFrame({"symbol": "BBB-USD", "ts": ts, "close": np.full(n, 50.0)})
    return pd.concat([up, flat], ignore_index=True)


def test_shape_and_columns():
    w = ma_crossover(_bars(30), fast=3, slow=10)
    assert list(w.columns) == ["AAA-USD", "BBB-USD"]
    assert w.index.is_monotonic_increasing
    assert len(w) == 30


def test_long_only_equal_weight_values():
    w = ma_crossover(_bars(30), fast=3, slow=10)
    # 2-symbol universe -> active weight is 1/2; values are only {0.0, 0.5}
    assert set(np.unique(w.values)) <= {0.0, 0.5}
    # the steadily-rising symbol is long once both SMAs warm up
    assert w["AAA-USD"].iloc[-1] == pytest.approx(0.5)
    # the flat symbol never triggers (fast SMA never exceeds slow SMA)
    assert (w["BBB-USD"] == 0.0).all()


def test_fast_must_be_less_than_slow():
    with pytest.raises(ValueError, match="fast .* must be < slow"):
        ma_crossover(_bars(30), fast=10, slow=5)
