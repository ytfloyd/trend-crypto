import numpy as np
import pandas as pd
import pytest

from scripts.research.tearsheet_common_v0 import scale_equity_to_start


def test_scale_equity_to_start_basic():
    series = pd.Series([1.0, 1.1, 1.05, 1.2], index=pd.date_range("2020-01-01", periods=4))
    scaled = scale_equity_to_start(series, 100000.0)
    assert scaled.iloc[0] == 100000.0
    assert scaled.iloc[1] == 110000.0
    assert scaled.iloc[2] == 105000.0
    assert scaled.iloc[3] == 120000.0


def test_scale_equity_to_start_preserves_shape():
    series = pd.Series([1.0, 2.0, 0.5], index=pd.date_range("2020-01-01", periods=3))
    scaled = scale_equity_to_start(series, 50.0)
    assert scaled.iloc[0] == 50.0
    assert scaled.iloc[1] == 100.0
    assert scaled.iloc[2] == 25.0


def test_scale_equity_to_start_empty_series():
    series = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Cannot scale empty series"):
        scale_equity_to_start(series, 100.0)


def test_scale_equity_to_start_zero_first_value():
    series = pd.Series([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="Cannot scale series with first value"):
        scale_equity_to_start(series, 100.0)


def test_scale_equity_to_start_nan_first_value():
    series = pd.Series([np.nan, 1.0, 2.0])
    with pytest.raises(ValueError, match="Cannot scale series with first value"):
        scale_equity_to_start(series, 100.0)
