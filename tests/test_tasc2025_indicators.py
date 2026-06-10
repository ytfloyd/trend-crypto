from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research" / "common"))

from tasc2025_indicators import (  # noqa: E402
    autocorr_regime_score,
    continuation_index,
    drawdown_duration,
    laguerre_filter,
    linear_regression_channel,
    no_lag_ema,
    supertrend,
    ulcer_index,
)


def test_autocorr_regime_score_separates_persistent_series() -> None:
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    rng = np.random.default_rng(42)
    persistent = pd.Series(np.tile([0.01, 0.01, -0.005, -0.005], 60), index=idx)
    randomish = pd.Series(rng.normal(0.0, 0.01, len(idx)), index=idx)

    p_score = autocorr_regime_score(persistent, window=80, min_lag=2, max_lag=8)
    r_score = autocorr_regime_score(randomish, window=80, min_lag=2, max_lag=8)

    assert p_score["ac_score"].iloc[-1] > r_score["ac_score"].iloc[-1]
    assert 0.0 <= p_score["ac_score"].iloc[-1] <= 1.0


def test_laguerre_and_continuation_index_follow_uptrend() -> None:
    close = pd.Series(np.linspace(100.0, 150.0, 120))
    lag = laguerre_filter(close, gamma=0.8)
    ci = continuation_index(close, gamma=0.8, length=10)

    assert lag.notna().sum() == len(close)
    assert lag.iloc[-1] < close.iloc[-1]
    assert ci.iloc[-1] == 1.0


def test_no_lag_ema_reduces_lag_vs_plain_ema() -> None:
    close = pd.Series(np.linspace(100.0, 150.0, 120))
    ema = close.ewm(span=20, adjust=False, min_periods=20).mean()
    nlema = no_lag_ema(close, 20)

    assert abs(close.iloc[-1] - nlema.iloc[-1]) < abs(close.iloc[-1] - ema.iloc[-1])


def test_supertrend_outputs_direction_and_bands() -> None:
    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    close = pd.Series(np.linspace(100.0, 140.0, 80), index=idx)
    df = pd.DataFrame(
        {
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        },
        index=idx,
    )
    st = supertrend(df, atr_len=13, multiplier=3.0)

    assert {"supertrend", "supertrend_dir", "st_upper", "st_lower"}.issubset(st.columns)
    assert st["supertrend_dir"].iloc[-1] == 1.0


def test_linear_regression_channel_detects_positive_slope() -> None:
    close = pd.Series(np.linspace(100.0, 130.0, 80))
    ch = linear_regression_channel(close, length=40, width=2.0)

    assert ch["lr_slope"].iloc[-1] > 0.0
    assert ch["lr_upper"].iloc[-1] > ch["lr_center"].iloc[-1] > ch["lr_lower"].iloc[-1]


def test_ulcer_and_drawdown_duration() -> None:
    nav = pd.Series([1.0, 1.2, 1.1, 1.0, 1.3, 1.25])

    assert ulcer_index(nav) > 0.0
    dd = drawdown_duration(nav)
    assert dd["max_drawdown_duration"] == 2
    assert dd["current_drawdown_duration"] == 1

