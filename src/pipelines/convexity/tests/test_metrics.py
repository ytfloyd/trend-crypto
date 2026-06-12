"""Unit tests for convexity_pipeline.metrics.

Covers: basic correctness on known inputs, edge cases (empty, NaN,
zero variance), convexity beta sign correctness.
"""
import numpy as np
import pandas as pd

from pipelines.convexity import metrics as M


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_skew_zero_on_symmetric():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 5000))
    assert abs(M.skew(s)) < 0.2


def test_skew_positive_on_right_skewed():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.exponential(1, 5000))  # positive skew
    assert M.skew(s) > 0.5


def test_skew_negative_on_left_skewed():
    rng = np.random.default_rng(2)
    s = pd.Series(-rng.exponential(1, 5000))
    assert M.skew(s) < -0.5


def test_sharpe_basic():
    r = pd.Series([0.01] * 252)  # 1% per day, zero variance
    assert M.sharpe(r) is None  # zero std -> None


def test_sharpe_positive():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    sh = M.sharpe(r)
    assert sh is not None
    assert 0 < sh < 3


def test_max_drawdown_known():
    e = pd.Series([1.0, 1.1, 1.05, 0.9, 1.0])
    dd = M.max_drawdown(e)
    assert abs(dd - (1 - 0.9 / 1.1)) < 0.01


def test_calmar_positive():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    c = M.calmar(r)
    assert c is None or c > -1.0  # well-defined


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------

def test_payoff_ratio_basic():
    p = pd.Series([1.0, -0.5, 2.0, -1.0])
    # mean win = 1.5; |mean loss| = 0.75; ratio = 2.0
    assert abs(M.payoff_ratio(p) - 2.0) < 1e-6


def test_profit_factor_basic():
    p = pd.Series([1.0, -0.5, 2.0, -1.0])
    # wins = 3.0; losses = -1.5; pf = 2.0
    assert abs(M.profit_factor(p) - 2.0) < 1e-6


def test_hit_rate():
    p = pd.Series([1.0, -0.5, 2.0, -1.0, 0.5])
    assert abs(M.hit_rate(p) - 0.6) < 1e-6


def test_max_consecutive_losses():
    p = pd.Series([1.0, -0.5, -0.5, -0.3, 0.5, -0.1, -0.2])
    # Longest losing streak: 3
    assert M.max_consecutive_losses(p) == 3


def test_max_consecutive_losses_no_losses():
    p = pd.Series([1.0, 0.5, 0.3])
    assert M.max_consecutive_losses(p) == 0


# ---------------------------------------------------------------------------
# Convexity-specific
# ---------------------------------------------------------------------------

def test_tail_capture_perfect_alignment():
    rng = np.random.default_rng(5)
    u = pd.Series(rng.normal(0, 0.01, 1000))
    # Alpha PERFECTLY captures top-decile underlying moves in correct direction
    a = pd.Series(np.zeros(1000), index=u.index)
    top_thresh = u.abs().quantile(0.90)
    top_mask = u.abs() >= top_thresh
    a[top_mask] = u[top_mask] * 1.0  # perfect capture
    tc = M.tail_capture(a, u)
    # Capture should be close to 1.0
    assert tc is not None
    assert tc > 0.8


def test_tail_capture_no_capture():
    rng = np.random.default_rng(6)
    u = pd.Series(rng.normal(0, 0.01, 1000))
    a = pd.Series(np.zeros(1000), index=u.index)
    tc = M.tail_capture(a, u)
    # No alpha returns -> capture ~ 0
    assert tc is not None
    assert tc < 0.05


def test_convexity_beta_long_vol():
    """Construct alpha = 0.5 * |underlying| - should yield positive convexity beta."""
    rng = np.random.default_rng(7)
    u = pd.Series(rng.normal(0, 0.012, 2000))
    a = 0.5 * u.abs() + rng.normal(0, 0.002, 2000)
    a.index = u.index
    b, p = M.convexity_beta(a, u)
    assert b is not None
    assert b > 0.3
    assert p < 0.01  # highly significant


def test_convexity_beta_concave():
    """Alpha = -0.5 * |underlying| should yield negative b."""
    rng = np.random.default_rng(8)
    u = pd.Series(rng.normal(0, 0.012, 2000))
    a = -0.5 * u.abs() + rng.normal(0, 0.002, 2000)
    a.index = u.index
    b, p = M.convexity_beta(a, u)
    assert b is not None
    assert b < -0.3


def test_composite_convexity_score_positive():
    rng = np.random.default_rng(9)
    n = 2000
    r = pd.Series(
        rng.normal(0.0008, 0.012, n)
        + rng.exponential(0.003, n),  # right-skew tail
        index=pd.date_range("2010-01-01", periods=n, freq="B"),
    )
    trades = pd.Series([1.5, -0.5, 2.0, -0.8, 1.2, -0.4])
    ccs = M.composite_convexity_score(returns=r, trade_pnls=trades)
    assert ccs is not None
    assert ccs > 0  # should be positive given positive-skew construction


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_returns():
    s = pd.Series(dtype=float)
    assert M.skew(s) is None
    assert M.sharpe(s) is None
    assert M.calmar(s) is None


def test_nan_handling():
    s = pd.Series([np.nan, 0.01, np.nan, 0.02, -0.01])
    sk = M.skew(s)
    # Should work on 3 non-NaN obs
    assert sk is not None


def test_zero_variance_returns():
    s = pd.Series([0.001] * 100)
    assert M.sharpe(s) is None
    assert M.calmar(s) is None or M.calmar(s) > 0  # tiny DD floor handles it


def test_payoff_ratio_no_losses():
    p = pd.Series([1.0, 0.5])
    assert M.payoff_ratio(p) is None
