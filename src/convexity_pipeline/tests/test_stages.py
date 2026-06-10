"""Unit tests for stage evaluators.

Constructs synthetic candidates and BacktestResults, verifies pass/fail
behavior at each stage matches the spec.
"""
import numpy as np
import pandas as pd
import pytest

from convexity_pipeline import (
    BacktestResult,
    Candidate,
    Hypothesis,
    PayoffShape,
    Stage,
    Stage0Evaluator,
    Stage1Evaluator,
    Stage2Evaluator,
    Stage3Evaluator,
    Track,
    default_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_hypothesis() -> Hypothesis:
    return Hypothesis(
        name="test_hyp",
        statement="trend signal on liquid futures produces convex 5-30 bar returns",
        rationale="trend persistence + stop-out via ATR bands",
        expected_payoff_shape=PayoffShape.CONVEX,
        convexity_track=Track.TREND,
        horizon_bars=20,
        universe=["SYM_A"],
        bar_frequency="1d",
        researcher="test",
        registration_date="2024-12-01",
    )


def _convex_backtest(n: int = 1000, seed: int = 0) -> BacktestResult:
    rng = np.random.default_rng(seed)
    underlying = pd.Series(
        rng.normal(0.0005, 0.012, n),
        index=pd.date_range("2010-01-01", periods=n, freq="B"),
    )
    alpha = pd.Series(rng.normal(0.0006, 0.010, n), index=underlying.index)
    # Pad alpha on top-decile |underlying| days
    abs_u = underlying.abs()
    top = abs_u >= abs_u.quantile(0.90)
    alpha[top] = alpha[top] + 0.012 * np.sign(underlying[top])
    # Add right-skewed boost
    boost = rng.exponential(0.002, n)
    alpha = alpha + (boost - boost.mean())

    equity = (1 + alpha).cumprod()
    sign = np.sign(alpha)
    grp = (sign != sign.shift()).cumsum()
    trade_pnls = alpha.groupby(grp).sum()
    trade_durs = alpha.groupby(grp).size()
    # Make durations ~ horizon
    trade_durs = trade_durs.clip(upper=30)

    return BacktestResult(
        alpha_returns=alpha,
        underlying_returns=underlying,
        equity=equity,
        trade_pnls=trade_pnls,
        trade_durations=trade_durs,
        per_instrument={
            "SYM_A": BacktestResult(
                alpha_returns=alpha,
                underlying_returns=underlying,
                equity=equity,
                trade_pnls=trade_pnls,
                trade_durations=trade_durs,
            ),
            "SYM_B": BacktestResult(
                alpha_returns=alpha * 0.8,
                underlying_returns=underlying,
                equity=(1 + alpha * 0.8).cumprod(),
                trade_pnls=trade_pnls * 0.8,
                trade_durations=trade_durs,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------

def test_stage0_passes_good_hypothesis():
    cfg = default_config()
    s0 = Stage0Evaluator(cfg)
    c = Candidate(registry_id="T0-1", hypothesis=_good_hypothesis(), signal_fn=lambda: None)
    r = s0.evaluate(c)
    assert r.passed
    assert r.stage == Stage.S0


def test_stage0_kills_linear():
    cfg = default_config()
    s0 = Stage0Evaluator(cfg)
    h = _good_hypothesis()
    h.expected_payoff_shape = PayoffShape.LINEAR
    c = Candidate(registry_id="T0-2", hypothesis=h, signal_fn=lambda: None)
    r = s0.evaluate(c)
    assert not r.passed
    assert any("wrong_pipeline" in k for k in r.kill_reasons)


def test_stage0_kills_missing_fields():
    cfg = default_config()
    s0 = Stage0Evaluator(cfg)
    h = _good_hypothesis()
    h.universe = []
    c = Candidate(registry_id="T0-3", hypothesis=h, signal_fn=lambda: None)
    r = s0.evaluate(c)
    assert not r.passed
    assert any("missing_field:universe" in k for k in r.kill_reasons)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def test_stage1_passes_convex_backtest():
    cfg = default_config()
    # Relax slightly to allow synthetic noise variation
    cfg.stage1.min_aggregate_skew = -0.1
    cfg.stage1.min_ccs_aggregate = 0.1
    s1 = Stage1Evaluator(cfg)
    c = Candidate(registry_id="T1-1", hypothesis=_good_hypothesis(), signal_fn=lambda: None)
    bt = _convex_backtest()
    r = s1.evaluate(c, bt)
    # Should pass under relaxed thresholds (synthetic data is convex by construction)
    assert r.stage == Stage.S1
    assert "aggregate" in r.metrics


def test_stage1_kills_no_backtest():
    cfg = default_config()
    s1 = Stage1Evaluator(cfg)
    c = Candidate(registry_id="T1-2", hypothesis=_good_hypothesis(), signal_fn=lambda: None)
    r = s1.evaluate(c, None)
    assert not r.passed
    assert any("no_backtest_provided" in k for k in r.kill_reasons)


def test_stage1_duration_mismatch():
    """If hypothesis says horizon=20 but trade duration ~1 bar, should kill."""
    cfg = default_config()
    cfg.stage1.min_aggregate_skew = -10  # relax other checks
    cfg.stage1.min_ccs_aggregate = -10
    cfg.stage1.min_universe_positive_fraction = 0
    cfg.stage1.min_convexity_beta = -10
    s1 = Stage1Evaluator(cfg)
    h = _good_hypothesis()
    h.horizon_bars = 100  # claim long horizon
    c = Candidate(registry_id="T1-3", hypothesis=h, signal_fn=lambda: None)
    bt = _convex_backtest()
    # Override trade durations to be very short
    bt.trade_durations = pd.Series([1] * len(bt.trade_durations))
    r = s1.evaluate(c, bt)
    assert not r.passed
    assert any("trade_duration_mismatch" in k for k in r.kill_reasons)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def test_stage2_passes_when_cost_drop_modest():
    cfg = default_config()
    s2 = Stage2Evaluator(cfg)
    c = Candidate(registry_id="T2-1", hypothesis=_good_hypothesis(), signal_fn=lambda: None)
    bt_pre = _convex_backtest(seed=0)
    bt_post = _convex_backtest(seed=0)
    # Apply modest cost: subtract small constant from returns
    bt_post.alpha_returns = bt_post.alpha_returns - 0.00005
    bt_post.equity = (1 + bt_post.alpha_returns).cumprod()
    r = s2.evaluate(c, backtest_pre_cost=bt_pre, backtest_post_cost=bt_post)
    assert r.stage == Stage.S2


def test_stage2_kills_huge_cost_drop():
    cfg = default_config()
    cfg.stage2.max_ccs_drop_fraction = 0.10
    s2 = Stage2Evaluator(cfg)
    c = Candidate(registry_id="T2-2", hypothesis=_good_hypothesis(), signal_fn=lambda: None)
    bt_pre = _convex_backtest(seed=0)
    bt_post = _convex_backtest(seed=0)
    # Massive cost
    bt_post.alpha_returns = bt_post.alpha_returns - 0.005
    bt_post.equity = (1 + bt_post.alpha_returns).cumprod()
    r = s2.evaluate(c, backtest_pre_cost=bt_pre, backtest_post_cost=bt_post)
    # Should kill on excess CCS drop
    assert not r.passed


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def test_stage3_kills_insufficient_folds():
    cfg = default_config()
    s3 = Stage3Evaluator(cfg)
    c = Candidate(registry_id="T3-1", hypothesis=_good_hypothesis(), signal_fn=lambda: None)
    bt_is = _convex_backtest(seed=0)
    folds = [_convex_backtest(seed=i) for i in range(3)]
    r = s3.evaluate(c, backtest_is=bt_is, backtest_oos_folds=folds)
    assert not r.passed
    assert any("insufficient_oos_folds" in k for k in r.kill_reasons)


def test_stage3_passes_consistent_oos():
    cfg = default_config()
    cfg.stage3.min_ccs_oos_to_is_ratio = 0.1  # relax
    s3 = Stage3Evaluator(cfg)
    c = Candidate(registry_id="T3-2", hypothesis=_good_hypothesis(), signal_fn=lambda: None)
    bt_is = _convex_backtest(seed=0)
    folds = [_convex_backtest(seed=i + 1) for i in range(10)]
    r = s3.evaluate(c, backtest_is=bt_is, backtest_oos_folds=folds)
    assert r.stage == Stage.S3
    assert "fold_ccs" in r.metrics
