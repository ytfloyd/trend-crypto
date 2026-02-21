"""Tests for the canonical metrics module (src/common/metrics.py).

Every assertion is against a hand-computed expected value.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from common.metrics import (
    compute_sharpe,
    equity_metrics,
    infer_periods_per_year,
)


def _make_equity(
    daily_returns: list[float],
    start_nav: float = 100_000.0,
    freq_hours: int = 24,
) -> pl.DataFrame:
    nav = [start_nav]
    for r in daily_returns:
        nav.append(nav[-1] * (1 + r))
    n = len(nav)
    ts = [datetime(2020, 1, 1) + timedelta(hours=freq_hours * i) for i in range(n)]
    return pl.DataFrame({"ts": ts, "nav": nav})


class TestInferPeriodsPerYear:
    def test_daily(self):
        eq = _make_equity([0.01] * 10, freq_hours=24)
        assert abs(infer_periods_per_year(eq) - 365.0) < 0.01

    def test_hourly(self):
        eq = _make_equity([0.001] * 10, freq_hours=1)
        assert abs(infer_periods_per_year(eq) - 8760.0) < 0.01


class TestComputeSharpe:
    def test_known_value(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0005, 0.01, 1000)
        series = pl.Series("r", rets.tolist())
        sharpe = compute_sharpe(series, 365.0)
        expected = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(365)
        assert abs(sharpe - expected) < 1e-8

    def test_constant_returns(self):
        series = pl.Series("r", [0.01] * 100)
        assert compute_sharpe(series, 365.0) == 0.0

    def test_single_return(self):
        series = pl.Series("r", [0.05])
        assert compute_sharpe(series, 365.0) == 0.0


class TestEquityMetrics:
    def test_basic(self):
        rets = [0.01, -0.005, 0.003] * 100
        eq = _make_equity(rets, freq_hours=24)
        m = equity_metrics(eq)
        assert abs(m["total_return"] - (eq["nav"][-1] / eq["nav"][0] - 1)) < 1e-8
        assert m["sharpe"] != 0.0
        assert m["max_drawdown"] <= 0.0

    def test_with_return_col(self):
        rets = [0.01, -0.005, 0.003] * 100
        eq = _make_equity(rets, freq_hours=24)
        eq = eq.with_columns(
            pl.Series("net_ret", [0.0] + rets)
        )
        m = equity_metrics(eq, return_col="net_ret")
        # Should skip the first 0.0 return
        nav = eq["nav"].to_numpy()
        nav_rets = np.diff(nav) / nav[:-1]
        expected_sharpe = (np.mean(nav_rets) / np.std(nav_rets, ddof=1)) * np.sqrt(365)
        assert abs(m["sharpe"] - expected_sharpe) < 0.01

    def test_consistent_with_engine(self):
        """Verify equity_metrics matches _summary_stats from engine."""
        from backtest.engine import _summary_stats

        rng = np.random.default_rng(99)
        rets = rng.normal(0.0003, 0.008, 500).tolist()
        eq = _make_equity(rets, freq_hours=24)

        engine_stats = _summary_stats(eq)
        canonical = equity_metrics(eq)

        assert abs(engine_stats["sharpe"] - canonical["sharpe"]) < 1e-6
        assert abs(engine_stats["total_return"] - canonical["total_return"]) < 1e-10
        assert abs(engine_stats["max_drawdown"] - canonical["max_drawdown"]) < 1e-10
