"""Tests that _summary_stats produces correct, analytically verifiable results.

These tests use hand-constructed equity curves with known properties so
every output can be verified against a manual calculation. No tolerance
for "close enough" — these must be exact to floating point precision.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from backtest.engine import _summary_stats


def _make_equity(
    daily_returns: list[float],
    start_nav: float = 100_000.0,
    start_date: datetime = datetime(2020, 1, 1),
    freq_hours: int = 24,
) -> pl.DataFrame:
    """Build an equity DataFrame from a list of daily returns."""
    nav = [start_nav]
    for r in daily_returns:
        nav.append(nav[-1] * (1 + r))
    n = len(nav)
    ts = [start_date + timedelta(hours=freq_hours * i) for i in range(n)]
    return pl.DataFrame({"ts": ts, "nav": nav})


class TestSharpeCalculation:

    def test_constant_positive_return_daily(self):
        """1% daily return for 100 days. Std = 0, Sharpe should be 0 (or inf)."""
        eq = _make_equity([0.01] * 100, freq_hours=24)
        stats = _summary_stats(eq)
        # Constant returns → std = 0 → Sharpe should be 0 (division guard)
        assert stats["sharpe"] == 0.0

    def test_known_sharpe_daily(self):
        """Construct returns with known mean and std, verify Sharpe.

        The engine recomputes returns via pct_change() on the NAV curve.
        Due to compounding, pct_change(NAV) ≈ input returns but not identical.
        We verify against pct_change(NAV) to test the engine's own math.
        """
        rng = np.random.default_rng(42)
        n = 1000
        rets = rng.normal(0.0005, 0.01, n).tolist()
        eq = _make_equity(rets, freq_hours=24)
        stats = _summary_stats(eq)

        # Recompute expected Sharpe the same way the engine should
        nav = eq["nav"].to_numpy()
        nav_rets = np.diff(nav) / nav[:-1]
        expected_sharpe = (np.mean(nav_rets) / np.std(nav_rets, ddof=1)) * np.sqrt(365)
        assert abs(stats["sharpe"] - expected_sharpe) < 1e-8, (
            f"Expected Sharpe {expected_sharpe:.10f}, got {stats['sharpe']:.10f}"
        )

    def test_known_sharpe_hourly(self):
        """Hourly data should annualize with sqrt(8760)."""
        rng = np.random.default_rng(7)
        n = 2000
        rets = rng.normal(0.00005, 0.002, n).tolist()
        eq = _make_equity(rets, freq_hours=1)
        stats = _summary_stats(eq)

        nav = eq["nav"].to_numpy()
        nav_rets = np.diff(nav) / nav[:-1]
        expected_sharpe = (np.mean(nav_rets) / np.std(nav_rets, ddof=1)) * np.sqrt(8760)
        assert abs(stats["sharpe"] - expected_sharpe) < 1e-8, (
            f"Expected Sharpe {expected_sharpe:.10f}, got {stats['sharpe']:.10f}"
        )

    def test_annualization_daily_is_365(self):
        """Explicitly verify that daily data uses 365 periods/year."""
        eq = _make_equity([0.01, -0.005, 0.003] * 100, freq_hours=24)
        stats = _summary_stats(eq)

        nav = eq["nav"].to_numpy()
        rets = np.diff(nav) / nav[:-1]
        manual_sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(365)
        assert abs(stats["sharpe"] - manual_sharpe) < 1e-8

    def test_annualization_hourly_is_8760(self):
        """Explicitly verify that hourly data uses 8760 periods/year."""
        eq = _make_equity([0.001, -0.0005] * 500, freq_hours=1)
        stats = _summary_stats(eq)

        nav = eq["nav"].to_numpy()
        rets = np.diff(nav) / nav[:-1]
        manual_sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(8760)
        assert abs(stats["sharpe"] - manual_sharpe) < 1e-8


class TestTotalReturn:

    def test_exact_total_return(self):
        """100k → 200k = 100% return."""
        eq = _make_equity([1.0], start_nav=100_000.0)  # one 100% return day
        stats = _summary_stats(eq)
        assert abs(stats["total_return"] - 1.0) < 1e-10

    def test_zero_return(self):
        eq = _make_equity([0.0] * 50, start_nav=100_000.0)
        stats = _summary_stats(eq)
        assert abs(stats["total_return"]) < 1e-10

    def test_compounding(self):
        """10 days of +10% = 1.1^10 - 1 = 159.37%"""
        eq = _make_equity([0.10] * 10, start_nav=100_000.0)
        stats = _summary_stats(eq)
        expected = 1.1 ** 10 - 1.0
        assert abs(stats["total_return"] - expected) < 1e-8


class TestMaxDrawdown:

    def test_no_drawdown(self):
        """Monotonically increasing NAV has 0 drawdown."""
        eq = _make_equity([0.01] * 50)
        stats = _summary_stats(eq)
        assert stats["max_drawdown"] == 0.0

    def test_known_drawdown(self):
        """100k → 120k → 90k. DD from peak = (90-120)/120 = -25%"""
        eq = _make_equity([0.20, -0.25])
        stats = _summary_stats(eq)
        assert abs(stats["max_drawdown"] - (-0.25)) < 1e-10

    def test_drawdown_recovery(self):
        """Drop then recover — max DD should be the trough, not current."""
        eq = _make_equity([0.10, -0.30, 0.50])
        stats = _summary_stats(eq)
        # Peak after +10%: 110k. After -30%: 77k. DD = (77-110)/110 = -30%
        # After +50%: 115.5k. Still below peak of 110? No, 115.5 > 110.
        # Max DD was -30% at the trough.
        expected_dd = -0.30
        assert abs(stats["max_drawdown"] - expected_dd) < 1e-10


class TestEdgeCases:

    def test_empty_equity(self):
        eq = pl.DataFrame({"ts": [], "nav": []}).cast({"ts": pl.Datetime, "nav": pl.Float64})
        stats = _summary_stats(eq)
        assert stats["total_return"] == 0.0
        assert stats["sharpe"] == 0.0
        assert stats["max_drawdown"] == 0.0

    def test_single_row(self):
        eq = pl.DataFrame({
            "ts": [datetime(2020, 1, 1)],
            "nav": [100_000.0],
        })
        stats = _summary_stats(eq)
        assert stats["total_return"] == 0.0
        assert stats["sharpe"] == 0.0
