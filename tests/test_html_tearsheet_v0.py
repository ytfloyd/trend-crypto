"""Tests for HTML tearsheet generation (tearsheet_common_v0)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure scripts/research is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))

from tearsheet_common_v0 import (
    _top_n_drawdown_periods,
    build_standard_html_tearsheet,
    compute_comprehensive_stats,
    compute_drawdown,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_equity(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate a synthetic equity curve with realistic-ish returns."""
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0003, 0.02, size=n)
    prices = np.cumprod(1 + daily_ret) * 1.0
    dates = pd.bdate_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="equity")


def _make_benchmark(equity: pd.Series, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0004, 0.03, size=len(equity))
    prices = np.cumprod(1 + daily_ret) * 1.0
    return pd.Series(prices, index=equity.index, name="benchmark_equity")


# ---------------------------------------------------------------------------
# tests: _top_n_drawdown_periods
# ---------------------------------------------------------------------------

class TestTopNDrawdownPeriods:
    def test_returns_list_of_dicts(self):
        eq = _make_equity()
        periods = _top_n_drawdown_periods(eq, n=5)
        assert isinstance(periods, list)
        assert len(periods) <= 5
        for p in periods:
            assert "start" in p
            assert "valley" in p
            assert "end" in p
            assert "drawdown" in p
            assert "duration_days" in p

    def test_periods_sorted_by_severity(self):
        eq = _make_equity()
        periods = _top_n_drawdown_periods(eq, n=5)
        dds = [p["drawdown"] for p in periods]
        assert dds == sorted(dds), "Periods should be sorted worst-first (most negative)"

    def test_drawdowns_are_negative(self):
        eq = _make_equity()
        periods = _top_n_drawdown_periods(eq, n=5)
        for p in periods:
            assert p["drawdown"] < 0

    def test_monotonic_equity_no_drawdowns(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        eq = pd.Series(np.arange(1, 101, dtype=float), index=dates)
        periods = _top_n_drawdown_periods(eq, n=5)
        assert periods == []


# ---------------------------------------------------------------------------
# tests: compute_comprehensive_stats
# ---------------------------------------------------------------------------

class TestComputeComprehensiveStats:
    def test_returns_all_expected_keys(self):
        eq = _make_equity()
        stats = compute_comprehensive_stats(eq)
        expected_keys = [
            "total_return", "cagr", "best_year", "worst_year",
            "best_month", "worst_month", "best_day", "worst_day",
            "vol", "downside_vol", "max_dd", "avg_dd", "max_dd_duration",
            "sharpe", "sortino", "calmar",
            "omega", "stability", "tail_ratio",
            "var_95", "cvar_95", "skewness", "kurtosis", "alpha",
            "beta", "correlation", "information_ratio",
            "win_rate", "profit_factor", "win_loss_ratio",
            "expectancy", "autocorrelation", "turnover_proxy",
            "n_days", "start", "end", "drawdown_periods",
            "benchmark_stats",
        ]
        for k in expected_keys:
            assert k in stats, f"Missing key: {k}"

    def test_n_days_matches_equity_length(self):
        eq = _make_equity(n=300)
        stats = compute_comprehensive_stats(eq)
        assert stats["n_days"] == 300

    def test_cagr_positive_for_uptrend(self):
        dates = pd.bdate_range("2020-01-01", periods=252)
        eq = pd.Series(np.linspace(1.0, 2.0, 252), index=dates)
        stats = compute_comprehensive_stats(eq)
        assert stats["cagr"] > 0

    def test_max_dd_negative(self):
        eq = _make_equity()
        stats = compute_comprehensive_stats(eq)
        assert stats["max_dd"] < 0

    def test_win_rate_between_0_and_1(self):
        eq = _make_equity()
        stats = compute_comprehensive_stats(eq)
        assert 0 <= stats["win_rate"] <= 1

    def test_with_benchmark(self):
        eq = _make_equity()
        bench = _make_benchmark(eq)
        stats = compute_comprehensive_stats(eq, benchmark_equity=bench)
        assert stats["benchmark_stats"] is not None
        assert "cagr" in stats["benchmark_stats"]
        assert not np.isnan(stats["beta"])
        assert not np.isnan(stats["correlation"])
        assert not np.isnan(stats["alpha"])

    def test_without_benchmark_factor_stats_are_nan(self):
        eq = _make_equity()
        stats = compute_comprehensive_stats(eq)
        assert stats["benchmark_stats"] is None
        assert np.isnan(stats["beta"])
        assert np.isnan(stats["alpha"])

    def test_drawdown_periods_populated(self):
        eq = _make_equity()
        stats = compute_comprehensive_stats(eq)
        assert isinstance(stats["drawdown_periods"], list)
        assert len(stats["drawdown_periods"]) > 0


# ---------------------------------------------------------------------------
# tests: build_standard_html_tearsheet
# ---------------------------------------------------------------------------

class TestBuildStandardHtmlTearsheet:
    def test_generates_html_file(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        result = build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test Strategy",
            strategy_equity=eq,
            equity_csv_path="/tmp/equity.csv",
        )
        assert result == out
        assert out.exists()
        content = out.read_text()
        assert content.startswith("<!DOCTYPE html>")

    def test_contains_all_four_sections(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test Strategy",
            strategy_equity=eq,
        )
        content = out.read_text()
        assert "Performance Overview" in content
        assert "Performance Statistics" in content
        assert "Rolling Analytics" in content
        assert "Factor &amp; Trade Analysis" in content

    def test_contains_strategy_label(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="My Custom Strategy Name",
            strategy_equity=eq,
        )
        content = out.read_text()
        assert "My Custom Strategy Name" in content

    def test_contains_plotly_cdn(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
        )
        content = out.read_text()
        assert "cdn.plot.ly/plotly" in content
        assert "Plotly.newPlot" in content

    def test_contains_hero_metrics(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
        )
        content = out.read_text()
        for metric in ["CAGR", "VOLATILITY", "SHARPE", "SORTINO", "MAX DD", "CALMAR"]:
            assert metric in content, f"Missing hero metric: {metric}"

    def test_contains_chart_divs(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
        )
        content = out.read_text()
        for div_id in ["eq-chart", "dd-chart", "rs-chart", "rv-chart", "hist-chart", "yr-chart"]:
            assert f"id='{div_id}'" in content, f"Missing chart div: {div_id}"

    def test_with_benchmark(self, tmp_path: Path):
        eq = _make_equity()
        bench = _make_benchmark(eq)
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
            benchmark_equity=bench,
            benchmark_label="BTC B&H",
        )
        content = out.read_text()
        assert "Strategy vs Benchmark" in content
        assert "BTC B&amp;H" in content or "BTC B&H" in content
        assert "rb-chart" in content  # rolling beta chart present

    def test_drawdown_periods_table(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
        )
        content = out.read_text()
        assert "Worst Drawdown Periods" in content

    def test_provenance_section(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
            equity_csv_path="/data/equity.csv",
            metrics_csv_path="/data/metrics.csv",
        )
        content = out.read_text()
        assert "Data Provenance" in content
        assert "/data/equity.csv" in content

    def test_confidential_footer(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
            confidential_footer=True,
        )
        content = out.read_text()
        assert "CONFIDENTIAL" in content

    def test_no_confidential_footer(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
            confidential_footer=False,
        )
        content = out.read_text()
        assert "CONFIDENTIAL" not in content

    def test_tail_risk_metrics(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
        )
        content = out.read_text()
        assert "Tail Risk Metrics" in content
        for m in ["VaR (95%)", "CVaR (95%)", "Skewness", "Kurtosis", "Tail Ratio", "Stability"]:
            assert m in content, f"Missing tail risk metric: {m}"

    def test_monthly_heatmap_present(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
        )
        content = out.read_text()
        assert "hm-chart" in content

    def test_creates_parent_dirs(self, tmp_path: Path):
        eq = _make_equity()
        out = tmp_path / "sub" / "dir" / "tearsheet.html"
        build_standard_html_tearsheet(
            out_html=out,
            strategy_label="Test",
            strategy_equity=eq,
        )
        assert out.exists()
