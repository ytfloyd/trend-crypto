"""Tests for Phase 5: Research Infrastructure.

Covers:
- ExperimentTracker: start/log/finish lifecycle, leaderboard
- AlphaPipeline: IC computation, hit rate, evaluate_alpha
- Research API: quick_backtest, quick_sweep
- ParameterOptimizer: walk-forward splits, deflated Sharpe, optimization
"""
from __future__ import annotations

import math
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from research.alpha_pipeline import (
    AlphaResult,
    compute_hit_rate,
    compute_ic,
    compute_signal_turnover,
    evaluate_alpha,
)
from research.api import quick_backtest, quick_sweep
from research.experiment import ExperimentRun, ExperimentTracker
from research.optimizer import (
    ParameterOptimizer,
    WalkForwardSplit,
    deflated_sharpe_ratio,
    walk_forward_splits,
)


def _make_bars(n: int = 200, base_price: float = 1000.0) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = base_price
    for i in range(n):
        ts = start + timedelta(hours=i)
        delta = 0.5 * (1 if (i * 7 + 3) % 5 < 3 else -1)
        o = price
        c = price + delta
        rows.append({
            "ts": ts, "symbol": "BTC-USD",
            "open": o, "high": max(o, c) + 0.1,
            "low": min(o, c) - 0.1, "close": c,
            "volume": 1000.0 + i * 10.0,
        })
        price = c
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------


class TestExperimentTracker:
    def setup_method(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_start_and_finish(self) -> None:
        tracker = ExperimentTracker(self.tmp_dir)
        run = tracker.start_run(run_name="test_run", params={"fast": 5})
        assert run.status == "running"
        assert run.run_name == "test_run"
        finished = tracker.finish_run()
        assert finished.status == "completed"
        assert finished.finished_at is not None

    def test_log_metrics(self) -> None:
        tracker = ExperimentTracker(self.tmp_dir)
        tracker.start_run(run_name="metrics_test")
        tracker.log_metrics({"sharpe": 1.5, "max_drawdown": -0.1})
        run = tracker.finish_run()
        assert run.metrics["sharpe"] == 1.5

    def test_log_params(self) -> None:
        tracker = ExperimentTracker(self.tmp_dir)
        tracker.start_run()
        tracker.log_params({"slow": 20, "fee_bps": 10.0})
        run = tracker.finish_run()
        assert run.params["slow"] == 20

    def test_log_artifact(self) -> None:
        tracker = ExperimentTracker(self.tmp_dir)
        tracker.start_run()
        tracker.log_artifact("/path/to/chart.png")
        run = tracker.finish_run()
        assert "/path/to/chart.png" in run.artifacts

    def test_load_run(self) -> None:
        tracker = ExperimentTracker(self.tmp_dir)
        run = tracker.start_run(run_name="load_test")
        tracker.finish_run()
        loaded = tracker.load_run(run.run_id)
        assert loaded is not None
        assert loaded.run_name == "load_test"

    def test_list_runs(self) -> None:
        tracker = ExperimentTracker(self.tmp_dir)
        for i in range(3):
            tracker.start_run(run_name=f"run_{i}")
            tracker.finish_run()
        runs = tracker.list_runs()
        assert len(runs) == 3

    def test_leaderboard(self) -> None:
        tracker = ExperimentTracker(self.tmp_dir)
        for sharpe in [0.5, 1.5, 1.0]:
            tracker.start_run()
            tracker.log_metrics({"sharpe": sharpe})
            tracker.finish_run()
        top = tracker.leaderboard(metric="sharpe", top_n=2)
        assert len(top) == 2
        assert top[0].metrics["sharpe"] == 1.5

    def test_no_active_run_raises(self) -> None:
        tracker = ExperimentTracker(self.tmp_dir)
        with pytest.raises(RuntimeError):
            tracker.log_metrics({"sharpe": 1.0})

    def test_round_trip_persistence(self) -> None:
        tracker1 = ExperimentTracker(self.tmp_dir)
        run = tracker1.start_run(params={"x": 42})
        tracker1.log_metrics({"y": 3.14})
        tracker1.finish_run()

        # New tracker instance reads from disk
        tracker2 = ExperimentTracker(self.tmp_dir)
        loaded = tracker2.load_run(run.run_id)
        assert loaded is not None
        assert loaded.params["x"] == 42
        assert loaded.metrics["y"] == 3.14


# ---------------------------------------------------------------------------
# Alpha Pipeline
# ---------------------------------------------------------------------------


class TestAlphaPipeline:
    def test_compute_ic_positive(self) -> None:
        signal = pl.Series("sig", list(range(100)))
        returns = pl.Series("ret", [float(x) * 0.01 for x in range(100)])
        ic, ic_std = compute_ic(signal, returns)
        assert ic > 0.5  # Strong positive IC for perfectly correlated

    def test_compute_ic_insufficient_data(self) -> None:
        ic, _ = compute_ic(pl.Series("s", [1.0]), pl.Series("r", [0.01]))
        assert ic == 0.0

    def test_hit_rate_perfect(self) -> None:
        signal = pl.Series("sig", [1.0, -1.0, 1.0, -1.0])
        returns = pl.Series("ret", [0.01, -0.01, 0.02, -0.02])
        hr = compute_hit_rate(signal, returns)
        assert abs(hr - 1.0) < 1e-10

    def test_hit_rate_zero(self) -> None:
        signal = pl.Series("sig", [1.0, -1.0, 1.0])
        returns = pl.Series("ret", [-0.01, 0.01, -0.01])
        hr = compute_hit_rate(signal, returns)
        assert abs(hr) < 1e-10

    def test_signal_turnover(self) -> None:
        signal = pl.Series("sig", [0.0, 1.0, 0.0, 1.0])
        to = compute_signal_turnover(signal)
        assert to == 1.0

    def test_evaluate_alpha_returns_result(self) -> None:
        n = 100
        signal = pl.Series("sig", [math.sin(i * 0.1) for i in range(n)])
        returns = pl.Series("ret", [0.001 * math.sin(i * 0.1) + 0.0001 for i in range(n)])
        result = evaluate_alpha("test_alpha", signal, returns)
        assert isinstance(result, AlphaResult)
        assert result.name == "test_alpha"
        assert result.n_bars > 0


# ---------------------------------------------------------------------------
# Research API
# ---------------------------------------------------------------------------


class TestResearchAPI:
    def test_quick_backtest_buy_and_hold(self) -> None:
        bars = _make_bars(100)
        equity_df, summary = quick_backtest(bars, strategy_mode="buy_and_hold")
        assert equity_df.height == 100
        assert "total_return" in summary

    def test_quick_backtest_ma_crossover(self) -> None:
        bars = _make_bars(100)
        equity_df, summary = quick_backtest(
            bars, strategy_mode="ma_crossover_long_only", fast=5, slow=20,
        )
        assert equity_df.height == 100
        assert "sharpe" in summary

    def test_quick_sweep_returns_dataframe(self) -> None:
        bars = _make_bars(100)
        results = quick_sweep(
            bars,
            param_grid={"fast": [3, 5], "slow": [15, 20]},
            strategy_mode="ma_crossover_long_only",
        )
        assert isinstance(results, pl.DataFrame)
        assert results.height == 4  # 2 * 2 combinations
        assert "total_return" in results.columns
        assert "sharpe" in results.columns


# ---------------------------------------------------------------------------
# Parameter Optimizer
# ---------------------------------------------------------------------------


class TestParameterOptimizer:
    def test_walk_forward_splits(self) -> None:
        splits = walk_forward_splits(n_bars=100, n_splits=5, train_frac=0.7)
        assert len(splits) == 5
        for s in splits:
            assert s.train_start <= s.train_end
            assert s.test_start <= s.test_end
            assert s.train_end < s.test_start

    def test_walk_forward_insufficient_data(self) -> None:
        splits = walk_forward_splits(n_bars=5, n_splits=5)
        assert len(splits) == 0

    def test_deflated_sharpe_ratio_bounds(self) -> None:
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0, n_trials=100, n_bars=1000,
        )
        assert 0.0 <= dsr <= 1.0

    def test_deflated_sharpe_ratio_single_trial(self) -> None:
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.0, n_trials=1, n_bars=100,
        )
        assert dsr == 1.0

    def test_optimizer_runs(self) -> None:
        bars = _make_bars(200)

        def evaluate_fn(bars_slice: pl.DataFrame, params: dict) -> float:
            # Simple metric: return the negative of fast (prefer smaller fast)
            return -float(params.get("fast", 5))

        optimizer = ParameterOptimizer(
            bars=bars,
            evaluate_fn=evaluate_fn,
            param_grid={"fast": [3, 5, 10]},
            n_splits=3,
            train_frac=0.7,
        )
        result = optimizer.optimize()
        assert result.n_trials == 3
        assert result.best_params["fast"] == 3  # -3 > -5 > -10
        assert result.deflated_sharpe is not None
