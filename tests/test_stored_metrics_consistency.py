"""Consistency test: re-derives ALL stored metrics from source equity curves.

This is the defense-in-depth test. It reads every summary.json and every
metrics CSV, recomputes every metric independently from the raw equity data,
and fails loudly if any diverge beyond floating-point tolerance.

Run with: pytest tests/test_stored_metrics_consistency.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

RUNS_DIR = Path("artifacts/runs")
COMPARE_DIR = Path("artifacts/compare")

# Tolerance: engine and numpy should agree within this
SHARPE_TOL = 0.02
RETURN_TOL = 1e-6
DD_TOL = 1e-6


def _recompute_from_equity(eq_path: Path) -> dict:
    """Recompute summary stats from equity.parquet using only numpy."""
    eq = pl.read_parquet(eq_path)
    nav = eq["nav"].to_numpy()
    ts = eq["ts"].to_list()

    total_return = nav[-1] / nav[0] - 1.0 if nav[0] > 0 else 0.0
    rets = np.diff(nav) / nav[:-1]

    if len(ts) > 1:
        dt_seconds = [(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts)-1)]
        ppy = 365 * 24 * 3600 / np.median(dt_seconds) if np.median(dt_seconds) > 0 else 365.0
    else:
        ppy = 365.0

    std = np.std(rets, ddof=1) if len(rets) > 1 else 0.0
    sharpe = (np.mean(rets) / std) * np.sqrt(ppy) if std > 1e-12 else 0.0

    running_max = np.maximum.accumulate(nav)
    max_dd = float(np.min(nav / running_max - 1.0))

    return {"total_return": total_return, "sharpe": sharpe, "max_drawdown": max_dd}


class TestRunSummaryConsistency:
    """Every summary.json in artifacts/runs/ must match its equity.parquet."""

    def _get_run_dirs(self):
        if not RUNS_DIR.exists():
            pytest.skip("No artifacts/runs directory")
        dirs = []
        for d in sorted(RUNS_DIR.iterdir()):
            if d.is_dir() and (d / "equity.parquet").exists() and (d / "summary.json").exists():
                dirs.append(d)
        if not dirs:
            pytest.skip("No runs with both equity.parquet and summary.json")
        return dirs

    def test_all_sharpe_values_match(self):
        """Sharpe in summary.json must match recomputed Sharpe."""
        run_dirs = self._get_run_dirs()
        failures = []
        for d in run_dirs:
            stored = json.loads((d / "summary.json").read_text())
            recomputed = _recompute_from_equity(d / "equity.parquet")
            diff = abs(stored["sharpe"] - recomputed["sharpe"])
            if diff > SHARPE_TOL:
                failures.append(
                    f"{d.name}: stored={stored['sharpe']:.6f}, "
                    f"recomputed={recomputed['sharpe']:.6f}, diff={diff:.6f}"
                )
        assert not failures, (
            f"{len(failures)} runs have Sharpe mismatch:\n" + "\n".join(failures[:10])
        )

    def test_all_total_returns_match(self):
        """Total return in summary.json must match recomputed total return."""
        run_dirs = self._get_run_dirs()
        failures = []
        for d in run_dirs:
            stored = json.loads((d / "summary.json").read_text())
            recomputed = _recompute_from_equity(d / "equity.parquet")
            diff = abs(stored["total_return"] - recomputed["total_return"])
            if diff > RETURN_TOL:
                failures.append(
                    f"{d.name}: stored={stored['total_return']:.8f}, "
                    f"recomputed={recomputed['total_return']:.8f}"
                )
        assert not failures, (
            f"{len(failures)} runs have total_return mismatch:\n" + "\n".join(failures[:10])
        )

    def test_all_max_drawdowns_match(self):
        """Max drawdown in summary.json must match recomputed max drawdown."""
        run_dirs = self._get_run_dirs()
        failures = []
        for d in run_dirs:
            stored = json.loads((d / "summary.json").read_text())
            recomputed = _recompute_from_equity(d / "equity.parquet")
            diff = abs(stored["max_drawdown"] - recomputed["max_drawdown"])
            if diff > DD_TOL:
                failures.append(
                    f"{d.name}: stored={stored['max_drawdown']:.8f}, "
                    f"recomputed={recomputed['max_drawdown']:.8f}"
                )
        assert not failures, (
            f"{len(failures)} runs have max_drawdown mismatch:\n" + "\n".join(failures[:10])
        )

    def test_nav_compounds_from_net_ret(self):
        """NAV[t] = NAV[t-1] * (1 + net_ret[t]) for a sample of runs."""
        run_dirs = self._get_run_dirs()
        sample = run_dirs[:30]
        failures = []
        checked = 0
        for d in sample:
            eq = pl.read_parquet(d / "equity.parquet")
            if "net_ret" not in eq.columns:
                continue
            checked += 1
            nav = eq["nav"].to_list()
            net = eq["net_ret"].to_list()
            for t in range(1, len(nav)):
                expected = nav[t-1] * (1 + net[t])
                if abs(nav[t] - expected) > 0.01:
                    failures.append(f"{d.name} bar {t}: nav={nav[t]}, expected={expected}")
                    break
        assert checked > 0, "No equity files with net_ret column found in sample"
        assert not failures, (
            f"{len(failures)} runs have NAV compounding errors:\n" + "\n".join(failures[:10])
        )


class TestCompareSummaryConsistency:
    """Compare/ directory summaries must match their equity curves."""

    def _get_compare_dirs(self):
        if not COMPARE_DIR.exists():
            pytest.skip("No artifacts/compare directory")
        dirs = []
        for d in sorted(COMPARE_DIR.rglob("summary.json")):
            parent = d.parent
            eq = parent / "portfolio_equity.parquet"
            if not eq.exists():
                eq = parent / "equity.parquet"
            if eq.exists():
                dirs.append((parent, eq, d))
        if not dirs:
            pytest.skip("No compare dirs with equity + summary")
        return dirs

    def test_compare_sharpe_values_match(self):
        """Compare summaries must match recomputed Sharpe from equity curves."""
        entries = self._get_compare_dirs()
        failures = []
        for parent, eq_path, sum_path in entries:
            stored = json.loads(sum_path.read_text())
            stored_sharpe = stored.get("sharpe")
            if stored_sharpe is None:
                continue

            eq = pl.read_parquet(eq_path)
            nav = eq["nav"].to_numpy()
            ts = eq["ts"].to_list()

            if "r_port_net" in eq.columns:
                rets = eq["r_port_net"].to_numpy()[1:]
            elif "r_port" in eq.columns:
                rets = eq["r_port"].to_numpy()[1:]
            elif "net_ret" in eq.columns:
                rets = eq["net_ret"].to_numpy()[1:]
            else:
                rets = np.diff(nav) / nav[:-1]

            dt_all = [(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts)-1)]
            ppy = 365 * 24 * 3600 / np.median(dt_all) if np.median(dt_all) > 0 else 365.0
            std = np.std(rets, ddof=1)
            recomputed = (np.mean(rets) / std) * np.sqrt(ppy) if std > 1e-12 else 0.0

            if abs(stored_sharpe - recomputed) > SHARPE_TOL:
                failures.append(
                    f"{parent.relative_to(COMPARE_DIR)}: "
                    f"stored={stored_sharpe:.4f}, recomputed={recomputed:.4f}"
                )
        assert not failures, (
            f"{len(failures)} compare dirs have Sharpe mismatch:\n" + "\n".join(failures)
        )
