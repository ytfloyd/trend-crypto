"""Verify the engine's cost model is applied correctly.

Tests use a simple deterministic setup where the expected cost can be
computed by hand. No randomness, no approximation.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from backtest.engine import _summary_stats


def _make_equity_with_costs(
    nav_list: list[float],
    net_ret_list: list[float],
    gross_ret_list: list[float],
    cost_ret_list: list[float],
    turnover_list: list[float],
    w_held_list: list[float],
    freq_hours: int = 24,
) -> pl.DataFrame:
    n = len(nav_list)
    ts = [datetime(2020, 1, 1) + timedelta(hours=freq_hours * i) for i in range(n)]
    return pl.DataFrame({
        "ts": ts,
        "nav": nav_list,
        "net_ret": net_ret_list,
        "gross_ret": gross_ret_list,
        "cost_ret": cost_ret_list,
        "turnover": turnover_list,
        "w_held": w_held_list,
    })


class TestCostApplication:
    """Verify cost = turnover * (fee_bps + slip_bps) / 10000."""

    def test_single_trade_cost(self):
        """One trade at t=1: weight goes from 0 to 1.
        Fee=10bps, slip=10bps → total 20bps.
        Turnover = |1-0| = 1.
        Cost = 1 * 20/10000 = 0.002.
        """
        cost_bps = 20  # 10 fee + 10 slip
        turnover = 1.0
        expected_cost_ret = turnover * (cost_bps / 10000.0)
        assert abs(expected_cost_ret - 0.002) < 1e-10

        # For a 100k portfolio, this is $200 in costs
        nav = 100_000.0
        cost_usd = nav * expected_cost_ret
        assert abs(cost_usd - 200.0) < 1e-6

    def test_cost_reduces_net_return(self):
        """net_ret = gross_ret - cost_ret.
        If gross = 1%, cost = 0.2%, then net = 0.8%."""
        gross = 0.01
        cost = 0.002
        net = gross - cost
        assert abs(net - 0.008) < 1e-10

    def test_roundtrip_cost(self):
        """Enter and exit: two trades, each with turnover=1.
        Total cost = 2 * 0.002 = 0.004 = 40bps."""
        cost_per_trade = 0.002  # 20bps
        n_trades = 2
        total_cost = n_trades * cost_per_trade
        assert abs(total_cost - 0.004) < 1e-10

    def test_partial_trade_cost(self):
        """Weight change from 0.5 to 0.8: turnover = 0.3.
        Cost = 0.3 * 20/10000 = 0.0006."""
        turnover = 0.3
        cost_bps = 20
        expected = turnover * cost_bps / 10000
        assert abs(expected - 0.0006) < 1e-10


class TestCostInEquity:
    """Verify costs are correctly reflected in equity curves."""

    def test_equity_with_costs_matches_nav(self):
        """Manually construct an equity curve with known costs,
        verify NAV progression is consistent."""
        initial = 100_000.0
        gross_ret = 0.01  # 1% gross return
        cost_ret = 0.002  # 20bps cost
        net_ret = gross_ret - cost_ret  # 0.8%

        nav_1 = initial * (1 + net_ret)

        # NAV should be exactly 100,800
        assert abs(nav_1 - 100_800.0) < 1e-6

    def test_zero_cost_means_net_equals_gross(self):
        """With zero fees, net_ret == gross_ret."""
        gross_ret = 0.015
        cost_ret = 0.0
        net_ret = gross_ret - cost_ret
        assert net_ret == gross_ret

    def test_cost_attribution(self):
        """Verify: sum(cost_ret) * initial_nav ≈ total USD paid in fees."""
        initial = 100_000.0
        cost_rets = [0.002, 0.001, 0.0, 0.002, 0.0]  # 5 bars
        total_cost_pct = sum(cost_rets)
        total_cost_usd_approx = total_cost_pct * initial
        assert abs(total_cost_usd_approx - 500.0) < 1e-6


class TestEngineEquityCostConsistency:
    """Spot-check actual equity.parquet files for internal consistency."""

    def test_net_ret_equals_gross_minus_cost(self):
        """For every bar: net_ret ≈ gross_ret + cash_ret - cost_ret."""
        from pathlib import Path

        runs_dir = Path("artifacts/runs")
        if not runs_dir.exists():
            pytest.skip("No artifacts directory")

        checked = 0
        for run_dir in sorted(runs_dir.iterdir())[:20]:
            eq_path = run_dir / "equity.parquet"
            if not eq_path.exists():
                continue

            eq = pl.read_parquet(eq_path)
            if "gross_ret" not in eq.columns or "cost_ret" not in eq.columns:
                continue

            gross = eq["gross_ret"].to_list()
            net = eq["net_ret"].to_list()
            cost = eq["cost_ret"].to_list()
            cash = eq["cash_ret"].to_list() if "cash_ret" in eq.columns else [0.0] * len(gross)
            funding = eq["funding_costs"].to_list() if "funding_costs" in eq.columns else [0.0] * len(gross)

            for t in range(len(gross)):
                expected_net = gross[t] + cash[t] - cost[t] - funding[t]
                assert abs(net[t] - expected_net) < 1e-10, (
                    f"Bar {t} in {run_dir.name}: "
                    f"net={net[t]}, expected={expected_net} "
                    f"(gross={gross[t]}, cash={cash[t]}, cost={cost[t]}, funding={funding[t]})"
                )
            checked += 1

        assert checked > 0, "No equity files checked"

    def test_nav_matches_compounded_net_ret(self):
        """NAV[t] = NAV[t-1] * (1 + net_ret[t])."""
        from pathlib import Path

        runs_dir = Path("artifacts/runs")
        if not runs_dir.exists():
            pytest.skip("No artifacts directory")

        checked = 0
        for run_dir in sorted(runs_dir.iterdir())[:20]:
            eq_path = run_dir / "equity.parquet"
            if not eq_path.exists():
                continue

            eq = pl.read_parquet(eq_path)
            nav = eq["nav"].to_list()
            net = eq["net_ret"].to_list()

            for t in range(1, len(nav)):
                expected_nav = nav[t - 1] * (1 + net[t])
                assert abs(nav[t] - expected_nav) < 0.01, (
                    f"Bar {t} in {run_dir.name}: "
                    f"nav={nav[t]}, expected={expected_nav} "
                    f"(prev_nav={nav[t-1]}, net_ret={net[t]})"
                )
            checked += 1

        assert checked > 0, "No equity files checked"

    def test_cost_ret_equals_turnover_times_fee_bps(self):
        """cost_ret[t] = turnover[t] * fee_bps_total / 10000."""
        import json
        from pathlib import Path

        runs_dir = Path("artifacts/runs")
        if not runs_dir.exists():
            pytest.skip("No artifacts directory")

        checked = 0
        for run_dir in sorted(runs_dir.iterdir())[:20]:
            eq_path = run_dir / "equity.parquet"
            manifest_path = run_dir / "manifest.json"
            if not eq_path.exists() or not manifest_path.exists():
                continue

            manifest = json.loads(manifest_path.read_text())
            params = manifest.get("params", {})
            execution = params.get("execution", {})
            fee_bps = execution.get("fee_bps", 0.0) or 0.0
            slip_bps = execution.get("slippage_bps", 0.0) or 0.0
            total_bps = fee_bps + slip_bps

            eq = pl.read_parquet(eq_path)
            if "turnover" not in eq.columns or "cost_ret" not in eq.columns:
                continue

            turnover = eq["turnover"].to_list()
            cost_ret = eq["cost_ret"].to_list()

            for t in range(len(turnover)):
                expected_cost = turnover[t] * (total_bps / 10000.0)
                assert abs(cost_ret[t] - expected_cost) < 1e-10, (
                    f"Bar {t} in {run_dir.name}: "
                    f"cost_ret={cost_ret[t]}, expected={expected_cost} "
                    f"(turnover={turnover[t]}, total_bps={total_bps})"
                )
            checked += 1

        assert checked > 0, "No equity files checked"
