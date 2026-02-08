"""Tests for Phase 3: Advanced Risk Framework.

Covers:
- VaR: historical, parametric, component
- Stress testing: scenario application, built-in scenarios
- PortfolioRiskManager: leverage/concentration constraints
- Regime detection: EWMA correlation, DCC estimator
- Risk attribution: factor OLS decomposition
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from common.config import PortfolioConfig, RiskConfigRaw, compile_config
from risk.attribution import FactorAttribution, factor_risk_attribution
from risk.regime import (
    RegimeState,
    detect_correlation_regime,
    ewma_correlation,
    ewma_volatility,
    rolling_dcc_correlation,
)
from risk.risk_manager import PortfolioRiskManager, RiskManager
from risk.stress import (
    BUILTIN_SCENARIOS,
    COVID_MARCH_2020,
    StressResult,
    StressScenario,
    run_all_stress_tests,
    run_stress_test,
)
from risk.var import component_var, historical_var, parametric_var


# ---------------------------------------------------------------------------
# VaR tests
# ---------------------------------------------------------------------------


class TestHistoricalVaR:
    def test_known_distribution(self) -> None:
        """VaR of uniform returns should approximate the quantile."""
        # 100 returns from -0.05 to +0.05
        returns = pl.Series("ret", [i / 1000.0 - 0.05 for i in range(101)])
        var_95 = historical_var(returns, confidence=0.95)
        # 5th percentile of [-0.05, 0.05] ≈ -0.045
        assert var_95 > 0  # positive (loss)
        assert abs(var_95 - 0.045) < 0.01

    def test_empty_returns(self) -> None:
        assert historical_var(pl.Series("r", []), confidence=0.95) == 0.0

    def test_single_return(self) -> None:
        assert historical_var(pl.Series("r", [0.01]), confidence=0.95) == 0.0


class TestParametricVaR:
    def test_single_asset(self) -> None:
        """Single asset parametric VaR = z * sigma * sqrt(hp)."""
        cov = {"A": {"A": 0.0004}}  # sigma = 0.02
        weights = {"A": 1.0}
        var = parametric_var(weights, cov, confidence=0.95)
        # z_95 ≈ 1.645, VaR ≈ 1.645 * 0.02 ≈ 0.033
        assert 0.02 < var < 0.05

    def test_two_uncorrelated_assets(self) -> None:
        """Uncorrelated assets: portfolio vol < sum of individual vols."""
        cov = {
            "A": {"A": 0.0004, "B": 0.0},
            "B": {"A": 0.0, "B": 0.0004},
        }
        single_var = parametric_var({"A": 1.0}, {"A": {"A": 0.0004}}, confidence=0.95)
        portfolio_var = parametric_var({"A": 0.5, "B": 0.5}, cov, confidence=0.95)
        # Diversification benefit: portfolio VaR < single asset VaR
        assert portfolio_var < single_var

    def test_empty_weights(self) -> None:
        assert parametric_var({}, {}, confidence=0.95) == 0.0


class TestComponentVaR:
    def test_components_sum_to_total(self) -> None:
        """Component VaRs should sum to total parametric VaR."""
        cov = {
            "A": {"A": 0.0004, "B": 0.0001},
            "B": {"A": 0.0001, "B": 0.0009},
        }
        weights = {"A": 0.6, "B": 0.4}
        total = parametric_var(weights, cov, confidence=0.95)
        components = component_var(weights, cov, confidence=0.95)
        component_sum = sum(components.values())
        assert abs(total - component_sum) < 1e-10


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------


class TestStressTesting:
    def test_simple_scenario(self) -> None:
        scenario = StressScenario(
            name="Test", asset_shocks={"BTC-USD": -0.30, "ETH-USD": -0.40},
        )
        weights = {"BTC-USD": 0.5, "ETH-USD": 0.5}
        result = run_stress_test(scenario, weights, nav=100_000.0)
        assert isinstance(result, StressResult)
        # PnL = 0.5 * -0.30 * 100k + 0.5 * -0.40 * 100k = -15k + -20k = -35k
        assert abs(result.portfolio_pnl - (-35_000.0)) < 0.01
        assert abs(result.stressed_nav - 65_000.0) < 0.01

    def test_default_shock_for_unknown_asset(self) -> None:
        scenario = StressScenario(
            name="Partial", asset_shocks={"BTC-USD": -0.20},
        )
        weights = {"BTC-USD": 0.5, "XYZ-USD": 0.5}
        result = run_stress_test(scenario, weights, nav=100_000.0, default_shock=-0.10)
        # BTC: 0.5 * -0.20 * 100k = -10k, XYZ: 0.5 * -0.10 * 100k = -5k
        assert abs(result.portfolio_pnl - (-15_000.0)) < 0.01

    def test_builtin_scenarios_exist(self) -> None:
        assert len(BUILTIN_SCENARIOS) == 3
        names = [s.name for s in BUILTIN_SCENARIOS]
        assert "COVID March 2020" in names
        assert "LUNA/UST May 2022" in names
        assert "FTX Nov 2022" in names

    def test_run_all_stress_tests(self) -> None:
        weights = {"BTC-USD": 0.5, "ETH-USD": 0.3, "SOL-USD": 0.2}
        results = run_all_stress_tests(weights, nav=100_000.0)
        assert len(results) == 3
        for r in results:
            assert r.portfolio_pnl < 0  # All scenarios should be negative
            assert r.stressed_nav < 100_000.0


# ---------------------------------------------------------------------------
# PortfolioRiskManager
# ---------------------------------------------------------------------------


class TestPortfolioRiskManager:
    def _make_history(self, n: int = 50) -> pl.DataFrame:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return pl.DataFrame({
            "ts": [start + timedelta(hours=i) for i in range(n)],
            "close": [100.0 + 0.5 * i for i in range(n)],
            "open": [100.0 + 0.5 * i for i in range(n)],
            "high": [101.0 + 0.5 * i for i in range(n)],
            "low": [99.0 + 0.5 * i for i in range(n)],
            "volume": [1000.0] * n,
        })

    def _make_rm(self) -> RiskManager:
        from common.config import RiskConfigResolved
        cfg = RiskConfigResolved(
            vol_window=10, target_vol_annual=None, max_weight=1.0, window_units="bars",
        )
        return RiskManager(cfg=cfg, periods_per_year=8760.0)

    def test_gross_leverage_clamping(self) -> None:
        portfolio_cfg = PortfolioConfig(
            symbols=["A", "B"],
            max_gross_leverage=0.5,
            max_single_name_weight=1.0,
        )
        rm = self._make_rm()
        prm = PortfolioRiskManager(per_asset_rm=rm, portfolio_cfg=portfolio_cfg)
        history = self._make_history()
        result = prm.apply(
            {"A": 0.8, "B": 0.8},
            {"A": history, "B": history},
        )
        gross = sum(abs(v) for v in result.values())
        assert gross <= 0.5 + 1e-10

    def test_single_name_limit(self) -> None:
        portfolio_cfg = PortfolioConfig(
            symbols=["A", "B"],
            max_gross_leverage=2.0,
            max_single_name_weight=0.3,
        )
        rm = self._make_rm()
        prm = PortfolioRiskManager(per_asset_rm=rm, portfolio_cfg=portfolio_cfg)
        history = self._make_history()
        result = prm.apply(
            {"A": 0.8, "B": 0.1},
            {"A": history, "B": history},
        )
        assert abs(result["A"]) <= 0.3 + 1e-10

    def test_passthrough_when_no_constraints_binding(self) -> None:
        portfolio_cfg = PortfolioConfig(
            symbols=["A"], max_gross_leverage=2.0, max_single_name_weight=1.0,
        )
        rm = self._make_rm()
        prm = PortfolioRiskManager(per_asset_rm=rm, portfolio_cfg=portfolio_cfg)
        history = self._make_history()
        result = prm.apply({"A": 0.5}, {"A": history})
        # With no vol targeting (target_vol_annual=None), weight should pass through
        assert abs(result["A"] - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


class TestRegimeDetection:
    def test_ewma_volatility_length(self) -> None:
        returns = [0.01, -0.02, 0.005, 0.01, -0.005]
        vols = ewma_volatility(returns, halflife=3)
        assert len(vols) == len(returns)
        assert vols[0] == 0.0  # First value is always 0

    def test_ewma_correlation_bounds(self) -> None:
        returns_a = [0.01, -0.02, 0.005, 0.01, -0.005] * 10
        returns_b = [0.01, -0.02, 0.005, 0.01, -0.005] * 10
        corrs = ewma_correlation(returns_a, returns_b, halflife=5)
        for c in corrs[1:]:  # Skip first (0)
            assert -1.0 <= c <= 1.0

    def test_perfect_correlation_detected(self) -> None:
        n = 100
        base = [0.01 * (i % 5 - 2) for i in range(n)]
        regime = detect_correlation_regime(
            {"A": base, "B": base}, halflife=10, crisis_threshold=0.7,
        )
        assert regime == RegimeState.CRISIS

    def test_dcc_matrices_shape(self) -> None:
        n = 50
        a = [0.01 * (i % 3 - 1) for i in range(n)]
        b = [0.01 * ((i + 2) % 3 - 1) for i in range(n)]
        matrices = rolling_dcc_correlation({"A": a, "B": b}, halflife=10)
        assert len(matrices) == n
        for m in matrices:
            assert m["A"]["A"] == 1.0
            assert m["B"]["B"] == 1.0

    def test_insufficient_data(self) -> None:
        regime = detect_correlation_regime({"A": [0.01], "B": [-0.01]})
        assert regime is None


# ---------------------------------------------------------------------------
# Risk attribution
# ---------------------------------------------------------------------------


class TestRiskAttribution:
    def test_single_factor_perfect_fit(self) -> None:
        """When portfolio = 2 * factor, beta should be ~2, R² ~1."""
        n = 100
        factor = [0.01 * (i % 7 - 3) for i in range(n)]
        portfolio = [2.0 * f for f in factor]
        result = factor_risk_attribution(portfolio, {"market": factor})
        assert isinstance(result, FactorAttribution)
        assert abs(result.factor_betas["market"] - 2.0) < 0.01
        assert result.r_squared > 0.99

    def test_no_factors(self) -> None:
        result = factor_risk_attribution([0.01, 0.02, -0.01], {})
        assert result.r_squared == 0.0
        assert result.factor_betas == {}

    def test_insufficient_data(self) -> None:
        result = factor_risk_attribution([0.01], {"f": [0.01]})
        assert result.r_squared == 0.0

    def test_factor_contributions_sum_bounded(self) -> None:
        """Factor contributions should be between 0 and 1 for well-behaved data."""
        n = 200
        factor_a = [0.01 * math.sin(i * 0.1) for i in range(n)]
        factor_b = [0.005 * math.cos(i * 0.15) for i in range(n)]
        portfolio = [1.5 * a + 0.8 * b + 0.001 * (i % 3 - 1) for i, (a, b) in enumerate(zip(factor_a, factor_b))]
        result = factor_risk_attribution(portfolio, {"A": factor_a, "B": factor_b})
        total_contrib = sum(result.factor_contributions.values())
        # Factor contributions should explain most of the variance
        assert total_contrib > 0.5
        assert result.r_squared > 0.8
