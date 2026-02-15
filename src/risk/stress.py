"""Stress testing framework for portfolio risk analysis.

Provides scenario definitions and a runner that applies shocks to portfolio
weights and computes stressed PnL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StressScenario:
    """Definition of a stress scenario.

    Attributes:
        name: Human-readable scenario name.
        asset_shocks: Symbol → return shock (e.g. -0.30 = -30% move).
        vol_multiplier: Multiplier applied to all asset volatilities.
        correlation_override: If set, all pairwise correlations are set to this value.
    """

    name: str
    asset_shocks: dict[str, float]
    vol_multiplier: float = 1.0
    correlation_override: Optional[float] = None


# Pre-built scenarios based on historical crypto events
COVID_MARCH_2020 = StressScenario(
    name="COVID March 2020",
    asset_shocks={
        "BTC-USD": -0.40,
        "ETH-USD": -0.45,
        "SOL-USD": -0.50,
        "BNB-USD": -0.42,
    },
    vol_multiplier=3.0,
    correlation_override=0.90,
)

LUNA_UST_MAY_2022 = StressScenario(
    name="LUNA/UST May 2022",
    asset_shocks={
        "BTC-USD": -0.25,
        "ETH-USD": -0.35,
        "SOL-USD": -0.50,
        "BNB-USD": -0.30,
    },
    vol_multiplier=2.5,
    correlation_override=0.85,
)

FTX_NOV_2022 = StressScenario(
    name="FTX Nov 2022",
    asset_shocks={
        "BTC-USD": -0.22,
        "ETH-USD": -0.28,
        "SOL-USD": -0.65,
        "BNB-USD": -0.20,
    },
    vol_multiplier=2.0,
    correlation_override=0.80,
)

BUILTIN_SCENARIOS = [COVID_MARCH_2020, LUNA_UST_MAY_2022, FTX_NOV_2022]


@dataclass
class StressResult:
    """Result of applying a stress scenario to a portfolio.

    Attributes:
        scenario_name: Name of the scenario applied.
        portfolio_pnl: Portfolio-level PnL from the stress event.
        asset_pnl: Per-asset PnL breakdown.
        stressed_nav: NAV after applying the stress (from initial NAV).
        nav_drawdown_pct: Drawdown as percentage of initial NAV.
    """

    scenario_name: str
    portfolio_pnl: float
    asset_pnl: dict[str, float]
    stressed_nav: float
    nav_drawdown_pct: float


def run_stress_test(
    scenario: StressScenario,
    weights: dict[str, float],
    nav: float = 100_000.0,
    default_shock: float = -0.20,
) -> StressResult:
    """Apply a stress scenario to portfolio weights and compute stressed PnL.

    For assets with a weight but no scenario-specific shock, ``default_shock``
    is applied.

    Args:
        scenario: The stress scenario to apply.
        weights: Symbol → current weight mapping.
        nav: Current portfolio NAV.
        default_shock: Shock applied to assets not in scenario.asset_shocks.

    Returns:
        StressResult with portfolio and per-asset PnL.
    """
    asset_pnl: dict[str, float] = {}
    total_pnl = 0.0

    for symbol, weight in weights.items():
        shock = scenario.asset_shocks.get(symbol, default_shock)
        pnl = weight * shock * nav
        asset_pnl[symbol] = pnl
        total_pnl += pnl

    stressed_nav = nav + total_pnl
    dd_pct = (total_pnl / nav) * 100.0 if nav > 0 else 0.0

    return StressResult(
        scenario_name=scenario.name,
        portfolio_pnl=total_pnl,
        asset_pnl=asset_pnl,
        stressed_nav=stressed_nav,
        nav_drawdown_pct=dd_pct,
    )


def run_all_stress_tests(
    weights: dict[str, float],
    nav: float = 100_000.0,
    scenarios: Optional[list[StressScenario]] = None,
    default_shock: float = -0.20,
) -> list[StressResult]:
    """Run all built-in or custom stress scenarios.

    Args:
        weights: Symbol → current weight mapping.
        nav: Current portfolio NAV.
        scenarios: Custom scenarios; defaults to BUILTIN_SCENARIOS.
        default_shock: Shock for assets not in a scenario.

    Returns:
        List of StressResult, one per scenario.
    """
    if scenarios is None:
        scenarios = BUILTIN_SCENARIOS
    return [run_stress_test(s, weights, nav, default_shock) for s in scenarios]
