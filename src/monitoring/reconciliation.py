"""Automated position reconciliation for live trading.

Compares expected portfolio state against actual broker positions
and reports drift metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from common.logging import get_logger

logger = get_logger("reconciliation")


@dataclass(frozen=True)
class DriftReport:
    """Report on position drift between expected and actual state.

    Attributes:
        ts: Timestamp of the reconciliation.
        expected_weights: Symbol → expected weight.
        actual_weights: Symbol → actual weight.
        drifts: Symbol → drift (actual - expected).
        max_drift: Maximum absolute drift across all symbols.
        mean_drift: Mean absolute drift.
        is_clean: True if all drifts are within tolerance.
        symbols_with_drift: Symbols exceeding tolerance.
    """

    ts: str
    expected_weights: dict[str, float]
    actual_weights: dict[str, float]
    drifts: dict[str, float]
    max_drift: float
    mean_drift: float
    is_clean: bool
    symbols_with_drift: list[str]


def reconcile_live_vs_target(
    expected_weights: dict[str, float],
    actual_weights: dict[str, float],
    tolerance: float = 0.02,
) -> DriftReport:
    """Compare expected vs actual portfolio weights.

    Args:
        expected_weights: Symbol → expected weight.
        actual_weights: Symbol → actual weight (from broker).
        tolerance: Maximum acceptable drift.

    Returns:
        DriftReport with drift analysis.
    """
    all_symbols = sorted(set(expected_weights) | set(actual_weights))
    drifts: dict[str, float] = {}
    symbols_with_drift: list[str] = []

    for sym in all_symbols:
        expected = expected_weights.get(sym, 0.0)
        actual = actual_weights.get(sym, 0.0)
        drift = actual - expected
        drifts[sym] = drift
        if abs(drift) > tolerance:
            symbols_with_drift.append(sym)

    abs_drifts = [abs(d) for d in drifts.values()]
    max_drift = max(abs_drifts) if abs_drifts else 0.0
    mean_drift = sum(abs_drifts) / len(abs_drifts) if abs_drifts else 0.0
    is_clean = len(symbols_with_drift) == 0

    report = DriftReport(
        ts=datetime.now(timezone.utc).isoformat(),
        expected_weights=dict(expected_weights),
        actual_weights=dict(actual_weights),
        drifts=drifts,
        max_drift=max_drift,
        mean_drift=mean_drift,
        is_clean=is_clean,
        symbols_with_drift=symbols_with_drift,
    )

    if not is_clean:
        logger.warning(
            "Position drift detected: max=%.4f, symbols=%s",
            max_drift, symbols_with_drift,
        )
    else:
        logger.info("Reconciliation clean: max_drift=%.4f", max_drift)

    return report


def reconcile_history(
    reports: list[DriftReport],
) -> dict[str, Any]:
    """Aggregate statistics across multiple reconciliation reports.

    Args:
        reports: List of DriftReports from multiple cycles.

    Returns:
        Dict with aggregated reconciliation statistics.
    """
    if not reports:
        return {
            "n_reports": 0,
            "n_clean": 0,
            "n_dirty": 0,
            "avg_max_drift": 0.0,
            "worst_max_drift": 0.0,
            "most_drifted_symbols": [],
        }

    n_clean = sum(1 for r in reports if r.is_clean)
    max_drifts = [r.max_drift for r in reports]
    all_drifted: dict[str, int] = {}
    for r in reports:
        for sym in r.symbols_with_drift:
            all_drifted[sym] = all_drifted.get(sym, 0) + 1

    return {
        "n_reports": len(reports),
        "n_clean": n_clean,
        "n_dirty": len(reports) - n_clean,
        "avg_max_drift": sum(max_drifts) / len(max_drifts),
        "worst_max_drift": max(max_drifts),
        "most_drifted_symbols": sorted(all_drifted.items(), key=lambda x: -x[1]),
    }
