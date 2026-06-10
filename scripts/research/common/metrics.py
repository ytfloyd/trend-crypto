"""
Shared performance metrics and formatting utilities.

DEPRECATED LOCATION — the implementation now lives in ``core.metrics`` (single
source of truth). This module re-exports it verbatim so existing
``from common.metrics import ...`` callers keep working with zero behavior
change. New code should import from ``core.metrics`` directly.

See docs/RESEARCH_PIPELINE_REORGANIZATION.md (Phase 1: unify the core stack).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the editable `src/` packages are importable even when this module is
# reached without `src` on the path (mirrors the repo's PYTHONPATH=src convention).
_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.metrics import (  # noqa: E402,F401
    ANN_FACTOR,
    _cross_sectional_ic,
    compute_ic_decay,
    compute_metrics,
    compute_regime,
    compute_yearly_sharpe_trend,
    format_metrics_table,
    information_horizon,
)

__all__ = [
    "ANN_FACTOR",
    "compute_metrics",
    "information_horizon",
    "compute_ic_decay",
    "compute_yearly_sharpe_trend",
    "format_metrics_table",
    "compute_regime",
    "_cross_sectional_ic",
]
