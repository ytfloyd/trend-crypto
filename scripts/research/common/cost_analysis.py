"""
Transaction-cost analysis helpers.

DEPRECATED LOCATION — the implementation now lives in ``core.cost_analysis``
(single source of truth). This module re-exports it verbatim so existing
``from common.cost_analysis import ...`` callers keep working with zero behavior
change. New code should import from ``core.cost_analysis`` directly.

See docs/RESEARCH_PIPELINE_REORGANIZATION.md (Phase 1: unify the core stack).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.cost_analysis import (  # noqa: E402,F401
    RuleCostReport,
    analyse_rule,
    analyse_rule_set,
    marginal_value,
    select_viable_rules,
)

__all__ = [
    "RuleCostReport",
    "analyse_rule",
    "analyse_rule_set",
    "marginal_value",
    "select_viable_rules",
]
