"""
Portfolio risk overlays (position limits, vol targeting, drawdown control, ...).

DEPRECATED LOCATION — the implementation now lives in ``core.risk_overlays``
(single source of truth). This module re-exports it verbatim so existing
``from common.risk_overlays import ...`` callers keep working with zero behavior
change. New code should import from ``core.risk_overlays`` directly.

See docs/RESEARCH_PIPELINE_REORGANIZATION.md (Phase 1: unify the core stack).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.risk_overlays import (  # noqa: E402,F401
    apply_dd_control,
    apply_position_limit_wide,
    apply_trailing_stop,
    apply_vol_concentration,
    apply_vol_targeting,
)

__all__ = [
    "apply_dd_control",
    "apply_position_limit_wide",
    "apply_trailing_stop",
    "apply_vol_concentration",
    "apply_vol_targeting",
]
