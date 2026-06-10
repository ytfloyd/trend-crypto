"""
Shared backtesting utilities.

DEPRECATED LOCATION — the implementation now lives in ``core.backtest`` (single
source of truth). This module re-exports it verbatim so existing
``from common.backtest import ...`` callers keep working with zero behavior
change. New code should import from ``core.backtest`` directly.

See docs/RESEARCH_PIPELINE_REORGANIZATION.md (Phase 1: unify the core stack).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.backtest import (  # noqa: E402,F401
    DEFAULT_COST_BPS,
    simple_backtest,
)

__all__ = ["DEFAULT_COST_BPS", "simple_backtest"]
