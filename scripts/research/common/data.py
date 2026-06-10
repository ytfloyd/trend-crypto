"""
Shared data loading and universe filtering utilities.

DEPRECATED LOCATION — the implementation now lives in ``core.data`` (single
source of truth). This module re-exports it verbatim so existing
``from common.data import ...`` callers keep working with zero behavior change.
New code should import from ``core.data`` directly.

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

from core.data import (  # noqa: E402,F401
    ANN_FACTOR,
    BARS_PER_DAY,
    DEFAULT_DB,
    FREQ_INTERVALS,
    compute_btc_benchmark,
    filter_universe,
    load_bars,
    load_daily_bars,
)

__all__ = [
    "ANN_FACTOR",
    "BARS_PER_DAY",
    "DEFAULT_DB",
    "FREQ_INTERVALS",
    "compute_btc_benchmark",
    "filter_universe",
    "load_bars",
    "load_daily_bars",
]
