"""Shared stage-result contracts for the cross_sectional and time_series pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class StageResult:
    """Outcome of a single pipeline stage for one alpha."""

    stage: str
    verdict: StageVerdict
    metrics: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
