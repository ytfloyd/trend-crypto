"""Automated alpha research pipeline — generate, screen, validate, catalog."""

from .types import (
    AlphaCandidate,
    GateConfig,
    PipelineReport,
    StageResult,
    StageVerdict,
)
from .pipeline import AlphaPipeline

__all__ = [
    "AlphaCandidate",
    "AlphaPipeline",
    "GateConfig",
    "PipelineReport",
    "StageResult",
    "StageVerdict",
]
