"""Automated alpha research pipeline — generate, screen, validate, catalog."""

from src.alpha_pipeline.types import (
    AlphaCandidate,
    GateConfig,
    PipelineReport,
    StageResult,
    StageVerdict,
)
from src.alpha_pipeline.pipeline import AlphaPipeline

__all__ = [
    "AlphaCandidate",
    "AlphaPipeline",
    "GateConfig",
    "PipelineReport",
    "StageResult",
    "StageVerdict",
]
