"""Shared contracts + gate utilities for the evaluation pipelines.

The cross_sectional and time_series pipelines share these stage-result types and
the CPCV embargo helper. (convexity defines its own StageResult.) Extracting
them here means time_series no longer reaches into cross_sectional. See
docs/RESEARCH_PIPELINE_REORGANIZATION.md.
"""
from .embargo import _apply_embargo
from .types import StageResult, StageVerdict

__all__ = ["StageResult", "StageVerdict", "_apply_embargo"]
