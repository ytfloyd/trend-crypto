"""Convexity Alpha Pipeline.

Implements the research process defined in
docs/research/convexity_alpha_pipeline_spec.md.

Parallel pipeline to src/alpha_pipeline/ (which handles cross-sectional /
linear quant-equity alpha). Routes candidates based on the Alpha Registry's
`expected_payoff_shape` field.

Public surface:
    from convexity_pipeline import (
        Track, PayoffShape, Stage,
        Hypothesis, Candidate, BacktestResult, StageResult,
        PipelineConfig, default_config,
        metrics,
        Stage0Evaluator, Stage1Evaluator, Stage2Evaluator,
        Stage3Evaluator, Stage4Evaluator,
        ConvexityPipelineRunner,
    )
"""
from .types import (
    Track,
    PayoffShape,
    Stage,
    Hypothesis,
    Candidate,
    BacktestResult,
    StageResult,
    KillDecision,
    PipelineConfig,
    default_config,
)
from . import metrics
from .stages import (
    Stage0Evaluator,
    Stage1Evaluator,
    Stage2Evaluator,
    Stage3Evaluator,
    Stage4Evaluator,
)
from .runner import ConvexityPipelineRunner

__all__ = [
    "Track",
    "PayoffShape",
    "Stage",
    "Hypothesis",
    "Candidate",
    "BacktestResult",
    "StageResult",
    "KillDecision",
    "PipelineConfig",
    "default_config",
    "metrics",
    "Stage0Evaluator",
    "Stage1Evaluator",
    "Stage2Evaluator",
    "Stage3Evaluator",
    "Stage4Evaluator",
    "ConvexityPipelineRunner",
]
