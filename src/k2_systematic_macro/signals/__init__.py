"""Research-only signal layers for K2 systematic macro."""
from __future__ import annotations

from .trade_candidates import (
    TradeCandidateArtifacts,
    TradeCandidateConfig,
    build_trade_candidate_artifacts,
    build_trade_candidates,
)

__all__ = [
    "TradeCandidateArtifacts",
    "TradeCandidateConfig",
    "build_trade_candidate_artifacts",
    "build_trade_candidates",
]
