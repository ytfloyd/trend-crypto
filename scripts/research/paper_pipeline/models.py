"""Data models for the paper discovery pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class StrategyType(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STAT_ARB = "stat_arb"
    VOLATILITY = "volatility"
    FACTOR = "factor"
    SEASONAL = "seasonal"
    ML_SIGNAL = "ml_signal"
    EXECUTION = "execution"
    OTHER = "other"


class FilterVerdict(str, Enum):
    PASS = "pass"
    FAIL_ALPHA = "fail_alpha"
    FAIL_RATIONALE = "fail_rationale"
    FAIL_ROBUSTNESS = "fail_robustness"
    FAIL_IMPLEMENTABILITY = "fail_implementability"
    FAIL_METHODOLOGY = "fail_methodology"
    FLAG_STALENESS = "flag_staleness"


@dataclass
class PaperMeta:
    """Metadata extracted from a paper listing."""

    paper_id: str
    title: str
    authors: list[str]
    publication_date: str
    abstract: str
    source: str  # "arxiv" or "ssrn"
    url: str
    pdf_url: str | None = None
    categories: list[str] = field(default_factory=list)

    asset_classes: list[str] = field(default_factory=list)
    strategy_type: StrategyType = StrategyType.OTHER
    data_period: str = ""
    claimed_result: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["strategy_type"] = self.strategy_type.value
        return d


@dataclass
class FilterResult:
    """Outcome of passing a paper through the filter stack."""

    paper_id: str
    verdict: FilterVerdict
    passed: bool
    staleness_flag: bool = False
    rejection_reason: str = ""
    filter_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


@dataclass
class StrategySpec:
    """Full strategy specification extracted from a paper."""

    paper_id: str
    strategy_name: str
    strategy_type: StrategyType
    asset_classes: list[str]
    signal_description: str
    entry_rule: str
    exit_rule: str
    position_sizing: str
    rebalance_frequency: str
    data_requirements: list[str]
    risk_management: str
    transaction_costs_noted: str
    claimed_sharpe: float | None = None
    claimed_cagr: float | None = None
    claimed_max_dd: float | None = None
    data_period: str = ""
    out_of_sample: bool = False
    multi_market: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["strategy_type"] = self.strategy_type.value
        return d


@dataclass
class CatalogueEntry:
    """Complete catalogue entry combining paper, filter, and strategy."""

    paper: PaperMeta
    filter_result: FilterResult
    strategy: StrategySpec | None = None
    discovered_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        d = {
            "paper": self.paper.to_dict(),
            "filter_result": self.filter_result.to_dict(),
            "strategy": self.strategy.to_dict() if self.strategy else None,
            "discovered_at": self.discovered_at,
        }
        return d
