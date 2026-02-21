"""Phase 2: Multi-stage filter stack for paper quality assessment.

Each filter scores a paper on a 0-1 scale and produces a pass/fail verdict
with reasoning. A paper must pass ALL mandatory filters to proceed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .models import FilterResult, FilterVerdict, PaperMeta, StrategyType


@dataclass
class _Score:
    passed: bool
    score: float
    reason: str


# ---------------------------------------------------------------------------
# Filter 1: Alpha filter
# ---------------------------------------------------------------------------
ALPHA_POSITIVE = [
    "alpha", "excess return", "abnormal return", "outperform",
    "profitable", "trading profit", "risk-adjusted return",
    "predictability", "predictive", "forecast",
    "long-short", "long short", "trading signal",
    "buy signal", "sell signal", "market timing",
]
ALPHA_NEGATIVE = [
    "risk factor decomposition", "asset pricing model",
    "no evidence of alpha", "cannot reject",
    "risk premium explanation", "priced by factor",
    "efficient market", "market efficiency",
]


def _alpha_filter(paper: PaperMeta) -> _Score:
    """Does the paper claim to uncover excess return (alpha)?"""
    text = f"{paper.title} {paper.abstract}".lower()

    pos_hits = sum(1 for kw in ALPHA_POSITIVE if kw in text)
    neg_hits = sum(1 for kw in ALPHA_NEGATIVE if kw in text)

    score = min(1.0, pos_hits / 3) - min(1.0, neg_hits / 2)
    passed = score > 0.1

    if not passed:
        return _Score(False, score, "No clear alpha claim; appears to be risk-factor decomposition")
    return _Score(True, score, f"Alpha signals: {pos_hits} positive, {neg_hits} negative indicators")


# ---------------------------------------------------------------------------
# Filter 2: Economic rationale
# ---------------------------------------------------------------------------
RATIONALE_KEYWORDS = [
    "behavioral bias", "overreaction", "underreaction",
    "disposition effect", "anchoring", "herding",
    "limits to arbitrage", "short-sale constraint",
    "liquidity premium", "illiquidity", "funding constraint",
    "institutional friction", "regulatory", "tax",
    "information asymmetry", "informed trader",
    "structural", "market microstructure",
    "risk premium", "carry", "convenience yield",
    "investor sentiment", "attention", "salience",
    "supply-demand imbalance", "flow", "rebalancing",
]


def _rationale_filter(paper: PaperMeta) -> _Score:
    """Is there a plausible non-data-mined reason the inefficiency exists?"""
    text = f"{paper.title} {paper.abstract}".lower()

    hits = sum(1 for kw in RATIONALE_KEYWORDS if kw in text)
    score = min(1.0, hits / 2)

    if hits == 0:
        return _Score(
            False, 0.0,
            "No economic rationale articulated; purely empirical without causal mechanism"
        )
    return _Score(True, score, f"Rationale keywords found: {hits}")


# ---------------------------------------------------------------------------
# Filter 3: Statistical robustness
# ---------------------------------------------------------------------------
OOS_KEYWORDS = [
    "out-of-sample", "out of sample", "oos",
    "holdout", "hold-out", "validation set",
    "walk-forward", "walk forward", "expanding window",
    "cross-validation", "cross validation",
    "backtesting", "backtest", "paper trading",
]

MULTI_MARKET_KEYWORDS = [
    "multiple market", "cross-country", "international",
    "multiple asset", "multi-asset", "several market",
    "diverse", "robustness check", "subsample",
    "different time period", "sub-period",
]

TCA_KEYWORDS = [
    "transaction cost", "trading cost", "slippage",
    "bid-ask", "bid ask", "spread", "commission",
    "market impact", "execution cost", "round-trip",
    "net of cost", "after cost", "cost-adjusted",
]


def _robustness_filter(paper: PaperMeta) -> _Score:
    """Does the paper demonstrate statistical robustness?"""
    text = f"{paper.title} {paper.abstract}".lower()

    has_oos = any(kw in text for kw in OOS_KEYWORDS)
    has_multi = any(kw in text for kw in MULTI_MARKET_KEYWORDS)
    has_tca = any(kw in text for kw in TCA_KEYWORDS)

    # Check for sample size mentions
    year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", paper.abstract)
    data_span_years = 0
    if len(year_matches) >= 2:
        years = sorted(int(y) for y in set(year_matches))
        data_span_years = years[-1] - years[0]
    has_sufficient_data = data_span_years >= 5

    checks = [has_oos, has_multi, has_tca, has_sufficient_data]
    score = sum(checks) / len(checks)
    passed = sum(checks) >= 2  # need at least OOS + one other

    reasons = []
    if not has_oos:
        reasons.append("no out-of-sample testing mentioned")
    if not has_multi:
        reasons.append("single market/period only")
    if not has_tca:
        reasons.append("no transaction cost analysis")
    if not has_sufficient_data:
        reasons.append(f"data span ~{data_span_years}y (need >=5)")

    if not passed:
        return _Score(False, score, "; ".join(reasons))

    return _Score(
        True, score,
        f"OOS={has_oos}, multi-market={has_multi}, TCA={has_tca}, data={data_span_years}y"
    )


# ---------------------------------------------------------------------------
# Filter 4: Implementability
# ---------------------------------------------------------------------------
PROPRIETARY_KEYWORDS = [
    "proprietary data", "tick data", "order book",
    "level 2", "level ii", "market maker",
    "co-location", "colocation", "fpga",
    "nanosecond", "microsecond", "latency arbitrage",
    "hft", "high-frequency", "high frequency",
    "private dataset", "internal data",
]

PUBLIC_DATA_KEYWORDS = [
    "yahoo finance", "crsp", "compustat",
    "bloomberg", "datastream", "wrds",
    "publicly available", "open source",
    "coinbase", "binance", "coingecko",
    "daily data", "daily returns", "monthly return",
    "weekly return",
]


def _implementability_filter(paper: PaperMeta) -> _Score:
    """Can a non-institutional trader implement this?"""
    text = f"{paper.title} {paper.abstract}".lower()

    proprietary_hits = sum(1 for kw in PROPRIETARY_KEYWORDS if kw in text)
    public_hits = sum(1 for kw in PUBLIC_DATA_KEYWORDS if kw in text)

    # HFT papers are auto-rejected
    is_hft = any(kw in text for kw in ["hft", "high-frequency", "high frequency", "nanosecond"])
    if is_hft:
        return _Score(False, 0.0, "HFT infrastructure required; not implementable by retail")

    if paper.strategy_type == StrategyType.EXECUTION:
        return _Score(
            False, 0.2,
            "Execution/market-making strategy; requires specialized infrastructure"
        )

    score = max(0.0, min(1.0, (public_hits - proprietary_hits + 1) / 3))
    passed = proprietary_hits < 2 and score > 0.2

    if not passed:
        return _Score(
            False, score,
            f"Likely requires proprietary data/infra ({proprietary_hits} proprietary indicators)"
        )
    return _Score(True, score, f"public_data={public_hits}, proprietary={proprietary_hits}")


# ---------------------------------------------------------------------------
# Filter 5: Staleness / crowding check
# ---------------------------------------------------------------------------
WELL_KNOWN_ANOMALIES = {
    "momentum": 1993,  # Jegadeesh & Titman
    "value": 1992,  # Fama-French
    "size": 1981,  # Banz
    "low volatility": 2006,  # Ang et al
    "carry": 2007,  # Koijen et al
    "post-earnings drift": 1968,  # Ball & Brown
    "accruals anomaly": 1996,  # Sloan
    "short-term reversal": 1990,  # Jegadeesh
    "pairs trading": 2006,  # Gatev et al
    "betting against beta": 2014,  # Frazzini & Pedersen
}


def _staleness_filter(paper: PaperMeta) -> _Score:
    """Has this anomaly been widely published for >10 years?"""
    text = f"{paper.title} {paper.abstract}".lower()
    current_year = date.today().year

    flagged_anomalies = []
    for anomaly, origin_year in WELL_KNOWN_ANOMALIES.items():
        if anomaly in text and (current_year - origin_year) > 10:
            flagged_anomalies.append(f"{anomaly} (since {origin_year})")

    if flagged_anomalies:
        # Flag but don't auto-reject - check if paper adds adaptation/novelty
        novelty_kw = [
            "novel", "new approach", "improvement", "enhanced", "adaptive",
            "machine learning", "deep learning", "cryptocurrency", "crypto",
            "alternative data", "regime", "dynamic",
        ]
        has_novelty = any(kw in text for kw in novelty_kw)
        score = 0.6 if has_novelty else 0.3

        return _Score(
            True, score,  # pass but flagged
            f"STALENESS FLAG: known anomalies [{', '.join(flagged_anomalies)}]"
            + ("; BUT paper adds novel adaptation" if has_novelty else "; no clear adaptation")
        )

    return _Score(True, 1.0, "No staleness concern")


# ---------------------------------------------------------------------------
# Composite filter
# ---------------------------------------------------------------------------
def apply_filters(paper: PaperMeta) -> FilterResult:
    """Run the full filter stack on a paper. Returns FilterResult."""
    scores: dict[str, dict] = {}

    # Run filters in order
    alpha = _alpha_filter(paper)
    scores["alpha"] = {"passed": alpha.passed, "score": alpha.score, "reason": alpha.reason}
    if not alpha.passed:
        return FilterResult(
            paper_id=paper.paper_id,
            verdict=FilterVerdict.FAIL_ALPHA,
            passed=False,
            rejection_reason=alpha.reason,
            filter_scores=scores,
        )

    rationale = _rationale_filter(paper)
    scores["rationale"] = {
        "passed": rationale.passed, "score": rationale.score, "reason": rationale.reason,
    }
    if not rationale.passed:
        return FilterResult(
            paper_id=paper.paper_id,
            verdict=FilterVerdict.FAIL_RATIONALE,
            passed=False,
            rejection_reason=rationale.reason,
            filter_scores=scores,
        )

    robustness = _robustness_filter(paper)
    scores["robustness"] = {
        "passed": robustness.passed, "score": robustness.score, "reason": robustness.reason,
    }
    if not robustness.passed:
        return FilterResult(
            paper_id=paper.paper_id,
            verdict=FilterVerdict.FAIL_ROBUSTNESS,
            passed=False,
            rejection_reason=robustness.reason,
            filter_scores=scores,
        )

    implementability = _implementability_filter(paper)
    scores["implementability"] = {
        "passed": implementability.passed,
        "score": implementability.score,
        "reason": implementability.reason,
    }
    if not implementability.passed:
        return FilterResult(
            paper_id=paper.paper_id,
            verdict=FilterVerdict.FAIL_IMPLEMENTABILITY,
            passed=False,
            rejection_reason=implementability.reason,
            filter_scores=scores,
        )

    staleness = _staleness_filter(paper)
    scores["staleness"] = {
        "passed": staleness.passed, "score": staleness.score, "reason": staleness.reason,
    }
    staleness_flag = "STALENESS FLAG" in staleness.reason

    return FilterResult(
        paper_id=paper.paper_id,
        verdict=FilterVerdict.PASS,
        passed=True,
        staleness_flag=staleness_flag,
        rejection_reason="",
        filter_scores=scores,
    )


def run_filters(papers: list[PaperMeta]) -> tuple[
    list[tuple[PaperMeta, FilterResult]],
    list[tuple[PaperMeta, FilterResult]],
]:
    """Run filter stack on all papers.

    Returns (passed, rejected) tuples of (PaperMeta, FilterResult).
    """
    print("\n" + "=" * 60)
    print("PHASE 2: FILTER STACK")
    print("=" * 60)

    passed: list[tuple[PaperMeta, FilterResult]] = []
    rejected: list[tuple[PaperMeta, FilterResult]] = []

    for paper in papers:
        result = apply_filters(paper)
        if result.passed:
            passed.append((paper, result))
            flag = " [STALENESS]" if result.staleness_flag else ""
            print(f"  PASS{flag}: {paper.title[:70]}")
        else:
            rejected.append((paper, result))
            print(f"  REJECT ({result.verdict.value}): {paper.title[:60]}  -- {result.rejection_reason[:60]}")

    print(f"\nFilter results: {len(passed)} passed, {len(rejected)} rejected")
    return passed, rejected
