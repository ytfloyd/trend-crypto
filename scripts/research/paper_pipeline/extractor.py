"""Phase 3: Strategy specification extraction from papers.

Extracts structured strategy details from paper abstracts and metadata.
For full-text extraction, extend with PDF parsing.
"""
from __future__ import annotations

import re

from .models import FilterResult, PaperMeta, StrategySpec, StrategyType

# ---------------------------------------------------------------------------
# Signal / rule extraction heuristics
# ---------------------------------------------------------------------------
SIGNAL_PATTERNS: dict[StrategyType, str] = {
    StrategyType.MOMENTUM: (
        "Trend-following / momentum signal. Likely uses price returns over a lookback "
        "window to determine signal direction."
    ),
    StrategyType.MEAN_REVERSION: (
        "Mean-reversion signal. Likely compares current price to a fair-value estimate "
        "(moving average, z-score, or cointegration residual)."
    ),
    StrategyType.STAT_ARB: (
        "Statistical arbitrage signal. Likely uses relative pricing between correlated "
        "assets (spread, ratio, or cointegration-based)."
    ),
    StrategyType.VOLATILITY: (
        "Volatility signal. Likely uses implied vs realized vol spread or vol surface "
        "dynamics."
    ),
    StrategyType.FACTOR: (
        "Factor-based signal. Uses fundamental or market-derived factor scores "
        "(value, quality, size, etc.)."
    ),
    StrategyType.SEASONAL: (
        "Calendar / seasonal signal. Based on recurring temporal patterns."
    ),
    StrategyType.ML_SIGNAL: (
        "Machine learning signal. Uses trained model predictions (features from "
        "price, volume, or alternative data)."
    ),
    StrategyType.EXECUTION: (
        "Execution optimization signal. Focused on trade timing and cost minimization."
    ),
    StrategyType.OTHER: "Signal type unclear from abstract. Requires full-text review.",
}

REBALANCE_HEURISTICS = {
    "daily": ["daily", "each day", "every day", "close-to-close"],
    "weekly": ["weekly", "each week", "every week"],
    "monthly": ["monthly", "each month", "every month", "month-end"],
    "quarterly": ["quarterly", "each quarter"],
    "intraday": ["intraday", "intra-day", "hourly", "minute"],
}


def _detect_rebalance(text: str) -> str:
    """Guess rebalance frequency from text."""
    text_lower = text.lower()
    for freq, keywords in REBALANCE_HEURISTICS.items():
        if any(kw in text_lower for kw in keywords):
            return freq
    return "unclear - requires full paper review"


def _extract_numeric(text: str, pattern: str) -> float | None:
    """Extract a numeric value using a regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            pass
    return None


def _detect_data_requirements(paper: PaperMeta) -> list[str]:
    """Infer data requirements from abstract."""
    text = f"{paper.title} {paper.abstract}".lower()
    reqs = []

    if any(kw in text for kw in ["daily price", "close price", "ohlc", "daily return"]):
        reqs.append("daily OHLCV bars")
    if any(kw in text for kw in ["volume", "dollar volume", "turnover"]):
        reqs.append("volume data")
    if any(kw in text for kw in ["fundamental", "earnings", "book value", "balance sheet"]):
        reqs.append("fundamental data")
    if any(kw in text for kw in ["option", "implied vol", "vol surface"]):
        reqs.append("options chain data")
    if any(kw in text for kw in ["sentiment", "twitter", "news", "social media"]):
        reqs.append("sentiment / alternative data")
    if any(kw in text for kw in ["order book", "depth", "bid-ask"]):
        reqs.append("L2 order book data")
    if any(kw in text for kw in ["intraday", "minute", "tick"]):
        reqs.append("intraday tick/minute data")

    if not reqs:
        reqs.append("daily price data (assumed)")

    return reqs


def extract_strategy(paper: PaperMeta, filter_result: FilterResult) -> StrategySpec:
    """Extract a structured strategy specification from paper metadata.

    This uses heuristic extraction from the abstract. For production use,
    extend with full-text PDF parsing and LLM-assisted extraction.
    """
    text = f"{paper.title} {paper.abstract}"
    text_lower = text.lower()

    signal_desc = SIGNAL_PATTERNS.get(paper.strategy_type, SIGNAL_PATTERNS[StrategyType.OTHER])

    # Entry / exit heuristics
    entry_rule = "Long when signal is positive" if paper.strategy_type in {
        StrategyType.MOMENTUM, StrategyType.FACTOR, StrategyType.ML_SIGNAL,
    } else "Enter when signal exceeds threshold"

    exit_rule = "Exit on signal reversal or stop-loss"

    if "stop" in text_lower or "stop-loss" in text_lower:
        exit_rule += " (explicit stop-loss mentioned)"
    if "trailing" in text_lower:
        exit_rule += " (trailing stop mentioned)"
    if "take profit" in text_lower or "profit target" in text_lower:
        exit_rule += " (profit target mentioned)"

    # Position sizing
    pos_sizing = "Equal weight (default assumption)"
    if "inverse volatility" in text_lower or "risk parity" in text_lower:
        pos_sizing = "Inverse volatility / risk parity"
    elif "kelly" in text_lower:
        pos_sizing = "Kelly criterion"
    elif "signal strength" in text_lower or "conviction" in text_lower:
        pos_sizing = "Signal-strength weighted"

    # Risk management
    risk_mgmt = "Not explicitly described"
    risk_kws = []
    if "drawdown" in text_lower or "max drawdown" in text_lower:
        risk_kws.append("drawdown monitoring")
    if "vol target" in text_lower or "volatility target" in text_lower:
        risk_kws.append("volatility targeting")
    if "stop-loss" in text_lower or "stop loss" in text_lower:
        risk_kws.append("stop-loss")
    if "position limit" in text_lower or "concentration" in text_lower:
        risk_kws.append("position limits")
    if risk_kws:
        risk_mgmt = "; ".join(risk_kws)

    # TCA
    tca = "Not mentioned"
    if "transaction cost" in text_lower or "trading cost" in text_lower:
        bps_match = re.search(r"(\d+)\s*(?:bps|basis point)", text_lower)
        if bps_match:
            tca = f"Modeled at {bps_match.group(1)} bps"
        else:
            tca = "Transaction costs considered (level unspecified)"

    # Claimed metrics
    claimed_sharpe = _extract_numeric(text, r"sharpe\s*(?:ratio)?\s*(?:of|=|:)?\s*([\d.]+)")
    claimed_cagr = _extract_numeric(text, r"(?:cagr|annual.*return)\s*(?:of|=|:)?\s*([\d.]+)")
    claimed_max_dd = _extract_numeric(text, r"(?:max.*drawdown|maximum.*drawdown)\s*(?:of|=|:)?\s*-?([\d.]+)")

    # OOS / multi-market from filter scores
    robustness = filter_result.filter_scores.get("robustness", {})
    has_oos = "OOS=True" in robustness.get("reason", "")
    has_multi = "multi-market=True" in robustness.get("reason", "")

    return StrategySpec(
        paper_id=paper.paper_id,
        strategy_name=f"{paper.strategy_type.value}::{paper.title[:60]}",
        strategy_type=paper.strategy_type,
        asset_classes=paper.asset_classes,
        signal_description=signal_desc,
        entry_rule=entry_rule,
        exit_rule=exit_rule,
        position_sizing=pos_sizing,
        rebalance_frequency=_detect_rebalance(text),
        data_requirements=_detect_data_requirements(paper),
        risk_management=risk_mgmt,
        transaction_costs_noted=tca,
        claimed_sharpe=claimed_sharpe,
        claimed_cagr=claimed_cagr,
        claimed_max_dd=claimed_max_dd,
        data_period=paper.data_period,
        out_of_sample=has_oos,
        multi_market=has_multi,
        notes=f"Staleness: {filter_result.staleness_flag}. "
              f"Auto-extracted from abstract; verify with full paper.",
    )


def run_extraction(
    passed_papers: list[tuple[PaperMeta, FilterResult]],
) -> list[tuple[PaperMeta, FilterResult, StrategySpec]]:
    """Extract strategy specs for all papers that passed filters.

    Returns list of (paper, filter_result, strategy_spec) tuples.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: STRATEGY EXTRACTION")
    print("=" * 60)

    results = []
    for paper, filt in passed_papers:
        spec = extract_strategy(paper, filt)
        results.append((paper, filt, spec))
        print(f"  Extracted: {spec.strategy_name[:70]}")
        print(f"    Type: {spec.strategy_type.value} | Assets: {spec.asset_classes}")
        print(f"    Rebalance: {spec.rebalance_frequency} | OOS: {spec.out_of_sample}")

    print(f"\nExtracted {len(results)} strategy specifications.")
    return results
