"""Curated equity universes for the gamma screener.

S&P 100 (OEF) is the default: 100 most heavily-traded US large caps,
all with deep options markets. Rarely changes; snapshot as of 2024/2025.

For v0 we stay on the S&P 100 to keep IB snapshot times reasonable
(~8 minutes for 100 names at 4-5 sec/ticker). Expand to S&P 500 once
the pipeline is stable.
"""
from __future__ import annotations

import json
from pathlib import Path

# ── S&P 100 (OEF) — stable large-cap set ──────────────────────────────
SP100: tuple[str, ...] = (
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU",
    "ISRG", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
    "TSLA", "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC",
    "WMT", "XOM",
)


def get_sp500(project_root: Path) -> tuple[str, ...]:
    """Load S&P 500 tickers from the repo's cached JSON."""
    path = project_root / "scripts" / "research" / "sp500_tickers.json"
    data = json.loads(path.read_text())
    return tuple(data["tickers"])


def get_universe(name: str, project_root: Path) -> tuple[str, ...]:
    """Resolve a universe name to a ticker tuple.

    Accepts: 'sp100', 'sp500'. For 'custom', callers pass tickers directly.
    """
    name = name.lower()
    if name == "sp100":
        return SP100
    if name == "sp500":
        return get_sp500(project_root)
    raise ValueError(f"Unknown universe '{name}'. Use 'sp100' or 'sp500'.")
