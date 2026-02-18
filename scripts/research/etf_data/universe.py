"""
Curated ETF universe definitions for research.

Provides ticker lists organised by asset class, matching the multi-asset
universe used in Kolanovic & Wei (2015) "Momentum Strategies Across
Asset Classes".  All ETFs are US-listed, highly liquid, and have long
enough histories for robust backtesting (most from 2005-2010+).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Asset-class buckets
# ---------------------------------------------------------------------------
US_EQUITY_SECTORS: list[str] = [
    "XLB",   # Materials
    "XLC",   # Communication Services (launched 2018)
    "XLE",   # Energy
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLK",   # Technology
    "XLP",   # Consumer Staples
    "XLRE",  # Real Estate (launched 2015)
    "XLU",   # Utilities
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
]

US_EQUITY_STYLE: list[str] = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq-100
    "IWM",   # Russell 2000
    "IWD",   # Russell 1000 Value
    "IWF",   # Russell 1000 Growth
    "MDY",   # S&P MidCap 400
    "VTV",   # Vanguard Value
    "VUG",   # Vanguard Growth
    "IJR",   # S&P SmallCap 600
]

INTERNATIONAL_EQUITY: list[str] = [
    "EFA",   # MSCI EAFE (Developed ex-US)
    "EEM",   # MSCI Emerging Markets
    "VGK",   # FTSE Europe
    "VPL",   # FTSE Pacific
    "EWJ",   # MSCI Japan
    "EWZ",   # MSCI Brazil
    "EWH",   # MSCI Hong Kong
    "FXI",   # FTSE China 50
    "INDA",  # MSCI India
    "VWO",   # FTSE Emerging Markets
]

FIXED_INCOME: list[str] = [
    "TLT",   # 20+ Year Treasury
    "IEF",   # 7-10 Year Treasury
    "SHY",   # 1-3 Year Treasury
    "AGG",   # US Aggregate Bond
    "LQD",   # Investment Grade Corporate
    "HYG",   # High Yield Corporate
    "TIP",   # TIPS (Inflation-Protected)
    "BND",   # Total Bond Market
    "EMB",   # Emerging Market Bonds
    "MBB",   # Mortgage-Backed Securities
]

COMMODITIES: list[str] = [
    "GLD",   # Gold
    "SLV",   # Silver
    "USO",   # Crude Oil
    "DBA",   # Agriculture
    "DBC",   # Broad Commodities
    "PDBC",  # Optimum Yield Diversified Commodity
    "IAU",   # Gold (iShares)
]

REAL_ESTATE: list[str] = [
    "VNQ",   # US REITs
    "IYR",   # US Real Estate
    "REM",   # Mortgage REITs
]

MULTI_ASSET: list[str] = [
    "AOM",   # Moderate Allocation
    "AOA",   # Aggressive Allocation
    "AOK",   # Conservative Allocation
]


# ---------------------------------------------------------------------------
# Composite universes
# ---------------------------------------------------------------------------
def get_full_universe() -> list[str]:
    """Return all ~64 ETFs across all asset classes."""
    return sorted(set(
        US_EQUITY_SECTORS
        + US_EQUITY_STYLE
        + INTERNATIONAL_EQUITY
        + FIXED_INCOME
        + COMMODITIES
        + REAL_ESTATE
        + MULTI_ASSET
    ))


def get_core_universe() -> list[str]:
    """Return a smaller ~40-ETF universe of the most liquid, longest-history ETFs.

    Excludes newer ETFs (XLC, XLRE, INDA, PDBC) and niche tickers.
    Good for backtests starting from 2006+.
    """
    return sorted(set(
        ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
        + ["SPY", "QQQ", "IWM", "IWD", "IWF", "MDY"]
        + ["EFA", "EEM", "VGK", "EWJ", "EWZ", "FXI", "VWO"]
        + ["TLT", "IEF", "SHY", "AGG", "LQD", "HYG", "TIP"]
        + ["GLD", "SLV", "USO", "DBA", "DBC"]
        + ["VNQ", "IYR"]
    ))


def get_asset_class_label(ticker: str) -> str:
    """Return the asset-class label for a given ticker."""
    if ticker in US_EQUITY_SECTORS:
        return "us_equity_sector"
    if ticker in US_EQUITY_STYLE:
        return "us_equity_style"
    if ticker in INTERNATIONAL_EQUITY:
        return "intl_equity"
    if ticker in FIXED_INCOME:
        return "fixed_income"
    if ticker in COMMODITIES:
        return "commodity"
    if ticker in REAL_ESTATE:
        return "real_estate"
    if ticker in MULTI_ASSET:
        return "multi_asset"
    return "unknown"


def get_universe_with_labels(universe: list[str] | None = None) -> list[dict]:
    """Return a list of {ticker, asset_class} dicts for the given universe."""
    if universe is None:
        universe = get_full_universe()
    return [{"ticker": t, "asset_class": get_asset_class_label(t)} for t in universe]
