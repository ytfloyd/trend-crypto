"""Cboe Global Indices available on the public EOD CDN.

Probe: ``https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYMBOL}_History.csv``

Symbols returning HTTP 403 as of Jun 2026 (not on CDN): DSPBX, COR3MD, VXNFLX,
VXTSLA, and most additional single-name VX* tickers beyond those listed below.
"""
from __future__ import annotations

# Equity-index implied volatility (levels and term structure)
EQUITY_VOL: tuple[str, ...] = (
    "VIX",      # SPX 30d
    "VVIX",     # vol of VIX
    "VIX9D",    # SPX 9d
    "VIX1D",    # SPX 1d
    "VIX3M",    # SPX 3m
    "VIX6M",    # SPX 6m
    "VIX1Y",    # SPX 1y
    "VXO",      # legacy S&P 100 methodology
    "VXN",      # Nasdaq-100
    "RVX",      # Russell 2000
    "SKEW",     # SPX tail risk
    "SMILE",    # SPX volatility smile
)

# Dispersion / constituent-vol complex
DISPERSION: tuple[str, ...] = (
    "VIXEQ",    # SPX constituent weighted 30d IV
    "DSPX",     # SPX implied dispersion
)

# Implied correlation (SPX top-50 basket)
CORRELATION_TENOR: tuple[str, ...] = (
    "COR1M",
    "COR3M",
    "COR6M",
    "COR9M",
    "COR1Y",
)

CORRELATION_SKEW: tuple[str, ...] = (
    "COR10D",
    "COR30D",
    "COR70D",
    "COR90D",
)

# Cross-asset / sector volatility
COMMODITY_VOL: tuple[str, ...] = (
    "OVX",      # crude oil ETF
    "GVZ",      # gold ETF
)

RATES_VOL: tuple[str, ...] = (
    "TYVIX",    # 10y Treasury
    "VXTLT",    # TLT ETF
)

THEMATIC_VOL: tuple[str, ...] = (
    "VXTH",     # semiconductors
    "VXEFA",    # EAFE
    "VXEEM",    # emerging markets
    "VXHYG",    # high yield
    "VXSLV",    # silver
    "VXGDX",    # gold miners
)

SINGLE_NAME_VOL: tuple[str, ...] = (
    "VXAPL",
    "VXAZN",
    "VXGOG",
    "VXGS",
    "VXIBM",
    "VXFXI",
    "VXBAC",
    "VXWMT",
)

VOL_CORRELATION_UNIVERSE: tuple[str, ...] = (
    *EQUITY_VOL,
    *DISPERSION,
    *CORRELATION_TENOR,
    *CORRELATION_SKEW,
    *COMMODITY_VOL,
    *RATES_VOL,
    *THEMATIC_VOL,
    *SINGLE_NAME_VOL,
)


def get_vol_correlation_universe() -> list[str]:
    return list(VOL_CORRELATION_UNIVERSE)
