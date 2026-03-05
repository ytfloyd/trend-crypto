"""Options pricing models and Greeks computation.

Provides Black-Scholes (equity/FX), Black-76 (futures/commodities),
and Bachelier (rates) pricing models with a unified Protocol interface.

Reference: Natenberg, *Option Volatility and Pricing*, Ch. 1-5, 8-13.
"""
from __future__ import annotations

from .base import PricingModel, Greeks, OptionContract, OptionType
from .black_scholes import BlackScholes, bs_price, bs_greeks, bs_delta, bs_iv
from .black76 import Black76, b76_price, b76_greeks, b76_iv
from .bachelier import Bachelier, bach_price, bach_greeks

__all__ = [
    "PricingModel",
    "Greeks",
    "OptionContract",
    "OptionType",
    "BlackScholes",
    "bs_price",
    "bs_greeks",
    "bs_delta",
    "bs_iv",
    "Black76",
    "b76_price",
    "b76_greeks",
    "b76_iv",
    "Bachelier",
    "bach_price",
    "bach_greeks",
]
