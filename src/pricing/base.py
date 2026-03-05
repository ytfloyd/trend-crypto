"""Pricing model protocol and shared data structures.

Defines the contract that all pricing models must satisfy, plus
lightweight containers for option contracts and Greeks.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class OptionType(Enum):
    CALL = "C"
    PUT = "P"


@dataclass(frozen=True)
class OptionContract:
    """Minimal option contract descriptor.

    Attributes
    ----------
    underlying : underlying identifier (e.g. "CL", "ES", "BTC-USD").
    strike : strike price.
    expiry_tte : time to expiry in years.
    option_type : CALL or PUT.
    multiplier : contract multiplier (e.g. 1000 for CL, 50 for ES).
    """

    underlying: str
    strike: float
    expiry_tte: float
    option_type: OptionType
    multiplier: float = 1.0


@dataclass
class Greeks:
    """Container for first- and second-order sensitivities.

    All values are per-unit (one option contract, multiplier=1).
    Scale by position * multiplier for portfolio-level Greeks.

    Natenberg Ch. 8-13: each Greek measures sensitivity to one
    risk factor, holding all others constant.
    """

    price: float
    delta: float
    gamma: float
    theta: float      # per calendar day
    vega: float       # per 1 point of vol (e.g. 0.01 = 1 vol point)
    rho: float = 0.0
    vanna: float = 0.0   # d(delta)/d(vol)
    volga: float = 0.0   # d(vega)/d(vol), aka vomma

    def scale(self, quantity: float, multiplier: float = 1.0) -> Greeks:
        """Scale all Greeks by position size and contract multiplier."""
        m = quantity * multiplier
        return Greeks(
            price=self.price * m,
            delta=self.delta * m,
            gamma=self.gamma * m,
            theta=self.theta * m,
            vega=self.vega * m,
            rho=self.rho * m,
            vanna=self.vanna * m,
            volga=self.volga * m,
        )


@runtime_checkable
class PricingModel(Protocol):
    """Protocol that all pricing models must implement."""

    def price(
        self,
        forward: float,
        strike: float,
        tte: float,
        vol: float,
        rate: float,
        is_call: bool,
    ) -> float:
        """Compute option price."""
        ...

    def greeks(
        self,
        forward: float,
        strike: float,
        tte: float,
        vol: float,
        rate: float,
        is_call: bool,
    ) -> Greeks:
        """Compute all Greeks."""
        ...

    def implied_vol(
        self,
        forward: float,
        strike: float,
        tte: float,
        price: float,
        rate: float,
        is_call: bool,
    ) -> float:
        """Invert from market price to implied volatility."""
        ...
