"""Black-76 model for European options on futures.

The standard model for commodity options (CL, NG, GC, SI, ...),
interest rate options (Eurodollar, SOFR, Treasury), and any
exchange-traded futures option.

Black-76 differs from Black-Scholes only in that the underlying
is already a forward/futures price, so there is no cost-of-carry
adjustment.  The discount factor applies only to the payoff.

Reference: Natenberg, *Option Volatility and Pricing*, Ch. 3-5.
           Fischer Black, "The Pricing of Commodity Contracts" (1976).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from .base import Greeks


# ── Scalar functions ─────────────────────────────────────────────────────

def _d1d2(
    futures: float,
    strike: float,
    tte: float,
    vol: float,
) -> tuple[float, float]:
    vol_sqrt_t = vol * np.sqrt(tte)
    d1 = (np.log(futures / strike) + 0.5 * vol ** 2 * tte) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def b76_price(
    futures: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    is_call: bool,
) -> float:
    """Black-76 price for a European futures option."""
    if tte <= 0 or vol <= 0:
        intrinsic = max(futures - strike, 0.0) if is_call else max(strike - futures, 0.0)
        return intrinsic * np.exp(-rate * max(tte, 0.0))

    d1, d2 = _d1d2(futures, strike, tte, vol)
    df = np.exp(-rate * tte)

    if is_call:
        return df * (futures * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        return df * (strike * norm.cdf(-d2) - futures * norm.cdf(-d1))


def b76_greeks(
    futures: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    is_call: bool,
) -> Greeks:
    """Compute all Black-76 Greeks analytically."""
    if tte <= 0 or vol <= 0:
        intrinsic = max(futures - strike, 0.0) if is_call else max(strike - futures, 0.0)
        d = 1.0 if (is_call and futures > strike) else (-1.0 if (not is_call and futures < strike) else 0.0)
        return Greeks(
            price=intrinsic * np.exp(-rate * max(tte, 0.0)),
            delta=d * np.exp(-rate * max(tte, 0.0)),
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            vanna=0.0, volga=0.0,
        )

    d1, d2 = _d1d2(futures, strike, tte, vol)
    df = np.exp(-rate * tte)
    sqrt_t = np.sqrt(tte)
    n_d1 = norm.pdf(d1)

    price = b76_price(futures, strike, tte, vol, rate, is_call)

    if is_call:
        delta = df * norm.cdf(d1)
    else:
        delta = -df * norm.cdf(-d1)

    gamma = df * n_d1 / (futures * vol * sqrt_t)

    theta_intrinsic = -(futures * vol * n_d1 * df) / (2.0 * sqrt_t)
    theta_rate = -rate * price
    theta = (theta_intrinsic + theta_rate) / 365.0

    vega = futures * df * sqrt_t * n_d1 * 0.01

    rho = -tte * price * 0.01

    vanna = -df * n_d1 * d2 / vol * 0.01
    volga = vega * d1 * d2 / vol

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        vanna=vanna,
        volga=volga,
    )


def b76_iv(
    futures: float,
    strike: float,
    tte: float,
    price: float,
    rate: float,
    is_call: bool,
    tol: float = 1e-8,
    max_vol: float = 5.0,
) -> float:
    """Black-76 implied volatility via Brent's method."""
    if tte <= 0 or price <= 0:
        return np.nan

    intrinsic = max(futures - strike, 0.0) if is_call else max(strike - futures, 0.0)
    intrinsic *= np.exp(-rate * tte)
    if price < intrinsic - tol:
        return np.nan

    def objective(v: float) -> float:
        return b76_price(futures, strike, tte, v, rate, is_call) - price

    try:
        return brentq(objective, 1e-6, max_vol, xtol=tol)
    except ValueError:
        return np.nan


# ── Class wrapper ────────────────────────────────────────────────────────

class Black76:
    """Black-76 pricing model (satisfies PricingModel protocol)."""

    def price(
        self, forward: float, strike: float, tte: float,
        vol: float, rate: float, is_call: bool,
    ) -> float:
        return b76_price(forward, strike, tte, vol, rate, is_call)

    def greeks(
        self, forward: float, strike: float, tte: float,
        vol: float, rate: float, is_call: bool,
    ) -> Greeks:
        return b76_greeks(forward, strike, tte, vol, rate, is_call)

    def implied_vol(
        self, forward: float, strike: float, tte: float,
        price: float, rate: float, is_call: bool,
    ) -> float:
        return b76_iv(forward, strike, tte, price, rate, is_call)
