"""Bachelier (normal) model for interest rate options.

Used for swaptions, caps/floors, and any option where the underlying
can go negative (rates).  The underlying follows arithmetic Brownian
motion rather than geometric.  Vol is quoted in absolute terms (bps
or percentage points) rather than as a percentage of spot.

This is the standard model for SOFR options, Treasury options,
and most rates vol surfaces.

Reference: Louis Bachelier (1900); Natenberg discusses the normal
           model as an alternative to lognormal in Ch. 4-5.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from .base import Greeks


def bach_price(
    forward: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    is_call: bool,
) -> float:
    """Bachelier (normal model) option price.

    Parameters
    ----------
    vol : normal (absolute) volatility, same units as forward/strike.
    """
    if tte <= 0 or vol <= 0:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        return intrinsic * np.exp(-rate * max(tte, 0.0))

    vol_sqrt_t = vol * np.sqrt(tte)
    d = (forward - strike) / vol_sqrt_t
    df = np.exp(-rate * tte)

    if is_call:
        return df * (vol_sqrt_t * (d * norm.cdf(d) + norm.pdf(d)))
    else:
        return df * (vol_sqrt_t * (-d * norm.cdf(-d) + norm.pdf(d)))


def bach_greeks(
    forward: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    is_call: bool,
) -> Greeks:
    """Bachelier Greeks."""
    if tte <= 0 or vol <= 0:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        d = 1.0 if (is_call and forward > strike) else (-1.0 if (not is_call and forward < strike) else 0.0)
        return Greeks(
            price=intrinsic * np.exp(-rate * max(tte, 0.0)),
            delta=d * np.exp(-rate * max(tte, 0.0)),
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
        )

    vol_sqrt_t = vol * np.sqrt(tte)
    d = (forward - strike) / vol_sqrt_t
    df = np.exp(-rate * tte)
    sqrt_t = np.sqrt(tte)

    price = bach_price(forward, strike, tte, vol, rate, is_call)

    if is_call:
        delta = df * norm.cdf(d)
    else:
        delta = -df * norm.cdf(-d)

    gamma = df * norm.pdf(d) / vol_sqrt_t

    theta_term = -vol * norm.pdf(d) / (2.0 * sqrt_t)
    theta = (df * theta_term - rate * price) / 365.0

    # Vega: sensitivity to 1bp of normal vol
    vega = df * sqrt_t * norm.pdf(d) * 0.0001

    rho = -tte * price * 0.01

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
    )


def bach_iv(
    forward: float,
    strike: float,
    tte: float,
    price: float,
    rate: float,
    is_call: bool,
    tol: float = 1e-10,
    max_vol: float = 1000.0,
) -> float:
    """Bachelier implied (normal) volatility via Brent's method."""
    if tte <= 0 or price <= 0:
        return np.nan

    def objective(v: float) -> float:
        return bach_price(forward, strike, tte, v, rate, is_call) - price

    try:
        return brentq(objective, 1e-8, max_vol, xtol=tol)
    except ValueError:
        return np.nan


class Bachelier:
    """Bachelier pricing model (satisfies PricingModel protocol)."""

    def price(
        self, forward: float, strike: float, tte: float,
        vol: float, rate: float, is_call: bool,
    ) -> float:
        return bach_price(forward, strike, tte, vol, rate, is_call)

    def greeks(
        self, forward: float, strike: float, tte: float,
        vol: float, rate: float, is_call: bool,
    ) -> Greeks:
        return bach_greeks(forward, strike, tte, vol, rate, is_call)

    def implied_vol(
        self, forward: float, strike: float, tte: float,
        price: float, rate: float, is_call: bool,
    ) -> float:
        return bach_iv(forward, strike, tte, price, rate, is_call)
