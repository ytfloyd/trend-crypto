"""Black-Scholes model for European options on spot underlyings.

Suitable for equities, FX, and crypto spot options.  For futures
options use Black-76 instead.

The model assumes:
    - Continuous geometric Brownian motion
    - Constant vol and rates over the life of the option
    - No dividends (or use adjusted forward)
    - European exercise only

Reference: Natenberg, *Option Volatility and Pricing*, Ch. 1-5.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from .base import Greeks


# ── Scalar functions (fast, no object overhead) ──────────────────────────

def _d1d2(
    forward: float,
    strike: float,
    tte: float,
    vol: float,
) -> tuple[float, float]:
    vol_sqrt_t = vol * np.sqrt(tte)
    d1 = (np.log(forward / strike) + 0.5 * vol ** 2 * tte) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(
    forward: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    is_call: bool,
) -> float:
    """Black-Scholes price for a European option.

    Uses forward price (S * exp(rT)) to handle dividends/carry.
    """
    if tte <= 0 or vol <= 0:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        return intrinsic * np.exp(-rate * max(tte, 0.0))

    d1, d2 = _d1d2(forward, strike, tte, vol)
    df = np.exp(-rate * tte)

    if is_call:
        return df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        return df * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def bs_delta(
    forward: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    is_call: bool,
) -> float:
    """Black-Scholes delta (dV/dS, forward delta)."""
    if tte <= 0 or vol <= 0:
        if is_call:
            return 1.0 if forward > strike else 0.0
        else:
            return -1.0 if forward < strike else 0.0

    d1, _ = _d1d2(forward, strike, tte, vol)
    df = np.exp(-rate * tte)
    if is_call:
        return df * norm.cdf(d1)
    else:
        return df * (norm.cdf(d1) - 1.0)


def bs_greeks(
    forward: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    is_call: bool,
) -> Greeks:
    """Compute all Black-Scholes Greeks analytically."""
    if tte <= 0 or vol <= 0:
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        delta = bs_delta(forward, strike, tte, vol, rate, is_call)
        return Greeks(
            price=intrinsic * np.exp(-rate * max(tte, 0.0)),
            delta=delta,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            vanna=0.0,
            volga=0.0,
        )

    d1, d2 = _d1d2(forward, strike, tte, vol)
    df = np.exp(-rate * tte)
    sqrt_t = np.sqrt(tte)
    n_d1 = norm.pdf(d1)

    price = bs_price(forward, strike, tte, vol, rate, is_call)

    if is_call:
        delta = df * norm.cdf(d1)
    else:
        delta = df * (norm.cdf(d1) - 1.0)

    gamma = df * n_d1 / (forward * vol * sqrt_t)

    # Theta: per calendar day (divide annual by 365)
    theta_term1 = -(forward * vol * n_d1 * df) / (2.0 * sqrt_t)
    if is_call:
        theta_term2 = -rate * strike * df * norm.cdf(d2)
        theta_term3 = rate * forward * df * norm.cdf(d1)
    else:
        theta_term2 = rate * strike * df * norm.cdf(-d2)
        theta_term3 = -rate * forward * df * norm.cdf(-d1)
    theta = (theta_term1 + theta_term2 + theta_term3) / 365.0

    # Vega: per 1% vol move (multiply by 0.01)
    vega = forward * df * sqrt_t * n_d1 * 0.01

    # Rho: per 1% rate move
    if is_call:
        rho = strike * tte * df * norm.cdf(d2) * 0.01
    else:
        rho = -strike * tte * df * norm.cdf(-d2) * 0.01

    # Vanna: d(delta)/d(vol) per 1% vol
    vanna = -df * n_d1 * d2 / vol * 0.01

    # Volga (vomma): d(vega)/d(vol) per 1% vol
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


def bs_iv(
    forward: float,
    strike: float,
    tte: float,
    price: float,
    rate: float,
    is_call: bool,
    tol: float = 1e-8,
    max_vol: float = 5.0,
) -> float:
    """Implied vol via Brent's method.

    Searches for the vol that reproduces the given market price.
    Returns NaN if no solution is found.
    """
    if tte <= 0 or price <= 0:
        return np.nan

    intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    intrinsic *= np.exp(-rate * tte)
    if price < intrinsic - tol:
        return np.nan

    def objective(v: float) -> float:
        return bs_price(forward, strike, tte, v, rate, is_call) - price

    try:
        return brentq(objective, 1e-6, max_vol, xtol=tol)
    except ValueError:
        return np.nan


# ── Class wrapper (implements PricingModel protocol) ─────────────────────

class BlackScholes:
    """Black-Scholes pricing model as a class (satisfies PricingModel)."""

    def price(
        self, forward: float, strike: float, tte: float,
        vol: float, rate: float, is_call: bool,
    ) -> float:
        return bs_price(forward, strike, tte, vol, rate, is_call)

    def greeks(
        self, forward: float, strike: float, tte: float,
        vol: float, rate: float, is_call: bool,
    ) -> Greeks:
        return bs_greeks(forward, strike, tte, vol, rate, is_call)

    def implied_vol(
        self, forward: float, strike: float, tte: float,
        price: float, rate: float, is_call: bool,
    ) -> float:
        return bs_iv(forward, strike, tte, price, rate, is_call)
