"""
Sizing module for the futures-only silver replicator.

Each sizing function returns an integer pd.Series of *signed* contract counts
indexed by bar timestamp, suitable to be fed into
`backtest_unscaled.simulate_futures_pnl`.

Three modes are provided:

    * fixed(n_contracts)               — same |N| contracts when state != 0
    * vol_target(target_dollar_vol_per_day, atr_dollar, bars_per_day)
                                        — size = target_$_per_day /
                                          (ATR_$ * sqrt(bars_per_day))
                                          where ATR_$ = ATR_per_oz * QI_MULT
    * confidence_scaled(base_contracts, max_contracts, confidence)
                                        — interpolates base..max by a 0..1
                                          confidence score (#filters firing)

Contract size is in QI mini-silver units (2,500 oz multiplier).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

QI_MULT = 2_500          # $ per $1/oz move per QI contract


# ---------------------------------------------------------------- fixed --


def fixed(state: pd.Series, n_contracts: int) -> pd.Series:
    """Always size |N| contracts when state != 0."""
    n = int(n_contracts)
    contracts = (state.astype(int) * n)
    return contracts.astype("int32").rename("contracts")


# --------------------------------------------------------- vol_target --


def vol_target(
    state: pd.Series,
    atr_per_oz: pd.Series,
    target_dollar_vol_per_day: float,
    bars_per_day: float,
    *,
    cap_contracts: int = 32,
    floor_contracts: int = 1,
) -> pd.Series:
    """
    Vol-targeted sizing.

    contracts = round( target_$_vol_per_day / (ATR_$ * sqrt(bars_per_day)) ),
    where ATR_$ = ATR_per_oz * 2500 (QI mini multiplier).

    The output is the signed integer contract count when state != 0, else 0.
    Cap and floor keep things sane when ATR collapses or explodes.
    """
    atr_dollar = atr_per_oz.astype(float) * QI_MULT
    denom = atr_dollar * np.sqrt(max(float(bars_per_day), 1e-9))
    raw = np.where(denom > 1e-9, float(target_dollar_vol_per_day) / denom, 0.0)
    sized = np.clip(np.round(raw), floor_contracts, int(cap_contracts))
    out = (state.astype(int).to_numpy() * sized.astype(int))
    return pd.Series(out.astype(np.int32), index=state.index, name="contracts")


# ---------------------------------------------------- confidence_scaled --


def _confidence_score(features: pd.DataFrame, p) -> pd.Series:
    """0..1 confidence: fraction of {trend, momentum, confirm, vol_gate} firing
    in agreement with the *raw* trend direction (independent of the state).

    Each filter contributes 0.25.
    """
    f = features
    sma_fast_col = f"sma_{p.fast}"
    sma_slow_col = f"sma_{p.slow}"
    sma_fast = f[sma_fast_col].ffill()
    sma_slow = f[sma_slow_col].ffill()
    rsi = f["rsi_14"].fillna(50.0)
    macd_hist = f["macd_hist"].fillna(0.0)
    adx = f["adx_14"].fillna(0.0)
    atr = f["atr_14"].fillna(0.0)
    close = f["c"].ffill()

    trend_long = (sma_fast > sma_slow).astype(int)
    trend_short = (sma_fast < sma_slow).astype(int)
    trend_dir = trend_long - trend_short  # -1, 0, +1

    mom_aligned = (
        ((trend_dir > 0) & (rsi >= p.rsi_long_thr))
        | ((trend_dir < 0) & (rsi <= p.rsi_short_thr))
    ).astype(int)

    confirm_aligned = (
        ((trend_dir > 0) & (macd_hist > 0))
        | ((trend_dir < 0) & (macd_hist < 0))
    ).astype(int)

    natr = (atr / close.replace(0, np.nan)).fillna(0.0)
    vol_gate = ((adx >= p.adx_min) | (natr <= p.atr_max)).astype(int)

    trend_present = (trend_dir != 0).astype(int)

    score = 0.25 * (trend_present + mom_aligned + confirm_aligned + vol_gate)
    return score.clip(0.0, 1.0)


def confidence_scaled(
    state: pd.Series,
    features: pd.DataFrame,
    signal_params,
    base_contracts: int,
    max_contracts: int,
) -> pd.Series:
    """
    Linearly interpolate contracts from `base_contracts` (low confidence) up to
    `max_contracts` (all four filter classes firing). Returns 0 when state==0.
    """
    conf = _confidence_score(features, signal_params).reindex(state.index).fillna(0.0)
    sized = np.round(
        float(base_contracts) + (float(max_contracts) - float(base_contracts)) * conf.values
    ).astype(int)
    sized = np.clip(sized, int(min(base_contracts, max_contracts)),
                    int(max(base_contracts, max_contracts)))
    out = state.astype(int).to_numpy() * sized
    return pd.Series(out.astype(np.int32), index=state.index, name="contracts")


# ------------------------------------------------------ utility helpers --


def bars_per_day(tf: str) -> float:
    """Approximate # bars per calendar day for our parquet bars."""
    return {
        "1H": 16.0,    # observed ~16h trading day
        "4H": 4.35,
        "8H": 2.26,
        "1D": 0.86,
    }[tf]


def bars_per_year(tf: str) -> float:
    """For Sharpe annualization."""
    return bars_per_day(tf) * 252.0
