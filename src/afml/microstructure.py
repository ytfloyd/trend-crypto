"""
Market microstructure features.

Implements tools from AFML Chapter 16 (and related literature):

  - **VPIN** (Volume-synchronized Probability of Informed Trading):
    Estimates the fraction of volume driven by informed traders.
    High VPIN → toxic flow → potential adverse selection risk.

  - **Amihud illiquidity**: Price impact per unit volume.
    Higher values → less liquid.

  - **Kyle's lambda**: Regression-based price impact estimator.

  - **Roll spread**: Implicit bid-ask spread from serial covariance.

Reference:
    López de Prado, M. (2018) *Advances in Financial Machine Learning*,
    Chapter 18: Entropy Features, Chapter 19: Microstructural Features.
    Easley, D. et al. (2012) Flow Toxicity and Liquidity (VPIN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =====================================================================
# VPIN  (Volume-synchronized Probability of Informed Trading)
# =====================================================================

def vpin(
    close: pd.Series,
    volume: pd.Series,
    n_buckets: int = 50,
    window: int = 50,
) -> pd.DataFrame:
    """Compute VPIN (Volume-synchronized Probability of Informed Trading).

    Splits volume into equal-sized buckets, classifies each bar's
    volume as buy or sell (using the tick rule on close-to-close returns),
    then computes VPIN as the rolling average of absolute order imbalance
    normalised by bucket volume.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume per bar (same index as close).
    n_buckets : int
        Number of volume buckets.  Each bucket holds
        ``total_volume / n_buckets`` volume.
    window : int
        Rolling window (in buckets) for VPIN averaging.

    Returns
    -------
    pd.DataFrame with columns: vpin, bucket_end_date.
    """
    # Tick rule: classify volume as buy/sell
    returns = close.diff()
    signed = np.sign(returns).fillna(0)
    buy_volume = volume * ((1 + signed) / 2)
    sell_volume = volume * ((1 - signed) / 2)

    total_vol = volume.sum()
    bucket_size = total_vol / n_buckets

    # Fill buckets
    buckets = []
    cum_buy, cum_sell = 0.0, 0.0
    cum_vol = 0.0

    for i in range(len(volume)):
        cum_buy += buy_volume.iloc[i]
        cum_sell += sell_volume.iloc[i]
        cum_vol += volume.iloc[i]

        while cum_vol >= bucket_size and bucket_size > 0:
            overflow = cum_vol - bucket_size
            # Proportion of current bar in this bucket
            frac = 1 - overflow / volume.iloc[i] if volume.iloc[i] > 0 else 1

            bucket_buy = cum_buy - buy_volume.iloc[i] * (1 - frac)
            bucket_sell = cum_sell - sell_volume.iloc[i] * (1 - frac)

            buckets.append({
                "date": close.index[i],
                "buy_vol": bucket_buy,
                "sell_vol": bucket_sell,
                "bucket_vol": bucket_size,
            })

            # Carry overflow to next bucket
            cum_buy = buy_volume.iloc[i] * (1 - frac)
            cum_sell = sell_volume.iloc[i] * (1 - frac)
            cum_vol = overflow

    if len(buckets) == 0:
        return pd.DataFrame(columns=["vpin", "bucket_end_date"])

    df_buckets = pd.DataFrame(buckets)
    df_buckets["abs_imbalance"] = abs(df_buckets["buy_vol"] - df_buckets["sell_vol"])
    df_buckets["vpin"] = (
        df_buckets["abs_imbalance"].rolling(window).mean()
        / df_buckets["bucket_vol"]
    )

    return df_buckets[["date", "vpin"]].rename(columns={"date": "bucket_end_date"}).dropna()


# =====================================================================
# Amihud illiquidity
# =====================================================================

def amihud_illiquidity(
    close: pd.Series,
    volume: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Amihud (2002) illiquidity ratio.

    ``λ = E[ |r_t| / V_t ]``

    Higher values → less liquid (larger price impact per unit volume).

    Parameters
    ----------
    close : pd.Series
    volume : pd.Series
    window : int
        Rolling window for averaging.

    Returns
    -------
    pd.Series — rolling Amihud illiquidity.
    """
    returns = np.log(close / close.shift(1)).abs()
    dollar_volume = close * volume
    ratio = returns / dollar_volume.replace(0, np.nan)
    return ratio.rolling(window).mean()


# =====================================================================
# Kyle's lambda
# =====================================================================

def kyle_lambda(
    close: pd.Series,
    volume: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Rolling Kyle's lambda (price impact coefficient).

    Regresses absolute returns on signed root-volume:
    ``|Δp| = λ · sign(Δp) · sqrt(V) + ε``

    Higher lambda → more price impact per unit of trade.

    Parameters
    ----------
    close : pd.Series
    volume : pd.Series
    window : int

    Returns
    -------
    pd.Series — rolling lambda estimates.
    """
    dp = close.diff()
    signed_root_vol = np.sign(dp) * np.sqrt(volume)
    abs_dp = dp.abs()

    # Rolling OLS: lambda = cov(|dp|, signed_sqrt_v) / var(signed_sqrt_v)
    cov = abs_dp.rolling(window).cov(signed_root_vol)
    var = signed_root_vol.rolling(window).var()
    return (cov / var.replace(0, np.nan)).dropna()


# =====================================================================
# Roll spread
# =====================================================================

def roll_spread(
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Roll (1984) implied bid-ask spread.

    Estimates the effective spread from the serial covariance of
    price changes:  ``spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1}))``

    Only defined when the serial covariance is negative (most liquid
    assets).  Returns NaN when positive.

    Parameters
    ----------
    close : pd.Series
    window : int

    Returns
    -------
    pd.Series — rolling implied spread.
    """
    dp = close.diff()
    dp_lag = dp.shift(1)
    cov = dp.rolling(window).cov(dp_lag)
    spread = 2 * np.sqrt((-cov).clip(lower=0))
    spread[cov >= 0] = np.nan
    return spread
