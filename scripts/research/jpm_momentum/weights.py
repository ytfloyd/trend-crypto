"""
Position sizing for JPM Momentum research.

Three methods from the paper:
- **Equal-weight**: 1/N across selected assets.
- **Inverse-volatility**: w_i = (1/vol_i) / sum(1/vol_j).
- **Risk-parity**: target equal risk contribution (simplified: same as inv-vol
  for uncorrelated assets).

All functions produce wide-format weight DataFrames (index=ts, columns=symbols).
Supports both crypto (ann_factor=365) and ETF (ann_factor=252) markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ANN_FACTOR_CRYPTO


def equal_weight(
    mask: pd.DataFrame,
) -> pd.DataFrame:
    """Equal-weight: 1/N across all True entries in *mask*.

    Parameters
    ----------
    mask : pd.DataFrame
        Wide-format boolean mask (index=ts, columns=symbols).  True = selected.

    Returns
    -------
    pd.DataFrame
        Wide-format weights (same shape), summing to 1.0 per row (or 0 if none).
    """
    n = mask.sum(axis=1).replace(0, np.nan)
    return mask.astype(float).div(n, axis=0).fillna(0.0)


def inverse_volatility(
    mask: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_lookback: int = 42,
    vol_floor: float = 0.10,
    ann_factor: float = ANN_FACTOR_CRYPTO,
) -> pd.DataFrame:
    """Inverse-volatility weighting: w_i = (1/vol_i) / sum(1/vol_j).

    Parameters
    ----------
    mask : pd.DataFrame
        Wide-format boolean selection mask.
    returns_wide : pd.DataFrame
        Wide-format daily returns (same column universe as mask).
    vol_lookback : int
        Rolling window for realised vol estimation.
    vol_floor : float
        Minimum annualised vol to prevent extreme weights.
    ann_factor : float
        Annualisation factor (365 for crypto, 252 for ETFs).

    Returns
    -------
    pd.DataFrame
        Wide-format weights.
    """
    common = mask.columns.intersection(returns_wide.columns)
    m = mask[common].reindex(returns_wide.index).fillna(False)
    r = returns_wide[common].reindex(m.index).fillna(0.0)

    vol_ann = r.rolling(vol_lookback, min_periods=max(10, vol_lookback // 2)).std() * np.sqrt(ann_factor)
    vol_ann = vol_ann.clip(lower=vol_floor)

    inv_vol = (1.0 / vol_ann).where(m, 0.0)
    row_sum = inv_vol.sum(axis=1).replace(0, np.nan)
    return inv_vol.div(row_sum, axis=0).fillna(0.0)


def risk_parity(
    mask: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_lookback: int = 42,
    vol_floor: float = 0.10,
    ann_factor: float = ANN_FACTOR_CRYPTO,
) -> pd.DataFrame:
    """Simplified risk-parity weighting.

    For long-only portfolios of (approximately) uncorrelated assets, risk-parity
    reduces to inverse-volatility weighting.  This is the paper's recommended
    approach when full covariance estimation is noisy.

    Parameters are identical to :func:`inverse_volatility`.
    """
    return inverse_volatility(
        mask, returns_wide, vol_lookback=vol_lookback, vol_floor=vol_floor,
        ann_factor=ann_factor,
    )


def build_weights(
    mask: pd.DataFrame,
    returns_wide: pd.DataFrame,
    method: str = "equal",
    vol_lookback: int = 42,
    vol_floor: float = 0.10,
    ann_factor: float = ANN_FACTOR_CRYPTO,
) -> pd.DataFrame:
    """Dispatch to the appropriate weighting method.

    Parameters
    ----------
    mask : pd.DataFrame
        Wide-format boolean selection mask.
    returns_wide : pd.DataFrame
        Wide-format daily returns.
    method : str
        One of: ``equal``, ``inv_vol``, ``risk_parity``.
    vol_lookback : int
        Rolling window for vol estimation (for inv_vol / risk_parity).
    vol_floor : float
        Minimum annualised vol (for inv_vol / risk_parity).
    ann_factor : float
        Annualisation factor (365 for crypto, 252 for ETFs).

    Returns
    -------
    pd.DataFrame
        Wide-format weights, summing to ~1.0 per row.
    """
    if method == "equal":
        return equal_weight(mask)
    elif method == "inv_vol":
        return inverse_volatility(mask, returns_wide, vol_lookback, vol_floor, ann_factor)
    elif method == "risk_parity":
        return risk_parity(mask, returns_wide, vol_lookback, vol_floor, ann_factor)
    else:
        raise ValueError(f"Unknown weighting method {method!r}. Choose from: equal, inv_vol, risk_parity")
