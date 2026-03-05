"""Carver-style position sizing: the canonical formula.

    position_weight = (vol_target * IDM * instrument_weight * forecast)
                      / (AVG_FORECAST * instrument_sigma)

This unifies volatility targeting, forecast strength, instrument
diversification, and portfolio allocation into a single formula.

For a long-only portfolio the position weight is floored at zero.

Reference: Robert Carver, *Systematic Trading*, Chapter 12.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy.forecast import TARGET_ABS_FORECAST


@dataclass
class CarverPositionSizer:
    """Full-formula position sizer per Carver's framework.

    Parameters
    ----------
    vol_target : annualised portfolio volatility target (e.g. 0.20 for 20%).
    idm : Instrument Diversification Multiplier (>= 1.0 when diversified).
    instrument_weight : this instrument's share of the portfolio (e.g. 0.33
        for a 3-instrument equal-weight portfolio).
    avg_forecast : the scaling constant, typically 10.0.
    max_position : hard cap on position weight (e.g. 2.0).
    long_only : if True, floor positions at 0.0.
    ann_factor : number of bars per year for annualising volatility.
    """

    vol_target: float = 0.20
    idm: float = 1.0
    instrument_weight: float = 1.0
    avg_forecast: float = TARGET_ABS_FORECAST
    max_position: float = 2.0
    long_only: bool = True
    ann_factor: float = 365.0

    def size(
        self,
        forecast: np.ndarray,
        instrument_sigma_ann: np.ndarray,
    ) -> np.ndarray:
        """Compute position weights from forecast and volatility.

        Parameters
        ----------
        forecast : 1-D array of combined, scaled, capped forecasts.
        instrument_sigma_ann : 1-D array of annualised instrument volatility
            (e.g. rolling 25-day daily return std * sqrt(365)).

        Returns
        -------
        1-D array of target position weights (notional fraction of capital).
        """
        sigma = np.where(
            (instrument_sigma_ann <= 0) | ~np.isfinite(instrument_sigma_ann),
            np.nan,
            instrument_sigma_ann,
        )
        position = (
            self.vol_target
            * self.idm
            * self.instrument_weight
            * forecast
        ) / (self.avg_forecast * sigma)

        position = np.where(np.isfinite(position), position, 0.0)

        if self.long_only:
            position = np.maximum(position, 0.0)

        return np.clip(position, -self.max_position, self.max_position)

    def size_series(
        self,
        forecast: pd.Series,
        instrument_sigma_ann: pd.Series,
    ) -> pd.Series:
        """Pandas wrapper around ``size()``."""
        pos = self.size(forecast.to_numpy(), instrument_sigma_ann.to_numpy())
        return pd.Series(pos, index=forecast.index, name="target_weight")


def compute_instrument_sigma_ann(
    returns: pd.Series | np.ndarray,
    lookback: int = 25,
    ann_factor: float = 365.0,
) -> np.ndarray:
    """Rolling annualised volatility of an instrument's returns.

    Uses exponentially weighted std for a Carver-consistent estimate.
    """
    s = pd.Series(returns)
    ewm_std = s.ewm(span=lookback, min_periods=max(10, lookback // 2)).std()
    return (ewm_std * np.sqrt(ann_factor)).to_numpy()


# ── Portfolio-level sizing (multi-instrument) ────────────────────────────
def size_portfolio(
    forecasts: dict[str, np.ndarray],
    returns_wide: pd.DataFrame,
    instrument_weights: dict[str, float],
    vol_target: float = 0.20,
    idm: float | np.ndarray = 1.0,
    vol_lookback: int = 25,
    ann_factor: float = 365.0,
    max_position: float = 2.0,
    long_only: bool = True,
) -> pd.DataFrame:
    """Size positions for all instruments at once.

    Parameters
    ----------
    forecasts : dict mapping symbol to 1-D forecast array.
    returns_wide : DataFrame, index=ts, columns=symbols, values=returns.
    instrument_weights : dict mapping symbol to portfolio weight fraction.
    vol_target : annualised portfolio vol target.
    idm : scalar or time-varying IDM array.
    vol_lookback : lookback for per-instrument vol estimate.
    ann_factor : annualisation factor.
    max_position : hard cap per instrument.
    long_only : floor at zero.

    Returns
    -------
    DataFrame of target position weights, same shape as returns_wide.
    """
    positions = pd.DataFrame(0.0, index=returns_wide.index, columns=returns_wide.columns)

    idm_arr = np.broadcast_to(
        np.asarray(idm, dtype=np.float64),
        len(returns_wide),
    )

    for sym in returns_wide.columns:
        if sym not in forecasts or sym not in instrument_weights:
            continue

        sigma_ann = compute_instrument_sigma_ann(
            returns_wide[sym].to_numpy(),
            lookback=vol_lookback,
            ann_factor=ann_factor,
        )

        fc = np.asarray(forecasts[sym], dtype=np.float64)
        iw = instrument_weights[sym]

        raw_pos = (vol_target * idm_arr * iw * fc) / (TARGET_ABS_FORECAST * sigma_ann)
        raw_pos = np.where(np.isfinite(raw_pos), raw_pos, 0.0)

        if long_only:
            raw_pos = np.maximum(raw_pos, 0.0)

        positions[sym] = np.clip(raw_pos, -max_position, max_position)

    return positions
