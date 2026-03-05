"""Combine multiple trading rule forecasts into a single blended forecast.

Implements Carver's forecast combination with the Forecast Diversification
Multiplier (FDM).  The FDM compensates for the dampening that occurs when
averaging imperfectly correlated forecasts, ensuring the combined forecast
retains the correct average absolute value.

Reference: Robert Carver, *Systematic Trading*, Chapters 8-9.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .forecast import FORECAST_CAP, TARGET_ABS_FORECAST


@dataclass
class ForecastCombiner:
    """Weighted combination of rule forecasts with diversification multiplier.

    Parameters
    ----------
    weights : dict mapping rule name to its combination weight.
        Weights must be non-negative and will be normalised to sum to 1.
    fdm : Forecast Diversification Multiplier. If None, it is estimated
        from the forecast correlation matrix after calling ``fit()``.
    cap : maximum absolute combined forecast.
    """

    weights: dict[str, float]
    fdm: float | None = None
    cap: float = FORECAST_CAP
    _fitted_fdm: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Forecast weights must sum to a positive number")
        self._norm_weights = {k: v / total for k, v in self.weights.items()}
        if self.fdm is not None:
            self._fitted_fdm = self.fdm

    def fit(self, forecasts: dict[str, np.ndarray]) -> "ForecastCombiner":
        """Estimate the FDM from forecast data (if not set manually).

        Parameters
        ----------
        forecasts : dict mapping rule name to 1-D forecast array.
            All arrays must have the same length.
        """
        if self.fdm is not None:
            return self

        self._fitted_fdm = compute_fdm(
            forecasts=forecasts,
            weights=self._norm_weights,
        )
        return self

    def combine(self, forecasts: dict[str, np.ndarray]) -> np.ndarray:
        """Produce the combined, FDM-scaled, capped forecast.

        Parameters
        ----------
        forecasts : dict mapping rule name to 1-D forecast array.

        Returns
        -------
        1-D array of combined forecasts.
        """
        n = None
        for v in forecasts.values():
            n = len(v)
            break
        if n is None:
            raise ValueError("No forecasts provided")

        combined = np.zeros(n, dtype=np.float64)
        for name, w in self._norm_weights.items():
            if name in forecasts:
                fc = np.asarray(forecasts[name], dtype=np.float64)
                combined += w * np.where(np.isfinite(fc), fc, 0.0)

        combined *= self._fitted_fdm
        return np.clip(combined, -self.cap, self.cap)

    def combine_df(self, forecasts_df: pd.DataFrame) -> pd.Series:
        """Combine forecasts from a DataFrame (columns = rule names).

        Parameters
        ----------
        forecasts_df : DataFrame where each column is a rule's forecast.

        Returns
        -------
        pd.Series of combined forecasts with the same index.
        """
        fc_dict = {col: forecasts_df[col].to_numpy() for col in forecasts_df.columns}
        combined = self.combine(fc_dict)
        return pd.Series(combined, index=forecasts_df.index, name="combined_forecast")


def compute_fdm(
    forecasts: dict[str, np.ndarray],
    weights: dict[str, float],
) -> float:
    """Compute the Forecast Diversification Multiplier.

    FDM = 1 / sqrt(w' @ C @ w)

    where w is the weight vector and C is the correlation matrix of the
    forecasts.  Capped at a maximum of 2.5 to avoid unrealistic scaling.

    Parameters
    ----------
    forecasts : dict mapping rule name to 1-D forecast array.
    weights : dict mapping rule name to normalised weight.
    """
    names = sorted(set(forecasts.keys()) & set(weights.keys()))
    if len(names) < 2:
        return 1.0

    df = pd.DataFrame({n: forecasts[n] for n in names}).dropna()
    if len(df) < 20:
        return 1.0

    corr = df.corr().values
    w = np.array([weights[n] for n in names])
    w = w / w.sum()

    portfolio_var = w @ corr @ w
    if portfolio_var <= 0:
        return 1.0

    fdm = 1.0 / math.sqrt(portfolio_var)
    return min(fdm, 2.5)


def equal_weight_combiner(
    rule_names: list[str],
    fdm: float | None = None,
    cap: float = FORECAST_CAP,
) -> ForecastCombiner:
    """Convenience: create a combiner with equal weights across rules."""
    n = len(rule_names)
    if n == 0:
        raise ValueError("Need at least one rule")
    weights = {name: 1.0 / n for name in rule_names}
    return ForecastCombiner(weights=weights, fdm=fdm, cap=cap)
