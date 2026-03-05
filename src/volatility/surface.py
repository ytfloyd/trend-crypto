"""Implied volatility surface representation and interpolation.

A vol surface is a 2D function: sigma(K, T) where K is strike (or
moneyness) and T is time to expiry.  This module provides:

    VolSlice  — a single-expiry smile (sigma vs strike)
    VolSurface — a collection of slices forming the full surface

The surface supports interpolation across both strike and expiry
dimensions, which is essential for pricing off-market strikes and
for computing vol metrics like skew, convexity, and term structure.

Reference: Natenberg, *Option Volatility and Pricing*, Ch. 18-19.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate


@dataclass
class VolSlice:
    """Single-expiry volatility smile.

    Attributes
    ----------
    expiry : expiration datetime (UTC).
    strikes : array of strikes.
    ivs : array of implied vols corresponding to each strike.
    forward : forward price for this expiry (needed for moneyness).
    underlying_price : spot price at snapshot time.
    snapshot_ts : when this slice was observed.
    """

    expiry: datetime
    strikes: np.ndarray
    ivs: np.ndarray
    forward: float
    underlying_price: float
    snapshot_ts: datetime
    bid_ivs: Optional[np.ndarray] = None
    ask_ivs: Optional[np.ndarray] = None

    @property
    def tte(self) -> float:
        """Time to expiry in years."""
        dt = (self.expiry - self.snapshot_ts).total_seconds()
        return max(dt / (365.25 * 86400), 1e-8)

    @property
    def moneyness(self) -> np.ndarray:
        """Log-moneyness: ln(K / F)."""
        return np.log(self.strikes / self.forward)

    @property
    def delta_strikes(self) -> np.ndarray:
        """Approximate BS delta for each strike (call delta)."""
        from pricing.black_scholes import bs_delta
        return np.array([
            bs_delta(self.forward, k, self.tte, iv, 0.0, is_call=True)
            for k, iv in zip(self.strikes, self.ivs)
        ])

    def iv_at_strike(self, strike: float) -> float:
        """Interpolate IV at an arbitrary strike using cubic spline."""
        if len(self.strikes) < 2:
            return float(self.ivs[0]) if len(self.ivs) > 0 else np.nan
        f = interpolate.interp1d(
            self.strikes, self.ivs,
            kind="cubic" if len(self.strikes) >= 4 else "linear",
            fill_value="extrapolate",
        )
        return float(f(strike))

    def atm_iv(self) -> float:
        """ATM implied vol (interpolated at forward price)."""
        return self.iv_at_strike(self.forward)

    def skew_25d(self) -> float:
        """25-delta risk reversal: IV(25d put) - IV(25d call).

        Positive skew = puts are more expensive (typical for equities).
        """
        deltas = self.delta_strikes
        if len(deltas) < 4:
            return np.nan
        f_put = interpolate.interp1d(
            deltas, self.ivs, kind="linear", fill_value="extrapolate",
        )
        return float(f_put(0.25) - f_put(0.75))

    def butterfly_25d(self) -> float:
        """25-delta butterfly: 0.5*(IV(25d put) + IV(25d call)) - ATM IV.

        Measures smile convexity / wing richness.
        """
        deltas = self.delta_strikes
        if len(deltas) < 4:
            return np.nan
        f = interpolate.interp1d(
            deltas, self.ivs, kind="linear", fill_value="extrapolate",
        )
        wing_avg = 0.5 * (float(f(0.25)) + float(f(0.75)))
        return wing_avg - self.atm_iv()

    def to_dataframe(self) -> pd.DataFrame:
        """Export slice as a DataFrame for analysis."""
        data = {
            "strike": self.strikes,
            "iv": self.ivs,
            "moneyness": self.moneyness,
        }
        if self.bid_ivs is not None:
            data["bid_iv"] = self.bid_ivs
        if self.ask_ivs is not None:
            data["ask_iv"] = self.ask_ivs
        df = pd.DataFrame(data)
        df["expiry"] = self.expiry
        df["tte"] = self.tte
        df["forward"] = self.forward
        return df


@dataclass
class VolSurface:
    """Collection of VolSlices forming a complete volatility surface.

    Supports interpolation across both strike and expiry dimensions
    and provides summary metrics for vol trading.
    """

    underlying: str
    slices: list[VolSlice] = field(default_factory=list)
    snapshot_ts: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.slices.sort(key=lambda s: s.expiry)
        if self.slices and self.snapshot_ts is None:
            self.snapshot_ts = self.slices[0].snapshot_ts

    def add_slice(self, s: VolSlice) -> None:
        self.slices.append(s)
        self.slices.sort(key=lambda s: s.expiry)

    @property
    def expiries(self) -> list[datetime]:
        return [s.expiry for s in self.slices]

    @property
    def ttes(self) -> np.ndarray:
        return np.array([s.tte for s in self.slices])

    def slice_by_expiry(self, expiry: datetime) -> Optional[VolSlice]:
        """Find the slice matching a specific expiry."""
        for s in self.slices:
            if s.expiry == expiry:
                return s
        return None

    def nearest_slice(self, tte: float) -> VolSlice:
        """Find the slice with the nearest time-to-expiry."""
        idx = int(np.argmin(np.abs(self.ttes - tte)))
        return self.slices[idx]

    def iv(self, strike: float, tte: float) -> float:
        """Interpolate IV at an arbitrary (strike, tte) point.

        Uses linear interpolation in the expiry dimension across
        the two bracketing slices, and cubic spline within each slice.
        """
        if not self.slices:
            return np.nan

        ttes = self.ttes
        if tte <= ttes[0]:
            return self.slices[0].iv_at_strike(strike)
        if tte >= ttes[-1]:
            return self.slices[-1].iv_at_strike(strike)

        idx = int(np.searchsorted(ttes, tte)) - 1
        idx = max(0, min(idx, len(self.slices) - 2))
        s0 = self.slices[idx]
        s1 = self.slices[idx + 1]

        iv0 = s0.iv_at_strike(strike)
        iv1 = s1.iv_at_strike(strike)
        w = (tte - ttes[idx]) / (ttes[idx + 1] - ttes[idx])

        # Interpolate in variance space (more stable than vol space)
        var0 = iv0 ** 2 * ttes[idx]
        var1 = iv1 ** 2 * ttes[idx + 1]
        total_var = var0 * (1 - w) + var1 * w
        return np.sqrt(max(total_var / tte, 0.0))

    def atm_term_structure(self) -> pd.DataFrame:
        """ATM vol across all expiries."""
        rows = []
        for s in self.slices:
            rows.append({
                "expiry": s.expiry,
                "tte": s.tte,
                "atm_iv": s.atm_iv(),
                "forward": s.forward,
            })
        return pd.DataFrame(rows)

    def skew_term_structure(self) -> pd.DataFrame:
        """25-delta skew across all expiries."""
        rows = []
        for s in self.slices:
            rows.append({
                "expiry": s.expiry,
                "tte": s.tte,
                "skew_25d": s.skew_25d(),
                "butterfly_25d": s.butterfly_25d(),
            })
        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the entire surface into a single DataFrame."""
        frames = [s.to_dataframe() for s in self.slices]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
