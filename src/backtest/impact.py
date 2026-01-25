"""Dynamic slippage / market impact model utilities."""
from __future__ import annotations

from typing import Iterable

import polars as pl
import math


def _rolling_volatility(bars: pl.DataFrame, window: int) -> list[float]:
    returns = bars.select(pl.col("close").pct_change().fill_null(0.0)).to_series()
    vol = returns.rolling_std(window_size=window, min_samples=window).fill_null(0.0)
    return vol.to_list()


def _volume_usd(bars: pl.DataFrame) -> list[float]:
    vol_usd = (bars["volume"] * bars["close"] + 1.0).to_list()
    return vol_usd


def compute_dynamic_slippage(
    bars: pl.DataFrame,
    trade_size_usd: Iterable[float],
    *,
    impact_coeff: float = 0.1,
    vol_window: int = 24,
) -> list[float]:
    """
    Compute impact in bps per bar.

    Impact(bps) = C * volatility * sqrt(trade_size_usd / volume_usd)
    - volatility is rolling std of pct returns
    - volume_usd = volume * close (+1.0 safety)
    - impact_bps clipped to [2.0, 500.0] when trade_size_usd > 0
    """
    vol = _rolling_volatility(bars, vol_window)
    vol_usd = _volume_usd(bars)
    impact_bps = []
    for v, vol_u, trade_usd in zip(vol, vol_usd, trade_size_usd):
        if trade_usd <= 0:
            impact_bps.append(0.0)
            continue
        raw = impact_coeff * v * math.sqrt(trade_usd / vol_u)
        clipped = min(500.0, max(2.0, raw))
        impact_bps.append(clipped)
    return impact_bps


def compute_impact_bps(
    *,
    volatility: float,
    trade_size_usd: float,
    volume_usd: float,
    impact_coeff: float = 0.1,
) -> float:
    """Compute impact bps for a single bar."""
    if trade_size_usd <= 0:
        return 0.0
    raw = impact_coeff * volatility * math.sqrt(trade_size_usd / volume_usd)
    return min(500.0, max(2.0, raw))
