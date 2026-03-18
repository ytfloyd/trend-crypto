"""
Cross-sectional factor model for token selection and sizing.

Ranks every token in the universe at each rebalance on five
orthogonal factors, then combines into a composite percentile
score that drives continuous allocation.

Factors
-------
1. Momentum        — 7d log return (trend strength)
2. Volume surge    — 24h volume / 7d avg volume (attention/flow)
3. Realized vol    — 7d annualised vol (opportunity for trend capture)
4. Proximity-to-high — closeness to 7d high (trend persistence)
5. Risk-adj momentum — 7d return / 7d vol (Sharpe-like quality filter)

All factors are cross-sectionally ranked to [0, 1] each hour
to eliminate scale differences across tokens and time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factors(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    high_wide: pd.DataFrame,
    *,
    lookback_momentum: int = 168,
    lookback_volume: int = 168,
    lookback_vol: int = 168,
    lookback_sharpe: int = 168,
) -> dict[str, pd.DataFrame]:
    """Compute raw factor values (wide-format, ts × symbol)."""
    ret = close_wide.pct_change()

    momentum = np.log(close_wide / close_wide.shift(lookback_momentum))

    vol_24h = volume_wide.rolling(24, min_periods=12).sum()
    vol_7d_avg = (
        volume_wide.rolling(lookback_volume, min_periods=lookback_volume // 2).mean()
        * 24
    )
    volume_surge = (vol_24h / vol_7d_avg.clip(lower=1)).clip(0, 5)

    realized_vol = (
        ret.rolling(lookback_vol, min_periods=lookback_vol // 2).std()
        * np.sqrt(8760)
    )

    rolling_high = high_wide.rolling(lookback_momentum, min_periods=24).max()
    proximity_to_high = 1 + (close_wide - rolling_high) / rolling_high.clip(lower=1e-8)

    roll_mean = ret.rolling(lookback_sharpe, min_periods=lookback_sharpe // 2).mean()
    roll_std = ret.rolling(lookback_sharpe, min_periods=lookback_sharpe // 2).std()
    rolling_sharpe = roll_mean / roll_std.clip(lower=1e-8)

    return {
        "momentum": momentum,
        "volume_surge": volume_surge,
        "realized_vol": realized_vol,
        "proximity_to_high": proximity_to_high,
        "rolling_sharpe": rolling_sharpe,
    }


def compute_composite_score(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Cross-sectional rank each factor, then take weighted average.

    Returns wide-format DataFrame with composite scores ∈ [0, 1].
    """
    if weights is None:
        weights = {
            "momentum": 0.30,
            "volume_surge": 0.15,
            "realized_vol": 0.15,
            "proximity_to_high": 0.15,
            "rolling_sharpe": 0.25,
        }

    ref = next(iter(factors.values()))
    composite = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)

    for name, w in weights.items():
        if name in factors:
            ranked = factors[name].rank(axis=1, pct=True)
            composite += w * ranked.fillna(0.5)

    return composite
