"""
V2 cross-sectional factor model — expanded factor set + multi-speed.

New factors on top of V1's five:
  6. Short-term reversal  — 24h return (contrarian: oversold bounce)
  7. Momentum acceleration — 7d vs 30d momentum differential
  8. BTC beta             — rolling correlation to BTC (low = diversifier)
  9. Volume profile shift — recent vs historical volume ratio
 10. Drawdown recovery    — distance from 30d low (bottom-fishing)

Multi-speed design: factors are computed at different lookback windows
and combined with configurable weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factors_v2(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    high_wide: pd.DataFrame,
    low_wide: pd.DataFrame,
    *,
    lookback_fast: int = 24,
    lookback_medium: int = 168,
    lookback_slow: int = 720,
) -> dict[str, pd.DataFrame]:
    """Compute 10 cross-sectional factors at multiple speeds."""
    ret = close_wide.pct_change()

    # ── V1 factors (tuned) ───────────────────────────────────────
    momentum_7d = np.log(close_wide / close_wide.shift(lookback_medium).clip(lower=1e-10))

    vol_24h = volume_wide.rolling(lookback_fast, min_periods=12).sum()
    vol_7d_avg = volume_wide.rolling(lookback_medium, min_periods=lookback_medium // 2).mean() * 24
    volume_surge = (vol_24h / vol_7d_avg.clip(lower=1)).clip(0, 5)

    realized_vol = ret.rolling(lookback_medium, min_periods=lookback_medium // 2).std() * np.sqrt(8760)

    rolling_high = high_wide.rolling(lookback_medium, min_periods=24).max()
    proximity_to_high = 1 + (close_wide - rolling_high) / rolling_high.clip(lower=1e-8)

    roll_mean = ret.rolling(lookback_medium, min_periods=lookback_medium // 2).mean()
    roll_std = ret.rolling(lookback_medium, min_periods=lookback_medium // 2).std()
    rolling_sharpe = roll_mean / roll_std.clip(lower=1e-8)

    # ── NEW: Short-term reversal (contrarian — 24h) ──────────────
    reversal_24h = -ret.rolling(lookback_fast, min_periods=12).sum()

    # ── NEW: Momentum acceleration ───────────────────────────────
    mom_30d = np.log(close_wide / close_wide.shift(lookback_slow).clip(lower=1e-10))
    mom_accel = momentum_7d - mom_30d * (lookback_medium / lookback_slow)

    # ── NEW: BTC beta (rolling correlation) ──────────────────────
    btc_ret = ret.get("BTC-USD")
    if btc_ret is not None:
        btc_corr = ret.rolling(lookback_medium, min_periods=lookback_medium // 2).corr(btc_ret)
        btc_beta_inv = 1 - btc_corr.clip(-1, 1).abs()
    else:
        btc_beta_inv = pd.DataFrame(0.5, index=close_wide.index, columns=close_wide.columns)

    # ── NEW: Volume profile shift (attention detector) ───────────
    vol_recent = volume_wide.rolling(lookback_fast * 3, min_periods=24).mean()
    vol_hist = volume_wide.rolling(lookback_slow, min_periods=lookback_slow // 2).mean()
    vol_profile = (vol_recent / vol_hist.clip(lower=1)).clip(0, 5)

    # ── NEW: Drawdown recovery ───────────────────────────────────
    rolling_low = low_wide.rolling(lookback_slow, min_periods=48).min()
    recovery = (close_wide - rolling_low) / rolling_low.clip(lower=1e-8)

    return {
        "momentum": momentum_7d,
        "volume_surge": volume_surge,
        "realized_vol": realized_vol,
        "proximity_to_high": proximity_to_high,
        "rolling_sharpe": rolling_sharpe,
        "reversal_24h": reversal_24h,
        "mom_acceleration": mom_accel,
        "btc_beta_inv": btc_beta_inv,
        "vol_profile_shift": vol_profile,
        "drawdown_recovery": recovery,
    }


# ── Composite score presets ──────────────────────────────────────

WEIGHT_PRESETS = {
    "v1_original": {
        "momentum": 0.30,
        "volume_surge": 0.15,
        "realized_vol": 0.15,
        "proximity_to_high": 0.15,
        "rolling_sharpe": 0.25,
    },
    "v2_momentum_heavy": {
        "momentum": 0.25,
        "rolling_sharpe": 0.20,
        "mom_acceleration": 0.15,
        "proximity_to_high": 0.10,
        "volume_surge": 0.10,
        "realized_vol": 0.05,
        "btc_beta_inv": 0.05,
        "vol_profile_shift": 0.05,
        "drawdown_recovery": 0.05,
    },
    "v2_diversified": {
        "momentum": 0.15,
        "rolling_sharpe": 0.15,
        "mom_acceleration": 0.10,
        "proximity_to_high": 0.10,
        "volume_surge": 0.10,
        "realized_vol": 0.05,
        "reversal_24h": 0.10,
        "btc_beta_inv": 0.10,
        "vol_profile_shift": 0.05,
        "drawdown_recovery": 0.10,
    },
    "v2_contrarian_blend": {
        "momentum": 0.15,
        "rolling_sharpe": 0.15,
        "reversal_24h": 0.20,
        "drawdown_recovery": 0.15,
        "btc_beta_inv": 0.10,
        "volume_surge": 0.10,
        "vol_profile_shift": 0.10,
        "proximity_to_high": 0.05,
    },
}


def compute_composite_v2(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
    preset: str = "v2_momentum_heavy",
) -> pd.DataFrame:
    """Cross-sectional rank each factor, then take weighted average.

    Returns wide-format DataFrame with composite scores in [0, 1].
    """
    if weights is None:
        weights = WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS["v2_momentum_heavy"])

    ref = next(iter(factors.values()))
    composite = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
    total_w = 0.0

    for name, w in weights.items():
        if name in factors and w > 0:
            ranked = factors[name].rank(axis=1, pct=True)
            composite += w * ranked.fillna(0.5)
            total_w += w

    if total_w > 0:
        composite /= total_w

    return composite
