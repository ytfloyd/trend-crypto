"""
V3 cross-sectional factor model — adaptive factors designed to push
useful signal deeper into the rank order.

Problems with V2 at lower ranks:
  - Momentum and Sharpe are dominated by large-cap names (BTC, ETH, SOL)
  - Small/mid-cap tokens have sparse, noisy return histories
  - Volume factors reward large tokens mechanically

V3 adds:
 11. Sector-relative momentum   — momentum vs same-cap-tier peers, not absolute
 12. Liquidity improvement       — recent ADV trend (attention before the move)
 13. Idiosyncratic momentum      — returns orthogonal to BTC/market
 14. Relative strength breadth   — what % of lookback windows is the token positive
 15. Volatility contraction      — vol compression as breakout precursor

Design principle: rank within liquidity tiers to avoid large-cap bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _tier_relative_rank(
    factor: pd.DataFrame,
    adv_wide: pd.DataFrame | None = None,
    n_tiers: int = 3,
) -> pd.DataFrame:
    """Rank a factor within liquidity tiers, then normalize to [0,1].

    This prevents large-cap tokens from mechanically dominating
    cross-sectional ranks due to scale effects.
    """
    if adv_wide is None:
        return factor.rank(axis=1, pct=True)

    median_adv = adv_wide.median()
    sorted_adv = median_adv.sort_values()
    tier_size = max(len(sorted_adv) // n_tiers, 1)

    tier_map = {}
    for i, sym in enumerate(sorted_adv.index):
        tier_map[sym] = min(i // tier_size, n_tiers - 1)

    result = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)

    for tier in range(n_tiers):
        tier_syms = [s for s, t in tier_map.items() if t == tier and s in factor.columns]
        if len(tier_syms) < 3:
            continue
        tier_ranks = factor[tier_syms].rank(axis=1, pct=True)
        result[tier_syms] = tier_ranks

    return result.fillna(0.5)


def compute_factors_v3(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    high_wide: pd.DataFrame,
    low_wide: pd.DataFrame,
    adv_wide: pd.DataFrame | None = None,
    *,
    lookback_fast: int = 24,
    lookback_medium: int = 168,
    lookback_slow: int = 720,
) -> dict[str, pd.DataFrame]:
    """Compute 15 cross-sectional factors with tier-relative ranking."""
    ret = close_wide.pct_change(fill_method=None)

    # ── V1 factors ─────────────────────────────────────────────
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

    # ── V2 factors ─────────────────────────────────────────────
    reversal_24h = -ret.rolling(lookback_fast, min_periods=12).sum()

    mom_30d = np.log(close_wide / close_wide.shift(lookback_slow).clip(lower=1e-10))
    mom_accel = momentum_7d - mom_30d * (lookback_medium / lookback_slow)

    btc_ret = ret.get("BTC-USD")
    if btc_ret is not None:
        btc_corr = ret.rolling(lookback_medium, min_periods=lookback_medium // 2).corr(btc_ret)
        btc_beta_inv = 1 - btc_corr.clip(-1, 1).abs()
    else:
        btc_beta_inv = pd.DataFrame(0.5, index=close_wide.index, columns=close_wide.columns)

    vol_recent = volume_wide.rolling(lookback_fast * 3, min_periods=24).mean()
    vol_hist = volume_wide.rolling(lookback_slow, min_periods=lookback_slow // 2).mean()
    vol_profile = (vol_recent / vol_hist.clip(lower=1)).clip(0, 5)

    rolling_low = low_wide.rolling(lookback_slow, min_periods=48).min()
    recovery = (close_wide - rolling_low) / rolling_low.clip(lower=1e-8)

    # ── V3 NEW factors ─────────────────────────────────────────

    # 11. Idiosyncratic momentum: residual returns after removing BTC exposure
    if btc_ret is not None:
        # Vectorised rolling beta via cov/var decomposition
        btc_arr = btc_ret.reindex(ret.index).fillna(0)
        min_p = lookback_medium // 2
        xy = ret.mul(btc_arr, axis=0).rolling(lookback_medium, min_periods=min_p).mean()
        x_bar = ret.rolling(lookback_medium, min_periods=min_p).mean()
        y_bar = btc_arr.rolling(lookback_medium, min_periods=min_p).mean()
        cov_xy = xy - x_bar.mul(y_bar, axis=0)
        var_y = btc_arr.rolling(lookback_medium, min_periods=min_p).var()
        beta = cov_xy.div(var_y.clip(lower=1e-10), axis=0)
        resid_ret = ret.sub(beta.mul(btc_arr, axis=0), axis=0)
        idio_mom = resid_ret.rolling(lookback_medium, min_periods=min_p).sum()
    else:
        idio_mom = momentum_7d.copy()

    # 12. Liquidity improvement: trend in dollar volume (rolling slope)
    dv = close_wide * volume_wide
    dv_short = dv.rolling(lookback_fast * 3, min_periods=24).mean()
    dv_long = dv.rolling(lookback_slow, min_periods=lookback_slow // 2).mean()
    liq_improvement = (dv_short / dv_long.clip(lower=1)).clip(0, 5) - 1

    # 13. Relative strength breadth: fraction of recent days with positive returns
    daily_ret = ret.resample("D").sum()
    breadth_daily = daily_ret.rolling(30, min_periods=10).apply(
        lambda x: (x > 0).sum() / len(x), raw=True
    )
    breadth = breadth_daily.reindex(close_wide.index, method="ffill")

    # 14. Volatility contraction: vol compression as breakout precursor
    vol_short = ret.rolling(lookback_fast * 2, min_periods=12).std()
    vol_long = ret.rolling(lookback_medium, min_periods=lookback_medium // 2).std()
    vol_contraction = 1 - (vol_short / vol_long.clip(lower=1e-8)).clip(0, 3)

    # 15. Multi-timeframe agreement: alignment of 1d, 3d, 7d momentum signs
    mom_1d = ret.rolling(lookback_fast, min_periods=12).sum()
    mom_3d = ret.rolling(lookback_fast * 3, min_periods=24).sum()
    mtf_agree = (
        (mom_1d > 0).astype(float) +
        (mom_3d > 0).astype(float) +
        (momentum_7d > 0).astype(float)
    ) / 3.0

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
        "idio_momentum": idio_mom,
        "liq_improvement": liq_improvement,
        "breadth": breadth,
        "vol_contraction": vol_contraction,
        "mtf_agreement": mtf_agree,
    }


WEIGHT_PRESETS_V3: dict[str, dict[str, float]] = {
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
    "v3_deep_signal": {
        "momentum": 0.15,
        "rolling_sharpe": 0.10,
        "idio_momentum": 0.15,
        "breadth": 0.12,
        "mtf_agreement": 0.12,
        "vol_contraction": 0.10,
        "liq_improvement": 0.08,
        "mom_acceleration": 0.08,
        "btc_beta_inv": 0.05,
        "vol_profile_shift": 0.05,
    },
    "v3_balanced": {
        "momentum": 0.12,
        "rolling_sharpe": 0.10,
        "idio_momentum": 0.12,
        "breadth": 0.10,
        "mtf_agreement": 0.10,
        "vol_contraction": 0.08,
        "liq_improvement": 0.08,
        "mom_acceleration": 0.08,
        "proximity_to_high": 0.07,
        "volume_surge": 0.05,
        "btc_beta_inv": 0.05,
        "drawdown_recovery": 0.05,
    },
    "v3_contrarian_deep": {
        "reversal_24h": 0.15,
        "vol_contraction": 0.15,
        "breadth": 0.12,
        "idio_momentum": 0.12,
        "liq_improvement": 0.10,
        "btc_beta_inv": 0.10,
        "drawdown_recovery": 0.08,
        "mtf_agreement": 0.08,
        "rolling_sharpe": 0.05,
        "momentum": 0.05,
    },
}


def compute_composite_v3(
    factors: dict[str, pd.DataFrame],
    adv_wide: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    preset: str = "v3_deep_signal",
    use_tier_relative: bool = True,
) -> pd.DataFrame:
    """Cross-sectional rank (optionally tier-relative), weighted combination."""
    if weights is None:
        weights = WEIGHT_PRESETS_V3.get(preset, WEIGHT_PRESETS_V3["v3_deep_signal"])

    ref = next(iter(factors.values()))
    composite = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
    total_w = 0.0

    for name, w in weights.items():
        if name in factors and w > 0:
            if use_tier_relative and adv_wide is not None:
                ranked = _tier_relative_rank(factors[name], adv_wide)
            else:
                ranked = factors[name].rank(axis=1, pct=True)
            composite += w * ranked.fillna(0.5)
            total_w += w

    if total_w > 0:
        composite /= total_w

    return composite
