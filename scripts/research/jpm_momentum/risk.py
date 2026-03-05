"""
Risk management overlays for JPM Momentum research.

Implements the paper's risk management toolkit (Chapter 3):

- **Volatility Targeting**: Scale portfolio to target annualised vol.
- **Trailing Stop-Loss**: Exit when drawdown from peak exceeds threshold.
- **Time-Based Re-Entry**: After stop triggers, wait N bars before re-entering.
- **Mean Reversion Overlay**: Reduce position when short-term return exceeds
  threshold (dampens turning-point losses).
- **Volatility Signal Filter**: Reduce exposure during high-vol regimes.

All functions operate on wide-format weight DataFrames (index=ts, columns=symbols)
and return modified weight DataFrames.

Supports both crypto (ann_factor=365) and ETF (ann_factor=252) markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ANN_FACTOR_CRYPTO


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------
def apply_vol_targeting(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_target: float = 0.20,
    lookback: int = 42,
    max_leverage: float = 2.0,
    ann_factor: float = ANN_FACTOR_CRYPTO,
) -> pd.DataFrame:
    """Scale portfolio weights so realised vol ~ ``vol_target``.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weight matrix.
    returns_wide : pd.DataFrame
        Wide-format return matrix (same column universe).
    vol_target : float
        Target annualised volatility.
    lookback : int
        Rolling window for realised portfolio vol estimate.
    max_leverage : float
        Cap on the scaling factor.
    ann_factor : float
        Annualisation factor (365 for crypto, 252 for ETFs).
    """
    common = weights.columns.intersection(returns_wide.columns)
    w = weights[common].copy()
    r = returns_wide[common].reindex(w.index).fillna(0.0)

    w_held = w.shift(1).fillna(0.0)
    port_ret = (w_held * r).sum(axis=1)
    realised_vol = (
        port_ret.rolling(lookback, min_periods=max(10, lookback // 2)).std()
        * np.sqrt(ann_factor)
    )
    scalar = (vol_target / realised_vol).clip(lower=0.0, upper=max_leverage).fillna(1.0)
    return w.mul(scalar, axis=0)


# ---------------------------------------------------------------------------
# Trailing stop-loss with optional time-based re-entry
# ---------------------------------------------------------------------------
def apply_trailing_stop(
    base_weights: pd.DataFrame,
    close_wide: pd.DataFrame,
    stop_pct: float,
    reentry_bars: int = 0,
) -> pd.DataFrame:
    """Zero out asset weight when price drops >stop_pct from peak while held.

    Parameters
    ----------
    base_weights : pd.DataFrame
        Wide-format weight matrix (pre-overlay).
    close_wide : pd.DataFrame
        Wide-format close price matrix.
    stop_pct : float
        Stop-loss trigger as a fraction (e.g. 0.15 for 15%).
    reentry_bars : int
        Minimum bars to wait before re-entering after a stop.
        0 = re-enter at next rebalance that selects the asset.
    """
    w = base_weights.copy()
    common = w.columns.intersection(close_wide.columns)
    w = w[common]
    cls = close_wide[common].reindex(w.index).ffill()

    stopped: dict[str, bool] = {c: False for c in common}
    stopped_at: dict[str, int] = {c: -9999 for c in common}
    peak: dict[str, float] = {c: 0.0 for c in common}
    sym_to_col = {s: j for j, s in enumerate(w.columns)}

    for i, dt in enumerate(w.index):
        for sym in common:
            col_idx = sym_to_col[sym]
            base_wt = base_weights.at[dt, sym] if sym in base_weights.columns else 0.0
            price = cls.iloc[i, col_idx] if not np.isnan(cls.iloc[i, col_idx]) else 0.0

            if base_wt > 0 and not stopped[sym]:
                peak[sym] = max(peak[sym], price) if peak[sym] > 0 else price
                if peak[sym] > 0 and price < peak[sym] * (1 - stop_pct):
                    stopped[sym] = True
                    stopped_at[sym] = i
                    w.iloc[i, col_idx] = 0.0
            elif base_wt > 0 and stopped[sym]:
                bars_since_stop = i - stopped_at[sym]
                if bars_since_stop >= reentry_bars:
                    stopped[sym] = False
                    peak[sym] = price
                else:
                    w.iloc[i, col_idx] = 0.0
            else:
                stopped[sym] = False
                peak[sym] = 0.0

    # Re-normalise so weights sum to original exposure on each day
    orig_sum = base_weights[common].sum(axis=1).replace(0, np.nan)
    new_sum = w.sum(axis=1).replace(0, np.nan)
    scale = (orig_sum / new_sum).fillna(0.0).clip(upper=2.0)
    w = w.multiply(scale, axis=0)
    return w


# ---------------------------------------------------------------------------
# Mean reversion overlay
# ---------------------------------------------------------------------------
def apply_mean_reversion_overlay(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    window: int = 5,
    threshold_sigma: float = 2.0,
    scale_factor: float = 0.5,
) -> pd.DataFrame:
    """Reduce position when short-term return exceeds threshold.

    When a symbol's ``window``-day return exceeds ``threshold_sigma`` standard
    deviations (estimated over a longer lookback), its weight is scaled to
    ``scale_factor`` of its base weight.  This dampens turning-point losses.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weight matrix.
    returns_wide : pd.DataFrame
        Wide-format daily return matrix.
    window : int
        Short-term return window (e.g. 5 days).
    threshold_sigma : float
        Number of standard deviations to trigger scaling.
    scale_factor : float
        Multiplier applied when threshold is breached (e.g. 0.5 = halve weight).
    """
    common = weights.columns.intersection(returns_wide.columns)
    w = weights[common].copy()
    r = returns_wide[common].reindex(w.index).fillna(0.0)

    cum_ret = (1 + r).rolling(window).apply(np.prod, raw=True) - 1.0
    long_vol = r.rolling(63, min_periods=21).std()
    z_score = cum_ret / (long_vol * np.sqrt(window)).replace(0, np.nan)

    breach = z_score.abs() > threshold_sigma
    w = w.where(~breach, w * scale_factor)
    return w


# ---------------------------------------------------------------------------
# Volatility signal filter
# ---------------------------------------------------------------------------
def apply_vol_filter(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    multiplier: float = 2.0,
    scale: float = 0.5,
    vol_lookback: int = 42,
    median_lookback: int = 252,
    ann_factor: float = ANN_FACTOR_CRYPTO,
) -> pd.DataFrame:
    """Reduce exposure during high-vol regimes.

    When realised portfolio vol exceeds ``multiplier`` times its rolling median,
    scale all weights to ``scale`` fraction of normal.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weight matrix.
    returns_wide : pd.DataFrame
        Wide-format daily return matrix.
    multiplier : float
        Trigger multiplier vs. median vol (e.g. 2.0 = reduce when vol > 2x median).
    scale : float
        Exposure fraction in high-vol regime.
    vol_lookback : int
        Rolling window for realised vol estimation.
    median_lookback : int
        Rolling window for median vol estimation.
    ann_factor : float
        Annualisation factor (365 for crypto, 252 for ETFs).
    """
    common = weights.columns.intersection(returns_wide.columns)
    w = weights[common].copy()
    r = returns_wide[common].reindex(w.index).fillna(0.0)

    w_held = w.shift(1).fillna(0.0)
    port_ret = (w_held * r).sum(axis=1)
    realised_vol = port_ret.rolling(vol_lookback, min_periods=max(10, vol_lookback // 2)).std() * np.sqrt(ann_factor)
    median_vol = realised_vol.rolling(median_lookback, min_periods=max(42, median_lookback // 2)).median()

    high_vol = realised_vol > (multiplier * median_vol)
    scalar = high_vol.map({True: scale, False: 1.0}).fillna(1.0)
    return w.mul(scalar, axis=0)
