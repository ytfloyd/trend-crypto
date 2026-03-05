"""
TSMOM weight construction — long-or-cash, per-asset absolute weighting.

Default position is 100 % cash.  When a trend signal fires (signal > 0)
on a given asset, we go long with volatility-scaled sizing.  When the
signal is non-positive, weight = 0 (cash).

Sizing methods:
  binary       : weight_i = vol_target / realized_vol_i   (equal risk)
  proportional : weight_i = f(signal) * vol_target / realized_vol_i
  capped       : same as proportional, clipped to max_weight

Max-weight cap handling: excess weight is sent to cash (NOT redistributed)
to keep per-position risk contribution clean.

Exit overlays:
  signal_reversal : exit when signal <= 0  (embedded in weight construction)
  vol_adaptive    : exit when price drops K * ATR_entry from peak
  vol_spike       : exit when realized vol > 2x entry-date realized vol
  combined        : signal_reversal OR vol_adaptive, whichever first
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from common.data import ANN_FACTOR
from common.risk_overlays import apply_vol_targeting


# Re-export under the name used by the runner for clarity
apply_portfolio_vol_target = apply_vol_targeting


# -----------------------------------------------------------------------
# Per-asset weight construction
# -----------------------------------------------------------------------

def build_tsmom_weights(
    signal: pd.DataFrame,
    universe: pd.DataFrame,
    returns_wide: pd.DataFrame,
    *,
    sizing: str = "binary",
    vol_target: float = 0.15,
    vol_lookback: int = 63,
    max_weight: float = 0.20,
) -> pd.DataFrame:
    """Build long-or-cash weights from an absolute trend signal.

    Parameters
    ----------
    signal : wide-format signal (index=ts, columns=symbols).
        Positive => uptrend, non-positive => cash.
    universe : wide-format boolean mask (True = tradeable).
    returns_wide : wide-format daily returns for vol estimation.
    sizing : 'binary', 'proportional', or 'capped'.
    vol_target : per-asset annualized vol target for position sizing.
    vol_lookback : rolling window for realized vol estimation.
    max_weight : hard cap per-asset.  Excess goes to cash.

    Returns
    -------
    pd.DataFrame  wide-format weights (>= 0, cash when 0).
    """
    sig = signal.copy()
    sig[~universe] = np.nan

    realized_vol = (
        returns_wide
        .rolling(vol_lookback, min_periods=max(10, vol_lookback // 2))
        .std() * np.sqrt(ANN_FACTOR)
    )
    realized_vol = realized_vol.replace(0, np.nan)

    # Long-only: only positive signals
    entry_mask = sig > 0

    if sizing == "binary":
        raw_w = (vol_target / realized_vol).where(entry_mask, 0.0)
    elif sizing == "proportional":
        sig_positive = sig.clip(lower=0)
        sig_norm = sig_positive / sig_positive.max(axis=1).replace(0, np.nan).values[:, None]
        raw_w = (sig_norm * vol_target / realized_vol).where(entry_mask, 0.0)
    elif sizing == "capped":
        sig_positive = sig.clip(lower=0)
        sig_norm = sig_positive / sig_positive.max(axis=1).replace(0, np.nan).values[:, None]
        raw_w = (sig_norm * vol_target / realized_vol).where(entry_mask, 0.0)
    else:
        raise ValueError(f"Unknown sizing method {sizing!r}")

    raw_w = raw_w.fillna(0.0)

    # Hard cap — excess goes to cash (not redistributed)
    raw_w = raw_w.clip(upper=max_weight)

    return raw_w


# -----------------------------------------------------------------------
# Exit overlays
# -----------------------------------------------------------------------

def apply_vol_adaptive_trailing_stop(
    base_weights: pd.DataFrame,
    close_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    atr_multiple: float = 2.0,
    atr_lookback: int = 14,
) -> pd.DataFrame:
    """Exit when price drops K * ATR_entry from peak while held.

    ATR_entry is the asset's ATR at the time the position was opened.
    After being stopped out, the asset can re-enter at the next bar
    where the base weight becomes non-zero from a previously-zero state.

    Parameters
    ----------
    base_weights : wide-format weight matrix (pre-overlay).
    close_wide : wide-format close prices.
    returns_wide : wide-format daily returns (for ATR estimation).
    atr_multiple : K — number of ATR multiples for the stop.
    atr_lookback : window for ATR computation.
    """
    common = base_weights.columns.intersection(close_wide.columns)
    w = base_weights[common].copy()
    cls = close_wide[common].reindex(w.index).ffill()

    atr = (
        returns_wide[common].abs()
        .rolling(atr_lookback, min_periods=max(5, atr_lookback // 2))
        .mean()
        .reindex(w.index)
        .ffill()
    )
    # ATR in price terms
    atr_price = atr * cls

    # State tracking per asset
    peak: dict[str, float] = {c: 0.0 for c in common}
    entry_atr: dict[str, float] = {c: 0.0 for c in common}
    stopped: dict[str, bool] = {c: False for c in common}
    was_in: dict[str, bool] = {c: False for c in common}

    w_vals = w.values.copy()
    cls_vals = cls.values
    atr_vals = atr_price.values
    col_map = {s: j for j, s in enumerate(common)}

    for i in range(len(w.index)):
        for sym in common:
            j = col_map[sym]
            base_wt = w_vals[i, j]
            price = cls_vals[i, j] if not np.isnan(cls_vals[i, j]) else 0.0
            cur_atr = atr_vals[i, j] if not np.isnan(atr_vals[i, j]) else 0.0

            if base_wt > 0 and not stopped[sym]:
                # New entry
                if not was_in[sym]:
                    peak[sym] = price
                    entry_atr[sym] = cur_atr
                else:
                    peak[sym] = max(peak[sym], price)

                # Check stop
                stop_dist = atr_multiple * entry_atr[sym]
                if stop_dist > 0 and price < peak[sym] - stop_dist:
                    stopped[sym] = True
                    w_vals[i, j] = 0.0

                was_in[sym] = True

            elif base_wt > 0 and stopped[sym]:
                # Still stopped — need a fresh signal (base went to 0 then back)
                if not was_in[sym]:
                    stopped[sym] = False
                    peak[sym] = price
                    entry_atr[sym] = cur_atr
                    was_in[sym] = True
                else:
                    w_vals[i, j] = 0.0

            else:
                # Base weight is 0 — reset state
                stopped[sym] = False
                was_in[sym] = False
                peak[sym] = 0.0
                entry_atr[sym] = 0.0

    return pd.DataFrame(w_vals, index=w.index, columns=common)


def apply_vol_spike_exit(
    base_weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_lookback: int = 21,
    spike_multiple: float = 2.0,
) -> pd.DataFrame:
    """Exit when current realized vol > spike_multiple * entry-date vol."""
    common = base_weights.columns.intersection(returns_wide.columns)
    w = base_weights[common].copy()
    r = returns_wide[common].reindex(w.index).fillna(0.0)

    realized_vol = r.rolling(vol_lookback, min_periods=max(5, vol_lookback // 2)).std()

    entry_vol: dict[str, float] = {c: 0.0 for c in common}
    was_in: dict[str, bool] = {c: False for c in common}

    w_vals = w.values.copy()
    vol_vals = realized_vol.values
    col_map = {s: j for j, s in enumerate(common)}

    for i in range(len(w.index)):
        for sym in common:
            j = col_map[sym]
            base_wt = w_vals[i, j]
            cur_vol = vol_vals[i, j] if not np.isnan(vol_vals[i, j]) else 0.0

            if base_wt > 0:
                if not was_in[sym]:
                    entry_vol[sym] = cur_vol
                    was_in[sym] = True
                elif entry_vol[sym] > 0 and cur_vol > spike_multiple * entry_vol[sym]:
                    w_vals[i, j] = 0.0
                    was_in[sym] = False
            else:
                was_in[sym] = False
                entry_vol[sym] = 0.0

    return pd.DataFrame(w_vals, index=w.index, columns=common)


def apply_combined_exit(
    base_weights: pd.DataFrame,
    close_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    atr_multiple: float = 2.0,
    atr_lookback: int = 14,
) -> pd.DataFrame:
    """Signal reversal OR vol-adaptive trailing stop — whichever first.

    Since signal reversal is already embedded in base_weights (zero when
    signal <= 0), we only need to layer the trailing stop on top.
    """
    return apply_vol_adaptive_trailing_stop(
        base_weights, close_wide, returns_wide,
        atr_multiple=atr_multiple, atr_lookback=atr_lookback,
    )
