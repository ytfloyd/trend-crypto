"""
Shared risk overlay utilities.

Used by all paper-recreation and multi-frequency research packages.
Provides portfolio-level risk management functions that operate on
wide-format weight DataFrames (index=ts, columns=symbols).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import simple_backtest
from .data import ANN_FACTOR


# ---------------------------------------------------------------------------
# Position limits
# ---------------------------------------------------------------------------
def apply_position_limit_wide(
    weights: pd.DataFrame,
    max_wt: float,
) -> pd.DataFrame:
    """Cap individual weights in a wide-format weight matrix.

    Iteratively caps and redistributes excess proportionally until
    no single weight exceeds ``max_wt`` as a fraction of row total.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format: index=ts, columns=symbols, values=target weights.
    max_wt : float
        Maximum allowed weight as a fraction of row total (e.g. 0.15).
    """
    w = weights.copy()
    for _ in range(10):
        row_sum = w.sum(axis=1).replace(0, np.nan)
        pct = w.div(row_sum, axis=0).fillna(0)
        over = pct > max_wt
        if not over.any().any():
            break
        w = w.where(~over, pct.clip(upper=max_wt).mul(row_sum, axis=0))
        new_sum = w.sum(axis=1).replace(0, np.nan)
        scale = (row_sum / new_sum).fillna(1.0)
        w = w.mul(scale, axis=0)
    return w


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------
def apply_vol_targeting(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vol_target: float = 0.20,
    lookback: int = 42,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Scale portfolio weights so realized vol ≈ ``vol_target``.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weight matrix.
    returns_wide : pd.DataFrame
        Wide-format return matrix (same column universe).
    vol_target : float
        Target annualized volatility (e.g. 0.20 for 20%).
    lookback : int
        Rolling window for realized portfolio vol estimate.
    max_leverage : float
        Cap on the scaling factor to prevent excessive leverage.
    """
    common = weights.columns.intersection(returns_wide.columns)
    w = weights[common].copy()
    r = returns_wide[common].reindex(w.index).fillna(0.0)

    w_held = w.shift(1).fillna(0.0)
    port_ret = (w_held * r).sum(axis=1)
    realized_vol = (
        port_ret.rolling(lookback, min_periods=max(10, lookback // 2)).std()
        * np.sqrt(ANN_FACTOR)
    )
    scalar = (vol_target / realized_vol).clip(lower=0.0, upper=max_leverage).fillna(1.0)
    return w.mul(scalar, axis=0)


# ---------------------------------------------------------------------------
# Drawdown control
# ---------------------------------------------------------------------------
def apply_dd_control(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    dd_threshold: float = 0.30,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Linearly scale down exposure as drawdown approaches threshold.

    At 0% DD: full weight.
    At dd_threshold: 50% weight.
    At 2 × dd_threshold: 0% weight (fully in cash).

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weight matrix.
    returns_wide : pd.DataFrame
        Wide-format return matrix.
    dd_threshold : float
        Drawdown level at which exposure is halved (e.g. 0.30 for 30%).
    cost_bps : float
        Transaction cost for the preliminary backtest (for DD calc).
    """
    common = weights.columns.intersection(returns_wide.columns)
    w = weights[common].copy()
    r = returns_wide[common].reindex(w.index).fillna(0.0)

    bt = simple_backtest(w, r, cost_bps=cost_bps)
    bt["ts"] = pd.to_datetime(bt["ts"])
    eq = bt.set_index("ts")["portfolio_equity"]
    dd = eq / eq.cummax() - 1.0
    scale = (1.0 + dd / (2.0 * dd_threshold)).clip(0.0, 1.0)
    return w.mul(scale, axis=0)


# ---------------------------------------------------------------------------
# Trailing stop-loss
# ---------------------------------------------------------------------------
def apply_trailing_stop(
    base_weights: pd.DataFrame,
    close_wide: pd.DataFrame,
    stop_pct: float,
) -> pd.DataFrame:
    """Zero out asset weight when price drops >stop_pct from peak while held.

    After being stopped out, the asset can re-enter at the next non-zero
    base weight (i.e., at the next rebalance that selects it).

    Parameters
    ----------
    base_weights : pd.DataFrame
        Wide-format weight matrix (pre-overlay).
    close_wide : pd.DataFrame
        Wide-format close price matrix.
    stop_pct : float
        Stop-loss trigger as a fraction (e.g. 0.15 for 15%).
    """
    w = base_weights.copy()
    common = w.columns.intersection(close_wide.columns)
    w = w[common]
    cls = close_wide[common].reindex(w.index).ffill()

    stopped: dict[str, bool] = {c: False for c in common}
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
                    w.iloc[i, col_idx] = 0.0
            elif base_wt > 0 and stopped[sym]:
                if i > 0:
                    prev_base = (
                        base_weights.at[w.index[i - 1], sym]
                        if sym in base_weights.columns else 0.0
                    )
                    if prev_base == 0.0 or base_wt != prev_base:
                        stopped[sym] = False
                        peak[sym] = price
                    else:
                        w.iloc[i, col_idx] = 0.0
                else:
                    w.iloc[i, col_idx] = 0.0
            else:
                stopped[sym] = False
                peak[sym] = 0.0

    # Re-normalize so weights sum to original exposure on each day
    orig_sum = base_weights[common].sum(axis=1).replace(0, np.nan)
    new_sum = w.sum(axis=1).replace(0, np.nan)
    scale = (orig_sum / new_sum).fillna(0.0).clip(upper=2.0)
    w = w.multiply(scale, axis=0)
    return w
