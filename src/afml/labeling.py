"""
Triple-barrier labeling and meta-labeling.

Implements the labeling methods from AFML Chapters 3–4:
  - Triple-barrier method (profit-taking, stop-loss, max holding period)
  - Vertical barrier (time expiry) labels
  - Meta-labeling (learn bet size given a primary model's side)
  - Daily volatility estimator used for dynamic barrier widths

All functions operate on pandas DataFrames of OHLCV bars (any bar type).

Reference:
    López de Prado, M. (2018) *Advances in Financial Machine Learning*,
    Chapter 3: Labeling (pp. 43–67).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Daily volatility (used to set dynamic barrier widths)
# -----------------------------------------------------------------------

def daily_volatility(
    close: pd.Series,
    span: int = 20,
) -> pd.Series:
    """Exponentially-weighted daily volatility of log returns.

    This is the ``getDailyVol`` function from AFML Snippet 3.1.
    Used to set dynamic profit-taking and stop-loss barriers that
    adapt to current volatility.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed by datetime.
    span : int
        EWMA span in bars.

    Returns
    -------
    pd.Series of volatility estimates, same index as *close*.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.ewm(span=span, min_periods=span).std()


# -----------------------------------------------------------------------
# Triple-barrier method (AFML Snippet 3.2 / 3.4)
# -----------------------------------------------------------------------

def triple_barrier_labels(
    close: pd.Series,
    events: pd.DataFrame,
    *,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    molecule: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Apply the triple-barrier method to label observations.

    For each event (entry point), we define three barriers:
      1. **Upper (profit-taking)**: price rises by ``pt_sl[0] * width``
      2. **Lower (stop-loss)**: price falls by ``pt_sl[1] * width``
      3. **Vertical (time expiry)**: max holding period expires

    The label is determined by which barrier is touched first.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed by datetime.
    events : pd.DataFrame
        Must have columns:
        - ``t1``: vertical barrier datetime (max holding period end)
        - ``trgt``: barrier width (e.g. daily volatility)
        - ``side`` (optional): +1 for long, -1 for short. If absent,
          labels are unsigned (1 = profit-take, -1 = stop-loss, 0 = expiry).
    pt_sl : tuple[float, float]
        Multipliers for (profit-taking, stop-loss) barriers.
        Set either to 0 to disable that barrier.
    molecule : pd.DatetimeIndex | None
        Subset of event indices to process (for parallelisation).

    Returns
    -------
    pd.DataFrame with columns:
        - ``t1``: timestamp when first barrier was touched
        - ``ret``: return at first barrier touch
        - ``label``: {-1, 0, 1} — the label
        - ``trgt``: the target/width used
    """
    if molecule is None:
        molecule = events.index

    has_side = "side" in events.columns

    records = []
    for loc in molecule:
        if loc not in events.index:
            continue

        t1 = events.at[loc, "t1"]
        trgt = events.at[loc, "trgt"]
        side = events.at[loc, "side"] if has_side else 1.0

        # Price path from entry to vertical barrier
        path = close.loc[loc:t1]
        if len(path) < 2:
            continue

        # Returns relative to entry
        entry_price = close.at[loc]
        path_ret = (path / entry_price - 1.0) * side

        # Compute barrier levels
        pt_level = trgt * pt_sl[0] if pt_sl[0] > 0 else np.inf
        sl_level = -trgt * pt_sl[1] if pt_sl[1] > 0 else -np.inf

        # Find first touch times
        pt_touch = path_ret[path_ret >= pt_level].index
        sl_touch = path_ret[path_ret <= sl_level].index

        earliest = t1  # default: vertical barrier
        label = 0
        touch_ret = float(path_ret.iloc[-1])

        if len(pt_touch) > 0 and (len(sl_touch) == 0 or pt_touch[0] <= sl_touch[0]):
            earliest = pt_touch[0]
            label = 1
            touch_ret = float(path_ret.at[earliest])
        elif len(sl_touch) > 0:
            earliest = sl_touch[0]
            label = -1
            touch_ret = float(path_ret.at[earliest])

        records.append({
            "t_entry": loc,
            "t1": earliest,
            "ret": touch_ret,
            "label": label,
            "trgt": trgt,
        })

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.set_index("t_entry")
    return out


# -----------------------------------------------------------------------
# Event generation helpers
# -----------------------------------------------------------------------

def make_events(
    close: pd.Series,
    vol: pd.Series,
    *,
    holding_periods: int = 10,
    side: pd.Series | None = None,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Create the *events* DataFrame for triple-barrier labeling.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed by datetime.
    vol : pd.Series
        Volatility series (from ``daily_volatility``), same index.
    holding_periods : int
        Number of bars for the vertical barrier.
    side : pd.Series | None
        +1 / -1 signal from a primary model.  If None, defaults to +1
        (long only, unsigned labels).
    min_ret : float
        Minimum target width (floor for ``trgt``).

    Returns
    -------
    pd.DataFrame with columns: t1, trgt, side.
    """
    vol = vol.dropna()
    idx = vol.index.intersection(close.index)

    t1 = close.index.searchsorted(idx) + holding_periods
    t1 = t1.clip(max=len(close.index) - 1)
    t1_dates = close.index[t1]

    events = pd.DataFrame({
        "t1": t1_dates,
        "trgt": vol.reindex(idx).clip(lower=min_ret).values,
        "side": side.reindex(idx).values if side is not None else np.ones(len(idx)),
    }, index=idx)

    return events.dropna(subset=["trgt"])


# -----------------------------------------------------------------------
# Meta-labeling (AFML Section 3.6)
# -----------------------------------------------------------------------

def meta_labels(
    primary_side: pd.Series,
    triple_barrier_out: pd.DataFrame,
) -> pd.DataFrame:
    """Create meta-labels from a primary model's side and triple-barrier output.

    Meta-labeling decomposes the trading decision into two steps:
      1. **Primary model** decides the *side* (long/short)
      2. **Meta model** decides whether to *take the bet* (and how much)

    The meta-label is binary: 1 if the primary model's side was
    profitable (i.e. the trade hit profit-take or expired with positive
    return), 0 otherwise.

    Parameters
    ----------
    primary_side : pd.Series
        +1 / -1 side predictions from the primary model, indexed by datetime.
    triple_barrier_out : pd.DataFrame
        Output of ``triple_barrier_labels`` with columns: t1, ret, label, trgt.

    Returns
    -------
    pd.DataFrame with columns:
        - ``side``: from primary model
        - ``ret``: realised return (signed by primary side)
        - ``meta_label``: 1 if profitable, 0 if not
        - ``t1``: exit time
    """
    common = primary_side.index.intersection(triple_barrier_out.index)
    if len(common) == 0:
        return pd.DataFrame(columns=["side", "ret", "meta_label", "t1"])

    side = primary_side.reindex(common)
    tb = triple_barrier_out.reindex(common)

    signed_ret = tb["ret"] * side
    meta = (signed_ret > 0).astype(int)

    return pd.DataFrame({
        "side": side.values,
        "ret": signed_ret.values,
        "meta_label": meta.values,
        "t1": tb["t1"].values,
    }, index=common)


# -----------------------------------------------------------------------
# Fixed-horizon labels (simple baseline for comparison)
# -----------------------------------------------------------------------

def fixed_horizon_labels(
    close: pd.Series,
    horizon: int = 1,
    method: Literal["sign", "threshold"] = "sign",
    threshold: float = 0.0,
) -> pd.Series:
    """Label by forward return sign or threshold crossing.

    This is the naive labeling approach that AFML argues against.
    Included for comparison in notebooks.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    horizon : int
        Forward return horizon in bars.
    method : str
        ``"sign"`` = sign of forward return,
        ``"threshold"`` = 1 if return > threshold, -1 if < -threshold, 0 otherwise.
    threshold : float
        Used only when method = ``"threshold"``.

    Returns
    -------
    pd.Series of labels {-1, 0, 1}.
    """
    fwd_ret = close.shift(-horizon) / close - 1.0

    if method == "sign":
        return np.sign(fwd_ret).fillna(0).astype(int)

    labels = pd.Series(0, index=close.index, dtype=int)
    labels[fwd_ret > threshold] = 1
    labels[fwd_ret < -threshold] = -1
    return labels
