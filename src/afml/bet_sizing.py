"""
Bet sizing: translating ML predictions into position sizes.

Standard ML outputs (probabilities or labels) don't directly map to
optimal position sizes.  This module implements the bet sizing approaches
from AFML Chapter 10:

  - **avg_active_signals**: current average signal (accounts for concurrency)
  - **discrete_signal**: discretize a continuous signal into buckets
  - **bet_size_sigmoid**: size positions using a sigmoid/CDF function
  - **bet_size_from_prob**: convert predicted probabilities to bet sizes
    using the inverse-CDF approach (accounts for divergence from 0.5)

Reference:
    López de Prado, M. (2018) *Advances in Financial Machine Learning*,
    Chapter 10: Bet Sizing (pp. 141–155).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# -----------------------------------------------------------------------
# Active signals  (AFML Snippet 10.1)
# -----------------------------------------------------------------------

def avg_active_signals(
    signals: pd.DataFrame,
    t1: pd.Series,
) -> pd.Series:
    """Compute the average active signal at each time step.

    At any point in time, multiple bets may be active (their labels
    haven't expired).  This function averages the signals of all
    concurrently active bets.

    Parameters
    ----------
    signals : pd.DataFrame
        Must have a column ``"signal"`` indexed by start time.
    t1 : pd.Series
        Label end times (index = start, value = end).

    Returns
    -------
    pd.Series — time-weighted average signal at each point.
    """
    out = pd.Series(dtype=float)
    t1 = t1.reindex(signals.index)

    for loc in signals.index:
        # All signals active at this time
        active = signals.loc[
            (signals.index <= loc) & (t1.loc[signals.index] >= loc),
            "signal",
        ]
        if len(active) > 0:
            out.loc[loc] = active.mean()

    return out


# -----------------------------------------------------------------------
# Discretize  (AFML Snippet 10.2)
# -----------------------------------------------------------------------

def discrete_signal(
    signal: pd.Series,
    step_size: float = 0.1,
) -> pd.Series:
    """Discretize a continuous signal into fixed step sizes.

    Maps a continuous [-1, 1] signal into discrete buckets of
    size ``step_size``.

    Parameters
    ----------
    signal : pd.Series
        Continuous signal values.
    step_size : float
        Bucket width.  E.g. 0.1 produces 21 levels from -1.0 to 1.0.

    Returns
    -------
    pd.Series — discretized signal.
    """
    return (signal / step_size).round() * step_size


# -----------------------------------------------------------------------
# Sigmoid bet size  (AFML Snippet 10.3)
# -----------------------------------------------------------------------

def bet_size_sigmoid(
    signal: pd.Series,
    w: float = 10.0,
) -> pd.Series:
    """Map a continuous signal to position size via sigmoid.

    Uses ``2 * sigmoid(w * signal) - 1`` so output ∈ (-1, 1).
    Parameter ``w`` controls aggressiveness: higher w = more binary bets.

    Parameters
    ----------
    signal : pd.Series
        Continuous signal (typically in [-1, 1]).
    w : float
        Slope parameter.  w → ∞ makes sizing binary {-1, +1}.

    Returns
    -------
    pd.Series — position sizes in (-1, 1).
    """
    return 2 / (1 + np.exp(-w * signal)) - 1


# -----------------------------------------------------------------------
# Probability → bet size  (AFML Snippet 10.4)
# -----------------------------------------------------------------------

def bet_size_from_prob(
    prob: pd.Series,
    pred_side: pd.Series | None = None,
    num_classes: int = 2,
) -> pd.Series:
    """Convert predicted probability to bet size using inverse-CDF.

    For a two-class problem, the signal equals:
        ``side * (2 * prob - 1)``
    then mapped through the normal CDF to get a position size.

    Parameters
    ----------
    prob : pd.Series
        Predicted probability of the positive class (or of the predicted class).
    pred_side : pd.Series | None
        +1 (long) or -1 (short) for each prediction.
        If None, assumes all long.
    num_classes : int
        Number of classes (default 2).

    Returns
    -------
    pd.Series — bet sizes in [-1, 1].
    """
    if pred_side is None:
        pred_side = pd.Series(1, index=prob.index)

    # z-score: how far the probability diverges from 1/num_classes
    z = (prob - 1.0 / num_classes) / (prob * (1 - prob)).clip(lower=1e-8).pow(0.5)
    size = pred_side * (2 * norm.cdf(z) - 1)
    return size


# -----------------------------------------------------------------------
# Budgeting / max concurrent positions
# -----------------------------------------------------------------------

def max_concurrent_signals(t1: pd.Series) -> int:
    """Count the maximum number of concurrently active bets.

    Useful for position-level budgeting: if max concurrency is C,
    each bet should be sized at most 1/C to keep gross exposure ≤ 1.

    Parameters
    ----------
    t1 : pd.Series
        Index = bet start, value = bet end.

    Returns
    -------
    int
    """
    events = []
    for start, end in zip(t1.index, t1.values):
        events.append((start, 0, 1))   # 0 = open (sort before close)
        events.append((end, 1, -1))     # 1 = close
    events.sort()

    max_c = 0
    current = 0
    for _, _, delta in events:
        current += delta
        max_c = max(max_c, current)
    return max_c
