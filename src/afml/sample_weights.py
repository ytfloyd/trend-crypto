"""
Sample weights for overlapping labels.

Implements the sample weighting and sequential bootstrap from AFML Chapter 4:
  - Label concurrency (how many labels are active at each bar)
  - Average uniqueness per label
  - Sample weights by return attribution
  - Sequential bootstrap (IID-like resampling respecting temporal overlaps)

When triple-barrier labels span multiple bars, observations are not
independent.  Naive ML training double-counts information in overlapping
regions.  These tools correct for that.

Reference:
    López de Prado, M. (2018) *Advances in Financial Machine Learning*,
    Chapter 4: Sample Weights (pp. 69–85).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Concurrency and uniqueness  (AFML Snippets 4.1–4.2)
# -----------------------------------------------------------------------

def label_concurrency(
    t1: pd.Series,
    close_index: pd.DatetimeIndex,
) -> pd.Series:
    """Count how many labels are active at each bar.

    Parameters
    ----------
    t1 : pd.Series
        Index = entry time, value = exit time (from triple-barrier ``t1``).
    close_index : pd.DatetimeIndex
        Index of the close price series (all bar timestamps).

    Returns
    -------
    pd.Series of integer concurrency counts, indexed by *close_index*.
    """
    concurrency = pd.Series(0, index=close_index, dtype=int)
    for entry, exit_t in t1.items():
        mask = (concurrency.index >= entry) & (concurrency.index <= exit_t)
        concurrency.loc[mask] += 1
    return concurrency


def average_uniqueness(
    t1: pd.Series,
    close_index: pd.DatetimeIndex,
) -> pd.Series:
    """Compute the average uniqueness of each label.

    A label's uniqueness at bar *t* is ``1 / concurrency[t]``.
    The average uniqueness over the label's lifespan measures how
    much exclusive information it carries.

    Labels with high uniqueness are more informative; those with low
    uniqueness overlap heavily with other labels and contribute
    less independent information.

    Parameters
    ----------
    t1 : pd.Series
        Index = entry time, value = exit time.
    close_index : pd.DatetimeIndex
        All bar timestamps.

    Returns
    -------
    pd.Series of average uniqueness per label, indexed by entry time.
        Values in (0, 1]; 1.0 = fully unique (no overlap).

    Reference
    ---------
    AFML Snippet 4.2.
    """
    conc = label_concurrency(t1, close_index)

    uniqueness = pd.Series(np.nan, index=t1.index, dtype=float)
    for entry, exit_t in t1.items():
        mask = (conc.index >= entry) & (conc.index <= exit_t)
        bars_in_label = conc.loc[mask]
        if len(bars_in_label) == 0:
            continue
        uniqueness.at[entry] = (1.0 / bars_in_label).mean()

    return uniqueness


# -----------------------------------------------------------------------
# Sample weights by return attribution  (AFML Snippet 4.3)
# -----------------------------------------------------------------------

def sample_weight_by_return(
    t1: pd.Series,
    close: pd.Series,
    concurrency: pd.Series | None = None,
) -> pd.Series:
    """Compute sample weights proportional to attributed returns.

    Each bar's return is split equally among all labels active at that
    bar.  A label's weight is the absolute sum of its attributed returns.

    This ensures that labels during high-concurrency periods (where
    many trades overlap) receive lower weight, preventing the ML model
    from over-fitting to crowded regimes.

    Parameters
    ----------
    t1 : pd.Series
        Index = entry time, value = exit time.
    close : pd.Series
        Close prices indexed by datetime.
    concurrency : pd.Series | None
        Pre-computed concurrency.  If None, computed internally.

    Returns
    -------
    pd.Series of non-negative sample weights, indexed by entry time.

    Reference
    ---------
    AFML Snippet 4.3.
    """
    if concurrency is None:
        concurrency = label_concurrency(t1, close.index)

    log_ret = np.log(close / close.shift(1))

    weights = pd.Series(np.nan, index=t1.index, dtype=float)
    for entry, exit_t in t1.items():
        mask = (close.index >= entry) & (close.index <= exit_t)
        bars = close.index[mask]
        if len(bars) < 2:
            weights.at[entry] = 0.0
            continue

        # Attributed return: each bar's log-return / concurrency at that bar
        attr_ret = log_ret.reindex(bars) / concurrency.reindex(bars).clip(lower=1)
        weights.at[entry] = attr_ret.abs().sum()

    return weights


def normalize_weights(weights: pd.Series) -> pd.Series:
    """Normalize sample weights to sum to the number of samples.

    This preserves the effective sample size when passed to sklearn's
    ``sample_weight`` parameter.
    """
    total = weights.sum()
    if total <= 0:
        return pd.Series(1.0, index=weights.index)
    return weights * len(weights) / total


# -----------------------------------------------------------------------
# Sequential bootstrap  (AFML Snippet 4.5)
# -----------------------------------------------------------------------

def _indicator_matrix(
    t1: pd.Series,
    close_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Build the indicator matrix: I[i, t] = 1 if label i spans bar t.

    Returns
    -------
    np.ndarray of shape (n_labels, n_bars), dtype bool.
    """
    bar_loc = {ts: j for j, ts in enumerate(close_index)}
    n_labels = len(t1)
    n_bars = len(close_index)
    ind = np.zeros((n_labels, n_bars), dtype=bool)

    for i, (entry, exit_t) in enumerate(t1.items()):
        start_idx = bar_loc.get(entry)
        end_idx = bar_loc.get(exit_t)
        if start_idx is not None and end_idx is not None:
            ind[i, start_idx : end_idx + 1] = True

    return ind


def sequential_bootstrap(
    t1: pd.Series,
    close_index: pd.DatetimeIndex,
    n_samples: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Draw a bootstrap sample respecting label overlaps.

    At each draw, the probability of selecting a label is proportional
    to its average uniqueness *conditioned on the labels already drawn*.
    This produces a more IID-like sample than naive bootstrapping.

    Parameters
    ----------
    t1 : pd.Series
        Index = entry time, value = exit time.
    close_index : pd.DatetimeIndex
        All bar timestamps.
    n_samples : int | None
        Number of samples to draw. Defaults to ``len(t1)``.
    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray of integer indices into ``t1`` (drawn with replacement).

    Reference
    ---------
    AFML Snippet 4.5.
    """
    rng = np.random.default_rng(random_state)
    ind = _indicator_matrix(t1, close_index)
    n_labels, n_bars = ind.shape

    if n_samples is None:
        n_samples = n_labels

    # Track how many selected labels are active at each bar
    phi = np.zeros(n_bars, dtype=float)
    selected: list[int] = []

    for _ in range(n_samples):
        avg_u = np.zeros(n_labels, dtype=float)
        for i in range(n_labels):
            active_bars = ind[i]
            if not active_bars.any():
                avg_u[i] = 0.0
                continue
            # Uniqueness = 1 / (concurrency of already-selected + this label)
            conc_if_added = phi[active_bars] + 1.0
            avg_u[i] = (1.0 / conc_if_added).mean()

        # Convert to probabilities
        total_u = avg_u.sum()
        if total_u <= 0:
            prob = np.ones(n_labels) / n_labels
        else:
            prob = avg_u / total_u

        chosen = rng.choice(n_labels, p=prob)
        selected.append(chosen)
        phi[ind[chosen]] += 1.0

    return np.array(selected)
