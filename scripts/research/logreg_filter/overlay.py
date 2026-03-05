"""
Probability overlay: entry filter, conviction sizing, regime throttle.

Takes base strategy candidate weights and model probabilities, and produces
adjusted wide-format weight matrices ready for simple_backtest().
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverlayConfig:
    p_enter: float = 0.60
    p_hold: float = 0.45  # only used in hold-filter policy
    hold_filter: bool = False
    max_weight: float = 0.20
    target_gross: float = 1.0
    max_positions: int = 15
    # Regime throttle: piecewise-linear map from p_regime → gross multiplier
    regime_thresholds: list[float] = field(default_factory=lambda: [0.4, 0.6])
    regime_multipliers: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])


def apply_entry_filter(
    base_weights: pd.DataFrame,
    probs: pd.DataFrame,
    p_enter: float = 0.60,
    p_hold: float = 0.45,
    hold_filter: bool = False,
) -> pd.DataFrame:
    """Zero out positions where model probability is below threshold.

    Parameters
    ----------
    base_weights : pd.DataFrame
        Wide-format (index=ts, columns=symbols) of base strategy weights.
    probs : pd.DataFrame
        Wide-format (index=ts, columns=symbols) of model probabilities.
    p_enter : float
        Minimum probability for new entries.
    p_hold : float
        Minimum probability for existing holds (if hold_filter=True).
    hold_filter : bool
        If True, also filter existing holds below p_hold.

    Returns
    -------
    pd.DataFrame: filtered weights.
    """
    common_cols = base_weights.columns.intersection(probs.columns)
    common_ts = base_weights.index.intersection(probs.index)
    w = base_weights.reindex(index=common_ts, columns=common_cols).fillna(0.0)
    p = probs.reindex(index=common_ts, columns=common_cols).fillna(0.5)

    if hold_filter:
        mask = p >= p_hold
    else:
        prev_w = w.shift(1).fillna(0.0)
        is_new = (prev_w == 0) & (w > 0)
        is_held = prev_w > 0
        mask = (~is_new | (p >= p_enter)) & (~is_held | True)
        mask = mask & ((~is_new) | (p >= p_enter))
        new_blocked = is_new & (p < p_enter)
        mask = ~new_blocked

    return w.where(mask, 0.0)


def apply_conviction_sizing(
    filtered_weights: pd.DataFrame,
    probs: pd.DataFrame,
    target_gross: float = 1.0,
    max_weight: float = 0.20,
    max_positions: int = 15,
) -> pd.DataFrame:
    """Scale weights by edge = max(0, p - 0.5), normalised to target gross.

    Parameters
    ----------
    filtered_weights : pd.DataFrame
        Wide-format weights after entry filter (zeros for excluded assets).
    probs : pd.DataFrame
        Wide-format model probabilities.
    target_gross : float
        Target gross exposure.
    max_weight : float
        Maximum weight per asset.
    max_positions : int
        Maximum number of positions.

    Returns
    -------
    pd.DataFrame: conviction-sized weights.
    """
    common_cols = filtered_weights.columns.intersection(probs.columns)
    common_ts = filtered_weights.index.intersection(probs.index)
    w = filtered_weights.reindex(index=common_ts, columns=common_cols).fillna(0.0)
    p = probs.reindex(index=common_ts, columns=common_cols).fillna(0.5)

    active = w > 0
    edge = (p - 0.5).clip(lower=0.0) * active

    # Keep only top-N by edge each day
    for ts in edge.index:
        row = edge.loc[ts]
        if row.sum() == 0:
            continue
        n_active = (row > 0).sum()
        if n_active > max_positions:
            threshold = row.nlargest(max_positions).iloc[-1]
            edge.loc[ts, row < threshold] = 0.0

    row_sum = edge.sum(axis=1).replace(0, np.nan)
    sized = edge.div(row_sum, axis=0).fillna(0.0) * target_gross
    sized = sized.clip(upper=max_weight)

    # Re-normalise after clipping
    clipped_sum = sized.sum(axis=1).replace(0, np.nan)
    sized = sized.div(clipped_sum, axis=0).fillna(0.0) * target_gross

    return sized


def apply_regime_throttle(
    weights: pd.DataFrame,
    p_regime: pd.Series,
    thresholds: list[float] | None = None,
    multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """Scale gross exposure based on regime probability.

    The piecewise-linear mapping has len(thresholds) breakpoints and
    len(thresholds)+1 multiplier values.

    Default: p_regime < 0.4 → 0.0, 0.4–0.6 → 0.5, >= 0.6 → 1.0

    Parameters
    ----------
    weights : pd.DataFrame
        Wide-format weights.
    p_regime : pd.Series
        Regime probability indexed by ts.
    thresholds : list[float]
        Breakpoints (ascending).
    multipliers : list[float]
        Gross multiplier for each region (len = len(thresholds) + 1).

    Returns
    -------
    pd.DataFrame: throttled weights.
    """
    if thresholds is None:
        thresholds = [0.4, 0.6]
    if multipliers is None:
        multipliers = [0.0, 0.5, 1.0]

    p = p_regime.reindex(weights.index).ffill().fillna(0.5)

    scale = pd.Series(multipliers[-1], index=weights.index)
    for i in range(len(thresholds) - 1, -1, -1):
        scale = scale.where(p >= thresholds[i], multipliers[i])

    return weights.mul(scale, axis=0)


def _to_wide_probs(preds: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format predictions to wide-format probability matrix."""
    return preds.pivot_table(
        index="ts", columns="symbol", values="p_success", aggfunc="last",
    )


def build_overlay_variants(
    base_weights: pd.DataFrame,
    predictions: pd.DataFrame,
    p_regime: pd.Series | None = None,
    cfg: OverlayConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all four ablation variants.

    Returns
    -------
    dict mapping variant name to wide-format weight DataFrame:
      - "baseline": base weights unchanged
      - "filter": entry filter applied
      - "filter_sizing": filter + conviction sizing
      - "filter_sizing_regime": filter + sizing + regime throttle
    """
    if cfg is None:
        cfg = OverlayConfig()

    probs_wide = _to_wide_probs(predictions)

    variants: dict[str, pd.DataFrame] = {"baseline": base_weights}

    filtered = apply_entry_filter(
        base_weights, probs_wide,
        p_enter=cfg.p_enter, p_hold=cfg.p_hold, hold_filter=cfg.hold_filter,
    )
    variants["filter"] = filtered

    sized = apply_conviction_sizing(
        filtered, probs_wide,
        target_gross=cfg.target_gross,
        max_weight=cfg.max_weight,
        max_positions=cfg.max_positions,
    )
    variants["filter_sizing"] = sized

    if p_regime is not None and len(p_regime) > 0:
        throttled = apply_regime_throttle(
            sized, p_regime,
            thresholds=cfg.regime_thresholds,
            multipliers=cfg.regime_multipliers,
        )
        variants["filter_sizing_regime"] = throttled

    return variants
