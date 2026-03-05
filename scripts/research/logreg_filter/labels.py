"""
Label construction for logistic regression trade-quality model.

Two label types:
  1. Barrier (TP-before-SL): y=1 if price hits take-profit before stop-loss
     within a fixed horizon.
  2. Forward return threshold: y=1 if forward return exceeds a configurable
     threshold (optionally volatility-scaled).

All labels are computed per-asset with correct temporal alignment.
Labels use future data (they are the training *target*) but are never
leaked into feature computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class LabelType(str, Enum):
    BARRIER = "barrier"
    FORWARD_RETURN = "forward_return"


class UnresolvedPolicy(str, Enum):
    CONSERVATIVE = "conservative"  # neither barrier hit → y=0
    SIGN = "sign"  # neither barrier hit → y=1 if terminal return > 0


@dataclass(frozen=True)
class BarrierLabelConfig:
    tp_atr: float = 2.0
    sl_atr: float = 1.0
    horizon: int = 20
    atr_window: int = 14
    unresolved_policy: UnresolvedPolicy = UnresolvedPolicy.CONSERVATIVE


@dataclass(frozen=True)
class ForwardReturnLabelConfig:
    horizon: int = 20
    threshold: float = 0.0
    vol_scaled: bool = False
    vol_scale_k: float = 0.5
    atr_window: int = 14


def _compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Average True Range computed from OHLC data."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def barrier_labels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    cfg: BarrierLabelConfig | None = None,
) -> pd.Series:
    """Compute barrier labels for a single asset.

    At each bar t (candidate entry):
      - entry price = close[t]
      - TP barrier = entry + tp_atr * ATR[t]
      - SL barrier = entry - sl_atr * ATR[t]
      - Scan forward up to H bars
      - y=1 if high touches TP before low touches SL

    Parameters
    ----------
    high, low, close : pd.Series
        Price series for a single asset, indexed by datetime.
    cfg : BarrierLabelConfig
        Label parameters.

    Returns
    -------
    pd.Series of {0, 1, NaN} indexed by datetime.
        NaN where ATR is unavailable or horizon extends beyond data.
    """
    if cfg is None:
        cfg = BarrierLabelConfig()

    atr = _compute_atr(high, low, close, cfg.atr_window)
    n = len(close)
    labels = np.full(n, np.nan)
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values
    atr_arr = atr.values

    for i in range(n):
        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            continue
        if i + cfg.horizon >= n:
            continue

        entry = close_arr[i]
        tp = entry + cfg.tp_atr * atr_arr[i]
        sl = entry - cfg.sl_atr * atr_arr[i]

        hit_tp = False
        hit_sl = False
        for j in range(i + 1, min(i + cfg.horizon + 1, n)):
            if high_arr[j] >= tp:
                hit_tp = True
                break
            if low_arr[j] <= sl:
                hit_sl = True
                break

        if hit_tp:
            labels[i] = 1.0
        elif hit_sl:
            labels[i] = 0.0
        else:
            if cfg.unresolved_policy == UnresolvedPolicy.CONSERVATIVE:
                labels[i] = 0.0
            else:
                end_idx = min(i + cfg.horizon, n - 1)
                labels[i] = 1.0 if close_arr[end_idx] > entry else 0.0

    return pd.Series(labels, index=close.index, name="label")


def forward_return_labels(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    cfg: ForwardReturnLabelConfig | None = None,
) -> pd.Series:
    """Compute forward-return threshold labels for a single asset.

    y=1 if forward return over H bars > threshold.
    If vol_scaled, y=1 if forward_return / ATR > k.

    Parameters
    ----------
    close : pd.Series
        Close prices for a single asset.
    high, low : pd.Series, optional
        Required only when vol_scaled=True (for ATR computation).
    cfg : ForwardReturnLabelConfig
        Label parameters.

    Returns
    -------
    pd.Series of {0, 1, NaN} indexed by datetime.
    """
    if cfg is None:
        cfg = ForwardReturnLabelConfig()

    fwd_ret = close.shift(-cfg.horizon) / close - 1.0

    if cfg.vol_scaled:
        if high is None or low is None:
            raise ValueError("high and low required for vol-scaled labels")
        atr = _compute_atr(high, low, close, cfg.atr_window)
        atr_norm = atr / close
        ratio = fwd_ret / atr_norm.replace(0, np.nan)
        raw = (ratio > cfg.vol_scale_k).astype(float)
    else:
        raw = (fwd_ret > cfg.threshold).astype(float)

    raw[fwd_ret.isna()] = np.nan
    return raw.rename("label")


def compute_labels_panel(
    panel: pd.DataFrame,
    label_type: LabelType = LabelType.BARRIER,
    barrier_cfg: BarrierLabelConfig | None = None,
    fwd_cfg: ForwardReturnLabelConfig | None = None,
) -> pd.DataFrame:
    """Compute labels for all assets in a long-format panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have columns: symbol, ts, open, high, low, close, volume.
    label_type : LabelType
        Which label method to use.
    barrier_cfg / fwd_cfg : config dataclasses
        Parameters for the chosen label type.

    Returns
    -------
    pd.DataFrame with columns: symbol, ts, label
    """
    results = []
    for sym, g in panel.groupby("symbol"):
        g = g.sort_values("ts").set_index("ts")
        if label_type == LabelType.BARRIER:
            lbl = barrier_labels(g["high"], g["low"], g["close"], barrier_cfg)
        else:
            lbl = forward_return_labels(
                g["close"], g["high"], g["low"], fwd_cfg,
            )
        lbl_df = lbl.reset_index()
        lbl_df.columns = ["ts", "label"]
        lbl_df["symbol"] = sym
        results.append(lbl_df)

    return pd.concat(results, ignore_index=True)
