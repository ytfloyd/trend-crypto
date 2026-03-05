"""
Curated feature set for logistic regression trade-quality model.

Designed for interpretability — small set (~15 features) computed from
OHLCV data only.  All features use data available up to and including
time t (no lookahead).  Optional cross-asset / regime features degrade
gracefully when inputs are unavailable.

Feature groups:
  - Trend / Momentum: trailing returns, MA ratio, MA slope
  - Breakout structure: distance to Donchian high/low
  - Volatility: normalised ATR, ATR percentile
  - Volume: volume z-score
  - Cross-asset / regime: BTC return, BTC ATR percentile, market breadth
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    ret_windows: list[int] = field(default_factory=lambda: [1, 5, 20])
    ma_fast: int = 10
    ma_slow: int = 50
    donchian_window: int = 20
    atr_window: int = 14
    atr_pctl_lookback: int = 252
    vol_zscore_window: int = 20
    btc_ret_window: int = 20
    breadth_ma_window: int = 50


def _compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def compute_features_single(
    g: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute features for a single asset (long-format group).

    Parameters
    ----------
    g : pd.DataFrame
        Single-asset slice with columns: ts, open, high, low, close, volume.
        Must be sorted by ts.
    cfg : FeatureConfig
        Feature parameters.

    Returns
    -------
    pd.DataFrame with original columns plus feature columns.
    """
    if cfg is None:
        cfg = FeatureConfig()

    g = g.copy()
    c = g["close"]
    h = g["high"]
    lo = g["low"]
    v = g["volume"]

    # --- Trend / Momentum ---
    for w in cfg.ret_windows:
        g[f"ret_{w}"] = c / c.shift(w) - 1.0

    ma_fast = c.rolling(cfg.ma_fast, min_periods=cfg.ma_fast).mean()
    ma_slow = c.rolling(cfg.ma_slow, min_periods=cfg.ma_slow).mean()
    g["ma_ratio"] = ma_fast / ma_slow - 1.0
    g["ma_slope"] = (ma_slow - ma_slow.shift(5)) / ma_slow.shift(5)

    # --- Breakout structure ---
    hh = h.rolling(cfg.donchian_window, min_periods=cfg.donchian_window).max()
    ll = lo.rolling(cfg.donchian_window, min_periods=cfg.donchian_window).min()
    g["dist_donch_high"] = (c - hh) / c
    g["dist_donch_low"] = (c - ll) / c

    # --- Volatility ---
    atr = _compute_atr(h, lo, c, cfg.atr_window)
    g["atr_norm"] = atr / c
    g["atr_pctl"] = atr.rolling(
        cfg.atr_pctl_lookback, min_periods=cfg.atr_window,
    ).rank(pct=True)

    # --- Volume ---
    v_ma = v.rolling(cfg.vol_zscore_window, min_periods=cfg.vol_zscore_window).mean()
    v_std = v.rolling(cfg.vol_zscore_window, min_periods=cfg.vol_zscore_window).std()
    g["vol_zscore"] = (v - v_ma) / v_std.replace(0, np.nan)

    return g


def get_feature_columns(cfg: FeatureConfig | None = None) -> list[str]:
    """Return the list of feature column names produced by compute_features_single."""
    if cfg is None:
        cfg = FeatureConfig()
    cols = [f"ret_{w}" for w in cfg.ret_windows]
    cols += [
        "ma_ratio", "ma_slope",
        "dist_donch_high", "dist_donch_low",
        "atr_norm", "atr_pctl",
        "vol_zscore",
    ]
    return cols


CROSS_ASSET_COLS = ["btc_ret", "btc_atr_pctl", "breadth"]


def compute_cross_asset_features(
    panel: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute market-wide features shared across all assets.

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel with columns: symbol, ts, close, high, low, volume.

    Returns
    -------
    pd.DataFrame indexed by ts with columns: btc_ret, btc_atr_pctl, breadth.
    """
    if cfg is None:
        cfg = FeatureConfig()

    out = pd.DataFrame(index=panel["ts"].drop_duplicates().sort_values())

    btc = panel.loc[panel["symbol"] == "BTC-USD"].sort_values("ts").set_index("ts")
    if len(btc) > cfg.btc_ret_window:
        out["btc_ret"] = btc["close"] / btc["close"].shift(cfg.btc_ret_window) - 1.0
        btc_atr = _compute_atr(btc["high"], btc["low"], btc["close"], cfg.atr_window)
        out["btc_atr_pctl"] = btc_atr.rolling(
            cfg.atr_pctl_lookback, min_periods=cfg.atr_window,
        ).rank(pct=True)
    else:
        out["btc_ret"] = np.nan
        out["btc_atr_pctl"] = np.nan

    ma_slow = panel.groupby("symbol")["close"].transform(
        lambda s: s.rolling(cfg.breadth_ma_window, min_periods=cfg.breadth_ma_window).mean()
    )
    above_ma = (panel["close"] > ma_slow).astype(float)
    above_ma.index = panel["ts"].values
    breadth = above_ma.groupby(above_ma.index).mean()
    breadth = breadth[~breadth.index.duplicated(keep="last")]
    out["breadth"] = breadth

    return out


def compute_features_panel(
    panel: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute all features for the full panel.

    Returns long-format DataFrame with original columns + per-asset features
    + cross-asset features joined via ts.
    """
    if cfg is None:
        cfg = FeatureConfig()

    parts = []
    for _, g in panel.groupby("symbol"):
        g = g.sort_values("ts")
        parts.append(compute_features_single(g, cfg))
    featured = pd.concat(parts, ignore_index=True)

    cross = compute_cross_asset_features(panel, cfg)
    featured = featured.merge(cross, left_on="ts", right_index=True, how="left")

    return featured


def get_all_feature_columns(cfg: FeatureConfig | None = None) -> list[str]:
    """Return the complete feature column list (per-asset + cross-asset)."""
    return get_feature_columns(cfg) + CROSS_ASSET_COLS
