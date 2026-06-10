"""Leakage-safe first-layer futures features."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Lookbacks for the first K2 feature layer."""

    atr_lookback: int = 14
    realized_vol_lookbacks: tuple[int, ...] = (12, 24, 60)
    vol_of_vol_lookback: int = 24
    compression_lookback: int = 60
    return_horizons: tuple[int, ...] = (1, 3, 6, 12, 24)
    breakout_lookback: int = 20
    trend_lookback: int = 20
    annualization_by_timeframe: dict[str, float] = field(
        default_factory=lambda: {"1h": 24.0 * 252.0, "4h": 6.0 * 252.0, "1d": 252.0}
    )


def build_feature_frame(bars: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Append the first research feature layer to an OHLCV bar frame.

    All rolling features use the current completed bar and prior bars only.
    Breakout references are shifted by one bar so the current high/low cannot
    define its own breakout threshold.
    """
    cfg = config or FeatureConfig()
    if bars.empty:
        return bars.copy()
    out = bars.sort_values("ts").reset_index(drop=True).copy()
    close = out["c"].astype(float)
    high = out["h"].astype(float)
    low = out["l"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["true_range"] = true_range
    out[f"atr_{cfg.atr_lookback}"] = true_range.rolling(cfg.atr_lookback).mean()
    out["log_return_1"] = np.log(close / close.shift(1))
    out["return_1"] = close.pct_change()
    for horizon in cfg.return_horizons:
        out[f"return_{horizon}"] = close.pct_change(horizon)
        out[f"log_return_{horizon}"] = np.log(close / close.shift(horizon))

    ann = _annualization_factor(out, cfg)
    for lookback in cfg.realized_vol_lookbacks:
        rv_col = f"realized_vol_{lookback}"
        out[rv_col] = out["log_return_1"].rolling(lookback).std() * np.sqrt(ann)
        out[f"vol_of_vol_{lookback}"] = out[rv_col].rolling(cfg.vol_of_vol_lookback).std()

    atr_col = f"atr_{cfg.atr_lookback}"
    out["atr_compression"] = out[atr_col] / out[atr_col].rolling(cfg.compression_lookback).median()
    range_pct = (high - low) / close
    out["range_compression"] = range_pct / range_pct.rolling(cfg.compression_lookback).median()

    prior_high = high.shift(1).rolling(cfg.breakout_lookback).max()
    prior_low = low.shift(1).rolling(cfg.breakout_lookback).min()
    atr_safe = out[atr_col].replace(0.0, np.nan)
    out["breakout_distance_high_atr"] = (close - prior_high) / atr_safe
    out["breakout_distance_low_atr"] = (close - prior_low) / atr_safe
    out["breakout_channel_width_atr"] = (prior_high - prior_low) / atr_safe

    sign = np.sign(out["log_return_1"])
    trend_direction = np.sign(close - close.shift(cfg.trend_lookback))
    directional_hits = (sign == trend_direction).astype(float)
    directional_hits[trend_direction == 0] = np.nan
    out["trend_persistence"] = directional_hits.rolling(cfg.trend_lookback).mean()
    out["trend_return"] = close.pct_change(cfg.trend_lookback)
    out["trend_return_atr"] = (close - close.shift(cfg.trend_lookback)) / atr_safe

    return out


def _annualization_factor(frame: pd.DataFrame, config: FeatureConfig) -> float:
    if "timeframe" in frame and frame["timeframe"].notna().any():
        timeframe = str(frame["timeframe"].dropna().iloc[-1]).lower()
        if timeframe in config.annualization_by_timeframe:
            return config.annualization_by_timeframe[timeframe]
    return 252.0
