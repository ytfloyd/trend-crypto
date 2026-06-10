"""Leakage-safe forward targets for K2 futures research."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetConfig:
    """Forward target settings expressed in completed bars."""

    horizons: tuple[str, ...] = ("4h", "1d", "3d", "5d")
    horizon_bars_by_timeframe: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "1h": {"4h": 4, "1d": 24, "3d": 72, "5d": 120},
            "4h": {"4h": 1, "1d": 6, "3d": 18, "5d": 30},
            "1d": {"1d": 1, "3d": 3, "5d": 5},
            "daily": {"1d": 1, "3d": 3, "5d": 5},
        }
    )
    atr_lookback: int = 14
    realized_vol_lookback: int = 20
    tail_atr_multiple: float = 2.0
    vol_expansion_threshold: float = 1.25


def build_target_frame(frame: pd.DataFrame, config: TargetConfig | None = None) -> pd.DataFrame:
    """Append forward research targets without using future data in features.

    Targets intentionally look forward; all normalizers use the current completed
    bar and prior bars only. The resulting frame is for model training and
    diagnostics, not live signal construction.
    """
    cfg = config or TargetConfig()
    if frame.empty:
        return frame.copy()

    out = frame.sort_values("ts").reset_index(drop=True).copy()
    close = pd.to_numeric(out["c"], errors="coerce")
    log_return = np.log(close / close.shift(1))
    atr = _atr(out, cfg.atr_lookback)
    current_vol = log_return.rolling(cfg.realized_vol_lookback).std()
    timeframe = _frame_timeframe(out)

    for horizon in cfg.horizons:
        bars = _horizon_bars(timeframe, horizon, cfg)
        if bars is None:
            continue
        future_close = close.shift(-bars)
        future_move = future_close - close
        future_log_move = np.log(future_close / close)
        future_vol = _forward_realized_vol(log_return, bars)
        normalized_move = future_move / atr.replace(0.0, np.nan)
        expansion_ratio = future_vol / current_vol.replace(0.0, np.nan)
        tail_event = normalized_move.abs() > cfg.tail_atr_multiple

        suffix = _suffix(horizon)
        out[f"target_future_return_{suffix}"] = future_close / close - 1.0
        out[f"target_future_log_return_{suffix}"] = future_log_move
        out[f"target_future_move_norm_{suffix}"] = normalized_move
        out[f"target_future_vol_expansion_{suffix}"] = expansion_ratio
        out[f"target_vol_expansion_event_{suffix}"] = (
            expansion_ratio > cfg.vol_expansion_threshold
        ).astype("float")
        out[f"target_tail_event_{suffix}"] = tail_event.astype("float")
        out[f"target_conditional_direction_{suffix}"] = np.where(
            tail_event,
            np.sign(future_move),
            0.0,
        )

    return out


def _atr(frame: pd.DataFrame, lookback: int) -> pd.Series:
    if f"atr_{lookback}" in frame:
        return pd.to_numeric(frame[f"atr_{lookback}"], errors="coerce")
    high = pd.to_numeric(frame["h"], errors="coerce")
    low = pd.to_numeric(frame["l"], errors="coerce")
    close = pd.to_numeric(frame["c"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(lookback).mean()


def _forward_realized_vol(log_return: pd.Series, bars: int) -> pd.Series:
    future_returns = [log_return.shift(-step) for step in range(1, bars + 1)]
    future = pd.concat(future_returns, axis=1)
    # Use an RMS realized-volatility estimator so a one-bar horizon is still
    # measurable as the absolute next return rather than all-NaN sample std.
    return np.sqrt((future.pow(2).sum(axis=1, min_count=bars)) / bars)


def _frame_timeframe(frame: pd.DataFrame) -> str:
    if "timeframe" not in frame or frame["timeframe"].dropna().empty:
        return "4h"
    return str(frame["timeframe"].dropna().iloc[-1]).lower()


def _horizon_bars(timeframe: str, horizon: str, config: TargetConfig) -> int | None:
    mapping = config.horizon_bars_by_timeframe.get(timeframe.lower(), {})
    bars = mapping.get(horizon.lower())
    if bars is None or bars <= 0:
        return None
    return int(bars)


def _suffix(horizon: str) -> str:
    return horizon.lower().replace(" ", "_")
