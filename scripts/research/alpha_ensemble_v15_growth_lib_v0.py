#!/usr/bin/env python
from __future__ import annotations

"""
Growth Sleeve V1.5 – research library (daily-only v0).

Implements:
- Regime filter (ADX + Ichimoku cloud)
- Multi-speed trend signals (slow breakout + fast DEWMA cross)
- Trailing exits (Chandelier + PSAR) and gap-risk handling
- Vol parity sizing, cluster caps, portfolio vol target with max scalar
- Hard single-name cap and simple turnover/equity tracking

Inputs are in-memory pandas DataFrames; no file I/O or CLI runner here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow importing alpha_utils when executed as a script from repo root or this dir
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

try:
    from alpha_utils import calc_adx, calc_atr, calc_bollinger_width, calc_dewma, calc_ichimoku, calc_keltner
except ImportError:  # pragma: no cover - fallback if used as module elsewhere
    from scripts.research.alpha_utils import (
        calc_adx,
        calc_atr,
        calc_bollinger_width,
        calc_dewma,
        calc_ichimoku,
        calc_keltner,
    )


ANN_FACTOR = 365.0


@dataclass
class GrowthSleeveConfig:
    """
    Parameter block for the Growth Sleeve.
    """

    # Signal params
    adx_window: int = 14
    adx_threshold: float = 25.0
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou: int = 52
    ichimoku_disp: int = 26
    bb_window: int = 20
    bb_k: float = 2.0
    bb_breakout_mult: float = 1.5
    keltner_window: int = 20
    keltner_atr_mult: float = 2.0
    volume_lookback: int = 30
    dewma_fast: int = 10
    dewma_slow: int = 40

    # Trailing exit / stops
    atr_window: int = 14
    chandelier_mult: float = 3.0
    psar_af_start: float = 0.02
    psar_af_step: float = 0.02
    psar_af_max: float = 0.2
    gap_atr_mult: float = 1.0
    gap_slippage: float = 0.0025  # 25 bps

    # Sizing and guardrails
    target_risk_bps: float = 50.0  # per-name risk budget vs stop distance
    max_name_weight: float = 0.08
    corr_threshold: float = 0.7
    cluster_cap: float = 0.40
    cov_lookback: int = 30
    target_vol: float = 0.20  # annualized
    max_scalar: float = 1.5


def _to_panel(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure bars are in a MultiIndex panel indexed by [ts, symbol], sorted.
    Required columns: ts, symbol, open, high, low, close, volume.
    """
    required = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    if not required.issubset(bars_df.columns):
        missing = required - set(bars_df.columns)
        raise ValueError(f"bars_df missing required columns: {missing}")
    out = bars_df.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    out = out.sort_values(["ts", "symbol"])
    out = out.set_index(["ts", "symbol"])
    return out


def _calc_psar_symbol(df: pd.DataFrame, cfg: GrowthSleeveConfig) -> pd.Series:
    """
    Lightweight Parabolic SAR (long-only) per symbol.
    Returns a Series aligned to df index.
    """
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    if n == 0:
        return pd.Series([], index=df.index, name="psar")

    psar = np.zeros(n)
    trend_up = True
    af = cfg.psar_af_start
    ep = high[0]  # extreme point
    psar[0] = low[0]

    for i in range(1, n):
        prev_psar = psar[i - 1]
        if trend_up:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < psar[i]:
                trend_up = False
                psar[i] = ep
                ep = low[i]
                af = cfg.psar_af_start
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > psar[i]:
                trend_up = True
                psar[i] = ep
                ep = high[i]
                af = cfg.psar_af_start

        if trend_up and high[i] > ep:
            ep = high[i]
            af = min(cfg.psar_af_max, af + cfg.psar_af_step)
        elif (not trend_up) and low[i] < ep:
            ep = low[i]
            af = min(cfg.psar_af_max, af + cfg.psar_af_step)

    return pd.Series(psar, index=df.index, name="psar")


def _compute_features(panel: pd.DataFrame, cfg: GrowthSleeveConfig) -> pd.DataFrame:
    """
    Compute indicator features required for the sleeve.
    Returns a DataFrame aligned to panel index with all feature columns.
    """
    feats = pd.DataFrame(index=panel.index)

    # ADX / ATR use group-aware helpers (keep a flat index to avoid symbol ambiguity)
    df_for_ind = panel.reset_index()
    atr_ser = calc_atr(df_for_ind, n=cfg.atr_window)
    atr_ser.index = panel.index  # align to multiindex
    feats["atr"] = atr_ser
    adx_ser = calc_adx(df_for_ind, n=cfg.adx_window)
    adx_ser.index = panel.index
    feats["adx"] = adx_ser

    # Per-symbol computations (Ichimoku, Keltner, Bollinger width, DEWMA, PSAR)
    ich_parts = []
    kelt_parts = []
    psar_parts = []
    bb_parts = []
    dewma_fast_parts = []
    dewma_slow_parts = []
    for sym, g in panel.groupby(level="symbol"):
        g_reset = g.reset_index()
        ichi = calc_ichimoku(
            g_reset,
            tenkan=cfg.ichimoku_tenkan,
            kijun=cfg.ichimoku_kijun,
            senkou=cfg.ichimoku_senkou,
            displacement=cfg.ichimoku_disp,
        )
        ichi.index = g.index
        ich_parts.append(ichi)

        kelt = calc_keltner(
            g_reset,
            n=cfg.keltner_window,
            atr_mult=cfg.keltner_atr_mult,
            ma="ema",
        )
        kelt.index = g.index
        kelt_parts.append(kelt)

        psar = _calc_psar_symbol(g_reset, cfg)
        psar.index = g.index
        psar_parts.append(psar)

        bb_width = calc_bollinger_width(g_reset["close"], n=cfg.bb_window, k=cfg.bb_k)
        bb_width.index = g.index
        bb_width.name = "bb_width"
        bb_parts.append(bb_width)

        dewma_f = calc_dewma(g_reset["close"], n=cfg.dewma_fast)
        dewma_f.index = g.index
        dewma_f.name = "dewma_fast"
        dewma_fast_parts.append(dewma_f)

        dewma_s = calc_dewma(g_reset["close"], n=cfg.dewma_slow)
        dewma_s.index = g.index
        dewma_s.name = "dewma_slow"
        dewma_slow_parts.append(dewma_s)

    ich_df = pd.concat(ich_parts).sort_index()
    kelt_df = pd.concat(kelt_parts).sort_index()
    feats = feats.join(ich_df)
    feats = feats.join(kelt_df.rename(columns={"mid": "keltner_mid", "upper": "keltner_upper", "lower": "keltner_lower"}))

    feats["psar"] = pd.concat(psar_parts).sort_index()
    feats["bb_width"] = pd.concat(bb_parts).sort_index()
    feats["dewma_fast"] = pd.concat(dewma_fast_parts).sort_index()
    feats["dewma_slow"] = pd.concat(dewma_slow_parts).sort_index()

    # Rolling helpers
    vol = panel["volume"]
    feats["avg_vol_30"] = (
        vol.groupby(level="symbol").rolling(window=cfg.volume_lookback, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    )
    feats["bb_width_mean_20"] = (
        feats["bb_width"]
        .groupby(level="symbol")
        .rolling(window=cfg.bb_window, min_periods=cfg.bb_window)
        .mean()
        .reset_index(level=0, drop=True)
        .shift(1)
    )

    return feats


def _cluster_components(corr: pd.DataFrame, threshold: float) -> List[List[str]]:
    """
    Build connected components where corr > threshold.
    """
    if corr.empty:
        return []
    symbols = corr.index.tolist()
    visited = set()
    clusters: List[List[str]] = []

    for sym in symbols:
        if sym in visited:
            continue
        stack = [sym]
        component = []
        while stack:
            s = stack.pop()
            if s in visited:
                continue
            visited.add(s)
            component.append(s)
            neighbors = [n for n in symbols if n != s and corr.loc[s, n] > threshold]
            stack.extend([n for n in neighbors if n not in visited])
        clusters.append(component)
    return clusters


def _apply_cluster_cap(weights: pd.Series, returns_window: pd.DataFrame, cfg: GrowthSleeveConfig) -> pd.Series:
    """
    Scale weights within highly correlated clusters to respect cluster_cap.
    """
    active = weights[weights > 0]
    if active.empty or returns_window.empty:
        return weights

    corr = returns_window.corr().loc[active.index, active.index].fillna(0.0)
    clusters = _cluster_components(corr, cfg.corr_threshold)
    scaled = weights.copy()
    for cluster in clusters:
        cluster_w = scaled[cluster].abs().sum()
        if cluster_w > cfg.cluster_cap and cluster_w > 0:
            factor = cfg.cluster_cap / cluster_w
            scaled[cluster] = scaled[cluster] * factor
    return scaled


def _apply_vol_scalar(weights: pd.Series, returns_window: pd.DataFrame, cfg: GrowthSleeveConfig) -> Tuple[pd.Series, float]:
    """
    Scale weights to target portfolio volatility (capped).
    Returns scaled weights and scalar applied.
    """
    active = weights[weights > 0]
    if active.empty or returns_window.empty:
        return weights, 1.0

    cov = returns_window.cov().loc[active.index, active.index].fillna(0.0)
    if cov.empty:
        return weights, 1.0

    w_vec = active.values
    cov_mat = cov.values
    port_var = float(w_vec @ cov_mat @ w_vec)
    exp_vol_daily = np.sqrt(port_var) if port_var > 0 else 0.0
    exp_vol_ann = exp_vol_daily * np.sqrt(ANN_FACTOR)
    if exp_vol_ann == 0 or not np.isfinite(exp_vol_ann):
        return weights, 1.0

    scalar = min(cfg.max_scalar, cfg.target_vol / exp_vol_ann)
    scaled = weights * scalar
    return scaled, scalar


def run_growth_sleeve_backtest(
    bars_df: pd.DataFrame,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the Growth Sleeve backtest over an in-memory bars DataFrame.

    Args:
        bars_df: DataFrame with columns ts, symbol, open, high, low, close, volume.
        start_ts: optional start timestamp (inclusive).
        end_ts: optional end timestamp (inclusive).
        params: optional dict to override GrowthSleeveConfig fields.

    Returns:
        dict with:
            weights_df: ts, symbol, weight
            equity_df: ts, portfolio_equity, portfolio_ret, gross_long, cash_weight, turnover
            trades_df: optional trade log (entry/exit)
            debug_df: optional flags (regime, slow, fast, exits, exposure_mult, scalar)
    """
    cfg = GrowthSleeveConfig(**(params or {}))
    panel = _to_panel(bars_df)
    if start_ts:
        panel = panel.loc[panel.index.get_level_values("ts") >= pd.to_datetime(start_ts)]
    if end_ts:
        panel = panel.loc[panel.index.get_level_values("ts") <= pd.to_datetime(end_ts)]
    if panel.empty:
        raise ValueError("No data after applying date filters.")

    # Features and returns
    feats = _compute_features(panel, cfg)
    close = panel["close"]
    open_ = panel["open"]
    high = panel["high"]
    ret_close = close.groupby(level="symbol").pct_change()
    prev_close = close.groupby(level="symbol").shift(1)
    returns_df = ret_close.unstack("symbol").dropna(how="all")

    dates = sorted(panel.index.get_level_values("ts").unique())
    symbols = sorted(panel.index.get_level_values("symbol").unique())

    # State per symbol
    state: Dict[str, Dict[str, Any]] = {
        sym: {"active": False, "entry_px": np.nan, "highest": np.nan, "stop": np.nan, "cooloff": False}
        for sym in symbols
    }

    weights_prev = pd.Series(0.0, index=symbols)
    weights_records: List[Dict[str, Any]] = []
    equity_records: List[Dict[str, Any]] = []
    debug_records: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    portfolio_equity = 1.0
    equity_records.append(
        {
            "ts": dates[0],
            "portfolio_equity": portfolio_equity,
            "portfolio_ret": 0.0,
            "gross_long": 0.0,
            "cash_weight": 1.0,
            "turnover": 0.0,
        }
    )

    for i in range(1, len(dates)):
        ts = dates[i]
        prev_ts = dates[i - 1]
        day_prev = panel.xs(prev_ts, level="ts")
        day = panel.xs(ts, level="ts")
        feat_day = feats.xs(ts, level="ts")
        feat_prev = feats.xs(prev_ts, level="ts")

        # Gap exit and realized returns for holdings carried into today
        realized = pd.Series(0.0, index=symbols)
        gap_flags: Dict[str, bool] = {}
        for sym in symbols:
            if sym not in day_prev.index or sym not in day.index:
                gap_flags[sym] = False
                continue
            w_prev = weights_prev.get(sym, 0.0)
            if w_prev <= 0 or sym not in day_prev.index:
                continue
            prev_px = prev_close.loc[(prev_ts, sym)] if (prev_ts, sym) in prev_close.index else np.nan
            if not np.isfinite(prev_px):
                continue
            open_px = day.loc[sym, "open"]
            close_px = day.loc[sym, "close"]
            atr_today = feat_day.loc[sym, "atr"]
            stop_prev = state[sym]["stop"]

            gap_trigger = (
                state[sym]["active"]
                and np.isfinite(stop_prev)
                and np.isfinite(atr_today)
                and open_px < (stop_prev - cfg.gap_atr_mult * atr_today)
            )
            if gap_trigger:
                exit_px = open_px * (1.0 - cfg.gap_slippage)
                realized_ret = exit_px / prev_px - 1.0
                state[sym].update({"active": False, "highest": np.nan, "stop": np.nan, "cooloff": True})
                gap_flags[sym] = True
                trades.append(
                    {"ts": ts, "symbol": sym, "side": "EXIT_GAP", "px": exit_px, "ret": realized_ret, "weight_prior": w_prev}
                )
            else:
                gap_flags[sym] = False
                realized_ret = close_px / prev_px - 1.0
            realized[sym] = realized_ret

        port_ret = float((weights_prev * realized).sum())
        portfolio_equity *= 1.0 + port_ret

        gross_long = float(weights_prev.sum())
        cash_weight = max(0.0, 1.0 - gross_long)

        # Update trailing stops and apply end-of-day exits (Chandelier, PSAR)
        desired_weights = pd.Series(0.0, index=symbols)
        exit_flags_close: Dict[str, bool] = {}
        exit_flags_psar: Dict[str, bool] = {}
        exposure_map: Dict[str, float] = {}

        for sym in symbols:
            if sym not in day.index:
                exit_flags_close[sym] = False
                exit_flags_psar[sym] = False
                exposure_map[sym] = 0.0
                desired_weights[sym] = 0.0
                continue

            # skip re-entry on the same day as a gap exit
            if state[sym].get("cooloff", False):
                desired_weights[sym] = 0.0
                exit_flags_close[sym] = False
                exit_flags_psar[sym] = False
                exposure_map[sym] = 0.0
                # clear cooloff for next day
                state[sym]["cooloff"] = False
                continue

            atr_t = feat_day.loc[sym, "atr"]
            adx_t = feat_day.loc[sym, "adx"]
            in_cloud = feat_day.loc[sym, "in_cloud"]
            below_cloud = feat_day.loc[sym, "below_cloud"]
            k_upper = feat_day.loc[sym, "keltner_upper"]
            bb_width = feat_day.loc[sym, "bb_width"]
            bb_mean = feat_day.loc[sym, "bb_width_mean_20"]
            vol_avg = feat_day.loc[sym, "avg_vol_30"]
            volume_today = day.loc[sym, "volume"]
            close_px = day.loc[sym, "close"]

            in_cloud_flag = bool(in_cloud) if pd.notna(in_cloud) else False
            below_cloud_flag = bool(below_cloud) if pd.notna(below_cloud) else False
            regime_on = bool((pd.notna(adx_t) and adx_t >= cfg.adx_threshold) and (not in_cloud_flag) and (not below_cloud_flag))

            slow_bb = bool(bb_width > cfg.bb_breakout_mult * bb_mean) if np.isfinite(bb_width) and np.isfinite(bb_mean) else False
            breakout = bool(close_px > k_upper) if np.isfinite(close_px) and np.isfinite(k_upper) else False
            vol_confirm = bool(volume_today > 1.5 * vol_avg) if np.isfinite(volume_today) and np.isfinite(vol_avg) else False
            slow_on = slow_bb and breakout and vol_confirm

            dew_fast = feat_day.loc[sym, "dewma_fast"]
            dew_slow = feat_day.loc[sym, "dewma_slow"]
            fast_on = bool(dew_fast > dew_slow) if np.isfinite(dew_fast) and np.isfinite(dew_slow) else False

            exposure_mult = (0.8 * slow_on + 0.2 * fast_on) if regime_on else 0.0
            exposure_map[sym] = exposure_mult

            # Update trailing state for active names that survived the gap check
            exit_close = False
            exit_psar = False
            if state[sym]["active"] and not gap_flags.get(sym, False):
                highest_prev = state[sym]["highest"]
                high_today = day.loc[sym, "high"]
                highest_now = np.nanmax([highest_prev, high_today])
                stop_today = highest_now - cfg.chandelier_mult * atr_t if np.isfinite(atr_t) else np.nan

                psar_t = feat_day.loc[sym, "psar"]
                psar_flip = np.isfinite(psar_t) and (psar_t > close_px)
                chandelier_hit = np.isfinite(stop_today) and (close_px < stop_today)
                exit_close = chandelier_hit
                exit_psar = psar_flip

                state[sym].update({"highest": highest_now, "stop": stop_today})

                if exit_close or exit_psar:
                    state[sym].update({"active": False, "highest": np.nan, "stop": np.nan})
                    prev_px_val = prev_close.loc[(prev_ts, sym)] if (prev_ts, sym) in prev_close.index else np.nan
                    trades.append(
                        {
                            "ts": ts,
                            "symbol": sym,
                            "side": "EXIT_CLOSE",
                            "px": close_px,
                            "ret": close_px / prev_px_val - 1.0 if np.isfinite(prev_px_val) else np.nan,
                            "weight_prior": weights_prev.get(sym, 0.0),
                            "reason": "psar" if exit_psar else "chandelier",
                        }
                    )

            exit_flags_close[sym] = exit_close
            exit_flags_psar[sym] = exit_psar

            # Compute desired raw weight (risk-based) if still eligible
            if exposure_mult > 0 and not exit_close and not exit_psar and not gap_flags.get(sym, False):
                stop_dist = cfg.chandelier_mult * atr_t
                if not np.isfinite(stop_dist) or stop_dist <= 0 or not np.isfinite(close_px):
                    raw_w = 0.0
                else:
                    raw_w = (cfg.target_risk_bps / 10000.0) * (close_px / stop_dist)
                desired_weights[sym] = raw_w * exposure_mult
            else:
                desired_weights[sym] = 0.0

        # Guardrails: cluster cap and vol targeting
        ret_window = returns_df.loc[:ts].tail(cfg.cov_lookback)
        weights_clustered = _apply_cluster_cap(desired_weights, ret_window, cfg)
        weights_scaled, scalar_applied = _apply_vol_scalar(weights_clustered, ret_window, cfg)

        # Hard cap per name and optional renorm to keep gross <= 1
        weights_capped = weights_scaled.clip(upper=cfg.max_name_weight)
        gross_after_cap = weights_capped.sum()
        if gross_after_cap > 1.0:
            weights_capped *= 1.0 / gross_after_cap

        turnover = 0.5 * float(np.abs(weights_capped - weights_prev).sum())

        # Update state for entries
        for sym in symbols:
            w_new = weights_capped.get(sym, 0.0)
            was_active = state[sym]["active"]
            if w_new > 0 and (not was_active):
                state[sym].update(
                    {
                        "active": True,
                        "entry_px": day.loc[sym, "close"],
                        "highest": day.loc[sym, "high"],
                        "stop": day.loc[sym, "high"] - cfg.chandelier_mult * feat_day.loc[sym, "atr"],
                    }
                )
                trades.append(
                    {"ts": ts, "symbol": sym, "side": "ENTRY", "px": day.loc[sym, "close"], "weight": w_new, "reason": "signals"}
                )
            elif w_new == 0:
                state[sym].update({"active": False, "highest": np.nan, "stop": np.nan})

        weights_records.extend([{"ts": ts, "symbol": sym, "weight": weights_capped[sym]} for sym in symbols])
        equity_records.append(
            {
                "ts": ts,
                "portfolio_equity": portfolio_equity,
                "portfolio_ret": port_ret,
                "gross_long": gross_long,
                "cash_weight": cash_weight,
                "turnover": turnover,
            }
        )
        for sym in symbols:
            if sym not in feat_day.index:
                continue
            feat_s = feat_day.loc[sym]
            in_cloud_flag = bool(feat_s["in_cloud"]) if pd.notna(feat_s["in_cloud"]) else False
            below_cloud_flag = bool(feat_s["below_cloud"]) if pd.notna(feat_s["below_cloud"]) else False
            debug_records.append(
                {
                    "ts": ts,
                    "symbol": sym,
                    "regime_on": bool((pd.notna(feat_s["adx"]) and feat_s["adx"] >= cfg.adx_threshold) and (not in_cloud_flag) and (not below_cloud_flag)),
                    "slow_on": bool(
                        (feat_s["bb_width"] > cfg.bb_breakout_mult * feat_s["bb_width_mean_20"])
                        if np.isfinite(feat_s["bb_width"]) and np.isfinite(feat_s["bb_width_mean_20"])
                        else False
                    ),
                    "fast_on": bool(
                        (feat_s["dewma_fast"] > feat_s["dewma_slow"])
                        if np.isfinite(feat_s["dewma_fast"]) and np.isfinite(feat_s["dewma_slow"])
                        else False
                    ),
                    "exit_chandelier": exit_flags_close.get(sym, False),
                    "exit_psar": exit_flags_psar.get(sym, False),
                    "gap_exit": gap_flags.get(sym, False),
                    "exposure_mult": exposure_map.get(sym, 0.0),
                    "vol_scalar": scalar_applied,
                }
            )

        weights_prev = weights_capped

    weights_df = pd.DataFrame(weights_records).sort_values(["ts", "symbol"])
    equity_df = pd.DataFrame(equity_records).sort_values("ts")
    debug_df = pd.DataFrame(debug_records).sort_values(["ts", "symbol"])
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["ts", "symbol", "side", "px", "reason"])

    return {
        "weights_df": weights_df,
        "equity_df": equity_df,
        "trades_df": trades_df,
        "debug_df": debug_df,
    }


__all__ = ["GrowthSleeveConfig", "run_growth_sleeve_backtest"]
