#!/usr/bin/env python3
"""
ETF-Native Alpha Strategies — BRAIN System Evaluation & PDF Report
===================================================================
Multi-asset alpha strategies designed specifically for an ETF universe:
time-series momentum, trend following, mean reversion, volatility,
carry proxies, and multi-signal combos inspired by AQR, Kolanovic,
and Moskowitz-Ooi-Pedersen.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

import alpha_brain
from alpha_brain import (
    AlphaDataSet,
    COST_BPS,
    evaluate_alpha,
    construct_portfolio,
    simulate,
    check_passing,
    rate_alpha,
    score_alpha,
)
from scripts.research.etf_data.universe import (
    get_expanded_universe, get_core_universe, get_full_universe,
    get_sector_map,
)

# ═══════════════════════════════════════════════════════════════════════════════
# ETF-NATIVE ALPHA CATALOGUE
# ═══════════════════════════════════════════════════════════════════════════════

ETF_ALPHAS: dict[str, tuple[str, str]] = {}

# ---------------------------------------------------------------------------
# Family 1: Time-Series & Cross-Sectional Momentum (Moskowitz et al.)
# ---------------------------------------------------------------------------
_MOMENTUM = {
    # Pure price momentum at various lookbacks
    "mom_1m":     ("rank(ts_returns(close, 21))",
                   "1-month cross-sectional momentum"),
    "mom_3m":     ("rank(ts_returns(close, 63))",
                   "3-month cross-sectional momentum"),
    "mom_6m":     ("rank(ts_returns(close, 126))",
                   "6-month cross-sectional momentum"),
    "mom_12m":    ("rank(ts_returns(close, 252))",
                   "12-month cross-sectional momentum"),
    # 12-1 momentum: skip the most recent month (reversal effect)
    "mom_12_1":   ("rank((close / (delay(close, 252) + 1e-8) - 1) - (close / (delay(close, 21) + 1e-8) - 1))",
                   "12-minus-1 month momentum (skip recent)"),
    # Volatility-adjusted momentum (Sharpe-momentum)
    "sharpe_mom_3m":  ("rank(ts_mean(returns, 63) / (ts_std(returns, 63) + 1e-8))",
                       "3-month Sharpe momentum"),
    "sharpe_mom_6m":  ("rank(ts_mean(returns, 126) / (ts_std(returns, 126) + 1e-8))",
                       "6-month Sharpe momentum"),
    "sharpe_mom_12m": ("rank(ts_mean(returns, 252) / (ts_std(returns, 252) + 1e-8))",
                       "12-month Sharpe momentum"),
    # Acceleration: momentum of momentum
    "mom_accel":  ("rank(ts_returns(close, 63) - delay(ts_returns(close, 63), 63))",
                   "Momentum acceleration (3m vs prior 3m)"),
    # Multi-horizon average
    "mom_combo":  ("rank(ts_returns(close, 21) + ts_returns(close, 63) + ts_returns(close, 126) + ts_returns(close, 252))",
                   "Multi-horizon momentum combo (1+3+6+12m)"),
}

# ---------------------------------------------------------------------------
# Family 2: Trend Following (CTA/managed futures style)
# ---------------------------------------------------------------------------
_TREND = {
    # SMA crossover signals
    "trend_sma50_200":  ("rank(ts_mean(close, 50) / (ts_mean(close, 200) + 1e-8) - 1)",
                         "50/200 SMA crossover trend"),
    "trend_sma20_100":  ("rank(ts_mean(close, 20) / (ts_mean(close, 100) + 1e-8) - 1)",
                         "20/100 SMA crossover trend"),
    "trend_sma10_50":   ("rank(ts_mean(close, 10) / (ts_mean(close, 50) + 1e-8) - 1)",
                         "10/50 SMA crossover trend"),
    # Price vs long-term moving average
    "trend_above_200":  ("rank(close / (ts_mean(close, 200) + 1e-8) - 1)",
                         "Distance above 200-day SMA"),
    "trend_above_100":  ("rank(close / (ts_mean(close, 100) + 1e-8) - 1)",
                         "Distance above 100-day SMA"),
    # Breakout: Donchian channel position
    "breakout_60d":     ("rank((close - ts_min(close, 60)) / (ts_max(close, 60) - ts_min(close, 60) + 1e-8))",
                         "60-day Donchian channel breakout"),
    "breakout_120d":    ("rank((close - ts_min(close, 120)) / (ts_max(close, 120) - ts_min(close, 120) + 1e-8))",
                         "120-day Donchian channel breakout"),
    # Linear decay weighted momentum (emphasise recent)
    "trend_decay_21":   ("rank(ts_decay_linear(returns, 21))",
                         "21-day decay-weighted returns"),
    "trend_decay_63":   ("rank(ts_decay_linear(returns, 63))",
                         "63-day decay-weighted returns"),
    # MACD-style
    "trend_macd":       ("rank(ts_mean(close, 12) - ts_mean(close, 26))",
                         "MACD-style 12/26 difference"),
}

# ---------------------------------------------------------------------------
# Family 3: Mean Reversion / Contrarian
# ---------------------------------------------------------------------------
_REVERSION = {
    # Short-term reversal (well-documented 1-week effect)
    "rev_1w":       ("-1 * rank(ts_returns(close, 5))",
                     "1-week reversal"),
    "rev_3d":       ("-1 * rank(ts_returns(close, 3))",
                     "3-day reversal"),
    # Bollinger band mean reversion
    "rev_boll_20":  ("-1 * rank((close - ts_mean(close, 20)) / (ts_std(close, 20) + 1e-8))",
                     "20-day Bollinger band reversion"),
    "rev_boll_60":  ("-1 * rank((close - ts_mean(close, 60)) / (ts_std(close, 60) + 1e-8))",
                     "60-day Bollinger band reversion"),
    # Z-score reversion
    "rev_zscore_20": ("-1 * rank(ts_zscore(close, 20))",
                      "20-day z-score reversion"),
    "rev_zscore_60": ("-1 * rank(ts_zscore(close, 60))",
                      "60-day z-score reversion"),
    # Distance from 52-week high (buy the dip)
    "rev_from_high": ("rank((close - ts_max(close, 252)) / (ts_max(close, 252) + 1e-8))",
                      "Buy the dip: distance from 52-week high"),
    # RSI-style: up days vs down days
    "rev_rsi_14":    ("-1 * rank(ts_sum(if_else(returns > 0, returns, 0), 14) / (ts_sum(abs(returns), 14) + 1e-8))",
                      "14-day RSI-style overbought/oversold reversal"),
}

# ---------------------------------------------------------------------------
# Family 4: Volatility / Low-Risk Anomaly
# ---------------------------------------------------------------------------
_VOLATILITY = {
    # Low volatility premium (Ang, Hodrick, Xing, Zhang 2006)
    "lowvol_20":    ("-1 * rank(ts_std(returns, 20))",
                     "20-day low volatility"),
    "lowvol_60":    ("-1 * rank(ts_std(returns, 60))",
                     "60-day low volatility"),
    # Inverse vol (risk-parity weighting proxy)
    "invvol_60":    ("rank(1.0 / (ts_std(returns, 60) + 1e-8))",
                     "Inverse 60-day volatility"),
    # Volatility-of-volatility (prefer stable regimes)
    "low_volvol":   ("-1 * rank(ts_std(ts_std(returns, 20), 60))",
                     "Low vol-of-vol (stability premium)"),
    # Downside deviation (prefer low drawdown risk)
    "low_downside": ("-1 * rank(ts_std(if_else(returns < 0, returns, 0), 60))",
                     "Low downside deviation"),
    # Max drawdown rank
    "low_maxdd":    ("rank(ts_min(close, 60) / (ts_max(close, 60) + 1e-8))",
                     "Low 60-day max drawdown"),
}

# ---------------------------------------------------------------------------
# Family 5: Volume / Liquidity
# ---------------------------------------------------------------------------
_VOLUME = {
    # Volume trend (rising volume = conviction)
    "vol_trend":      ("rank(ts_mean(volume, 5) / (ts_mean(volume, 60) + 1e-8))",
                       "Short-term volume surge vs 60d average"),
    # Money flow (OBV-style)
    "money_flow_20":  ("rank(ts_sum(if_else(close > delay(close, 1), volume, -volume), 20) / (ts_sum(volume, 20) + 1e-8))",
                       "20-day on-balance volume flow"),
    # Volume-weighted momentum
    "vwap_mom":       ("rank(ts_sum(returns * volume, 20) / (ts_sum(volume, 20) + 1e-8))",
                       "20-day volume-weighted returns"),
}

# ---------------------------------------------------------------------------
# Family 6: Cross-Asset / Sector-Relative
# ---------------------------------------------------------------------------
_CROSS_ASSET = {
    # Within asset-class relative strength
    "sector_rel_1m":  ("group_rank(ts_returns(close, 21), sector)",
                       "1-month return ranked within asset class"),
    "sector_rel_3m":  ("group_rank(ts_returns(close, 63), sector)",
                       "3-month return ranked within asset class"),
    # Sector-neutralised momentum
    "sector_neut_mom": ("group_neutralize(rank(ts_returns(close, 63)), sector)",
                        "3-month sector-neutralised momentum"),
    # Cross-sectional mean reversion within sector
    "sector_rev_1w":  ("-1 * group_rank(ts_returns(close, 5), sector)",
                       "1-week within-sector reversal"),
}

# ---------------------------------------------------------------------------
# Family 7: Combo / Multi-Factor
# ---------------------------------------------------------------------------
_COMBO = {
    # Momentum + trend filter: momentum only when above MA
    "combo_mom_trend": ("rank(ts_returns(close, 126)) * rank(close / (ts_mean(close, 200) + 1e-8))",
                        "6m momentum x trend filter"),
    # Momentum + low vol
    "combo_mom_lowvol": ("rank(ts_returns(close, 126)) * rank(1.0 / (ts_std(returns, 60) + 1e-8))",
                         "6m momentum x inverse vol"),
    # Trend + reversion hybrid
    "combo_trend_rev": ("rank(ts_mean(close, 50) / (ts_mean(close, 200) + 1e-8) - 1) + (-1 * rank(ts_zscore(close, 20)))",
                        "Trend + short-term reversion"),
    # Multi-signal: momentum, trend, low-vol
    "combo_triple":    ("rank(ts_returns(close, 126)) + rank(close / (ts_mean(close, 200) + 1e-8)) + (-1 * rank(ts_std(returns, 60)))",
                        "Triple factor: momentum + trend + low vol"),
    # Sharpe momentum with trend confirm
    "combo_sharpe_trend": ("rank(ts_mean(returns, 63) / (ts_std(returns, 63) + 1e-8)) * sign(ts_mean(close, 50) - ts_mean(close, 200))",
                           "3m Sharpe momentum, confirmed by trend"),
    # All-in combo: momentum + trend + reversal + lowvol
    "combo_kitchen_sink": ("rank(ts_returns(close, 252)) + rank(ts_mean(close, 50) / (ts_mean(close, 200) + 1e-8)) + (-1 * rank(ts_zscore(close, 20))) + (-1 * rank(ts_std(returns, 60)))",
                           "Kitchen sink: 12m mom + trend + reversion + lowvol"),
    # Risk-parity momentum (vol-weighted multi-horizon)
    "combo_rp_mom":   ("rank((ts_returns(close, 21) / (ts_std(returns, 21) + 1e-8)) + (ts_returns(close, 63) / (ts_std(returns, 63) + 1e-8)) + (ts_returns(close, 126) / (ts_std(returns, 126) + 1e-8)))",
                       "Risk-parity multi-horizon momentum"),
    # Sector-neutral combo
    "combo_sector_neutral": ("group_neutralize(rank(ts_returns(close, 126)) + (-1 * rank(ts_std(returns, 60))), sector)",
                             "Sector-neutral momentum + lowvol"),
}

# ---------------------------------------------------------------------------
# Family 8: Surprise / Anticipation — position before/during large moves
# ---------------------------------------------------------------------------
_SURPRISE = {
    # --- Vol Compression → Breakout (low vol precedes explosive moves) ---
    # Short-term vol collapsed vs long-term = coiled spring
    "surp_vol_compress":     ("-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8))",
                              "Vol compression: buy when 5d vol << 60d vol"),
    "surp_vol_compress_10":  ("-1 * rank(ts_std(returns, 10) / (ts_std(returns, 120) + 1e-8))",
                              "Vol compression: 10d vs 120d ratio"),
    # Vol compression + momentum direction (compressed AND trending = imminent breakout)
    "surp_coiled_spring":    ("rank(ts_returns(close, 63)) * (-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8)))",
                              "Coiled spring: momentum direction x vol compression"),
    # Range compression: Donchian channel width shrinking
    "surp_range_compress":   ("-1 * rank((ts_max(close, 20) - ts_min(close, 20)) / (ts_max(close, 60) - ts_min(close, 60) + 1e-8))",
                              "Range compression: 20d range << 60d range"),

    # --- Vol Surprise / Regime Break (realized vol spikes → continuation) ---
    # When current vol suddenly exceeds recent norm, the move continues
    "surp_vol_spike":        ("rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8)) * sign(ts_sum(returns, 5))",
                              "Vol spike continuation: high vol x recent direction"),
    # Vol breakout: 5-day vol exceeding 1-year vol by a lot + direction
    "surp_vol_breakout":     ("rank(ts_std(returns, 5) / (ts_std(returns, 252) + 1e-8)) * rank(ts_returns(close, 5))",
                              "Vol breakout: extreme short-term vol x 1-week return"),
    # Vol-of-vol spike: instability itself increasing
    "surp_volvol_spike":     ("rank(ts_std(ts_std(returns, 10), 20)) * sign(ts_sum(returns, 10))",
                              "Vol-of-vol spike x recent direction"),

    # --- Early Momentum / Ignition (catch the first days of a new trend) ---
    # Recent return is strong but longer-term was flat (new move starting)
    "surp_ignition_5_60":    ("rank(ts_returns(close, 5)) * (-1 * rank(abs(ts_returns(close, 60))))",
                              "Momentum ignition: strong 5d on flat 60d base"),
    "surp_ignition_3_20":    ("rank(ts_returns(close, 3)) * (-1 * rank(abs(ts_returns(close, 20))))",
                              "Momentum ignition: strong 3d on flat 20d base"),
    # Breakout from consolidation: price near range top after tight range
    "surp_breakout_setup":   ("rank((close - ts_min(close, 20)) / (ts_max(close, 20) - ts_min(close, 20) + 1e-8)) * (-1 * rank((ts_max(close, 20) - ts_min(close, 20)) / (close + 1e-8)))",
                              "Breakout setup: near range top + tight range"),

    # --- Volume Anticipation (smart money accumulation before moves) ---
    # Volume increasing while price is quiet = accumulation
    "surp_accumulation":     ("rank(ts_mean(volume, 5) / (ts_mean(volume, 60) + 1e-8)) * (-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8)))",
                              "Accumulation: rising volume + compressed vol"),
    # Volume surge confirming new breakout direction
    "surp_vol_confirm":      ("rank(ts_mean(volume, 3) / (ts_mean(volume, 20) + 1e-8)) * rank(ts_returns(close, 5))",
                              "Volume-confirmed breakout: vol surge x 5d return"),

    # --- Regime Change Detection (structural shift signals) ---
    # Price crossing a long-term level after being below it
    "surp_regime_cross_200": ("rank(if_else(close > ts_mean(close, 200), close / close, close * 0) - if_else(delay(close, 5) > delay(ts_mean(close, 200), 5), close / close, close * 0))",
                              "Regime cross: just crossed above 200d SMA"),
    # New 60-day high (fresh breakout territory) — ratio near 1.0 = at high
    "surp_new_high_60":      ("rank(close / (ts_max(close, 60) + 1e-8))",
                              "New 60-day high breakout"),
    # New 252-day high
    "surp_new_high_252":     ("rank(close / (ts_max(close, 252) + 1e-8))",
                              "New 52-week high breakout"),

    # --- Convexity / Tail Positioning ---
    # Skewness: prefer positively-skewed assets (convex payoff profile)
    "surp_pos_skew":         ("-1 * rank(ts_sum(if_else(returns < 0, returns * returns, 0), 60) / (ts_sum(returns * returns, 60) + 1e-8))",
                              "Positive skew: avoid left-tail-heavy assets"),
    # Recent large move continuation (gap-and-go)
    "surp_gap_go":           ("rank(abs(ts_returns(close, 1))) * sign(ts_returns(close, 1)) * rank(ts_returns(close, 5))",
                              "Gap-and-go: large daily move continues if aligned with 5d trend"),

    # --- Combos ---
    # Vol compression + trend confirmation
    "surp_compress_trend":   ("(-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8))) * rank(close / (ts_mean(close, 50) + 1e-8))",
                              "Vol compression + trend: compressed vol in uptrend"),
    # Full surprise combo: compression + momentum + volume
    "surp_full_combo":       ("(-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8))) + rank(ts_returns(close, 63)) + rank(ts_mean(volume, 5) / (ts_mean(volume, 60) + 1e-8))",
                              "Full surprise: vol compress + momentum + volume surge"),
    # Anticipation score: new high + low vol + volume building
    "surp_anticipation":     ("rank(close / (ts_max(close, 60) + 1e-8)) + (-1 * rank(ts_std(returns, 10) / (ts_std(returns, 60) + 1e-8))) + rank(ts_mean(volume, 5) / (ts_mean(volume, 60) + 1e-8))",
                              "Anticipation: near 60d high + compressed vol + volume building"),
}

# Merge all families
for _family in [_MOMENTUM, _TREND, _REVERSION, _VOLATILITY, _VOLUME, _CROSS_ASSET, _COMBO, _SURPRISE]:
    ETF_ALPHAS.update(_family)

FAMILY_LABELS = {}
for name in _MOMENTUM:   FAMILY_LABELS[name] = "Momentum"
for name in _TREND:       FAMILY_LABELS[name] = "Trend"
for name in _REVERSION:   FAMILY_LABELS[name] = "Reversion"
for name in _VOLATILITY:  FAMILY_LABELS[name] = "Volatility"
for name in _VOLUME:      FAMILY_LABELS[name] = "Volume"
for name in _CROSS_ASSET: FAMILY_LABELS[name] = "CrossAsset"
for name in _COMBO:       FAMILY_LABELS[name] = "Combo"
for name in _SURPRISE:    FAMILY_LABELS[name] = "Surprise"


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_alphas(
    data: AlphaDataSet,
    mode: str = "long_only",
    max_weight: float = 0.05,
    decay: int = 0,
    neutralize: str = "none",
    delay: int = 1,
    annualization: int = 252,
) -> tuple[pd.DataFrame, dict, dict]:
    rows = []
    pnl_dict: dict[str, pd.Series] = {}
    nav_dict: dict[str, pd.Series] = {}

    total = len(ETF_ALPHAS)
    for i, (name, (expr, desc)) in enumerate(sorted(ETF_ALPHAS.items()), 1):
        t0 = time.time()
        try:
            alpha_signal = evaluate_alpha(expr, data)
            weights = construct_portfolio(
                alpha_signal, mode=mode, max_weight=max_weight,
                decay=decay, neutralize=neutralize, sector_df=data.sector,
            )
            if delay > 1:
                weights = weights.shift(delay - 1)
            metrics = simulate(weights, data.returns, annualization=annualization)
            elapsed = time.time() - t0

            rating = rate_alpha(metrics)
            sc = score_alpha(metrics)
            checks = check_passing(metrics, delay=delay)
            passes = all(v[0] for v in checks.values())

            pnl_dict[name] = metrics["daily_pnl"].dropna()
            nav_dict[name] = metrics["cum_nav"].dropna()

            rows.append({
                "name": name, "family": FAMILY_LABELS.get(name, "Other"),
                "description": desc, "expression": expr,
                "sharpe": metrics["sharpe"], "sortino": metrics["sortino"],
                "turnover": metrics["turnover"], "fitness": metrics["fitness"],
                "brain_fitness": metrics["brain_fitness"],
                "returns": metrics["returns"], "total_return": metrics["total_return"],
                "drawdown": metrics["drawdown"], "margin": metrics["margin"],
                "win_rate": metrics["win_rate"], "n_bars": metrics["n_bars"],
                "rating": rating, "score": sc, "passes": passes,
                "elapsed": elapsed, "error": None,
            })
            print(f"  [{i:3d}/{total}] {name:<24s} Sharpe={metrics['sharpe']:7.2f}  "
                  f"Return={metrics['returns']*100:6.1f}%  DD={metrics['drawdown']*100:6.1f}%  "
                  f"{rating:<12s} [{FAMILY_LABELS.get(name, '?')}]")

        except Exception as e:
            elapsed = time.time() - t0
            rows.append({
                "name": name, "family": FAMILY_LABELS.get(name, "Other"),
                "description": desc, "expression": expr,
                "sharpe": 0, "sortino": 0, "turnover": 0, "fitness": 0,
                "brain_fitness": 0, "returns": 0, "total_return": 0,
                "drawdown": 0, "margin": 0, "win_rate": 0, "n_bars": 0,
                "rating": "Error", "score": 0, "passes": False,
                "elapsed": elapsed, "error": str(e),
            })
            print(f"  [{i:3d}/{total}] {name:<24s} ERROR: {str(e)[:80]}")

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df, pnl_dict, nav_dict


# ═══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ═══════════════════════════════════════════════════════════════════════════════

DARK_BG = "#1a1a2e"
ACCENT = "#00d2ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd93d"
ACCENT4 = "#a78bfa"
ACCENT5 = "#4ade80"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#333355"
TABLE_ROW_ALT = "#222244"

FAMILY_COLORS = {
    "Momentum": "#00d2ff",
    "Trend": "#4ade80",
    "Reversion": "#ff6b6b",
    "Volatility": "#a78bfa",
    "Volume": "#ffd93d",
    "CrossAsset": "#f97316",
    "Combo": "#ec4899",
    "Surprise": "#06b6d4",
    "Other": "#888888",
}

def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": DARK_BG,
        "axes.edgecolor": GRID_COLOR, "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR, "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR, "grid.color": GRID_COLOR,
        "grid.alpha": 0.3, "font.family": "sans-serif", "font.size": 9,
    })


def generate_report(
    results_df: pd.DataFrame,
    pnl_dict: dict[str, pd.Series],
    nav_dict: dict[str, pd.Series],
    data: AlphaDataSet,
    mode: str,
    output_path: Path,
    annualization: int = 252,
):
    _setup_style()
    valid = results_df[results_df["error"].isna()].copy()
    n_total = len(results_df)
    n_valid = len(valid)
    n_errors = n_total - n_valid
    n_pass = int(valid["passes"].sum())

    with PdfPages(str(output_path)) as pdf:
        # ──────────────────────────────────────────────────────────────────
        # PAGE 1: Title & Executive Summary
        # ──────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.88, "ETF-Native Alpha Strategies", fontsize=28, fontweight="bold",
                 ha="center", color=ACCENT)
        fig.text(0.5, 0.81, "Multi-Asset BRAIN System Evaluation", fontsize=16,
                 ha="center", color=TEXT_COLOR)
        fig.text(0.5, 0.75, f"{data.date_start.date()} to {data.date_end.date()}  |  "
                 f"{len(data.symbols)} ETFs  |  Mode: {mode}  |  Cost: {COST_BPS:.0f} bps",
                 fontsize=11, ha="center", color="#888888")

        summary_lines = [
            f"Alphas Evaluated:   {n_total}  ({n_valid} OK, {n_errors} errors)",
            f"Passing Criteria:   {n_pass}  ({n_pass/max(n_valid,1)*100:.0f}%)",
            "",
            f"Best Sharpe:        {valid['sharpe'].max():.2f}  ({valid.iloc[0]['name']})",
            f"Median Sharpe:      {valid['sharpe'].median():.2f}",
            f"Mean Ann. Return:   {valid['returns'].mean()*100:.1f}%",
            f"Mean Max Drawdown:  {valid['drawdown'].mean()*100:.1f}%",
            f"Mean Turnover:      {valid['turnover'].mean()*100:.1f}%",
            "",
            "Performance by Strategy Family:",
        ]
        for fam in ["Momentum", "Trend", "Reversion", "Volatility", "Volume", "CrossAsset", "Combo", "Surprise"]:
            fam_df = valid[valid["family"] == fam]
            if len(fam_df) > 0:
                summary_lines.append(
                    f"  {fam:<12s}  n={len(fam_df):2d}  "
                    f"Sharpe: {fam_df['sharpe'].median():+.2f} (med)  "
                    f"Return: {fam_df['returns'].mean()*100:+.1f}%"
                )

        y = 0.63
        for line in summary_lines:
            fig.text(0.12, y, line, fontsize=10.5, fontfamily="monospace", color=TEXT_COLOR)
            y -= 0.032
        pdf.savefig(fig)
        plt.close(fig)

        # ──────────────────────────────────────────────────────────────────
        # PAGE 2: Sharpe by Family (grouped bar) + Distributions
        # ──────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Strategy Quality Overview", fontsize=14, fontweight="bold", color=ACCENT)

        # Sharpe distribution by family
        ax = axes[0, 0]
        families = ["Momentum", "Trend", "Reversion", "Volatility", "Volume", "CrossAsset", "Combo", "Surprise"]
        box_data = [valid[valid["family"]==f]["sharpe"].values for f in families if len(valid[valid["family"]==f])>0]
        box_labels = [f for f in families if len(valid[valid["family"]==f])>0]
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
        for patch, label in zip(bp["boxes"], box_labels):
            patch.set_facecolor(FAMILY_COLORS.get(label, "#888888"))
            patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color(TEXT_COLOR)
        ax.axhline(0, color=ACCENT2, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(1.25, color=ACCENT5, linewidth=0.8, linestyle="--", alpha=0.5, label="Pass=1.25")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sharpe by Strategy Family")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.2)

        # Sharpe histogram
        ax = axes[0, 1]
        sharpes = valid["sharpe"].dropna()
        ax.hist(sharpes, bins=25, color=ACCENT, alpha=0.7, edgecolor=DARK_BG)
        ax.axvline(sharpes.median(), color=ACCENT3, linestyle="--", linewidth=1.5,
                   label=f"Median={sharpes.median():.2f}")
        ax.axvline(0, color=ACCENT2, linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Count")
        ax.set_title("Sharpe Distribution (all)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Return vs Drawdown scatter
        ax = axes[1, 0]
        for fam in families:
            fam_df = valid[valid["family"] == fam]
            if len(fam_df) > 0:
                ax.scatter(fam_df["drawdown"]*100, fam_df["returns"]*100,
                          color=FAMILY_COLORS.get(fam, "#888"), alpha=0.8, s=40,
                          label=fam, edgecolors="none")
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Ann. Return (%)")
        ax.set_title("Risk-Return by Family")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)

        # Sharpe vs Turnover scatter
        ax = axes[1, 1]
        for fam in families:
            fam_df = valid[valid["family"] == fam]
            if len(fam_df) > 0:
                ax.scatter(fam_df["turnover"]*100, fam_df["sharpe"],
                          color=FAMILY_COLORS.get(fam, "#888"), alpha=0.8, s=40,
                          label=fam, edgecolors="none")
        ax.set_xlabel("Avg Daily Turnover (%)")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sharpe vs Turnover")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ──────────────────────────────────────────────────────────────────
        # PAGE 3: Top 15 Equity Curves
        # ──────────────────────────────────────────────────────────────────
        top_n = valid.head(15)
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5))
        fig.suptitle("Top 15 Alpha Equity Curves", fontsize=14, fontweight="bold", color=ACCENT)

        # Top 5
        ax = axes[0]
        for _, row in top_n.head(5).iterrows():
            name = row["name"]
            if name in nav_dict:
                nav = nav_dict[name]
                color = FAMILY_COLORS.get(FAMILY_LABELS.get(name, "Other"), "#888")
                ax.plot(nav.index, nav.values, label=f"{name} ({row['sharpe']:.2f})",
                       linewidth=1.5, color=color)
        ax.set_ylabel("Cumulative NAV")
        ax.set_title("Rank 1-5")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.2)

        # Rank 6-10
        ax = axes[1]
        for _, row in top_n.iloc[5:10].iterrows():
            name = row["name"]
            if name in nav_dict:
                nav = nav_dict[name]
                color = FAMILY_COLORS.get(FAMILY_LABELS.get(name, "Other"), "#888")
                ax.plot(nav.index, nav.values, label=f"{name} ({row['sharpe']:.2f})",
                       linewidth=1.5, color=color)
        ax.set_ylabel("Cumulative NAV")
        ax.set_title("Rank 6-10")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.2)

        # Rank 11-15
        ax = axes[2]
        for _, row in top_n.iloc[10:15].iterrows():
            name = row["name"]
            if name in nav_dict:
                nav = nav_dict[name]
                color = FAMILY_COLORS.get(FAMILY_LABELS.get(name, "Other"), "#888")
                ax.plot(nav.index, nav.values, label=f"{name} ({row['sharpe']:.2f})",
                       linewidth=1.5, color=color)
        ax.set_ylabel("Cumulative NAV")
        ax.set_title("Rank 11-15")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.2)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ──────────────────────────────────────────────────────────────────
        # PAGE 4: Rolling Sharpe for Top 10
        # ──────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("Rolling 252-day Sharpe — Top 10 Strategies", fontsize=14,
                     fontweight="bold", color=ACCENT)

        window = 252
        for idx, ax in enumerate(axes):
            start_i = idx * 5
            for _, row in top_n.iloc[start_i:start_i+5].iterrows():
                name = row["name"]
                if name in pnl_dict:
                    pnl = pnl_dict[name]
                    roll_mean = pnl.rolling(window, min_periods=60).mean()
                    roll_std = pnl.rolling(window, min_periods=60).std()
                    rs = (roll_mean / roll_std) * np.sqrt(annualization)
                    rs = rs.dropna()
                    color = FAMILY_COLORS.get(FAMILY_LABELS.get(name, "Other"), "#888")
                    ax.plot(rs.index, rs.values, label=name, linewidth=1.2, color=color)
            ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)
            ax.axhline(1.25, color=ACCENT5, linewidth=0.8, linestyle="--", alpha=0.4)
            ax.set_ylabel("Rolling Sharpe")
            ax.legend(fontsize=7, ncol=3, loc="upper left")
            ax.grid(True, alpha=0.2)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ──────────────────────────────────────────────────────────────────
        # PAGE 5: Drawdown profiles for Top 10
        # ──────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("Drawdown Profiles — Top 10 Strategies", fontsize=14,
                     fontweight="bold", color=ACCENT)

        for idx, ax in enumerate(axes):
            start_i = idx * 5
            for _, row in top_n.iloc[start_i:start_i+5].iterrows():
                name = row["name"]
                if name in nav_dict:
                    nav = nav_dict[name]
                    peak = nav.cummax()
                    dd = (nav - peak) / peak
                    color = FAMILY_COLORS.get(FAMILY_LABELS.get(name, "Other"), "#888")
                    ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color)
                    ax.plot(dd.index, dd.values, linewidth=0.8, color=color, label=name)
            ax.set_ylabel("Drawdown")
            ax.legend(fontsize=7, ncol=3, loc="lower left")
            ax.grid(True, alpha=0.2)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ──────────────────────────────────────────────────────────────────
        # PAGE 6: Correlation Heatmap of Top 20 PnL Streams
        # ──────────────────────────────────────────────────────────────────
        top20_names = valid.head(20)["name"].tolist()
        top20_pnl = pd.DataFrame({n: pnl_dict[n] for n in top20_names if n in pnl_dict})
        if top20_pnl.shape[1] >= 3:
            corr = top20_pnl.corr()
            fig, ax = plt.subplots(figsize=(11, 8.5))
            im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns, fontsize=7)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if abs(corr.iloc[i,j]) < 0.5 else "white")
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("PnL Correlation — Top 20 Strategies", fontsize=14,
                        fontweight="bold", color=ACCENT)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ──────────────────────────────────────────────────────────────────
        # PAGE 7: Equal-Weight Ensemble of Top N
        # ──────────────────────────────────────────────────────────────────
        for ensemble_n in [5, 10, 20]:
            ens_names = valid.head(ensemble_n)["name"].tolist()
            ens_pnl_df = pd.DataFrame({n: pnl_dict[n] for n in ens_names if n in pnl_dict})
            if ens_pnl_df.shape[1] < 2:
                continue

            fig, axes = plt.subplots(3, 1, figsize=(11, 8.5))
            fig.suptitle(f"Equal-Weight Ensemble — Top {ensemble_n} Strategies",
                        fontsize=14, fontweight="bold", color=ACCENT)

            ens_pnl = ens_pnl_df.mean(axis=1).dropna()
            ens_nav = (1 + ens_pnl).cumprod()
            std = float(ens_pnl.std())
            ens_sharpe = float(ens_pnl.mean() / std * np.sqrt(annualization)) if std > 1e-12 else 0.0
            total_ret = float(ens_nav.iloc[-1] - 1.0)
            n_yrs = len(ens_pnl) / annualization
            ann_ret = float((1 + total_ret)**(1/n_yrs) - 1) if n_yrs > 0 else 0.0
            peak = ens_nav.cummax()
            dd = (ens_nav - peak) / peak
            max_dd = float(dd.min())

            ax = axes[0]
            ax.plot(ens_nav.index, ens_nav.values, color=ACCENT, linewidth=2)
            ax.set_ylabel("NAV")
            ax.set_title(f"Equity Curve  |  Sharpe={ens_sharpe:.2f}  Return={ann_ret*100:.1f}%  MaxDD={max_dd*100:.1f}%",
                        fontsize=11)
            ax.grid(True, alpha=0.2)

            ax = axes[1]
            roll_mean = ens_pnl.rolling(252, min_periods=60).mean()
            roll_std = ens_pnl.rolling(252, min_periods=60).std()
            rs = (roll_mean / roll_std) * np.sqrt(annualization)
            rs = rs.dropna()
            ax.plot(rs.index, rs.values, color=ACCENT4, linewidth=1.2)
            ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)
            ax.set_ylabel("Rolling Sharpe (252d)")
            ax.grid(True, alpha=0.2)

            ax = axes[2]
            ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color=ACCENT2)
            ax.plot(dd.index, dd.values, color=ACCENT2, linewidth=0.8)
            ax.set_ylabel("Drawdown")
            ax.grid(True, alpha=0.2)

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)

        # ──────────────────────────────────────────────────────────────────
        # PAGE 8+: Detailed Results Tables
        # ──────────────────────────────────────────────────────────────────
        rows_per_page = 25
        for page_start in range(0, len(valid), rows_per_page):
            chunk = valid.iloc[page_start:page_start+rows_per_page]
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")

            col_labels = ["#", "Alpha", "Family", "Sharpe", "Return", "MaxDD",
                          "Turn", "Win%", "Score", "Pass"]
            cell_data = []
            for rank_i, (_, row) in enumerate(chunk.iterrows(), page_start + 1):
                cell_data.append([
                    str(rank_i),
                    row["name"],
                    row["family"],
                    f"{row['sharpe']:.2f}",
                    f"{row['returns']*100:.1f}%",
                    f"{row['drawdown']*100:.1f}%",
                    f"{row['turnover']*100:.1f}%",
                    f"{row['win_rate']*100:.0f}%",
                    str(row["score"]),
                    "Y" if row["passes"] else "",
                ])

            table = ax.table(
                cellText=cell_data, colLabels=col_labels,
                loc="center", cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1, 1.15)

            for (row_i, col_i), cell in table.get_celld().items():
                cell.set_edgecolor(GRID_COLOR)
                if row_i == 0:
                    cell.set_facecolor("#2a2a4a")
                    cell.set_text_props(color=ACCENT, fontweight="bold")
                else:
                    cell.set_facecolor(TABLE_ROW_ALT if row_i % 2 == 0 else DARK_BG)
                    cell.set_text_props(color=TEXT_COLOR)
                    if col_i == len(col_labels) - 1 and cell.get_text().get_text() == "Y":
                        cell.set_text_props(color="#4ade80", fontweight="bold")

            ax.set_title(f"Detailed Results — Rank {page_start+1} to {page_start+len(chunk)}",
                         fontsize=12, fontweight="bold", color=ACCENT, pad=20)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n  PDF report saved: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_asset_class(args):
    """Return (db_path, symbols, sector_map, asset_label) based on --asset-class."""
    if args.asset_class == "stocks":
        import json
        import duckdb
        sp500_json = PROJECT_ROOT / "scripts" / "research" / "sp500_tickers.json"
        with open(sp500_json) as f:
            sp_data = json.load(f)
        all_tickers = sp_data["tickers"]
        sector_map = sp_data["sectors"]
        db = args.db or str(PROJECT_ROOT / ".." / "data" / "stocks_market.duckdb")
        con = duckdb.connect(db, read_only=True)
        good = con.execute(
            "SELECT symbol FROM bars_1d WHERE ts < '2015-01-01' GROUP BY symbol HAVING COUNT(*) >= 200"
        ).fetchdf()["symbol"].tolist()
        con.close()
        symbols = sorted(set(all_tickers) & set(good))
        return db, symbols, sector_map, "S&P 500"
    else:
        universe_fn = {"expanded": get_expanded_universe, "core": get_core_universe, "full": get_full_universe}
        symbols = universe_fn.get(args.universe, get_expanded_universe)()
        db = args.db or str(PROJECT_ROOT / ".." / "data" / "etf_market.duckdb")
        return db, symbols, get_sector_map(symbols), "ETF"


def main():
    parser = argparse.ArgumentParser(
        description="Run ETF-native alpha strategies through BRAIN system"
    )
    parser.add_argument("--db", default=None, help="DuckDB path")
    parser.add_argument("--asset-class", choices=["etf", "stocks"], default="etf",
                        help="Asset class: etf (default) or stocks")
    parser.add_argument("--universe", choices=["expanded", "core", "full"],
                        default="expanded")
    parser.add_argument("--mode", choices=["long_only", "long_short"], default="long_only")
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--decay", type=int, default=0)
    parser.add_argument("--neutralize", choices=["none", "sector", "market"], default="none")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=None, help="Output PDF path")
    args = parser.parse_args()

    ann = 252
    alpha_brain.ANNUALIZATION = ann

    db_path, symbols, sector_map, asset_label = _resolve_asset_class(args)

    print(f"\n{'='*70}")
    print(f"  ETF-Native Alpha Strategies — BRAIN Evaluation ({asset_label})")
    print(f"  Mode: {args.mode}  Delay: {args.delay}  Cost: {COST_BPS:.0f} bps")
    print(f"  Universe: {len(symbols)} tickers")
    print(f"{'='*70}\n")

    t0 = time.time()
    data = AlphaDataSet(
        db_path, symbols, start=args.start, end=args.end,
        table="bars_1d", annualization=ann, sector_map=sector_map,
    )

    print(f"\nEvaluating {len(ETF_ALPHAS)} ETF-native alphas...\n")
    results_df, pnl_dict, nav_dict = run_all_alphas(
        data, mode=args.mode, max_weight=args.max_weight,
        decay=args.decay, neutralize=args.neutralize, delay=args.delay,
        annualization=ann,
    )

    out_dir = PROJECT_ROOT / "artifacts" / "research" / "alpha_brain"
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_suffix = f"_{asset_label.lower().replace(' ', '_').replace('&', '')}" if asset_label != "ETF" else ""
    mode_suffix = f"_{args.mode}" if args.mode != "long_only" else ""
    csv_path = out_dir / f"etf_native_results{asset_suffix}{mode_suffix}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    if args.output:
        pdf_path = Path(args.output)
    else:
        pdf_path = out_dir / f"etf_native_report{asset_suffix}{mode_suffix}.pdf"
    generate_report(results_df, pnl_dict, nav_dict, data, args.mode, pdf_path,
                    annualization=ann)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
