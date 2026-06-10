#!/usr/bin/env python3
"""
Stock Surprise / Anticipation Alpha Strategies — BRAIN Evaluation
=================================================================
Stock-specific surprise alphas exploiting microstructure, overnight
gaps, close location value, volume intelligence, volatility regime
shifts, and momentum ignition. Designed for the S&P 500 cross-section.

Theory: surprise alphas aim to detect *information arrival* before
the market fully prices it — vol compression breakouts, abnormal
volume accumulation, gap continuation, and regime change signals.
"""
from __future__ import annotations

import argparse
import json
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

# ═══════════════════════════════════════════════════════════════════════════════
# STOCK SURPRISE ALPHA CATALOGUE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each entry: name -> (expression, description, category)

SURPRISE_ALPHAS: dict[str, tuple[str, str, str]] = {}


# ---------------------------------------------------------------------------
# Cat 1: Overnight Gap & Open-Close Dynamics
# ---------------------------------------------------------------------------
# Stocks have meaningful overnight sessions; gap direction/size carries info.
_GAP = {
    # Overnight return: (open - prev close) / prev close
    "gap_overnight_mom":
        ("rank((open - delay(close, 1)) / (delay(close, 1) + 1e-8))",
         "Overnight gap momentum: buy stocks gapping up",
         "Gap"),
    "gap_overnight_rev":
        ("-1 * rank((open - delay(close, 1)) / (delay(close, 1) + 1e-8))",
         "Overnight gap reversal: fade the gap",
         "Gap"),
    # Gap continuation: gap up + intraday follow-through
    "gap_continuation":
        ("rank((open - delay(close, 1)) / (delay(close, 1) + 1e-8)) * rank((close - open) / (open + 1e-8))",
         "Gap continuation: gap up that closes higher than open",
         "Gap"),
    # Cumulative gap signal over 5 days (persistent gapping = institutional flow)
    "gap_5d_trend":
        ("rank(ts_sum((open - delay(close, 1)) / (delay(close, 1) + 1e-8), 5))",
         "5-day cumulative gap direction (persistent institutional flow)",
         "Gap"),
    # Intraday vs overnight decomposition: which session drives returns?
    "intraday_return":
        ("rank(ts_mean((close - open) / (open + 1e-8), 20))",
         "20d avg intraday return (open-to-close)",
         "Gap"),
    "overnight_return":
        ("rank(ts_mean((open - delay(close, 1)) / (delay(close, 1) + 1e-8), 20))",
         "20d avg overnight return (close-to-open)",
         "Gap"),
    # Gap size absolute: large gaps = information events
    "gap_size_rank":
        ("rank(ts_mean(abs(open - delay(close, 1)) / (delay(close, 1) + 1e-8), 10))",
         "Avg 10d absolute gap size (information intensity)",
         "Gap"),
}

# ---------------------------------------------------------------------------
# Cat 2: Close Location Value (CLV) & Price Structure
# ---------------------------------------------------------------------------
# Where in the day's range did price close? Closing near highs = buying pressure.
_CLV = {
    "clv_raw":
        ("rank(ts_mean((close - low) / (high - low + 1e-8), 10))",
         "10d avg close location value (CLV near 1 = closing at highs)",
         "CLV"),
    "clv_trend":
        ("rank(ts_mean((close - low) / (high - low + 1e-8), 5) - ts_mean((close - low) / (high - low + 1e-8), 20))",
         "CLV trend: improving CLV = building buying pressure",
         "CLV"),
    # Upper shadow ratio: tall upper shadow = selling pressure at highs
    "upper_shadow":
        ("-1 * rank(ts_mean((high - max(open, close)) / (high - low + 1e-8), 10))",
         "Short upper shadow: no selling pressure at highs",
         "CLV"),
    # Body ratio: large body = conviction, small body = indecision
    "body_ratio":
        ("rank(ts_mean(abs(close - open) / (high - low + 1e-8), 10)) * sign(ts_sum(close - open, 10))",
         "Body ratio x direction: large bullish candles = conviction",
         "CLV"),
    # Accumulation/Distribution line proxy
    "ad_line":
        ("rank(ts_sum(((close - low) - (high - close)) / (high - low + 1e-8) * volume, 20))",
         "20d accumulation/distribution (CLV-weighted volume)",
         "CLV"),
    # CLV improving + volume expanding = smart money buying
    "clv_vol_expand":
        ("rank(ts_mean((close - low) / (high - low + 1e-8), 5)) * rank(ts_mean(volume, 5) / (ts_mean(volume, 20) + 1e-8))",
         "CLV improving + volume expanding",
         "CLV"),
}

# ---------------------------------------------------------------------------
# Cat 3: Volatility Regime & Compression
# ---------------------------------------------------------------------------
# Exploit vol compression → breakout and vol regime shifts.
_VOL_REGIME = {
    # Parkinson volatility (range-based, more efficient estimator)
    "parkinson_low":
        ("-1 * rank(ts_mean(power(log(high / (low + 1e-8)), 2), 20) / (4 * 0.693))",
         "Low Parkinson (range-based) volatility",
         "VolRegime"),
    # Garman-Klass / simple vol ratio: if GK << simple, overnight vol dominates
    "range_vol_ratio":
        ("rank(ts_std(returns, 20) / (ts_mean((high - low) / (close + 1e-8), 20) + 1e-8))",
         "Return vol / range vol ratio (intraday vs close-to-close)",
         "VolRegime"),
    # Short-term vol compression (multiple horizons)
    "vol_compress_3_20":
        ("-1 * rank(ts_std(returns, 3) / (ts_std(returns, 20) + 1e-8))",
         "3d/20d vol compression",
         "VolRegime"),
    "vol_compress_5_60":
        ("-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8))",
         "5d/60d vol compression",
         "VolRegime"),
    "vol_compress_10_120":
        ("-1 * rank(ts_std(returns, 10) / (ts_std(returns, 120) + 1e-8))",
         "10d/120d vol compression",
         "VolRegime"),
    # Vol term structure: short > long means vol expansion regime
    "vol_term_structure":
        ("rank(ts_std(returns, 60) / (ts_std(returns, 10) + 1e-8))",
         "Vol term structure: long vol > short vol = calm after storm",
         "VolRegime"),
    # Range compression: tight Donchian channels
    "range_compress_20_60":
        ("-1 * rank((ts_max(close, 20) - ts_min(close, 20)) / (ts_max(close, 60) - ts_min(close, 60) + 1e-8))",
         "20d/60d range compression",
         "VolRegime"),
    "range_compress_10_40":
        ("-1 * rank((ts_max(close, 10) - ts_min(close, 10)) / (ts_max(close, 40) - ts_min(close, 40) + 1e-8))",
         "10d/40d range compression",
         "VolRegime"),
    # Vol surprise: vol suddenly spiking (new information)
    "vol_spike_direction":
        ("rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8)) * sign(ts_sum(returns, 5))",
         "Vol spike x direction (information surprise continuation)",
         "VolRegime"),
    # Compressed vol + directional trend = coiled spring
    "coiled_spring":
        ("rank(ts_returns(close, 63)) * (-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8)))",
         "Coiled spring: trending + compressed vol",
         "VolRegime"),
    "coiled_spring_short":
        ("rank(ts_returns(close, 21)) * (-1 * rank(ts_std(returns, 3) / (ts_std(returns, 20) + 1e-8)))",
         "Short coiled spring: 1m trend + compressed 3d vol",
         "VolRegime"),
}

# ---------------------------------------------------------------------------
# Cat 4: Volume Intelligence
# ---------------------------------------------------------------------------
# Volume anomalies proxy for information asymmetry and institutional activity.
_VOLUME_INT = {
    # Abnormal volume z-score
    "vol_abnormal":
        ("rank((volume - ts_mean(volume, 60)) / (ts_std(volume, 60) + 1e-8))",
         "Volume z-score vs 60d history (abnormal activity)",
         "VolumeInt"),
    # Volume surge: short vs long term
    "vol_surge_5_20":
        ("rank(ts_mean(volume, 5) / (ts_mean(volume, 20) + 1e-8))",
         "5d/20d volume surge ratio",
         "VolumeInt"),
    "vol_surge_3_60":
        ("rank(ts_mean(volume, 3) / (ts_mean(volume, 60) + 1e-8))",
         "3d/60d volume surge ratio (short burst)",
         "VolumeInt"),
    # On-balance volume trend
    "obv_trend":
        ("rank(ts_sum(if_else(close > delay(close, 1), volume, if_else(close < delay(close, 1), -volume, 0)), 20))",
         "20d on-balance volume (directional flow)",
         "VolumeInt"),
    # Volume-price divergence: high volume but small moves = absorption
    "vol_price_diverge":
        ("-1 * rank(ts_mean(volume, 5) / (ts_mean(volume, 20) + 1e-8)) * rank(ts_std(returns, 5) / (ts_std(returns, 20) + 1e-8))",
         "Volume up + vol down = quiet accumulation",
         "VolumeInt"),
    # Dollar-volume weighted return direction
    "dvol_momentum":
        ("rank(ts_sum(returns * dollar_volume, 20) / (ts_sum(dollar_volume, 20) + 1e-8))",
         "20d dollar-volume-weighted returns",
         "VolumeInt"),
    # Up-volume ratio: proportion of volume on up days
    "up_vol_ratio":
        ("rank(ts_sum(if_else(returns > 0, volume, 0), 20) / (ts_sum(volume, 20) + 1e-8))",
         "20d up-volume ratio (buying vs selling pressure)",
         "VolumeInt"),
    # Volume climax (extreme volume) + direction = potential exhaustion or breakout
    "vol_climax_continue":
        ("rank(ts_max(volume, 5) / (ts_mean(volume, 60) + 1e-8)) * sign(ts_returns(close, 5))",
         "Volume climax continuation: extreme volume x direction",
         "VolumeInt"),
}

# ---------------------------------------------------------------------------
# Cat 5: Momentum Ignition & Regime Change
# ---------------------------------------------------------------------------
# Detect the early stages of new trends and structural shifts.
_IGNITION = {
    # Fresh move on quiet base: strong 5d return but flat 60d
    "ignition_5_60":
        ("rank(ts_returns(close, 5)) * (-1 * rank(abs(ts_returns(close, 60))))",
         "5d strength on flat 60d base (new move igniting)",
         "Ignition"),
    "ignition_3_20":
        ("rank(ts_returns(close, 3)) * (-1 * rank(abs(ts_returns(close, 20))))",
         "3d strength on flat 20d base",
         "Ignition"),
    # 52-week high breakout with volume
    "new_high_252_vol":
        ("rank(close / (ts_max(close, 252) + 1e-8)) * rank(volume / (ts_mean(volume, 20) + 1e-8))",
         "52-week high proximity x volume surge",
         "Ignition"),
    "new_high_60_vol":
        ("rank(close / (ts_max(close, 60) + 1e-8)) * rank(volume / (ts_mean(volume, 20) + 1e-8))",
         "60d high proximity x volume surge",
         "Ignition"),
    # SMA crossover recency: just crossed above 50d SMA
    "sma_cross_50":
        ("rank(close / (ts_mean(close, 50) + 1e-8) - 1) * (-1 * rank(abs(delay(close, 10) / (delay(ts_mean(close, 50), 10) + 1e-8) - 1)))",
         "Above 50d SMA now but near it 10d ago (fresh crossover)",
         "Ignition"),
    # Momentum acceleration: return speeding up
    "mom_acceleration":
        ("rank(ts_returns(close, 5) - delay(ts_returns(close, 5), 5))",
         "Momentum acceleration (5d return improving vs prior 5d)",
         "Ignition"),
    "mom_accel_21":
        ("rank(ts_returns(close, 21) - delay(ts_returns(close, 21), 21))",
         "21d momentum acceleration vs prior 21d",
         "Ignition"),
    # Sector-relative breakout: outperforming peers by abnormal amount
    "sector_breakout":
        ("rank(group_rank(ts_returns(close, 5), sector)) * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8))",
         "Outperforming sector + elevated vol = sector-relative breakout",
         "Ignition"),
}

# ---------------------------------------------------------------------------
# Cat 6: Tail Risk & Skewness
# ---------------------------------------------------------------------------
# Position for asymmetric payoffs and convexity.
_TAIL = {
    # Historical return skewness proxy: ratio of up-variance to total variance
    "pos_skew":
        ("rank(ts_sum(if_else(returns > 0, returns * returns, 0), 60) / (ts_sum(returns * returns, 60) + 1e-8))",
         "Positive skew: higher proportion of variance from up-moves",
         "Tail"),
    # Negative skew avoidance
    "avoid_neg_skew":
        ("-1 * rank(ts_sum(if_else(returns < 0, returns * returns, 0), 60) / (ts_sum(returns * returns, 60) + 1e-8))",
         "Avoid left-tail-heavy stocks",
         "Tail"),
    # Upside/downside capture
    "updown_capture":
        ("rank(ts_mean(if_else(returns > 0, returns, 0), 60) / (ts_mean(if_else(returns < 0, -returns, 0), 60) + 1e-8))",
         "Upside/downside capture ratio (prefer convex payoffs)",
         "Tail"),
    # Kurtosis proxy: frequency of large moves
    "low_kurtosis":
        ("-1 * rank(ts_sum(if_else(abs(returns) > 2 * ts_std(returns, 60), 1, 0), 60))",
         "Low kurtosis: avoid stocks with frequent tail events",
         "Tail"),
    # Maximum favorable excursion: how far up does it go before pulling back?
    "max_up_excursion":
        ("rank((ts_max(close, 20) - close) / (close + 1e-8))",
         "Max favorable excursion in last 20d (stocks that run then hold)",
         "Tail"),
}

# ---------------------------------------------------------------------------
# Cat 7: Multi-Signal Surprise Combos
# ---------------------------------------------------------------------------
# Combine surprise signals for higher conviction.
_COMBOS = {
    # Vol compress + volume surge + trend
    "surp_triple":
        ("(-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8))) + rank(ts_mean(volume, 5) / (ts_mean(volume, 60) + 1e-8)) + rank(ts_returns(close, 63))",
         "Triple: vol compress + volume surge + 3m momentum",
         "Combo"),
    # CLV + volume + momentum alignment
    "surp_clv_vol_mom":
        ("rank(ts_mean((close - low) / (high - low + 1e-8), 5)) * rank(ts_mean(volume, 5) / (ts_mean(volume, 20) + 1e-8)) * rank(ts_returns(close, 21))",
         "CLV buying pressure x volume surge x 1m momentum",
         "Combo"),
    # Gap + volume + coiled spring
    "surp_gap_coiled":
        ("rank((open - delay(close, 1)) / (delay(close, 1) + 1e-8)) * (-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8)))",
         "Positive gap on compressed vol (surprise breakout)",
         "Combo"),
    # New high + low vol + rising volume
    "surp_anticipation":
        ("rank(close / (ts_max(close, 60) + 1e-8)) + (-1 * rank(ts_std(returns, 10) / (ts_std(returns, 60) + 1e-8))) + rank(ts_mean(volume, 5) / (ts_mean(volume, 60) + 1e-8))",
         "Near 60d high + compressed vol + rising volume",
         "Combo"),
    # Accumulation breakout: quiet buildup then move
    "surp_accumulation_break":
        ("rank(ts_sum(((close - low) - (high - close)) / (high - low + 1e-8) * volume, 20)) * (-1 * rank(ts_std(returns, 5) / (ts_std(returns, 20) + 1e-8)))",
         "A/D accumulation + compressed short-term vol",
         "Combo"),
    # Sector breakout + vol compress + volume
    "surp_sector_ignite":
        ("group_rank(ts_returns(close, 5), sector) * (-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8))) * rank(ts_mean(volume, 3) / (ts_mean(volume, 20) + 1e-8))",
         "Sector outperformer + compressed vol + volume burst",
         "Combo"),
    # Everything surprise: CLV + gap + vol regime + volume + momentum
    "surp_full_model":
        ("rank(ts_mean((close - low) / (high - low + 1e-8), 5)) + rank(ts_sum((open - delay(close, 1)) / (delay(close, 1) + 1e-8), 5)) + (-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8))) + rank(ts_mean(volume, 5) / (ts_mean(volume, 60) + 1e-8)) + rank(ts_returns(close, 21))",
         "Full model: CLV + gap + vol compress + vol surge + momentum",
         "Combo"),
    # Risk-adjusted surprise: full model scaled by inverse vol
    "surp_risk_adj":
        ("(rank(ts_mean((close - low) / (high - low + 1e-8), 5)) + (-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8))) + rank(ts_mean(volume, 5) / (ts_mean(volume, 60) + 1e-8))) * rank(1.0 / (ts_std(returns, 20) + 1e-8))",
         "Surprise combo x inverse vol (risk-adjusted)",
         "Combo"),
    # Overnight-session-driven combo
    "surp_overnight_combo":
        ("rank(ts_mean((open - delay(close, 1)) / (delay(close, 1) + 1e-8), 10)) + rank(ts_mean((close - low) / (high - low + 1e-8), 10)) + (-1 * rank(ts_std(returns, 5) / (ts_std(returns, 60) + 1e-8)))",
         "Overnight trend + CLV + vol compress",
         "Combo"),
}

# Merge all categories
for _cat in [_GAP, _CLV, _VOL_REGIME, _VOLUME_INT, _IGNITION, _TAIL, _COMBOS]:
    SURPRISE_ALPHAS.update(_cat)

CATEGORY_LABELS = {}
for name, (_, _, cat) in SURPRISE_ALPHAS.items():
    CATEGORY_LABELS[name] = cat

CATEGORY_COLORS = {
    "Gap": "#f97316",
    "CLV": "#06b6d4",
    "VolRegime": "#a78bfa",
    "VolumeInt": "#ffd93d",
    "Ignition": "#4ade80",
    "Tail": "#ff6b6b",
    "Combo": "#ec4899",
}


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_alphas(
    data: AlphaDataSet,
    mode: str = "long_only",
    max_weight: float = 0.01,
    decay: int = 0,
    neutralize: str = "none",
    delay: int = 1,
    annualization: int = 252,
    cost_bps: float = COST_BPS,
) -> tuple[pd.DataFrame, dict, dict]:
    rows = []
    pnl_dict: dict[str, pd.Series] = {}
    nav_dict: dict[str, pd.Series] = {}

    total = len(SURPRISE_ALPHAS)
    for i, (name, (expr, desc, cat)) in enumerate(sorted(SURPRISE_ALPHAS.items()), 1):
        t0 = time.time()
        try:
            alpha_signal = evaluate_alpha(expr, data)
            weights = construct_portfolio(
                alpha_signal, mode=mode, max_weight=max_weight,
                decay=decay, neutralize=neutralize, sector_df=data.sector,
            )
            if delay > 1:
                weights = weights.shift(delay - 1)
            metrics = simulate(weights, data.returns, cost_bps=cost_bps,
                             annualization=annualization)
            elapsed = time.time() - t0

            rating = rate_alpha(metrics)
            sc = score_alpha(metrics)
            checks = check_passing(metrics, delay=delay)
            passes = all(v[0] for v in checks.values())

            pnl_dict[name] = metrics["daily_pnl"].dropna()
            nav_dict[name] = metrics["cum_nav"].dropna()

            rows.append({
                "name": name, "category": cat,
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
            print(f"  [{i:3d}/{total}] {name:<28s} Sharpe={metrics['sharpe']:7.2f}  "
                  f"Return={metrics['returns']*100:6.1f}%  DD={metrics['drawdown']*100:6.1f}%  "
                  f"[{cat}]")

        except Exception as e:
            elapsed = time.time() - t0
            rows.append({
                "name": name, "category": cat,
                "description": desc, "expression": expr,
                "sharpe": 0, "sortino": 0, "turnover": 0, "fitness": 0,
                "brain_fitness": 0, "returns": 0, "total_return": 0,
                "drawdown": 0, "margin": 0, "win_rate": 0, "n_bars": 0,
                "rating": "Error", "score": 0, "passes": False,
                "elapsed": elapsed, "error": str(e),
            })
            print(f"  [{i:3d}/{total}] {name:<28s} ERROR: {str(e)[:80]}")

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
    cost_bps: float = COST_BPS,
):
    _setup_style()
    valid = results_df[results_df["error"].isna()].copy()
    n_total = len(results_df)
    n_valid = len(valid)
    n_errors = n_total - n_valid

    with PdfPages(str(output_path)) as pdf:
        # PAGE 1: Title & Summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.88, "Stock Surprise / Anticipation Alphas",
                 fontsize=26, fontweight="bold", ha="center", color=ACCENT)
        fig.text(0.5, 0.82, f"S&P 500 Cross-Sectional BRAIN Evaluation — {mode.replace('_', ' ').title()}",
                 fontsize=14, ha="center", color=TEXT_COLOR)
        fig.text(0.5, 0.77,
                 f"{data.date_start.date()} to {data.date_end.date()}  |  "
                 f"{len(data.symbols)} stocks  |  Cost: {cost_bps:.0f} bps",
                 fontsize=11, ha="center", color="#888888")

        lines = [
            f"Alphas Evaluated:   {n_total}  ({n_valid} OK, {n_errors} errors)",
            "",
            f"Best Sharpe:        {valid['sharpe'].max():.2f}  ({valid.iloc[0]['name']})",
            f"Median Sharpe:      {valid['sharpe'].median():.2f}",
            f"Positive Sharpe:    {(valid['sharpe'] > 0).sum()} / {n_valid}",
            f"Mean Ann. Return:   {valid['returns'].mean()*100:.1f}%",
            f"Mean Max Drawdown:  {valid['drawdown'].mean()*100:.1f}%",
            f"Mean Turnover:      {valid['turnover'].mean()*100:.1f}%",
            "",
            "Performance by Category:",
        ]
        for cat in ["Gap", "CLV", "VolRegime", "VolumeInt", "Ignition", "Tail", "Combo"]:
            cdf = valid[valid["category"] == cat]
            if len(cdf) > 0:
                pos = (cdf["sharpe"] > 0).sum()
                lines.append(
                    f"  {cat:<12s}  n={len(cdf):2d}  "
                    f"Sharpe: {cdf['sharpe'].median():+.2f} (med)  "
                    f"Best: {cdf['sharpe'].max():+.2f}  "
                    f"Pos: {pos}/{len(cdf)}"
                )

        y = 0.65
        for line in lines:
            fig.text(0.12, y, line, fontsize=10.5, fontfamily="monospace", color=TEXT_COLOR)
            y -= 0.032
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 2: Sharpe by Category (box plot + bar chart)
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("Sharpe Ratio by Surprise Category", fontsize=14,
                     fontweight="bold", color=ACCENT)

        cats_order = ["Gap", "CLV", "VolRegime", "VolumeInt", "Ignition", "Tail", "Combo"]
        cats_present = [c for c in cats_order if c in valid["category"].values]

        ax = axes[0]
        box_data = [valid[valid["category"] == c]["sharpe"].values for c in cats_present]
        bp = ax.boxplot(box_data, tick_labels=cats_present, patch_artist=True, widths=0.6)
        for patch, cat in zip(bp["boxes"], cats_present):
            patch.set_facecolor(CATEGORY_COLORS.get(cat, "#888"))
            patch.set_alpha(0.7)
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.5)
        ax.set_ylabel("Sharpe Ratio")
        ax.grid(True, alpha=0.2)

        ax = axes[1]
        for _, row in valid.iterrows():
            color = CATEGORY_COLORS.get(row["category"], "#888")
            ax.bar(row["name"], row["sharpe"], color=color, alpha=0.7)
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.5)
        ax.set_ylabel("Sharpe Ratio")
        ax.tick_params(axis="x", rotation=90, labelsize=5)
        ax.grid(True, alpha=0.2, axis="y")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 3: Top 10 Equity Curves
        top10 = valid.head(10)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[3, 1])
        fig.suptitle("Top 10 Surprise Alphas — Equity Curves", fontsize=14,
                     fontweight="bold", color=ACCENT)
        cmap = plt.colormaps["tab10"]
        for i, (_, row) in enumerate(top10.iterrows()):
            name = row["name"]
            if name in nav_dict:
                nav = nav_dict[name]
                ax1.plot(nav.index, nav.values, linewidth=1.3,
                        label=f"{name} (S={row['sharpe']:.2f})",
                        color=cmap(i % 10))
        ax1.set_ylabel("NAV")
        ax1.legend(fontsize=7, loc="upper left", ncol=2)
        ax1.grid(True, alpha=0.2)

        for i, (_, row) in enumerate(top10.iterrows()):
            name = row["name"]
            if name in nav_dict:
                nav = nav_dict[name]
                peak = nav.cummax()
                dd = (nav - peak) / peak
                ax2.plot(dd.index, dd.values, linewidth=0.8, color=cmap(i % 10))
        ax2.set_ylabel("Drawdown")
        ax2.grid(True, alpha=0.2)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 4: Top 10 Rolling Sharpe
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.suptitle("Top 10 Surprise Alphas — Rolling Sharpe (252d)", fontsize=14,
                     fontweight="bold", color=ACCENT)
        for i, (_, row) in enumerate(top10.iterrows()):
            name = row["name"]
            if name in pnl_dict:
                pnl = pnl_dict[name]
                rm = pnl.rolling(252, min_periods=60).mean()
                rs = pnl.rolling(252, min_periods=60).std()
                rolling_s = (rm / rs) * np.sqrt(annualization)
                rolling_s = rolling_s.dropna()
                ax.plot(rolling_s.index, rolling_s.values, linewidth=1,
                       label=name, color=cmap(i % 10))
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)
        ax.set_ylabel("Rolling Sharpe")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 5: Correlation heatmap of top 20
        top20_names = valid.head(20)["name"].tolist()
        top20_pnl = pd.DataFrame({n: pnl_dict[n] for n in top20_names if n in pnl_dict})
        if top20_pnl.shape[1] >= 3:
            corr = top20_pnl.corr()
            fig, ax = plt.subplots(figsize=(11, 9))
            im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns, fontsize=7)
            for ci in range(len(corr)):
                for cj in range(len(corr)):
                    ax.text(cj, ci, f"{corr.iloc[ci, cj]:.2f}", ha="center", va="center",
                            fontsize=5, color="black" if abs(corr.iloc[ci, cj]) < 0.5 else "white")
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("PnL Correlation — Top 20 Surprise Strategies", fontsize=14,
                         fontweight="bold", color=ACCENT)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # PAGE 6: Ensembles
        for ens_n in [5, 10, 20]:
            ens_names = valid.head(ens_n)["name"].tolist()
            ens_pnl_df = pd.DataFrame({n: pnl_dict[n] for n in ens_names if n in pnl_dict})
            if ens_pnl_df.shape[1] < 2:
                continue
            ens_pnl = ens_pnl_df.mean(axis=1).dropna()
            if len(ens_pnl) < 30:
                continue
            ens_nav = (1 + ens_pnl).cumprod()
            std = float(ens_pnl.std())
            ens_sharpe = float(ens_pnl.mean() / std * np.sqrt(annualization)) if std > 1e-12 else 0.0
            total_ret = float(ens_nav.iloc[-1] - 1.0)
            n_yrs = len(ens_pnl) / annualization
            ann_ret = float((1 + total_ret)**(1/n_yrs) - 1) if n_yrs > 0 else 0.0
            peak = ens_nav.cummax()
            dd = (ens_nav - peak) / peak
            max_dd = float(dd.min())

            fig, axes = plt.subplots(3, 1, figsize=(11, 8.5))
            fig.suptitle(f"Equal-Weight Ensemble — Top {ens_n} Surprise Strategies",
                         fontsize=14, fontweight="bold", color=ACCENT)

            axes[0].plot(ens_nav.index, ens_nav.values, color=ACCENT, linewidth=2)
            axes[0].set_ylabel("NAV")
            axes[0].set_title(
                f"Sharpe={ens_sharpe:.2f}  CAGR={ann_ret*100:.1f}%  MaxDD={max_dd*100:.1f}%",
                fontsize=11)
            axes[0].grid(True, alpha=0.2)

            roll_m = ens_pnl.rolling(252, min_periods=60).mean()
            roll_s = ens_pnl.rolling(252, min_periods=60).std()
            rs = (roll_m / roll_s) * np.sqrt(annualization)
            rs = rs.dropna()
            axes[1].plot(rs.index, rs.values, color=ACCENT4, linewidth=1.2)
            axes[1].axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)
            axes[1].set_ylabel("Rolling Sharpe (252d)")
            axes[1].grid(True, alpha=0.2)

            axes[2].fill_between(dd.index, dd.values, 0, alpha=0.4, color=ACCENT2)
            axes[2].set_ylabel("Drawdown")
            axes[2].grid(True, alpha=0.2)

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)

        # PAGE 7+: Scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.suptitle("Surprise Alpha Trade-offs", fontsize=14,
                     fontweight="bold", color=ACCENT)

        ax = axes[0]
        for cat in CATEGORY_COLORS:
            sub = valid[valid["category"] == cat]
            if not sub.empty:
                ax.scatter(sub["turnover"] * 100, sub["sharpe"],
                          label=cat, color=CATEGORY_COLORS[cat], s=40, alpha=0.8)
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Turnover (%)")
        ax.set_ylabel("Sharpe")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        ax = axes[1]
        sc = ax.scatter(valid["drawdown"] * 100, valid["returns"] * 100,
                       c=valid["sharpe"], cmap="coolwarm", s=40, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="Sharpe")
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Ann. Return (%)")
        ax.grid(True, alpha=0.2)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 8+: Results Tables
        rows_per_page = 25
        for page_start in range(0, len(valid), rows_per_page):
            chunk = valid.iloc[page_start:page_start + rows_per_page]
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")

            col_labels = ["#", "Alpha", "Cat", "Sharpe", "Return", "MaxDD",
                          "Turn", "Win%", "Description"]
            cell_data = []
            for rank_i, (_, row) in enumerate(chunk.iterrows(), page_start + 1):
                cell_data.append([
                    str(rank_i), row["name"], row["category"],
                    f"{row['sharpe']:.2f}", f"{row['returns']*100:.1f}%",
                    f"{row['drawdown']*100:.1f}%", f"{row['turnover']*100:.1f}%",
                    f"{row['win_rate']*100:.0f}%",
                    row["description"][:45],
                ])

            table = ax.table(cellText=cell_data, colLabels=col_labels,
                           loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.15)

            for (row_i, col_i), cell in table.get_celld().items():
                cell.set_edgecolor(GRID_COLOR)
                if row_i == 0:
                    cell.set_facecolor("#2a2a4a")
                    cell.set_text_props(color=ACCENT, fontweight="bold")
                else:
                    cell.set_facecolor(TABLE_ROW_ALT if row_i % 2 == 0 else DARK_BG)
                    cell.set_text_props(color=TEXT_COLOR)

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

def main():
    parser = argparse.ArgumentParser(
        description="Run stock surprise alphas through BRAIN system")
    parser.add_argument("--db", default=None, help="DuckDB path")
    parser.add_argument("--mode", choices=["long_only", "long_short"], default="long_only")
    parser.add_argument("--max-weight", type=float, default=0.01)
    parser.add_argument("--decay", type=int, default=0)
    parser.add_argument("--neutralize", choices=["none", "sector", "market"], default="none")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--cost", type=float, default=5.0,
                        help="Transaction cost in bps (default: 5 for S&P 500)")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=None, help="Output PDF path")
    args = parser.parse_args()

    ann = 252
    alpha_brain.ANNUALIZATION = ann

    sp500_json = PROJECT_ROOT / "scripts" / "research" / "sp500_tickers.json"
    with open(sp500_json) as f:
        sp_data = json.load(f)
    all_tickers = sp_data["tickers"]
    sector_map = sp_data["sectors"]

    db_path = args.db or str(PROJECT_ROOT / ".." / "data" / "stocks_market.duckdb")

    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    good = con.execute(
        "SELECT symbol FROM bars_1d WHERE ts < '2015-01-01' "
        "GROUP BY symbol HAVING COUNT(*) >= 200"
    ).fetchdf()["symbol"].tolist()
    con.close()
    symbols = sorted(set(all_tickers) & set(good))

    print(f"\n{'='*72}")
    print(f"  Stock Surprise / Anticipation Alphas — BRAIN Evaluation")
    print(f"  Mode: {args.mode}  Delay: {args.delay}  Cost: {args.cost:.0f} bps")
    print(f"  Universe: {len(symbols)} S&P 500 stocks")
    print(f"  Alphas: {len(SURPRISE_ALPHAS)}")
    print(f"{'='*72}\n")

    t0 = time.time()
    data = AlphaDataSet(
        db_path, symbols, start=args.start, end=args.end,
        table="bars_1d", annualization=ann, sector_map=sector_map,
    )

    print(f"\nEvaluating {len(SURPRISE_ALPHAS)} surprise alphas...\n")
    results_df, pnl_dict, nav_dict = run_all_alphas(
        data, mode=args.mode, max_weight=args.max_weight,
        decay=args.decay, neutralize=args.neutralize, delay=args.delay,
        annualization=ann, cost_bps=args.cost,
    )

    out_dir = PROJECT_ROOT / "artifacts" / "research" / "alpha_brain"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = f"_{args.mode}" if args.mode != "long_only" else ""
    cost_suffix = f"_{int(args.cost)}bps" if args.cost != 5.0 else ""
    csv_path = out_dir / f"surprise_stocks_results{mode_suffix}{cost_suffix}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    if args.output:
        pdf_path = Path(args.output)
    else:
        pdf_path = out_dir / f"surprise_stocks_report{mode_suffix}{cost_suffix}.pdf"
    generate_report(results_df, pnl_dict, nav_dict, data, args.mode, pdf_path,
                    annualization=ann, cost_bps=args.cost)

    # Print top 10 summary
    print(f"\n  {'='*72}")
    print(f"  TOP 10 SURPRISE ALPHAS ({args.mode})")
    print(f"  {'='*72}")
    print(f"  {'#':>3s}  {'Name':<28s} {'Cat':<10s} {'Sharpe':>7s} {'Return':>8s} {'MaxDD':>7s} {'Turn':>6s}")
    print(f"  {'─'*72}")
    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"  {i:3d}  {row['name']:<28s} {row['category']:<10s} "
              f"{row['sharpe']:7.2f} {row['returns']*100:7.1f}% "
              f"{row['drawdown']*100:6.1f}% {row['turnover']*100:5.1f}%")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
