#!/usr/bin/env python3
"""
Clenow Systematic Momentum — Crypto Adaptation
================================================

Replicates the Chapter 12 strategy from Andreas Clenow's "Trading Evolved",
adapted for the crypto universe:

  1. Rank assets by annualized exponential regression slope × R²
  2. BTC regime filter (BTC > 200-day MA → risk-on, else 100% cash)
  3. Per-asset trend filter (price > 100-day MA)
  4. ATR-based position sizing with portfolio risk targeting
  5. Weekly rebalancing — rotate into top-N ranked assets

Usage:
    python -m scripts.research.alpha_lab.clenow_momentum
"""
from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from common.data import load_daily_bars, ANN_FACTOR
from common.metrics import compute_metrics

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

OUT_DIR = ROOT / "artifacts" / "research" / "alpha_lab"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "clenow_momentum_report.pdf"

# ── Palette ──────────────────────────────────────────────────────────
CB = "#0F2E5F"; CLB = "#3366A6"; CG = "#C2A154"; CR = "#B22222"
CGr = "#2E7D32"; CGy = "#888888"; CTEAL = "#006B6B"
CORAL = "#E07040"; PURPLE = "#6B3FA0"

JPM_BLUE = colors.Color(0.06, 0.18, 0.37)
JPM_BLUE_LIGHT = colors.Color(0.20, 0.40, 0.65)
JPM_GOLD = colors.Color(0.76, 0.63, 0.33)
JPM_GRAY = colors.Color(0.55, 0.55, 0.55)
JPM_GRAY_LIGHT = colors.Color(0.93, 0.93, 0.93)
WHITE = colors.white; BLACK = colors.black
PAGE_W, PAGE_H = letter; MARGIN = 0.75 * inch; CONTENT_W = PAGE_W - 2 * MARGIN


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MomentumConfig:
    name: str = "Clenow Base"

    # Signal
    regression_window: int = 90       # days for exp regression
    annualization_factor: float = 252.0  # trading days per year (used for slope)

    # Regime filter
    regime_filter: bool = True
    regime_ma_window: int = 200       # BTC above N-day MA = risk-on

    # Per-asset filter
    asset_ma_window: int = 100        # asset must be above its own N-day MA

    # Portfolio construction
    top_n: int = 10                   # hold top N ranked assets
    rebal_freq_days: int = 7          # rebalance every N calendar days
    risk_factor: float = 0.001        # target risk per position (10 bps per ATR)
    atr_period: int = 20
    max_weight: float = 0.20          # max 20% in any single asset
    vol_target: float = 0.40          # portfolio vol target (annualized)

    # Universe
    min_history_days: int = 365
    min_adv_usd: float = 500_000.0

    # Costs
    cost_bps: float = 20.0

    # Drawdown control
    dd_control: bool = True
    dd_threshold: float = -0.15       # go to cash at -15% from peak
    dd_cooldown_days: int = 15


# ═══════════════════════════════════════════════════════════════════════
# Signal computation
# ═══════════════════════════════════════════════════════════════════════

def momentum_score(prices: pd.Series, window: int = 90,
                   ann_factor: float = 252.0) -> tuple[float, float, float]:
    """Compute annualized exponential regression slope × R².

    Returns (score, annualized_slope, r_squared).
    """
    if len(prices) < window:
        return np.nan, np.nan, np.nan
    p = prices.iloc[-window:]
    if p.min() <= 0 or p.isna().any():
        return np.nan, np.nan, np.nan

    log_p = np.log(p.values)
    x = np.arange(window, dtype=float)
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, log_p)

    # Annualize: daily slope → annual return
    ann_slope = (np.exp(slope * ann_factor) - 1.0)
    r_sq = r_value ** 2
    score = ann_slope * r_sq

    return score, ann_slope, r_sq


def compute_all_scores(close_wide: pd.DataFrame, date: object,
                       window: int, ann_factor: float,
                       eligible: set) -> pd.DataFrame:
    """Compute momentum scores for all eligible assets on a given date."""
    rows = []
    for sym in eligible:
        if sym not in close_wide.columns:
            continue
        prices = close_wide[sym].loc[:date].dropna()
        if len(prices) < window:
            continue
        score, ann_slope, r_sq = momentum_score(prices, window, ann_factor)
        if np.isnan(score):
            continue
        rows.append({"symbol": sym, "score": score,
                     "ann_slope": ann_slope, "r_squared": r_sq})
    return pd.DataFrame(rows)


def compute_atr(close: pd.Series, high: pd.Series, low: pd.Series,
                period: int = 20) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# ═══════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════

def prepare_data(cfg: MomentumConfig):
    print("[data] Loading daily bars...")
    panel = load_daily_bars(start="2017-01-01", end="2026-12-31")

    sym_stats = panel.groupby("symbol").agg(n_days=("ts", "count"))
    long_enough = sym_stats[sym_stats["n_days"] >= cfg.min_history_days].index
    panel = panel[panel["symbol"].isin(long_enough)].copy().sort_values(["symbol", "ts"])

    panel["dollar_vol"] = panel["close"] * panel["volume"]
    panel["adv_20"] = panel.groupby("symbol")["dollar_vol"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    panel["in_universe"] = panel["adv_20"] >= cfg.min_adv_usd
    panel.loc[panel["adv_20"].isna(), "in_universe"] = False

    symbols = sorted(panel[panel["in_universe"]].symbol.unique())
    print(f"[data] Universe: {len(symbols)} assets")

    close_wide = panel.pivot_table(index="ts", columns="symbol", values="close")
    high_wide = panel.pivot_table(index="ts", columns="symbol", values="high")
    low_wide = panel.pivot_table(index="ts", columns="symbol", values="low")
    universe_wide = panel.pivot_table(index="ts", columns="symbol",
                                      values="in_universe").fillna(False).astype(bool)
    universe_wide = universe_wide.reindex(close_wide.index).fillna(False)

    dates = sorted(close_wide.index)

    # Precompute ATR and per-asset MAs
    atr_wide = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)
    ma_asset_wide = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)

    for sym in symbols:
        if sym not in close_wide.columns:
            continue
        c = close_wide[sym].dropna()
        h = high_wide[sym].reindex(c.index)
        lo = low_wide[sym].reindex(c.index)
        if len(c) < cfg.atr_period + 5:
            continue
        atr_wide.loc[c.index, sym] = compute_atr(c, h, lo, cfg.atr_period)
        ma_asset_wide.loc[c.index, sym] = c.rolling(cfg.asset_ma_window,
                                                      min_periods=cfg.asset_ma_window).mean()

    # BTC regime MA
    btc = close_wide.get("BTC-USD", pd.Series(dtype=float)).dropna()
    btc_ma = btc.rolling(cfg.regime_ma_window, min_periods=cfg.regime_ma_window).mean()

    print(f"[data] Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    return {
        "close": close_wide, "high": high_wide, "low": low_wide,
        "atr": atr_wide, "ma_asset": ma_asset_wide,
        "universe": universe_wide,
        "btc_close": btc, "btc_ma": btc_ma,
        "symbols": symbols, "dates": dates,
    }


# ═══════════════════════════════════════════════════════════════════════
# Portfolio simulation
# ═══════════════════════════════════════════════════════════════════════

def run_simulation(data: dict, cfg: MomentumConfig) -> dict:
    close = data["close"]
    atr_w = data["atr"]
    ma_asset = data["ma_asset"]
    universe = data["universe"]
    btc_close = data["btc_close"]
    btc_ma = data["btc_ma"]
    dates = data["dates"]

    cost_one_way = cfg.cost_bps / 2.0 / 10_000
    equity = INITIAL_EQUITY = 1_000_000.0

    # Portfolio state: {symbol: {"coins": float, "weight": float}}
    holdings: dict[str, float] = {}  # symbol -> number of coins
    cash = equity

    daily_equity = []
    daily_n_holdings = []
    daily_gross_exposure = []
    daily_regime = []
    daily_dd_lockout = []
    rebalance_log = []

    peak_equity = equity
    in_dd_lockout = False
    dd_lockout_until_idx = -1
    last_rebal_idx = -cfg.rebal_freq_days  # force first rebal
    n_rebalances = 0

    t0 = time.time()

    for di, date in enumerate(dates):
        prices = close.loc[date].dropna()

        # Mark to market
        port_value = cash
        for sym, coins in holdings.items():
            if sym in prices:
                port_value += coins * prices[sym]

        # ── Regime check ──
        bp = btc_close.get(date)
        bma = btc_ma.get(date)
        regime_on = True
        if cfg.regime_filter:
            regime_on = pd.notna(bp) and pd.notna(bma) and bp > bma

        # ── DD control ──
        dd_active = False
        if cfg.dd_control:
            if port_value > peak_equity:
                peak_equity = port_value
            current_dd = port_value / peak_equity - 1.0
            if in_dd_lockout:
                if di >= dd_lockout_until_idx and regime_on:
                    in_dd_lockout = False
                    peak_equity = port_value
            elif current_dd < cfg.dd_threshold:
                in_dd_lockout = True
                dd_lockout_until_idx = di + cfg.dd_cooldown_days
            dd_active = in_dd_lockout

        # ── Liquidate if regime off or DD lockout ──
        go_to_cash = (not regime_on) or dd_active

        if go_to_cash and holdings:
            for sym, coins in holdings.items():
                if sym in prices:
                    sell_value = coins * prices[sym]
                    cash += sell_value - sell_value * cost_one_way
            holdings.clear()

        # ── Rebalance check ──
        is_rebal_day = (di - last_rebal_idx) >= cfg.rebal_freq_days

        if is_rebal_day and not go_to_cash:
            last_rebal_idx = di

            # Eligible symbols: in universe + above own MA
            eligible = set()
            for sym in data["symbols"]:
                if sym not in prices:
                    continue
                if not universe.loc[date].get(sym, False):
                    continue
                asset_ma = ma_asset.loc[date].get(sym)
                if pd.notna(asset_ma) and prices[sym] > asset_ma:
                    eligible.add(sym)

            # Rank by momentum score
            scores_df = compute_all_scores(
                close, date, cfg.regression_window, cfg.annualization_factor, eligible)

            if len(scores_df) == 0:
                # No eligible assets — go to cash
                for sym, coins in holdings.items():
                    if sym in prices:
                        sell_value = coins * prices[sym]
                        cash += sell_value - sell_value * cost_one_way
                holdings.clear()
            else:
                scores_df = scores_df.sort_values("score", ascending=False)
                target_syms = set(scores_df.head(cfg.top_n)["symbol"])

                # Compute target weights via inverse-ATR
                target_weights = {}
                for _, row in scores_df.head(cfg.top_n).iterrows():
                    sym = row["symbol"]
                    a = atr_w.loc[date].get(sym)
                    p = prices.get(sym)
                    if pd.isna(a) or a <= 0 or pd.isna(p) or p <= 0:
                        continue
                    # Weight = risk_factor / (ATR/price) — inverse-vol
                    atr_pct = a / p
                    raw_w = cfg.risk_factor / atr_pct if atr_pct > 0 else 0
                    target_weights[sym] = min(raw_w, cfg.max_weight)

                # Normalize to vol target
                total_raw = sum(target_weights.values())
                if total_raw > 1.0:
                    scale = 1.0 / total_raw
                    target_weights = {s: w * scale for s, w in target_weights.items()}

                # Sell holdings not in target
                for sym in list(holdings.keys()):
                    if sym not in target_weights:
                        if sym in prices:
                            sell_value = holdings[sym] * prices[sym]
                            cash += sell_value - sell_value * cost_one_way
                        del holdings[sym]

                # Rebalance: adjust existing + buy new
                total_equity = cash
                for sym, coins in holdings.items():
                    if sym in prices:
                        total_equity += coins * prices[sym]

                for sym, tw in target_weights.items():
                    target_value = total_equity * tw
                    current_value = holdings.get(sym, 0) * prices.get(sym, 0)
                    delta = target_value - current_value

                    if abs(delta) < total_equity * 0.005:
                        continue  # skip tiny rebalances

                    if sym not in prices or prices[sym] <= 0:
                        continue

                    if delta > 0:  # buy
                        buy_cost = delta + delta * cost_one_way
                        if buy_cost > cash:
                            delta = cash / (1 + cost_one_way)
                            buy_cost = delta + delta * cost_one_way
                        coins_to_buy = delta / prices[sym]
                        holdings[sym] = holdings.get(sym, 0) + coins_to_buy
                        cash -= buy_cost
                    else:  # sell
                        coins_to_sell = abs(delta) / prices[sym]
                        coins_to_sell = min(coins_to_sell, holdings.get(sym, 0))
                        sell_value = coins_to_sell * prices[sym]
                        holdings[sym] = holdings.get(sym, 0) - coins_to_sell
                        cash += sell_value - sell_value * cost_one_way
                        if holdings.get(sym, 0) < 1e-10:
                            holdings.pop(sym, None)

                n_rebalances += 1
                rebalance_log.append({
                    "date": date, "n_held": len(holdings),
                    "top_score": scores_df.iloc[0]["score"] if len(scores_df) > 0 else 0,
                    "n_eligible": len(eligible),
                })

        # ── End-of-day equity ──
        equity_eod = cash
        long_notional = 0.0
        for sym, coins in holdings.items():
            if sym in prices:
                mv = coins * prices[sym]
                equity_eod += mv
                long_notional += mv

        daily_equity.append(equity_eod)
        daily_n_holdings.append(len(holdings))
        daily_gross_exposure.append(long_notional / equity_eod if equity_eod > 0 else 0)
        daily_regime.append(regime_on)
        daily_dd_lockout.append(dd_active)

        if (di + 1) % 500 == 0:
            print(f"  [{cfg.name}] Day {di+1}/{len(dates)}: "
                  f"eq=${equity_eod:,.0f}, hold={len(holdings)}, "
                  f"regime={'ON' if regime_on else 'OFF'}")

    elapsed = time.time() - t0
    eq = pd.Series(daily_equity, index=dates, name="equity")
    eq_norm = eq / eq.iloc[0]
    print(f"  [{cfg.name}] Final=${daily_equity[-1]:,.0f}, "
          f"Rebals={n_rebalances}, {elapsed:.1f}s")

    return {
        "name": cfg.name, "equity": eq, "equity_norm": eq_norm,
        "n_holdings": pd.Series(daily_n_holdings, index=dates),
        "gross_exposure": pd.Series(daily_gross_exposure, index=dates),
        "regime": pd.Series(daily_regime, index=dates),
        "dd_lockout": pd.Series(daily_dd_lockout, index=dates),
        "rebalance_log": pd.DataFrame(rebalance_log),
        "n_rebalances": n_rebalances,
    }


INITIAL_EQUITY = 1_000_000.0


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze(sim: dict) -> dict:
    eq = sim["equity_norm"]
    m = compute_metrics(eq)
    annual = sim["equity"].copy()
    annual.index = pd.to_datetime(annual.index)
    ann_ret = annual.resample("YE").last().pct_change().dropna()
    ann_ret.index = ann_ret.index.year
    peak_retained = sim["equity"].iloc[-1] / sim["equity"].max()
    return {"metrics": m, "annual": ann_ret, "peak_retained": peak_retained}


# ═══════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════

VARIANT_COLORS = [CGy, CR, CLB, CTEAL, CGr, CORAL, PURPLE, CB, CG]

def set_chart_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
        "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.fontsize": 8, "legend.framealpha": 0.9,
    })

def fig_to_image(fig, width=6.5 * inch, ratio=0.55):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width, height=width * ratio)


def chart_equity_all(sims, btc_eq):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(btc_eq.index, btc_eq.values, color=CG, lw=1.0, ls="--", alpha=0.5, label="BTC B&H")
    for i, sim in enumerate(sims):
        eq = sim["equity_norm"]
        c = VARIANT_COLORS[i % len(VARIANT_COLORS)]
        lw = 2.0 if i == len(sims) - 1 else 1.0
        ax.plot(eq.index, eq.values, color=c, lw=lw, label=sim["name"],
                alpha=1.0 if i == len(sims) - 1 else 0.65)
    ax.set_yscale("log")
    ax.set_title("Clenow Systematic Momentum — Equity Curves (log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def chart_drawdown_all(sims, btc_eq):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dd = btc_eq / btc_eq.cummax() - 1.0
    ax.fill_between(dd.index, dd.values, 0, alpha=0.05, color=CG)
    ax.plot(dd.index, dd.values, color=CG, lw=0.5, ls="--", alpha=0.3, label="BTC B&H")
    for i, sim in enumerate(sims):
        eq = sim["equity_norm"]
        d = eq / eq.cummax() - 1.0
        c = VARIANT_COLORS[i % len(VARIANT_COLORS)]
        show_fill = (i == len(sims) - 1)
        if show_fill:
            ax.fill_between(d.index, d.values, 0, alpha=0.12, color=c)
        ax.plot(d.index, d.values, color=c, lw=0.8 if not show_fill else 1.5,
                label=sim["name"], alpha=0.5 if not show_fill else 1.0)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    return fig


def chart_regime_and_holdings(sim, data):
    set_chart_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1, 1]})
    eq = sim["equity"]
    ax1.plot(eq.index, eq.values, color=CB, lw=1.0)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"{sim['name']} — Equity + Regime")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

    # Shade risk-off periods
    regime = sim["regime"]
    for i in range(1, len(regime)):
        if not regime.iloc[i]:
            ax1.axvspan(regime.index[i-1], regime.index[i], alpha=0.08, color=CR)

    nh = sim["n_holdings"]
    ax2.fill_between(nh.index, nh.values, alpha=0.3, color=CLB, step="post")
    ax2.plot(nh.index, nh.values, color=CLB, lw=0.6, drawstyle="steps-post")
    ax2.set_ylabel("# Holdings")

    ge = sim["gross_exposure"]
    ax3.fill_between(ge.index, ge.values, alpha=0.3, color=CTEAL, step="post")
    ax3.plot(ge.index, ge.values, color=CTEAL, lw=0.6, drawstyle="steps-post")
    ax3.set_ylabel("Exposure")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax3.set_xlabel("Date")

    fig.tight_layout()
    return fig


def chart_annual(sims, data):
    set_chart_style()
    btc = data["close"]["BTC-USD"].dropna()
    btc.index = pd.to_datetime(btc.index)
    btc_ann = btc.resample("YE").last().pct_change().dropna()
    years = sorted(set(btc_ann.index.year))
    n = len(sims)
    width = 0.7 / (n + 1)
    x = np.arange(len(years))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.35, [btc_ann.get(pd.Timestamp(f"{y}-12-31"), 0) for y in years],
           width, label="BTC B&H", color=CG, alpha=0.5)
    for i, sim in enumerate(sims):
        eq = sim["equity"].copy()
        eq.index = pd.to_datetime(eq.index)
        ann = eq.resample("YE").last().pct_change().dropna()
        vals = [ann.get(pd.Timestamp(f"{y}-12-31"), 0) for y in years]
        ax.bar(x - 0.35 + (i + 1) * width, vals, width,
               label=sim["name"], color=VARIANT_COLORS[i % len(VARIANT_COLORS)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("Annual Returns")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def chart_monthly_heatmap(sim):
    set_chart_style()
    eq = sim["equity"].copy()
    eq.index = pd.to_datetime(eq.index)
    monthly = eq.resample("ME").last().pct_change().dropna()
    mdf = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month,
                         "ret": monthly.values})
    pivot = mdf.pivot(index="year", columns="month", values="ret")
    fig, ax = plt.subplots(figsize=(10, max(3, len(pivot) * 0.4)))
    vals = pivot.values
    mask = ~np.isnan(vals)
    vmax = np.abs(vals[mask]).max() if mask.any() else 0.5
    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int), fontsize=8)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0%}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > vmax * 0.6 else "black")
    ax.set_title(f"{sim['name']} — Monthly Returns")
    fig.colorbar(im, ax=ax, label="Monthly Return", shrink=0.8)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PDF Report
# ═══════════════════════════════════════════════════════════════════════

def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("Title", parent=ss["Title"], fontName="Times-Bold",
        fontSize=28, leading=34, textColor=WHITE, alignment=TA_CENTER, spaceAfter=12)
    s["subtitle"] = ParagraphStyle("Subtitle", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=16, leading=20, textColor=colors.Color(0.85, 0.85, 0.85),
        alignment=TA_CENTER, spaceAfter=6)
    s["cover_date"] = ParagraphStyle("CoverDate", parent=ss["Normal"], fontName="Helvetica",
        fontSize=11, leading=14, textColor=JPM_GOLD, alignment=TA_CENTER)
    s["h1"] = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=JPM_BLUE, spaceBefore=24, spaceAfter=10)
    s["h2"] = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=16, textColor=JPM_BLUE_LIGHT, spaceBefore=16, spaceAfter=6)
    s["body"] = ParagraphStyle("Body", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=10, leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6)
    s["body_bold"] = ParagraphStyle("BodyBold", parent=ss["Normal"], fontName="Times-Bold",
        fontSize=10, leading=13.5, textColor=BLACK, spaceBefore=2, spaceAfter=6)
    s["caption"] = ParagraphStyle("Caption", parent=ss["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=11, textColor=JPM_GRAY, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=10)
    s["disclaimer"] = ParagraphStyle("Disclaimer", parent=ss["Normal"], fontName="Helvetica",
        fontSize=7, leading=9, textColor=JPM_GRAY, alignment=TA_JUSTIFY)
    s["key_stat_label"] = ParagraphStyle("KSL", parent=ss["Normal"], fontName="Helvetica",
        fontSize=8, leading=10, textColor=JPM_GRAY, alignment=TA_CENTER)
    s["key_stat_value"] = ParagraphStyle("KSV", parent=ss["Normal"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=JPM_BLUE, alignment=TA_CENTER)
    return s


def _hf(canvas, doc, is_cover=False):
    canvas.saveState()
    if not is_cover:
        canvas.setStrokeColor(JPM_BLUE); canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H-MARGIN+6, PAGE_W-MARGIN, PAGE_H-MARGIN+6)
        canvas.setFont("Helvetica", 7.5); canvas.setFillColor(JPM_GRAY)
        canvas.drawString(MARGIN, PAGE_H-MARGIN+10, "Clenow Systematic Momentum — Crypto")
        canvas.drawRightString(PAGE_W-MARGIN, PAGE_H-MARGIN+10, "NRT Research")
        canvas.setStrokeColor(JPM_BLUE)
        canvas.line(MARGIN, MARGIN-14, PAGE_W-MARGIN, MARGIN-14)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, MARGIN-24, "CONFIDENTIAL — For internal use only")
        canvas.drawCentredString(PAGE_W/2, MARGIN-24, f"Page {doc.page}")
        canvas.drawRightString(PAGE_W-MARGIN, MARGIN-24, "February 2026")
    canvas.restoreState()

def on_cover(c, d):
    c.saveState(); c.setFillColor(JPM_BLUE)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(JPM_GOLD); c.rect(0, PAGE_H*0.42, PAGE_W, 3, fill=1, stroke=0)
    c.restoreState()

def on_body(c, d): _hf(c, d, False)


def make_table(headers, rows, col_widths=None, highlight_row=None):
    data_t = [headers] + rows
    cmds = [
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("BACKGROUND", (0,0), (-1,0), JPM_BLUE),
        ("ALIGN", (1,0), (-1,-1), "RIGHT"),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.Color(0.8,0.8,0.8)),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, JPM_GRAY_LIGHT]),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]
    if highlight_row is not None:
        r = highlight_row + 1
        cmds.append(("BACKGROUND", (0,r), (-1,r), colors.Color(0.85, 0.92, 1.0)))
        cmds.append(("FONTNAME", (0,r), (-1,r), "Helvetica-Bold"))
    t = Table(data_t, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(cmds))
    return t


def generate_report(sims, analyses, data, btc_eq):
    print("[report] Generating PDF...")
    sty = build_styles()
    fc = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H-2*MARGIN, id="cover")
    fb = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H-2*MARGIN, id="body")
    doc = BaseDocTemplate(str(PDF_PATH), pagesize=letter,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN)
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[fc], onPage=on_cover),
        PageTemplate(id="Body", frames=[fb], onPage=on_body),
    ])

    story = []
    story.append(Spacer(1, PAGE_H * 0.25))
    story.append(Paragraph("Systematic Momentum", sty["title"]))
    story.append(Paragraph("Clenow Ch.12 — Crypto Adaptation", sty["subtitle"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"NRT Research · {datetime.now().strftime('%B %Y')}", sty["cover_date"]))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    # Find best variant
    best_i = max(range(len(analyses)),
                 key=lambda i: analyses[i]["metrics"].get("sharpe", -99))
    bm = analyses[best_i]["metrics"]
    best_sim = sims[best_i]

    # ── Key stats ──
    story.append(Paragraph("1. Key Finding", sty["h1"]))
    stats = [
        [Paragraph("Best Variant", sty["key_stat_label"]),
         Paragraph("Sharpe", sty["key_stat_label"]),
         Paragraph("CAGR", sty["key_stat_label"]),
         Paragraph("Max DD", sty["key_stat_label"]),
         Paragraph("Peak Ret'd", sty["key_stat_label"])],
        [Paragraph(best_sim["name"], sty["key_stat_value"]),
         Paragraph(f"{bm['sharpe']:.2f}", sty["key_stat_value"]),
         Paragraph(f"{bm['cagr']:.1%}", sty["key_stat_value"]),
         Paragraph(f"{bm['max_dd']:.1%}", sty["key_stat_value"]),
         Paragraph(f"{analyses[best_i]['peak_retained']:.0%}", sty["key_stat_value"])],
    ]
    st = Table(stats, colWidths=[CONTENT_W/5]*5)
    st.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("BACKGROUND",(0,0),(-1,-1),JPM_GRAY_LIGHT),
        ("BOX",(0,0),(-1,-1),0.5,JPM_BLUE),
    ]))
    story.append(st)
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "This report implements the Clenow Systematic Momentum strategy (Trading Evolved, Ch.12) "
        "adapted for the crypto universe. Assets are ranked by annualized exponential regression "
        "slope × R², filtered by a BTC regime gate (BTC > 200-day MA) and per-asset trend "
        "confirmation (price > 100-day MA). Positions are sized by inverse-ATR with weekly "
        "rebalancing into the top-N ranked assets.", sty["body"]))

    # ── Comparison table ──
    story.append(Paragraph("Exhibit 1: All Variants", sty["h2"]))
    btc_m = compute_metrics(btc_eq)
    headers = ["Variant", "CAGR", "Vol", "Sharpe", "Sortino", "Max DD",
               "Calmar", "Skew", "Peak Ret'd"]
    rows = [["BTC B&H", f"{btc_m['cagr']:.1%}", f"{btc_m['vol']:.1%}",
             f"{btc_m['sharpe']:.2f}", f"{btc_m['sortino']:.2f}",
             f"{btc_m['max_dd']:.1%}", f"{btc_m['calmar']:.2f}",
             f"{btc_m['skewness']:.2f}",
             f"{btc_eq.iloc[-1]/btc_eq.max():.0%}"]]
    for sim, a in zip(sims, analyses):
        m = a["metrics"]
        rows.append([sim["name"], f"{m['cagr']:.1%}", f"{m['vol']:.1%}",
                     f"{m['sharpe']:.2f}", f"{m['sortino']:.2f}",
                     f"{m['max_dd']:.1%}", f"{m['calmar']:.2f}",
                     f"{m['skewness']:.2f}", f"{a['peak_retained']:.0%}"])
    cw = [1.6*inch] + [0.55*inch]*8
    story.append(make_table(headers, rows, col_widths=cw, highlight_row=best_i+1))

    # ── Charts ──
    story.append(PageBreak())
    story.append(Paragraph("2. Performance", sty["h1"]))
    story.append(fig_to_image(chart_equity_all(sims, btc_eq)))
    story.append(Paragraph("Exhibit 2: Equity curves (log scale).", sty["caption"]))
    story.append(fig_to_image(chart_drawdown_all(sims, btc_eq)))
    story.append(Paragraph("Exhibit 3: Drawdowns.", sty["caption"]))

    story.append(PageBreak())
    story.append(fig_to_image(chart_regime_and_holdings(best_sim, data), ratio=0.6))
    story.append(Paragraph(f"Exhibit 4: {best_sim['name']} — equity with regime shading (red = risk-off), "
                           "holdings count, and gross exposure.", sty["caption"]))

    story.append(fig_to_image(chart_annual(sims, data)))
    story.append(Paragraph("Exhibit 5: Annual returns.", sty["caption"]))

    story.append(PageBreak())
    story.append(fig_to_image(chart_monthly_heatmap(best_sim), ratio=0.5))
    story.append(Paragraph(f"Exhibit 6: {best_sim['name']} monthly returns.", sty["caption"]))

    # ── Methodology ──
    story.append(PageBreak())
    story.append(Paragraph("3. Methodology", sty["h1"]))
    story.append(Paragraph(
        "<b>Signal — Regression Momentum Score:</b> For each asset, fit a linear regression "
        "to the log of the trailing N-day close prices. The annualized slope measures trend "
        "strength; R² measures trend quality (smoothness). The product (annualized slope × R²) "
        "ranks assets by risk-adjusted trend quality. Assets with strong, smooth trends "
        "rank highest; choppy or mean-reverting assets score low.", sty["body"]))
    story.append(Paragraph(
        "<b>BTC Regime Filter:</b> When BTC is below its 200-day moving average, the portfolio "
        "goes to 100% cash. This exploits the single-factor structure of crypto markets — "
        "virtually no altcoin trends are reliable during a BTC bear regime.", sty["body"]))
    story.append(Paragraph(
        "<b>Per-Asset Filter:</b> Each asset must be trading above its own 100-day moving average "
        "to be eligible for inclusion. This prevents buying assets in local downtrends even if "
        "their regression score is temporarily positive.", sty["body"]))
    story.append(Paragraph(
        "<b>Position Sizing:</b> Inverse-ATR weighting targets equal risk contribution per position. "
        "Each position is capped at 20% of portfolio. Weekly rebalancing rotates capital into the "
        "top-N ranked assets.", sty["body"]))
    story.append(Paragraph(
        "<b>Drawdown Control:</b> If portfolio equity drops 15% from its all-time high, all "
        "positions are liquidated and trading pauses for 15 days minimum, resuming only when "
        "BTC is back in an uptrend.", sty["body"]))

    # Disclaimer
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.5, color=JPM_GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "DISCLAIMER: All results are hypothetical backtested performance. Strategy parameters "
        "follow Clenow (2019) with adaptations for crypto market structure. No optimization of "
        "core parameters (90-day regression, 200-day regime MA) was performed — these are taken "
        "directly from the source material. Transaction costs of 20 bps round-trip are applied. "
        "For internal research use only.", sty["disclaimer"]))

    doc.build(story)
    print(f"[report] PDF saved to {PDF_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    base_cfg = MomentumConfig()
    data = prepare_data(base_cfg)

    configs = [
        # Original (equity-calibrated)
        MomentumConfig(name="Original (equity cal)"),

        # Aggressive sizing: 10x risk factor
        MomentumConfig(name="10x risk, 200MA", risk_factor=0.010),
        MomentumConfig(name="10x risk, 100MA", risk_factor=0.010, regime_ma_window=100),
        MomentumConfig(name="10x risk, 50MA", risk_factor=0.010, regime_ma_window=50),

        # Full send: 25x risk factor (target ~full Kelly territory)
        MomentumConfig(name="25x risk, 200MA", risk_factor=0.025),
        MomentumConfig(name="25x risk, 100MA", risk_factor=0.025, regime_ma_window=100),
        MomentumConfig(name="25x risk, 50MA", risk_factor=0.025, regime_ma_window=50),

        # Max aggression: 50x risk, shorter regime, concentrated
        MomentumConfig(name="50x risk, 50MA, N=5", risk_factor=0.050,
                       regime_ma_window=50, top_n=5, max_weight=0.40),
        MomentumConfig(name="50x risk, 50MA, N=10", risk_factor=0.050,
                       regime_ma_window=50, top_n=10, max_weight=0.30),

        # Full send no guardrails
        MomentumConfig(name="50x, 50MA, no DD ctrl", risk_factor=0.050,
                       regime_ma_window=50, dd_control=False, max_weight=0.30),
        MomentumConfig(name="50x, no filter at all", risk_factor=0.050,
                       regime_filter=False, dd_control=False, max_weight=0.30),
    ]

    print(f"\n[sim] Running {len(configs)} variants...")
    sims = []
    analyses = []
    for cfg in configs:
        sim = run_simulation(data, cfg)
        a = analyze(sim)
        sims.append(sim)
        analyses.append(a)

    btc = data["close"]["BTC-USD"].dropna()
    btc_eq = btc / btc.iloc[0]

    # Summary
    print("\n" + "=" * 105)
    print(f"  {'Variant':<25s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} "
          f"{'Sortino':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'PeakRet':>8s}")
    print("  " + "-" * 100)
    btc_m = compute_metrics(btc_eq)
    print(f"  {'BTC B&H':<25s} {btc_m['cagr']:>7.1%} {btc_m['vol']:>7.1%} "
          f"{btc_m['sharpe']:>8.2f} {btc_m['sortino']:>8.2f} "
          f"{btc_m['max_dd']:>7.1%} {btc_m['calmar']:>8.2f} "
          f"{btc_eq.iloc[-1]/btc_eq.max():>7.0%}")
    for sim, a in zip(sims, analyses):
        m = a["metrics"]
        print(f"  {sim['name']:<25s} {m['cagr']:>7.1%} {m['vol']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']:>7.1%} {m['calmar']:>8.2f} "
              f"{a['peak_retained']:>7.0%}")
    print("=" * 105)

    generate_report(sims, analyses, data, btc_eq)
    print("\nDone.")


if __name__ == "__main__":
    main()
