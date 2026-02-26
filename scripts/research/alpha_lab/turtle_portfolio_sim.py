#!/usr/bin/env python3
"""
Turtle Trading — Full Portfolio Simulation
==========================================

Faithful Turtle system replication across the entire crypto universe.
Implements System 1 (20/10) + System 2 (55/20) with:
  - ATR-based position sizing (1% equity risk per unit)
  - Pyramiding up to 4 units per asset
  - Hard stops at 2 ATR from last entry
  - System 1 "last trade winner" filter
  - Portfolio-level unit caps

Generates a JP Morgan-styled PDF report.

Usage:
    python -m scripts.research.alpha_lab.turtle_portfolio_sim
"""
from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from common.data import load_daily_bars, ANN_FACTOR
from common.metrics import compute_metrics

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
    HRFlowable,
)

OUT_DIR = ROOT / "artifacts" / "research" / "alpha_lab"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "turtle_portfolio_report.pdf"

# ── Configuration ────────────────────────────────────────────────────
INITIAL_EQUITY = 1_000_000.0
SYS1_ENTRY, SYS1_EXIT = 20, 10
SYS2_ENTRY, SYS2_EXIT = 55, 20
ATR_PERIOD = 20
RISK_PER_UNIT = 0.01       # 1% of equity risked per ATR unit
ATR_STOP_MULT = 2.0
MAX_UNITS_PER_ASSET = 4
PYRAMID_ATR_MULT = 0.5     # add unit every 0.5 ATR above last entry
MAX_UNITS_TOTAL = 24       # total portfolio unit cap
MAX_UNITS_LONG = 15        # single-direction cap
COST_BPS = 20.0
MIN_HISTORY_DAYS = 365
MIN_ADV_USD = 500_000.0

# ── Chart palette ────────────────────────────────────────────────────
CB = "#0F2E5F"; CLB = "#3366A6"; CG = "#C2A154"; CR = "#B22222"
CGr = "#2E7D32"; CGy = "#888888"; CTEAL = "#006B6B"

JPM_BLUE = colors.Color(0.06, 0.18, 0.37)
JPM_BLUE_LIGHT = colors.Color(0.20, 0.40, 0.65)
JPM_GOLD = colors.Color(0.76, 0.63, 0.33)
JPM_GRAY = colors.Color(0.55, 0.55, 0.55)
JPM_GRAY_LIGHT = colors.Color(0.93, 0.93, 0.93)
WHITE = colors.white
BLACK = colors.black
PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Data preparation
# ═══════════════════════════════════════════════════════════════════════

def prepare_data():
    """Load daily bars, filter universe, pivot to wide format, precompute signals."""
    print("[sim] Loading daily bars...")
    panel = load_daily_bars(start="2017-01-01", end="2026-12-31")

    # Per-asset filtering: min history + min ADV
    sym_stats = panel.groupby("symbol").agg(
        n_days=("ts", "count"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
    )
    long_enough = sym_stats[sym_stats["n_days"] >= MIN_HISTORY_DAYS].index

    panel = panel[panel["symbol"].isin(long_enough)].copy()
    panel = panel.sort_values(["symbol", "ts"])

    # Rolling ADV filter
    panel["dollar_vol"] = panel["close"] * panel["volume"]
    panel["adv_20"] = panel.groupby("symbol")["dollar_vol"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    panel["in_universe"] = panel["adv_20"] >= MIN_ADV_USD
    panel.loc[panel["adv_20"].isna(), "in_universe"] = False

    symbols = sorted(panel[panel["in_universe"]].symbol.unique())
    print(f"[sim] Universe: {len(symbols)} assets with {MIN_HISTORY_DAYS}+ days and ${MIN_ADV_USD/1e6:.1f}M+ ADV")

    # Pivot to wide-format
    close_wide = panel.pivot_table(index="ts", columns="symbol", values="close")
    high_wide = panel.pivot_table(index="ts", columns="symbol", values="high")
    low_wide = panel.pivot_table(index="ts", columns="symbol", values="low")
    universe_wide = panel.pivot_table(index="ts", columns="symbol", values="in_universe")
    universe_wide = universe_wide.fillna(False).astype(bool)

    dates = sorted(close_wide.index)
    print(f"[sim] Date range: {dates[0]} to {dates[-1]} ({len(dates)} trading days)")

    # Precompute signals for all assets
    print("[sim] Precomputing channels and ATR...")
    atr_wide = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)
    s1_entry_wide = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)
    s1_exit_wide = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)
    s2_entry_wide = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)
    s2_exit_wide = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)

    for sym in symbols:
        if sym not in close_wide.columns:
            continue
        c = close_wide[sym].dropna()
        h = high_wide[sym].reindex(c.index)
        l = low_wide[sym].reindex(c.index)
        if len(c) < max(SYS2_ENTRY, ATR_PERIOD) + 5:
            continue

        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

        s1e = h.shift(1).rolling(SYS1_ENTRY, min_periods=SYS1_ENTRY).max()
        s1x = l.shift(1).rolling(SYS1_EXIT, min_periods=SYS1_EXIT).min()
        s2e = h.shift(1).rolling(SYS2_ENTRY, min_periods=SYS2_ENTRY).max()
        s2x = l.shift(1).rolling(SYS2_EXIT, min_periods=SYS2_EXIT).min()

        atr_wide.loc[c.index, sym] = atr
        s1_entry_wide.loc[c.index, sym] = s1e
        s1_exit_wide.loc[c.index, sym] = s1x
        s2_entry_wide.loc[c.index, sym] = s2e
        s2_exit_wide.loc[c.index, sym] = s2x

    return {
        "close": close_wide, "high": high_wide, "low": low_wide,
        "atr": atr_wide, "universe": universe_wide,
        "s1_entry": s1_entry_wide, "s1_exit": s1_exit_wide,
        "s2_entry": s2_entry_wide, "s2_exit": s2_exit_wide,
        "symbols": symbols, "dates": dates,
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Portfolio simulation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Unit:
    coins: float
    entry_price: float
    entry_date: object

@dataclass
class Position:
    symbol: str
    system: int
    units: list = field(default_factory=list)
    stop_price: float = 0.0

    @property
    def n_units(self): return len(self.units)
    @property
    def total_coins(self): return sum(u.coins for u in self.units)
    @property
    def last_entry_price(self): return self.units[-1].entry_price if self.units else 0
    @property
    def cost_basis(self): return sum(u.coins * u.entry_price for u in self.units)


@dataclass
class TradeRecord:
    symbol: str
    system: int
    entry_date: object
    exit_date: object
    entry_price: float
    exit_price: float
    n_units: int
    pnl_dollar: float
    pnl_pct: float
    exit_reason: str
    holding_days: int


def run_simulation(data: dict) -> dict:
    """Run the full Turtle portfolio simulation day by day."""
    close = data["close"]
    atr = data["atr"]
    universe = data["universe"]
    s1_entry = data["s1_entry"]
    s1_exit = data["s1_exit"]
    s2_entry = data["s2_entry"]
    s2_exit = data["s2_exit"]
    dates = data["dates"]

    cash = INITIAL_EQUITY
    positions: dict[str, Position] = {}
    last_s1_winner: dict[str, bool] = {}

    # Daily tracking
    daily_equity = []
    daily_cash = []
    daily_n_positions = []
    daily_n_units = []
    daily_gross_exposure = []
    daily_long_notional = []
    trades: list[TradeRecord] = []
    daily_assets_held = []

    cost_one_way = COST_BPS / 2.0 / 10_000

    print(f"[sim] Running simulation over {len(dates)} days...")
    t0 = time.time()

    for di, date in enumerate(dates):
        prices = close.loc[date].dropna()
        atrs = atr.loc[date].dropna()
        univ = universe.loc[date]

        # ── 1. Mark-to-market ──
        port_value = cash
        for sym, pos in positions.items():
            if sym in prices:
                port_value += pos.total_coins * prices[sym]

        # ── 2. Check exits (stops + channel exits) ──
        to_close = []
        for sym, pos in positions.items():
            if sym not in prices:
                continue
            price = prices[sym]

            # Hard stop
            if price <= pos.stop_price:
                to_close.append((sym, "stop"))
                continue

            # Channel exit
            if pos.system == 1:
                exit_level = s1_exit.loc[date].get(sym)
            else:
                exit_level = s2_exit.loc[date].get(sym)

            if pd.notna(exit_level) and price < exit_level:
                to_close.append((sym, "channel"))

        for sym, reason in to_close:
            pos = positions[sym]
            price = prices[sym]
            exit_value = pos.total_coins * price
            cost = exit_value * cost_one_way
            entry_avg = pos.cost_basis / pos.total_coins if pos.total_coins > 0 else price
            pnl = exit_value - pos.cost_basis - cost
            pnl_pct = (price / entry_avg - 1.0) if entry_avg > 0 else 0.0

            cash += exit_value - cost

            entry_dt = pos.units[0].entry_date
            trades.append(TradeRecord(
                symbol=sym, system=pos.system,
                entry_date=entry_dt, exit_date=date,
                entry_price=entry_avg, exit_price=price,
                n_units=pos.n_units, pnl_dollar=pnl, pnl_pct=pnl_pct,
                exit_reason=reason,
                holding_days=(date - entry_dt).days if hasattr(date - entry_dt, 'days') else 0,
            ))

            if pos.system == 1:
                last_s1_winner[sym] = pnl > 0
            del positions[sym]

        # ── 3. Check new entries ──
        total_units = sum(p.n_units for p in positions.values())

        # Sort available symbols by ATR (most volatile first, like original Turtles)
        candidates = []
        for sym in data["symbols"]:
            if sym in positions:
                continue
            if sym not in prices or not univ.get(sym, False):
                continue
            a = atrs.get(sym)
            if pd.isna(a) or a <= 0:
                continue
            s1e = s1_entry.loc[date].get(sym)
            s2e = s2_entry.loc[date].get(sym)
            if pd.isna(s1e) and pd.isna(s2e):
                continue
            candidates.append((sym, prices[sym], a, s1e, s2e))

        for sym, price, a, s1e, s2e in candidates:
            if total_units >= MAX_UNITS_TOTAL:
                break

            system = None
            # System 1: enter if breakout AND last S1 trade wasn't a winner
            if pd.notna(s1e) and price > s1e:
                if not last_s1_winner.get(sym, False):
                    system = 1
                else:
                    last_s1_winner[sym] = False  # reset filter after skip

            # System 2: always enter on breakout (catches filtered S1)
            if system is None and pd.notna(s2e) and price > s2e:
                system = 2

            if system is None:
                continue

            # Size: 1 unit = risk_per_unit * equity / ATR
            equity = port_value
            unit_coins = (equity * RISK_PER_UNIT) / a
            unit_cost = unit_coins * price
            trade_cost = unit_cost * cost_one_way

            if unit_cost + trade_cost > cash:
                continue

            cash -= unit_cost + trade_cost
            stop = price - ATR_STOP_MULT * a

            positions[sym] = Position(
                symbol=sym, system=system,
                units=[Unit(coins=unit_coins, entry_price=price, entry_date=date)],
                stop_price=stop,
            )
            total_units += 1

        # ── 4. Check pyramiding ──
        for sym, pos in list(positions.items()):
            if pos.n_units >= MAX_UNITS_PER_ASSET:
                continue
            if total_units >= MAX_UNITS_TOTAL:
                break
            if sym not in prices or sym not in atrs.index:
                continue

            price = prices[sym]
            a = atrs[sym]
            if pd.isna(a) or a <= 0:
                continue

            if price >= pos.last_entry_price + PYRAMID_ATR_MULT * a:
                equity = port_value
                unit_coins = (equity * RISK_PER_UNIT) / a
                unit_cost = unit_coins * price
                trade_cost = unit_cost * cost_one_way

                if unit_cost + trade_cost > cash:
                    continue

                cash -= unit_cost + trade_cost
                pos.units.append(Unit(coins=unit_coins, entry_price=price, entry_date=date))
                pos.stop_price = price - ATR_STOP_MULT * a
                total_units += 1

        # ── 5. Record daily state ──
        equity_eod = cash
        long_notional = 0.0
        held = []
        for sym, pos in positions.items():
            if sym in prices:
                mv = pos.total_coins * prices[sym]
                equity_eod += mv
                long_notional += mv
                held.append(sym)

        daily_equity.append(equity_eod)
        daily_cash.append(cash)
        daily_n_positions.append(len(positions))
        daily_n_units.append(sum(p.n_units for p in positions.values()))
        daily_gross_exposure.append(long_notional / equity_eod if equity_eod > 0 else 0)
        daily_long_notional.append(long_notional)
        daily_assets_held.append(held)

        if (di + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Day {di+1}/{len(dates)}: equity=${equity_eod:,.0f}, "
                  f"positions={len(positions)}, units={sum(p.n_units for p in positions.values())} "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"[sim] Simulation complete in {elapsed:.1f}s")
    print(f"[sim] Final equity: ${daily_equity[-1]:,.0f}")
    print(f"[sim] Total trades: {len(trades)}")

    eq_series = pd.Series(daily_equity, index=dates, name="equity")
    eq_norm = eq_series / eq_series.iloc[0]

    return {
        "equity": eq_series,
        "equity_norm": eq_norm,
        "cash": pd.Series(daily_cash, index=dates),
        "n_positions": pd.Series(daily_n_positions, index=dates),
        "n_units": pd.Series(daily_n_units, index=dates),
        "gross_exposure": pd.Series(daily_gross_exposure, index=dates),
        "long_notional": pd.Series(daily_long_notional, index=dates),
        "trades": trades,
        "assets_held": daily_assets_held,
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Analysis
# ═══════════════════════════════════════════════════════════════════════

def build_btc_benchmark(data: dict) -> pd.Series:
    """BTC buy-and-hold equity normalised to 1.0."""
    btc = data["close"]["BTC-USD"].dropna()
    return btc / btc.iloc[0]


def analyze(sim: dict, data: dict) -> dict:
    """Compute all analysis tables and metrics."""
    eq = sim["equity_norm"]
    metrics = compute_metrics(eq)

    btc_eq = build_btc_benchmark(data)
    btc_metrics = compute_metrics(btc_eq)

    # Trade analysis
    trades = sim["trades"]
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    if len(tdf) > 0:
        tdf["winner"] = tdf["pnl_dollar"] > 0
        n_trades = len(tdf)
        n_win = tdf["winner"].sum()
        win_rate = n_win / n_trades
        avg_win = tdf.loc[tdf["winner"], "pnl_dollar"].mean() if n_win > 0 else 0
        avg_loss = tdf.loc[~tdf["winner"], "pnl_dollar"].mean() if (n_trades - n_win) > 0 else 0
        profit_factor = abs(avg_win * n_win / (avg_loss * (n_trades - n_win))) if avg_loss != 0 else np.inf
        avg_hold = tdf["holding_days"].mean()
        best_trade = tdf.loc[tdf["pnl_dollar"].idxmax()]
        worst_trade = tdf.loc[tdf["pnl_dollar"].idxmin()]

        s1_trades = tdf[tdf["system"] == 1]
        s2_trades = tdf[tdf["system"] == 2]
        s1_pnl = s1_trades["pnl_dollar"].sum()
        s2_pnl = s2_trades["pnl_dollar"].sum()

        # Per-asset attribution
        asset_pnl = tdf.groupby("symbol")["pnl_dollar"].sum().sort_values(ascending=False)

        # Stop vs channel exit
        stop_exits = tdf[tdf["exit_reason"] == "stop"]
        channel_exits = tdf[tdf["exit_reason"] == "channel"]
    else:
        tdf = pd.DataFrame()
        n_trades = win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0
        s1_pnl = s2_pnl = 0
        asset_pnl = pd.Series(dtype=float)
        best_trade = worst_trade = None
        stop_exits = channel_exits = pd.DataFrame()

    # Annual returns
    eq_dollar = sim["equity"]
    eq_dollar.index = pd.to_datetime(eq_dollar.index)
    annual = eq_dollar.resample("YE").last().pct_change().dropna()
    annual.index = annual.index.year

    btc_close = data["close"]["BTC-USD"].dropna()
    btc_close.index = pd.to_datetime(btc_close.index)
    btc_annual = btc_close.resample("YE").last().pct_change().dropna()
    btc_annual.index = btc_annual.index.year

    return {
        "metrics": metrics,
        "btc_metrics": btc_metrics,
        "btc_eq": btc_eq,
        "tdf": tdf,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_hold": avg_hold,
        "s1_pnl": s1_pnl,
        "s2_pnl": s2_pnl,
        "asset_pnl": asset_pnl,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "stop_exits": stop_exits,
        "channel_exits": channel_exits,
        "annual": annual,
        "btc_annual": btc_annual,
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Charts
# ═══════════════════════════════════════════════════════════════════════

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


def chart_equity(sim, analysis):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    eq = sim["equity_norm"]
    btc = analysis["btc_eq"].reindex(eq.index, method="ffill")
    ax.plot(btc.index, btc.values, color=CGy, lw=1.0, label="BTC Buy & Hold", alpha=0.7)
    ax.plot(eq.index, eq.values, color=CB, lw=1.5, label="Turtle Portfolio")
    ax.set_yscale("log")
    ax.set_title("Turtle Portfolio vs BTC Buy & Hold — Equity (log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def chart_drawdowns(sim, analysis):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    eq = sim["equity_norm"]
    dd = eq / eq.cummax() - 1.0
    btc = analysis["btc_eq"].reindex(eq.index, method="ffill")
    btc_dd = btc / btc.cummax() - 1.0
    ax.fill_between(btc_dd.index, btc_dd.values, 0, alpha=0.1, color=CGy)
    ax.plot(btc_dd.index, btc_dd.values, color=CGy, lw=0.6, label="BTC B&H")
    ax.fill_between(dd.index, dd.values, 0, alpha=0.15, color=CB)
    ax.plot(dd.index, dd.values, color=CB, lw=0.8, label="Turtle Portfolio")
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def chart_positions(sim):
    set_chart_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.fill_between(sim["n_positions"].index, sim["n_positions"].values,
                     alpha=0.3, color=CLB, step="post")
    ax1.plot(sim["n_positions"].index, sim["n_positions"].values,
             color=CLB, lw=0.6, drawstyle="steps-post")
    ax1.set_title("Number of Open Positions")
    ax1.set_ylabel("Positions")

    ax2.fill_between(sim["gross_exposure"].index, sim["gross_exposure"].values,
                     alpha=0.3, color=CTEAL, step="post")
    ax2.plot(sim["gross_exposure"].index, sim["gross_exposure"].values,
             color=CTEAL, lw=0.6, drawstyle="steps-post")
    ax2.set_title("Gross Exposure (% of Equity)")
    ax2.set_ylabel("Exposure")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    return fig


def chart_asset_attribution(analysis, top_n=20):
    set_chart_style()
    pnl = analysis["asset_pnl"]
    if len(pnl) == 0:
        return None
    top = pd.concat([pnl.head(top_n // 2), pnl.tail(top_n // 2)]).drop_duplicates()
    top = top.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.3)))
    colors_bar = [CGr if v > 0 else CR for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(f"P&L Attribution — Top/Bottom {top_n // 2} Assets")
    ax.set_xlabel("Total P&L ($)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def chart_system_decomp(analysis):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    systems = ["System 1", "System 2"]
    vals = [analysis["s1_pnl"], analysis["s2_pnl"]]
    clrs = [CLB, CG]
    ax.bar(systems, vals, color=clrs, alpha=0.8, width=0.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("P&L by System")
    ax.set_ylabel("Total P&L ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def chart_annual_returns(analysis):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    annual = analysis["annual"]
    btc_annual = analysis["btc_annual"]
    common_years = annual.index.intersection(btc_annual.index)
    x = np.arange(len(common_years))
    w = 0.35
    ax.bar(x - w/2, [btc_annual.get(y, 0) for y in common_years],
           w, label="BTC B&H", color=CGy, alpha=0.7)
    ax.bar(x + w/2, [annual.get(y, 0) for y in common_years],
           w, label="Turtle Portfolio", color=CB, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(common_years.astype(int))
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("Annual Returns")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    fig.tight_layout()
    return fig


def chart_monthly_heatmap(sim):
    set_chart_style()
    eq = sim["equity"].copy()
    eq.index = pd.to_datetime(eq.index)
    monthly = eq.resample("ME").last().pct_change().dropna()
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="ret")

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
    ax.set_title("Monthly Returns Heatmap")
    fig.colorbar(im, ax=ax, label="Monthly Return", shrink=0.8)
    fig.tight_layout()
    return fig


def chart_units_over_time(sim):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(sim["n_units"].index, sim["n_units"].values,
                    alpha=0.3, color=CG, step="post")
    ax.plot(sim["n_units"].index, sim["n_units"].values,
            color=CG, lw=0.6, drawstyle="steps-post")
    ax.axhline(MAX_UNITS_TOTAL, color=CR, ls="--", lw=0.8, alpha=0.6, label=f"Cap ({MAX_UNITS_TOTAL})")
    ax.set_title("Total Units Deployed")
    ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PART 5: PDF report
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
        fontSize=11, leading=14, textColor=JPM_GOLD, alignment=TA_CENTER, spaceAfter=4)
    s["h1"] = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=JPM_BLUE, spaceBefore=24, spaceAfter=10)
    s["h2"] = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=16, textColor=JPM_BLUE_LIGHT, spaceBefore=16, spaceAfter=6)
    s["body"] = ParagraphStyle("Body", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=10, leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6)
    s["body_bold"] = ParagraphStyle("BodyBold", parent=ss["Normal"], fontName="Times-Bold",
        fontSize=10, leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6)
    s["body_italic"] = ParagraphStyle("BodyItalic", parent=ss["Normal"], fontName="Times-Italic",
        fontSize=10, leading=13.5, textColor=JPM_GRAY, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6)
    s["bullet"] = ParagraphStyle("Bullet", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=10, leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        leftIndent=18, bulletIndent=6, spaceBefore=1, spaceAfter=2)
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


def _header_footer(canvas, doc, is_cover=False):
    canvas.saveState()
    if not is_cover:
        canvas.setStrokeColor(JPM_BLUE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - MARGIN + 6, PAGE_W - MARGIN, PAGE_H - MARGIN + 6)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(JPM_GRAY)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 10,
                          "Turtle Trading — Full Portfolio Simulation")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 10, "NRT Research")
        canvas.setStrokeColor(JPM_BLUE)
        canvas.line(MARGIN, MARGIN - 14, PAGE_W - MARGIN, MARGIN - 14)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, MARGIN - 24, "CONFIDENTIAL — For internal use only")
        canvas.drawCentredString(PAGE_W / 2, MARGIN - 24, f"Page {doc.page}")
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 24, "February 2026")
    canvas.restoreState()


def on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(JPM_BLUE)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(JPM_GOLD)
    canvas.rect(0, PAGE_H * 0.42, PAGE_W, 3, fill=1, stroke=0)
    canvas.restoreState()

def on_body(canvas, doc):
    _header_footer(canvas, doc, is_cover=False)


def make_table(headers, rows, col_widths=None, highlight_row=None):
    data = [headers] + rows
    cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 0), (-1, 0), JPM_BLUE),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.8, 0.8, 0.8)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, JPM_GRAY_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if highlight_row is not None:
        r = highlight_row + 1
        cmds.append(("BACKGROUND", (0, r), (-1, r), colors.Color(0.85, 0.92, 1.0)))
        cmds.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(cmds))
    return t


def generate_report(sim, analysis, data):
    """Build the PDF report."""
    print("[report] Generating PDF...")
    sty = build_styles()

    frame_cover = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="cover")
    frame_body = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="body")
    doc = BaseDocTemplate(str(PDF_PATH), pagesize=letter,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN)
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[frame_cover], onPage=on_cover),
        PageTemplate(id="Body", frames=[frame_body], onPage=on_body),
    ])

    story = []
    m = analysis["metrics"]
    bm = analysis["btc_metrics"]

    # ── Cover ──
    story.append(Spacer(1, PAGE_H * 0.25))
    story.append(Paragraph("Turtle Trading", sty["title"]))
    story.append(Paragraph("Full Crypto Universe Portfolio Simulation", sty["subtitle"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"NRT Research · {datetime.now().strftime('%B %Y')}", sty["cover_date"]))
    story.append(Spacer(1, 40))
    story.append(Paragraph(
        f"System 1 ({SYS1_ENTRY}/{SYS1_EXIT}) + System 2 ({SYS2_ENTRY}/{SYS2_EXIT}) · "
        f"{len(data['symbols'])} assets · "
        f"{data['dates'][0].strftime('%Y') if hasattr(data['dates'][0], 'strftime') else data['dates'][0]} – "
        f"{data['dates'][-1].strftime('%Y') if hasattr(data['dates'][-1], 'strftime') else data['dates'][-1]}",
        sty["cover_date"]))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    # ── Executive Summary ──
    story.append(Paragraph("1. Executive Summary", sty["h1"]))

    # Key stats boxes
    stats_data = [
        [Paragraph("CAGR", sty["key_stat_label"]),
         Paragraph("Sharpe", sty["key_stat_label"]),
         Paragraph("Max Drawdown", sty["key_stat_label"]),
         Paragraph("Win Rate", sty["key_stat_label"]),
         Paragraph("Profit Factor", sty["key_stat_label"]),
         Paragraph("Total Trades", sty["key_stat_label"])],
        [Paragraph(f"{m['cagr']:.1%}", sty["key_stat_value"]),
         Paragraph(f"{m['sharpe']:.2f}", sty["key_stat_value"]),
         Paragraph(f"{m['max_dd']:.1%}", sty["key_stat_value"]),
         Paragraph(f"{analysis['win_rate']:.1%}", sty["key_stat_value"]),
         Paragraph(f"{analysis['profit_factor']:.2f}" if analysis['profit_factor'] != np.inf else "∞",
                   sty["key_stat_value"]),
         Paragraph(f"{analysis['n_trades']}", sty["key_stat_value"])],
    ]
    stats_table = Table(stats_data, colWidths=[CONTENT_W / 6] * 6)
    stats_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, -1), JPM_GRAY_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, JPM_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.Color(0.85, 0.85, 0.85)),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"This report presents a faithful replication of the complete Turtle Trading system "
        f"across {len(data['symbols'])} crypto assets from "
        f"{str(data['dates'][0])[:10]} to {str(data['dates'][-1])[:10]}. "
        f"The system combines System 1 (20-day/10-day breakout) and System 2 (55-day/20-day breakout) "
        f"with ATR-based position sizing (1% equity risk per unit), pyramiding up to 4 units per asset, "
        f"hard stops at 2× ATR, and portfolio-level unit caps ({MAX_UNITS_TOTAL} total, "
        f"{MAX_UNITS_LONG} single direction). Transaction costs of {COST_BPS:.0f} bps round-trip "
        f"are applied on all trades.", sty["body"]))

    # Comparison table
    story.append(Spacer(1, 6))
    story.append(Paragraph("Exhibit 1: Portfolio vs BTC Buy & Hold", sty["h2"]))
    comp_rows = [
        ["Turtle Portfolio", f"{m['cagr']:.1%}", f"{m['vol']:.1%}", f"{m['sharpe']:.2f}",
         f"{m['sortino']:.2f}", f"{m['max_dd']:.1%}", f"{m['calmar']:.2f}",
         f"{m['skewness']:.2f}"],
        ["BTC Buy & Hold", f"{bm['cagr']:.1%}", f"{bm['vol']:.1%}", f"{bm['sharpe']:.2f}",
         f"{bm['sortino']:.2f}", f"{bm['max_dd']:.1%}", f"{bm['calmar']:.2f}",
         f"{bm['skewness']:.2f}"],
    ]
    story.append(make_table(
        ["Strategy", "CAGR", "Vol", "Sharpe", "Sortino", "Max DD", "Calmar", "Skew"],
        comp_rows, highlight_row=0))
    story.append(Spacer(1, 10))

    # ── Equity + Drawdowns ──
    story.append(Paragraph("2. Performance", sty["h1"]))
    story.append(fig_to_image(chart_equity(sim, analysis)))
    story.append(Paragraph("Exhibit 2: Equity curves on log scale. The Turtle portfolio "
                           "compounds through systematic trend-following across the full "
                           "crypto universe.", sty["caption"]))
    story.append(fig_to_image(chart_drawdowns(sim, analysis)))
    story.append(Paragraph("Exhibit 3: Drawdown comparison. Turtle system aims to exit "
                           "before the worst of bear-market drawdowns.", sty["caption"]))

    # Annual returns
    story.append(PageBreak())
    story.append(fig_to_image(chart_annual_returns(analysis)))
    story.append(Paragraph("Exhibit 4: Year-by-year returns.", sty["caption"]))

    story.append(fig_to_image(chart_monthly_heatmap(sim), ratio=0.5))
    story.append(Paragraph("Exhibit 5: Monthly returns heatmap.", sty["caption"]))

    # ── Position management ──
    story.append(PageBreak())
    story.append(Paragraph("3. Position Management", sty["h1"]))
    story.append(fig_to_image(chart_positions(sim), ratio=0.5))
    story.append(Paragraph("Exhibit 6: Number of simultaneous positions and gross portfolio "
                           "exposure over time.", sty["caption"]))
    story.append(fig_to_image(chart_units_over_time(sim), ratio=0.35))
    story.append(Paragraph("Exhibit 7: Total Turtle units deployed. Dashed line shows "
                           f"the {MAX_UNITS_TOTAL}-unit portfolio cap.", sty["caption"]))

    avg_pos = sim["n_positions"].mean()
    max_pos = sim["n_positions"].max()
    avg_exp = sim["gross_exposure"].mean()
    max_exp = sim["gross_exposure"].max()
    story.append(Paragraph(
        f"The portfolio held an average of {avg_pos:.1f} positions (peak: {max_pos:.0f}) "
        f"with average gross exposure of {avg_exp:.1%} (peak: {max_exp:.1%}). "
        f"The ATR-based sizing naturally scales positions smaller for volatile assets, "
        f"keeping per-unit risk constant at {RISK_PER_UNIT:.0%} of equity.", sty["body"]))

    # ── Trade Analysis ──
    story.append(PageBreak())
    story.append(Paragraph("4. Trade Analysis", sty["h1"]))
    trade_stats = [
        ["Total trades", f"{analysis['n_trades']}"],
        ["Win rate", f"{analysis['win_rate']:.1%}"],
        ["Average winner", f"${analysis['avg_win']:,.0f}"],
        ["Average loser", f"${analysis['avg_loss']:,.0f}"],
        ["Profit factor", f"{analysis['profit_factor']:.2f}" if analysis['profit_factor'] != np.inf else "∞"],
        ["Avg holding period", f"{analysis['avg_hold']:.0f} days"],
        ["System 1 P&L", f"${analysis['s1_pnl']:,.0f}"],
        ["System 2 P&L", f"${analysis['s2_pnl']:,.0f}"],
    ]
    if analysis["best_trade"] is not None:
        bt = analysis["best_trade"]
        trade_stats.append(["Best trade", f"{bt['symbol']} +${bt['pnl_dollar']:,.0f} ({bt['pnl_pct']:+.1%})"])
        wt = analysis["worst_trade"]
        trade_stats.append(["Worst trade", f"{wt['symbol']} ${wt['pnl_dollar']:,.0f} ({wt['pnl_pct']:+.1%})"])

    story.append(Paragraph("Exhibit 8: Trade Statistics", sty["h2"]))
    story.append(make_table(["Metric", "Value"], trade_stats,
                            col_widths=[2.5 * inch, 4 * inch]))
    story.append(Spacer(1, 10))

    if len(analysis["tdf"]) > 0:
        tdf = analysis["tdf"]
        stop_n = len(analysis["stop_exits"])
        chan_n = len(analysis["channel_exits"])
        story.append(Paragraph(
            f"Of {analysis['n_trades']} total trades, {stop_n} ({stop_n/analysis['n_trades']:.0%}) "
            f"exited via hard stop and {chan_n} ({chan_n/analysis['n_trades']:.0%}) via channel exit. "
            f"System 1 generated {len(tdf[tdf.system==1])} trades; "
            f"System 2 generated {len(tdf[tdf.system==2])} trades.", sty["body"]))

    # System decomp
    story.append(fig_to_image(chart_system_decomp(analysis), width=4 * inch, ratio=0.65))
    story.append(Paragraph("Exhibit 9: P&L contribution by system.", sty["caption"]))

    # ── Asset Attribution ──
    story.append(PageBreak())
    story.append(Paragraph("5. Asset Attribution", sty["h1"]))
    attr_fig = chart_asset_attribution(analysis)
    if attr_fig:
        story.append(fig_to_image(attr_fig, ratio=0.6))
        story.append(Paragraph("Exhibit 10: P&L waterfall — top and bottom contributors.", sty["caption"]))

    # Top 10 assets table
    pnl = analysis["asset_pnl"]
    if len(pnl) > 0:
        top10 = pnl.head(10)
        top10_rows = [[sym, f"${v:,.0f}"] for sym, v in top10.items()]
        story.append(Paragraph("Exhibit 11: Top 10 P&L Contributors", sty["h2"]))
        story.append(make_table(["Asset", "Total P&L"], top10_rows,
                                col_widths=[3 * inch, 3.5 * inch]))

    # ── Implementation ──
    story.append(PageBreak())
    story.append(Paragraph("6. Implementation Details", sty["h1"]))
    params = [
        ["Initial equity", f"${INITIAL_EQUITY:,.0f}"],
        ["System 1 entry/exit", f"{SYS1_ENTRY}-day high / {SYS1_EXIT}-day low"],
        ["System 2 entry/exit", f"{SYS2_ENTRY}-day high / {SYS2_EXIT}-day low"],
        ["ATR period", f"{ATR_PERIOD} days"],
        ["Risk per unit", f"{RISK_PER_UNIT:.0%} of equity"],
        ["Hard stop", f"{ATR_STOP_MULT:.0f}× ATR below last entry"],
        ["Max units per asset", f"{MAX_UNITS_PER_ASSET}"],
        ["Pyramid interval", f"{PYRAMID_ATR_MULT}× ATR"],
        ["Max portfolio units", f"{MAX_UNITS_TOTAL}"],
        ["Max long units", f"{MAX_UNITS_LONG}"],
        ["Transaction costs", f"{COST_BPS:.0f} bps round-trip"],
        ["Universe filter", f"≥{MIN_HISTORY_DAYS} days history, ≥${MIN_ADV_USD/1e6:.1f}M 20-day ADV"],
        ["Universe size", f"{len(data['symbols'])} assets"],
        ["S1 last-trade filter", "Skip entry if last S1 trade was a winner"],
    ]
    story.append(make_table(["Parameter", "Value"], params,
                            col_widths=[2.5 * inch, 4 * inch]))

    # ── Disclaimer ──
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.5, color=JPM_GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "DISCLAIMER: All results are hypothetical backtested performance and do not represent "
        "actual trading. Past performance is not indicative of future results. This analysis "
        "does not account for slippage beyond the stated transaction cost assumption, market impact, "
        "funding costs, or operational risk. The Turtle system parameters are fixed at classical "
        "values and have not been optimized for crypto markets. For internal research use only.",
        sty["disclaimer"]))

    doc.build(story)
    print(f"[report] PDF saved to {PDF_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = prepare_data()
    sim = run_simulation(data)
    analysis = analyze(sim, data)

    m = analysis["metrics"]
    bm = analysis["btc_metrics"]
    print("\n" + "=" * 70)
    print("  TURTLE PORTFOLIO SIMULATION — RESULTS")
    print("=" * 70)
    print(f"  {'Metric':<25s} {'Turtle':>12s} {'BTC B&H':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for k, fmt in [("cagr", ".1%"), ("vol", ".1%"), ("sharpe", ".2f"),
                   ("sortino", ".2f"), ("max_dd", ".1%"), ("calmar", ".2f"),
                   ("skewness", ".2f"), ("kurtosis", ".2f")]:
        print(f"  {k:<25s} {m[k]:>12{fmt}} {bm[k]:>12{fmt}}")
    print(f"  {'Win rate':<25s} {analysis['win_rate']:>12.1%}")
    print(f"  {'Profit factor':<25s} {analysis['profit_factor']:>12.2f}")
    print(f"  {'Total trades':<25s} {analysis['n_trades']:>12d}")
    print(f"  {'Avg holding (days)':<25s} {analysis['avg_hold']:>12.0f}")
    print("=" * 70)

    generate_report(sim, analysis, data)
    print("\nDone.")


if __name__ == "__main__":
    main()
