#!/usr/bin/env python3
"""
Turtle Trading v2 — Gain Protection Overlays
=============================================

Runs the full Turtle simulation with configurable gain-protection mechanisms
and compares variants against the classic (unprotected) system.

Protection mechanisms:
  1. Portfolio drawdown control  — liquidate when equity drops X% from peak
  2. BTC regime filter           — only enter when BTC is trending up
  3. Concentrated universe       — only trade top-N assets by ADV

Usage:
    python -m scripts.research.alpha_lab.turtle_portfolio_v2
"""
from __future__ import annotations

import io
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
    HRFlowable,
)

OUT_DIR = ROOT / "artifacts" / "research" / "alpha_lab"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "turtle_portfolio_v2_report.pdf"

# ── Base Turtle config ───────────────────────────────────────────────
INITIAL_EQUITY = 1_000_000.0
SYS1_ENTRY, SYS1_EXIT = 20, 10
SYS2_ENTRY, SYS2_EXIT = 55, 20
ATR_PERIOD = 20
RISK_PER_UNIT = 0.01
ATR_STOP_MULT = 2.0
MAX_UNITS_PER_ASSET = 4
PYRAMID_ATR_MULT = 0.5
MAX_UNITS_TOTAL = 24
MAX_UNITS_LONG = 15
COST_BPS = 20.0
MIN_HISTORY_DAYS = 365
MIN_ADV_USD = 500_000.0

# ── Chart palette ────────────────────────────────────────────────────
CB = "#0F2E5F"; CLB = "#3366A6"; CG = "#C2A154"; CR = "#B22222"
CGr = "#2E7D32"; CGy = "#888888"; CTEAL = "#006B6B"
CORAL = "#E07040"; PURPLE = "#6B3FA0"

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
# Overlay configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OverlayConfig:
    name: str = "Classic"

    # Portfolio drawdown control
    dd_control: bool = False
    dd_threshold: float = -0.20      # liquidate at -20% from peak
    dd_cooldown_days: int = 20       # stay in cash this many days after trigger
    dd_require_btc_for_reentry: bool = True  # require BTC uptrend to re-enter

    # BTC regime filter
    btc_filter: bool = False
    btc_ma_window: int = 50          # BTC must be above N-day MA
    btc_exit_ma_window: int = 100    # liquidate all if BTC breaks below this MA

    # Concentrated universe
    concentrated: bool = False
    top_n: int = 15                  # only trade top N by ADV


# ═══════════════════════════════════════════════════════════════════════
# Data preparation (shared across variants)
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


def prepare_data():
    """Load daily bars, filter universe, precompute signals and BTC trend."""
    print("[data] Loading daily bars...")
    panel = load_daily_bars(start="2017-01-01", end="2026-12-31")

    sym_stats = panel.groupby("symbol").agg(n_days=("ts", "count"))
    long_enough = sym_stats[sym_stats["n_days"] >= MIN_HISTORY_DAYS].index
    panel = panel[panel["symbol"].isin(long_enough)].copy().sort_values(["symbol", "ts"])

    panel["dollar_vol"] = panel["close"] * panel["volume"]
    panel["adv_20"] = panel.groupby("symbol")["dollar_vol"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    panel["in_universe"] = panel["adv_20"] >= MIN_ADV_USD
    panel.loc[panel["adv_20"].isna(), "in_universe"] = False

    symbols = sorted(panel[panel["in_universe"]].symbol.unique())
    print(f"[data] Universe: {len(symbols)} assets")

    close_wide = panel.pivot_table(index="ts", columns="symbol", values="close")
    high_wide = panel.pivot_table(index="ts", columns="symbol", values="high")
    low_wide = panel.pivot_table(index="ts", columns="symbol", values="low")
    universe_wide = panel.pivot_table(index="ts", columns="symbol", values="in_universe").fillna(False).astype(bool)

    # ADV ranking for concentrated universe — reindex to match close_wide
    adv_wide = panel.pivot_table(index="ts", columns="symbol", values="adv_20")
    adv_wide = adv_wide.reindex(close_wide.index)

    dates = sorted(close_wide.index)
    print(f"[data] Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Precompute channels + ATR
    print("[data] Precomputing channels and ATR...")
    atr_wide = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)
    s1_entry_wide = atr_wide.copy()
    s1_exit_wide = atr_wide.copy()
    s2_entry_wide = atr_wide.copy()
    s2_exit_wide = atr_wide.copy()

    for sym in symbols:
        if sym not in close_wide.columns:
            continue
        c = close_wide[sym].dropna()
        h = high_wide[sym].reindex(c.index)
        lo = low_wide[sym].reindex(c.index)
        if len(c) < max(SYS2_ENTRY, ATR_PERIOD) + 5:
            continue

        prev_c = c.shift(1)
        tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
        atr_wide.loc[c.index, sym] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
        s1_entry_wide.loc[c.index, sym] = h.shift(1).rolling(SYS1_ENTRY, min_periods=SYS1_ENTRY).max()
        s1_exit_wide.loc[c.index, sym] = lo.shift(1).rolling(SYS1_EXIT, min_periods=SYS1_EXIT).min()
        s2_entry_wide.loc[c.index, sym] = h.shift(1).rolling(SYS2_ENTRY, min_periods=SYS2_ENTRY).max()
        s2_exit_wide.loc[c.index, sym] = lo.shift(1).rolling(SYS2_EXIT, min_periods=SYS2_EXIT).min()

    # BTC moving averages for regime filter
    btc = close_wide.get("BTC-USD", pd.Series(dtype=float)).dropna()
    btc_ma50 = btc.rolling(50, min_periods=50).mean()
    btc_ma100 = btc.rolling(100, min_periods=100).mean()

    return {
        "close": close_wide, "high": high_wide, "low": low_wide,
        "atr": atr_wide, "universe": universe_wide, "adv": adv_wide,
        "s1_entry": s1_entry_wide, "s1_exit": s1_exit_wide,
        "s2_entry": s2_entry_wide, "s2_exit": s2_exit_wide,
        "btc_close": btc, "btc_ma50": btc_ma50, "btc_ma100": btc_ma100,
        "symbols": symbols, "dates": dates,
    }


# ═══════════════════════════════════════════════════════════════════════
# Simulation engine (parameterized by OverlayConfig)
# ═══════════════════════════════════════════════════════════════════════

def run_simulation(data: dict, cfg: OverlayConfig) -> dict:
    """Run the Turtle simulation with the given overlay configuration."""
    close = data["close"]
    atr = data["atr"]
    universe = data["universe"]
    adv = data["adv"]
    s1_entry = data["s1_entry"]
    s1_exit = data["s1_exit"]
    s2_entry = data["s2_entry"]
    s2_exit = data["s2_exit"]
    btc_close = data["btc_close"]
    btc_ma50 = data["btc_ma50"]
    btc_ma100 = data["btc_ma100"]
    dates = data["dates"]

    cash = INITIAL_EQUITY
    positions: dict[str, Position] = {}
    last_s1_winner: dict[str, bool] = {}
    cost_one_way = COST_BPS / 2.0 / 10_000

    # Overlay state
    peak_equity = INITIAL_EQUITY
    dd_lockout_until = None
    in_dd_lockout = False

    # Daily tracking
    daily_equity = []
    daily_n_positions = []
    daily_n_units = []
    daily_gross_exposure = []
    daily_dd_lockout = []
    daily_btc_filter = []
    trades = []

    t0 = time.time()

    for di, date in enumerate(dates):
        prices = close.loc[date].dropna()
        atrs = atr.loc[date].dropna()
        univ = universe.loc[date]

        # ── Mark to market ──
        port_value = cash
        for sym, pos in positions.items():
            if sym in prices:
                port_value += pos.total_coins * prices[sym]

        # ── OVERLAY: Portfolio drawdown control ──
        dd_triggered = False
        if cfg.dd_control:
            if port_value > peak_equity:
                peak_equity = port_value
            current_dd = port_value / peak_equity - 1.0

            if in_dd_lockout:
                cooldown_expired = dd_lockout_until is not None and date >= dd_lockout_until
                if cooldown_expired:
                    # After cooldown, require BTC uptrend to resume (if configured)
                    btc_clear = True
                    if cfg.dd_require_btc_for_reentry:
                        bp = btc_close.get(date)
                        bma = btc_ma50.get(date)
                        btc_clear = pd.notna(bp) and pd.notna(bma) and bp > bma
                    if btc_clear:
                        in_dd_lockout = False
                        dd_lockout_until = None
                        peak_equity = port_value  # reset HWM to current level
            elif current_dd < cfg.dd_threshold:
                dd_triggered = True
                in_dd_lockout = True
                try:
                    lockout_end_idx = min(di + cfg.dd_cooldown_days, len(dates) - 1)
                    dd_lockout_until = dates[lockout_end_idx]
                except (IndexError, TypeError):
                    dd_lockout_until = None

        # ── OVERLAY: BTC regime filter ──
        btc_ok = True
        btc_exit_trigger = False
        if cfg.btc_filter:
            bp = btc_close.get(date)
            bma50 = btc_ma50.get(date)
            bma100 = btc_ma100.get(date)
            if pd.notna(bp) and pd.notna(bma50):
                btc_ok = bp > bma50
            else:
                btc_ok = False
            if pd.notna(bp) and pd.notna(bma100):
                if bp < bma100:
                    btc_exit_trigger = True

        # ── Force liquidation if DD triggered or BTC exit ──
        force_liquidate = dd_triggered or (cfg.btc_filter and btc_exit_trigger and len(positions) > 0)

        if force_liquidate:
            for sym in list(positions.keys()):
                pos = positions[sym]
                if sym not in prices:
                    continue
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
                    exit_reason="dd_control" if dd_triggered else "btc_filter",
                    holding_days=(date - entry_dt).days if hasattr(date - entry_dt, 'days') else 0,
                ))
                if pos.system == 1:
                    last_s1_winner[sym] = pnl > 0
            positions.clear()

        # ── Normal exits (stops + channels) ──
        if not force_liquidate:
            to_close = []
            for sym, pos in positions.items():
                if sym not in prices:
                    continue
                price = prices[sym]
                if price <= pos.stop_price:
                    to_close.append((sym, "stop"))
                    continue
                exit_level = (s1_exit if pos.system == 1 else s2_exit).loc[date].get(sym)
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

        # ── OVERLAY: Determine which assets are eligible ──
        can_enter = not in_dd_lockout
        if cfg.btc_filter:
            can_enter = can_enter and btc_ok

        # ── New entries ──
        if can_enter:
            total_units = sum(p.n_units for p in positions.values())

            # Build candidate list
            eligible_symbols = set(data["symbols"])
            if cfg.concentrated:
                day_adv = adv.loc[date].dropna().sort_values(ascending=False)
                day_adv = day_adv[day_adv.index.isin(eligible_symbols)]
                top_n_syms = set(day_adv.head(cfg.top_n).index)
                eligible_symbols = eligible_symbols & top_n_syms

            candidates = []
            for sym in eligible_symbols:
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
                if pd.notna(s1e) and price > s1e:
                    if not last_s1_winner.get(sym, False):
                        system = 1
                    else:
                        last_s1_winner[sym] = False
                if system is None and pd.notna(s2e) and price > s2e:
                    system = 2
                if system is None:
                    continue

                equity = port_value
                unit_coins = (equity * RISK_PER_UNIT) / a
                unit_cost = unit_coins * price
                trade_cost = unit_cost * cost_one_way
                if unit_cost + trade_cost > cash:
                    continue

                cash -= unit_cost + trade_cost
                positions[sym] = Position(
                    symbol=sym, system=system,
                    units=[Unit(coins=unit_coins, entry_price=price, entry_date=date)],
                    stop_price=price - ATR_STOP_MULT * a,
                )
                total_units += 1

            # Pyramiding
            for sym, pos in list(positions.items()):
                if pos.n_units >= MAX_UNITS_PER_ASSET or total_units >= MAX_UNITS_TOTAL:
                    continue
                if sym not in prices:
                    continue
                price = prices[sym]
                a = atrs.get(sym)
                if pd.isna(a) or a <= 0:
                    continue
                if price >= pos.last_entry_price + PYRAMID_ATR_MULT * a:
                    unit_coins = (port_value * RISK_PER_UNIT) / a
                    unit_cost = unit_coins * price
                    trade_cost = unit_cost * cost_one_way
                    if unit_cost + trade_cost > cash:
                        continue
                    cash -= unit_cost + trade_cost
                    pos.units.append(Unit(coins=unit_coins, entry_price=price, entry_date=date))
                    pos.stop_price = price - ATR_STOP_MULT * a
                    total_units += 1

        # ── Record ──
        equity_eod = cash
        long_notional = 0.0
        for sym, pos in positions.items():
            if sym in prices:
                equity_eod += pos.total_coins * prices[sym]
                long_notional += pos.total_coins * prices[sym]

        daily_equity.append(equity_eod)
        daily_n_positions.append(len(positions))
        daily_n_units.append(sum(p.n_units for p in positions.values()))
        daily_gross_exposure.append(long_notional / equity_eod if equity_eod > 0 else 0)
        daily_dd_lockout.append(in_dd_lockout)
        daily_btc_filter.append(not btc_ok if cfg.btc_filter else False)

    elapsed = time.time() - t0
    eq_series = pd.Series(daily_equity, index=dates, name="equity")
    eq_norm = eq_series / eq_series.iloc[0]

    print(f"  [{cfg.name}] Final=${daily_equity[-1]:,.0f}, "
          f"Trades={len(trades)}, {elapsed:.1f}s")

    return {
        "name": cfg.name,
        "equity": eq_series,
        "equity_norm": eq_norm,
        "n_positions": pd.Series(daily_n_positions, index=dates),
        "n_units": pd.Series(daily_n_units, index=dates),
        "gross_exposure": pd.Series(daily_gross_exposure, index=dates),
        "dd_lockout": pd.Series(daily_dd_lockout, index=dates),
        "btc_blocked": pd.Series(daily_btc_filter, index=dates),
        "trades": trades,
    }


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_variant(sim: dict) -> dict:
    eq = sim["equity_norm"]
    m = compute_metrics(eq)
    trades = sim["trades"]
    tdf = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()
    n_trades = len(tdf)
    n_win = int((tdf["pnl_dollar"] > 0).sum()) if n_trades > 0 else 0
    win_rate = n_win / n_trades if n_trades > 0 else 0
    avg_win = float(tdf.loc[tdf["pnl_dollar"] > 0, "pnl_dollar"].mean()) if n_win > 0 else 0
    avg_loss = float(tdf.loc[tdf["pnl_dollar"] <= 0, "pnl_dollar"].mean()) if (n_trades - n_win) > 0 else 0
    pf = abs(avg_win * n_win / (avg_loss * (n_trades - n_win))) if avg_loss != 0 and (n_trades - n_win) > 0 else np.inf
    avg_hold = float(tdf["holding_days"].mean()) if n_trades > 0 else 0

    # Time in cash from overlays
    dd_days = int(sim["dd_lockout"].sum())
    btc_days = int(sim["btc_blocked"].sum())
    total_days = len(sim["equity"])

    return {
        "metrics": m, "n_trades": n_trades, "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss, "profit_factor": pf,
        "avg_hold": avg_hold, "dd_lockout_days": dd_days,
        "btc_blocked_days": btc_days, "total_days": total_days,
        "tdf": tdf,
    }


# ═══════════════════════════════════════════════════════════════════════
# Charts
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


VARIANT_COLORS = [CGy, CR, CLB, CTEAL, CGr, CORAL, PURPLE, CB]


def chart_equity_comparison(sims, btc_eq):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(btc_eq.index, btc_eq.values, color=CG, lw=1.0, ls="--",
            alpha=0.6, label="BTC B&H")
    for i, sim in enumerate(sims):
        eq = sim["equity_norm"]
        c = VARIANT_COLORS[i % len(VARIANT_COLORS)]
        lw = 2.0 if i == len(sims) - 1 else 1.0
        ax.plot(eq.index, eq.values, color=c, lw=lw, label=sim["name"],
                alpha=1.0 if i == len(sims) - 1 else 0.7)
    ax.set_yscale("log")
    ax.set_title("Equity Curves — All Variants (log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def chart_drawdown_comparison(sims, btc_eq):
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    btc_dd = btc_eq / btc_eq.cummax() - 1.0
    ax.fill_between(btc_dd.index, btc_dd.values, 0, alpha=0.05, color=CG)
    ax.plot(btc_dd.index, btc_dd.values, color=CG, lw=0.5, ls="--",
            alpha=0.4, label="BTC B&H")
    for i, sim in enumerate(sims):
        eq = sim["equity_norm"]
        dd = eq / eq.cummax() - 1.0
        c = VARIANT_COLORS[i % len(VARIANT_COLORS)]
        if i == len(sims) - 1:
            ax.fill_between(dd.index, dd.values, 0, alpha=0.12, color=c)
        ax.plot(dd.index, dd.values, color=c, lw=0.8 if i < len(sims) - 1 else 1.5,
                label=sim["name"], alpha=0.6 if i < len(sims) - 1 else 1.0)
    ax.set_title("Drawdowns — All Variants")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    return fig


def chart_overlay_timeline(sim_protected, data):
    """Show when DD lockout and BTC filter were active."""
    set_chart_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1, 1]})

    eq = sim_protected["equity"]
    ax1.plot(eq.index, eq.values, color=CB, lw=1.0)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"{sim_protected['name']} — Overlay Activity Timeline")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

    dd = sim_protected["dd_lockout"]
    ax2.fill_between(dd.index, dd.astype(float).values, 0, alpha=0.4, color=CR, step="post")
    ax2.set_ylabel("DD Lockout")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Active", "Locked"])

    btc = sim_protected["btc_blocked"]
    ax3.fill_between(btc.index, btc.astype(float).values, 0, alpha=0.4, color=CORAL, step="post")
    ax3.set_ylabel("BTC Filter")
    ax3.set_ylim(-0.05, 1.15)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Clear", "Blocked"])
    ax3.set_xlabel("Date")

    fig.tight_layout()
    return fig


def chart_annual_comparison(sims, data):
    """Annual returns bar chart for all variants."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    n_variants = len(sims)
    width = 0.7 / n_variants

    btc = data["close"]["BTC-USD"].dropna()
    btc.index = pd.to_datetime(btc.index)
    btc_ann = btc.resample("YE").last().pct_change().dropna()
    years = sorted(set(btc_ann.index.year))
    x = np.arange(len(years))

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
    ax.set_title("Annual Returns — All Variants")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def chart_peak_preservation(sims):
    """Show peak equity and how much was preserved."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [s["name"] for s in sims]
    peaks = [s["equity"].max() for s in sims]
    finals = [s["equity"].iloc[-1] for s in sims]
    preserved = [f / p for f, p in zip(finals, peaks)]

    x = np.arange(len(names))
    ax.bar(x, [p / 1e6 for p in peaks], 0.35, label="Peak Equity",
           color=CLB, alpha=0.6)
    ax.bar(x + 0.35, [f / 1e6 for f in finals], 0.35, label="Final Equity",
           color=CB, alpha=0.8)
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Equity ($M)")
    ax.set_title("Peak vs Final Equity")
    ax.legend()

    for i, (p, f, pres) in enumerate(zip(peaks, finals, preserved)):
        ax.text(i + 0.175, max(p, f) / 1e6 + 0.2, f"{pres:.0%}\nretained",
                ha="center", va="bottom", fontsize=7, color=CB)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PDF report
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


def _header_footer(canvas, doc, is_cover=False):
    canvas.saveState()
    if not is_cover:
        canvas.setStrokeColor(JPM_BLUE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - MARGIN + 6, PAGE_W - MARGIN, PAGE_H - MARGIN + 6)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(JPM_GRAY)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 10,
                          "Turtle Trading v2 — Gain Protection Analysis")
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
    data_t = [headers] + rows
    cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 0), (-1, 0), JPM_BLUE),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.8, 0.8, 0.8)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, JPM_GRAY_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_row is not None:
        r = highlight_row + 1
        cmds.append(("BACKGROUND", (0, r), (-1, r), colors.Color(0.85, 0.92, 1.0)))
        cmds.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
    t = Table(data_t, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(cmds))
    return t


def generate_report(sims, analyses, data, btc_eq):
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

    # Cover
    story.append(Spacer(1, PAGE_H * 0.25))
    story.append(Paragraph("Turtle Trading v2", sty["title"]))
    story.append(Paragraph("Gain Protection Overlay Analysis", sty["subtitle"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"NRT Research · {datetime.now().strftime('%B %Y')}", sty["cover_date"]))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    # ── Key finding ──
    best_idx = max(range(len(analyses)),
                   key=lambda i: analyses[i]["metrics"].get("sharpe", -99))
    best = analyses[best_idx]
    best_sim = sims[best_idx]
    bm = best["metrics"]

    story.append(Paragraph("1. Key Finding", sty["h1"]))

    stats_data = [
        [Paragraph("Best Variant", sty["key_stat_label"]),
         Paragraph("Sharpe", sty["key_stat_label"]),
         Paragraph("CAGR", sty["key_stat_label"]),
         Paragraph("Max DD", sty["key_stat_label"]),
         Paragraph("Peak Retained", sty["key_stat_label"])],
        [Paragraph(best_sim["name"], sty["key_stat_value"]),
         Paragraph(f"{bm['sharpe']:.2f}", sty["key_stat_value"]),
         Paragraph(f"{bm['cagr']:.1%}", sty["key_stat_value"]),
         Paragraph(f"{bm['max_dd']:.1%}", sty["key_stat_value"]),
         Paragraph(f"{best_sim['equity'].iloc[-1] / best_sim['equity'].max():.0%}",
                   sty["key_stat_value"])],
    ]
    st = Table(stats_data, colWidths=[CONTENT_W / 5] * 5)
    st.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, -1), JPM_GRAY_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, JPM_BLUE),
    ]))
    story.append(st)
    story.append(Spacer(1, 12))

    classic_a = analyses[0]
    story.append(Paragraph(
        f"The classic Turtle system produced a {classic_a['metrics']['max_dd']:.0%} max drawdown, "
        f"with the portfolio running from $1M to ${sims[0]['equity'].max()/1e6:.0f}M before collapsing "
        f"to ${sims[0]['equity'].iloc[-1]/1e6:.1f}M — retaining only "
        f"{sims[0]['equity'].iloc[-1]/sims[0]['equity'].max():.0%} of its peak value. "
        f"The gain protection overlays address this by adding portfolio-level drawdown control, "
        f"a BTC regime filter, and universe concentration.", sty["body"]))

    # ── Comparison table ──
    story.append(Spacer(1, 8))
    story.append(Paragraph("Exhibit 1: All Variants — Full Comparison", sty["h2"]))

    headers = ["Variant", "CAGR", "Vol", "Sharpe", "Sortino", "Max DD",
               "Calmar", "Skew", "Trades", "Win%", "PF", "Peak Ret'd"]
    rows = []
    btc_m = compute_metrics(btc_eq)
    rows.append(["BTC B&H", f"{btc_m['cagr']:.1%}", f"{btc_m['vol']:.1%}",
                 f"{btc_m['sharpe']:.2f}", f"{btc_m['sortino']:.2f}",
                 f"{btc_m['max_dd']:.1%}", f"{btc_m['calmar']:.2f}",
                 f"{btc_m['skewness']:.2f}", "—", "—", "—",
                 f"{btc_eq.iloc[-1] / btc_eq.max():.0%}"])
    for sim, a in zip(sims, analyses):
        m = a["metrics"]
        pf_str = f"{a['profit_factor']:.2f}" if a['profit_factor'] != np.inf else "∞"
        rows.append([
            sim["name"], f"{m['cagr']:.1%}", f"{m['vol']:.1%}",
            f"{m['sharpe']:.2f}", f"{m['sortino']:.2f}",
            f"{m['max_dd']:.1%}", f"{m['calmar']:.2f}",
            f"{m['skewness']:.2f}", f"{a['n_trades']}",
            f"{a['win_rate']:.0%}", pf_str,
            f"{sim['equity'].iloc[-1] / sim['equity'].max():.0%}",
        ])

    cw = [1.6 * inch] + [0.45 * inch] * 11
    story.append(make_table(headers, rows, col_widths=cw, highlight_row=best_idx + 1))

    # ── Equity curves ──
    story.append(PageBreak())
    story.append(Paragraph("2. Performance Comparison", sty["h1"]))
    story.append(fig_to_image(chart_equity_comparison(sims, btc_eq)))
    story.append(Paragraph("Exhibit 2: Equity curves for all variants on log scale.", sty["caption"]))

    story.append(fig_to_image(chart_drawdown_comparison(sims, btc_eq)))
    story.append(Paragraph("Exhibit 3: Drawdown comparison.", sty["caption"]))

    # Peak preservation
    story.append(PageBreak())
    story.append(fig_to_image(chart_peak_preservation(sims), ratio=0.5))
    story.append(Paragraph("Exhibit 4: Peak equity vs final equity. The percentage retained "
                           "measures how much of the high-water mark the system preserved.", sty["caption"]))

    # Annual returns
    story.append(fig_to_image(chart_annual_comparison(sims, data)))
    story.append(Paragraph("Exhibit 5: Annual returns by variant.", sty["caption"]))

    # ── Overlay activity ──
    story.append(PageBreak())
    story.append(Paragraph("3. Overlay Activity", sty["h1"]))
    story.append(Paragraph(
        "The following timeline shows when each protection mechanism was active for the "
        "best-performing variant. Red shading indicates the portfolio was in drawdown lockout "
        "(all positions liquidated, no new entries). Orange indicates the BTC trend filter "
        "was blocking new entries.", sty["body"]))
    story.append(fig_to_image(chart_overlay_timeline(best_sim, data), ratio=0.6))
    story.append(Paragraph(
        f"Exhibit 6: Overlay activity for {best_sim['name']}. "
        f"DD lockout was active for {best['dd_lockout_days']} days "
        f"({best['dd_lockout_days']/best['total_days']:.0%} of trading days). "
        f"BTC filter blocked entries for {best['btc_blocked_days']} days "
        f"({best['btc_blocked_days']/best['total_days']:.0%}).", sty["caption"]))

    # ── Methodology ──
    story.append(PageBreak())
    story.append(Paragraph("4. Methodology", sty["h1"]))
    story.append(Paragraph("The following overlays were tested atop the classical Turtle system:", sty["body"]))
    story.append(Paragraph(
        "<b>Portfolio Drawdown Control:</b> When portfolio equity drops 20% from its all-time high, "
        "all positions are liquidated and the system enters a cooldown period (20 trading days minimum). "
        "The system resumes trading only after the cooldown expires AND the drawdown has recovered "
        "above -10%. This prevents re-entering into a continuing crash.", sty["body"]))
    story.append(Paragraph(
        "<b>BTC Regime Filter:</b> New entries require BTC to be above its 50-day moving average. "
        "If BTC falls below its 100-day moving average, ALL positions are liquidated regardless of "
        "per-asset signals. This exploits the empirical fact that crypto is a single-factor market — "
        "when BTC is in a downtrend, virtually no altcoin trend signal is reliable.", sty["body"]))
    story.append(Paragraph(
        "<b>Concentrated Universe:</b> Instead of trading all 250 qualifying assets, only the "
        "top N assets by 20-day average dollar volume are eligible for entry. This avoids "
        "death-by-a-thousand-cuts from breakout signals on illiquid altcoins.", sty["body"]))

    params = [
        ["DD threshold", "−20% from peak"],
        ["DD cooldown", "20 trading days"],
        ["DD re-entry", "Drawdown recovers above −10%"],
        ["BTC entry filter", "BTC close > 50-day MA"],
        ["BTC exit trigger", "BTC close < 100-day MA → liquidate all"],
        ["Concentrated top-N", "15 assets by 20-day ADV"],
    ]
    story.append(Paragraph("Exhibit 7: Overlay Parameters", sty["h2"]))
    story.append(make_table(["Parameter", "Value"], params,
                            col_widths=[2.5 * inch, 4 * inch]))

    # Disclaimer
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.5, color=JPM_GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "DISCLAIMER: All results are hypothetical backtested performance. The overlay parameters "
        "were selected based on judgment, not optimization — they represent reasonable institutional "
        "risk management, not curve-fitted thresholds. Past performance is not indicative of future "
        "results. For internal research use only.", sty["disclaimer"]))

    doc.build(story)
    print(f"[report] PDF saved to {PDF_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = prepare_data()

    configs = [
        OverlayConfig(name="Classic"),
        OverlayConfig(name="+ DD Control",
                      dd_control=True, dd_require_btc_for_reentry=False),
        OverlayConfig(name="+ BTC Filter",
                      btc_filter=True),
        OverlayConfig(name="+ Concentrated (N=15)",
                      concentrated=True, top_n=15),
        OverlayConfig(name="DD + BTC",
                      dd_control=True, btc_filter=True),
        OverlayConfig(name="DD + BTC + Top15",
                      dd_control=True, btc_filter=True,
                      concentrated=True, top_n=15),
        OverlayConfig(name="DD + BTC + Top10",
                      dd_control=True, btc_filter=True,
                      concentrated=True, top_n=10),
    ]

    print(f"\n[sim] Running {len(configs)} variants...")
    sims = []
    analyses = []
    for cfg in configs:
        sim = run_simulation(data, cfg)
        a = analyze_variant(sim)
        sims.append(sim)
        analyses.append(a)

    # BTC benchmark
    btc = data["close"]["BTC-USD"].dropna()
    btc_eq = btc / btc.iloc[0]

    # Summary
    print("\n" + "=" * 100)
    print(f"  {'Variant':<25s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>8s} "
          f"{'MaxDD':>8s} {'Calmar':>8s} {'PeakRet':>8s} {'Trades':>8s} {'Win%':>8s}")
    print("  " + "-" * 95)
    btc_m = compute_metrics(btc_eq)
    print(f"  {'BTC B&H':<25s} {btc_m['cagr']:>7.1%} {btc_m['vol']:>7.1%} "
          f"{btc_m['sharpe']:>8.2f} {btc_m['max_dd']:>7.1%} {btc_m['calmar']:>8.2f} "
          f"{btc_eq.iloc[-1]/btc_eq.max():>7.0%} {'—':>8s} {'—':>8s}")
    for sim, a in zip(sims, analyses):
        m = a["metrics"]
        peak_ret = sim["equity"].iloc[-1] / sim["equity"].max()
        print(f"  {sim['name']:<25s} {m['cagr']:>7.1%} {m['vol']:>7.1%} "
              f"{m['sharpe']:>8.2f} {m['max_dd']:>7.1%} {m['calmar']:>8.2f} "
              f"{peak_ret:>7.0%} {a['n_trades']:>8d} {a['win_rate']:>7.0%}")
    print("=" * 100)

    generate_report(sims, analyses, data, btc_eq)
    print("\nDone.")


if __name__ == "__main__":
    main()
