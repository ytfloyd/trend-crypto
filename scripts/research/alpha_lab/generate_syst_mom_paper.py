#!/usr/bin/env python3
"""
NRT Alternative Thinking 2026 Issue 2
======================================
Systematic Momentum in Digital Assets: What Works, What Doesn't, and Why

AQR-styled research report synthesising the Turtle breakout and Clenow
regression momentum results across the full crypto universe.

v2 — incorporates research desk feedback:
  • BTC-timing passive benchmark (the critical null hypothesis)
  • Survivorship bias disclosure
  • Standardised BTC filter (50/100d for both systems)
  • Transaction cost sensitivity (20/50/100 bps)
  • Rolling 12-month Sharpe
  • Parameter sensitivity heatmaps (DD threshold × BTC MA)
  • Annual return attribution
  • Clenow raw-slope variant (R² isolation test)
  • Methodology appendix
  • External-ready headers (no confidential markings)

Usage:
    python -m scripts.research.alpha_lab.generate_syst_mom_paper
"""
from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

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
    PageTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    KeepTogether,
)

OUT_DIR = ROOT / "artifacts" / "research" / "alpha_lab"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "nrt_alt_thinking_syst_mom_crypto.pdf"

# ── Palette ──────────────────────────────────────────────────────────
AQR_NAVY   = "#1B2A4A"
AQR_GREEN  = "#2D8C6E"
AQR_GREEN_LT = "#5CB88A"
AQR_GOLD   = "#C9A84C"
AQR_RED    = "#B03A2E"
AQR_GRAY   = "#6B7280"
AQR_GRAY_LT = "#E5E7EB"
AQR_BLUE_LT = "#3B82C4"
AQR_PURPLE = "#7C5BBF"

RL_NAVY   = colors.Color(0.106, 0.165, 0.290)
RL_GREEN  = colors.Color(0.176, 0.549, 0.431)
RL_GOLD   = colors.Color(0.788, 0.659, 0.298)
RL_GRAY   = colors.Color(0.42, 0.45, 0.50)
RL_GRAY_LT = colors.Color(0.90, 0.91, 0.92)
RL_RED    = colors.Color(0.69, 0.23, 0.18)
RL_PURPLE = colors.Color(0.486, 0.357, 0.749)
WHITE = colors.white; BLACK = colors.black
PAGE_W, PAGE_H = letter; MARGIN = 0.75 * inch; CONTENT_W = PAGE_W - 2 * MARGIN

# ── Config ───────────────────────────────────────────────────────────
INITIAL_EQUITY = 1_000_000.0
COST_BPS = 20.0
MIN_HISTORY_DAYS = 365
MIN_ADV_USD = 500_000.0
BTC_ENTRY_MA = 50       # Standardised across both systems
BTC_EXIT_MA = 100


# ═══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════

def prepare_data():
    print("[data] Loading daily bars...")
    panel = load_daily_bars(start="2017-01-01", end="2026-12-31")

    # Universe stats for survivorship disclosure
    all_syms = panel.symbol.unique()
    sym_stats = panel.groupby("symbol").agg(
        n_days=("ts", "count"),
        first_date=("ts", "min"),
        last_date=("ts", "max"),
    )
    max_date = panel.ts.max()
    dead_mask = sym_stats["last_date"] < max_date - pd.Timedelta(days=90)
    n_dead = int(dead_mask.sum())
    n_total = len(all_syms)

    long_enough = sym_stats[sym_stats["n_days"] >= MIN_HISTORY_DAYS].index
    panel = panel[panel["symbol"].isin(long_enough)].copy().sort_values(["symbol", "ts"])
    panel["dollar_vol"] = panel["close"] * panel["volume"]
    panel["adv_20"] = panel.groupby("symbol")["dollar_vol"].transform(
        lambda x: x.rolling(20, min_periods=20).mean())
    panel["in_universe"] = (panel["adv_20"] >= MIN_ADV_USD) & panel["adv_20"].notna()

    symbols = sorted(panel[panel["in_universe"]].symbol.unique())
    close_w = panel.pivot_table(index="ts", columns="symbol", values="close")
    high_w  = panel.pivot_table(index="ts", columns="symbol", values="high")
    low_w   = panel.pivot_table(index="ts", columns="symbol", values="low")
    univ_w  = panel.pivot_table(index="ts", columns="symbol", values="in_universe").fillna(False).astype(bool)
    univ_w  = univ_w.reindex(close_w.index).fillna(False)
    adv_w   = panel.pivot_table(index="ts", columns="symbol", values="adv_20").reindex(close_w.index)
    dates   = sorted(close_w.index)

    atr_w   = pd.DataFrame(index=close_w.index, columns=close_w.columns, dtype=float)
    ma100_w = atr_w.copy()
    s1e_w = atr_w.copy(); s1x_w = atr_w.copy()
    s2e_w = atr_w.copy(); s2x_w = atr_w.copy()
    for sym in symbols:
        if sym not in close_w.columns:
            continue
        c  = close_w[sym].dropna()
        h  = high_w[sym].reindex(c.index)
        lo = low_w[sym].reindex(c.index)
        if len(c) < 60:
            continue
        pc = c.shift(1)
        tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
        atr_w.loc[c.index, sym]   = tr.rolling(20, min_periods=20).mean()
        ma100_w.loc[c.index, sym] = c.rolling(100, min_periods=100).mean()
        s1e_w.loc[c.index, sym] = h.shift(1).rolling(20, min_periods=20).max()
        s1x_w.loc[c.index, sym] = lo.shift(1).rolling(10, min_periods=10).min()
        s2e_w.loc[c.index, sym] = h.shift(1).rolling(55, min_periods=55).max()
        s2x_w.loc[c.index, sym] = lo.shift(1).rolling(20, min_periods=20).min()

    btc = close_w.get("BTC-USD", pd.Series(dtype=float)).dropna()
    btc_mas = {}
    for w in [30, 40, 50, 60, 70, 100, 200]:
        btc_mas[w] = btc.rolling(w, min_periods=w).mean()

    print(f"[data] {len(symbols)} assets, {len(dates)} days, "
          f"{n_dead}/{n_total} dead symbols")
    return {
        "close": close_w, "high": high_w, "low": low_w, "atr": atr_w,
        "ma100": ma100_w, "universe": univ_w, "adv": adv_w,
        "s1_entry": s1e_w, "s1_exit": s1x_w, "s2_entry": s2e_w, "s2_exit": s2x_w,
        "btc": btc, "btc_mas": btc_mas,
        "symbols": symbols, "dates": dates,
        "n_total_syms": n_total, "n_dead_syms": n_dead,
        "n_universe": len(symbols),
    }


# ═══════════════════════════════════════════════════════════════════════
# BTC-TIMING PASSIVE BENCHMARK
# ═══════════════════════════════════════════════════════════════════════

def run_btc_timing_benchmark(data, *, top_n=10, cost_bps=COST_BPS,
                             label="BTC Timing (EW top-10)"):
    """Long equal-weight top-N alts by ADV when BTC > 50d MA; cash otherwise."""
    close = data["close"]; adv = data["adv"]; univ = data["universe"]
    btc_c = data["btc"]; btc_ma_e = data["btc_mas"][BTC_ENTRY_MA]
    btc_ma_x = data["btc_mas"][BTC_EXIT_MA]
    dates = data["dates"]; cow = cost_bps / 2 / 10_000
    cash = INITIAL_EQUITY; holdings = {}
    d_eq = []; d_np = []; d_ge = []; last_reb = -7

    for di, dt in enumerate(dates):
        pr = close.loc[dt].dropna()
        pv = cash
        for s, c in holdings.items():
            if s in pr:
                pv += c * pr[s]

        bp = btc_c.get(dt)
        bm_e = btc_ma_e.get(dt)
        bm_x = btc_ma_x.get(dt)
        btc_ok = pd.notna(bp) and pd.notna(bm_e) and bp > bm_e
        btc_exit = pd.notna(bp) and pd.notna(bm_x) and bp < bm_x

        if btc_exit and holdings:
            for s, c in holdings.items():
                if s in pr:
                    cash += c * pr[s] * (1 - cow)
            holdings.clear()

        if (di - last_reb) >= 7 and btc_ok:
            last_reb = di
            da = adv.loc[dt].dropna().sort_values(ascending=False)
            eligible = da[da.index.isin(set(data["symbols"]))]
            eligible = eligible[eligible.index.map(lambda s: univ.loc[dt].get(s, False))]
            eligible = eligible[eligible.index != "BTC-USD"]
            top = list(eligible.head(top_n).index)
            if not top:
                d_eq.append(pv); d_np.append(len(holdings))
                d_ge.append(sum(holdings.get(s, 0) * pr.get(s, 0) for s in holdings) / pv if pv > 0 else 0)
                continue

            te = cash
            for s, c in holdings.items():
                if s in pr:
                    te += c * pr[s]
            tw = 1.0 / len(top)
            target = {s: tw for s in top if s in pr and pr[s] > 0}

            for s in list(holdings):
                if s not in target:
                    if s in pr:
                        cash += holdings[s] * pr[s] * (1 - cow)
                    del holdings[s]

            te2 = cash
            for s, c in holdings.items():
                if s in pr:
                    te2 += c * pr[s]

            for s, w in target.items():
                tv = te2 * w
                cv = holdings.get(s, 0) * pr.get(s, 0)
                delta = tv - cv
                if abs(delta) < te2 * 0.005 or pr.get(s, 0) <= 0:
                    continue
                if delta > 0:
                    bc = delta * (1 + cow)
                    if bc > cash:
                        delta = cash / (1 + cow)
                        bc = delta * (1 + cow)
                    holdings[s] = holdings.get(s, 0) + delta / pr[s]
                    cash -= bc
                else:
                    cs = min(abs(delta) / pr[s], holdings.get(s, 0))
                    cash += cs * pr[s] * (1 - cow)
                    holdings[s] = holdings.get(s, 0) - cs
                    if holdings.get(s, 0) < 1e-10:
                        holdings.pop(s, None)

        eq_eod = cash; ln = 0
        for s, c in holdings.items():
            if s in pr:
                mv = c * pr[s]; eq_eod += mv; ln += mv
        d_eq.append(eq_eod); d_np.append(len(holdings))
        d_ge.append(ln / eq_eod if eq_eod > 0 else 0)

    eq = pd.Series(d_eq, index=dates); eqn = eq / eq.iloc[0]
    print(f"  [{label}] ${d_eq[-1]:,.0f}")
    return {"name": label, "equity": eq, "equity_norm": eqn,
            "n_pos": pd.Series(d_np, index=dates),
            "gross_exp": pd.Series(d_ge, index=dates)}


# ═══════════════════════════════════════════════════════════════════════
# TURTLE SIMULATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TUnit:
    coins: float; entry_price: float; entry_date: object

@dataclass
class TPos:
    symbol: str; system: int; units: list = field(default_factory=list); stop_price: float = 0.0
    @property
    def n_units(self): return len(self.units)
    @property
    def total_coins(self): return sum(u.coins for u in self.units)
    @property
    def cost_basis(self): return sum(u.coins * u.entry_price for u in self.units)
    @property
    def last_ep(self): return self.units[-1].entry_price if self.units else 0

def run_turtle(data, *, btc_filter=False, dd_control=False, concentrated=False,
               top_n=10, cost_bps=COST_BPS, btc_entry_ma=BTC_ENTRY_MA,
               btc_exit_ma=BTC_EXIT_MA, dd_threshold=-0.20, label="Turtle"):
    close = data["close"]; atr = data["atr"]; univ = data["universe"]; adv = data["adv"]
    s1e = data["s1_entry"]; s1x = data["s1_exit"]
    s2e = data["s2_entry"]; s2x = data["s2_exit"]
    btc_c = data["btc"]
    btc_m_entry = data["btc_mas"][btc_entry_ma]
    btc_m_exit  = data["btc_mas"][btc_exit_ma]
    dates = data["dates"]; cow = cost_bps / 2 / 10_000
    cash = INITIAL_EQUITY; positions = {}; last_s1w = {}
    peak_eq = INITIAL_EQUITY; in_lock = False; lock_until = -1
    d_eq = []; d_np = []; d_ge = []

    for di, dt in enumerate(dates):
        pr = close.loc[dt].dropna(); at = atr.loc[dt].dropna(); uv = univ.loc[dt]
        pv = cash
        for s, p in positions.items():
            if s in pr:
                pv += p.total_coins * pr[s]

        dd_act = False
        if dd_control:
            if pv > peak_eq:
                peak_eq = pv
            if in_lock:
                if di >= lock_until:
                    bp = btc_c.get(dt); bm = btc_m_entry.get(dt)
                    if pd.notna(bp) and pd.notna(bm) and bp > bm:
                        in_lock = False; peak_eq = pv
            elif pv / peak_eq - 1 < dd_threshold:
                in_lock = True; lock_until = di + 20
            dd_act = in_lock

        btc_ok = True; btc_exit = False
        if btc_filter:
            bp = btc_c.get(dt)
            bm_e = btc_m_entry.get(dt); bm_x = btc_m_exit.get(dt)
            btc_ok = pd.notna(bp) and pd.notna(bm_e) and bp > bm_e
            btc_exit = pd.notna(bp) and pd.notna(bm_x) and bp < bm_x

        force_liq = dd_act or (btc_filter and btc_exit and positions)
        if force_liq:
            for s in list(positions):
                if s in pr:
                    p = positions[s]; cash += p.total_coins * pr[s] * (1 - cow)
                    if p.system == 1:
                        last_s1w[s] = p.total_coins * pr[s] > p.cost_basis
            positions.clear()

        if not force_liq:
            tc = []
            for s, p in positions.items():
                if s not in pr:
                    continue
                if pr[s] <= p.stop_price:
                    tc.append(s); continue
                xl = (s1x if p.system == 1 else s2x).loc[dt].get(s)
                if pd.notna(xl) and pr[s] < xl:
                    tc.append(s)
            for s in tc:
                p = positions[s]
                if s in pr:
                    cash += p.total_coins * pr[s] * (1 - cow)
                    if p.system == 1:
                        last_s1w[s] = p.total_coins * pr[s] > p.cost_basis
                del positions[s]

        can_enter = not dd_act and (not btc_filter or btc_ok)
        if can_enter:
            tu = sum(p.n_units for p in positions.values())
            elig = set(data["symbols"])
            if concentrated:
                da = adv.loc[dt].dropna().sort_values(ascending=False)
                da = da[da.index.isin(elig)]
                elig = set(da.head(top_n).index)
            for sym in sorted(elig):
                if tu >= 24 or sym in positions:
                    continue
                if sym not in pr or not uv.get(sym, False):
                    continue
                a = at.get(sym)
                if pd.isna(a) or a <= 0:
                    continue
                s1 = s1e.loc[dt].get(sym); s2 = s2e.loc[dt].get(sym)
                sys_n = None
                if pd.notna(s1) and pr[sym] > s1:
                    if not last_s1w.get(sym, False):
                        sys_n = 1
                    else:
                        last_s1w[sym] = False
                if sys_n is None and pd.notna(s2) and pr[sym] > s2:
                    sys_n = 2
                if sys_n is None:
                    continue
                uc = (pv * 0.01) / a; cost = uc * pr[sym] * (1 + cow)
                if cost > cash:
                    continue
                cash -= cost
                positions[sym] = TPos(
                    symbol=sym, system=sys_n,
                    units=[TUnit(coins=uc, entry_price=pr[sym], entry_date=dt)],
                    stop_price=pr[sym] - 2 * a)
                tu += 1

            for sym, pos in list(positions.items()):
                if pos.n_units >= 4 or tu >= 24:
                    continue
                if sym not in pr:
                    continue
                a = at.get(sym)
                if pd.isna(a) or a <= 0:
                    continue
                if pr[sym] >= pos.last_ep + 0.5 * a:
                    uc = (pv * 0.01) / a; cost = uc * pr[sym] * (1 + cow)
                    if cost > cash:
                        continue
                    cash -= cost
                    pos.units.append(TUnit(coins=uc, entry_price=pr[sym], entry_date=dt))
                    pos.stop_price = pr[sym] - 2 * a; tu += 1

        eq_eod = cash; ln = 0
        for s, p in positions.items():
            if s in pr:
                mv = p.total_coins * pr[s]; eq_eod += mv; ln += mv
        d_eq.append(eq_eod); d_np.append(len(positions))
        d_ge.append(ln / eq_eod if eq_eod > 0 else 0)

    eq = pd.Series(d_eq, index=dates); eqn = eq / eq.iloc[0]
    print(f"  [{label}] ${d_eq[-1]:,.0f}")
    return {"name": label, "equity": eq, "equity_norm": eqn,
            "n_pos": pd.Series(d_np, index=dates),
            "gross_exp": pd.Series(d_ge, index=dates)}


# ═══════════════════════════════════════════════════════════════════════
# CLENOW SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def mom_score(prices, window=90, use_r_squared=True):
    if len(prices) < window:
        return np.nan
    p = prices.iloc[-window:]
    if p.min() <= 0 or p.isna().any():
        return np.nan
    x = np.arange(window, dtype=float)
    lp = np.log(p.values)
    sl, _, rv, _, _ = sp_stats.linregress(x, lp)
    ann_slope = np.exp(sl * 252) - 1
    if use_r_squared:
        return ann_slope * rv ** 2
    return ann_slope

def run_clenow(data, *, risk_factor=0.001, top_n=10,
               dd_control=True, regime_filter=True,
               use_r_squared=True, cost_bps=COST_BPS,
               label="Clenow"):
    close = data["close"]; atr_w = data["atr"]; ma100 = data["ma100"]
    univ = data["universe"]; dates = data["dates"]
    btc_c = data["btc"]
    btc_ma_e = data["btc_mas"][BTC_ENTRY_MA]
    btc_ma_x = data["btc_mas"][BTC_EXIT_MA]
    cow = cost_bps / 2 / 10_000
    cash = INITIAL_EQUITY; holdings = {}
    d_eq = []; d_np = []; d_ge = []
    peak_eq = INITIAL_EQUITY; in_lock = False; lock_until = -1; last_reb = -7

    for di, dt in enumerate(dates):
        pr = close.loc[dt].dropna(); at = atr_w.loc[dt].dropna()
        pv = cash
        for s, c in holdings.items():
            if s in pr:
                pv += c * pr[s]

        bp = btc_c.get(dt)
        bm_e = btc_ma_e.get(dt); bm_x = btc_ma_x.get(dt)
        ron = (not regime_filter) or (pd.notna(bp) and pd.notna(bm_e) and bp > bm_e)
        btc_exit = regime_filter and pd.notna(bp) and pd.notna(bm_x) and bp < bm_x

        dd_act = False
        if dd_control:
            if pv > peak_eq:
                peak_eq = pv
            if in_lock:
                if di >= lock_until and ron:
                    in_lock = False; peak_eq = pv
            elif pv / peak_eq - 1 < -0.15:
                in_lock = True; lock_until = di + 15
            dd_act = in_lock

        go_cash = (not ron) or dd_act or btc_exit
        if go_cash and holdings:
            for s, c in holdings.items():
                if s in pr:
                    cash += c * pr[s] * (1 - cow)
            holdings.clear()

        if (di - last_reb) >= 7 and not go_cash:
            last_reb = di
            elig = set()
            for sym in data["symbols"]:
                if sym not in pr or not univ.loc[dt].get(sym, False):
                    continue
                am = ma100.loc[dt].get(sym)
                if pd.notna(am) and pr[sym] > am:
                    elig.add(sym)
            scores = []
            for sym in elig:
                p = close[sym].loc[:dt].dropna()
                sc = mom_score(p, 90, use_r_squared=use_r_squared)
                if not np.isnan(sc):
                    scores.append((sym, sc))
            scores.sort(key=lambda x: x[1], reverse=True)
            target = {}
            for sym, sc in scores[:top_n]:
                a = at.get(sym); p = pr.get(sym)
                if pd.isna(a) or a <= 0 or pd.isna(p) or p <= 0:
                    continue
                w = risk_factor / (a / p) if a / p > 0 else 0
                target[sym] = min(w, 0.30)
            tw = sum(target.values())
            if tw > 1:
                target = {s: w / tw for s, w in target.items()}
            for s in list(holdings):
                if s not in target:
                    if s in pr:
                        cash += holdings[s] * pr[s] * (1 - cow)
                    del holdings[s]
            te = cash
            for s, c in holdings.items():
                if s in pr:
                    te += c * pr[s]
            for s, w in target.items():
                tv = te * w; cv = holdings.get(s, 0) * pr.get(s, 0); delta = tv - cv
                if abs(delta) < te * 0.005 or s not in pr or pr[s] <= 0:
                    continue
                if delta > 0:
                    bc = delta * (1 + cow)
                    if bc > cash:
                        delta = cash / (1 + cow); bc = delta * (1 + cow)
                    holdings[s] = holdings.get(s, 0) + delta / pr[s]; cash -= bc
                else:
                    cs = min(abs(delta) / pr[s], holdings.get(s, 0))
                    cash += cs * pr[s] * (1 - cow)
                    holdings[s] = holdings.get(s, 0) - cs
                    if holdings.get(s, 0) < 1e-10:
                        holdings.pop(s, None)

        eq_eod = cash; ln = 0
        for s, c in holdings.items():
            if s in pr:
                mv = c * pr[s]; eq_eod += mv; ln += mv
        d_eq.append(eq_eod); d_np.append(len(holdings))
        d_ge.append(ln / eq_eod if eq_eod > 0 else 0)

    eq = pd.Series(d_eq, index=dates); eqn = eq / eq.iloc[0]
    print(f"  [{label}] ${d_eq[-1]:,.0f}")
    return {"name": label, "equity": eq, "equity_norm": eqn,
            "n_pos": pd.Series(d_np, index=dates),
            "gross_exp": pd.Series(d_ge, index=dates)}


# ═══════════════════════════════════════════════════════════════════════
# ANALYTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def annual_returns(sim):
    eq = sim["equity_norm"]
    eq.index = pd.to_datetime(eq.index)
    years = sorted(eq.index.year.unique())
    out = {}
    for yr in years:
        chunk = eq[eq.index.year == yr]
        if len(chunk) < 20:
            continue
        out[yr] = float(chunk.iloc[-1] / chunk.iloc[0] - 1)
    return out

def rolling_sharpe(sim, window=365):
    eq = sim["equity_norm"]
    ret = eq.pct_change().dropna()
    mu = ret.rolling(window, min_periods=window).mean()
    sigma = ret.rolling(window, min_periods=window).std()
    rs = (mu / sigma) * np.sqrt(ANN_FACTOR)
    return rs.dropna()


# ═══════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.fontsize": 8, "legend.framealpha": 0.9,
    })

def fig2img(fig, width=6.5 * inch, ratio=0.55):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0); plt.close(fig)
    return Image(buf, width=width, height=width * ratio)

def chart_main_equity(turtle_sims, clenow_sims, btc_eq, bench):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(btc_eq.index, btc_eq.values, color=AQR_GRAY, lw=1, ls="--",
            alpha=0.5, label="BTC Buy & Hold")
    ax.plot(bench["equity_norm"].index, bench["equity_norm"].values,
            color=AQR_PURPLE, lw=1.4, ls="-.", label=bench["name"])
    clrs_t = [AQR_RED, AQR_BLUE_LT, AQR_GREEN_LT, AQR_GREEN]
    for i, s in enumerate(turtle_sims):
        ax.plot(s["equity_norm"].index, s["equity_norm"].values,
                color=clrs_t[i % len(clrs_t)],
                lw=1.5 if i == len(turtle_sims) - 1 else 0.9,
                label=s["name"], alpha=1 if i == len(turtle_sims) - 1 else 0.6)
    clrs_c = [AQR_GOLD, "#C07000"]
    for i, s in enumerate(clenow_sims[:2]):
        ax.plot(s["equity_norm"].index, s["equity_norm"].values,
                color=clrs_c[i % len(clrs_c)], lw=1, ls="--",
                label=s["name"], alpha=0.7)
    ax.set_yscale("log")
    ax.set_title("Exhibit 1: Equity Curves — All Strategies (log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig

def chart_drawdowns(sims_dict, btc_eq):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for lbl, eq, c in sims_dict:
        dd = eq / eq.cummax() - 1
        ax.fill_between(dd.index, dd.values, 0, alpha=0.08, color=c)
        ax.plot(dd.index, dd.values, color=c, lw=0.8, label=lbl)
    ax.set_title("Exhibit 2: Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return fig

def chart_overlay_decomp(sims, bench):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_s = [bench] + sims
    names = [s["name"] for s in all_s]
    sharpes = [compute_metrics(s["equity_norm"])["sharpe"] for s in all_s]
    clrs = []
    for i, sr in enumerate(sharpes):
        if i == 0:
            clrs.append(AQR_PURPLE)
        elif sr < 0.6:
            clrs.append(AQR_RED)
        elif sr < 1.0:
            clrs.append(AQR_BLUE_LT)
        else:
            clrs.append(AQR_GREEN)
    x = np.arange(len(names))
    ax.bar(x, sharpes, color=clrs, alpha=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Exhibit 3: Overlay Decomposition — Sharpe Ratio")
    ax.axhline(1.0, color=AQR_GRAY, ls="--", lw=0.8, alpha=0.5)
    fig.tight_layout()
    return fig

def chart_rolling_sharpe(sims_w_labels, window=365):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for lbl, sim, c in sims_w_labels:
        rs = rolling_sharpe(sim, window)
        ax.plot(rs.index, rs.values, color=c, lw=1, label=lbl)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"Exhibit 6: Rolling {window}-Day Sharpe Ratio")
    ax.set_ylabel("Annualised Sharpe")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig

def chart_annual_returns(sims_for_annual):
    set_style()
    years = sorted(set().union(*(annual_returns(s).keys() for _, s, _ in sims_for_annual)))
    n_strats = len(sims_for_annual)
    width = 0.8 / n_strats
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    for i, (lbl, sim, c) in enumerate(sims_for_annual):
        ar = annual_returns(sim)
        vals = [ar.get(y, 0) for y in years]
        ax.bar(x + i * width, vals, width=width, color=c, alpha=0.8, label=lbl)
    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels([str(y) for y in years], fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Exhibit 7: Calendar Year Returns by Strategy")
    ax.set_ylabel("Annual Return")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig

def chart_param_heatmap(grid_results, x_label, y_label, title):
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    x_vals = sorted(set(r[0] for r in grid_results))
    y_vals = sorted(set(r[1] for r in grid_results))
    mat = np.full((len(y_vals), len(x_vals)), np.nan)
    for xv, yv, sr in grid_results:
        xi = x_vals.index(xv); yi = y_vals.index(yv)
        mat[yi, xi] = sr
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.6,
                   origin="lower")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], fontsize=8)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:.0%}" for v in y_vals], fontsize=8)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_title(title)
    for yi in range(len(y_vals)):
        for xi in range(len(x_vals)):
            v = mat[yi, xi]
            if not np.isnan(v):
                ax.text(xi, yi, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if v < 0.7 or v > 1.4 else "black")
    plt.colorbar(im, ax=ax, label="Sharpe Ratio", shrink=0.8)
    fig.tight_layout()
    return fig

def chart_clenow_r2_test(sims):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [s["name"] for s in sims]
    sharpes = [compute_metrics(s["equity_norm"])["sharpe"] for s in sims]
    clrs = [AQR_GOLD if "R²" in s["name"] or "Slope" not in s["name"] else AQR_BLUE_LT
            for s in sims]
    x = np.arange(len(names))
    ax.bar(x, sharpes, color=clrs, alpha=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Exhibit 8: R² Isolation Test — Weighted vs Raw Slope")
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout()
    return fig

def chart_txcost_sensitivity(cost_results):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [r["name"] for r in cost_results]
    sharpes = [r["sharpe"] for r in cost_results]
    clrs = [AQR_GREEN if "All Overlays" in n else AQR_RED for n in names]
    x = np.arange(len(names))
    ax.bar(x, sharpes, color=clrs, alpha=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Exhibit 5: Transaction Cost Sensitivity (20 / 50 / 100 bps)")
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PDF STYLES
# ═══════════════════════════════════════════════════════════════════════

def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("Title", parent=ss["Title"], fontName="Times-Bold",
        fontSize=26, leading=32, textColor=WHITE, alignment=TA_CENTER, spaceAfter=10)
    s["subtitle"] = ParagraphStyle("Sub", parent=ss["Normal"], fontName="Times-Italic",
        fontSize=14, leading=18, textColor=colors.Color(0.85, 0.85, 0.85),
        alignment=TA_CENTER, spaceAfter=6)
    s["cover_date"] = ParagraphStyle("CD", parent=ss["Normal"], fontName="Helvetica",
        fontSize=10, leading=13, textColor=RL_GOLD, alignment=TA_CENTER)
    s["cover_authors"] = ParagraphStyle("CA", parent=ss["Normal"], fontName="Helvetica",
        fontSize=9, leading=12, textColor=colors.Color(0.75, 0.75, 0.75),
        alignment=TA_CENTER)
    s["h1"] = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=RL_NAVY, spaceBefore=22, spaceAfter=8)
    s["h2"] = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=RL_GREEN, spaceBefore=14, spaceAfter=5)
    s["body"] = ParagraphStyle("Body", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=10, leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6)
    s["body_bold"] = ParagraphStyle("BB", parent=ss["Normal"], fontName="Times-Bold",
        fontSize=10, leading=13.5, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=6)
    s["fn"] = ParagraphStyle("FN", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=7.5, leading=9.5, textColor=RL_GRAY, alignment=TA_JUSTIFY,
        spaceBefore=1, spaceAfter=1)
    s["caption"] = ParagraphStyle("Cap", parent=ss["Normal"], fontName="Helvetica",
        fontSize=8, leading=10.5, textColor=RL_GRAY, alignment=TA_JUSTIFY,
        spaceBefore=3, spaceAfter=8)
    s["exec_body"] = ParagraphStyle("EB", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=10.5, leading=14, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=2, spaceAfter=8)
    s["disclaimer"] = ParagraphStyle("Disc", parent=ss["Normal"], fontName="Helvetica",
        fontSize=7, leading=9, textColor=RL_GRAY, alignment=TA_JUSTIFY)
    s["app_body"] = ParagraphStyle("App", parent=ss["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=11, textColor=BLACK, alignment=TA_JUSTIFY,
        spaceBefore=1, spaceAfter=4)
    return s


def on_cover(c, d):
    c.saveState()
    c.setFillColor(RL_NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(RL_GREEN)
    c.rect(0, PAGE_H * 0.40, PAGE_W, 3, fill=1, stroke=0)
    c.restoreState()

def on_body(c, d):
    c.saveState()
    c.setStrokeColor(RL_NAVY); c.setLineWidth(0.5)
    c.line(MARGIN, PAGE_H - MARGIN + 6, PAGE_W - MARGIN, PAGE_H - MARGIN + 6)
    c.setFont("Helvetica", 7.5); c.setFillColor(RL_GRAY)
    c.drawString(MARGIN, PAGE_H - MARGIN + 10,
                 "NRT Alternative Thinking | 2026 Issue 2")
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 10,
                      "Systematic Momentum in Digital Assets")
    c.line(MARGIN, MARGIN - 14, PAGE_W - MARGIN, MARGIN - 14)
    c.drawCentredString(PAGE_W / 2, MARGIN - 24, f"Page {d.page}")
    c.restoreState()

def mktbl(headers, rows, col_widths=None, highlight_row=None):
    d = [headers] + rows
    cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BACKGROUND", (0, 0), (-1, 0), RL_NAVY),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.82, 0.82, 0.82)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, RL_GRAY_LT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if highlight_row is not None:
        r = highlight_row + 1
        cmds += [
            ("BACKGROUND", (0, r), (-1, r), colors.Color(0.85, 0.95, 0.90)),
            ("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"),
        ]
    t = Table(d, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(cmds))
    return t


# ═══════════════════════════════════════════════════════════════════════
# PDF REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def generate_report(turtle_sims, clenow_sims, clenow_r2_test, bench, data,
                    cost_results, grid_results, annual_tbl):
    print("[report] Building PDF...")
    sty = build_styles()
    btc = data["btc"]; btc_eq = btc / btc.iloc[0]
    btc_m = compute_metrics(btc_eq)
    bench_m = compute_metrics(bench["equity_norm"])

    all_m = []
    for s in turtle_sims + clenow_sims:
        m = compute_metrics(s["equity_norm"])
        m["name"] = s["name"]
        m["peak_ret"] = s["equity"].iloc[-1] / s["equity"].max()
        all_m.append(m)

    # Turtle All Overlays is turtle_sims[-1], i.e. all_m[len(turtle_sims)-1]
    t_best_m = all_m[len(turtle_sims) - 1]
    t_best_pr = turtle_sims[-1]["equity"].iloc[-1] / turtle_sims[-1]["equity"].max()

    fc = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="cover")
    fb = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="body")
    doc = BaseDocTemplate(str(PDF_PATH), pagesize=letter,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN)
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[fc], onPage=on_cover),
        PageTemplate(id="Body", frames=[fb], onPage=on_body),
    ])

    S = []

    # ── COVER ─────────────────────────────────────────────────────────
    S.append(Spacer(1, PAGE_H * 0.22))
    S.append(Paragraph("Systematic Momentum<br/>in Digital Assets", sty["title"]))
    S.append(Spacer(1, 8))
    S.append(Paragraph("What Works, What Doesn't, and Why", sty["subtitle"]))
    S.append(Spacer(1, 30))
    S.append(Paragraph("NRT Alternative Thinking · 2026 Issue 2", sty["cover_date"]))
    S.append(Spacer(1, 12))
    S.append(Paragraph("Quantitative Strategy Group", sty["cover_authors"]))
    S.append(NextPageTemplate("Body"))
    S.append(PageBreak())

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────
    S.append(Paragraph("Executive Summary", sty["h1"]))
    S.append(Paragraph(
        "In crypto, timing the regime is everything. We test two implementations of systematic "
        "momentum — channel breakout (Turtle Trading, 1983) and regression momentum "
        "(Clenow, 2019) — across 250 crypto assets over nine years (Jan 2017 – Feb 2026). "
        "Both produce competitive risk-adjusted returns. But a passive benchmark that simply "
        f"holds the top 10 altcoins during Bitcoin uptrends achieves a {bench_m['sharpe']:.2f} "
        "Sharpe ratio, <b>outperforming both active systems on a risk-adjusted basis</b>. "
        "The most important finding in this paper is what <i>doesn't</i> matter.<super>1</super>",
        sty["exec_body"]))
    S.append(Paragraph(
        "A single binary decision — be invested when Bitcoin is above its 50-day moving "
        "average, hold cash otherwise — dominates signal construction, position sizing, "
        "pyramiding, and stop-loss architecture combined. Active signals add value primarily "
        "through <i>drawdown compression</i>: the best Turtle overlay system achieves a "
        f"{t_best_m['max_dd']:.0%} maximum drawdown versus {bench_m['max_dd']:.0%} for the "
        f"passive benchmark, and retains {t_best_m['peak_ret']:.0%} of peak equity versus "
        f"{bench['equity'].iloc[-1] / bench['equity'].max():.0%} for the benchmark. "
        "For investors whose binding constraint is drawdown tolerance rather than Sharpe "
        "maximisation, the Turtle overlay system's tighter risk management justifies its "
        "incremental complexity.",
        sty["exec_body"]))
    S.append(Paragraph(
        "Two subsidiary findings are noteworthy. First, regression momentum (Clenow) "
        f"performs better than previously reported when given the same regime filter — a "
        f"{compute_metrics(clenow_sims[1]['equity_norm'])['sharpe']:.2f} Sharpe at "
        "moderate sizing — suggesting that the regime filter, not the signal type, was the "
        "binding constraint in earlier tests. Second, the R² trend-quality weighting, which "
        "we hypothesised would be harmful in crypto, is modestly <i>helpful</i>: stripping R² "
        "slightly degrades performance, contradicting our prior expectations.",
        sty["exec_body"]))
    S.append(Paragraph(
        "<super>1</super> See Moskowitz, Ooi, and Pedersen (2012), Asness, Moskowitz, and "
        "Pedersen (2013), and Hurst, Ooi, and Pedersen (2017) for evidence on systematic "
        "momentum across traditional asset classes.", sty["fn"]))

    # ── PART 1: THE QUESTION ──────────────────────────────────────────
    S.append(PageBreak())
    S.append(Paragraph("Part 1: The Question", sty["h1"]))
    S.append(Paragraph(
        "Can systematic momentum, which has delivered positive risk-adjusted returns across "
        "equities, fixed income, commodities, and currencies for over a century, be profitably "
        "applied to digital assets?", sty["body"]))
    S.append(Paragraph(
        "The question is non-trivial. Crypto markets differ from traditional asset classes in at "
        "least three structurally important ways: (1) extremely high volatility (60–100% annualised "
        "for major assets, vs. 15–25% for equities), (2) high cross-asset correlation (virtually all "
        "tokens move with Bitcoin during drawdowns), and (3) discontinuous, gappy price dynamics "
        "(large moves occur in hours, not weeks).", sty["body"]))
    S.append(Paragraph(
        "We test two canonical approaches. The first, <b>channel breakout</b>, enters when price "
        "exceeds a trailing N-day high and exits when it falls below a trailing M-day low — the "
        "Turtle Trading system (Faith, 2003), with ATR-based sizing, pyramiding, and hard stops. "
        "The second, <b>regression momentum</b>, ranks assets by the annualised slope of an "
        "exponential regression on log prices, weighted by R² (Clenow, 2019). Both are "
        "well-documented with decades of live track records in traditional markets.", sty["body"]))

    # ── DATA AND UNIVERSE ─────────────────────────────────────────────
    S.append(Paragraph("Part 2: Data and Universe Construction", sty["h1"]))
    S.append(Paragraph(
        "Our data consists of daily OHLCV bars for all USD-denominated spot pairs on Coinbase "
        f"Advanced from January 2017 through February 2026, comprising {data['n_total_syms']} "
        f"unique symbols. After filtering for minimum listing history (365 days) and liquidity "
        f"($500K 20-day average daily volume), {data['n_universe']} assets enter the tradeable "
        "universe at some point during the sample. Universe membership is evaluated dynamically — "
        "assets enter and exit as their volume crosses the threshold.", sty["body"]))
    S.append(Paragraph(
        "<b>Survivorship bias.</b> Our database includes all assets that have ever traded on Coinbase, "
        "including those that subsequently lost liquidity or were delisted from the exchange. Of the "
        f"{data['n_total_syms']} symbols in the raw database, {data['n_dead_syms']} have their "
        "last recorded bar more than 90 days before the sample end date. However, survivorship "
        "bias remains a concern for two reasons. First, tokens that were <i>never</i> listed on "
        "Coinbase — which includes most of the 2017–2019 altcoin graveyard (small ICO tokens "
        "that went to zero on smaller exchanges) — are absent from our universe. Second, Coinbase's "
        "listing standards impose a quality filter that excludes the most speculative assets. The net "
        "effect is that our CAGR estimates for the broad-universe simulations (\"Turtle Classic\") are "
        "likely overstated. The concentrated top-10 variants are largely immune to this bias, as the "
        "most liquid assets in crypto have persisted throughout the sample.", sty["body"]))
    S.append(Paragraph(
        "<b>BTC filter standardisation.</b> Both systems use identical Bitcoin regime parameters: "
        f"new entries require BTC above its {BTC_ENTRY_MA}-day moving average; all positions "
        f"are liquidated if BTC falls below its {BTC_EXIT_MA}-day moving average. Using a faster "
        "entry MA and slower exit MA creates a buffer that avoids whipsaws while still capturing "
        "the onset of downtrends.", sty["body"]))

    # ── PART 3: BREAKOUT MOMENTUM ─────────────────────────────────────
    S.append(PageBreak())
    S.append(Paragraph("Part 3: Breakout Momentum (Turtle Trading)", sty["h1"]))
    S.append(Paragraph(
        "We implement the complete Turtle system: System 1 (20-day entry, 10-day exit) and "
        "System 2 (55-day entry, 20-day exit), with 1% equity risk per ATR unit, pyramiding up "
        "to 4 units per asset, 2-ATR hard stops, and portfolio caps of 24 total units.",
        sty["body"]))
    S.append(Paragraph(
        "The unmodified results are disastrous. The system captures a peak of "
        f"${turtle_sims[0]['equity'].max() / 1e6:.0f}M — but subsequently gives back "
        f"{1 - turtle_sims[0]['equity'].iloc[-1] / turtle_sims[0]['equity'].max():.0%} "
        f"of that peak (max drawdown: {all_m[0]['max_dd']:.0%}). "
        "Crypto's high cross-asset correlation means that when the market turns, "
        "<i>every</i> breakout signal fails simultaneously.", sty["body"]))

    S.append(fig2img(chart_main_equity(turtle_sims, clenow_sims, btc_eq, bench)))
    S.append(Paragraph(
        "Source: NRT Research. Coinbase Advanced spot data, daily bars, Jan 2017 – Feb 2026. "
        f"{data['n_universe']} assets. Transaction costs of {COST_BPS:.0f} bps round-trip. "
        "All results are hypothetical backtested performance.", sty["caption"]))

    # ── PART 4: REGIME FILTER + BENCHMARK ─────────────────────────────
    S.append(Paragraph("Part 4: The Regime Filter and the Passive Benchmark", sty["h1"]))
    S.append(Paragraph(
        "The key insight is that crypto is effectively a single-factor market. When Bitcoin "
        "trends down, virtually no altcoin trend signal is reliable. This motivates a simple "
        f"overlay: use Bitcoin's {BTC_ENTRY_MA}-day MA as a portfolio-level regime switch.",
        sty["body"]))
    S.append(Paragraph(
        "But this raises a critical question: <b>how much of the return is the regime filter "
        "itself, and how much is the Turtle's breakout signals?</b> To answer this, we construct "
        "a passive benchmark: equal-weight the top 10 altcoins by ADV whenever BTC is above "
        f"its {BTC_ENTRY_MA}-day MA, and hold cash otherwise. No entry signals, no stops, "
        "no position sizing — just the macro filter plus passive exposure.", sty["body"]))

    S.append(fig2img(chart_overlay_decomp(turtle_sims, bench)))
    S.append(Paragraph(
        "Source: NRT Research. Each bar shows the Sharpe ratio. The purple bar is the passive "
        "BTC-timing benchmark. The gap between the benchmark and the full overlay system "
        "measures the alpha contributed by breakout signals, stops, and position sizing.",
        sty["caption"]))

    S.append(Paragraph(
        f"The results are humbling. The passive benchmark achieves a {bench_m['sharpe']:.2f} "
        f"Sharpe with a {bench_m['max_dd']:.0%} maximum drawdown and a {bench_m['cagr']:.0%} "
        f"CAGR. The full Turtle overlay system achieves a <i>lower</i> Sharpe of "
        f"{t_best_m['sharpe']:.2f}, a lower CAGR of {t_best_m['cagr']:.0%}, but a better "
        f"maximum drawdown of {t_best_m['max_dd']:.0%} and substantially better peak "
        f"retention ({t_best_m['peak_ret']:.0%} vs "
        f"{bench['equity'].iloc[-1] / bench['equity'].max():.0%}).", sty["body"]))
    S.append(Paragraph(
        "This is the paper's central finding: <b>the regime filter, not the breakout signal, "
        "is the dominant source of risk-adjusted return.</b> The Turtle's entry signals, ATR "
        "sizing, pyramiding, and hard stops collectively add complexity but do not improve "
        "Sharpe over naive equal-weight exposure. What the active system <i>does</i> provide "
        "is superior drawdown management — the breakout filter prevents entering at local "
        "highs during choppy BTC uptrends, and the hard stops enforce discipline when "
        "individual positions fail. Whether this drawdown compression justifies the system's "
        "complexity depends on the investor's loss aversion and mandate constraints.",
        sty["body"]))

    # Full comparison table
    S.append(PageBreak())
    S.append(Paragraph("Exhibit 4: Full Performance Comparison", sty["h2"]))
    hdrs = ["Strategy", "CAGR", "Vol", "Sharpe", "Sortino", "Max DD", "Calmar", "Peak Ret'd"]
    rows = []
    rows.append(["BTC Buy & Hold", f"{btc_m['cagr']:.1%}", f"{btc_m['vol']:.1%}",
                  f"{btc_m['sharpe']:.2f}", f"{btc_m['sortino']:.2f}",
                  f"{btc_m['max_dd']:.1%}", f"{btc_m['calmar']:.2f}",
                  f"{btc_eq.iloc[-1] / btc_eq.max():.0%}"])
    rows.append([bench["name"], f"{bench_m['cagr']:.1%}", f"{bench_m['vol']:.1%}",
                  f"{bench_m['sharpe']:.2f}", f"{bench_m['sortino']:.2f}",
                  f"{bench_m['max_dd']:.1%}", f"{bench_m['calmar']:.2f}",
                  f"{bench['equity'].iloc[-1] / bench['equity'].max():.0%}"])
    best_i = 0; best_sr = -99
    for i, m in enumerate(all_m):
        rows.append([m["name"], f"{m['cagr']:.1%}", f"{m['vol']:.1%}",
                     f"{m['sharpe']:.2f}", f"{m['sortino']:.2f}",
                     f"{m['max_dd']:.1%}", f"{m['calmar']:.2f}",
                     f"{m['peak_ret']:.0%}"])
        if m["sharpe"] > best_sr:
            best_sr = m["sharpe"]; best_i = i
    cw = [1.6 * inch] + [0.55 * inch] * 7
    S.append(mktbl(hdrs, rows, col_widths=cw, highlight_row=best_i + 2))

    # Drawdowns
    S.append(Spacer(1, 12))
    dd_sims = [
        ("BTC B&H", btc_eq, AQR_GRAY),
        (bench["name"], bench["equity_norm"], AQR_PURPLE),
        (turtle_sims[-1]["name"], turtle_sims[-1]["equity_norm"], AQR_GREEN),
    ]
    S.append(fig2img(chart_drawdowns(dd_sims, btc_eq)))
    S.append(Paragraph(
        "Source: NRT Research. Drawdowns from daily equity curves. The Turtle with "
        "overlays compresses bear-market drawdowns substantially relative to both BTC "
        "buy-and-hold and the passive benchmark.", sty["caption"]))

    # ── PART 5: ROBUSTNESS ────────────────────────────────────────────
    S.append(PageBreak())
    S.append(Paragraph("Part 5: Robustness", sty["h1"]))

    # Transaction cost sensitivity
    S.append(Paragraph("Transaction Cost Sensitivity", sty["h2"]))
    S.append(Paragraph(
        f"Our base case assumes {COST_BPS:.0f} bps round-trip transaction costs. This is "
        "reasonable for the concentrated top-10 universe (major pairs on institutional venues) "
        "but likely understated for the full 250-asset Turtle Classic, where illiquid altcoins "
        "in 2017–2019 had effective spreads of 50–200+ bps.", sty["body"]))
    S.append(fig2img(chart_txcost_sensitivity(cost_results)))
    S.append(Paragraph(
        "Source: NRT Research. Transaction costs are applied symmetrically on entry and exit. "
        "Higher costs disproportionately hurt the broad-universe Classic variant, which generates "
        "substantially more turnover from its larger number of positions.", sty["caption"]))

    # Parameter sensitivity heatmap
    S.append(Paragraph("Parameter Sensitivity", sty["h2"]))
    S.append(Paragraph(
        "A key concern with any backtested system is parameter sensitivity — are we sitting "
        "in a broad optimum or a narrow spike? We evaluate the Turtle with All Overlays across "
        "a grid of the two most important overlay parameters: drawdown threshold "
        "(10%–30%) and BTC entry MA window (30–70 days).", sty["body"]))
    S.append(fig2img(chart_param_heatmap(
        grid_results, "BTC Entry MA (days)", "DD Threshold",
        "Exhibit 5a: Parameter Sensitivity — Sharpe Ratio")))
    S.append(Paragraph(
        "Source: NRT Research. Each cell shows the Sharpe ratio for the Turtle + All Overlays "
        "configuration at that parameter combination. Top-10 concentration and 100-day BTC exit "
        "MA are held constant. The broad green region indicates parameter robustness — results "
        "are not dependent on a single point estimate.", sty["caption"]))

    # Rolling Sharpe
    S.append(PageBreak())
    S.append(Paragraph("Rolling Performance", sty["h2"]))
    rs_sims = [
        ("BTC B&H", {"equity_norm": btc_eq}, AQR_GRAY),
        (bench["name"], bench, AQR_PURPLE),
        (turtle_sims[-1]["name"], turtle_sims[-1], AQR_GREEN),
    ]
    S.append(fig2img(chart_rolling_sharpe(rs_sims)))
    S.append(Paragraph(
        "Source: NRT Research. 365-day rolling annualised Sharpe. The Turtle overlay system's "
        "edge is not concentrated in a single exceptional year — it maintains positive rolling "
        "Sharpe through most of the sample, though with meaningful variation.", sty["caption"]))

    # Annual returns
    S.append(Paragraph("Annual Return Attribution", sty["h2"]))
    ann_sims = [
        ("BTC B&H", {"equity_norm": btc_eq}, AQR_GRAY),
        (bench["name"], bench, AQR_PURPLE),
        (turtle_sims[-1]["name"], turtle_sims[-1], AQR_GREEN),
    ]
    S.append(fig2img(chart_annual_returns(ann_sims), ratio=0.48))
    S.append(Paragraph(
        "Source: NRT Research. Calendar year returns. Note: the passive benchmark's extreme "
        "2017 return reflects that the top-10 ADV universe in early 2017 was concentrated in "
        "a small number of assets (ETH, LTC, XRP and peers) that subsequently appreciated "
        "by orders of magnitude during the ICO bubble. This creates a partially favorable "
        "starting condition that is not representative of typical forward-looking conditions. "
        "From 2018 onward, the universe is more diversified and results are more indicative.",
        sty["caption"]))

    # Annual table
    S.append(Paragraph("Exhibit 7a: Annual Returns Table", sty["h2"]))
    a_hdrs = ["Year"] + [k for k in annual_tbl[0].keys() if k != "year"]
    a_rows = []
    for row in annual_tbl:
        a_rows.append([str(row["year"])] +
                      [f"{row[k]:.1%}" if not np.isnan(row[k]) else "—"
                       for k in a_hdrs[1:]])
    a_cw = [0.6 * inch] + [1.1 * inch] * (len(a_hdrs) - 1)
    S.append(mktbl(a_hdrs, a_rows, col_widths=a_cw))

    # ── PART 6: REGRESSION MOMENTUM ──────────────────────────────────
    S.append(PageBreak())
    S.append(Paragraph(
        "Part 6: Regression Momentum — Rehabilitation and the R² Test", sty["h1"]))
    S.append(Paragraph(
        "We implement the Clenow systematic momentum strategy: 90-day exponential regression "
        "on log prices, annualised slope × R² ranking, BTC regime filter (standardised to "
        f"{BTC_ENTRY_MA}/{BTC_EXIT_MA}-day to match the Turtle), per-asset trend confirmation "
        "(above 100-day MA), weekly rebalancing into the top 10 ranked assets, inverse-ATR "
        "position sizing.", sty["body"]))
    S.append(Paragraph(
        "With the standardised regime filter — and this is the key methodological point — "
        f"the Clenow system produces a {all_m[len(turtle_sims) + 1]['sharpe']:.2f} Sharpe "
        f"at 25× risk scaling and a {all_m[len(turtle_sims) + 1]['cagr']:.0%} CAGR. This is "
        "a substantial improvement over our earlier tests using a 200-day BTC MA, which "
        "kept the system in cash for extended periods and produced a 0.53 Sharpe. The lesson: "
        "the Clenow system was not broken — it was miscalibrated. The regime filter speed "
        "was the binding constraint, not the regression signal.", sty["body"]))
    S.append(Paragraph(
        "<b>The R² isolation test.</b> We hypothesised that the R² trend-quality weighting "
        "would be harmful in crypto, penalising explosive trends with high variance around "
        "the regression line. To test this, we run identical configurations with and without "
        "R² weighting. This is a clean, single-variable test.", sty["body"]))
    S.append(fig2img(chart_clenow_r2_test(clenow_r2_test)))
    S.append(Paragraph(
        "Source: NRT Research. All configurations use identical parameters (50× risk, top-10, "
        f"{BTC_ENTRY_MA}/{BTC_EXIT_MA}-day BTC filter, DD control). Only the ranking metric "
        "differs: slope × R² vs raw slope.", sty["caption"]))

    # Collect R² test metrics
    r2_m = [compute_metrics(s["equity_norm"]) for s in clenow_r2_test]
    S.append(Paragraph(
        f"The result contradicts our prior hypothesis. At 25× risk, Slope × R² achieves a "
        f"{r2_m[0]['sharpe']:.2f} Sharpe versus {r2_m[1]['sharpe']:.2f} for raw slope. At "
        f"50× risk, the gap is {r2_m[2]['sharpe']:.2f} vs {r2_m[3]['sharpe']:.2f}. <b>The R² "
        "weighting is modestly <i>helpful</i>, not harmful.</b> This suggests the R² filter "
        "is serving as a useful noise filter — assets with high slope but low R² are volatile "
        "trend-chasers, and penalising them slightly improves signal quality.",
        sty["body"]))
    S.append(Paragraph(
        "This finding reverses our earlier assessment (conducted with a slower 200-day BTC "
        "regime filter that kept the Clenow system in cash for extended periods). Once both "
        "systems share the same responsive 50/100-day regime filter, the Clenow system "
        f"performs competitively — a {r2_m[0]['sharpe']:.2f} Sharpe that is comparable to "
        f"the Turtle overlay's {t_best_m['sharpe']:.2f}. The prior apparent failure of "
        "regression momentum was primarily a regime filter miscalibration, not a "
        "fundamental signal deficiency.",
        sty["body"]))

    # ── PART 7: WHY THE REGIME FILTER DOMINATES ──────────────────────
    S.append(Paragraph(
        "Part 7: Why the Regime Filter Dominates", sty["h1"]))
    S.append(Paragraph(
        "Three structural features of crypto markets explain why the BTC regime filter "
        "captures the lion's share of risk-adjusted returns, leaving little residual alpha "
        "for active signal construction:", sty["body"]))
    S.append(Paragraph(
        "<b>1. Single-factor correlation structure.</b> In a market where all assets move with "
        "Bitcoin, the macro regime decision — invested or cash — dominates all other portfolio "
        "choices. During BTC downtrends, virtually no altcoin trend signal is reliable. During "
        "uptrends, nearly everything goes up. The cross-sectional question (\"which asset?\") "
        "is less important than the time-series question (\"be invested now, or not?\").",
        sty["body"]))
    S.append(Paragraph(
        "<b>2. Extreme return concentration.</b> The bulk of long-term crypto returns come from "
        "a small number of explosive bull runs (2017, late 2020–2021, 2024–2025). Being invested "
        "during these periods matters far more than entry precision. The passive benchmark "
        "captures these periods fully; active systems, by requiring additional confirmation "
        "(breakout above N-day high, regression rank, etc.), sacrifice some upside for "
        "downside protection that is only incrementally better than the macro filter alone.",
        sty["body"]))
    S.append(Paragraph(
        "<b>3. Noise in asset-level signals.</b> Both breakout and regression signals are "
        "applied to individual assets that are themselves 80–95% correlated to Bitcoin in "
        "drawdowns. The asset-level signal adds information only in the residual 5–20% of "
        "return variance not explained by the market factor. This leaves very little room for "
        "asset-level alpha, which is consistent with both active systems achieving Sharpe "
        "ratios comparable to — but not exceeding — the passive benchmark.", sty["body"]))

    # ── CONCLUDING THOUGHTS ───────────────────────────────────────────
    S.append(PageBreak())
    S.append(Paragraph("Concluding Thoughts", sty["h1"]))
    S.append(Paragraph(
        "Momentum works in crypto — but the alpha is in the regime, not the signal.", sty["body"]))
    S.append(Paragraph(
        "The most important result in this paper is the passive BTC-timing benchmark. A "
        "strategy that requires no signal construction, no position sizing, no stop-loss "
        "architecture — just equal-weighting the top 10 altcoins during BTC uptrends — "
        f"achieves a {bench_m['sharpe']:.2f} Sharpe ratio over nine years. Both active "
        "systems (breakout and regression) produce comparable but slightly lower Sharpe "
        "ratios, suggesting that the incremental value of sophisticated signal construction "
        "is small relative to getting the macro timing right.", sty["body"]))
    S.append(Paragraph(
        "This does not mean active signal systems are useless. The Turtle overlay system "
        f"achieves a {t_best_m['max_dd']:.0%} maximum drawdown versus {bench_m['max_dd']:.0%} "
        "for the passive benchmark — a meaningful improvement for investors with strict "
        "drawdown mandates. And the Clenow regression system, once properly calibrated with "
        f"the same regime filter, produces a {r2_m[0]['sharpe']:.2f} Sharpe with more "
        "predictable position sizes and lower turnover. The choice between systems depends "
        "on the investor's constraints: Sharpe maximisation favours the passive benchmark, "
        "drawdown minimisation favours the Turtle, and operational simplicity favours "
        "the Clenow.", sty["body"]))
    S.append(Paragraph(
        "One result deserves emphasis as an open research question: <b>the regime filter's "
        "effectiveness is contingent on crypto remaining a single-factor market.</b> If the "
        "asset class matures — if individual tokens develop independent return drivers "
        "(sector-specific fundamentals, differentiated cash flows, distinct institutional "
        "ownership bases) — the BTC filter's dominance may erode, and asset-level signals "
        "may become more valuable. Monitoring the cross-asset correlation structure over "
        "time is essential. A structural break toward lower correlations would be "
        "the most important signal for switching from macro-timing to asset-selection "
        "strategies.", sty["body"]))
    S.append(Paragraph(
        "Additional caveats: all results are in-sample on a single nine-year period that "
        "includes exceptional market conditions. The survivorship bias in our Coinbase-sourced "
        "universe likely overstates broad-universe results but has minimal impact on "
        "concentrated top-10 variants. Walk-forward validation and out-of-sample testing are "
        "necessary before any deployment. Transaction costs at institutional scale require "
        "further analysis, particularly for the Turtle's pyramiding and hard-stop mechanics "
        "which generate higher turnover than the Clenow or passive approaches.", sty["body"]))

    # ── APPENDIX A: METHODOLOGY ───────────────────────────────────────
    S.append(PageBreak())
    S.append(Paragraph("Appendix A: Implementation Details", sty["h1"]))
    S.append(Paragraph("<b>Turtle Trading System</b>", sty["h2"]))
    app = sty["app_body"]
    S.append(Paragraph(
        "<b>Data:</b> Daily OHLCV bars from Coinbase Advanced, Jan 2017 – Feb 2026. "
        f"{data['n_total_syms']} unique symbols.", app))
    S.append(Paragraph(
        "<b>Universe filter:</b> Minimum 365 days of history; 20-day average daily volume "
        "≥ $500K. Evaluated dynamically each day.", app))
    S.append(Paragraph(
        "<b>System 1:</b> Entry on close > 20-day highest high (shifted 1 day to avoid "
        "look-ahead). Exit on close < 10-day lowest low. System 1 filter: skip entry if "
        "last System 1 trade on this asset was a winner (\"last trade winner\" filter). "
        "If S1 is skipped, System 2 may still trigger.", app))
    S.append(Paragraph(
        "<b>System 2:</b> Entry on close > 55-day highest high. Exit on close < 20-day "
        "lowest low. No winner filter.", app))
    S.append(Paragraph(
        "<b>ATR calculation:</b> 20-day simple moving average of True Range.", app))
    S.append(Paragraph(
        "<b>Position sizing:</b> 1 unit = (1% of portfolio equity) / ATR. "
        "Dollar cost of 1 unit = unit_coins × price.", app))
    S.append(Paragraph(
        "<b>Pyramiding:</b> Up to 4 units per asset. Each pyramid requires price ≥ "
        "last entry price + 0.5 × ATR.", app))
    S.append(Paragraph(
        "<b>Hard stop:</b> 2 × ATR below the most recent entry price. "
        "Updated on each pyramid addition.", app))
    S.append(Paragraph(
        "<b>Portfolio limits:</b> Maximum 24 total units across all assets.", app))
    S.append(Paragraph(
        f"<b>BTC regime filter:</b> New entries require BTC > {BTC_ENTRY_MA}-day MA. "
        f"All positions liquidated if BTC < {BTC_EXIT_MA}-day MA. Asymmetric entry/exit "
        "MAs reduce whipsaw.", app))
    S.append(Paragraph(
        "<b>Drawdown control:</b> If portfolio equity falls 20% from all-time high, "
        "liquidate all positions. 20-day cooldown before re-entry. Re-entry requires "
        "BTC above entry MA. Peak equity resets on re-entry.", app))
    S.append(Paragraph(
        "<b>Concentrated universe:</b> Top N assets by rolling 20-day ADV.", app))
    S.append(Paragraph(
        f"<b>Transaction costs:</b> {COST_BPS:.0f} bps round-trip (applied half on "
        "entry, half on exit).", app))

    S.append(Spacer(1, 12))
    S.append(Paragraph("<b>Clenow Regression Momentum</b>", sty["h2"]))
    S.append(Paragraph(
        "<b>Signal:</b> For each asset, fit OLS regression on log(price) over trailing "
        "90-day window. Score = (exp(slope × 252) − 1) × R². Raw slope variant omits "
        "the R² multiplier.", app))
    S.append(Paragraph(
        "<b>Regime filter:</b> Same as Turtle — BTC above/below "
        f"{BTC_ENTRY_MA}/{BTC_EXIT_MA}-day MA.", app))
    S.append(Paragraph(
        "<b>Per-asset filter:</b> Asset price must be above its own 100-day MA.", app))
    S.append(Paragraph(
        "<b>Rebalancing:</b> Weekly (every 7 calendar days).", app))
    S.append(Paragraph(
        "<b>Position sizing:</b> Weight = risk_factor / (ATR / price). "
        "Maximum per-asset weight: 30%. Total weights capped at 100%.", app))
    S.append(Paragraph(
        "<b>Drawdown control:</b> Liquidate if equity falls 15% from peak. "
        "15-day cooldown. Re-entry requires BTC regime on.", app))
    S.append(Paragraph(
        "<b>Missing data:</b> Assets with NaN close prices on a given day are excluded "
        "from that day's eligible universe. No forward-filling of prices.", app))

    # ── REFERENCES ────────────────────────────────────────────────────
    S.append(PageBreak())
    S.append(Paragraph("References", sty["h2"]))
    refs = [
        "Asness, C., Moskowitz, T., and Pedersen, L. (2013). \"Value and Momentum "
        "Everywhere.\" <i>Journal of Finance</i>, 68(3), 929–985.",
        "Clenow, A. (2019). <i>Trading Evolved: Anyone Can Build Killer Trading "
        "Strategies in Python.</i> Clenow Media.",
        "Faith, C. (2003). <i>Way of the Turtle: The Secret Methods that Turned "
        "Ordinary People into Legendary Traders.</i> McGraw-Hill.",
        "Hurst, B., Ooi, Y.H., and Pedersen, L. (2017). \"A Century of Evidence on "
        "Trend-Following Investing.\" AQR Capital Management White Paper.",
        "Moskowitz, T., Ooi, Y.H., and Pedersen, L. (2012). \"Time Series Momentum.\" "
        "<i>Journal of Financial Economics</i>, 104(2), 228–250.",
    ]
    for r in refs:
        S.append(Paragraph(r, sty["fn"]))

    # ── DISCLAIMER ────────────────────────────────────────────────────
    S.append(Spacer(1, 25))
    S.append(HRFlowable(width="100%", thickness=0.5, color=RL_GRAY))
    S.append(Spacer(1, 6))
    S.append(Paragraph(
        "This document is provided for informational and educational purposes only and does "
        "not constitute investment advice, an offer, or solicitation. All performance data "
        "represents hypothetical backtested results and does not reflect actual trading. "
        "Past performance is not indicative of future results. The strategies described "
        "involve substantial risk of loss. Transaction costs, slippage, market impact, and "
        "other real-world frictions may differ materially from those modelled. The authors "
        "may hold positions in the assets discussed. No representation is made that any "
        "account will achieve results similar to those shown.", sty["disclaimer"]))

    doc.build(S)
    print(f"[report] PDF saved to {PDF_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = prepare_data()

    # ── Core simulations ──
    print("\n[sim] Passive BTC-timing benchmark...")
    bench = run_btc_timing_benchmark(data)

    print("\n[sim] Turtle variants...")
    turtle_sims = [
        run_turtle(data, label="Turtle Classic"),
        run_turtle(data, btc_filter=True, label="+ BTC Filter"),
        run_turtle(data, btc_filter=True, dd_control=True, label="+ BTC + DD Ctrl"),
        run_turtle(data, btc_filter=True, dd_control=True, concentrated=True,
                   top_n=10, label="Turtle All Overlays"),
    ]

    print("\n[sim] Clenow variants...")
    clenow_sims = [
        run_clenow(data, risk_factor=0.001, label="Clenow 1× (equity)"),
        run_clenow(data, risk_factor=0.025, label="Clenow 25×"),
        run_clenow(data, risk_factor=0.050, label="Clenow 50×"),
        run_clenow(data, risk_factor=0.050, dd_control=False,
                   regime_filter=False, label="Clenow 50× (naked)"),
    ]

    # ── R² isolation test ──
    print("\n[sim] R² isolation test...")
    clenow_r2_test = [
        run_clenow(data, risk_factor=0.025, use_r_squared=True,
                   label="Slope × R² (25×)"),
        run_clenow(data, risk_factor=0.025, use_r_squared=False,
                   label="Raw Slope (25×)"),
        run_clenow(data, risk_factor=0.050, use_r_squared=True,
                   label="Slope × R² (50×)"),
        run_clenow(data, risk_factor=0.050, use_r_squared=False,
                   label="Raw Slope (50×)"),
    ]

    # ── Transaction cost sensitivity ──
    print("\n[sim] Transaction cost sensitivity...")
    cost_results = []
    for bps in [20, 50, 100]:
        for label_base, kwargs in [
            ("Classic", dict(btc_filter=False, dd_control=False, concentrated=False)),
            ("All Overlays", dict(btc_filter=True, dd_control=True,
                                  concentrated=True, top_n=10)),
        ]:
            lbl = f"{label_base} @ {bps}bp"
            s = run_turtle(data, cost_bps=bps, label=lbl, **kwargs)
            m = compute_metrics(s["equity_norm"])
            cost_results.append({"name": lbl, "sharpe": m["sharpe"],
                                 "cagr": m["cagr"], "max_dd": m["max_dd"]})

    # ── Parameter sensitivity grid ──
    print("\n[sim] Parameter sensitivity grid...")
    grid_results = []
    dd_thresholds = [-0.10, -0.15, -0.20, -0.25, -0.30]
    btc_entry_mas = [30, 40, 50, 60, 70]
    for dd_t in dd_thresholds:
        for bma in btc_entry_mas:
            s = run_turtle(data, btc_filter=True, dd_control=True,
                           concentrated=True, top_n=10,
                           dd_threshold=dd_t, btc_entry_ma=bma,
                           label=f"DD={dd_t:.0%} BTC={bma}d")
            m = compute_metrics(s["equity_norm"])
            grid_results.append((bma, dd_t, m["sharpe"]))

    # ── Annual returns ──
    btc_eq_sim = {"equity_norm": data["btc"] / data["btc"].iloc[0]}
    key_sims = [
        ("BTC B&H", btc_eq_sim),
        (bench["name"], bench),
        (turtle_sims[-1]["name"], turtle_sims[-1]),
    ]
    all_years = sorted(set().union(*(annual_returns(s).keys() for _, s in key_sims)))
    annual_tbl = []
    for yr in all_years:
        row = {"year": yr}
        for lbl, sim in key_sims:
            ar = annual_returns(sim)
            row[lbl] = ar.get(yr, np.nan)
        annual_tbl.append(row)

    # ── Summary table ──
    btc_eq = data["btc"] / data["btc"].iloc[0]
    btc_m = compute_metrics(btc_eq)
    bench_m = compute_metrics(bench["equity_norm"])
    print("\n" + "=" * 110)
    hdr = f"  {'Strategy':<30s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'PeakRet':>8s}"
    print(hdr); print("  " + "-" * 65)
    print(f"  {'BTC B&H':<30s} {btc_m['cagr']:>7.1%} {btc_m['sharpe']:>8.2f} "
          f"{btc_m['max_dd']:>7.1%} {btc_eq.iloc[-1] / btc_eq.max():>7.0%}")
    print(f"  {bench['name']:<30s} {bench_m['cagr']:>7.1%} {bench_m['sharpe']:>8.2f} "
          f"{bench_m['max_dd']:>7.1%} "
          f"{bench['equity'].iloc[-1] / bench['equity'].max():>7.0%}")
    for s in turtle_sims + clenow_sims + clenow_r2_test:
        m = compute_metrics(s["equity_norm"])
        pr = s["equity"].iloc[-1] / s["equity"].max()
        print(f"  {s['name']:<30s} {m['cagr']:>7.1%} {m['sharpe']:>8.2f} "
              f"{m['max_dd']:>7.1%} {pr:>7.0%}")
    print("=" * 110)

    generate_report(turtle_sims, clenow_sims, clenow_r2_test, bench, data,
                    cost_results, grid_results, annual_tbl)
    print("\nDone.")


if __name__ == "__main__":
    main()
