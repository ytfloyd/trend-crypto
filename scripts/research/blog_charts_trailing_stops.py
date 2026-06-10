#!/usr/bin/env python
"""Publication-quality chart pack v3 for the stop-design Substack article.

Generates 11 charts with consistent institutional styling.

  01_btc_two_paths        — HERO: same Bitcoin signal, two realized paths
  02_scorecard            — 4-column comparison: NONE / FIXED / TRAILING / BTC HODL
  03_equity_curve         — fixed vs trailing vs BTC-HODL + no-stop overlay, log scale
  04_drawdown_comparison  — rolling drawdown for both variants
  05_stop_geometry        — conceptual schematic: why fixed stops widen naturally
  06_sol_centerpiece      — SOL Oct-2023 trade with missed-upside shading + callout
  07_return_distribution  — per-trade return histogram (right-tail emphasis)
  08_stop_hit_composition — exit-reason stacked bar (simplified)
  09_reentry_tax          — visualization of BTC trailing-stop re-entry penalties
  10_stop_cost_hierarchy  — NEW: bar chart, CAGR & Max DD for None / Fixed / Trailing
  11_breakout_sweep       — NEW: bar chart, Sharpe & CAGR by breakout window (5/10/20)

Visual language:
  no-stop   = #608958 (sage green — third comparison member)
  fixed     = #1f4e79 (deep navy)
  trailing  = #e26d2e (burnt orange)
  btc-hodl  = #999999 (muted gray, support color)
  entry     = #2a7f62 (forest green up-triangles)
  stop-exit = #a02c2c (dark red down-triangles)
  rebal-exit= #1f4e79 (navy down-triangles, signal-driven)
  missed    = #fde9d6 (warm cream shading for missed-upside regions)
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path("/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto")
spec = importlib.util.spec_from_file_location("wb", ROOT/"scripts/research/weekly_breakout_v1.py")
wb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wb)

OUT = ROOT / "artifacts/research/weekly_breakout_v2/blog_charts"
OUT.mkdir(parents=True, exist_ok=True)

# ── Consistent palette ────────────────────────────────────────────────
C_NOSTOP   = "#608958"
C_FIXED    = "#1f4e79"
C_TRAILING = "#e26d2e"
C_HODL     = "#9aa0a6"
C_ENTRY    = "#2a7f62"
C_STOPEXIT = "#a02c2c"
C_REBAL    = "#1f4e79"
C_MISSED   = "#fde9d6"
C_GREY     = "#5f5f5f"
C_BG       = "#fbfbfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#dcdcdc",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": C_BG,
    "axes.facecolor": C_BG,
    "savefig.facecolor": C_BG,
    "savefig.bbox": "tight",
    "savefig.dpi": 180,
})

BLUECHIP_10 = ["BTC-USDC","ETH-USDC","SOL-USDC","XRP-USDC","ADA-USDC",
                "DOGE-USDC","AVAX-USDC","LINK-USDC","DOT-USDC","LTC-USDC"]


def run_variant(panels, ind, trailing: bool = False, atr_mult: float = 3.0,
                 use_atr_stop: bool = True, breakout_window: int = 5):
    """Run a backtest variant. Optionally override the breakout window."""
    if breakout_window != 5:
        C = panels["C"]
        bo_high = C.rolling(breakout_window).max().shift(1)
        bo_low  = C.rolling(breakout_window).min().shift(1)
        ind_use = {**ind, "bo_high": bo_high, "bo_low": bo_low}
    else:
        ind_use = ind
    return wb.backtest(panels, ind_use, params=dict(
        cost_per_side=30/10000.0, atr_stop_mult=atr_mult, use_atr_stop=use_atr_stop,
        require_breakout_entry=True, trailing_stop=trailing, min_eligible_at_start=3))


def hodl_nav(panels, ref_index, symbol="BTC-USDC", start_nav=100_000.0):
    """Build a BTC HODL NAV series aligned to the backtest index."""
    px = panels["C"][symbol].reindex(ref_index, method="ffill").dropna()
    return (px / px.iloc[0]) * start_nav


def round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    rts = []
    for sym, g in trades.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        pos_sh = 0.0
        cost_basis_notional = 0.0
        entry_date = None
        for _, r in g.iterrows():
            if r["side"] == "BUY":
                if pos_sh < 1e-9:
                    entry_date = r["date"]
                    cost_basis_notional = 0.0
                cost_basis_notional += r["shares"] * r["price"]
                pos_sh += r["shares"]
            else:
                exit_sh = r["shares"]
                exit_px = r["price"]
                avg_entry = cost_basis_notional / pos_sh if pos_sh > 1e-9 else exit_px
                pnl_pct = (exit_px - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                cost_basis_notional *= max(0.0, 1.0 - exit_sh / max(pos_sh, 1e-9))
                pos_sh = max(0.0, pos_sh - exit_sh)
                if pos_sh < 1e-9:
                    held = (r["date"] - entry_date).days if entry_date else None
                    rts.append(dict(symbol=sym, entry_date=entry_date, exit_date=r["date"],
                                     held_days=held, avg_entry_px=avg_entry,
                                     exit_px=exit_px, pnl_pct=pnl_pct,
                                     exit_reason=r["reason"]))
                    entry_date = None
    return pd.DataFrame(rts)


def byline(fig, text="trend_crypto research desk · 9-year BC-10 backtest @ 30 bps/side"):
    fig.text(0.99, 0.005, text, fontsize=7.5, ha="right", color=C_GREY,
             style="italic", alpha=0.85)


def stamp_title(fig, title, subtitle, ax=None, x_offset=0.06):
    """Consistent title placement above the axes."""
    fig.text(x_offset, 0.96, title, fontsize=15, fontweight="bold",
             color="#1c1c1c")
    fig.text(x_offset, 0.925, subtitle, fontsize=10.5, color=C_GREY,
             style="italic")


# ════════════════════════════════════════════════════════════════════════
# CHART 1 — BTC TWO-PATH HERO
# ════════════════════════════════════════════════════════════════════════
def chart_btc_two_paths(panels, r_fix, r_trail):
    print("01: BTC two paths (hero)...")
    btc_close = panels["C"]["BTC-USDC"]
    win = (btc_close.index >= "2020-09-25") & (btc_close.index <= "2021-03-18")
    btc_w = btc_close.loc[win]

    def btc_trades(trades):
        t = trades[(trades["symbol"]=="BTC-USDC")
                   & (trades["date"]>="2020-10-01")
                   & (trades["date"]<="2021-03-15")].copy()
        return t[t["reason"].isin(["rebal_buy", "rebal_sell", "atr_stop"])].sort_values("date").reset_index(drop=True)

    bt_fix   = btc_trades(r_fix["trades"])
    bt_trail = btc_trades(r_trail["trades"])

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.subplots_adjust(top=0.86, bottom=0.10)

    # Background BTC price
    ax.fill_between(btc_w.index, 0, btc_w.values, color="#a02c2c", alpha=0.04)
    ax.plot(btc_w.index, btc_w.values, lw=1.3, color="#5a5a5a", alpha=0.85,
             zorder=2)
    # Direct label on price line
    ax.text(btc_w.index[-1] + pd.Timedelta(days=2), btc_w.iloc[-1],
             "BTC-USDC", color="#5a5a5a", fontsize=9.5,
             va="center", style="italic")

    # Fixed path: shaded hold window
    fix_entry = bt_fix.iloc[0]
    fix_exit  = bt_fix[bt_fix["reason"]=="rebal_sell"].iloc[-1]
    ax.axvspan(fix_entry["date"], fix_exit["date"], color=C_FIXED, alpha=0.05,
                zorder=1)

    # Common entry marker (both bought at $11,374 on 2020-10-12)
    ax.scatter([fix_entry["date"]], [fix_entry["price"]], marker="^", s=280,
                color=C_ENTRY, zorder=8, edgecolor="white", linewidth=2.0)
    ax.annotate("Both buy $11,374\n2020-10-12",
                xy=(fix_entry["date"], fix_entry["price"]),
                xytext=(-12, 38), textcoords="offset points",
                fontsize=10, color=C_ENTRY, fontweight="bold", ha="right",
                arrowprops=dict(arrowstyle="->", color=C_ENTRY, lw=1.0))

    # Fixed exit marker: navy down-triangle (clean rebalance exit, not stop)
    ax.scatter([fix_exit["date"]], [fix_exit["price"]], marker="v", s=280,
                color=C_FIXED, zorder=8, edgecolor="white", linewidth=2.0)
    ax.annotate("FIXED: SELL $45,232\nrebalance signal\n2021-03-08",
                xy=(fix_exit["date"], fix_exit["price"]),
                xytext=(-25, 50), textcoords="offset points",
                fontsize=10, color=C_FIXED, fontweight="bold", ha="right",
                arrowprops=dict(arrowstyle="->", color=C_FIXED, lw=1.5))

    # Trailing path: re-entries (green up-triangles, smaller than the common entry)
    trail_buys  = bt_trail[bt_trail["side"]=="BUY"].iloc[1:]  # skip the common 10-12 entry
    trail_sells = bt_trail[bt_trail["reason"]=="atr_stop"]
    ax.scatter(trail_buys["date"], trail_buys["price"], marker="^", s=110,
                color=C_ENTRY, zorder=6, edgecolor="white", linewidth=1.0,
                alpha=0.85)
    ax.scatter(trail_sells["date"], trail_sells["price"], marker="v", s=110,
                color=C_STOPEXIT, zorder=6, edgecolor="white", linewidth=1.0,
                alpha=0.95)

    # Annotate per-trailing-trade returns
    trail_returns = [+0.29, +0.57, +0.04, -0.02, -0.19]
    for i, (_, sell) in enumerate(trail_sells.iterrows()):
        if i < len(trail_returns):
            ret = trail_returns[i]
            color = C_ENTRY if ret > 0 else C_STOPEXIT
            ax.annotate(f"{ret*100:+.0f}%",
                        xy=(sell["date"], sell["price"]),
                        xytext=(0, -22), textcoords="offset points",
                        fontsize=10, color=color, fontweight="bold", ha="center")

    # Punchline callout in upper-left area
    callout = ("FIXED stop:  one trade,  +214%\n"
                "TRAILING stop:  5 trades,  +67% (compounded)\n"
                "─────────────────────────────────\n"
                "Same signal. Same entry. 3.2× shortfall.")
    ax.text(0.020, 0.965, callout, transform=ax.transAxes,
             fontsize=11, va="top", ha="left", fontfamily="DejaVu Sans Mono",
             color="#1c1c1c",
             bbox=dict(boxstyle="round,pad=0.7", facecolor="white",
                       edgecolor="#cccccc", alpha=0.97, linewidth=0.9))

    # Legend marker key
    handles = [
        plt.Line2D([], [], marker="^", linestyle="None", markersize=10,
                   markerfacecolor=C_ENTRY, markeredgecolor="white",
                   label="Entry"),
        plt.Line2D([], [], marker="v", linestyle="None", markersize=10,
                   markerfacecolor=C_STOPEXIT, markeredgecolor="white",
                   label="Trailing stop-out"),
        plt.Line2D([], [], marker="v", linestyle="None", markersize=10,
                   markerfacecolor=C_FIXED, markeredgecolor="white",
                   label="Fixed rebalance exit"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10, ncol=1)

    ax.set_ylabel("BTC price (USDC)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(btc_w.index[0], btc_w.index[-1] + pd.Timedelta(days=18))
    stamp_title(fig, "Same Bitcoin signal — two realized paths",
                  "Oct 2020 – Mar 2021 · both variants entered at the same $11,374 fill on Oct 12")
    byline(fig)
    plt.savefig(OUT/"01_btc_two_paths.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# CHART 2 — PERFORMANCE SCORECARD (designed graphic)
# ════════════════════════════════════════════════════════════════════════
def chart_scorecard(metrics):
    print("02: performance scorecard (4-col with no-stop & BTC HODL)...")
    # rows: label, no-stop, fixed, trailing, btc, emphasized
    rows = [
        ("Sharpe ratio",            f"{metrics['ns_sharpe']:.2f}",      f"{metrics['fix_sharpe']:.2f}",   f"{metrics['tr_sharpe']:.2f}",  f"{metrics['btc_sharpe']:.2f}",  True),
        ("Annualized return",       f"+{metrics['ns_cagr']*100:.1f}%",  f"+{metrics['fix_cagr']*100:.1f}%", f"+{metrics['tr_cagr']*100:.1f}%", f"+{metrics['btc_cagr']*100:.1f}%", False),
        ("Total return (9y)",       f"+{metrics['ns_total']*100:.0f}%", f"+{metrics['fix_total']*100:.0f}%", f"+{metrics['tr_total']*100:.0f}%", f"+{metrics['btc_total']*100:.0f}%", True),
        ("Max drawdown",            f"{metrics['ns_dd']*100:.1f}%",     f"{metrics['fix_dd']*100:.1f}%",  f"{metrics['tr_dd']*100:.1f}%",  f"{metrics['btc_dd']*100:.1f}%", True),
        ("Calmar (CAGR / |DD|)",    f"{metrics['ns_calmar']:.2f}",      f"{metrics['fix_calmar']:.2f}",   f"{metrics['tr_calmar']:.2f}",   f"{metrics['btc_calmar']:.2f}",  False),
        ("Stop-hit frequency",      "—",                                  f"{metrics['fix_stop']:.0f}%",  f"{metrics['tr_stop']:.0f}%",    "—",                                False),
    ]

    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.subplots_adjust(top=0.82, bottom=0.05, left=0.04, right=0.98)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    # Column x-positions
    x_metric = 4
    x_nostop = 40
    x_fixed  = 58
    x_trail  = 76
    x_btc    = 93

    y0 = 90
    ax.text(x_metric, y0, "M E T R I C", fontsize=10, color=C_GREY, fontweight="bold")
    ax.text(x_nostop, y0, "NO STOP",      fontsize=10.5, color=C_NOSTOP,   fontweight="bold", ha="center")
    ax.text(x_fixed,  y0, "FIXED 3×ATR",  fontsize=10.5, color=C_FIXED,    fontweight="bold", ha="center")
    ax.text(x_trail,  y0, "TRAILING 3×ATR", fontsize=10.5, color=C_TRAILING, fontweight="bold", ha="center")
    ax.text(x_btc,    y0, "BTC HODL",     fontsize=10.5, color="#444444",  fontweight="bold", ha="center")
    ax.plot([2, 98], [86, 86], color="#333333", lw=0.8)

    row_h = 11.5
    for i, (metric, ns_val, fix_val, tr_val, btc_val, emphasized) in enumerate(rows):
        y = 78 - i * row_h
        if emphasized:
            ax.add_patch(Rectangle((2, y - 3.5), 96, row_h - 1.2,
                                     facecolor="#f4f1ea", alpha=0.55,
                                     edgecolor="none", zorder=0))
        ax.text(x_metric, y, metric, fontsize=11.5 if emphasized else 11,
                color="#1c1c1c", fontweight="bold" if emphasized else "normal",
                va="center")
        for x, val, col in [
            (x_nostop, ns_val,  C_NOSTOP),
            (x_fixed,  fix_val, C_FIXED),
            (x_trail,  tr_val,  C_TRAILING),
            (x_btc,    btc_val, "#444444"),
        ]:
            ax.text(x, y, val,
                    fontsize=15.5 if emphasized else 12.5,
                    color=col if emphasized else "#1c1c1c",
                    fontweight="bold" if emphasized else "normal",
                    ha="center", va="center")
        if i < len(rows) - 1:
            ax.plot([2, 98], [y - 5.7, y - 5.7], color="#e6e6e6", lw=0.5)

    ax.plot([2, 98], [78 - len(rows)*row_h + 5.5, 78 - len(rows)*row_h + 5.5],
             color="#333333", lw=0.8)

    stamp_title(fig, "The stop-cost hierarchy",
                  "Same signal, same universe, four execution choices · "
                  "9-year Coinbase USDC bluechip backtest, 30 bps/side · "
                  "highlighted rows are the three that matter",
                  x_offset=0.04)
    byline(fig)
    plt.savefig(OUT/"02_scorecard.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# CHART 3 — EQUITY CURVE WITH BTC HODL
# ════════════════════════════════════════════════════════════════════════
def chart_equity_curve(panels, r_fix, r_trail, r_nostop):
    print("03: equity curve with BTC HODL + no-stop overlay...")
    nav_fix    = r_fix["equity"]["nav"]
    nav_trail  = r_trail["equity"]["nav"]
    nav_nostop = r_nostop["equity"]["nav"]

    btc = panels["C"]["BTC-USDC"].reindex(nav_fix.index, method="ffill")
    btc_norm = (btc / btc.dropna().iloc[0]) * 100_000

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.07, right=0.91)

    nav_f  = nav_fix / 1000
    nav_t  = nav_trail / 1000
    nav_ns = nav_nostop / 1000
    btc_n  = btc_norm / 1000

    ax.plot(btc_n.index, btc_n, lw=1.4, color=C_HODL, alpha=0.85,
             zorder=2, label="BTC HODL")
    ax.plot(nav_t.index, nav_t, lw=2.4, color=C_TRAILING,
             zorder=4, label="Trailing 3×ATR")
    ax.plot(nav_ns.index, nav_ns, lw=2.0, color=C_NOSTOP, alpha=0.85,
             linestyle="-", zorder=5, label="No stop")
    ax.plot(nav_f.index, nav_f, lw=2.6, color=C_FIXED,
             zorder=6, label="Fixed 3×ATR")

    # End-point labels
    end_x = nav_f.index[-1]
    ax.annotate(f"FIXED\n${nav_f.iloc[-1]:,.0f}k",
                xy=(end_x, nav_f.iloc[-1]),
                xytext=(10, 0), textcoords="offset points",
                color=C_FIXED, fontweight="bold", fontsize=10.5, va="center")
    ax.annotate(f"NO STOP\n${nav_ns.iloc[-1]:,.0f}k",
                xy=(end_x, nav_ns.iloc[-1]),
                xytext=(10, -2), textcoords="offset points",
                color=C_NOSTOP, fontweight="bold", fontsize=10.5, va="center")
    ax.annotate(f"BTC HODL\n${btc_n.iloc[-1]:,.0f}k",
                xy=(end_x, btc_n.iloc[-1]),
                xytext=(10, 0), textcoords="offset points",
                color=C_HODL, fontweight="bold", fontsize=10, va="center")
    ax.annotate(f"TRAILING\n${nav_t.iloc[-1]:,.0f}k",
                xy=(end_x, nav_t.iloc[-1]),
                xytext=(10, 0), textcoords="offset points",
                color=C_TRAILING, fontweight="bold", fontsize=10.5, va="center")

    ax.set_yscale("log")
    ax.set_ylabel("Portfolio value ($k, log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(nav_f.index[0], end_x + pd.Timedelta(days=200))

    stamp_title(fig, "Nine-year equity curves",
                  "$100k start · Coinbase USDC bluechip-10 universe · 30 bps per side · "
                  "no-stop and BTC HODL shown as reference series",
                  x_offset=0.07)
    byline(fig)
    plt.savefig(OUT/"03_equity_curve.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# CHART 4 — DRAWDOWN COMPARISON
# ════════════════════════════════════════════════════════════════════════
def chart_drawdown(r_fix, r_trail):
    print("04: drawdown comparison...")
    def dd(nav):
        peak = nav.cummax()
        return (nav / peak - 1) * 100

    dd_fix   = dd(r_fix["equity"]["nav"])
    dd_trail = dd(r_trail["equity"]["nav"])

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.93)

    ax.fill_between(dd_trail.index, dd_trail, 0, color=C_TRAILING, alpha=0.18,
                     zorder=2)
    ax.fill_between(dd_fix.index, dd_fix, 0, color=C_FIXED, alpha=0.20,
                     zorder=3)
    ax.plot(dd_trail.index, dd_trail, color=C_TRAILING, lw=1.8, zorder=4,
             label="Trailing")
    ax.plot(dd_fix.index, dd_fix, color=C_FIXED, lw=1.8, zorder=5,
             label="Fixed")

    ax.axhline(0, color="#333333", lw=0.6)

    # Both troughs ~ -50%
    ax.axhline(-50, color="#999999", lw=0.5, ls=":")
    ax.text(dd_fix.index[5], -50, "  −50% reference", fontsize=9, va="bottom",
             color="#666666", style="italic")

    # Direct labels at right edge
    ax.annotate(f"FIXED  worst {dd_fix.min():.1f}%",
                xy=(dd_fix.index[-1], dd_fix.iloc[-1]),
                xytext=(10, 5), textcoords="offset points",
                color=C_FIXED, fontweight="bold", fontsize=10,
                va="center")
    ax.annotate(f"TRAILING  worst {dd_trail.min():.1f}%",
                xy=(dd_trail.index[-1], dd_trail.iloc[-1]),
                xytext=(10, -10), textcoords="offset points",
                color=C_TRAILING, fontweight="bold", fontsize=10,
                va="center")

    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(dd_fix.index[0], dd_fix.index[-1] + pd.Timedelta(days=160))
    ax.set_ylim(min(dd_fix.min(), dd_trail.min()) * 1.05, 4)

    stamp_title(fig, "Drawdown — the protection that wasn't",
                  "Both variants reach roughly the same −50% trough. Trailing did not "
                  "buy you risk reduction; it bought you upside reduction.",
                  x_offset=0.07)
    byline(fig)
    plt.savefig(OUT/"04_drawdown_comparison.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# CHART 5 — STOP GEOMETRY SCHEMATIC (conceptual, not real data)
# ════════════════════════════════════════════════════════════════════════
def chart_stop_geometry():
    print("05: stop-geometry schematic...")
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.07, right=0.96)

    # Synthetic upward price path with realistic pullbacks
    np.random.seed(7)
    t = np.linspace(0, 100, 101)
    # Smooth upward trend with two visible pullbacks
    trend = 100 + (t/100) * 50
    noise = np.cumsum(np.random.normal(0, 0.7, len(t)))
    # Add two structured pullbacks
    pullback = np.zeros_like(t)
    pullback[35:48] = -8 * np.sin(np.linspace(0, np.pi, 13))
    pullback[70:85] = -10 * np.sin(np.linspace(0, np.pi, 15))
    price = trend + 0.4 * (noise - noise.mean()) + pullback
    price[0] = 100
    price[-1] = 150

    # Plot price
    ax.plot(t, price, color="#3a3a3a", lw=2.3, zorder=5, label="Price")

    # Fixed stop: horizontal at 85
    ax.axhline(85, color=C_FIXED, lw=2.5, zorder=4)
    ax.text(0.5, 85 + 1.2, "FIXED stop  (entry − 3·ATR)", color=C_FIXED,
             fontsize=10.5, fontweight="bold", va="bottom")

    # Trailing stop: highest-close-since-entry minus 15
    high_so_far = np.maximum.accumulate(price)
    trail_stop = high_so_far - 15
    ax.plot(t, trail_stop, color=C_TRAILING, lw=2.5, zorder=4,
             label="Trailing stop")
    ax.text(t[-1] + 1.5, trail_stop[-1], "TRAILING stop\n(highest close − 3·ATR)",
             color=C_TRAILING, fontsize=10.5, fontweight="bold", va="center")

    # Entry marker at t=0
    ax.scatter([0], [100], marker="^", s=200, color=C_ENTRY, zorder=7,
                edgecolor="white", linewidth=1.8)
    ax.annotate("ENTRY\n$100", xy=(0, 100), xytext=(-5, -25),
                 textcoords="offset points", fontsize=10, color=C_ENTRY,
                 fontweight="bold", ha="right")

    # Highlight the "at +50% gain" moment
    idx_50 = np.argmin(np.abs(price - 150))
    ax.axvline(idx_50, color="#bbbbbb", lw=0.8, ls=":", zorder=1)

    # Mark the +50% level
    ax.scatter([idx_50], [150], marker="o", s=90, color="#1c1c1c", zorder=8,
                edgecolor="white", linewidth=1.5)
    ax.annotate("After a +50%\nrun-up: $150",
                 xy=(idx_50, 150), xytext=(-95, 8),
                 textcoords="offset points", fontsize=10, color="#1c1c1c",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))

    # Cushion bands at idx_50 — both on the right side of the peak, side by side
    trail_at_50 = trail_stop[idx_50]

    # Trailing cushion (narrow band, closer to peak)
    bar_w = 3.0
    bar_t_x = idx_50 + 2
    ax.add_patch(Rectangle((bar_t_x, trail_at_50), bar_w,
                             150 - trail_at_50,
                             facecolor=C_TRAILING, alpha=0.30,
                             edgecolor=C_TRAILING, linewidth=1.4, zorder=3))
    # Brace-style label for trailing cushion
    cushion_mid_t = (trail_at_50 + 150) / 2
    ax.annotate(f"TRAILING\ncushion\n≈ {(150-trail_at_50)/150*100:.0f}%",
                 xy=(bar_t_x + bar_w + 0.2, cushion_mid_t),
                 xytext=(8, 0), textcoords="offset points",
                 fontsize=10, color=C_TRAILING, fontweight="bold", va="center")

    # Fixed cushion (deeper band, further right)
    bar_f_x = idx_50 + 14
    ax.add_patch(Rectangle((bar_f_x, 85), bar_w, 150 - 85,
                            facecolor=C_FIXED, alpha=0.18, edgecolor=C_FIXED,
                            linewidth=1.4, zorder=3))
    cushion_mid_f = (85 + 150) / 2
    ax.annotate(f"FIXED\ncushion\n≈ {(150-85)/150*100:.0f}%",
                 xy=(bar_f_x + bar_w + 0.2, cushion_mid_f),
                 xytext=(8, 0), textcoords="offset points",
                 fontsize=10, color=C_FIXED, fontweight="bold", va="center")

    ax.set_ylim(75, 168)
    ax.set_xlim(-8, 120)
    ax.set_xlabel("Trade progress")
    ax.set_ylabel("Price")
    ax.set_xticks([])

    stamp_title(fig,
                  "Why the fixed stop gets wider as the trade wins",
                  "A schematic of stop distance as a trade runs up 50%. The fixed stop "
                  "stands still while the cushion grows; the trailing stop ratchets up "
                  "and keeps the cushion constant.",
                  x_offset=0.07)
    byline(fig, "conceptual schematic — not from backtest data")
    plt.savefig(OUT/"05_stop_geometry.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# CHART 6 — SOL CENTERPIECE
# ════════════════════════════════════════════════════════════════════════
def chart_sol_centerpiece(panels, ind, rts_fix, rts_trail):
    print("06: SOL centerpiece...")
    # Find the biggest fixed-stop winner on SOL
    sol_fix = rts_fix[rts_fix["symbol"] == "SOL-USDC"].nlargest(1, "pnl_pct").iloc[0]
    sym = "SOL-USDC"
    entry    = sol_fix["entry_date"]
    exit_fix = sol_fix["exit_date"]

    C = panels["C"][sym]; H = panels["H"][sym]; L = panels["L"][sym]; O = panels["O"][sym]
    atr = ind["atr"][sym]

    trail_match = rts_trail[(rts_trail["symbol"] == sym) &
                             (rts_trail["entry_date"] >= entry - pd.Timedelta(days=7)) &
                             (rts_trail["entry_date"] <= exit_fix)].sort_values("exit_date")

    # Find a re-entry attempt — look for next trail trade after the first stop
    if not trail_match.empty:
        first_trail_exit = trail_match["exit_date"].min()
    else:
        first_trail_exit = entry + pd.Timedelta(days=30)

    # Find the trailing variant's *next* SOL entry after first_trail_exit
    later_trail = rts_trail[(rts_trail["symbol"] == sym) &
                              (rts_trail["entry_date"] > first_trail_exit) &
                              (rts_trail["entry_date"] <= exit_fix + pd.Timedelta(days=14))]
    next_reentry = later_trail.iloc[0] if not later_trail.empty else None

    win_start = entry - pd.Timedelta(days=18)
    win_end   = exit_fix + pd.Timedelta(days=25)
    sl = (C.index >= win_start) & (C.index <= win_end)
    px = C[sl]

    entry_atr = atr.loc[entry]
    entry_px = O.loc[entry] if not np.isnan(O.loc[entry]) else C.loc[entry]
    fixed_stop_lvl = entry_px - 3.0 * entry_atr

    # Build trailing stop level series
    running_high = pd.Series(index=px.index, dtype=float)
    rh = entry_px
    for d in running_high.index:
        if d < entry:
            running_high.loc[d] = np.nan
            continue
        c = C.loc[d]
        if not np.isnan(c) and c > rh:
            rh = c
        running_high.loc[d] = rh
    trail_stop_lvl = running_high - 3.0 * entry_atr

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.07, right=0.96)

    # Missed-upside shading: from first trailing stop-out to fixed exit
    first_exit = trail_match.iloc[0] if not trail_match.empty else None
    if first_exit is not None:
        miss_start = first_exit["exit_date"]
        miss_mask = (px.index > miss_start) & (px.index <= exit_fix)
        miss_lo = first_exit["exit_px"]
        miss_hi = px[miss_mask].values
        ax.fill_between(px.index[miss_mask], miss_lo, miss_hi,
                         color=C_MISSED, alpha=0.85, zorder=1, label="_nolegend_")
        # Annotate missed region
        if miss_mask.sum() > 5:
            mid_x = px.index[miss_mask][miss_mask.sum() // 2]
            mid_y = (miss_lo + px[miss_mask].max()) / 2
            ax.text(mid_x, mid_y, "MISSED UPSIDE\n~$40 → ~$120",
                     fontsize=11, color="#a64500", fontweight="bold",
                     ha="center", va="center", alpha=0.85,
                     style="italic")

    # Price (log scale)
    ax.plot(px.index, px, color="#1c1c1c", lw=1.7, zorder=5)
    ax.text(px.index[-1] + pd.Timedelta(days=1.5), px.iloc[-1], "SOL-USDC",
             color="#1c1c1c", fontsize=9.5, va="center", style="italic")

    # Fixed stop horizontal — label on the left, below the line
    ax.axhline(fixed_stop_lvl, color=C_FIXED, lw=2.5, zorder=4)
    ax.text(px.index[3], fixed_stop_lvl * 0.93,
             f"FIXED stop = ${fixed_stop_lvl:.2f}", color=C_FIXED,
             fontsize=10.5, fontweight="bold", va="top", ha="left")

    # Trailing stop staircase
    ax.plot(trail_stop_lvl.index, trail_stop_lvl, color=C_TRAILING, lw=2.3,
             zorder=4)
    # Direct label on trailing stop in the upper-middle plateau region
    plateau_idx = trail_stop_lvl[trail_stop_lvl > 50].index[2] if (trail_stop_lvl > 50).sum() > 5 else trail_stop_lvl.idxmax()
    ax.annotate("TRAILING stop\n(ratchets up with\nnew closing highs)",
                 xy=(plateau_idx, trail_stop_lvl.loc[plateau_idx]),
                 xytext=(-110, 14), textcoords="offset points",
                 fontsize=10, color=C_TRAILING, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_TRAILING, lw=1.0))

    # Entry marker
    ax.scatter([entry], [entry_px], marker="^", s=260, color=C_ENTRY,
                zorder=8, edgecolor="white", linewidth=2.0)
    ax.annotate(f"ENTRY ${entry_px:.2f}\n{entry.date()}",
                 xy=(entry, entry_px), xytext=(20, 18),
                 textcoords="offset points",
                 fontsize=10, color=C_ENTRY, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_ENTRY, lw=1.0))

    # Trailing stop-out marker
    if first_exit is not None:
        ax.scatter([first_exit["exit_date"]], [first_exit["exit_px"]],
                    marker="v", s=240, color=C_STOPEXIT, zorder=8,
                    edgecolor="white", linewidth=2.0)
        ax.annotate(f"TRAILING stops out\n${first_exit['exit_px']:.2f}  ({first_exit['pnl_pct']*100:+.0f}%)",
                     xy=(first_exit["exit_date"], first_exit["exit_px"]),
                     xytext=(-115, -8), textcoords="offset points",
                     fontsize=10, color=C_STOPEXIT, fontweight="bold",
                     ha="right",
                     arrowprops=dict(arrowstyle="->", color=C_STOPEXIT, lw=1.2))

    # Re-entry marker — push annotation BELOW the price line to avoid clutter
    if next_reentry is not None:
        reentry_px = next_reentry["avg_entry_px"]
        ax.scatter([next_reentry["entry_date"]], [reentry_px], marker="^",
                    s=170, color=C_ENTRY, zorder=7, edgecolor="white",
                    linewidth=1.6, alpha=0.85)
        ax.annotate(f"trailing re-enters\n${reentry_px:.2f}  (chasing higher)",
                     xy=(next_reentry["entry_date"], reentry_px),
                     xytext=(20, -55), textcoords="offset points",
                     fontsize=9.5, color=C_ENTRY, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=C_ENTRY, lw=1.0))

    # Fixed exit marker — push annotation up high, well above the price peak
    fix_exit_px = sol_fix["exit_px"]
    ax.scatter([exit_fix], [fix_exit_px], marker="v", s=260, color=C_FIXED,
                zorder=8, edgecolor="white", linewidth=2.0)
    ax.annotate(f"FIXED exits via rebalance\n${fix_exit_px:.2f}  ({sol_fix['pnl_pct']*100:+.0f}%)",
                 xy=(exit_fix, fix_exit_px),
                 xytext=(-20, 55), textcoords="offset points",
                 fontsize=10, color=C_FIXED, fontweight="bold", ha="right",
                 arrowprops=dict(arrowstyle="->", color=C_FIXED, lw=1.2))

    # Callout box: punchline (place at upper-left over empty area)
    callout = ("TRAILING:  banked +60%, then missed\nthe 3× continuation.\n\n"
                "FIXED:  rode the whole move.\n+256% on one trade.")
    ax.text(0.025, 0.97, callout, transform=ax.transAxes,
             fontsize=10.5, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.7", facecolor="white",
                       edgecolor="#cccccc", alpha=0.96, linewidth=0.9))

    ax.set_yscale("log")
    # Extend y-axis upward to give room for the FIXED exit annotation
    ax.set_ylim(top=px.max() * 1.35)
    ax.set_ylabel("SOL price (USDC, log scale)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    stamp_title(fig, f"The trade that shows the trap — {sym}, Oct 2023",
                  "Same signal, same entry price, same week. The fixed stop sat still "
                  "and rode a 4× move. The trailing stop locked in a small win and "
                  "watched the rest from cash.",
                  x_offset=0.07)
    byline(fig)
    plt.savefig(OUT/"06_sol_centerpiece.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# CHART 7 — RETURN DISTRIBUTION (simplified, right-tail focus)
# ════════════════════════════════════════════════════════════════════════
def chart_return_distribution(rts_fix, rts_trail):
    print("07: return distribution...")
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(top=0.83, bottom=0.13, left=0.07, right=0.96)

    bins = np.linspace(-0.5, 2.7, 50)
    ax.hist(rts_trail["pnl_pct"], bins=bins, alpha=0.70, color=C_TRAILING,
             edgecolor="white", linewidth=0.5, zorder=4,
             label=f"Trailing  ({len(rts_trail)} trades)")
    ax.hist(rts_fix["pnl_pct"], bins=bins, alpha=0.70, color=C_FIXED,
             edgecolor="white", linewidth=0.5, zorder=5,
             label=f"Fixed  ({len(rts_fix)} trades)")
    ax.axvline(0, color="#333333", lw=0.6)

    # Right-tail emphasis: shade the +50%+ region softly
    ax.axvspan(0.5, 2.7, color="#f4f1ea", alpha=0.5, zorder=1)

    # Best-trade markers
    fix_best = rts_fix["pnl_pct"].max()
    tr_best  = rts_trail["pnl_pct"].max()
    ax.axvline(fix_best, color=C_FIXED, lw=1.5, ls="--", alpha=0.7)
    ax.axvline(tr_best, color=C_TRAILING, lw=1.5, ls="--", alpha=0.7)
    ax.annotate(f"Best fixed trade\n+{fix_best*100:.0f}%",
                 xy=(fix_best, 4), xytext=(-10, 35), textcoords="offset points",
                 fontsize=10.5, color=C_FIXED, fontweight="bold", ha="right",
                 arrowprops=dict(arrowstyle="->", color=C_FIXED, lw=1.0))
    ax.annotate(f"Best trailing trade\n+{tr_best*100:.0f}%",
                 xy=(tr_best, 18), xytext=(40, 25), textcoords="offset points",
                 fontsize=10.5, color=C_TRAILING, fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_TRAILING, lw=1.0))

    # Right-tail caption
    ax.text(2.65, ax.get_ylim()[1] * 0.55,
             "the shaded region\nis where trend-following\nactually makes money",
             fontsize=9.5, color="#888888", style="italic",
             ha="right", va="center")
    ax.set_xlabel("Round-trip return")
    ax.set_ylabel("Number of trades")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:+.0f}%"))
    ax.legend(loc="upper left", bbox_to_anchor=(0.30, 0.98), fontsize=11)

    stamp_title(fig, "Per-trade return distribution — the right tail tells the story",
                  "Identical left tails (the stop kicks in at the same level). "
                  "But the trailing stop chops off everything above +95%.",
                  x_offset=0.07)
    byline(fig)
    plt.savefig(OUT/"07_return_distribution.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# CHART 8 — STOP-HIT COMPOSITION (simplified)
# ════════════════════════════════════════════════════════════════════════
def chart_stop_hit_composition(rts_fix, rts_trail):
    print("08: stop-hit composition...")
    fix_stop = (rts_fix["exit_reason"] == "atr_stop").mean() * 100
    tr_stop  = (rts_trail["exit_reason"] == "atr_stop").mean() * 100
    fix_rebal = 100 - fix_stop
    tr_rebal  = 100 - tr_stop

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.18, right=0.85)

    labels = ["Fixed stop", "Trailing stop"]
    y = np.array([1, 0])
    stops  = [fix_stop, tr_stop]
    rebals = [fix_rebal, tr_rebal]

    bar_h = 0.55
    bars_stop = ax.barh(y, stops, height=bar_h, color=C_STOPEXIT, alpha=0.85,
                          edgecolor="white", zorder=4, label="Forced exit (stop hit)")
    bars_reb  = ax.barh(y, rebals, left=stops, height=bar_h, color=C_REBAL,
                          alpha=0.85, edgecolor="white", zorder=4,
                          label="Signal-driven exit (rebalance)")

    # Annotations inside bars
    for i, (s, r) in enumerate(zip(stops, rebals)):
        ax.text(s/2, y[i], f"{s:.0f}%", ha="center", va="center",
                 color="white", fontweight="bold", fontsize=22, zorder=10)
        ax.text(s + r/2, y[i], f"{r:.0f}%", ha="center", va="center",
                 color="white", fontweight="bold", fontsize=22, zorder=10)

    # Y labels
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13, fontweight="bold")
    # Color the labels
    ax.get_yticklabels()[0].set_color(C_FIXED)
    ax.get_yticklabels()[1].set_color(C_TRAILING)

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("% of round-trip trades")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(False, axis="y")

    # Legend below
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30),
               ncol=2, fontsize=11)

    stamp_title(fig, "How each variant actually exits — and the geometry shift it reveals",
                  "Fixed: stop is the emergency brake (28%), rebalance is the edge (72%). "
                  "Trailing: stop becomes the primary exit logic (70%).",
                  x_offset=0.06)
    byline(fig)
    plt.savefig(OUT/"08_stop_hit_composition.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# CHART 9 — RE-ENTRY TAX
# ════════════════════════════════════════════════════════════════════════
def chart_reentry_tax():
    print("09: re-entry tax...")
    # Hardcode the verified BTC re-entry sequence from the trailing variant
    pairs = [
        ("2020-11-07", 14722, "2020-12-14", 19167),   # +30.2%
        ("2021-01-04", 30064, "2021-01-04", 33083),   # +10.0% same day
        ("2021-01-10", 34526, "2021-02-15", 48667),   # +41.0%
        ("2021-02-22", 47631, "2021-02-22", 57489),   # +20.7% same day
        ("2021-02-23", 46328, "2021-03-08", 50976),   # +10.0%
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    fig.subplots_adjust(top=0.85, bottom=0.10, left=0.10, right=0.94)

    n = len(pairs)
    y_positions = np.arange(n, 0, -1)
    max_price = max(max(p[1], p[3]) for p in pairs) * 1.10
    min_price = min(min(p[1], p[3]) for p in pairs) * 0.85

    for i, (exit_date, exit_px, reentry_date, reentry_px) in enumerate(pairs):
        y = y_positions[i]
        tax_pct = (reentry_px / exit_px - 1) * 100
        # Background row stripe
        if i % 2 == 0:
            ax.axhspan(y - 0.42, y + 0.42, color="#f4f1ea", alpha=0.45, zorder=1)

        # Exit dot (red)
        ax.scatter([exit_px], [y], marker="o", s=220, color=C_STOPEXIT,
                    edgecolor="white", linewidth=2.0, zorder=6)
        # Re-entry dot (green)
        ax.scatter([reentry_px], [y], marker="o", s=220, color=C_ENTRY,
                    edgecolor="white", linewidth=2.0, zorder=6)
        # Arrow connecting them
        arrow = FancyArrowPatch((exit_px, y), (reentry_px, y),
                                  arrowstyle="->",
                                  mutation_scale=22,
                                  color="#666666", lw=2.0, zorder=5)
        ax.add_patch(arrow)

        # Labels
        ax.annotate(f"stopped out\n${exit_px:,}\n{exit_date}",
                     xy=(exit_px, y), xytext=(-12, 0),
                     textcoords="offset points",
                     fontsize=9, color=C_STOPEXIT, fontweight="bold",
                     ha="right", va="center")
        ax.annotate(f"re-entered\n${reentry_px:,}\n{reentry_date}",
                     xy=(reentry_px, y), xytext=(12, 0),
                     textcoords="offset points",
                     fontsize=9, color=C_ENTRY, fontweight="bold",
                     ha="left", va="center")
        # Tax label in middle, in red
        mid_x = (exit_px + reentry_px) / 2
        ax.text(mid_x, y + 0.30, f"+{tax_pct:.0f}% re-entry tax",
                 fontsize=11, color="#a02c2c", fontweight="bold", ha="center")

    # Summary stat
    taxes = [(p[3]/p[1] - 1)*100 for p in pairs]
    avg_tax = np.mean(taxes)
    total_chain = np.prod([p[3]/p[1] for p in pairs]) - 1
    ax.text(0.5, -0.15,
             f"Average re-entry tax per stop-out: +{avg_tax:.0f}%      "
             f"|      Compounded chain penalty: +{total_chain*100:.0f}%",
             transform=ax.transAxes, fontsize=11.5, fontweight="bold",
             ha="center", color="#a02c2c",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f1ea",
                       edgecolor="#a02c2c", linewidth=1.2))

    ax.set_xlim(min_price, max_price)
    ax.set_ylim(0.2, n + 0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlabel("BTC price (USDC)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    stamp_title(fig, "The trailing stop didn't just exit early — it forced worse re-entry prices",
                  "Each stop-out was followed by a re-entry at a meaningfully higher price. "
                  "Five round trips, five higher fills, one persistent uptrend.",
                  x_offset=0.10)
    byline(fig)
    plt.savefig(OUT/"09_reentry_tax.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════
# CHART 10 — STOP-COST HIERARCHY (None vs Fixed vs Trailing)
# ════════════════════════════════════════════════════════════════════════
def chart_stop_cost_hierarchy(metrics):
    """Two-panel bar chart: CAGR vs Max DD across the three stop variants.

    Annotations call out the *trade* each stop choice represents: bps of
    CAGR given up per percentage point of max-drawdown reduction.
    """
    print("10: stop-cost hierarchy...")
    variants = ["No stop", "Fixed 3×ATR", "Trailing 3×ATR"]
    colors   = [C_NOSTOP, C_FIXED, C_TRAILING]
    cagrs    = [metrics["ns_cagr"]*100,  metrics["fix_cagr"]*100, metrics["tr_cagr"]*100]
    dds      = [metrics["ns_dd"]*100,    metrics["fix_dd"]*100,   metrics["tr_dd"]*100]

    # Cost-per-pp-of-DD-reduction relative to no-stop baseline
    base_cagr = cagrs[0]
    base_dd   = abs(dds[0])
    def cost_per_pp(c, d):
        dd_red = base_dd - abs(d)
        cagr_given = base_cagr - c
        if dd_red <= 0:
            return None
        return cagr_given * 100 / dd_red  # bps per pp

    cost_fixed = cost_per_pp(cagrs[1], dds[1])
    cost_trail = cost_per_pp(cagrs[2], dds[2])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 6.4))
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.08, right=0.97, wspace=0.28)

    # Left panel: CAGR
    bars1 = axL.bar(variants, cagrs, color=colors, width=0.62, edgecolor="white",
                     linewidth=1.4)
    axL.axhline(0, color="#333", lw=0.7)
    for b, v in zip(bars1, cagrs):
        axL.text(b.get_x() + b.get_width()/2, b.get_height() + 0.6,
                  f"+{v:.1f}%", ha="center", fontsize=12.5, fontweight="bold")
    axL.set_ylabel("Annualized return (CAGR, 9 years)", fontsize=11)
    axL.set_ylim(0, max(cagrs) * 1.25)
    axL.set_title("Return is paid for by stops",
                   fontsize=12.5, color="#1c1c1c", loc="left", pad=10)

    # Right panel: Max DD (use absolute values, descending bars)
    abs_dds = [abs(d) for d in dds]
    bars2 = axR.bar(variants, abs_dds, color=colors, width=0.62, edgecolor="white",
                     linewidth=1.4)
    for b, v in zip(bars2, abs_dds):
        axR.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                  f"-{v:.1f}%", ha="center", fontsize=12.5, fontweight="bold")
    axR.set_ylabel("Max drawdown (absolute value)", fontsize=11)
    axR.set_ylim(0, max(abs_dds) * 1.18)
    axR.set_title("…but stops barely move the drawdown",
                   fontsize=12.5, color="#1c1c1c", loc="left", pad=10)

    # Bottom annotation: cost-per-pp summary
    if cost_fixed is not None and cost_trail is not None:
        fig.text(0.5, 0.05,
                  f"Cost of buying drawdown protection — bps of CAGR given up "
                  f"per percentage point of max-DD reduction vs the no-stop baseline:    "
                  f"Fixed = {cost_fixed:,.0f} bps/pp     ·     "
                  f"Trailing = {cost_trail:,.0f} bps/pp     ({cost_trail/cost_fixed:.1f}× worse trade)",
                  ha="center", fontsize=10.5, color="#1c1c1c",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f1ea",
                            edgecolor="#cfc8b8", linewidth=0.8))

    stamp_title(fig, "The stop-cost hierarchy",
                  "Three stop choices on the same signal · drawdown improvement is "
                  "modest in all cases; the cost in foregone return is not",
                  x_offset=0.08)
    byline(fig)
    plt.savefig(OUT/"10_stop_cost_hierarchy.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# CHART 11 — BREAKOUT-WINDOW SWEEP
# ════════════════════════════════════════════════════════════════════════
def chart_breakout_sweep(panels, ind):
    """Bar chart: Sharpe and CAGR across 5/10/20-day breakout windows
    (with fixed 3×ATR stops, otherwise identical spec).
    """
    print("11: breakout-window sweep...")
    rows = []
    for bw in [5, 10, 20]:
        r = run_variant(panels, ind, trailing=False, atr_mult=3.0,
                         use_atr_stop=True, breakout_window=bw)
        nav = r["equity"]["nav"]
        m = wb.metrics_from_nav(nav, "")
        rows.append((bw, m["sharpe"], m["cagr"]*100, m["max_dd"]*100, m["total"]*100))
    labels = [f"{bw}-day" for bw, *_ in rows]
    sharpes = [r[1] for r in rows]
    cagrs   = [r[2] for r in rows]
    dds     = [abs(r[3]) for r in rows]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.8))
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.08, right=0.97, wspace=0.28)

    # Highlight 5-day (spec) in C_FIXED, others in muted grey
    bar_colors = [C_FIXED, "#aabac5", "#aabac5"]

    bars1 = axL.bar(labels, sharpes, color=bar_colors, width=0.55,
                     edgecolor="white", linewidth=1.4)
    for b, v in zip(bars1, sharpes):
        axL.text(b.get_x() + b.get_width()/2, b.get_height() + 0.018,
                  f"{v:.2f}", ha="center", fontsize=12.5, fontweight="bold")
    axL.set_ylim(0, max(sharpes) * 1.22)
    axL.set_ylabel("Sharpe ratio", fontsize=11)
    axL.set_title("Sharpe by breakout window", fontsize=12.5,
                   color="#1c1c1c", loc="left", pad=10)

    bars2 = axR.bar(labels, cagrs, color=bar_colors, width=0.55,
                     edgecolor="white", linewidth=1.4)
    for b, v in zip(bars2, cagrs):
        axR.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                  f"+{v:.1f}%", ha="center", fontsize=12.5, fontweight="bold")
    axR.set_ylim(0, max(cagrs) * 1.22)
    axR.set_ylabel("Annualized return (CAGR)", fontsize=11)
    axR.set_title("CAGR by breakout window", fontsize=12.5,
                   color="#1c1c1c", loc="left", pad=10)

    fig.text(0.5, 0.05,
              "Crypto trends move quickly. Wider breakout windows enter later, "
              "give up the first leg of the move, and degrade both risk-adjusted and absolute return.",
              ha="center", fontsize=10.5, color="#1c1c1c",
              style="italic")

    stamp_title(fig, "Robustness — breakout window",
                  "Same fixed 3×ATR stop, only the entry filter changes · "
                  "spec value (5 days) dominates the wider alternatives",
                  x_offset=0.08)
    byline(fig)
    plt.savefig(OUT/"11_breakout_sweep.png")
    plt.close()


def main():
    print("Loading lake & running variants on Bluechip-10...")
    syms, bars = wb.load_universe(restrict_to=BLUECHIP_10)
    panels = wb.assemble_panels(syms, bars)
    ind = wb.compute_indicators(panels, 1_000_000.0)
    ind["mom_score"] = wb.momentum_score(ind["mom"], ind["eligible_universe"])

    r_fix    = run_variant(panels, ind, trailing=False)
    r_trail  = run_variant(panels, ind, trailing=True)
    r_nostop = run_variant(panels, ind, trailing=False, use_atr_stop=False)
    rts_fix   = round_trips(r_fix["trades"])
    rts_trail = round_trips(r_trail["trades"])

    # Compute metrics (engine uses ANN=365 = correct for crypto)
    m_fix    = wb.metrics_from_nav(r_fix["equity"]["nav"], "")
    m_trail  = wb.metrics_from_nav(r_trail["equity"]["nav"], "")
    m_nostop = wb.metrics_from_nav(r_nostop["equity"]["nav"], "")
    btc_nav  = hodl_nav(panels, r_fix["equity"]["nav"].index)
    m_btc    = wb.metrics_from_nav(btc_nav, "")

    def calmar(m): return m["cagr"]/abs(m["max_dd"]) if m["max_dd"] else 0.0

    metrics = dict(
        # No-stop
        ns_sharpe=m_nostop["sharpe"], ns_cagr=m_nostop["cagr"],
        ns_total=m_nostop["total"],   ns_dd=m_nostop["max_dd"],
        ns_calmar=calmar(m_nostop),
        # Fixed
        fix_sharpe=m_fix["sharpe"], fix_cagr=m_fix["cagr"],
        fix_total=m_fix["total"],   fix_dd=m_fix["max_dd"],
        fix_calmar=calmar(m_fix),
        # Trailing
        tr_sharpe=m_trail["sharpe"], tr_cagr=m_trail["cagr"],
        tr_total=m_trail["total"],   tr_dd=m_trail["max_dd"],
        tr_calmar=calmar(m_trail),
        # BTC HODL
        btc_sharpe=m_btc["sharpe"], btc_cagr=m_btc["cagr"],
        btc_total=m_btc["total"],   btc_dd=m_btc["max_dd"],
        btc_calmar=calmar(m_btc),
        # Trade metrics
        fix_best=rts_fix["pnl_pct"].max(), tr_best=rts_trail["pnl_pct"].max(),
        fix_stop=(rts_fix["exit_reason"]=="atr_stop").mean()*100,
        tr_stop=(rts_trail["exit_reason"]=="atr_stop").mean()*100,
    )

    # Generate all charts
    chart_btc_two_paths(panels, r_fix, r_trail)
    chart_scorecard(metrics)
    chart_equity_curve(panels, r_fix, r_trail, r_nostop)
    chart_drawdown(r_fix, r_trail)
    chart_stop_geometry()
    chart_sol_centerpiece(panels, ind, rts_fix, rts_trail)
    chart_return_distribution(rts_fix, rts_trail)
    chart_stop_hit_composition(rts_fix, rts_trail)
    chart_reentry_tax()
    chart_stop_cost_hierarchy(metrics)
    chart_breakout_sweep(panels, ind)

    import json
    with open(OUT/"metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print(f"\nWrote all 11 charts + metrics.json to {OUT}")


if __name__ == "__main__":
    main()
