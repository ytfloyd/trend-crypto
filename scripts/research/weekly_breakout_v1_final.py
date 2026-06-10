#!/usr/bin/env python
"""Weekly Breakout v1 — FINAL: bluechip universe (the configuration that works).

Produces the full deliverable set for the report:
  - Bluechip-10 and Bluechip-20 results, as-specified (30 bps/side, 3 ATR, 20/40/90 mom)
  - Sensitivity at 10 bps/side and 0 bps (frictionless)
  - Side-by-side vs BTC-USDC B&H and MA(5/40) daily on same universe
  - Equity curve, drawdown, calendar-year, positions over time, scatter vs B&H,
    contribution-by-asset
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path("/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto")
spec = importlib.util.spec_from_file_location("wb", ROOT/"scripts/research/weekly_breakout_v1.py")
wb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wb)

OUT = ROOT / "artifacts/research/weekly_breakout_v1_final"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BLUECHIP_10 = ["BTC-USDC","ETH-USDC","SOL-USDC","XRP-USDC","ADA-USDC",
                "DOGE-USDC","AVAX-USDC","LINK-USDC","DOT-USDC","LTC-USDC"]
BLUECHIP_20 = BLUECHIP_10 + ["BCH-USDC","ATOM-USDC","ALGO-USDC","NEAR-USDC","OP-USDC",
                              "ARB-USDC","UNI-USDC","AAVE-USDC","MATIC-USDC","FIL-USDC"]


def ma_5_40_daily_basket(panels, eligible_mask, start_date, cost_per_side):
    """MA(5/40) equal-weight basket of all hold-eligible names, DAILY rebalance
    (matches the prior MA(5/40) study's construction). Apples-to-apples baseline."""
    C = panels["C"]; O = panels["O"]
    ma5 = C.rolling(5).mean(); ma40 = C.rolling(40).mean()
    pos = (ma5 > ma40) & eligible_mask
    n = pos.sum(axis=1).replace(0, np.nan)
    w = pos.div(n, axis=0).fillna(0.0)
    ret = C.pct_change(fill_method=None).fillna(0.0)
    strat_r = (w.shift(1) * ret).sum(axis=1)  # 1-day lag (signal yesterday, return today)
    tov = w.shift(1).diff().abs().sum(axis=1).fillna(0.0)
    strat_r = strat_r - tov * cost_per_side
    nav = (1 + strat_r[strat_r.index >= start_date]).cumprod() * 100_000.0
    return nav


def btc_bh(panels, start_date, cost_per_side):
    o = panels["O"]["BTC-USDC"]; c = panels["C"]["BTC-USDC"]
    o, c = o.align(c, join="inner")
    o = o.loc[o.index >= start_date]; c = c.loc[c.index >= start_date]
    r = (c/o - 1).fillna(0.0)
    return 100_000 * (1+r).cumprod() * (1 - cost_per_side)


def run_breakout_bluechip(syms_filter, label, cost_per_side=30/10000.0,
                            min_eligible=3, atr_mult=3.0):
    syms, bars = wb.load_universe(restrict_to=syms_filter)
    panels = wb.assemble_panels(syms, bars)
    ind = wb.compute_indicators(panels, 1_000_000.0)
    ind["mom_score"] = wb.momentum_score(ind["mom"], ind["eligible_universe"])
    result = wb.backtest(panels, ind, params=dict(
        cost_per_side=cost_per_side,
        atr_stop_mult=atr_mult, use_atr_stop=True,
        require_breakout_entry=True,
        min_eligible_at_start=min_eligible,
    ))
    return result, panels, ind


def calendar_year_returns(nav):
    yrly = nav.resample("YE").last()
    ret = yrly.pct_change().dropna()
    first_year = nav.index[0].year
    first_full = pd.Timestamp(f"{first_year}-12-31")
    if nav.index[0] <= first_full <= nav.index[-1]:
        ret[first_full] = nav.loc[:first_full].iloc[-1] / nav.iloc[0] - 1
        ret = ret.sort_index()
    ret.index = ret.index.year
    return ret


def main():
    print("=== Weekly Breakout v1 FINAL — bluechip universe ===\n")

    # Primary run: Bluechip-10, as-specified
    print("[Primary] Bluechip-10, as-specified (30bps/side, 3-ATR, 20/40/90 mom)")
    r10, panels10, ind10 = run_breakout_bluechip(BLUECHIP_10, "BC10-spec",
                                                   cost_per_side=30/10000.0)
    nav10 = r10["equity"]["nav"]
    m10 = wb.metrics_from_nav(nav10, "BC10 spec 30bps")
    start_date = nav10.index[0]
    print(f"  Start: {start_date.date()}  End: {nav10.index[-1].date()}")
    print(f"  CAGR={m10['cagr']*100:+6.2f}%  Sh={m10['sharpe']:+5.2f}  DD={m10['max_dd']*100:+6.1f}%  Tot={m10['total']*100:+8.1f}%")
    print(f"  Median positions: {r10['equity']['n_pos'].median():.1f}  trades: {len(r10['trades'])}")

    # Persistent artifacts for primary run
    r10["equity"].to_csv(OUT/"primary_equity.csv")
    r10["trades"].to_csv(OUT/"primary_trades.csv", index=False)
    r10["rebal_log"].to_csv(OUT/"primary_rebal_log.csv", index=False)

    # Bluechip-20 same config
    print("\n[B20] Bluechip-20, as-specified")
    r20, panels20, ind20 = run_breakout_bluechip(BLUECHIP_20, "BC20-spec",
                                                   cost_per_side=30/10000.0, min_eligible=5)
    nav20 = r20["equity"]["nav"]
    m20 = wb.metrics_from_nav(nav20, "BC20 spec 30bps")
    print(f"  CAGR={m20['cagr']*100:+6.2f}%  Sh={m20['sharpe']:+5.2f}  DD={m20['max_dd']*100:+6.1f}%  Tot={m20['total']*100:+8.1f}%")

    # Bluechip-10 sensitivities
    print("\n[Sensitivity at varying cost]")
    sens_runs = {}
    sens_metrics = []
    for bps in [0, 5, 10, 20, 30, 50]:
        r, _, _ = run_breakout_bluechip(BLUECHIP_10, f"BC10@{bps}bps",
                                         cost_per_side=bps/10000.0)
        nav = r["equity"]["nav"]
        m = wb.metrics_from_nav(nav, f"@{bps}bps")
        sens_runs[bps] = nav
        sens_metrics.append(dict(bps_per_side=bps, **m, n_trades=len(r["trades"])))
        print(f"  @{bps:>2d}bps/side: CAGR={m['cagr']*100:+6.2f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+8.1f}%")

    print("\n[Sensitivity at varying ATR stop]")
    for atr in [2.0, 2.5, 3.0, 3.5, 4.0]:
        r, _, _ = run_breakout_bluechip(BLUECHIP_10, f"BC10 atr={atr}",
                                         cost_per_side=30/10000.0, atr_mult=atr)
        m = wb.metrics_from_nav(r["equity"]["nav"], f"atr={atr}")
        print(f"  atr={atr}: CAGR={m['cagr']*100:+6.2f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+8.1f}%")

    # Benchmarks on BC10 universe
    btc_nav = btc_bh(panels10, start_date, 30/10000.0)
    m_btc = wb.metrics_from_nav(btc_nav, "BTC B&H")
    print(f"\n[Benchmarks at 30bps]")
    print(f"  BTC-USDC B&H: CAGR={m_btc['cagr']*100:+6.2f}%  Sh={m_btc['sharpe']:+5.2f}  DD={m_btc['max_dd']*100:+6.1f}%  Tot={m_btc['total']*100:+8.1f}%")

    ma_nav = ma_5_40_daily_basket(panels10, ind10["eligible_universe"], start_date, 30/10000.0)
    m_ma = wb.metrics_from_nav(ma_nav, "MA(5/40) daily basket")
    print(f"  MA(5/40) daily EW basket: CAGR={m_ma['cagr']*100:+6.2f}%  Sh={m_ma['sharpe']:+5.2f}  DD={m_ma['max_dd']*100:+6.1f}%  Tot={m_ma['total']*100:+8.1f}%")

    # 50/50 blend BC10 breakout + BTC HODL
    blend = (nav10/nav10.iloc[0]*0.5 + btc_nav/btc_nav.iloc[0]*0.5)*100_000.0
    m_blend = wb.metrics_from_nav(blend, "50/50 Breakout+BTC")
    print(f"  50/50 Breakout + BTC HODL: CAGR={m_blend['cagr']*100:+6.2f}%  Sh={m_blend['sharpe']:+5.2f}  DD={m_blend['max_dd']*100:+6.1f}%  Tot={m_blend['total']*100:+8.1f}%")

    # ─── Figures ─────────────────────────────────────────────────────
    # Fig 1: master equity + drawdown
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios":[3,1]})
    ax = axes[0]
    ax.plot(nav10.index, nav10/1e3, lw=2.0, color="#1f77b4",
             label=f"Weekly Breakout v1 (BC-10)  Sh={m10['sharpe']:.2f}, DD={m10['max_dd']:.0%}, CAGR={m10['cagr']*100:.0f}%")
    ax.plot(nav20.index, nav20/1e3, lw=1.4, color="#1f77b4", ls=":",
             label=f"Weekly Breakout v1 (BC-20)  Sh={m20['sharpe']:.2f}, DD={m20['max_dd']:.0%}, CAGR={m20['cagr']*100:.0f}%")
    ax.plot(btc_nav.index, btc_nav/1e3, lw=1.6, color="#d62728", ls="--",
             label=f"BTC-USDC B&H  Sh={m_btc['sharpe']:.2f}, DD={m_btc['max_dd']:.0%}, CAGR={m_btc['cagr']*100:.0f}%")
    ax.plot(ma_nav.index, ma_nav/1e3, lw=1.4, color="#2ca02c",
             label=f"MA(5/40) daily basket (BC-10)  Sh={m_ma['sharpe']:.2f}, DD={m_ma['max_dd']:.0%}, CAGR={m_ma['cagr']*100:.0f}%")
    ax.plot(blend.index, blend/1e3, lw=1.4, color="#9467bd",
             label=f"50/50 Breakout + BTC HODL  Sh={m_blend['sharpe']:.2f}, DD={m_blend['max_dd']:.0%}, CAGR={m_blend['cagr']*100:.0f}%")
    ax.set_yscale("log"); ax.set_ylabel("NAV ($k, log)")
    ax.set_title("Weekly Breakout v1 — bluechip universe (start $100k, 30 bps/side cost)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    for nav, color, label in [
        (nav10, "#1f77b4", "Breakout BC-10"),
        (btc_nav, "#d62728", "BTC B&H"),
        (ma_nav, "#2ca02c", "MA(5/40) daily basket"),
    ]:
        dd = (nav/nav.cummax() - 1)*100
        ax.plot(dd.index, dd.values, lw=1.0, color=color, alpha=0.85, label=label)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("DD (%)"); ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG/"01_master_equity_drawdown.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Fig 2: cost sensitivity
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(sens_runs)))
    for (bps, nav), c in zip(sens_runs.items(), colors):
        ax.plot(nav.index, nav/1e3, lw=1.4, color=c, label=f"{bps} bps/side")
    ax.plot(btc_nav.index, btc_nav/1e3, lw=1.6, color="#d62728", ls="--", label="BTC B&H @30bps")
    ax.set_yscale("log")
    ax.set_title("Weekly Breakout v1 (BC-10) — cost sensitivity")
    ax.set_ylabel("NAV ($k, log)"); ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG/"02_cost_sensitivity.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Fig 3: calendar-year returns
    cy_s = calendar_year_returns(nav10) * 100
    cy_b = calendar_year_returns(btc_nav) * 100
    cy_m = calendar_year_returns(ma_nav) * 100
    common = sorted(set(cy_s.index) | set(cy_b.index) | set(cy_m.index))
    x = np.arange(len(common)); w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, [cy_s.get(y, np.nan) for y in common], w, label="Breakout BC-10", color="#1f77b4")
    ax.bar(x,     [cy_b.get(y, np.nan) for y in common], w, label="BTC B&H",        color="#d62728")
    ax.bar(x + w, [cy_m.get(y, np.nan) for y in common], w, label="MA(5/40) basket",color="#2ca02c")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in common])
    ax.set_ylabel("Calendar-year return (%)")
    ax.set_title("Calendar-year returns: Breakout BC-10 vs BTC B&H vs MA(5/40) daily basket")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG/"03_calendar_year.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Fig 4: positions over time
    eq = r10["equity"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax = axes[0]
    ax.fill_between(eq.index, eq["n_pos"], 0, color="#1f77b4", alpha=0.5)
    ax.set_ylabel("# positions"); ax.grid(True, alpha=0.3)
    ax.set_title("Weekly Breakout BC-10 — exposure timeline")
    ax = axes[1]
    ax.plot(eq.index, eq["gross_exposure"]*100, color="#1f77b4", lw=1.0)
    ax.fill_between(eq.index, eq["gross_exposure"]*100, 0, color="#1f77b4", alpha=0.2)
    ax.set_ylabel("Gross exposure (%)"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG/"04_exposure_timeline.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Fig 5: rolling 90-day return scatter vs BTC
    nav_s = nav10.copy()
    nav_b = btc_nav.copy()
    common_idx = nav_s.index.intersection(nav_b.index)
    rs = nav_s.loc[common_idx].pct_change(63).dropna()
    rb = nav_b.loc[common_idx].pct_change(63).dropna()
    common2 = rs.index.intersection(rb.index)
    rs = rs.loc[common2]; rb = rb.loc[common2]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(rb*100, rs*100, alpha=0.5, s=10, c="#1f77b4")
    lim = max(abs(rb.min()), abs(rb.max()), abs(rs.min()), abs(rs.max())) * 100
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="k", alpha=0.4, label="y = x")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    if len(rb) > 30:
        lr = LinearRegression().fit(rb.values.reshape(-1, 1), rs.values)
        slope = lr.coef_[0]; intercept = lr.intercept_
        xs = np.linspace(-lim/100, lim/100, 100)
        ys = lr.predict(xs.reshape(-1, 1))
        ax.plot(xs*100, ys*100, color="#d62728", lw=1.5,
                label=f"slope={slope:.2f}, intercept={intercept*100:.1f}%")
    ax.set_xlabel("BTC-USDC B&H 63-day return (%)")
    ax.set_ylabel("Breakout BC-10 63-day return (%)")
    ax.set_title("Synthetic-call payoff? Rolling 63-day returns vs BTC")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG/"05_scatter_vs_btc.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Save summary
    summary = dict(
        primary_bc10=m10, bc20=m20,
        btc_bh=m_btc, ma_5_40_daily=m_ma, blend_50_50=m_blend,
        cost_sensitivity=sens_metrics,
        start=str(start_date.date()), end=str(nav10.index[-1].date()),
        n_trades_bc10=len(r10["trades"]), n_rebals_bc10=len(r10["rebal_log"]),
        median_positions_bc10=float(eq["n_pos"].median()),
        median_gross_exposure_bc10=float(eq["gross_exposure"].median()),
    )
    with open(OUT/"summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] Wrote artifacts to {OUT}")


if __name__ == "__main__":
    main()
