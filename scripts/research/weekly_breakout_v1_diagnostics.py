#!/usr/bin/env python
"""Diagnostic comparison runs to isolate WHY Weekly Breakout v1 is losing:
  A) Longer momentum windows (60/180/365 days) instead of 20/40/90
  B) MA(5/40) on the SAME universe + same start date (apples-to-apples baseline)
  C) Equal-weight basket of all hold-eligible names (the "throw away ranking" baseline)
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto")
spec = importlib.util.spec_from_file_location("wb", ROOT/"scripts/research/weekly_breakout_v1.py")
wb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wb)

OUT = ROOT / "artifacts/research/weekly_breakout_v1_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def momentum_score_custom(C, eligible_mask, windows):
    """Recompute momentum score with custom windows."""
    parts = []
    for w in windows:
        m = C.pct_change(w, fill_method=None)
        masked = m.where(eligible_mask & m.notna())
        rk = masked.rank(axis=1, pct=True) * 100.0
        parts.append(rk)
    return sum(parts) / len(parts)


def run_breakout_with_windows(panels, ind, windows, liq_min):
    """Re-run the strategy but with custom momentum windows."""
    C = panels["C"]
    elig_uni = ind["eligible_universe"]
    score = momentum_score_custom(C, elig_uni, windows)
    new_ind = dict(ind)
    new_ind["mom_score"] = score
    new_ind["mom"] = {w: C.pct_change(w, fill_method=None) for w in windows}
    result = wb.backtest(panels, new_ind, params=dict(
        cost_per_side=30/10000.0, atr_stop_mult=3.0,
        use_atr_stop=True, require_breakout_entry=True,
        min_eligible_at_start=15,
    ))
    return result


def run_ma_5_40_basket(panels, eligible_mask, start_date, cost_per_side=30/10000.0):
    """MA(5/40) equal-weight basket of all hold-eligible names, weekly rebalanced.
    Same cost & universe & start date as breakout."""
    C = panels["C"]; O = panels["O"]
    ma5 = C.rolling(5).mean()
    ma40 = C.rolling(40).mean()
    pos = (ma5 > ma40) & eligible_mask  # asset eligible only if liquid+live+trending
    # Returns at daily frequency from close-to-close
    ret = C.pct_change(fill_method=None).fillna(0.0)
    # Weekly rebalance: weights are decided on Monday based on Sunday's signals,
    # applied throughout next week (then rebalanced next Monday).
    # We'll approximate with a simple weekly equal-weight rebalance.
    dates = C.index
    dates_mon = dates[(dates.dayofweek == 0) & (dates >= start_date)]
    # Compute weights row-by-row at rebalances using prior-day's `pos`, equal-weight.
    weights = pd.DataFrame(0.0, index=dates, columns=C.columns)
    last_w = pd.Series(0.0, index=C.columns)
    for mon in dates_mon:
        yd_idx = dates.get_loc(mon) - 1
        if yd_idx < 0:
            continue
        sel = pos.iloc[yd_idx]
        sel_syms = sel.index[sel.values]
        n = len(sel_syms)
        w = pd.Series(0.0, index=C.columns)
        if n > 0:
            w.loc[sel_syms] = 1.0 / n
        last_w = w
        # Apply weights from `mon` forward until next rebalance
        weights.loc[mon:, :] = w.values
    # Daily strat return = sum(w_i * ret_i) - turnover * cost_per_side
    daily_w = weights.shift(0)  # held through the day
    strat_r = (daily_w * ret).sum(axis=1)
    # Turnover at rebalance days
    tov = (daily_w.diff().abs()).sum(axis=1).fillna(0.0)
    strat_r = strat_r - tov * cost_per_side
    # Equity starting from start_date
    mask = strat_r.index >= start_date
    nav = (1 + strat_r[mask]).cumprod() * 100_000.0
    return nav


def main():
    print("=== Weekly Breakout v1 — diagnostic study ===\n")
    syms, bars = wb.load_universe()
    panels = wb.assemble_panels(syms, bars)
    print(f"  {len(panels['C'].columns)} pairs, {len(panels['C'])} rows\n")

    LIQ = 1_000_000.0
    ind = wb.compute_indicators(panels, LIQ)
    ind["mom_score"] = wb.momentum_score(ind["mom"], ind["eligible_universe"])

    # Common start date for all comparisons
    n_warm = 95
    elig_size = ind["eligible_universe"].sum(axis=1)
    candidate_starts = elig_size.index[(elig_size >= 15) & (elig_size.index >= panels["C"].index[n_warm])]
    start_date = candidate_starts[0]
    print(f"Common start: {start_date.date()}\n")

    runs = {}

    # A) Original spec (20/40/90 momentum windows)
    print("[A] Spec: 20/40/90-day momentum + breakout + 3-ATR stop")
    r = wb.backtest(panels, ind, params=dict(
        cost_per_side=30/10000.0, atr_stop_mult=3.0,
        use_atr_stop=True, require_breakout_entry=True, min_eligible_at_start=15))
    runs["A: Spec (20/40/90 mom)"] = r["equity"]["nav"]
    m = wb.metrics_from_nav(r["equity"]["nav"], "A")
    print(f"   CAGR={m['cagr']*100:+6.1f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+7.1f}%")

    # B) Longer momentum windows (60/180/365)
    print("\n[B] Long-horizon momentum: 60/180/365-day windows + breakout + 3-ATR stop")
    r = run_breakout_with_windows(panels, ind, windows=(60, 180, 365), liq_min=LIQ)
    runs["B: Long-horizon mom (60/180/365)"] = r["equity"]["nav"]
    m = wb.metrics_from_nav(r["equity"]["nav"], "B")
    print(f"   CAGR={m['cagr']*100:+6.1f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+7.1f}%")

    # B2) Medium: 30/90/180
    print("\n[B2] Medium momentum: 30/90/180-day windows + breakout + 3-ATR stop")
    r = run_breakout_with_windows(panels, ind, windows=(30, 90, 180), liq_min=LIQ)
    runs["B2: 30/90/180 mom"] = r["equity"]["nav"]
    m = wb.metrics_from_nav(r["equity"]["nav"], "B2")
    print(f"   CAGR={m['cagr']*100:+6.1f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+7.1f}%")

    # C) MA(5/40) on same universe (baseline comparison)
    print("\n[C] MA(5/40) equal-weight basket of all hold-eligible names")
    # Use the same eligibility filter
    elig = ind["eligible_universe"]
    nav_c = run_ma_5_40_basket(panels, elig, start_date)
    runs["C: MA(5/40) basket (baseline)"] = nav_c
    m = wb.metrics_from_nav(nav_c, "C")
    print(f"   CAGR={m['cagr']*100:+6.1f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+7.1f}%")

    # BTC bench
    bench = wb.compute_benchmarks(panels, ind, start_date)
    btc = bench["btc_bh"]
    m_b = wb.metrics_from_nav(btc, "BTC")
    print(f"\n[BTC] CAGR={m_b['cagr']*100:+6.1f}%  Sh={m_b['sharpe']:+5.2f}  DD={m_b['max_dd']*100:+6.1f}%  Tot={m_b['total']*100:+7.1f}%")

    # Figure
    fig, ax = plt.subplots(figsize=(13, 7))
    for label, nav in runs.items():
        nav_r = nav / nav.iloc[0] * 100
        ax.plot(nav_r.index, nav_r.values, lw=1.6, alpha=0.9, label=label)
    if not btc.empty:
        b_r = btc / btc.iloc[0] * 100
        ax.plot(b_r.index, b_r.values, color="#d62728", lw=2.0, ls="--", label="BTC-USDC B&H")
    ax.set_yscale("log")
    ax.set_ylabel("Indexed NAV (start = 100, log)")
    ax.set_title("Weekly Breakout v1 — diagnostics: momentum horizons & baseline comparison")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT/"diagnostics_equity.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {OUT/'diagnostics_equity.png'}")


if __name__ == "__main__":
    main()
