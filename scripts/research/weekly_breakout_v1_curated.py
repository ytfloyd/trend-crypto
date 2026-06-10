#!/usr/bin/env python
"""Weekly Breakout v1 on the curated L1+L2+DeFi universe (same set as the
MA(5/40) study used for its 'high-quality basket').

Three runs:
  (i)   As-specified: 30 bps per side, 3-ATR stop, breakout filter ON
  (ii)  Same strategy but at 10 bps per side (matches the MA(5/40) study's 20 bps RT)
  (iii) MA(5/40) equal-weight basket of the same universe (apples-to-apples benchmark)
        at both cost levels.

Outputs: equity curves + metrics CSV + comparison figure.
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

ROOT = Path("/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto")
spec = importlib.util.spec_from_file_location("wb", ROOT/"scripts/research/weekly_breakout_v1.py")
wb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wb)

OUT = ROOT / "artifacts/research/weekly_breakout_v1_curated"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def load_hq_symbols() -> list[str]:
    df = pd.read_csv(ROOT/"artifacts/research/ma_5_40_usdc_universe/usdc_universe_ma_5_40_results_with_category.csv")
    mask = df["category"].isin(["L1", "L2", "DeFi"])
    return df.loc[mask, "symbol"].tolist()


def run_ma_5_40_basket(panels, eligible_mask, start_date, cost_per_side):
    """Equal-weight MA(5/40) on hold-eligible names, weekly rebalanced."""
    C = panels["C"]
    ma5 = C.rolling(5).mean(); ma40 = C.rolling(40).mean()
    pos = (ma5 > ma40) & eligible_mask
    ret = C.pct_change(fill_method=None).fillna(0.0)
    dates = C.index
    dates_mon = dates[(dates.dayofweek == 0) & (dates >= start_date)]
    weights = pd.DataFrame(0.0, index=dates, columns=C.columns)
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
        weights.loc[mon:, :] = w.values
    strat_r = (weights * ret).sum(axis=1)
    tov = weights.diff().abs().sum(axis=1).fillna(0.0)
    strat_r = strat_r - tov * cost_per_side
    mask = strat_r.index >= start_date
    nav = (1 + strat_r[mask]).cumprod() * 100_000.0
    return nav


def main():
    print("=== Weekly Breakout v1 on curated L1+L2+DeFi universe ===\n")
    hq_syms = load_hq_symbols()
    print(f"Curated set: {len(hq_syms)} L1+L2+DeFi pairs\n")
    syms, bars = wb.load_universe(restrict_to=hq_syms)
    panels = wb.assemble_panels(syms, bars)
    print(f"After structural filters: {len(panels['C'].columns)} pairs\n")

    LIQ = 1_000_000.0
    ind = wb.compute_indicators(panels, LIQ)
    ind["mom_score"] = wb.momentum_score(ind["mom"], ind["eligible_universe"])

    n_warm = 95
    elig_size = ind["eligible_universe"].sum(axis=1)
    candidate_starts = elig_size.index[(elig_size >= 10) & (elig_size.index >= panels["C"].index[n_warm])]
    start_date = candidate_starts[0]
    print(f"Backtest start: {start_date.date()}  (first day with ≥10 eligible names)\n")

    runs = {}

    for cost_label, bps_per_side in [("@30bps/side (as spec)", 30), ("@10bps/side (MA(5/40) parity)", 10)]:
        print(f"--- Cost level: {cost_label} ---")
        # Weekly breakout
        r = wb.backtest(panels, ind, params=dict(
            cost_per_side=bps_per_side / 10_000.0,
            atr_stop_mult=3.0, use_atr_stop=True,
            require_breakout_entry=True,
            min_eligible_at_start=10,
        ))
        nav_b = r["equity"]["nav"]
        m = wb.metrics_from_nav(nav_b, f"Breakout {cost_label}")
        print(f"  Breakout v1:        CAGR={m['cagr']*100:+7.1f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+9.1f}%  medpos={r['equity']['n_pos'].median():.1f}")
        runs[f"Breakout v1 {cost_label}"] = nav_b
        r["equity"].to_csv(OUT/f"breakout_equity_{bps_per_side}bps.csv")
        r["trades"].to_csv(OUT/f"breakout_trades_{bps_per_side}bps.csv", index=False)

        # MA(5/40) basket on same universe
        nav_m = run_ma_5_40_basket(panels, ind["eligible_universe"], start_date,
                                    cost_per_side=bps_per_side / 10_000.0)
        m_m = wb.metrics_from_nav(nav_m, f"MA(5/40) {cost_label}")
        print(f"  MA(5/40) EW basket: CAGR={m_m['cagr']*100:+7.1f}%  Sh={m_m['sharpe']:+5.2f}  DD={m_m['max_dd']*100:+6.1f}%  Tot={m_m['total']*100:+9.1f}%")
        runs[f"MA(5/40) EW basket {cost_label}"] = nav_m
        print()

    bench = wb.compute_benchmarks(panels, ind, start_date)
    btc = bench["btc_bh"]
    m_b = wb.metrics_from_nav(btc, "BTC")
    print(f"BTC-USDC B&H:  CAGR={m_b['cagr']*100:+7.1f}%  Sh={m_b['sharpe']:+5.2f}  DD={m_b['max_dd']*100:+6.1f}%  Tot={m_b['total']*100:+9.1f}%")

    # Metrics CSV
    rows = []
    for label, nav in runs.items():
        m = wb.metrics_from_nav(nav, label)
        rows.append(dict(label=label, **{k: m[k] for k in ("cagr","vol","sharpe","max_dd","total","years")}))
    rows.append(dict(label="BTC-USDC B&H",
                     cagr=m_b["cagr"], vol=m_b["vol"], sharpe=m_b["sharpe"],
                     max_dd=m_b["max_dd"], total=m_b["total"], years=m_b["years"]))
    pd.DataFrame(rows).to_csv(OUT/"summary_metrics.csv", index=False)
    print(f"\nWrote {OUT/'summary_metrics.csv'}")

    # Figure
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios":[3,1]})
    colors = {
        "Breakout v1 @30bps/side (as spec)":  "#1f77b4",
        "Breakout v1 @10bps/side (MA(5/40) parity)": "#1f77b4",
        "MA(5/40) EW basket @30bps/side (as spec)": "#2ca02c",
        "MA(5/40) EW basket @10bps/side (MA(5/40) parity)": "#2ca02c",
    }
    ls_map = {
        "@30bps/side (as spec)": "-",
        "@10bps/side (MA(5/40) parity)": ":",
    }

    ax = axes[0]
    for label, nav in runs.items():
        nav_r = nav / nav.iloc[0] * 100
        color = colors.get(label, "k")
        ls = "-"
        for k, v in ls_map.items():
            if k in label:
                ls = v; break
        ax.plot(nav_r.index, nav_r.values, lw=1.6, alpha=0.95, color=color,
                ls=ls, label=label)
    if not btc.empty:
        b_r = btc / btc.iloc[0] * 100
        ax.plot(b_r.index, b_r.values, color="#d62728", lw=2.0, ls="--", label="BTC-USDC B&H")
    ax.set_yscale("log")
    ax.set_ylabel("Indexed NAV (start = 100, log)")
    ax.set_title("Weekly Breakout v1 vs MA(5/40) EW basket — L1+L2+DeFi curated universe")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=9)

    ax = axes[1]
    for label, nav in runs.items():
        color = colors.get(label, "k")
        ls = "-"
        for k, v in ls_map.items():
            if k in label:
                ls = v; break
        dd = (nav / nav.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, lw=1.0, color=color, ls=ls, alpha=0.8, label=label)
    if not btc.empty:
        dd_b = (btc / btc.cummax() - 1) * 100
        ax.plot(dd_b.index, dd_b.values, color="#d62728", lw=1.2, ls="--", label="BTC B&H")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("DD (%)"); ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG/"curated_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG/'curated_comparison.png'}")


if __name__ == "__main__":
    main()
