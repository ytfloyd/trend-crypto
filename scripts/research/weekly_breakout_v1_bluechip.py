#!/usr/bin/env python
"""Weekly Breakout v1 across multiple universe definitions to pin down where (if at all)
the strategy generates positive risk-adjusted return.

Universes tested:
  U1: All Coinbase USDC pairs surviving structural filters (~273)
  U2: L1+L2+DeFi curated subset (85, from prior MA(5/40) study)
  U3: Top-10 caps blue-chip ("BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, LINK, DOT, LTC")
  U4: Top-20 caps blue-chip (adds BCH, ATOM, ALGO, NEAR, OP, ARB, UNI, AAVE, MATIC, FIL)

For each universe: as-spec (30 bps/side, 3 ATR, 20/40/90 mom) + variant with 60/180/365 mom windows.
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

OUT = ROOT / "artifacts/research/weekly_breakout_v1_universes"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Universe definitions
BLUECHIP_10 = ["BTC-USDC","ETH-USDC","SOL-USDC","XRP-USDC","ADA-USDC",
                "DOGE-USDC","AVAX-USDC","LINK-USDC","DOT-USDC","LTC-USDC"]
BLUECHIP_20 = BLUECHIP_10 + ["BCH-USDC","ATOM-USDC","ALGO-USDC","NEAR-USDC","OP-USDC",
                              "ARB-USDC","UNI-USDC","AAVE-USDC","MATIC-USDC","FIL-USDC"]


def load_hq_symbols() -> list[str]:
    df = pd.read_csv(ROOT/"artifacts/research/ma_5_40_usdc_universe/usdc_universe_ma_5_40_results_with_category.csv")
    mask = df["category"].isin(["L1", "L2", "DeFi"])
    return df.loc[mask, "symbol"].tolist()


def momentum_score_custom(C, eligible_mask, windows):
    parts = []
    for w in windows:
        m = C.pct_change(w, fill_method=None)
        masked = m.where(eligible_mask & m.notna())
        rk = masked.rank(axis=1, pct=True) * 100.0
        parts.append(rk)
    return sum(parts) / len(parts)


def run_breakout(panels, mom_windows, liq_min, cost_per_side, min_eligible,
                  atr_mult=3.0, max_pos=20, max_weight=0.15):
    ind = wb.compute_indicators(panels, liq_min)
    C = panels["C"]
    ind["mom"] = {w: C.pct_change(w, fill_method=None) for w in mom_windows}
    ind["mom_score"] = momentum_score_custom(C, ind["eligible_universe"], mom_windows)
    result = wb.backtest(panels, ind, params=dict(
        cost_per_side=cost_per_side,
        atr_stop_mult=atr_mult, use_atr_stop=True,
        require_breakout_entry=True, max_positions=max_pos, max_weight=max_weight,
        min_eligible_at_start=min_eligible,
    ))
    return result, ind


def run_btc_bench(panels, start_date, cost_per_side=30/10000.0):
    if "BTC-USDC" not in panels["C"].columns:
        return pd.Series(dtype=float)
    o = panels["O"]["BTC-USDC"]; c = panels["C"]["BTC-USDC"]
    o, c = o.align(c, join="inner")
    o = o.loc[o.index >= start_date]; c = c.loc[c.index >= start_date]
    r = (c/o - 1).fillna(0.0)
    return 100_000 * (1+r).cumprod() * (1 - cost_per_side)


def main():
    print("=== Weekly Breakout v1 — universe study ===\n")
    universes = {
        "U1: All USDC (~273)":           None,
        "U2: L1+L2+DeFi curated (85)":   load_hq_symbols(),
        "U3: Bluechip-10":               BLUECHIP_10,
        "U4: Bluechip-20":               BLUECHIP_20,
    }
    mom_variants = {
        "Spec mom (20/40/90)":     (20, 40, 90),
        "Long mom (60/180/365)":   (60, 180, 365),
    }

    rows = []
    runs_for_plot = {}

    for uname, syms_filter in universes.items():
        syms, bars = wb.load_universe(restrict_to=syms_filter)
        if len(syms) < 5:
            print(f"  Skipping {uname} (only {len(syms)} pairs)\n")
            continue
        panels = wb.assemble_panels(syms, bars)
        # Pick min_eligible based on universe size
        min_elig = max(3, min(15, int(len(syms) * 0.4)))
        if syms_filter is not None and len(syms_filter) <= 20:
            min_elig = max(3, int(len(syms) * 0.3))

        for mom_label, windows in mom_variants.items():
            label = f"{uname} | {mom_label}"
            print(f"--- {label} (min_eligible_at_start={min_elig})")
            try:
                result, ind = run_breakout(panels, windows, 1_000_000.0,
                                            30/10000.0, min_elig)
                nav = result["equity"]["nav"]
                m = wb.metrics_from_nav(nav, label)
                print(f"   CAGR={m['cagr']*100:+7.1f}%  Sh={m['sharpe']:+5.2f}  "
                      f"DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+9.1f}%  "
                      f"medpos={result['equity']['n_pos'].median():.1f}")
                rows.append(dict(label=label, **m, n_trades=len(result["trades"]),
                                  n_rebals=len(result["rebal_log"]),
                                  median_pos=float(result["equity"]["n_pos"].median()),
                                  median_gross=float(result["equity"]["gross_exposure"].median())))
                runs_for_plot[label] = nav
            except Exception as e:
                print(f"   FAILED: {e}")

        # BTC bench for this universe's data
        if "BTC-USDC" in panels["C"].columns:
            # Use the earliest start across both variants for this universe
            first_nav_start = min(
                (runs_for_plot[k].index[0] for k in runs_for_plot if k.startswith(uname)),
                default=None,
            )
            if first_nav_start is not None and uname == "U1: All USDC (~273)":
                btc_nav = run_btc_bench(panels, first_nav_start)
                runs_for_plot["BTC-USDC B&H"] = btc_nav
        print()

    pd.DataFrame(rows).to_csv(OUT/"universe_results.csv", index=False)
    print(f"\nWrote {OUT/'universe_results.csv'}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = {
        "U1": "#1f77b4", "U2": "#2ca02c", "U3": "#ff7f0e", "U4": "#9467bd",
    }
    for label, nav in runs_for_plot.items():
        nav_r = nav / nav.iloc[0] * 100
        if label == "BTC-USDC B&H":
            ax.plot(nav_r.index, nav_r.values, lw=2.2, color="#d62728", ls="--",
                    label=label)
            continue
        uname = label.split(" | ")[0]
        prefix = uname.split(":")[0]
        color = cmap.get(prefix, "k")
        ls = ":" if "Long mom" in label else "-"
        ax.plot(nav_r.index, nav_r.values, lw=1.4, alpha=0.85, color=color, ls=ls,
                label=label)
    ax.set_yscale("log")
    ax.set_ylabel("Indexed NAV (start = 100, log)")
    ax.set_title("Weekly Breakout v1 — universe study (30 bps/side cost)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG/"universe_study.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG/'universe_study.png'}")


if __name__ == "__main__":
    main()
