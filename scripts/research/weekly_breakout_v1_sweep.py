#!/usr/bin/env python
"""Sensitivity sweep over the Weekly Breakout v1 strategy.

Varies:
  - Liquidity threshold ($500k / $1M / $5M / $10M median $-volume)
  - ATR stop multiplier (2.0 / 3.0 / disabled)
  - Whether breakout-up is required for entry (True / False)
  - Cost per side (10 / 20 / 30 bps)

Writes summary CSV + a comparison figure.
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

OUT = ROOT / "artifacts/research/weekly_breakout_v1_sweep"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def run_one(syms, bars, panels, *,
            liq_min: float,
            atr_mult: float | None,
            require_breakout: bool,
            cost_bps_per_side: float,
            min_eligible: int = 15,
            label: str = "") -> dict:
    """Run a single backtest configuration. Returns metrics + nav."""
    ind = wb.compute_indicators(panels, liq_min)
    ind["mom_score"] = wb.momentum_score(ind["mom"], ind["eligible_universe"])
    params = dict(
        cost_per_side=cost_bps_per_side / 10_000.0,
        atr_stop_mult=atr_mult if atr_mult is not None else 3.0,
        use_atr_stop=(atr_mult is not None),
        require_breakout_entry=require_breakout,
        min_eligible_at_start=min_eligible,
    )
    result = wb.backtest(panels, ind, params=params)
    eq = result["equity"]
    if eq.empty:
        return dict(label=label, error="empty")
    nav = eq["nav"]
    m = wb.metrics_from_nav(nav, label)

    bench = wb.compute_benchmarks(panels, ind, nav.index[0])
    btc = bench["btc_bh"]
    m_btc = wb.metrics_from_nav(btc, "BTC-USDC B&H")

    return dict(
        label=label,
        liq_min=liq_min, atr_mult=atr_mult, require_breakout=require_breakout,
        cost_bps_per_side=cost_bps_per_side,
        start=str(nav.index[0].date()), end=str(nav.index[-1].date()),
        years=m["years"],
        cagr=m["cagr"], vol=m["vol"], sharpe=m["sharpe"], max_dd=m["max_dd"],
        total=m["total"], final=m["final"],
        n_trades=len(result["trades"]), n_rebalances=len(result["rebal_log"]),
        median_pos=float(eq["n_pos"].median()),
        median_gross=float(eq["gross_exposure"].median()),
        median_cash=float(eq["cash_pct"].median()),
        btc_cagr=m_btc["cagr"], btc_sharpe=m_btc["sharpe"], btc_total=m_btc["total"],
        edge_vs_btc=m["total"] - m_btc["total"],
        nav=nav, btc_nav=btc,
    )


def main():
    print("=== Weekly Breakout v1 — sensitivity sweep ===\n")
    print("Loading lake once (shared across runs)...")
    syms, bars = wb.load_universe()
    panels = wb.assemble_panels(syms, bars)
    print(f"  {len(panels['C'].columns)} pairs, {len(panels['C'])} rows\n")

    configs = []

    # 1. As-specified baseline at varying liquidity thresholds
    for liq in [500_000, 1_000_000, 5_000_000, 10_000_000]:
        configs.append(dict(label=f"AS-SPEC liq=${int(liq/1000)}k",
                            liq_min=liq, atr_mult=3.0,
                            require_breakout=True, cost_bps_per_side=30))

    # 2. ATR multiplier sweep at $5M liquidity (largest realistic universe)
    for atr_m in [2.0, 3.0, 4.0, None]:
        lbl = f"$5M liq, atr={atr_m if atr_m else 'OFF'}, req_bo"
        configs.append(dict(label=lbl, liq_min=5_000_000, atr_mult=atr_m,
                            require_breakout=True, cost_bps_per_side=30))

    # 3. Drop the breakout filter (just trend + momentum + universe)
    for liq in [1_000_000, 5_000_000]:
        configs.append(dict(label=f"NO-BO liq=${int(liq/1000)}k",
                            liq_min=liq, atr_mult=3.0,
                            require_breakout=False, cost_bps_per_side=30))

    # 4. Lower cost (10 bps per side = 20 bps RT, matches MA(5/40) study)
    for liq in [1_000_000, 5_000_000]:
        configs.append(dict(label=f"LOW-COST liq=${int(liq/1000)}k",
                            liq_min=liq, atr_mult=3.0,
                            require_breakout=True, cost_bps_per_side=10))

    # 5. No ATR stop, no breakout (closest to "pure trend + cross-sectional mom rank")
    for liq in [1_000_000, 5_000_000]:
        configs.append(dict(label=f"PURE-TREND liq=${int(liq/1000)}k",
                            liq_min=liq, atr_mult=None,
                            require_breakout=False, cost_bps_per_side=30))

    rows = []
    navs = {}
    btc_nav_ref = None
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg['label']}")
        out = run_one(syms, bars, panels, **cfg)
        if "error" in out:
            print(f"   ERROR: {out['error']}")
            continue
        rows.append({k: v for k, v in out.items() if k not in ("nav", "btc_nav")})
        navs[cfg["label"]] = out["nav"]
        if btc_nav_ref is None or len(out["btc_nav"]) > len(btc_nav_ref):
            btc_nav_ref = out["btc_nav"]
        print(f"   CAGR={out['cagr']*100:+6.1f}%  Sh={out['sharpe']:+5.2f}  "
              f"DD={out['max_dd']*100:+6.1f}%  Tot={out['total']*100:+8.1f}%  "
              f"medpos={out['median_pos']:.1f}  trades={out['n_trades']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT/"sweep_results.csv", index=False)
    print(f"\nWrote {OUT/'sweep_results.csv'}")
    print(f"\n--- TOP 5 by Sharpe ---")
    print(df.nlargest(5, "sharpe")[
        ["label","sharpe","cagr","max_dd","total","median_pos","median_gross","n_trades"]
    ].round(3).to_string(index=False))

    # Comparison figure
    fig, ax = plt.subplots(figsize=(14, 8))
    for label, nav in navs.items():
        nav_r = nav / nav.iloc[0] * 100
        ax.plot(nav_r.index, nav_r.values, lw=1.1, alpha=0.85, label=label)
    if btc_nav_ref is not None and not btc_nav_ref.empty:
        b = btc_nav_ref / btc_nav_ref.iloc[0] * 100
        ax.plot(b.index, b.values, color="#d62728", lw=2.0, ls="--", label="BTC-USDC B&H")
    ax.set_yscale("log")
    ax.set_ylabel("Indexed NAV (start = 100, log)")
    ax.set_title("Weekly Breakout v1 — sensitivity sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG/"sweep_equity.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG/'sweep_equity.png'}")


if __name__ == "__main__":
    main()
