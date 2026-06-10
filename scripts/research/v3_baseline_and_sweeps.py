#!/usr/bin/env python
"""Reality-check the fixed-stop baseline and run two requested sweeps.

Produces a single artifact `v3_baseline.json` with:
  - Risk metrics for fixed, trailing, and BTC HODL (Sharpe, Sortino, Calmar,
    MAR, vol, max DD)
  - Breakout-window sweep (5 / 10 / 20 day) at the spec ATR stop
  - No-ATR-stop variant
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto")
spec = importlib.util.spec_from_file_location("wb", ROOT/"scripts/research/weekly_breakout_v1.py")
wb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wb)

OUT = ROOT / "artifacts/research/weekly_breakout_v3"
OUT.mkdir(parents=True, exist_ok=True)

BC = ["BTC-USDC","ETH-USDC","SOL-USDC","XRP-USDC","ADA-USDC",
       "DOGE-USDC","AVAX-USDC","LINK-USDC","DOT-USDC","LTC-USDC"]


def risk_metrics(nav: pd.Series) -> dict:
    """Full risk-metric set on a daily NAV series.
    Uses ANN=365 (calendar days) which is correct for crypto 24/7 markets."""
    ANN = 365.0
    rets = nav.pct_change(fill_method=None).dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1/years) - 1
    vol = rets.std() * np.sqrt(ANN)
    sharpe = (rets.mean() * ANN) / vol if vol > 0 else 0.0
    downside = rets[rets < 0]
    sortino = (rets.mean() * ANN) / (downside.std() * np.sqrt(ANN)) if len(downside) and downside.std() > 0 else 0.0
    peak = nav.cummax()
    dd = (nav / peak - 1)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    # Pain index and Ulcer index
    ulcer = np.sqrt((dd**2).mean()) if len(dd) else 0.0
    # Days in drawdown > 20%
    days_dd20 = int((dd < -0.20).sum())
    days_dd50 = int((dd < -0.50).sum())
    return dict(
        cagr=float(cagr), vol=float(vol), sharpe=float(sharpe),
        sortino=float(sortino), max_dd=float(max_dd), calmar=float(calmar),
        ulcer=float(ulcer), years=float(years),
        total_return=float(nav.iloc[-1] / nav.iloc[0] - 1),
        final_nav=float(nav.iloc[-1]),
        days_dd20=days_dd20, days_dd50=days_dd50,
    )


def run_one(panels, ind, trailing=False, atr_mult=3.0, use_atr=True,
             breakout_window=5):
    """Run a variant. If breakout_window != 5, recompute breakout-high
    using the requested window."""
    if breakout_window != 5:
        # Rebuild breakout high/low at the requested window.
        # Engine reads close-vs-prior-window-high (NOT high-vs-prior-window-high),
        # so build from C consistently with compute_indicators.
        C = panels["C"]
        bo_high = C.rolling(breakout_window).max().shift(1)
        bo_low  = C.rolling(breakout_window).min().shift(1)
        ind2 = {**ind, "bo_high": bo_high, "bo_low": bo_low}
    else:
        ind2 = ind

    return wb.backtest(panels, ind2, params=dict(
        cost_per_side=30/10000.0, atr_stop_mult=atr_mult, use_atr_stop=use_atr,
        require_breakout_entry=True, trailing_stop=trailing,
        min_eligible_at_start=3))


def main():
    print("Loading universe…")
    syms, bars = wb.load_universe(restrict_to=BC)
    panels = wb.assemble_panels(syms, bars)
    ind = wb.compute_indicators(panels, 1_000_000.0)
    ind["mom_score"] = wb.momentum_score(ind["mom"], ind["eligible_universe"])

    results = {}

    # ── 1. Baseline: fixed vs trailing vs BTC HODL ────────────────────
    print("\n── Baseline metrics ──")
    r_fix = run_one(panels, ind, trailing=False, atr_mult=3.0, use_atr=True)
    r_tr  = run_one(panels, ind, trailing=True,  atr_mult=3.0, use_atr=True)
    nav_fix = r_fix["equity"]["nav"]
    nav_tr  = r_tr["equity"]["nav"]
    # BTC HODL: re-base BTC price to $100k start
    btc = panels["C"]["BTC-USDC"].reindex(nav_fix.index, method="ffill").dropna()
    btc_hodl = (btc / btc.iloc[0]) * 100_000

    results["fixed_3atr"]    = risk_metrics(nav_fix)
    results["trailing_3atr"] = risk_metrics(nav_tr)
    results["btc_hodl"]      = risk_metrics(btc_hodl)

    print(f"{'metric':<18} {'fixed':>14} {'trailing':>14} {'BTC HODL':>14}")
    for k in ["cagr","vol","sharpe","sortino","max_dd","calmar","ulcer"]:
        fmt = "{:>14.2%}" if k in ("cagr","vol","max_dd") else "{:>14.2f}"
        print(f"{k:<18}", fmt.format(results['fixed_3atr'][k]),
                          fmt.format(results['trailing_3atr'][k]),
                          fmt.format(results['btc_hodl'][k]))
    for k in ["total_return","days_dd20","days_dd50","final_nav"]:
        if k == "final_nav":
            print(f"{k:<18} {results['fixed_3atr'][k]:>14,.0f} "
                  f"{results['trailing_3atr'][k]:>14,.0f} "
                  f"{results['btc_hodl'][k]:>14,.0f}")
        elif k == "total_return":
            print(f"{k:<18} {results['fixed_3atr'][k]:>13.0%}  "
                  f"{results['trailing_3atr'][k]:>13.0%}  "
                  f"{results['btc_hodl'][k]:>13.0%}")
        else:
            print(f"{k:<18} {results['fixed_3atr'][k]:>14} "
                  f"{results['trailing_3atr'][k]:>14} "
                  f"{results['btc_hodl'][k]:>14}")

    # ── 2. Breakout-window sweep ──────────────────────────────────────
    print("\n── Breakout-window sweep (fixed 3×ATR stop) ──")
    sweep_bo = {}
    for bw in [5, 10, 20]:
        r = run_one(panels, ind, trailing=False, atr_mult=3.0, use_atr=True,
                     breakout_window=bw)
        nav = r["equity"]["nav"]
        m = risk_metrics(nav)
        sweep_bo[f"breakout_{bw}d"] = m
        print(f"  {bw:>2}-day breakout: Sharpe {m['sharpe']:.2f}  CAGR {m['cagr']:+.1%}  "
              f"DD {m['max_dd']:.1%}  Total {m['total_return']:+.0%}  Final ${m['final_nav']:,.0f}")
    results["breakout_sweep"] = sweep_bo

    # ── 3. No-ATR-stop variant ────────────────────────────────────────
    print("\n── No-stop variant (fixed-stop spec but stops disabled) ──")
    r_nostop = run_one(panels, ind, trailing=False, atr_mult=3.0, use_atr=False)
    nav_nostop = r_nostop["equity"]["nav"]
    m_nostop = risk_metrics(nav_nostop)
    results["no_stop"] = m_nostop
    print(f"  Sharpe {m_nostop['sharpe']:.2f}  CAGR {m_nostop['cagr']:+.1%}  "
          f"DD {m_nostop['max_dd']:.1%}  Total {m_nostop['total_return']:+.0%}  "
          f"Final ${m_nostop['final_nav']:,.0f}")

    # Save
    with open(OUT/"v3_baseline.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {OUT/'v3_baseline.json'}")

    # Also save the NAV series for plotting
    pd.DataFrame({
        "fixed":  nav_fix,
        "trail":  nav_tr,
        "btc":    btc_hodl,
        "no_stop": nav_nostop,
    }).to_csv(OUT/"v3_navs.csv")
    print(f"Wrote {OUT/'v3_navs.csv'}")


if __name__ == "__main__":
    main()
