#!/usr/bin/env python
"""Stop-design comparison on the FULL USDC universe (not just BC-10).

Runs the three exit specifications (no-stop, fixed 3×ATR, trailing 3×ATR) on
every USDC-quoted pair that passes the structural filters in
weekly_breakout_v1.py (365-day history, 90% coverage, $500k median DV,
non-stablecoin). Reports headline metrics and writes JSON + CSV.

Notes on universe construction:
  - Pairs are filtered point-in-time-correctly *within* the backtest via
    eligible_universe = live & liquid (a pair only counts on dates where
    it has a price and meets the liquidity floor).
  - However, the *set* of pairs considered is sourced from the lake's
    currently-listed pairs, so delisted pairs are not represented. This is
    a survivorship bias on absolute returns; it does not affect the
    fixed-vs-trailing *delta* this script studies.
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

OUT = ROOT / "artifacts/research/weekly_breakout_v4"
OUT.mkdir(parents=True, exist_ok=True)

ANN = 365.0


def risk_metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
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
    return dict(
        years=float(years),
        cagr=float(cagr), vol=float(vol), sharpe=float(sharpe), sortino=float(sortino),
        max_dd=float(max_dd), calmar=float(calmar),
        total_return=float(nav.iloc[-1] / nav.iloc[0] - 1),
        final_nav=float(nav.iloc[-1]),
        days_dd20=int((dd < -0.20).sum()),
        days_dd50=int((dd < -0.50).sum()),
    )


def run_variant(panels, ind, trailing=False, use_atr=True):
    return wb.backtest(panels, ind, params=dict(
        cost_per_side=30/10000.0, atr_stop_mult=3.0,
        use_atr_stop=use_atr, require_breakout_entry=True,
        trailing_stop=trailing,
        # leave min_eligible_at_start at the default 15 (vs 3 for BC-10) so
        # the backtest only starts once the wide universe is broad enough
        # for top-N selection to be meaningful
    ))


def hodl_nav(panels, ref_index, symbol="BTC-USDC", start_nav=100_000.0):
    px = panels["C"][symbol].reindex(ref_index, method="ffill").dropna()
    return (px / px.iloc[0]) * start_nav


def main():
    print("Loading FULL USDC universe (no restrict_to)...")
    syms, bars = wb.load_universe()  # all USDC pairs
    panels = wb.assemble_panels(syms, bars)
    ind = wb.compute_indicators(panels, wb.LIQ_MIN_USD)
    ind["mom_score"] = wb.momentum_score(ind["mom"], ind["eligible_universe"])
    print(f"Universe size: {len(syms)} USDC pairs")

    print("\nRunning three variants on full universe...")
    r_ns    = run_variant(panels, ind, trailing=False, use_atr=False)
    r_fix   = run_variant(panels, ind, trailing=False, use_atr=True)
    r_trail = run_variant(panels, ind, trailing=True,  use_atr=True)

    nav_ns    = r_ns["equity"]["nav"]
    nav_fix   = r_fix["equity"]["nav"]
    nav_trail = r_trail["equity"]["nav"]
    btc_nav   = hodl_nav(panels, nav_fix.index)

    m = dict(
        no_stop=risk_metrics(nav_ns),
        fixed_3atr=risk_metrics(nav_fix),
        trailing_3atr=risk_metrics(nav_trail),
        btc_hodl=risk_metrics(btc_nav),
        universe_size=len(syms),
        start_date=str(nav_fix.index[0].date()),
        end_date=str(nav_fix.index[-1].date()),
    )

    # Stop-cost trade
    base_cagr = m["no_stop"]["cagr"] * 100
    base_dd   = abs(m["no_stop"]["max_dd"]) * 100
    def cost_per_pp(label, mv):
        dd_red = base_dd - abs(mv["max_dd"]*100)
        cost   = base_cagr - mv["cagr"]*100
        return dict(cagr_cost_pp=cost, dd_reduction_pp=dd_red,
                     cost_per_pp_bps=(cost*100/dd_red) if dd_red>0 else None)
    m["stop_cost_trade"] = dict(
        fixed=cost_per_pp("fixed", m["fixed_3atr"]),
        trailing=cost_per_pp("trailing", m["trailing_3atr"]),
    )

    # Print headline
    print(f"\nBacktest window: {m['start_date']} → {m['end_date']} "
          f"({m['no_stop']['years']:.1f} years)")
    print(f"{'variant':<14} {'Sharpe':>7} {'CAGR':>8} {'Vol':>7} {'MaxDD':>8} {'Calmar':>7} {'Total':>9}")
    for k in ["no_stop","fixed_3atr","trailing_3atr","btc_hodl"]:
        v = m[k]
        print(f"{k:<14} {v['sharpe']:>7.2f} {v['cagr']:>+7.1%} {v['vol']:>6.1%} "
              f"{v['max_dd']:>7.1%} {v['calmar']:>7.2f} {v['total_return']:>+8.0%}")

    sc = m["stop_cost_trade"]
    if sc["fixed"]["cost_per_pp_bps"] is not None:
        print(f"\nStop-cost trade (vs no-stop):")
        print(f"  fixed:    {sc['fixed']['cagr_cost_pp']:+.1f}pp CAGR cost, "
              f"{sc['fixed']['dd_reduction_pp']:+.1f}pp DD protection, "
              f"= {sc['fixed']['cost_per_pp_bps']:.0f} bps/pp")
    if sc["trailing"]["cost_per_pp_bps"] is not None:
        print(f"  trailing: {sc['trailing']['cagr_cost_pp']:+.1f}pp CAGR cost, "
              f"{sc['trailing']['dd_reduction_pp']:+.1f}pp DD protection, "
              f"= {sc['trailing']['cost_per_pp_bps']:.0f} bps/pp")
        if sc["fixed"]["cost_per_pp_bps"]:
            ratio = sc["trailing"]["cost_per_pp_bps"] / sc["fixed"]["cost_per_pp_bps"]
            print(f"  → trailing is {ratio:.1f}× worse trade")

    # Save
    with open(OUT/"v4_full_usdc.json", "w") as f:
        json.dump(m, f, indent=2, default=str)
    pd.DataFrame({
        "no_stop":  nav_ns,
        "fixed":    nav_fix,
        "trailing": nav_trail,
        "btc_hodl": btc_nav,
    }).to_csv(OUT/"v4_navs.csv")
    print(f"\nWrote {OUT/'v4_full_usdc.json'}")
    print(f"Wrote {OUT/'v4_navs.csv'}")


if __name__ == "__main__":
    main()
