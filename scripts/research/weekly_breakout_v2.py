#!/usr/bin/env python
"""Weekly Breakout v2 — combined test of the five v1-suggested enhancements:

  #1 Deterministic dynamic universe: top-N pairs by trailing 90-day median dollar
     volume, refreshed monthly.
  #2 Flip default ATR stop from 3.0x to 2.0x.
  #3 Trailing ATR stop variant (highest close since entry - mult * entry-day ATR).
  #4 Walk-forward optimization on the bluechip universe (730d train / 365d OOS).
  #5 Optimal blend weight with BTC HODL (frontier curve).

The deterministic top-N universe (N=10, monthly refresh) is the recommended
production-grade universe definition for the bluechip strategy.
"""
from __future__ import annotations
import importlib.util
import json
from itertools import product
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

OUT = ROOT / "artifacts/research/weekly_breakout_v2"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ANN = 365.0
INITIAL = 100_000.0
COST_30_PER_SIDE = 30 / 10_000.0
BLUECHIP_10 = ["BTC-USDC","ETH-USDC","SOL-USDC","XRP-USDC","ADA-USDC",
                "DOGE-USDC","AVAX-USDC","LINK-USDC","DOT-USDC","LTC-USDC"]


# ── Test #1: Deterministic dynamic universe ────────────────────────────
def build_dynamic_universe_mask(panels: dict, top_n: int = 10,
                                  refresh: str = "MS") -> pd.DataFrame:
    """Return a (dates x symbols) boolean mask: True if the symbol is in the
    top-N pairs by trailing 90-day median dollar volume, refreshed on each
    `refresh` boundary (default = month start).

    Selection is point-in-time correct: each refresh date uses only data up to
    the day BEFORE that date.
    """
    C, V = panels["C"], panels["V"]
    DV90 = (V * C).rolling(90).median()  # 90-day median dollar volume

    # Refresh dates are the first trading day of each month
    dates = DV90.index
    monthly = pd.DataFrame(index=dates)
    monthly["month"] = dates.to_period("M").to_timestamp()
    first_per_month = monthly.groupby("month").apply(lambda x: x.index[0]).values
    first_per_month = pd.DatetimeIndex(first_per_month)

    mask = pd.DataFrame(False, index=dates, columns=DV90.columns)
    cur_universe: set[str] = set()
    next_refresh_idx = 0
    refresh_dates_set = set(first_per_month)

    for d in dates:
        if d in refresh_dates_set:
            # Use PRIOR day's DV90 (point-in-time)
            idx = dates.get_indexer([d])[0]
            if idx > 0:
                ranking = DV90.iloc[idx - 1].dropna().sort_values(ascending=False)
                cur_universe = set(ranking.head(top_n).index)
            else:
                cur_universe = set()
        if cur_universe:
            mask.loc[d, list(cur_universe)] = True
    return mask


def run_with_universe_mask(panels, mask, *, atr_mult=3.0, trailing=False,
                            cost_per_side=COST_30_PER_SIDE, min_eligible=3,
                            mom_floor=40.0):
    """Run breakout strategy with a custom dynamic-universe mask layered on top
    of the structural eligibility (live + liquid + coverage)."""
    ind = wb.compute_indicators(panels, 1_000_000.0)
    ind["mom_score"] = wb.momentum_score(ind["mom"], ind["eligible_universe"] & mask)
    result = wb.backtest(panels, ind, params=dict(
        cost_per_side=cost_per_side,
        atr_stop_mult=atr_mult, use_atr_stop=True,
        require_breakout_entry=True,
        mom_floor=mom_floor,
        trailing_stop=trailing,
        min_eligible_at_start=min_eligible,
    ), extra_universe_mask=mask)
    return result


def btc_bh(panels, start_date, cost_per_side):
    o = panels["O"]["BTC-USDC"]; c = panels["C"]["BTC-USDC"]
    o, c = o.align(c, join="inner")
    o = o.loc[o.index >= start_date]; c = c.loc[c.index >= start_date]
    r = (c/o - 1).fillna(0.0)
    return 100_000 * (1+r).cumprod() * (1 - cost_per_side)


# ── Test #4: Walk-forward optimization ────────────────────────────────
def walk_forward_optimize(panels, mask=None, train_days=730, test_days=365,
                            param_grid: dict | None = None,
                            cost_per_side=COST_30_PER_SIDE,
                            min_eligible=3) -> dict:
    """Roll forward in test_days steps. At each step, train on the prior
    train_days using each parameter combo; pick the combo with the highest
    in-sample Sharpe; record its OOS returns over the next test_days.

    Returns dict with selections (per window) and stitched OOS NAV.
    """
    if param_grid is None:
        param_grid = dict(
            atr_mult=[2.0, 2.5, 3.0],
            mom_floor=[30.0, 40.0, 50.0],
            trailing=[False, True],
        )

    ind_full = wb.compute_indicators(panels, 1_000_000.0)
    eligible_universe = ind_full["eligible_universe"]
    if mask is not None:
        eligible_universe = eligible_universe & mask.reindex_like(eligible_universe).fillna(False).astype(bool)
    score_full = wb.momentum_score(ind_full["mom"], eligible_universe)

    dates = panels["C"].index
    elig_n = eligible_universe.sum(axis=1)
    n_warm = 95
    first_valid = dates[(elig_n >= min_eligible).values & (dates >= dates[n_warm])][0]

    # Window start: first date where we have train_days+1 of history past first_valid
    start = first_valid + pd.Timedelta(days=train_days)
    selections = []
    oos_nav_pieces = []
    cur_nav = INITIAL

    cur = start
    end = dates[-1]
    grid = list(product(param_grid["atr_mult"], param_grid["mom_floor"], param_grid["trailing"]))

    while cur <= end:
        train_start = cur - pd.Timedelta(days=train_days)
        train_end = cur - pd.Timedelta(days=1)
        test_start = cur
        test_end = min(cur + pd.Timedelta(days=test_days - 1), end)

        # Build a panels-subset for training/testing by slicing dates (but compute
        # on full data — we'll just slice the resulting NAV).
        best_sh = -np.inf
        best_combo = None
        best_train_nav_end = None
        for atr_m, mom_f, trail in grid:
            params = dict(
                cost_per_side=cost_per_side,
                atr_stop_mult=atr_m,
                use_atr_stop=True,
                require_breakout_entry=True,
                mom_floor=mom_f,
                trailing_stop=trail,
                min_eligible_at_start=min_eligible,
            )
            ind = dict(ind_full)
            ind["mom_score"] = score_full
            r = wb.backtest(panels, ind, params=params,
                             extra_universe_mask=(mask if mask is not None else None))
            nav = r["equity"]["nav"]
            tr_nav = nav.loc[(nav.index >= train_start) & (nav.index <= train_end)]
            if len(tr_nav) < 60:
                continue
            ret = tr_nav.pct_change().dropna()
            if ret.std() == 0:
                sh = -np.inf
            else:
                sh = ret.mean() / ret.std() * np.sqrt(ANN)
            if sh > best_sh:
                best_sh = sh
                best_combo = (atr_m, mom_f, trail, r)

        if best_combo is None:
            break
        atr_m, mom_f, trail, r = best_combo
        nav = r["equity"]["nav"]
        oos_nav = nav.loc[(nav.index >= test_start) & (nav.index <= test_end)]
        if len(oos_nav) == 0:
            cur = cur + pd.Timedelta(days=test_days)
            continue
        oos_ret = oos_nav.pct_change().fillna(0.0)
        oos_ret.iloc[0] = 0.0
        # Stitch — start at cur_nav, compound by oos_ret
        nav_piece = cur_nav * (1 + oos_ret).cumprod()
        oos_nav_pieces.append(nav_piece)
        cur_nav = nav_piece.iloc[-1]
        selections.append(dict(window_start=str(test_start.date()),
                                window_end=str(test_end.date()),
                                atr_mult=atr_m, mom_floor=mom_f, trailing=trail,
                                train_sharpe=best_sh,
                                oos_total_ret=float(nav_piece.iloc[-1]/nav_piece.iloc[0]-1)))
        cur = cur + pd.Timedelta(days=test_days)

    if oos_nav_pieces:
        oos_nav = pd.concat(oos_nav_pieces).sort_index()
        oos_nav = oos_nav[~oos_nav.index.duplicated(keep="first")]
    else:
        oos_nav = pd.Series(dtype=float)

    return dict(selections=pd.DataFrame(selections), oos_nav=oos_nav)


# ── Test #5: Optimal blend with BTC HODL ──────────────────────────────
def blend_frontier(strat_nav, btc_nav, weights=None):
    """Compute Sharpe / MaxDD / CAGR / Total return frontier for blends of
    `strat` and `btc` at weights `weights`."""
    if weights is None:
        weights = np.linspace(0, 1, 21)
    # Align
    common = strat_nav.index.intersection(btc_nav.index)
    s = strat_nav.loc[common]; b = btc_nav.loc[common]
    sr = s.pct_change().fillna(0.0)
    br = b.pct_change().fillna(0.0)
    rows = []
    for w in weights:
        port = w * sr + (1 - w) * br
        nav = (1 + port).cumprod() * INITIAL
        ann = port.mean() * ANN
        vol = port.std() * np.sqrt(ANN)
        sh = ann / vol if vol > 0 else 0.0
        dd = (nav / nav.cummax() - 1).min()
        cagr = (nav.iloc[-1]/nav.iloc[0]) ** (ANN/len(nav)) - 1
        rows.append(dict(w_strategy=w, w_btc=1-w, cagr=cagr, sharpe=sh,
                          max_dd=float(dd), total=nav.iloc[-1]/nav.iloc[0]-1,
                          final=nav.iloc[-1]))
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=== Weekly Breakout v2 — 5-enhancement test suite ===\n")

    # Load FULL USDC universe (we need everything for the dynamic-universe rule
    # to be able to pick whatever pairs are top-N at each refresh)
    syms_all, bars_all = wb.load_universe()
    panels_all = wb.assemble_panels(syms_all, bars_all)
    print(f"Loaded {len(panels_all['C'].columns)} pairs from lake\n")

    # ── Test #1: Deterministic dynamic universe — top-10 by 90d $-vol, monthly ──
    print("─" * 70)
    print("[Test #1] Deterministic dynamic universe — top-10 by 90d $-vol, monthly refresh")
    print("─" * 70)
    mask10 = build_dynamic_universe_mask(panels_all, top_n=10, refresh="MS")
    n_members = mask10.sum(axis=1)
    print(f"  Universe size over time: median={int(n_members.median())}  min={n_members.min()}  max={n_members.max()}")
    n_unique_ever = (mask10.any(axis=0)).sum()
    print(f"  Unique pairs ever in the dynamic universe: {n_unique_ever}")
    # Show how the membership evolves
    snapshots = ["2018-01-01","2019-01-01","2020-01-01","2021-01-01","2022-01-01","2023-01-01","2024-01-01","2025-01-01","2026-01-01"]
    for d in snapshots:
        ts = pd.Timestamp(d)
        if ts in mask10.index:
            row = mask10.loc[ts]
            members = sorted(row[row].index.tolist())
            print(f"    {d}: {members}")
    print()

    result_dyn_3atr = run_with_universe_mask(panels_all, mask10, atr_mult=3.0, trailing=False)
    nav_dyn_3 = result_dyn_3atr["equity"]["nav"]
    m_dyn_3 = wb.metrics_from_nav(nav_dyn_3, "Dynamic Top-10 @ 3.0 ATR fixed")
    print(f"  Dynamic Top-10  / 3.0x ATR / fixed:    CAGR={m_dyn_3['cagr']*100:+6.2f}%  Sh={m_dyn_3['sharpe']:+5.2f}  DD={m_dyn_3['max_dd']*100:+6.1f}%  Tot={m_dyn_3['total']*100:+8.1f}%  start={nav_dyn_3.index[0].date()}")

    # ── Test #2: ATR stop 2.0x on deterministic universe ──
    print("\n" + "─" * 70)
    print("[Test #2] ATR stop = 2.0x (proposed default) on dynamic universe")
    print("─" * 70)
    result_dyn_2atr = run_with_universe_mask(panels_all, mask10, atr_mult=2.0, trailing=False)
    nav_dyn_2 = result_dyn_2atr["equity"]["nav"]
    m_dyn_2 = wb.metrics_from_nav(nav_dyn_2, "Dynamic Top-10 @ 2.0 ATR fixed")
    print(f"  Dynamic Top-10  / 2.0x ATR / fixed:    CAGR={m_dyn_2['cagr']*100:+6.2f}%  Sh={m_dyn_2['sharpe']:+5.2f}  DD={m_dyn_2['max_dd']*100:+6.1f}%  Tot={m_dyn_2['total']*100:+8.1f}%")

    # Also full ATR sweep on dynamic universe
    atr_sweep = []
    for atr_m in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        r = run_with_universe_mask(panels_all, mask10, atr_mult=atr_m, trailing=False)
        nav = r["equity"]["nav"]
        m = wb.metrics_from_nav(nav, f"ATR={atr_m}")
        atr_sweep.append(dict(atr_mult=atr_m, **m))
        print(f"    atr={atr_m}:  CAGR={m['cagr']*100:+6.2f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+8.1f}%")

    # ── Test #3: Trailing ATR stop on deterministic universe ──
    print("\n" + "─" * 70)
    print("[Test #3] Trailing ATR stop on dynamic universe")
    print("─" * 70)
    trail_sweep = []
    for atr_m in [2.0, 2.5, 3.0, 3.5, 4.0]:
        r = run_with_universe_mask(panels_all, mask10, atr_mult=atr_m, trailing=True)
        nav = r["equity"]["nav"]
        m = wb.metrics_from_nav(nav, f"Trail ATR={atr_m}")
        trail_sweep.append(dict(atr_mult=atr_m, **m))
        print(f"    trail atr={atr_m}: CAGR={m['cagr']*100:+6.2f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+8.1f}%")

    # Also test trailing on hand-picked bluechip-10 for completeness
    print("\n  [On hand-picked Bluechip-10, for reference]")
    syms_bc, bars_bc = wb.load_universe(restrict_to=BLUECHIP_10)
    panels_bc = wb.assemble_panels(syms_bc, bars_bc)
    ind_bc = wb.compute_indicators(panels_bc, 1_000_000.0)
    ind_bc["mom_score"] = wb.momentum_score(ind_bc["mom"], ind_bc["eligible_universe"])
    for atr_m, trail in [(3.0, False), (2.0, False), (3.0, True), (2.0, True)]:
        r = wb.backtest(panels_bc, ind_bc, params=dict(
            cost_per_side=COST_30_PER_SIDE, atr_stop_mult=atr_m, use_atr_stop=True,
            require_breakout_entry=True, trailing_stop=trail, min_eligible_at_start=3))
        nav = r["equity"]["nav"]
        m = wb.metrics_from_nav(nav, "")
        flag = "trail" if trail else "fixed"
        print(f"    BC-10 / {atr_m}x ATR / {flag}: CAGR={m['cagr']*100:+6.2f}%  Sh={m['sharpe']:+5.2f}  DD={m['max_dd']*100:+6.1f}%  Tot={m['total']*100:+8.1f}%")

    # ── Test #4: Walk-forward optimization on deterministic universe ──
    print("\n" + "─" * 70)
    print("[Test #4] Walk-forward optimization on dynamic universe (730d train / 365d OOS)")
    print("─" * 70)
    wfo = walk_forward_optimize(panels_all, mask=mask10,
                                  train_days=730, test_days=365,
                                  param_grid=dict(
                                      atr_mult=[2.0, 2.5, 3.0],
                                      mom_floor=[30.0, 40.0, 50.0],
                                      trailing=[False, True],
                                  ))
    sel = wfo["selections"]
    sel.to_csv(OUT/"wfo_selections.csv", index=False)
    oos_nav = wfo["oos_nav"]
    if len(oos_nav) > 1:
        m_wfo = wb.metrics_from_nav(oos_nav, "WFO OOS")
        print(f"\n  WFO OOS stitched: CAGR={m_wfo['cagr']*100:+6.2f}%  Sh={m_wfo['sharpe']:+5.2f}  DD={m_wfo['max_dd']*100:+6.1f}%  Tot={m_wfo['total']*100:+8.1f}%")
        print(f"  Number of windows: {len(sel)}")
        print(f"  Window selections:")
        for _, r in sel.iterrows():
            print(f"    {r['window_start']} → {r['window_end']}: atr={r['atr_mult']}, mom_floor={r['mom_floor']}, trail={r['trailing']}, train_sh={r['train_sharpe']:.2f}, oos_ret={r['oos_total_ret']*100:+.1f}%")
    oos_nav.to_csv(OUT/"wfo_oos_nav.csv")

    # ── Test #5: Optimal blend with BTC HODL ──
    print("\n" + "─" * 70)
    print("[Test #5] Optimal blend weight with BTC HODL (deterministic universe strategy)")
    print("─" * 70)
    btc_nav = btc_bh(panels_all, nav_dyn_3.index[0], COST_30_PER_SIDE)
    frontier = blend_frontier(nav_dyn_3, btc_nav)
    frontier.to_csv(OUT/"blend_frontier.csv", index=False)
    print(f"\n  Weight Strategy | Weight BTC | Sharpe | CAGR    | MaxDD   | Total")
    print(f"  ──────────────  | ────────── | ────── | ─────── | ─────── | ──────")
    for _, r in frontier.iloc[::2].iterrows():
        print(f"   {r['w_strategy']:>6.2f}        |  {r['w_btc']:>6.2f}    |  {r['sharpe']:+5.2f} | {r['cagr']*100:+6.2f}% | {r['max_dd']*100:+6.1f}% | {r['total']*100:+7.1f}%")
    best_sh = frontier.loc[frontier["sharpe"].idxmax()]
    print(f"\n  ★ Max Sharpe blend: {best_sh['w_strategy']*100:.0f}% strategy + {best_sh['w_btc']*100:.0f}% BTC HODL "
          f"→ Sh={best_sh['sharpe']:.2f}  DD={best_sh['max_dd']*100:.0f}%  CAGR={best_sh['cagr']*100:.1f}%")
    # Best Sharpe at MaxDD ≤ 50% constraint
    constr = frontier[frontier["max_dd"] >= -0.50]
    if len(constr) > 0:
        best_constr = constr.loc[constr["sharpe"].idxmax()]
        print(f"  ★ Max Sharpe @ DD ≤ 50%: {best_constr['w_strategy']*100:.0f}% strategy + {best_constr['w_btc']*100:.0f}% BTC HODL "
              f"→ Sh={best_constr['sharpe']:.2f}  DD={best_constr['max_dd']*100:.0f}%  CAGR={best_constr['cagr']*100:.1f}%")

    # ── Save artifacts ───────────────────────────────────────────────────
    nav_dyn_3.to_csv(OUT/"primary_dynamic_top10_3atr_fixed.csv")
    nav_dyn_2.to_csv(OUT/"alt_dynamic_top10_2atr_fixed.csv")
    pd.DataFrame(atr_sweep).to_csv(OUT/"atr_sweep_dynamic.csv", index=False)
    pd.DataFrame(trail_sweep).to_csv(OUT/"trail_sweep_dynamic.csv", index=False)

    # ── Figures ─────────────────────────────────────────────────────────
    # Fig 1: Dynamic vs hand-picked universe equity curves
    syms_bc, bars_bc = wb.load_universe(restrict_to=BLUECHIP_10)
    panels_bc = wb.assemble_panels(syms_bc, bars_bc)
    ind_bc = wb.compute_indicators(panels_bc, 1_000_000.0)
    ind_bc["mom_score"] = wb.momentum_score(ind_bc["mom"], ind_bc["eligible_universe"])
    r_bc = wb.backtest(panels_bc, ind_bc, params=dict(
        cost_per_side=COST_30_PER_SIDE, atr_stop_mult=3.0, use_atr_stop=True,
        require_breakout_entry=True, min_eligible_at_start=3))
    nav_bc = r_bc["equity"]["nav"]
    m_bc = wb.metrics_from_nav(nav_bc, "Hand-picked BC-10")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios":[3,1]})
    ax = axes[0]
    ax.plot(nav_dyn_3.index, nav_dyn_3/1e3, lw=2.0, color="#1f77b4",
             label=f"Dynamic Top-10  Sh={m_dyn_3['sharpe']:.2f} DD={m_dyn_3['max_dd']:.0%} CAGR={m_dyn_3['cagr']*100:.0f}%")
    ax.plot(nav_bc.index, nav_bc/1e3, lw=1.5, color="#1f77b4", ls=":",
             label=f"Hand-picked BC-10  Sh={m_bc['sharpe']:.2f} DD={m_bc['max_dd']:.0%} CAGR={m_bc['cagr']*100:.0f}%")
    ax.plot(btc_nav.index, btc_nav/1e3, lw=1.6, color="#d62728", ls="--",
             label=f"BTC-USDC HODL  Sh={wb.metrics_from_nav(btc_nav,'')['sharpe']:.2f}")
    ax.set_yscale("log"); ax.set_ylabel("NAV ($k, log)")
    ax.set_title("Test #1: Dynamic Top-10 (monthly refresh) vs hand-picked Bluechip-10")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    ax = axes[1]
    dd_dyn = (nav_dyn_3/nav_dyn_3.cummax()-1)*100
    dd_bc  = (nav_bc/nav_bc.cummax()-1)*100
    dd_btc = (btc_nav/btc_nav.cummax()-1)*100
    ax.plot(dd_dyn.index, dd_dyn, color="#1f77b4", lw=1.0, label="Dynamic Top-10")
    ax.plot(dd_bc.index, dd_bc, color="#1f77b4", lw=1.0, ls=":", label="Hand-picked BC-10")
    ax.plot(dd_btc.index, dd_btc, color="#d62728", lw=1.0, ls="--", label="BTC HODL")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("DD (%)"); ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG/"01_dynamic_vs_handpicked.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Fig 2: ATR sensitivity (fixed vs trailing) on dynamic universe
    fig, ax = plt.subplots(figsize=(13, 6))
    df_a = pd.DataFrame(atr_sweep)
    df_t = pd.DataFrame(trail_sweep)
    ax.plot(df_a["atr_mult"], df_a["sharpe"], marker="o", lw=2, color="#1f77b4", label="Fixed stop")
    ax.plot(df_t["atr_mult"], df_t["sharpe"], marker="s", lw=2, color="#ff7f0e", label="Trailing stop")
    ax.set_xlabel("ATR multiplier"); ax.set_ylabel("Sharpe (Dynamic Top-10 universe)")
    ax.set_title("Test #2 + #3: ATR stop sensitivity — fixed vs trailing")
    ax.grid(True, alpha=0.3); ax.legend()
    ax2 = ax.twinx()
    ax2.plot(df_a["atr_mult"], df_a["max_dd"]*100, marker="o", lw=1, color="#1f77b4", ls=":", alpha=0.6, label="Fixed DD")
    ax2.plot(df_t["atr_mult"], df_t["max_dd"]*100, marker="s", lw=1, color="#ff7f0e", ls=":", alpha=0.6, label="Trail DD")
    ax2.set_ylabel("MaxDD (%)")
    ax2.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG/"02_atr_sensitivity_fixed_vs_trailing.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Fig 3: WFO equity vs in-sample baseline
    if len(oos_nav) > 1:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(oos_nav.index, oos_nav/1e3, lw=2, color="#9467bd", label=f"WFO stitched OOS (Sh={m_wfo['sharpe']:.2f}, CAGR={m_wfo['cagr']*100:.0f}%, DD={m_wfo['max_dd']*100:.0f}%)")
        nav_dyn_oos = nav_dyn_3.loc[nav_dyn_3.index >= oos_nav.index[0]]
        nav_dyn_oos = nav_dyn_oos / nav_dyn_oos.iloc[0] * oos_nav.iloc[0]
        ax.plot(nav_dyn_oos.index, nav_dyn_oos/1e3, lw=1.4, color="#1f77b4", ls=":",
                 label=f"Fixed-params (3 ATR, mom_floor=40, fixed) over same window")
        btc_oos = btc_nav.loc[btc_nav.index >= oos_nav.index[0]]
        btc_oos = btc_oos / btc_oos.iloc[0] * oos_nav.iloc[0]
        ax.plot(btc_oos.index, btc_oos/1e3, lw=1.4, color="#d62728", ls="--", label="BTC HODL")
        ax.set_yscale("log"); ax.set_ylabel("NAV ($k, log)")
        ax.set_title("Test #4: Walk-forward OOS vs fixed-param baseline")
        ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIG/"03_wfo_oos.png", dpi=120, bbox_inches="tight")
        plt.close()

    # Fig 4: Blend frontier
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(frontier["w_strategy"]*100, frontier["sharpe"], marker="o", lw=2, color="#1f77b4")
    ax.axvline(best_sh["w_strategy"]*100, color="#d62728", ls="--", alpha=0.7,
                label=f"Max Sh @ {best_sh['w_strategy']*100:.0f}% strat")
    ax.set_xlabel("% in Strategy (rest = BTC HODL)")
    ax.set_ylabel("Portfolio Sharpe")
    ax.set_title("Sharpe vs blend weight")
    ax.grid(True, alpha=0.3); ax.legend()
    ax = axes[1]
    ax.scatter(frontier["max_dd"]*100, frontier["cagr"]*100, c=frontier["w_strategy"], cmap="viridis", s=60)
    for _, r in frontier.iloc[::5].iterrows():
        ax.annotate(f"{r['w_strategy']*100:.0f}%", (r['max_dd']*100, r['cagr']*100), fontsize=7)
    ax.set_xlabel("Max drawdown (%)"); ax.set_ylabel("CAGR (%)")
    ax.set_title("Frontier: CAGR vs MaxDD across blend weights")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG/"04_blend_frontier.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Summary
    with open(OUT/"v2_summary.json", "w") as f:
        json.dump({
            "test_1_dynamic_top10_3atr": m_dyn_3,
            "test_2_dynamic_top10_2atr": m_dyn_2,
            "test_2_atr_sweep": [dict(atr_mult=r["atr_mult"], sharpe=r["sharpe"], cagr=r["cagr"], max_dd=r["max_dd"], total=r["total"]) for r in atr_sweep],
            "test_3_trailing_sweep": [dict(atr_mult=r["atr_mult"], sharpe=r["sharpe"], cagr=r["cagr"], max_dd=r["max_dd"], total=r["total"]) for r in trail_sweep],
            "test_4_wfo": (m_wfo if len(oos_nav) > 1 else None),
            "test_5_best_sharpe_blend": dict(best_sh),
            "test_5_best_constrained_blend": (dict(best_constr) if len(constr) > 0 else None),
        }, f, indent=2, default=str)
    print(f"\n[done] Wrote {OUT}\n")


if __name__ == "__main__":
    main()
