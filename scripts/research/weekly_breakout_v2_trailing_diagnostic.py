#!/usr/bin/env python
"""Diagnostic: WHY does the trailing ATR stop underperform the fixed stop?

Compares fixed vs trailing stop variants on the same Bluechip-10 universe:
  - Stop-reason distribution
  - Holding-period distribution
  - P&L distribution by exit reason
  - For the same winning trades under FIXED, traces what TRAILING would have done
  - Visualizes one specific winning trade with both stop trajectories

Outputs to artifacts/research/weekly_breakout_v2/figures/05_*
and saves a CSV with the round-trip comparison.
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

OUT = ROOT / "artifacts/research/weekly_breakout_v2"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BLUECHIP_10 = ["BTC-USDC","ETH-USDC","SOL-USDC","XRP-USDC","ADA-USDC",
                "DOGE-USDC","AVAX-USDC","LINK-USDC","DOT-USDC","LTC-USDC"]


def round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct closed round-trips from the trade ledger.
    Round-trip = first BUY → final SELL when position fully closes.
    Uses average entry price weighted by share notional."""
    rts = []
    for sym, g in trades.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        pos_sh = 0.0
        cost_basis_notional = 0.0  # sum of (shares * price) on buys
        entry_date = None
        for _, r in g.iterrows():
            if r["side"] == "BUY":
                if pos_sh < 1e-9:
                    entry_date = r["date"]
                    cost_basis_notional = 0.0
                cost_basis_notional += r["shares"] * r["price"]
                pos_sh += r["shares"]
            else:
                # SELL: close (or trim) the position
                exit_sh = r["shares"]
                exit_px = r["price"]
                # Average entry price (weighted by remaining basis)
                if pos_sh > 1e-9:
                    avg_entry = cost_basis_notional / pos_sh
                else:
                    avg_entry = exit_px
                pnl_pct = (exit_px - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                # Reduce basis proportionally
                cost_basis_notional *= max(0.0, 1.0 - exit_sh / max(pos_sh, 1e-9))
                pos_sh = max(0.0, pos_sh - exit_sh)
                if pos_sh < 1e-9:
                    held = (r["date"] - entry_date).days if entry_date else None
                    rts.append(dict(
                        symbol=sym,
                        entry_date=entry_date, exit_date=r["date"],
                        held_days=held,
                        avg_entry_px=avg_entry, exit_px=exit_px,
                        pnl_pct=pnl_pct, exit_reason=r["reason"],
                    ))
                    entry_date = None
    return pd.DataFrame(rts)


def main():
    print("=== Trailing-stop diagnostic ===\n")
    syms, bars = wb.load_universe(restrict_to=BLUECHIP_10)
    panels = wb.assemble_panels(syms, bars)
    ind = wb.compute_indicators(panels, 1_000_000.0)
    ind["mom_score"] = wb.momentum_score(ind["mom"], ind["eligible_universe"])

    print("Running FIXED 3.0x ATR stop variant...")
    r_fix = wb.backtest(panels, ind, params=dict(
        cost_per_side=30/10000.0, atr_stop_mult=3.0, use_atr_stop=True,
        require_breakout_entry=True, trailing_stop=False, min_eligible_at_start=3))
    print("Running TRAILING 3.0x ATR stop variant...")
    r_trail = wb.backtest(panels, ind, params=dict(
        cost_per_side=30/10000.0, atr_stop_mult=3.0, use_atr_stop=True,
        require_breakout_entry=True, trailing_stop=True, min_eligible_at_start=3))

    rts_fix   = round_trips(r_fix["trades"])
    rts_trail = round_trips(r_trail["trades"])

    print(f"\nRound-trip counts: fixed={len(rts_fix)}, trailing={len(rts_trail)}")

    print(f"\n── Exit-reason distribution ──")
    print("                fixed         trailing")
    fix_counts   = rts_fix["exit_reason"].value_counts(normalize=True) * 100
    trail_counts = rts_trail["exit_reason"].value_counts(normalize=True) * 100
    all_reasons = sorted(set(fix_counts.index) | set(trail_counts.index))
    for reason in all_reasons:
        print(f"  {reason:<14} {fix_counts.get(reason, 0):>6.1f}%        {trail_counts.get(reason, 0):>6.1f}%")
    print(f"  total count  {len(rts_fix):>6d}        {len(rts_trail):>6d}")

    print(f"\n── Mean return by exit reason ──")
    print("                fixed         trailing")
    for reason in all_reasons:
        f = rts_fix[rts_fix["exit_reason"] == reason]["pnl_pct"].mean() * 100 if not rts_fix.empty else 0
        t = rts_trail[rts_trail["exit_reason"] == reason]["pnl_pct"].mean() * 100 if not rts_trail.empty else 0
        nf = (rts_fix["exit_reason"] == reason).sum()
        nt = (rts_trail["exit_reason"] == reason).sum()
        print(f"  {reason:<14} {f:>+7.1f}% (n={nf:>3d})  {t:>+7.1f}% (n={nt:>3d})")

    print(f"\n── Holding period (days) ──")
    print(f"  fixed:    median={rts_fix['held_days'].median():.0f}  mean={rts_fix['held_days'].mean():.0f}  90th pct={rts_fix['held_days'].quantile(0.9):.0f}")
    print(f"  trailing: median={rts_trail['held_days'].median():.0f}  mean={rts_trail['held_days'].mean():.0f}  90th pct={rts_trail['held_days'].quantile(0.9):.0f}")

    print(f"\n── Per-trade return distribution ──")
    print(f"  fixed:    median={rts_fix['pnl_pct'].median()*100:+.1f}%  mean={rts_fix['pnl_pct'].mean()*100:+.1f}%  std={rts_fix['pnl_pct'].std()*100:.1f}%")
    print(f"           best={rts_fix['pnl_pct'].max()*100:+.1f}%  worst={rts_fix['pnl_pct'].min()*100:+.1f}%  win rate={(rts_fix['pnl_pct']>0).mean()*100:.0f}%")
    print(f"  trailing: median={rts_trail['pnl_pct'].median()*100:+.1f}%  mean={rts_trail['pnl_pct'].mean()*100:+.1f}%  std={rts_trail['pnl_pct'].std()*100:.1f}%")
    print(f"           best={rts_trail['pnl_pct'].max()*100:+.1f}%  worst={rts_trail['pnl_pct'].min()*100:+.1f}%  win rate={(rts_trail['pnl_pct']>0).mean()*100:.0f}%")

    print(f"\n── Top 5 fixed-variant winners — what happened in trailing? ──")
    top5 = rts_fix.nlargest(5, "pnl_pct")
    for _, w in top5.iterrows():
        print(f"\n  Fixed:    {w['symbol']:<10} {w['entry_date'].date()} → {w['exit_date'].date()} ({w['held_days']:>3d}d)  "
              f"px {w['avg_entry_px']:>8.2f} → {w['exit_px']:>8.2f}  ret {w['pnl_pct']*100:+.0f}%  ({w['exit_reason']})")
        # Find the trailing-variant trades that include this entry
        match = rts_trail[(rts_trail["symbol"] == w["symbol"]) &
                          (rts_trail["entry_date"] >= w["entry_date"] - pd.Timedelta(days=7)) &
                          (rts_trail["entry_date"] <= w["exit_date"])]
        if not match.empty:
            for _, t in match.iterrows():
                print(f"  Trailing: {t['symbol']:<10} {t['entry_date'].date()} → {t['exit_date'].date()} ({t['held_days']:>3d}d)  "
                      f"px {t['avg_entry_px']:>8.2f} → {t['exit_px']:>8.2f}  ret {t['pnl_pct']*100:+.0f}%  ({t['exit_reason']})")
        else:
            print(f"  Trailing: (no trade entered in this window)")

    # ── Figure: visualize a specific trade with both stop trajectories ──
    print(f"\n── Generating illustrative trade chart ──")
    # Take the biggest winner under fixed and chart it
    biggest = rts_fix.nlargest(1, "pnl_pct").iloc[0]
    sym = biggest["symbol"]
    entry = biggest["entry_date"]
    exit_fix = biggest["exit_date"]
    print(f"  Biggest fixed winner: {sym} {entry.date()} → {exit_fix.date()}  ({biggest['pnl_pct']*100:+.0f}%)")

    # Pull the OHLC + ATR for this trade
    C = panels["C"][sym]
    H = panels["H"][sym]
    L = panels["L"][sym]
    O = panels["O"][sym]
    atr = ind["atr"][sym]

    # Window: entry - 5 days through max(exit_fix, exit_trail + 30d)
    # Find matching trailing exit if any
    trail_match = rts_trail[(rts_trail["symbol"] == sym) &
                             (rts_trail["entry_date"] >= entry - pd.Timedelta(days=7)) &
                             (rts_trail["entry_date"] <= exit_fix)]
    if not trail_match.empty:
        first_trail_exit = trail_match["exit_date"].min()
    else:
        first_trail_exit = entry + pd.Timedelta(days=30)

    win_start = entry - pd.Timedelta(days=20)
    win_end = max(exit_fix, first_trail_exit) + pd.Timedelta(days=30)
    sl = (C.index >= win_start) & (C.index <= win_end)
    px_close = C[sl]
    px_high  = H[sl]
    px_low   = L[sl]
    px_open  = O[sl]

    # Entry-day ATR and entry price (use open of entry day per engine)
    entry_atr = atr.loc[entry]
    entry_px = O.loc[entry]
    if np.isnan(entry_px):
        entry_px = C.loc[entry]
    fixed_stop = entry_px - 3.0 * entry_atr
    # Trailing stop trajectory: ratchets up with highest close since entry
    running_high = pd.Series(index=px_close.index, dtype=float)
    rh = entry_px
    for d in running_high.index:
        if d < entry:
            running_high.loc[d] = np.nan
            continue
        c = C.loc[d]
        if not np.isnan(c) and c > rh:
            rh = c
        running_high.loc[d] = rh
    trail_stop = running_high - 3.0 * entry_atr

    fig, ax = plt.subplots(figsize=(13, 7))
    # Candle-like representation using close
    ax.plot(px_close.index, px_close, color="#1f77b4", lw=1.5, label=f"{sym} close")
    ax.fill_between(px_close.index, px_low, px_high, color="#1f77b4", alpha=0.12, label="daily H-L range")

    # Entry marker
    ax.axvline(entry, color="#2ca02c", lw=1.0, ls="--", alpha=0.7)
    ax.scatter([entry], [entry_px], marker="^", color="#2ca02c", s=180, zorder=5, label=f"ENTRY @ {entry_px:.2f}")

    # Fixed stop line (constant)
    ax.axhline(fixed_stop, color="#9467bd", lw=2, ls="-",
                label=f"FIXED stop = entry − 3·ATR = {fixed_stop:.2f}  ({(fixed_stop/entry_px-1)*100:+.0f}% from entry)")

    # Trailing stop trajectory
    ax.plot(trail_stop.index, trail_stop, color="#ff7f0e", lw=2, ls="-",
             label="TRAILING stop = highest close since entry − 3·ATR")

    # Mark fixed exit
    fix_exit_px = biggest["exit_px"]
    ax.scatter([exit_fix], [fix_exit_px], marker="v", color="#d62728", s=180, zorder=5,
                label=f"FIXED exit @ {fix_exit_px:.2f}  ({biggest['pnl_pct']*100:+.0f}%)  reason: {biggest['exit_reason']}")

    # Mark trailing exit if any
    if not trail_match.empty:
        first = trail_match.sort_values("exit_date").iloc[0]
        ax.scatter([first["exit_date"]], [first["exit_px"]], marker="v", color="#ff7f0e", s=180, zorder=5,
                    label=f"TRAILING exit @ {first['exit_px']:.2f}  ({first['pnl_pct']*100:+.0f}%)  reason: {first['exit_reason']}")

    ax.set_title(f"Fixed vs trailing ATR stop — {sym} winning trade  "
                 f"(entry {entry.date()}, fixed exit {exit_fix.date()})")
    ax.set_ylabel("Price (USDC)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG/"05_trailing_diagnostic_winning_trade.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {FIG/'05_trailing_diagnostic_winning_trade.png'}")

    # ── Figure 2: PnL distribution + win-rate by reason ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    bins = np.linspace(-0.6, 2.0, 40)
    ax.hist(rts_fix["pnl_pct"], bins=bins, alpha=0.6, color="#1f77b4", label=f"Fixed (n={len(rts_fix)})")
    ax.hist(rts_trail["pnl_pct"], bins=bins, alpha=0.6, color="#ff7f0e", label=f"Trailing (n={len(rts_trail)})")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Round-trip return"); ax.set_ylabel("Count")
    ax.set_title("Per-trade return distribution")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    # Holding period distribution
    bins2 = np.linspace(0, 200, 30)
    ax.hist(rts_fix["held_days"], bins=bins2, alpha=0.6, color="#1f77b4", label="Fixed")
    ax.hist(rts_trail["held_days"], bins=bins2, alpha=0.6, color="#ff7f0e", label="Trailing")
    ax.set_xlabel("Holding period (days)"); ax.set_ylabel("Count")
    ax.set_title("Holding-period distribution")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(FIG/"06_trailing_diagnostic_distributions.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {FIG/'06_trailing_diagnostic_distributions.png'}")

    # Save round-trip CSVs
    rts_fix.to_csv(OUT/"roundtrips_fixed.csv", index=False)
    rts_trail.to_csv(OUT/"roundtrips_trailing.csv", index=False)
    print(f"\n[done] Saved round-trip CSVs to {OUT}/roundtrips_*.csv")


if __name__ == "__main__":
    main()
