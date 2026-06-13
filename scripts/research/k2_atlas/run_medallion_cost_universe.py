"""Medallion Lite — COST-ROBUST UNIVERSE SEARCH (pre-registered: protocol Amendment B).

top-100 fails the cost gate (Amendment A) because its alpha is in the illiquid tail. This tests
whether a LIQUID-leaning universe survives realistic tiered costs, trading capacity for robustness.

For each universe: frozen 5-factor strategy, within-universe rank, point-in-time; report OOS Sortino
under S0 flat-30 (reference) and S2 realistic-tiered costs, plus the S2+impact capacity curve.
GC-B: cost-robust if S2 OOS Sortino > 2.0; deployable if soft capacity (S2+impact) >= $5M.

Run: PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_cost_universe.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "scripts" / "research" / "k2_atlas"), str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_medallion_costs as cst  # noqa: E402  (tier_matrix, backtest_costed, oos_sortino, SCENARIOS, AUM_GRID)
import run_medallion_universe as uni  # noqa: E402
from medallion_lite.portfolio import build_factor_portfolio  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "medallion_audit"
FLAGSHIP = {"entry_threshold": 0.65, "trailing_stop_pct": 0.15}
UNIVERSES = [("top_10", "top", 10), ("top_25", "top", 25), ("top_50", "top", 50),
             ("top_100", "top", 100), ("adv>=$50M", "floor", 50e6), ("adv>=$20M", "floor", 20e6)]


def weights_for(factors, rw, regime, elig_h):
    comp = uni.composite_within(factors, elig_h)
    W, _ = build_factor_portfolio(
        comp, rw, regime, entry_threshold=FLAGSHIP["entry_threshold"], exit_score_threshold=0.40,
        max_hold_hours=336, trailing_stop_pct=FLAGSHIP["trailing_stop_pct"], rebalance_every_hours=24,
        max_positions=25, max_weight=0.10)
    return W


def main() -> None:
    print("COST-ROBUST UNIVERSE SEARCH — pre-registered (protocol Amendment B)\n")
    factors, rw, regime, dd, memb = uni.load_universe_panel()
    cols, idx = factors["momentum"].columns, factors["momentum"].index
    adv_daily = dd.pivot_table(index="ts", columns="symbol", values="adv20").reindex(columns=cols)
    adv_h = adv_daily.reindex(pd.DatetimeIndex(idx).normalize())
    adv_h.index = idx

    flat30 = lambda W: cst.tier_matrix(adv_h, cst.SCENARIOS["S0_flat30"])  # noqa: E731
    s2_mat = cst.tier_matrix(adv_h, cst.SCENARIOS["S2_base"])

    rows, manifest_rows = [], {}
    print(f"{'universe':<12}{'#nm':>5}{'S0 flat30':>11}{'S2 tiered':>11}   capacity (S2+impact) by AUM")
    print("-" * 92)
    for label, kind, value in UNIVERSES:
        elig_h = uni.eligibility(dd, memb, kind, value, cols, idx)
        nm = float(elig_h.sum(axis=1).mean())
        W = weights_for(factors, rw, regime, elig_h)
        s0 = cst.oos_sortino(cst.backtest_costed(W, rw, cst.tier_matrix(adv_h, cst.SCENARIOS["S0_flat30"])))
        s2 = cst.oos_sortino(cst.backtest_costed(W, rw, s2_mat))
        cap = {}
        for aum in cst.AUM_GRID:
            cap[f"{int(aum/1e6)}M"] = cst.oos_sortino(
                cst.backtest_costed(W, rw, s2_mat, aum=aum, adv_h=adv_h, spread_mat=s2_mat))
        soft_cap = max([int(k[:-1]) for k, v in cap.items() if v > 2.0], default=0)
        rows.append((label, nm, s0, s2, cap, soft_cap))
        manifest_rows[label] = {"avg_names": round(nm, 1), "s0_flat30": s0, "s2_tiered": s2,
                                "capacity_curve": cap, "soft_capacity_usdM": soft_cap}
        capstr = "  ".join(f"{k}:{v:.2f}" for k, v in cap.items())
        print(f"{label:<12}{nm:>5.0f}{s0:>11.2f}{s2:>11.2f}   {capstr}")

    print("-" * 92)
    robust = [r for r in rows if r[3] > 2.0]
    deployable = [r for r in robust if r[5] >= 5]
    print(f"\nGC-B cost-robust (S2 > 2.0): {[r[0] for r in robust] or 'NONE'}")
    print(f"GC-B deployable (soft capacity >= $5M): {[(r[0], f'${r[5]}M') for r in deployable] or 'NONE'}")
    verdict = ("GRADUATES to a liquid-universe variant" if deployable
               else "DOES NOT graduate — no liquid universe is cost-robust + tradable")
    print(f"VERDICT: {verdict}")

    manifest = {
        "protocol": "docs/research/medallion_validation_protocol.md (Amendment B)",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                     capture_output=True, text=True).stdout.strip(),
        "tiers_usd": cst.TIERS_USD, "s2_bps": cst.SCENARIOS["S2_base"], "impact_c_bps": cst.IMPACT_C,
        "universes": manifest_rows,
        "cost_robust": [r[0] for r in robust], "deployable": [r[0] for r in deployable],
        "verdict": verdict,
        "reproduce": "PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_cost_universe.py",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "medallion_cost_universe.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {OUT_DIR / 'medallion_cost_universe.json'}")


if __name__ == "__main__":
    main()
