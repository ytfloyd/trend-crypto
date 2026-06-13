"""Medallion Lite — TIERED-COST sensitivity (pre-registered: protocol Amendment A).

The headline uses a flat 30 bps. This tests whether the edge survives liquidity-dependent costs
(smaller names slip more) and quantifies capacity via a square-root market-impact model.

Costs are applied per-name, per-bar as one-way turnover x the name's cost, replicating
backtest_portfolio's accounting exactly. GC0: the flat-30 control must reproduce the 2.84 headline.

Run: PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_costs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "scripts" / "research" / "k2_atlas"), str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_medallion_universe as uni  # noqa: E402
import run_medallion_walkforward as wf  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402
from medallion_lite.portfolio import ANN_FACTOR, backtest_portfolio, build_factor_portfolio  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "medallion_audit"
FLAGSHIP = {"entry_threshold": 0.65, "trailing_stop_pct": 0.15}
OOS_START = "2023-01-01"
TIERS_USD = (50e6, 20e6, 5e6)  # T1>=50M, T2>=20M, T3>=5M, else T4
SCENARIOS = {  # one-way bps per tier (T1,T2,T3,T4)
    "S0_flat30": (30, 30, 30, 30), "S1_benign": (10, 20, 40, 70),
    "S2_base": (20, 40, 70, 120), "S3_punitive": (35, 70, 130, 220),
}
AUM_GRID = (5e6, 25e6, 50e6, 100e6, 250e6)
IMPACT_C = 100.0  # bps at participation = 1 (100% of ADV), square-root law
CASH_RATE = 0.04


def oos_sortino(daily: pd.Series) -> float:
    d = daily[daily.index >= OOS_START].dropna()
    return round(float(compute_metrics((1.0 + d).cumprod())["sortino"]), 4)


def _daily(net_hourly: pd.Series) -> pd.Series:
    return wf._daily(net_hourly)


def backtest_costed(W, R, cost_bps_mat, *, aum=None, adv_h=None, spread_mat=None) -> pd.Series:
    """Mirror backtest_portfolio exactly, but cost is per-name: sum_i tau_i * c_i,t.
    tau_i = |w_i - prev_w_i| / 2 (engine's one-way convention). If aum is given, add
    square-root impact: c_i = spread_i + IMPACT_C*sqrt(aum*tau_i/ADV_i)."""
    hourly_cash = (1 + CASH_RATE) ** (1 / ANN_FACTOR) - 1
    aligned = W.index.intersection(R.index)
    W, R = W.loc[aligned], R.loc[aligned]
    prev_w = pd.Series(0.0, index=W.columns)
    out = {}
    for dt in aligned:
        w = W.loc[dt]
        r = R.loc[dt].fillna(0.0)
        tau = (w - prev_w).abs() / 2.0
        if aum is None:
            cost = float((tau * (cost_bps_mat.loc[dt] / 1e4)).sum())
        else:
            adv = adv_h.loc[dt].clip(lower=1e3)
            part = (aum * tau / adv).clip(upper=1.0)        # cap participation at 100% of ADV
            imp = np.minimum(IMPACT_C * np.sqrt(part), 1000.0)  # cap impact at 1000 bps
            cost = float((tau * (spread_mat.loc[dt] + imp) / 1e4).sum())
        gross = float((w * r).sum())
        cash_frac = max(1.0 - float(w.abs().sum()), 0.0)
        out[dt] = gross - cost + cash_frac * hourly_cash
        drifted = w * (1 + r)
        tot = drifted.sum()
        prev_w = drifted / tot if abs(tot) > 1e-10 else drifted
    return _daily(pd.Series(out))


def tier_matrix(adv_h, bps):
    a = adv_h.to_numpy()
    c = np.select([a >= TIERS_USD[0], a >= TIERS_USD[1], a >= TIERS_USD[2]],
                  [bps[0], bps[1], bps[2]], default=bps[3]).astype("float64")
    return pd.DataFrame(c, index=adv_h.index, columns=adv_h.columns)


def main() -> None:
    print("TIERED-COST SENSITIVITY — pre-registered (protocol Amendment A)\n")
    factors, rw, regime, dd, memb = uni.load_universe_panel()
    cols, idx = factors["momentum"].columns, factors["momentum"].index
    elig_h = uni.eligibility(dd, memb, "top", 100, cols, idx)
    comp5 = uni.composite_within(factors, elig_h)
    W, _ = build_factor_portfolio(
        comp5, rw, regime, entry_threshold=FLAGSHIP["entry_threshold"], exit_score_threshold=0.40,
        max_hold_hours=336, trailing_stop_pct=FLAGSHIP["trailing_stop_pct"], rebalance_every_hours=24,
        max_positions=25, max_weight=0.10)

    # point-in-time ADV ($) mapped to the hourly grid
    adv_daily = dd.pivot_table(index="ts", columns="symbol", values="adv20").reindex(columns=cols)
    adv_h = adv_daily.reindex(pd.DatetimeIndex(idx).normalize())
    adv_h.index = idx

    # GC0 reconciliation: flat-30 via engine vs via our per-name backtester
    eng = backtest_portfolio(W, rw, tc_bps=30.0)
    eng_daily = _daily(pd.Series(eng["net_ret"].values, index=pd.to_datetime(eng["ts"])))
    eng_sortino = oos_sortino(eng_daily)
    ours_s0 = oos_sortino(backtest_costed(W, rw, tier_matrix(adv_h, SCENARIOS["S0_flat30"])))
    print(f"GC0 reconciliation: engine flat-30 OOS Sortino {eng_sortino:.2f}  | "
          f"per-name flat-30 {ours_s0:.2f}  | match={abs(eng_sortino-ours_s0)<0.05}")

    # Part A — tiered flat scenarios
    print("\n=== PART A: liquidity-tiered flat costs (OOS Sortino) ===")
    partA = {}
    for name, bps in SCENARIOS.items():
        s = oos_sortino(backtest_costed(W, rw, tier_matrix(adv_h, bps)))
        partA[name] = s
        print(f"  {name:<14} bps(T1/T2/T3/T4)={bps}  OOS Sortino {s:.2f}")

    # Part B — participation/impact capacity curve (spread = S2 base tiers)
    print("\n=== PART B: capacity curve (S2 spreads + sqrt-impact, c=100bps) ===")
    spread_mat = tier_matrix(adv_h, SCENARIOS["S2_base"])
    partB = {}
    for aum in AUM_GRID:
        s = oos_sortino(backtest_costed(W, rw, spread_mat, aum=aum, adv_h=adv_h, spread_mat=spread_mat))
        partB[f"{int(aum/1e6)}M"] = s
        print(f"  AUM ${int(aum/1e6):>4}M   OOS Sortino {s:.2f}")
    soft_cap = max([int(k[:-1]) for k, v in partB.items() if v > 2.0], default=0)

    gates = {
        "GC0_reconciles": bool(abs(eng_sortino - ours_s0) < 0.05),
        "GC1_S2_gt_2.0": bool(partA["S2_base"] > 2.0),
        "GC2_S3_gt_1.5": bool(partA["S3_punitive"] > 1.5),
        "GC3_soft_capacity_usdM": soft_cap,
    }
    gates["PASS"] = gates["GC0_reconciles"] and gates["GC1_S2_gt_2.0"] and gates["GC2_S3_gt_1.5"]
    print(f"\nGATES: {gates}")

    manifest = {
        "protocol": "docs/research/medallion_validation_protocol.md (Amendment A)",
        "git_commit": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip(),
        "tiers_usd": TIERS_USD, "scenarios_bps": SCENARIOS, "impact_c_bps": IMPACT_C,
        "reconciliation": {"engine_flat30": eng_sortino, "per_name_flat30": ours_s0},
        "part_a_tiered_oos_sortino": partA, "part_b_capacity_oos_sortino": partB,
        "gates": gates, "headline_flat30_oos_sortino": 2.84,
        "reproduce": "PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_costs.py",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "medallion_cost_sensitivity.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {OUT_DIR / 'medallion_cost_sensitivity.json'}")


if __name__ == "__main__":
    main()
