"""Medallion Lite — AUDITABLE validation run (pre-registered).

Implements docs/research/medallion_validation_protocol.md exactly. Deterministic (no RNG on the
5-factor path), fully provenanced. Emits a JSON manifest (git commit, package versions, data
fingerprint, config, gates, every result) and a daily-return CSV so a third party can re-derive
the metrics independently.

Headline = FROZEN-param OOS; walk-forward = labeled UPPER BOUND. 5-factor composite, point-in-time
top-100 universe, survivorship-free, within-universe ranking, 30 bps.

Run: PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_audit.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "scripts" / "research" / "k2_atlas"), str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import duckdb  # noqa: E402
import scipy  # noqa: E402

import run_medallion_universe as uni  # noqa: E402
import run_medallion_walkforward as wf  # noqa: E402
from afml.backtest_stats import deflated_sharpe_ratio  # noqa: E402  (PSR via benchmark=0)
from core.metrics import compute_metrics  # noqa: E402
from medallion_lite.portfolio import backtest_portfolio, build_factor_portfolio  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "medallion_audit"
FLAGSHIP = {"entry_threshold": 0.65, "trailing_stop_pct": 0.15}
OOS_START = "2023-01-01"
COST_GRID = (0.0, 10.0, 20.0, 30.0, 50.0)


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def provenance() -> dict:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "duckdb": duckdb.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def daily_at_cost(comp_pit, rw, regime, params, tc_bps) -> pd.Series:
    w, _ = build_factor_portfolio(
        comp_pit, rw, regime, entry_threshold=params["entry_threshold"],
        exit_score_threshold=0.40, max_hold_hours=336, trailing_stop_pct=params["trailing_stop_pct"],
        rebalance_every_hours=24, max_positions=25, max_weight=0.10)
    bt = backtest_portfolio(w, rw, tc_bps=tc_bps)
    return wf._daily(pd.Series(bt["net_ret"].values, index=pd.to_datetime(bt["ts"])))


def metrics(daily: pd.Series, lo=None) -> dict:
    d = (daily if lo is None else daily[daily.index >= lo]).dropna()
    m = compute_metrics((1.0 + d).cumprod())
    return {k: round(float(m[k]), 4) for k in
            ("total_return", "cagr", "vol", "sharpe", "sortino", "calmar", "max_dd", "hit_rate", "n_days")}


def psr_zero(daily_oos: pd.Series) -> dict:
    d = daily_oos.dropna()
    sr_obs = float(d.mean() / d.std()) if d.std() > 0 else 0.0
    p = deflated_sharpe_ratio(sr_obs, 0.0, len(d), skewness=float(skew(d)), excess_kurtosis=float(kurtosis(d)))
    return {"per_obs_sharpe": round(sr_obs, 4), "n_obs": int(len(d)),
            "skew": round(float(skew(d)), 3), "excess_kurt": round(float(kurtosis(d)), 3),
            "psr_vs_zero": round(float(p), 4)}


def main() -> None:
    print("AUDIT RUN — pre-registered protocol (deterministic, provenanced)\n")
    factors, rw, regime, dd, memb = uni.load_universe_panel()
    cols, idx = factors["momentum"].columns, factors["momentum"].index
    elig_h = uni.eligibility(dd, memb, "top", 100, cols, idx)
    comp5 = uni.composite_within(factors, elig_h)              # within-universe 5-factor composite
    btc = wf._daily(rw["BTC-USD"])  # daily series — do NOT reindex onto the hourly index

    # headline (frozen) + cost sweep + per-fold
    frozen = daily_at_cost(comp5, rw, regime, FLAGSHIP, 30.0)
    cost_sweep = {f"{int(c)}bps": metrics(daily_at_cost(comp5, rw, regime, FLAGSHIP, c), OOS_START)["sortino"]
                  for c in COST_GRID}
    per_fold = {}
    for _, _, te_lo, te_hi in wf.FOLDS:
        seg = frozen[(frozen.index >= te_lo) & (frozen.index <= te_hi)]
        per_fold[f"{te_lo[:7]}..{te_hi[:7]}"] = metrics(seg)["sortino"]

    # walk-forward upper bound (reuses the validated harness path)
    wf_oos = uni.walk_forward_oos(factors, rw, regime, elig_h)

    res = {
        "headline_frozen": {"full": metrics(frozen), "oos": metrics(frozen, OOS_START)},
        "upper_bound_walk_forward": {"oos": metrics(wf_oos)},
        "benchmark_btc": {"oos": metrics(btc, OOS_START), "full": metrics(btc)},
        "per_fold_oos_sortino_frozen": per_fold,
        "significance_psr": psr_zero(frozen[frozen.index >= OOS_START]),
        "cost_sensitivity_oos_sortino": cost_sweep,
    }

    h, b = res["headline_frozen"]["oos"], res["benchmark_btc"]["oos"]
    pos_folds = sum(1 for v in per_fold.values() if v > 0)
    gates = {
        "G1_beats_btc": bool(h["sortino"] > b["sortino"]),
        "G2_psr_ge_0.95": bool(res["significance_psr"]["psr_vs_zero"] >= 0.95),
        "G3_majority_folds_positive": bool(pos_folds >= 2),
        "G4_cost_robust": bool(cost_sweep["30bps"] > 1.5 and cost_sweep["50bps"] > 1.0),
    }
    gates["ALL_PASS"] = all(gates.values())

    manifest = {
        "protocol": "docs/research/medallion_validation_protocol.md",
        "provenance": provenance(),
        "data": {"lake": uni.LAKE.split("/")[-1], "date_range": [uni.START, uni.END], "oos_start": OOS_START,
                 "n_symbols_panel": int(len(cols)), "n_hourly_bars": int(len(idx)),
                 "universe": "point-in-time top-100 by 20d trailing ADV (survivorship-free, within-universe rank)"},
        "config": {"signal": "5-factor composite (medallion_lite.factors)", "params_frozen": FLAGSHIP,
                   "exit_score_threshold": 0.40, "max_hold_hours": 336, "rebalance_hours": 24,
                   "max_positions": 25, "max_weight": 0.10, "costs_bps_headline": 30.0, "folds": wf.FOLDS},
        "results": res, "gates": gates,
        "reproduce": "PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_audit.py",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = (manifest["provenance"]["git_commit"] or "nocommit")[:10]
    mpath = OUT_DIR / f"medallion_audit_{commit}.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    pd.DataFrame({"date": frozen.index, "strat_frozen": frozen.values,
                  "strat_wf": wf_oos.reindex(frozen.index).values,
                  "btc": btc.reindex(frozen.index).values}).to_csv(
        OUT_DIR / f"medallion_audit_{commit}_daily_returns.csv", index=False)

    print(f"HEADLINE (frozen, OOS 2023+):  Sortino {h['sortino']:.2f}  Sharpe {h['sharpe']:.2f}  "
          f"Calmar {h['calmar']:.2f}  CAGR {h['cagr']:.0%}  MaxDD {h['max_dd']:.0%}")
    print(f"UPPER BOUND (walk-forward):    Sortino {res['upper_bound_walk_forward']['oos']['sortino']:.2f}")
    print(f"BTC buy&hold (OOS):            Sortino {b['sortino']:.2f}  Sharpe {b['sharpe']:.2f}")
    print(f"PSR vs 0 (frozen OOS):         {res['significance_psr']['psr_vs_zero']:.3f}")
    print(f"per-fold OOS Sortino (frozen): {per_fold}")
    print(f"cost sweep OOS Sortino:        {cost_sweep}")
    print(f"GATES: {gates}")
    print(f"\nmanifest  -> {mpath}")


if __name__ == "__main__":
    main()
