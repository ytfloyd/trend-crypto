"""Medallion Lite — multiple-testing validation of the bagged-ensemble result.

Gates the vote-bagging finding (medallion_bagging_experiment.md) against selection bias:

  * Deflated Sharpe Ratio (DSR) — reuses src/afml/backtest_stats (Bailey & Lopez de Prado).
    Deflates the winner's Sharpe for the number of ensemble designs searched (n_trials), the
    dispersion of trial Sharpes, sample length, and non-normality. p>0.95 => genuine skill.
  * PBO via CSCV (Combinatorially Symmetric Cross-Validation, AFML Ch.12) — matrix form over
    the config family: split the daily-return matrix into S blocks, take all C(S,S/2) train/test
    partitions, and measure how often the best in-sample config lands below the OOS median.
    PBO <= 0.5 acceptable; lower is better.

The "trials" are the ensemble DESIGN space (agg in {mean,vote} x subset m x members K), each
scored at frozen flagship params on the top-100 within-universe-ranked universe, 30 bps.

Run: PYTHONPATH=scripts/research/k2_atlas:scripts/research:src \
       python scripts/research/k2_atlas/run_medallion_validation.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, spearmanr

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "scripts" / "research" / "k2_atlas"), str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_medallion_bagging as bag  # noqa: E402
import run_medallion_factors as fac  # noqa: E402
import run_medallion_universe as uni  # noqa: E402
import run_medallion_walkforward as wf  # noqa: E402
from afml.backtest_stats import deflated_sharpe_ratio, expected_max_sharpe  # noqa: E402

SEED = 42
OOS_START = "2023-01-01"
PARAMS = {"entry_threshold": 0.65, "trailing_stop_pct": 0.15}


def sr_obs(r: pd.Series) -> float:
    """Per-observation (non-annualised) Sharpe — the unit the DSR formula expects."""
    r = r.dropna()
    return float(r.mean() / r.std()) if r.std() > 0 else 0.0


def cscv_pbo(M: pd.DataFrame, S: int = 12) -> dict:
    """Matrix-form CSCV PBO (Bailey & Lopez de Prado, AFML Ch.12)."""
    arr = M.to_numpy()
    T, N = arr.shape
    blocks = np.array_split(np.arange(T), S)
    combos = list(itertools.combinations(range(S), S // 2))

    def block_sr(rows):
        x = arr[rows]
        mu, sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        return np.where(sd > 0, mu / sd, 0.0)

    lambdas, relranks, corrs = [], [], []
    for test in combos:
        test_idx = np.concatenate([blocks[b] for b in test])
        train_idx = np.concatenate([blocks[b] for b in range(S) if b not in test])
        is_sr, oos_sr = block_sr(train_idx), block_sr(test_idx)
        nstar = int(np.argmax(is_sr))
        rank = int(oos_sr.argsort().argsort()[nstar]) + 1  # 1..N
        omega = min(max(rank / (N + 1.0), 1e-6), 1 - 1e-6)
        lambdas.append(np.log(omega / (1 - omega)))
        relranks.append(omega)
        corrs.append(spearmanr(is_sr, oos_sr).statistic)
    lam = np.array(lambdas)
    return {"pbo": float((lam <= 0).mean()), "n_combos": len(combos),
            "median_oos_relrank": float(np.median(relranks)),
            "mean_rank_corr": float(np.nanmean(corrs))}


def main() -> None:
    print("loading panel + rank cache ...")
    dd, rw, regime, factors5, cw, sym_arr = fac.load_panel()
    full_index = cw.index
    elig_h = uni.eligibility(dd, None, "top", 100, cw.columns, full_index)
    R, cols = bag.build_rank_cache(fac.factor_specs(), sym_arr, full_index, elig_h)

    # ---- build the trial family (ensemble design space) ----
    comp5 = fac.rerank(uni.composite_within(factors5, elig_h), elig_h)
    comp100 = fac.rerank(bag.to_df(np.nanmean(R, axis=2), full_index, cols, elig_h), elig_h)
    family = {"5-factor": comp5, "100-factor": comp100}
    for agg in ("mean", "vote"):
        for m in (10, 20, 40, 60):
            for K in (50, 100):
                comp = fac.rerank(bag.to_df(bag.ensemble(R, m, K, SEED, agg=agg), full_index, cols, elig_h), elig_h)
                family[f"{agg}_m{m}_K{K}"] = comp
    print(f"  trial family: {len(family)} ensemble designs (frozen params, 30bps)")

    # ---- frozen-param daily returns per config ----
    daily = {}
    for name, comp in family.items():
        daily[name], _ = uni._config_daily(comp, rw, regime, PARAMS)

    # ---- rank the family by OOS Sortino; identify the winner ----
    print("\n=== trial family — OOS 2023+ ===")
    rows = []
    for name, d in daily.items():
        m = uni._metrics(d[d.index >= OOS_START])
        rows.append((name, m["sortino"], m["sharpe"], sr_obs(d[d.index >= OOS_START])))
    rows.sort(key=lambda x: -x[1])
    for name, so, sh, _ in rows:
        print(f"  {name:<16} OOS Sortino {so:>5.2f}  Sharpe {sh:>4.2f}")
    winner = rows[0][0]
    print(f"  --> winner by OOS Sortino: {winner}")

    # ---- Deflated Sharpe Ratio (selection-bias corrected) ----
    oos = {n: d[d.index >= OOS_START] for n, d in daily.items()}
    trial_obs_sr = np.array([sr_obs(oos[n]) for n in family])
    n_trials = len(family)
    benchmark = expected_max_sharpe(n_trials, mean_sharpe=0.0, std_sharpe=float(trial_obs_sr.std(ddof=1)))
    w = oos[winner].dropna()
    obs = sr_obs(w)
    dsr = deflated_sharpe_ratio(obs, benchmark, len(w),
                                skewness=float(skew(w)), excess_kurtosis=float(kurtosis(w)))
    print("\n=== DEFLATED SHARPE (selection bias over the ensemble design space) ===")
    print(f"  winner per-obs Sharpe {obs:.3f}  | trials {n_trials}  trial-SR std {trial_obs_sr.std(ddof=1):.3f}")
    print(f"  E[max SR] benchmark   {benchmark:.3f}  (haircut from {n_trials} designs)")
    print(f"  skew {skew(w):.2f}  excess-kurt {kurtosis(w):.2f}  n_obs {len(w)}")
    print(f"  DSR p-value = {dsr:.3f}   (> 0.95 => genuine after deflation)")

    # ---- PBO via CSCV over the full-period daily-return matrix ----
    M = pd.DataFrame(daily).dropna(how="all").fillna(0.0)
    pbo12 = cscv_pbo(M, S=12)
    pbo16 = cscv_pbo(M, S=16)
    print("\n=== PBO via CSCV (full period 2021-2026) ===")
    for tag, r in (("S=12", pbo12), ("S=16", pbo16)):
        print(f"  {tag}: PBO {r['pbo']:.2f}  ({r['n_combos']} splits)  "
              f"median OOS rel-rank {r['median_oos_relrank']:.2f}  IS/OOS rank-corr {r['mean_rank_corr']:+.2f}")
    print("\n  Interpretation: DSR>0.95 => winner survives multiple-testing; "
          "PBO<=0.5 => low overfitting (IS-best stays above OOS median). 30bps, top-100.")


if __name__ == "__main__":
    main()
