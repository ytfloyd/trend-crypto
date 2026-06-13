"""Medallion Lite — BAGGED ENSEMBLE of the 100-factor zoo.

Prior finding (medallion_factor_count_experiment.md): naively averaging 100 collinear
factor ranks compresses dispersion -> concentrated book -> inflated/unstable OOS; the
fair, equal-selectivity comparison is the RE-RANKED composite (5-factor 2.52, 100-factor 2.85).

Here we test whether BAGGING the factors does better. Random-subspace ensemble: build K
members, each from a random subset of m factors; RE-RANK each member to uniform selectivity
(the guardrail that defeats the concentration artifact); aggregate across members. Two
aggregations: (a) mean of member rank-scores, (b) vote = fraction of members ranking a name in
its top tercile. Final score re-ranked, then the SAME survivorship-free param-frozen
walk-forward (top-100 universe, within-universe ranking, 30 bps).

Run: PYTHONPATH=scripts/research/k2_atlas:scripts/research:src \
       python scripts/research/k2_atlas/run_medallion_bagging.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "scripts" / "research" / "k2_atlas"), str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_medallion_factors as fac  # noqa: E402  (load_panel, factor_specs, value_frame, avg_holdings, wf_from_composite, rerank, _disp)
import run_medallion_universe as uni  # noqa: E402
import run_medallion_walkforward as wf  # noqa: E402

SEED = 42
OOS_START = "2023-01-01"


def pct_rank(a: np.ndarray) -> np.ndarray:
    """NaN-aware cross-sectional percentile rank in [0,1] along axis=1 (per row)."""
    mask = ~np.isnan(a)
    filled = np.where(mask, a, np.inf)
    ranks = filled.argsort(axis=1).argsort(axis=1).astype("float32")  # 0..N-1; NaNs sort last
    cnt = mask.sum(axis=1, keepdims=True).astype("float32")
    pr = ranks / np.maximum(cnt - 1.0, 1.0)
    pr[~mask] = np.nan
    return pr


def build_rank_cache(specs, sym_arr, full_index, elig_h):
    """(T, N, F) float32 cache of within-universe, convention-oriented factor ranks."""
    cols = elig_h.columns
    em = elig_h.to_numpy()
    T, N, F = len(full_index), len(cols), len(specs)
    print(f"  rank cache {T}x{N}x{F} float32 ~{T*N*F*4/1e9:.1f} GB ...")
    R = np.full((T, N, F), np.nan, dtype="float32")
    for i, (label, fn, sign) in enumerate(specs):
        v = fac.value_frame(fn, sym_arr, full_index).reindex(columns=cols).to_numpy("float32")
        v = np.where(em, v, np.nan)
        pr = pct_rank(v)
        R[:, :, i] = (1.0 - pr) if sign < 0 else pr
        if (i + 1) % 25 == 0:
            print(f"    ... {i+1}/{F}")
    return R, cols


def ensemble(R, m, K, seed, agg="mean", replace=False):
    """Random-subspace bagging. Each member = re-ranked mean of m random factors.
    agg='mean' averages member rank-scores; agg='vote' averages top-tercile indicators."""
    T, N, F = R.shape
    rng = np.random.RandomState(seed)
    acc = np.zeros((T, N), dtype="float32")
    for _ in range(K):
        idx = rng.choice(F, size=m, replace=replace)
        member = pct_rank(np.nanmean(R[:, :, idx], axis=2))  # re-rank each member (guardrail)
        acc += (member > (2.0 / 3.0)).astype("float32") if agg == "vote" else member
    return acc / float(K)


def to_df(arr, full_index, cols, elig_h):
    return pd.DataFrame(arr, index=full_index, columns=cols).where(elig_h)


def report(name, comp_pit, rw, regime, n):
    oos = fac.wf_from_composite(comp_pit, rw, regime)
    m, mvt = uni._metrics(oos), uni._metrics(wf.vol_target(oos))
    nh = fac.avg_holdings(comp_pit, rw, regime)
    print(f"  {name:<34} n={n:<4} disp={fac._disp(comp_pit):.3f} hold~{nh:4.1f} | "
          f"Sortino {m['sortino']:>5.2f}  Sharpe {m['sharpe']:>4.2f}  "
          f"CAGR {m['cagr']:>6.0%}  DD {m['max_dd']:>5.0%}  (+vt {mvt['sortino']:.2f})")
    return m["sortino"]


def main() -> None:
    stability = len(sys.argv) > 1 and sys.argv[1] == "stability"
    print("loading panel (symbols ever in top-100) ...")
    dd, rw, regime, factors5, cw, sym_arr = fac.load_panel()
    full_index = cw.index
    elig_h = uni.eligibility(dd, None, "top", 100, cw.columns, full_index)
    specs = fac.factor_specs()
    R, cols = build_rank_cache(specs, sym_arr, full_index, elig_h)

    if stability:
        print("\n=== SEED STABILITY — vote bag (m=20, K=50) across seeds ===")
        sortinos = []
        for sd in (42, 7, 99, 2024, 31):
            ens = fac.rerank(to_df(ensemble(R, 20, 50, sd, agg="vote"), full_index, cols, elig_h), elig_h)
            s = report(f"vote m=20 K=50 seed={sd}", ens, rw, regime, "20x50")
            sortinos.append(s)
        print(f"\n  Sortino across seeds: mean {np.mean(sortinos):.2f}  "
              f"std {np.std(sortinos):.2f}  min {np.min(sortinos):.2f}  max {np.max(sortinos):.2f}")
        return

    # baselines (re-ranked = equal selectivity)
    comp5 = fac.rerank(uni.composite_within(factors5, elig_h), elig_h)
    comp100 = fac.rerank(to_df(np.nanmean(R, axis=2), full_index, cols, elig_h), elig_h)

    print("\n=== BAGGED ENSEMBLE vs baselines (top-100, within-universe rank, walk-forward, 30bps) ===")
    report("5-factor [re-ranked]", comp5, rw, regime, 5)
    report("100-factor [re-ranked]", comp100, rw, regime, 100)
    grid = [("mean", 10, 50), ("mean", 20, 50), ("mean", 40, 50),
            ("mean", 20, 100), ("vote", 20, 50)]
    for agg, m, K in grid:
        ens = to_df(ensemble(R, m, K, SEED, agg=agg), full_index, cols, elig_h)
        ens = fac.rerank(ens, elig_h)
        report(f"bag[{agg} m={m} K={K}]", ens, rw, regime, f"{m}x{K}")

    btc = wf._daily(rw["BTC-USD"])
    bo = uni._metrics(btc[btc.index >= OOS_START])
    print(f"\n  {'BTC buy&hold':<34} {'':<20} | Sortino {bo['sortino']:>5.2f}  "
          f"Sharpe {bo['sharpe']:>4.2f}  CAGR {bo['cagr']:>6.0%}  DD {bo['max_dd']:>5.0%}")
    print("\nMethod: random-subspace bagging; each member re-ranked to uniform selectivity "
          "(defeats the concentration artifact), aggregated, re-ranked. Equal-weight oriented "
          "ranks (no return-fitted weights/signs). 30bps, OOS 2023+.")


if __name__ == "__main__":
    main()
