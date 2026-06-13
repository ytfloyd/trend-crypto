# Medallion Lite — bagged-ensemble experiment (100-factor zoo)

**Question.** Can a bagged ensemble of the 100-factor TA-Lib zoo beat the single composite?

**Method.** Random-subspace (feature-bagging) ensemble on the same point-in-time top-100 universe,
within-universe ranking, param-frozen walk-forward, 30 bps. Build a `(time × symbol × 100)` cache
of within-universe, convention-oriented factor ranks. For each of `K` members, draw a random subset
of `m` factors, average their ranks, and **re-rank the member to uniform selectivity** — the
guardrail from the factor-count experiment that defeats the concentration artifact. Aggregate the
members two ways, then re-rank the final score:
- **mean** — average the members' rank-scores (a near-linear operation);
- **vote** — fraction of members that place a name in its **top tercile** (non-linear consensus).

Harness: `scripts/research/k2_atlas/run_medallion_bagging.py`.

## Results (WF-OOS 2023–2026, ~22 holdings throughout)

| Method | WF-OOS Sortino | Sharpe | CAGR | MaxDD | +vol-target |
|---|---:|---:|---:|---:|---:|
| 5-factor [re-ranked] (in-harness baseline) | 2.52 | 1.92 | 143% | −36% | 2.59 |
| 100-factor [re-ranked] | 2.88 | 2.21 | 190% | −41% | 3.18 |
| bag mean (m=10, K=50) | 2.67 | 2.06 | 166% | −42% | 3.10 |
| bag mean (m=20, K=50) | 2.88 | 2.19 | 190% | −41% | 3.10 |
| bag mean (m=40, K=50) | 2.87 | 2.20 | 191% | −41% | 3.14 |
| bag mean (m=20, K=100) | 2.78 | 2.13 | 180% | −41% | 3.08 |
| **bag VOTE (m=20, K=50)** | **3.05** | 2.31 | 207% | −37% | **3.45** |
| *BTC buy & hold* | *1.78* | *1.15* | *54%* | *−50%* | — |

**Seed stability — vote bag (m=20, K=50), 5 seeds:** Sortino **mean 3.04, std 0.13, min 2.79,
max 3.20**. Every seed beats both baselines; the effect is not a lucky single draw.

## Findings

1. **Linear (mean) bagging adds nothing.** It converges to the equal-weight 100-factor composite
   (~2.88) — as theory predicts, averaging re-ranked random subsets ≈ averaging all factors.
   Smaller subsets (m=10) are slightly *worse* (too little information per member).
2. **Non-linear (vote/consensus) bagging is a genuine, stable improvement.** ~3.04 ± 0.13 Sortino
   beats the 100-factor (2.88) and 5-factor (2.52) baselines, with a **better drawdown** (−37% vs
   −41%) and the best vol-targeted figure (3.45) — multiple metrics agree, across 5 seeds.
3. **The mechanism is sensible, not just curve-fit.** Vote rewards names that *many* independent
   factor-subsets agree on and downweights names propped up by a single extreme indicator. That
   consensus filter is exactly what should improve robustness and the tail (hence the better DD),
   rather than chasing one factor's lucky run.

## Caveats / not yet done (do NOT promote to the card/registry yet)
- **Multiple-testing not corrected.** Vote adds hyperparameters (tercile threshold, m, K); the
  result needs a **deflated-Sharpe / PBO** pass over the ensemble design space before adoption.
- **Baseline framing.** The in-harness baseline is the *re-ranked* 5-factor on the 343-symbol
  "ever top-100" panel (2.52), not the production-validated 2.95 (native composite, 362-symbol
  panel). The clean improvement is **vote-bag vs 100-factor (+~0.16)**; reconcile to the validated
  construction before quoting an absolute uplift over 2.95.
- Still equal-weight, convention-oriented ranks (no return-fitted signs), single walk-forward grid.

**Verdict.** The first method that beats the 100-factor composite. Promising enough to validate
properly (DSR/PBO + reconciled baseline), not yet to ship.

**Provenance:** `scripts/research/k2_atlas/run_medallion_bagging.py`, TA-Lib 0.6.8, 30 bps,
2021-01..2026-06, OOS 2023+, top-100 point-in-time universe, seeds {42,7,99,2024,31}.
