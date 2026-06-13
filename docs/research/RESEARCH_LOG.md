# Research Log — Medallion Lite / K2-Atlas crypto Sortino study

Chronological journal of experiments, results, and decisions. Newest first. Each entry links to
the detailed writeup and the harness that produced it. All figures are net of 30 bps one-way,
survivorship-free (point-in-time universe), param-frozen walk-forward unless stated. OOS = 2023+.

Standing references:
- Strategy card: [`medallion_lite_strategy_card.md`](medallion_lite_strategy_card.md)
- Investor report: [`medallion_research_report.md`](medallion_research_report.md)
- Registry entry: `registry/alphas/2026-06-medallion-lite.yaml`
- Harnesses: `scripts/research/k2_atlas/run_medallion_*.py`

---

## 2026-06-13 · Pre-registered auditable validation → frozen headline 2.84
**Question.** Establish one defensible, auditable performance figure (frozen-as-headline, WF as
upper bound) and correct the param-selection optimism in the prior 2.95.
**Method.** Pre-registered protocol committed *before* the run (`medallion_validation_protocol.md`,
commit `2ed0402`); deterministic harness (`run_medallion_audit.py`) emitting a JSON provenance
manifest (`audit_code_clean`, versions, data fingerprint, gates, all results) + daily-return CSV.
5-factor, top-100 PIT, within-universe rank, 30 bps. Acceptance gates G1–G4.
**Result — ALL GATES PASS.** Frozen-param **OOS Sortino 2.84** (Sharpe 2.15, Calmar 4.73, CAGR
173%, MaxDD −37%) vs BTC **1.78**; walk-forward upper bound 2.95. Per-fold 4.72/2.72/1.67 (all
positive). Cost-robust 3.50→2.84(30bps)→2.42(50bps). PSR vs 0 = 1.00.
**Decision.** **2.84 is the headline** (no params fit on data); 2.95 is the WF upper bound. Card,
registry (`oos_sortino` 2.95→2.84), test, and investor report revised. Process fix: a benchmark
reindex bug (BTC read 0.07) was caught and fixed (`65dd0c7`) before publishing; manifest re-run on
a clean commit. The factor zoo + ensembling remain rejected (PBO).
**Detail:** [`medallion_validation_protocol.md`](medallion_validation_protocol.md) ·
**Harness:** `run_medallion_audit.py` · **Manifest:** `artifacts/medallion_audit/`

---

## 2026-06-13 · Multiple-testing validation (DSR + PBO) — bagging overturned
**Question.** Does the vote-bagging lift survive selection-bias correction?
**Method.** Ensemble DESIGN space (mean/vote × m × K = 18 trials, **frozen** params to isolate the
design choice) through `src/afml/backtest_stats` Deflated Sharpe + matrix-form CSCV PBO.
**Result.** **No.** Under frozen params all 18 designs cluster at OOS Sortino **2.66–2.91**
(5-factor 2.77, 100-factor 2.79) — statistically indistinguishable. **DSR p≈1.00** confirms the
*base* factor signal's Sharpe is genuine, but the trial dispersion is ~0 so it credits the ensemble
with nothing. **PBO 0.70–0.77 FAILS** the ≤0.5 gate (IS-best design lands below OOS median ~70%;
rank-corr ≈ 0). The earlier 3.05 was an artifact of the *inner walk-forward param selection*, not
the vote aggregation (frozen vote = 2.82).
**Decision.** **Do NOT adopt the factor zoo or any ensemble of it.** Keep the simple 5-factor
composite on top-100. Dependable levers remain universe breadth + risk overlays, not factor
count/ensembling. Textbook PBO catch (QF-21).
**Detail:** [`medallion_bagging_experiment.md`](medallion_bagging_experiment.md) (validation section) ·
**Harness:** `run_medallion_validation.py`

---

## 2026-06-13 · Bagged ensemble of the 100-factor zoo  ⚠️ superseded by validation above
**Question.** Can a bagged ensemble of the 100 TA-Lib factors beat the single composite?
**Method.** Random-subspace bagging (K members, each a random subset of m factors, each member
re-ranked to uniform selectivity — the guardrail); aggregate by **mean** vs **vote** (top-tercile
consensus); same top-100 / within-universe / walk-forward / 30 bps. Seed-stability check on the
winner.
**Result.** **Mean (linear) bagging adds nothing** — converges to the equal-weight 100-factor
composite (~2.88). **Vote (non-linear consensus) bagging is a genuine, stable lift:** Sortino
**3.04 ± 0.13** across 5 seeds (vs 100-factor 2.88, 5-factor 2.52), with a *better* drawdown
(−37% vs −41%) and the best vol-target (3.45). Mechanism is sensible: consensus downweights names
propped up by a single extreme indicator.
**Decision.** **Most promising result so far, but NOT promoted.** Needs deflated-Sharpe / PBO over
the ensemble design space and reconciliation to the validated 2.95 baseline construction before
adoption. Kept in research, not in the card/registry.
**Detail:** [`medallion_bagging_experiment.md`](medallion_bagging_experiment.md) ·
**Harness:** `run_medallion_bagging.py`

---

## 2026-06-12 · Universe stress-test — top-250
**Question.** Does pushing breadth further, to top-250, help beyond the adopted top-100?
**Method.** Added `top_250` to the canonical universe sweep (same point-in-time / within-universe /
walk-forward method).
**Result.** **No — it hurts.** top_250 (~177 avg names) gives honest WF-OOS Sortino **2.28** vs
top_100's **2.95**, with a worse drawdown (**−47%** vs −35%) and a lower vol-target uplift
(2.21 vs 2.90). It sits in the over-widening regime alongside top_200 / all_usd (~2.3–2.5), still
ahead of BTC (1.78) but clearly inferior to the ~70–100 optimum.
**Decision.** **Stay at top-100.** Confirms the breadth optimum is ~70–100 names; the deep illiquid
tail dilutes the edge.
**Detail:** [`medallion_universe_sweep.md`](medallion_universe_sweep.md) ·
**Harness:** `run_medallion_universe.py`

---

## 2026-06-12 · Factor-count experiment — 5 vs 100 TA-Lib factors
**Question.** Does expanding the composite from 5 hand-chosen factors to a 100-indicator TA-Lib
zoo improve OOS performance?
**Method.** Same top-100 universe / within-universe ranking / walk-forward; equal-weight,
convention-oriented ranks (no return-fitted weights or signs). Diagnostics: avg holdings + a
re-ranked (equal-selectivity) control.
**Result.** The naïve 100-factor result (Sortino **5.95**, CAGR **1497%**, DD **−55%**) was a
**concentration artifact** — averaging collinear ranks compressed dispersion (0.188→0.146), so the
book held ~12 names vs ~22 and piled into a lucky handful. Re-ranked to equal selectivity it
collapses to Sortino **2.85** vs the 5-factor **2.52** — a small, fragile lift with a worse drawdown.
**Decision.** **Not adopted.** Breadth (~+1.0 Sortino) dominates factor count (~+0.3 at best).
Demonstrates the QF-21 multiple-testing trap. Principled follow-up (IC screen + decorrelation +
deflated-Sharpe/PBO) noted, not yet run.
**Detail:** [`medallion_factor_count_experiment.md`](medallion_factor_count_experiment.md) ·
**Harness:** `run_medallion_factors.py` (requires TA-Lib 0.6.8) · **Commit:** `ab7cb93`

---

## 2026-06-12 · Universe-definition sweep + reconciliation → adopt top-100
**Question.** As a capacity-constrained shop able to trade small assets, should we widen beyond the
top-50 universe, and to what?
**Method.** Point-in-time (20d trailing-ADV, survivorship-free) sweep over breadth; per-symbol
factors computed once, ranked **within** each candidate universe; param-frozen walk-forward. A
`membership` spec replays the committed top-50 table to reconcile against the validated baseline.
**Result.** Reconciliation ✓ — `membership` reproduces WF-OOS Sortino **1.97** (≈ prior 1.97–2.03),
confirming the harness (an earlier rank-over-all-then-mask version spuriously read top-50 at 2.65;
fixed by within-universe ranking). Widening to **top-100 (~93 names)** lifts honest WF-OOS Sortino
**1.97 → 2.95** with a *better* drawdown (−35%), +vol-target 3.04. `adv ≥ $1M` (~72 names) is the
liquidity-floor alternative (2.46). Over-widening (top_200 / adv≥$250k / all_usd) fades to ~2.47.
**Decision.** **Adopted top-100** (registry + card + investor report updated; `oos_sortino` 2.03→2.95).
Caveats: 30 bps optimistic for rank 50–100 names (tiered-cost re-test pending); universe is
reconstructed, not from a committed membership table.
**Detail:** [`medallion_universe_sweep.md`](medallion_universe_sweep.md) ·
**Harness:** `run_medallion_universe.py` · **Commits:** `b56d405`, `a0b4ebd`

---

## 2026-06-12 · Honest revalidation of the flagship (survivorship + walk-forward)
**Question.** Does Medallion Lite genuinely beat buy-and-hold, and what is the defensible Sortino?
**Method.** Rebuilt the flagship on a survivorship-free point-in-time universe; param-frozen
walk-forward; vol-target / regime overlays (QF-07 / MR-09).
**Result.** As-shipped OOS Sortino **2.70** was look-ahead survivorship (full-period-ADV universe).
Point-in-time + walk-forward corrects to **1.97–2.03** (top-50), vs BTC **1.78**, with smaller
drawdowns. Vol-target overlay → 2.33. Per-fold OOS decays 3.49 (2023) → 1.97 (2024) → 1.11 (2025-26).
Regime-tilt overlay *hurt* (built-in gate already suffices).
**Decision.** Registered with an honest `validation` block (new schema) + provenance. The ~0.7
gap between 2.70 and 2.0 was bias.
**Harnesses:** `run_medallion_pit.py`, `run_medallion_walkforward.py`, `run_medallion_sortino.py`

---

## Open follow-ups (backlog)
- **Tiered-cost sensitivity test** — gating item before treating the top-100 2.95 as production-grade.
- **Committed point-in-time top-100 membership table** in the lake (replace on-the-fly reconstruction).
- **Disciplined factor selection** — per-fold IC screen + cluster-decorrelation + deflated-Sharpe/PBO.
- **Per-fold decomposition of the top-100 2.95** (check the decay pattern seen at top-50).
