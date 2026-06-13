# Medallion Lite — Research Sleeve Summary (capstone)

**K2 TRADE ATLAS · systematic digital-asset Sortino study.** Branch `research/k2-atlas-sortino`.
This document consolidates the full sleeve: what was tested, every result across rigor / universe /
feature / cost dimensions, the validation chain, the current status, and how to reproduce it.
It is the single source of truth; the per-experiment docs and harnesses it links are the detail.

> **Status (2026-06-13): ON HOLD pending one input — small-cap round-trip execution cost.**
> Signal validated (survivorship-free, walk-forward, DSR p≈1.0, PBO-checked, gates G1–G4 pass).
> Not a deploy and not a kill until the execution-cost number is known.

---

## 1. Mandate & strategy

**Mandate.** Identify systematic crypto (OHLCV) strategies with **Sortino > 2.0**, long-convexity
bias, from the K2 TRADE ATLAS rulebook (QF-01 cross-sectional momentum, QF-07 vol-targeting,
QF-10/CV-17 trend-convexity, QF-21 data-snooping discipline, MR-09 regime).

**Medallion Lite.** Cross-sectional crypto factor strategy: rank the liquid universe each bar on a
**5-factor composite** (momentum, volume surge, realised vol, proximity-to-high, risk-adjusted
momentum); gate gross exposure with an **ensemble market-regime** score; run an **event-driven
portfolio** (enter >0.65 / exit <0.40, ≤25 positions, ≤10%/name, 15% trailing stop, ≤14-day hold,
24h rebalance). Long-biased; convexity is indirect (stops + regime gate truncate the left tail).
Registry id `2026-06-medallion-lite`.

---

## 2. Performance by rigor stage (the honesty arc)

Each step removed an optimistic assumption; nothing about the strategy changed.

| Stage | OOS Sortino | What was removed |
|---|--:|---|
| As-shipped flagship | 2.70 | — (full-period-ADV universe = look-ahead survivorship) |
| Point-in-time universe + walk-forward (top-50) | 1.97–2.03 | survivorship + parameter look-ahead |
| Widen to top-100 (walk-forward) | 2.95 | (added breadth — genuine) |
| **Pre-registered audit, frozen params (top-100)** | **2.84** | parameter-selection optimism → **headline** |
| Realistic liquidity-tiered costs | 1.42 | flat-cost assumption (← open hurdle, §5) |

All costs are **round-trip** (the engine charges `tc_bps` round-trip ≈ 15 bps/side). The 2.84
headline is a **~30 bps round-trip** assumption.

---

## 3. Performance by universe

Point-in-time (20d trailing ADV), survivorship-free, **within-universe** cross-sectional ranking,
30 bps round trip. Frozen = no params fit; WF = walk-forward upper bound. (`medallion_universe_sweep.md`)

| Universe | ~names | Frozen OOS Sortino | WF OOS Sortino | MaxDD |
|---|--:|--:|--:|--:|
| membership top-50 (validated baseline) | ~50 | 1.83 | 1.97 | −38% |
| top-25 | 25 | 2.09 | — | −35% |
| top-50 | 49 | 2.60 | 2.29 | −38% |
| **top-100 (adopted)** | 93 | **2.84** | **2.95** | −37% |
| top-200 | 161 | 2.47 | — | −43% |
| top-250 | 177 | 2.46 | 2.28 | −41% |
| adv ≥ $1M | 72 | 2.76 | 2.46 | −32% |
| adv ≥ $250k | 124 | 2.48 | — | −43% |
| all USD (~360) | 193 | 2.48 | — | −41% |
| *BTC buy & hold* | — | *1.78* | *1.78* | *−50%* |

**Finding:** breadth is a real lever up to an optimum of ~70–100 names; over-widening (top-200+/
all-USD) fades to ~2.4–2.5 with worse drawdowns. Reconciliation ✓ — the committed membership table
reproduces the 1.97 baseline under within-universe ranking.

---

## 4. Performance by feature set (factor count / ensembling)

Same top-100 / within-universe / walk-forward. (`medallion_factor_count_experiment.md`,
`medallion_bagging_experiment.md`, `run_medallion_validation.py`)

| Feature set | OOS Sortino | Verdict |
|---|--:|---|
| **5-factor composite (adopted)** | **2.84** (frozen) / 2.95 (WF) | baseline |
| 100-factor TA-Lib zoo (naïve avg) | 5.95 *(artifact)* | concentration artifact (12 vs 22 holdings) — rejected |
| 100-factor, equal-selectivity (re-ranked) | 2.85 | ≈ baseline, no real lift |
| Bagged ensemble — mean aggregation | ~2.88 | converges to the composite, adds nothing |
| Bagged ensemble — vote aggregation | 3.04 *(pre-DSR)* | **rejected**: PBO 0.70–0.77; under frozen params all designs cluster 2.66–2.91 |

**Finding:** more features did **not** robustly help. The naïve 100-factor "win" was a
concentration artifact; vote-bagging's apparent lift was a parameter-selection artifact that failed
the overfitting test (PBO). **Breadth (~+1.0 Sortino) dominates factor count (~+0.3 at best,
non-robust).** We run the simple 5-factor model.

---

## 5. Performance by cost — the open hurdle

`tc_bps` is round-trip. The headline assumes ~30 bps RT. Cost is the **one unresolved gating
input** (realized small-cap execution cost, TBD by separate execution research).
(`medallion_validation_protocol.md` Amendments A/B, `medallion_cost_sensitivity.json`,
`medallion_cost_universe.json`)

**Flat round-trip sweep (top-100):**

| Round-trip cost | 0 | 10 | 20 | **30 (headline)** | 50 | ~70 (breakeven) |
|---|--:|--:|--:|--:|--:|--:|
| OOS Sortino | 3.50 | 3.27 | 3.05 | **2.84** | 2.42 | ≈2.0 |

**Liquidity-tiered (cost rises as ADV falls) + market impact:**

| Scenario | OOS Sortino |
|---|--:|
| S1 benign (10/20/40/70 RT) | 2.26 |
| **S2 realistic (20/40/70/120 RT)** | **1.42** |
| S3 punitive (35/70/130/220 RT) | −0.13 |
| S2 + √-impact @ $25M AUM | 0.67 |

**Cost-robust universe search** (can liquid-only names dodge the hurdle?): under S2, top-25 1.29 /
top-50 1.45 / top-100 1.42 / **adv≥$50M 1.72 (best)** / adv≥$20M 1.33 — all below 2.0. Liquid-only
is cost-robust but too thin to carry the alpha.

**The crux.** The alpha concentrates in **small-cap names**, so the binding input is *their*
round-trip cost. If small-caps execute near liquid-name levels (≲30–40 bps RT) the 2.84 stands; if
they run 100 bps+ RT, the realistic-tiered case (~1.42) governs. **PASS/FAIL is decided by the
execution-cost number, not by more backtesting.**

---

## 6. Validation chain (why the signal is trusted)

| Control | Result |
|---|---|
| Survivorship-free point-in-time universe | ✓ (membership reconciles to 1.97 baseline) |
| Param-frozen walk-forward (select-on-train/freeze/score-on-test) | ✓ |
| Per-fold OOS Sortino (frozen, top-100) | 4.72 / 2.72 / 1.67 (’23/’24/’25-26) — all positive |
| Probabilistic Sharpe vs 0 (PSR) | **1.00** — base Sharpe is genuine |
| PBO on the feature/ensemble design space | 0.70–0.77 → design selection overfits ⇒ use simplest model |
| Pre-registered audit, gates G1–G4 | **all pass** (deterministic, provenance manifest + CSV) |

Reusable, deterministic, auditable: protocol pre-registered before the run; JSON manifest records
git commit, package versions, data fingerprint, config, gates, results; daily-return CSV for
independent re-derivation.

---

## 7. Was it the best in this sleeve? — yes (pre-cost)

The sleeve evaluated, on the same crypto OHLCV / survivorship-free / OOS-2023+ basis:
- **`sortino_hunt`** — a broad sweep of trend/momentum/breakout/carry-style rules: best OOS Sortino **~1.0**.
- **BTC buy & hold** (benchmark): OOS Sortino **1.78**.
- **Medallion Lite**: honest OOS Sortino **~2.0–2.95**.

**Pre-cost, Medallion Lite was clearly the best strategy in this pipeline** — the only one to clear
the >2.0 mandate, beating both the broad sweep and passive BTC. Caveat for context: the bar it
cleared was a pipeline whose other candidates mostly sat near 1.0, and "best pre-cost" is precisely
what the cost analysis put the asterisk on. The cross-sectional / convexity tracks in the wider
registry (`continuation-index`, `ma-5-40-trend`) are separate payoff shapes, not run in this
crypto-Sortino sleeve and not directly comparable here.

---

## 8. Bottom line & next step

- **Real, validated cross-sectional signal**; best in the sleeve; honest headline **2.84 @ ~30 bps RT**.
- **One open hurdle:** realized small-cap round-trip execution cost (TBD). It clears 2.0 to ~70 bps
  flat-equivalent, but the small-cap tier is the binding input.
- **Decision is execution-driven, not research-driven.** When the cost number lands: ≲30–40 bps RT
  on small-caps → paper-trade candidate; 100 bps+ → shelve.
- **Reusable infrastructure** left for the next candidate: point-in-time survivorship-free universe,
  within-universe ranking, cost-tiering + impact model, DSR/PBO, and the pre-registered auditable
  harness pattern.

---

## Artifact index

| Topic | Doc | Harness | Manifest |
|---|---|---|---|
| Strategy card | `medallion_lite_strategy_card.md` | — | — |
| Investor report | `medallion_research_report.md` | `medallion_report_pdf.py` | — |
| Research journal | `RESEARCH_LOG.md` | — | — |
| Universe sweep | `medallion_universe_sweep.md` | `run_medallion_universe.py` | — |
| Factor count | `medallion_factor_count_experiment.md` | `run_medallion_factors.py` | — |
| Bagging | `medallion_bagging_experiment.md` | `run_medallion_bagging.py` | — |
| DSR/PBO + audit | `medallion_validation_protocol.md` | `run_medallion_validation.py`, `run_medallion_audit.py` | `medallion_audit_<commit>.json` |
| Cost sensitivity | `medallion_validation_protocol.md` (Amd A/B) | `run_medallion_costs.py`, `run_medallion_cost_universe.py` | `medallion_cost_sensitivity.json`, `medallion_cost_universe.json` |
| Broad sweep (comparison) | — | `sortino_hunt.py` | — |

**Data:** `coinbase_crypto_ohlcv_lake.duckdb` (`bars_1h`, `bars_1d_usd_universe_clean`,
`…top50_adv10m_membership`). **Period:** 2021-01→2026-06, OOS 2023+. **Costs:** 30 bps round trip
(headline). All figures net of cost, survivorship-free.
