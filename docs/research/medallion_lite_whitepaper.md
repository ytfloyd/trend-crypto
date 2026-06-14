---
title: "Medallion Lite — Technical Whitepaper"
subtitle: "A survivorship-free, cost-aware study of a cross-sectional crypto factor strategy"
author: "K2 TRADE ATLAS · Systematic Digital-Asset Research"
date: "2026-06-14"
---

# Medallion Lite — Technical Whitepaper

**Audience:** internal research team. **Status:** validated signal, **on hold** pending one input
(realized small-cap round-trip execution cost). **Registry id:** `2026-06-medallion-lite`.
**Repository:** `research/k2-atlas-sortino` (merged to `main`); all results reproducible from the
harnesses and manifests cited in Appendix D.

> **One-paragraph thesis.** Medallion Lite is a cross-sectional crypto factor strategy that ranks a
> point-in-time liquid universe on a five-factor composite, gates exposure by an ensemble regime
> score, and trades an event-driven, vol-scaled portfolio. On a survivorship-free, walk-forward,
> 30 bps round-trip basis its out-of-sample Sortino is **2.84 (frozen params) / 2.95 (walk-forward
> upper bound)** vs **1.78** for buy-and-hold Bitcoin — the best validated signal in our crypto
> book. The probabilistic Sharpe ratio confirms the signal is genuine (PSR≈1.0). **However**, the
> edge concentrates in small-capitalization names, and under realistic liquidity-tiered transaction
> costs the out-of-sample Sortino falls to **1.42** with negligible capacity. The strategy is
> therefore *not deployable as specified*; its viability reduces to a single empirical question —
> the achievable round-trip execution cost on sub-\$20M-ADV crypto — which separate execution
> research must answer.

---

## 1. Executive summary

- **What it is.** Renaissance-inspired cross-sectional factor model on liquid USD crypto spot
  pairs: rank → regime-gate → event-driven portfolio with ATR stops and an upward-only trailing
  stop. Long-biased; convexity is indirect (stops + regime gate truncate the left tail).
- **Validated performance (OOS 2023–2026, 30 bps round trip, survivorship-free, top-100 universe):**
  frozen-param Sortino **2.84**, Sharpe **2.15**, Calmar **4.73**, CAGR **173%**, max drawdown
  **−37%**; walk-forward upper bound **2.95**. Benchmark BTC: Sortino 1.78. All four pre-registered
  acceptance gates pass; PSR vs 0 ≈ **1.00**.
- **Why it is not yet deployable.** Those figures assume a uniform 30 bps round-trip cost. The alpha
  is concentrated in small-cap names; under liquidity-tiered costs the Sortino drops to **1.42**
  (below BTC), and a square-root market-impact model gives soft capacity **< \$5M AUM**.
- **What we ruled out along the way.** A 100-indicator TA-Lib factor zoo and bagged ensembles were
  tested and **rejected** — their apparent gains were a concentration artifact and a parameter-
  selection artifact respectively, both caught by a probability-of-backtest-overfitting (PBO) test.
- **The decision is execution-driven, not research-driven.** If small-cap round-trip cost is
  ≲ 30–40 bps, Medallion Lite is a paper-trade candidate; if it is 100 bps+, the realistic-cost
  case governs and the strategy is shelved.

---

## 2. Strategy specification

### 2.1 Signal — five-factor cross-sectional composite
At each bar, every name in the eligible universe is scored on five factors, each **cross-sectionally
percentile-ranked to [0,1]** within the universe, then combined by fixed weights. (Hourly lookbacks
of 168 = 7 days in the flagship pipeline.)

| Factor | Definition | Weight | Captures |
|---|---|--:|---|
| Momentum | `log(close / close[−168])` | 0.30 | trend strength |
| Volume surge | `vol_24h / mean(vol, 7d)`, clipped [0,5] | 0.15 | attention / flow |
| Realized vol | `std(ret, 168) · √8760` | 0.15 | trend-capture opportunity |
| Proximity-to-high | `1 + (close − max(high,168)) / max(high,168)` | 0.15 | trend persistence |
| Risk-adj momentum | `mean(ret,168) / std(ret,168)` | 0.25 | quality filter |

`composite = Σ wᵢ · rank_pct(factorᵢ)`, where `rank_pct` is the within-universe cross-sectional
percentile. The composite ∈ [0,1]; higher = more attractive long.

### 2.2 Regime gate
An ensemble score on BTC (trend + volatility state) scales gross exposure: full when constructive,
reduced or zero when the broad market deteriorates. This is the primary left-tail control at the
book level (distinct from the per-trade stop).

### 2.3 Portfolio — event-driven, vol-scaled
Not continuous rebalancing (which 30 bps costs would destroy). Signal-event entries/exits:

- **Enter** when smoothed composite > **0.65** and regime ≥ entry floor.
- **Exit** on any of: composite < **0.40** (factor decay, checked at rebalance), regime collapse,
  **15% trailing stop**, or **max hold 336h (14d)**.
- **Sizing:** inverse-vol within holdings, scaled by regime; **≤ 25 positions**, **≤ 10% per name**;
  rebalance every **24h**.

### 2.4 Rulebook lineage (K2 TRADE ATLAS)
QF-01 cross-sectional momentum · QF-07 vol-targeting (overlay) · QF-10 / CV-17 trend-convexity
framing · QF-21 data-snooping discipline · MR-09 crypto regime. Routes to the **convexity pipeline**
(gates on convexity, with Sortino reported).

---

## 3. Data & universe

- **Source.** Coinbase USD spot pairs, `coinbase_crypto_ohlcv_lake.duckdb` (`bars_1h` for the
  flagship pipeline; `bars_1d_usd_universe_clean` + a top-50 ADV membership table for reconciliation).
- **Period.** 2021-01-01 → 2026-06-01. **Out-of-sample = 2023-01-01 onward** (1,248 trading days).
- **Universe.** Point-in-time **top-100 by 20-day trailing dollar-ADV**, survivorship-free
  (membership known only as-of each date). Factors are ranked **within** the eligible set per date.
- **Costs.** 30 bps **round trip** headline (the backtest engine charges `tc_bps` as a round-trip
  cost ≈ 15 bps/side; see §7.1). Sensitivity and tiering in §7.

---

## 4. Methodology & research integrity

Every figure in this paper is governed by four disciplines, adopted after an earlier version of the
strategy was found to be inflated by look-ahead and cost-optimism:

1. **Survivorship-free, point-in-time universe** — membership reconstructed as known on each
   historical date from trailing liquidity; never with hindsight about which assets survived.
2. **Walk-forward parameter selection** — parameters chosen on each fold's *train* window, frozen,
   scored on the subsequent *test* window. The **headline uses fully frozen flagship parameters**
   (nothing fit on the data); the walk-forward number is reported only as an upper bound, because
   parameter selection is itself a source of optimism.
3. **Costs always on** — 30 bps round trip in every headline number; full tiered-cost + market-
   impact stress in §7.
4. **Pre-register, then run** — hypotheses and acceptance gates fixed in a committed protocol before
   fitting (`medallion_validation_protocol.md`); the run is deterministic and emits a provenance
   manifest (Appendix D).

A **transparent baseline must be beaten out-of-sample** before any complex model is trusted; where a
model could not, we investigated features / target / validation / costs rather than shipping it
(see §6).

---

## 5. The honesty arc — how the number evolved with rigor

Nothing about the strategy changed across these steps; each removed an optimistic assumption.

| Stage | OOS Sortino | Assumption removed |
|---|--:|---|
| As-shipped flagship | 2.70 | — (universe was top-50 by *full-period* ADV = look-ahead survivorship) |
| Point-in-time universe + walk-forward (top-50) | 1.97–2.03 | survivorship + parameter look-ahead |
| Widen to top-100, walk-forward | 2.95 | (added breadth — genuine) |
| **Frozen params, pre-registered audit (top-100)** | **2.84** | parameter-selection optimism → **headline** |
| Realistic liquidity-tiered costs | 1.42 | uniform-cost assumption (the binding constraint, §7) |

Roughly **0.7 of the original apparent edge was look-ahead bias**; a further large fraction is
contingent on the cost assumption. We report 2.84 as the defensible, cost-optimistic headline and
treat §7 as the gating analysis.

---

## 6. Empirical results

### 6.1 Headline (frozen params, OOS 2023–2026, 30 bps round trip)

| Metric | Medallion Lite | BTC buy & hold |
|---|--:|--:|
| Sortino | **2.84** | 1.78 |
| Sortino (walk-forward upper bound) | 2.95 | — |
| Sharpe | 2.15 | 1.15 |
| Calmar | 4.73 | 1.09 |
| CAGR | 173% | 54% |
| Max drawdown | −37% | −50% |
| Daily hit rate | 71% | 50% |

**Per-fold OOS Sortino (frozen):** 4.72 (2023) · 2.72 (2024) · 1.67 (2025-26). All positive; the
edge softens over time but does not vanish — healthier than the top-50 predecessor (3.49 → 1.11).

### 6.2 By universe (within-universe rank, walk-forward, 30 bps)
Breadth is a real lever to an optimum of ~70–100 names; over-widening fades.

| Universe | ~names | Frozen OOS Sortino | WF OOS Sortino | MaxDD |
|---|--:|--:|--:|--:|
| top-25 | 25 | 2.09 | — | −35% |
| top-50 | 49 | 2.60 | 2.29 | −38% |
| **top-100 (adopted)** | 93 | **2.84** | **2.95** | −37% |
| top-200 | 161 | 2.47 | — | −43% |
| top-250 | 177 | 2.46 | 2.28 | −41% |
| ADV ≥ \$1M | 72 | 2.76 | 2.46 | −32% |
| all USD | 193 | 2.48 | — | −41% |

The committed top-50 membership table reproduces the prior **1.97** baseline under within-universe
ranking — a reconciliation that confirms the harness (an intermediate rank-over-all-then-mask
version had spuriously read 2.65).

### 6.3 By feature set — what we rejected
| Feature set | OOS Sortino | Verdict |
|---|--:|---|
| **5-factor composite (adopted)** | **2.84** / 2.95 WF | baseline |
| 100-factor TA-Lib zoo (naïve average) | 5.95 *(artifact)* | **rejected** — concentration artifact: averaging ~100 collinear ranks compresses dispersion, the book holds ~12 vs ~22 names and piles into a lucky few |
| 100-factor, equal-selectivity (re-ranked) | 2.85 | ≈ baseline; no real lift |
| Bagged ensemble — mean aggregation | ~2.88 | converges to the composite; adds nothing |
| Bagged ensemble — vote (consensus) | 3.04 *(pre-validation)* | **rejected** — failed PBO (0.70–0.77); under frozen params all designs cluster 2.66–2.91, so the apparent gain was parameter selection, not the aggregation |

**Conclusion: breadth (~+1.0 Sortino) dominates factor count (~+0.3 at best, non-robust).** We run
the simple five-factor model. This is a textbook multiple-testing result and the reason §4's
"beat the transparent baseline OOS" rule exists.

---

## 7. Cost analysis & capacity — the binding constraint

### 7.1 Cost convention (important)
The backtest charges `cost = (Σ|Δwᵢ|/2) · tc_bps` per bar. Working a round trip of a position shows
this applies one-way cost = `tc_bps/2` to each leg, so **`tc_bps` is the round-trip cost**. The 2.84
headline therefore assumes ~**30 bps round trip (≈15 bps/side)** — not 30 one-way.

### 7.2 Flat round-trip sensitivity (top-100)
| Round-trip cost (bps) | 0 | 10 | 20 | **30** | 50 | ~70 |
|---|--:|--:|--:|--:|--:|--:|
| OOS Sortino | 3.50 | 3.27 | 3.05 | **2.84** | 2.42 | ≈ 2.0 (breakeven) |

### 7.3 Liquidity-tiered costs + market impact
Costs assigned per name by point-in-time ADV tier (T1 ≥\$50M … T4 <\$5M), applied per-trade in the
engine's convention; a square-root impact term `c·√(participation)` (c = 100 bps) adds capacity.

| Scenario (round-trip bps, T1/T2/T3/T4) | OOS Sortino |
|---|--:|
| S1 benign (10/20/40/70) | 2.26 |
| **S2 realistic (20/40/70/120)** | **1.42** |
| S3 punitive (35/70/130/220) | −0.13 |
| S2 + impact @ \$5M AUM | 0.99 |
| S2 + impact @ \$25M AUM | 0.67 |

### 7.4 Cost-robust universe search
Can a liquid-only universe dodge the hurdle? No: under S2, top-25 = 1.29, top-50 = 1.45, top-100 =
1.42, **ADV≥\$50M = 1.72 (best)**, ADV≥\$20M = 1.33 — all below the 2.0 gate and below BTC.
**Breadth buys flat-cost alpha in expensive names; liquidity buys cost-robustness but too few
names** — neither end clears the bar.

### 7.5 The crux
The alpha concentrates in small-cap names, so the binding input is *their* round-trip cost. **If
small-caps execute near liquid-name levels (≲ 30–40 bps RT), the 2.84 stands; if they run 100 bps+,
the realistic case (~1.42) governs.** PASS/FAIL is decided by the execution-cost number, not by more
backtesting.

---

## 8. Validation chain

| Control | Method | Result |
|---|---|---|
| Survivorship | point-in-time membership; reconcile to committed table | membership replays 1.97 baseline ✓ |
| Look-ahead | features end at signal date; execution lag (decide@close, fill@next-open) | enforced by construction ✓ |
| Parameter overfit | param-frozen walk-forward + **frozen-param headline** | headline fits nothing on data ✓ |
| Significance | Probabilistic Sharpe Ratio vs 0 (Bailey & López de Prado), `src/afml/backtest_stats` | PSR ≈ **1.00** |
| Design-space overfit | PBO via CSCV over the feature/ensemble design space | PBO **0.70–0.77** ⇒ design selection overfits ⇒ use simplest model |
| Pre-registration | gates G1–G4 fixed in a committed protocol before the run | **all pass**, deterministic, manifest emitted |

**PSR / Deflated Sharpe.** PSR(0) = Φ[ SR̂·√(n−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) ], with skew γ₃ and
kurtosis γ₄; deflation against the expected maximum Sharpe of N trials guards selection bias.
**PBO / CSCV.** The daily-return matrix is split into S blocks; over all C(S, S/2) train/test
partitions we measure how often the in-sample-best configuration lands below the out-of-sample
median (PBO). A high PBO means "which config looks best in-sample doesn't generalize."

---

## 9. Risk factors & failure modes

- **Cost fragility (primary).** As specified, the strategy does not survive realistic small-cap
  costs (§7). This is the gating risk.
- **Capacity.** Square-root impact implies soft capacity < \$5M AUM in the realistic-cost case.
- **Edge decay.** Per-fold Sortino declines 4.72 → 2.72 → 1.67; crypto factor premia are not
  guaranteed to persist.
- **Directional exposure.** Despite the regime gate and stops, the book is net long crypto and will
  participate in broad drawdowns (with a smaller maximum loss than passive holding).
- **Universe reconstruction.** The top-100 universe is recomputed from trailing ADV; production
  should be driven by a committed point-in-time membership table.
- **Mapped controls:** false breakout → volume + path-quality filters; trend exhaustion → trend-
  extension features; stop-in-noise → ATR-relative stop viability; gap risk → gap-distribution
  features; crowding/correlated stop-outs → portfolio heat + correlation clustering.

---

## 10. Current status & deployment gate

**Status: ON HOLD pending one input — realized small-cap round-trip execution cost (TBD by separate
execution research).** Registry `stage: S3`, `status: queued`.

Deployment decision rule (pre-stated): obtain the achievable round-trip cost on sub-\$20M-ADV crypto
from execution research; if small-cap RT ≲ 30–40 bps → promote to **paper trading**; if 100 bps+ →
**shelve**. No further backtesting changes this — it is an execution-capability question.

---

## 11. Open questions & next steps

1. **Execution cost on small-caps** — the gating input (above).
2. **Per-fold decay** — is the 2023→2025 softening regime-driven or structural decay? Decompose.
3. **Committed PIT membership table** — replace on-the-fly ADV reconstruction for production.
4. **Trailing give-back** — explore partial profit-taking / trail-multiplier tuning (a related
   spot-convexity sleeve quantifies that ~45% of trades touch +1R but far fewer close there).
5. **Lower-cost venues / asset classes** — the same factor logic on liquid futures, where costs are
   a fraction of small-cap crypto, may convert the signal into a deployable strategy.

---

## Appendix A — Metric definitions
Sortino = annualized mean / annualized downside deviation (MAR = 0); Calmar = CAGR / |max drawdown|;
hit rate = fraction of positive daily returns; PSR as in §8. All return series net of cost,
survivorship-free, OOS = 2023+.

## Appendix B — Acceptance gates (pre-registered)
**G1** headline OOS Sortino > BTC OOS Sortino · **G2** PSR(0) ≥ 0.95 · **G3** ≥ 2 of 3 OOS folds
positive · **G4** cost-robust (OOS Sortino > 1.5 @30 bps and > 1.0 @50 bps, *flat*). All pass at the
30 bps-flat headline; the tiered-cost gate (§7) is the separate, currently-failing hurdle.

## Appendix C — Honest-arc provenance
2.70 (survivorship) → 1.97/2.03 (PIT+WF, top-50) → 2.95 WF / **2.84 frozen** (top-100) → 1.42
(realistic tiered cost). Each transition is a committed experiment (Appendix D).

## Appendix D — Reproducibility & artifact index
Deterministic; one-command reproduce in each manifest. Universe = point-in-time top-100, 30 bps RT,
OOS 2023+.

| Topic | Doc | Harness | Manifest |
|---|---|---|---|
| Capstone summary | `medallion_lite_research_summary.md` | — | — |
| Strategy card | `medallion_lite_strategy_card.md` | — | — |
| Universe sweep | `medallion_universe_sweep.md` | `run_medallion_universe.py` | — |
| Factor count | `medallion_factor_count_experiment.md` | `run_medallion_factors.py` | — |
| Bagging | `medallion_bagging_experiment.md` | `run_medallion_bagging.py` | — |
| DSR/PBO + audit | `medallion_validation_protocol.md` | `run_medallion_validation.py`, `run_medallion_audit.py` | `medallion_audit_<commit>.json` |
| Cost sensitivity | `medallion_validation_protocol.md` (Amd A/B) | `run_medallion_costs.py`, `run_medallion_cost_universe.py` | `medallion_cost_sensitivity.json`, `medallion_cost_universe.json` |
| Research journal | `RESEARCH_LOG.md` | — | — |

## Disclaimer
For internal research use. Performance figures are **hypothetical and based on backtested, simulated
results**, which have inherent limitations (constructed with hindsight; do not represent actual
trading; may not reflect real-world execution). Past or simulated performance is not indicative of
future results. Digital-asset trading involves a high degree of risk, including total loss.
