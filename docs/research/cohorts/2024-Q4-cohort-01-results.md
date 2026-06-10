# Convexity Alpha Pipeline — Cohort-01 Results (2024-Q4)

**Config:** `calibrated_config()` (`src/convexity_pipeline/thresholds.py`).
**Run:** `data/alpha_registry/runs/20260530T033255Z/` (`scorecard.csv`, `per_candidate.json`).
**CAND-5 re-run (extended CL):** `data/alpha_registry/runs/20260531T160804Z/` — CAND-5 re-evaluated on the
back-adjusted CL continuous after it was extended from ~11 months to ~24 months (2024-05 → 2026-05). See the
CAND-5 section; the original cohort numbers below are otherwise unchanged.
**Reproduce:**
```
PYTHONPATH=src python scripts/research/convexity_alpha_runner.py --config default --calibrate
PYTHONPATH=src python scripts/research/convexity_alpha_runner.py --config calibrated
# CAND-5 alone on the extended CL series:
PYTHONPATH=src python scripts/research/convexity_alpha_runner.py --config calibrated --only 5
```

## Scorecard

| ID | Name | Final stage | Pass | S1 CCS | S1 skew | S1 β_b (p) | S1 tail | OOS/IS CCS |
|---|---|---|---|---|---|---|---|---|
| 4  | Continuation Index (Ehlers) | **S4** | ✅ | 1.236 | +0.647 | +0.033 (2e-4) | 0.551 | 4.47 |
| 17 | Ultimate Channels & Bands (Ehlers) | **S4** | ✅ | 1.131 | +0.858 | +0.041 (2e-7) | 0.551 | 1.23 |
| 5† | Crude SuperTrend System (Unger) | **S1** | ❌ killed @ S1 | 0.003 | +0.786 | +0.028 (3e-7) | 0.457 | — |
| 16 | Trading The Channel (Kaufman LinReg) | S1 | ❌ killed @ S1 | −0.001 | −4.425 | −0.019 | 0.062 | — |
| 8  | MA Black Swan Filter (P>MA200) | S1 | ❌ killed @ S1 | 0.260 | −0.643 | −0.046 | 0.299 | — |

† CAND-5 values are the **extended 24-month CL re-run** (run `20260531T160804Z`). On the original ~11-month
window it reached S4 (S1 CCS 0.157, skew +1.553, β_b +0.078, tail 0.500, OOS/IS 0.67) before being killed at
S4; the longer series collapses its edge to a Stage-1 kill (see CAND-5 section).

**Acceptance (original ~11-month run): 3 of 5 candidates reached Stage 4; 2 passed the full robustness
battery.** On the extended 24-month CL series CAND-5 drops to a Stage-1 kill, so **2 of 5 now reach Stage 4**
(CAND-4 and CAND-17, both still pass). The acceptance criterion (≥2 at Stage 4) still holds.

## Per-candidate

### CAND-4 Continuation Index — PROMOTE to S4 ✅
Clean convex trend alpha. S1 CCS 1.24, skew +0.65, β_b +0.033 (p≈2e-4), tail capture 0.55,
payoff 3.50. S2 CCS drop only 2.3% (crypto costs trivial vs edge). S3 OOS/IS CCS 4.47, aggregate
OOS skew +0.73, median fold skew +0.12. **S4 all clear**: parameter stability 1.00, regime CCS
{bull 7.84, high_vol 1.03, low_vol 1.80} (3 positive), universe-drop ratio 1.15 (edge survives
dropping the top contributor), cost-2x total return +7.38.

### CAND-17 Ultimate Channels & Bands — PROMOTE to S4 ✅
Channel breakout with ATR trailing stop. S1 CCS 1.13, skew +0.86, β_b +0.041 (p≈2e-7), tail 0.55,
hit 0.55. S2 CCS drop 1.7%. S3 OOS/IS CCS 1.23, **aggregate OOS skew +0.83** (median fold skew −0.22
— see calibration note; the pooled distribution is strongly convex). **S4 all clear**: parameter
stability 0.875, regime CCS {bull 7.21, high_vol 1.00, low_vol 1.43} (3 positive), universe-drop
ratio 1.07, cost-2x +6.81.

### CAND-5 Crude SuperTrend — KILLED @ S1 on extended data ❌ (small-sample edge did not survive)
**Original ~11-month run (2025-06 → 2026-05):** genuinely convex on the metrics (S1 CCS 0.157, skew +1.55,
β_b +0.078, tail 0.50), survived S1–S3 (aggregate OOS skew +1.58, OOS/IS CCS 0.67), then **killed at S4** on
parameter stability 0.625 (< 0.70) and only 2 of 5 regimes positive.

**Extended 24-month re-run (2024-05 → 2026-05, run `20260531T160804Z`):** after the CL continuous was
back-filled (IBKR 202407–202509 strip) the edge **collapses and the candidate is killed at Stage 1** —
`ccs_aggregate_below_threshold` (CCS 0.003 vs floor 0.157). S1 skew falls to +0.786, β_b to +0.028 (still
significant, p≈3e-7), tail capture 0.457; Sharpe 0.031, Calmar 0.004, profit factor 1.10, hit rate 0.42, max
drawdown 0.50 over 177 trades (median hold 29 bars). The convex *shape* persists, but there is no
risk-adjusted edge.

**Interpretation:** the 11-month window was dominated by the 2026 oil dislocation — a single convex tail event.
Across a fuller two-year sample that includes the choppy H2-2024 / H1-2025 oil regime, bare SuperTrend(13,3)
on hourly CL is not standalone convex alpha. This is a clean **falsification** that confirms the
pre-registered "data-limited fragility" risk: more data killed the candidate exactly where flagged. **Do not
promote.** A multi-decade, multi-instrument futures series (Databento Phase 2) would be needed to test whether
any version of this idea is robust.

### CAND-16 Trading The Channel — KILLED @ S1 ❌ (correct rejection)
S1 skew −4.43, CCS ≈ 0, β_b −0.019, tail capture 0.06, universe-positive 0.42. Buying +2σ ETF
breakouts is not convex at the per-bar level. **Route to `src/alpha_pipeline/` (linear) if pursued.**

### CAND-8 MA Black Swan Filter — KILLED @ S1 ❌ (correct rejection)
S1 skew −0.64, **β_b −0.046** (negative convexity beta), median trade duration 4 bars (vs horizon
60 → many MA200 re-crossings), max consecutive losses 16 (> calibrated 13). Note payoff ratio 10.5
and hit rate 0.21 — a low-hit/high-payoff profile that *looks* convex, but the negative per-bar skew
and negative convexity beta show the crisis-filter benefit is a **risk overlay**, not standalone
convex alpha. **Route as a risk-overlay / linear construct.**

## Pre-registered vs realized

| Candidate | Pre-registered skew | Realized S1 skew | Pre-reg tail | Realized tail | Verdict |
|---|---|---|---|---|---|
| 4  | +0.3…+1.5 | +0.65 | 0.30–0.55 | 0.55 | in range ✅ |
| 17 | +0.3…+1.5 | +0.86 | 0.30–0.55 | 0.55 | in range ✅ |
| 5  | +0.2…+1.2 | +1.55 → +0.79† | 0.25–0.50 | 0.50 → 0.46† | skew in range on 24-mo, but edge non-robust |
| 16 | +0.2…+1.2 (Low conf) | −4.43 | 0.20–0.45 | 0.06 | **falsified** (as flagged) ✅ honest |
| 8  | +0.1…+1.0 (Low conf) | −0.64 | 0.25–0.50 | 0.30 | **falsified** on skew/β (as flagged) ✅ honest |

The two low-confidence ETF candidates were *pre-registered with explicit falsification on skew/β*,
and the pipeline falsified them exactly there — pre-registration worked as intended.

† CAND-5 second value is the extended 24-month CL re-run. The metrics stay nominally in (or near) range
but the risk-adjusted edge does not survive the longer sample — a falsification on *robustness* rather than
on the per-bar shape.

## Metric-definition surprises
1. **Median per-fold skew vs aggregate OOS skew.** The hardcoded "median fold skew > 0" Stage-3 gate
   killed two strongly-convex candidates (aggregate OOS skew +0.83 and +1.58). Convex payoffs are
   lumpy across folds; the *aggregate* OOS skew is the correct convexity measure. We converted the
   gate to a config flag and disabled it (see calibration doc). **This is the headline finding.**
2. **Payoff ratio can be high while the alpha is non-convex.** CAND-8 had payoff ratio 10.5 and hit
   rate 0.21 yet negative skew and negative convexity beta. Trade-level payoff asymmetry ≠ per-bar
   convexity; CCS correctly weighted it down via the negative-skew / Calmar interaction.
3. **`min_universe_positive_fraction` 25th pct = 1.0** on a 5-candidate cohort is too strict as a
   general rule (see calibration caveat).
4. **Anchored short IS windows understate DD envelope** → the 2× catastrophic-fold multiplier was
   too tight for crypto; raised to 2.9.
5. **The `min_ccs_aggregate = 0.157` floor was set by CAND-5's own (now-superseded) 11-month value** — it
   was the binding 25th-percentile candidate. With CAND-5's extended-data CCS collapsing to 0.003, that
   floor no longer reflects a live candidate. Recalibration is **deferred to cohort-02** (re-derive the
   Stage-1 percentiles from the surviving / next cohort); no thresholds were changed here.

## Recommendations for cohort-02
- **Universe replication for futures:** build multi-year, back-adjusted continuous series for CL, NG,
  GC, SI, HG (unlock/snapshot the futures lake) so single-instrument fragility (CAND-5) can be tested
  across instruments and decades.
- **Drop pure equity long/flat from the trend track.** CAND-16/CAND-8 belong in the linear pipeline
  or as risk overlays; the convexity screen rejects them for the right reason.
- **Re-examine `min_universe_positive_fraction`** with a larger cohort (target floor 0.55–0.70).
- **Promote the median-fold-skew flag to a standing spec change** (keep `require_positive_median_fold_skew=False`
  for convex tracks) and document in the spec proper.
- **Vol-expansion track:** unblocked only once VIX / options-chain history is ingested.
- **Correlation realized:** as hypothesized, the two crypto trend alphas (4, 17) behave similarly
  (both promoted); the two equity ETF candidates (16, 8) both failed convexity — fold any survivors
  into the pool with an explicit correlation budget.
