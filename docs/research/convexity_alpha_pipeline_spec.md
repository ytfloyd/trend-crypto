# Convexity Alpha Pipeline Spec

**Intended path:** `docs/research/convexity_alpha_pipeline_spec.md`
**Status:** v1.0 (draft for review)
**Companion docs:** `docs/research/alpha_hypothesis_template.md`, `alpha_registry.xlsx`

---

## 1. Purpose & scope

This spec defines a research pipeline for **long-convexity alphas** — trading ideas where the payoff curve bends upward. In practice that means two related but distinct tracks:

- **Trend-with-stops track** — directional alphas with bounded downside and unbounded (or long-tailed) upside. Trend-following, breakout, momentum, stop-reentry systems.
- **Vol-expansion track** — alphas that take the long side of |move| or implied volatility. Range-compression breakouts, long-gamma options structures, long-vol futures, tail-hedge overlays.

Both tracks share metric design, robustness checks, and operational machinery. They differ in entry semantics, exit dynamics, and a small set of track-specific gates.

This pipeline **runs in parallel** to the existing `src/alpha_pipeline/` (cross-sectional equity / standard quant). The Alpha Registry routes candidates into the correct pipeline based on `convexity_track`. Neither pipeline replaces the other.

### Out of scope

- Cross-sectional, dollar-neutral, market-neutral equity alphas. Those stay in the existing pipeline.
- Pure mean-reversion strategies whose return distributions are negatively skewed by design.
- Pure carry strategies whose payoffs are linear or concave.

If a candidate idea has linear or concave expected payoff, it gets routed away from this pipeline at Stage 0.

## 2. Why a separate pipeline

The default quant-equity research process is Sharpe-first, IR-first, and cross-sectional by default. Each of those decisions actively destroys long-convexity alphas:

- **Sharpe-first** penalizes the lumpy, positively-skewed, low-hit-rate return profiles convex strategies *must* have. A trend system can be a real alpha with Sharpe 0.4, +1.5 skew, and Calmar 1.2 — Sharpe-first kills it.
- **Cross-sectional by default** strips out the directional / regime / vol-expansion bet the strategy is supposed to make. Many convex ideas die when forced into a market-neutral mold.
- **Single-fold IS/OOS** tests miss path-dependent risk. Trend systems hold winners for months; a single bad fold doesn't prove the alpha is broken.

We accept the existing pipeline for what it does well. We add this one for what it doesn't.

## 3. Track separation

```
                     +-----------------------+
                     |   Alpha Registry      |
                     |  (Hypothesis routed)  |
                     +----------+------------+
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
   +---------------------+              +---------------------+
   |  Existing equity    |              |  Convexity pipeline |
   |  alpha pipeline     |              |  (this spec)        |
   |  (Sharpe/IR-first)  |              |                     |
   +---------------------+              |  +---------------+  |
                                        |  | Trend-w/Stops |  |
                                        |  +---------------+  |
                                        |                     |
                                        |  +---------------+  |
                                        |  | Vol-Expansion |  |
                                        |  +---------------+  |
                                        +---------+-----------+
                                                  |
                                                  v
                                        +---------------------+
                                        |  Production combiner|
                                        |  (alpha pool        |
                                        |   weighting)        |
                                        +---------------------+
```

**Routing decision (Stage 0):** based on `expected_payoff_shape` registered in the hypothesis form.

- `convex` → convexity pipeline.
- `linear` → existing equity pipeline (or other risk-budgeted process).
- `concave` (option-selling, mean-reversion) → not in scope here. Separate path.
- `ambiguous` → research lead decides; default to convexity pipeline if the alpha contains stops, breakouts, vol triggers, or breakout-from-compression logic.

## 4. Convexity-first metrics

These replace Sharpe as the primary screen. Sharpe is *reported* throughout but doesn't gate.

### 4.1 Definitions

All metrics are computed over net returns (after costs) at the strategy level unless noted.

**Return skewness:** Standard third-moment skew of daily/bar returns.
> `skew = E[(r - mu)^3] / sigma^3`

**Calmar ratio:** Annualized return / |max drawdown|.
> `Calmar = R_ann / |MaxDD|`

**Profit factor:** Gross wins / gross losses.
> `PF = sum(positive trade PnL) / |sum(negative trade PnL)|`

**Payoff ratio:** Average win / average loss.
> `Payoff = mean(positive trade PnL) / |mean(negative trade PnL)|`

**Tail capture ratio:** Fraction of the underlying's top-decile absolute moves the alpha captured in the *intended direction*.
> For trend-with-stops on instrument X:
> ```
> top_moves = top decile of |X return| over backtest window
> capture = sum(alpha_pnl during top_moves with correct sign) /
>           sum(|X return| during top_moves)
> ```
> Tail capture ≥ 0.3 means the strategy is converting tail events to PnL. < 0.15 means it isn't — almost certainly *not* a convex alpha regardless of headline metrics.

**Convexity beta:** Slope of strategy PnL against absolute underlying move.
> Fit: `r_alpha = a + b·|r_X| + c·r_X + e`
> Reject H0 (b = 0) at p < 0.05.
> b > 0 = long-vol payoff (convex). b ≈ 0 = neutral. b < 0 = concave.

**Tail Sharpe asymmetry:** Sharpe of positive-return regime vs negative-return regime.
> `TSA = Sharpe(returns | r > 0) / |Sharpe(returns | r < 0)|`
> Convex alphas have TSA > 1.

**Pain ratio:** Average drawdown / average underwater duration. Lower is better — captures "how painful was it to hold this."

### 4.2 Composite Convexity Score (CCS)

For comparison across alphas and as the **primary screening metric** at each pipeline stage, define:

```
CCS = NetReturnAnn / max(|MaxDD|, MinDD_floor)        # Calmar baseline
    × skew_factor(skew)                                # Skew adjustment
    × payoff_factor(payoff_ratio)                      # Payoff adjustment
    × tail_capture                                     # Tail capture multiplier
```

Where:

- `MinDD_floor = 0.05` (5% min DD to prevent CCS exploding for very low-DD samples)
- `skew_factor(s) = 1 + 0.5 · clip(s, -1, 2)` → modest penalty if skew negative, boost if positive, capped to avoid runaway
- `payoff_factor(p) = sqrt(max(p, 0.5))` → penalize <1 modestly, reward >2 moderately
- `tail_capture` enters as a multiplicative weight in [0, 1]

Reference CCS ranges (illustrative, calibrate on first cohort):

- CCS < 0.5 → weak. Likely killed at Stage 1.
- 0.5 ≤ CCS < 1.0 → marginal. Survives Stage 1 only if convexity beta and tail capture are both strong.
- 1.0 ≤ CCS < 2.0 → solid. Standard production candidate.
- CCS ≥ 2.0 → exceptional. Spend extra time on robustness; check for curve-fitting.

**Per-instrument CCS distribution matters as much as the aggregate.** Report median and IQR across the universe; not just the mean.

### 4.3 OOS / IS ratio — using CCS not Sharpe

The kill criterion at Stage 3 uses **CCS ratio**, not Sharpe ratio:
> `CCS_OOS / CCS_IS ≥ 0.5` to pass.

Sharpe ratio of OOS/IS is reported alongside but is not the gate. This addresses the dev's modification: Sharpe is secondary, so don't gate on it.

## 5. The 8-stage pipeline

Each stage has: **inputs**, **outputs**, **time budget**, **kill criteria**, **success criteria**. Kill criteria use convexity-first metrics, applied at the *pooled / universe-aggregate* level except where noted.

### Stage 0 — Hypothesis registration (1 hr)

**Input:** Idea + reference (TASC article, paper, internal note).
**Output:** Filled-out hypothesis template (`docs/research/alpha_hypothesis_template.md`); Alpha Registry entry created with `stage = S0`, `convexity_track`, `expected_payoff_shape`.
**Time budget:** 1 hour.

**Kill criteria:**
- `expected_payoff_shape` is not convex *and* the alpha doesn't contain stops/breakouts/vol triggers.
- Hypothesis is incoherent (no economic rationale).
- Identical to an existing live alpha.

**Success:** Registry status updates to `S0_complete → S1_queued`.

### Stage 1 — Fast vectorized screen (1 day)

**Input:** Alpha specification + standard universe (Appendix B) + default cost model (Appendix A).
**Output:** Per-universe and pooled metrics report.
**Time budget:** 1 day including review.

**Compute, both aggregate-over-universe AND per-instrument:**
- CCS, Calmar, payoff ratio, profit factor, skew, tail capture, avg trade duration, hit rate, max consecutive losses, % of universe positive.
- Sharpe (reported, not gating).
- Convexity beta vs each instrument's absolute returns.

**Kill criteria (any single failure):**
- **Aggregate skew < 0** (pooled, not per-instrument — addresses dev modification).
- **Per-instrument: skew below −1.0 on more than 1/3 of universe** (catastrophic-skew check).
- **CCS_aggregate < 0.5.**
- **% of universe with positive CCS < 60%.**
- **Convexity beta b ≤ 0** at the aggregate level (the alpha is not convex by construction).
- **Trade duration grossly mismatched to hypothesis horizon** (e.g., a "weekly" hypothesis with median 1-bar holds, or a "swing" hypothesis with 200-bar holds). *Note: no fixed bar-count threshold — checked against the registered hypothesis horizon (dev modification).*
- **Max consecutive losses > 40.** Even convex strategies need a recoverable streak.

**Success:** All above pass on default settings. Promote to Stage 2.

### Stage 2 — Realistic backtest (3 days)

**Input:** Stage-1 spec + realistic cost model.
**Output:** Net metrics report; Stage 1 / Stage 2 delta diagnostic.

**Realistic cost layers (additive):**
- Bid-ask: 0.5 × instrument-specific spread (futures: ticks; equities: bps).
- Linear impact: `cost_bps ≈ k · sqrt(participation_rate)`, k calibrated per asset class (Appendix A).
- Commissions: per-asset.
- Short borrow: per-asset rate for any short days.
- Roll cost: futures-only, per-curve.

**Kill criteria:**
- **CCS drops > 50% from Stage 1.**
- **Net annual return < 1.5 × estimated annual cost.**
- **Per-instrument: > 30% of universe goes net-negative under realistic costs.**
- **Convexity beta b loses significance** (p > 0.10 after costs).

**Success:** Promote to Stage 3.

### Stage 3 — Walk-forward OOS (1 week)

**Input:** Stage-2 spec; full historical window for the universe.
**Output:** OOS metric panel by fold; per-fold per-instrument CCS.

**Method:**
- Anchored walk-forward. Train window starts at earliest available; expands; test window is 1 year, rolling.
- Or block-bootstrap for short samples (block size = expected trade duration × 5).
- Minimum 8 OOS folds for promotion.

**Kill criteria (revised per dev feedback):**
- **`CCS_OOS / CCS_IS < 0.5`** (composite ratio, NOT Sharpe ratio — dev modification).
- **Median OOS skew not positive** OR **aggregate OOS skew not positive** (replaces "any OOS fold has negative skew", which the dev correctly flagged as too strict).
- **Any OOS fold has catastrophic left-tail loss** (single-fold max DD > 2 × aggregate MaxDD).
- **≥ 3 consecutive negative-CCS folds.**

**Success:** Promote to Stage 4.

### Stage 4 — Robustness battery (1 week)

**Input:** Stage-3 OOS-validated alpha.
**Output:** Robustness scorecard.

**Tests:**

1. **Parameter perturbation.** ±20% on every parameter independently; ±10% jointly on the two most sensitive. Metric stability requirement: aggregate CCS within 30% of base in 70%+ of perturbations.

2. **Regime decomposition.** Split test period into:
   - Bull (S&P trailing 1yr > +10%)
   - Bear (S&P trailing 1yr < -10%)
   - Sideways
   - High-vol (VIX > 75th percentile)
   - Low-vol (VIX < 25th percentile)

   Require: positive expectancy in **≥ 3 of 5** regimes; aggregate-positive in high-vol regime (convexity earns its keep there).

3. **Universe substitution.** Drop the single most-profitable instrument from the universe. Aggregate CCS must remain > 60% of original.

4. **Cost sensitivity.** Run at 2× assumed costs. Net return must remain positive.

5. **Look-ahead audit.** Code review of any variable that uses future information (forward-fill, cross-bar exits, end-of-day labels). Documented sign-off required.

6. **Curve-fit detection.** Check parameter-grid heatmap: if the strategy works *only* at a peak grid point (not on a plateau ≥3 cells wide), kill it.

**Kill criteria:**
- Any of the six checks fails materially (defined per check above).
- Curve-fit detected.

**Success:** Promote to Stage 5; write robustness scorecard to registry.

### Stage 5 — Pre-production (variable: see below)

**Input:** Stage-4 cleared alpha.
**Output:** Pre-production report; capacity model; documentation.

**Paper-trading window (revised per dev feedback):**
> `duration = max(min_calendar_window, min_signal_events_window)`
> where `min_calendar_window = 30 days` and `min_signal_events_window = 20 signals OR 10 round-trips`, whichever yields a longer window.

This handles both high-frequency (calendar bound) and low-frequency (signal-count bound) cases without using a fixed 2-4 week rule.

**Tasks:**
- Daily PnL tracking, expected vs realized.
- Capacity model: fit linear impact slope; report AUM at which IR halves.
- Operational documentation: signals/day, time-to-signal, execution mechanics, monitoring alerts.

**Kill criteria:**
- Live tracking error > 2σ of OOS expectation over the window.
- Operational red flags (execution lag > tolerance, signal volume off prediction).
- Capacity at target AUM ≤ 50% of planned allocation.

**Success:** Promote to Stage 6 (live shadow).

### Stage 6 — Live shadow (3-6 months)

**Input:** Stage-5 cleared alpha + small live allocation (0.25-0.5% NAV).
**Output:** Live performance vs expectation; promotion / demotion / kill decision.

**Tasks:**
- Track daily PnL.
- Compare live CCS to OOS CCS rolling window.
- Compare live skew, payoff, tail capture to OOS.

**Kill criteria:**
- 3-month live CCS < 0.3 × OOS expectation.
- Live skew turns clearly negative.
- Operational failure (exchange lockouts, signal failures > tolerance).

**Promotion criteria:**
- 6-month live CCS ≥ 0.7 × OOS expectation.
- Live skew at least non-negative.
- No operational failures.

**Success:** Move to full pool allocation under the combiner.

### Stage 7 — Monitoring & retirement (continuous)

**Cadence:**
- **Daily:** live PnL, drawdown vs expectation, skew rolling.
- **Weekly:** alpha-pool correlation matrix updated.
- **Quarterly:** re-test on most recent year; re-fit capacity model.
- **Annually:** full pipeline re-validation (Stages 3-4 redone with most recent data).

**Decay detection:**
- Trailing 6-month live CCS < 0.5 × initial OOS estimate → **reduce weight 50%**.
- Trailing 12-month live CCS < 0.3 × initial OOS estimate → **retire**.
- Alpha-pool correlation with another live alpha rises above 0.7 over 6 months → review for consolidation.

**Retirement is final.** No restoration without going through the pipeline again from Stage 3.

## 6. Track-specific overlays

The 8-stage skeleton runs both tracks. Each adds a small set of track-specific gates.

### 6.1 Trend-with-stops track

Additional Stage 1 checks:
- **Tail capture ≥ 0.30.** Fraction of underlying's top-decile moves the alpha captured in the correct direction (definition in 4.1).
- **Stop/no-stop comparison.** Run the alpha with and without stops; compute both. Stops should improve risk-adjusted metrics OR leave them unchanged. If stops *destroy* metrics, the stop is wrong (per Kaufman Apr 2025).
- **Reentry test.** Run with mandatory reentry on new HH/LL after stop-out; reentry must improve average trade. Per Kaufman's NVDA finding.

Additional Stage 4 robustness:
- Trend strategy must demonstrate **non-zero PnL in 2008, 2020, 2022** crisis sub-periods on the equity sleeve (these are where trend earns its keep).

### 6.2 Vol-expansion track

Additional Stage 1 checks:
- **Convexity beta b ≥ specific threshold.** Higher bar than the general 0: require b ≥ 0.5 (strategy genuinely long vol) and p < 0.05.
- **Compression detection precision ≥ 35%.** Of all signal fires, fraction that precede a vol expansion within signal horizon. Below this and the math doesn't work given typical payoff ratios.
- **Theta-bleed bound.** In quiescent periods (no vol expansion within horizon), maximum cumulative loss / NAV per quarter.

Additional Stage 2 cost layers:
- **Term-structure roll cost** for vol futures vehicles (VXX bleed). Calibrated empirically.
- **Options model** if vehicle is options: implied vol surface, gamma, theta, vega P&L decomposition. Track separately from net P&L.

Additional Stage 4 robustness:
- Vol-expansion strategy must demonstrate **positive expectancy during low-vol regimes** (when the signal fires expecting expansion). It's the inverse stress test for vol products.

## 7. Integration with existing systems

### 7.1 Alpha Registry routing

The registry is the single source of truth for which pipeline an alpha belongs to.

**New / updated fields:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `expected_payoff_shape` | enum {convex, linear, concave, ambiguous} | Y | Set at Stage 0. Convex = this pipeline. Linear = existing equity pipeline. |
| `convexity_track` | enum {trend, vol_expansion, both, N/A} | Y if convex | Set at Stage 0. |
| `stage` | enum {S0..S7, Live, Retired} | Y | Current pipeline stage. |
| `kill_reason` | text | If stage = Killed | Free text. |
| `tail_capture` | float [0, 1] | Y after Stage 1 | Per-instrument and aggregate stored. |
| `convexity_beta` | float | Y after Stage 1 | Aggregate b coefficient. |
| `convexity_beta_p` | float | Y after Stage 1 | p-value of b. |
| `ccs_is` | float | Y after Stage 1 | In-sample CCS. |
| `ccs_oos` | float | Y after Stage 3 | OOS aggregate CCS. |
| `promotion_decision` | text | At each promotion | Stage transitions, gates passed, sign-off. |
| `last_review_date` | date | Quarterly | Updated each review. |

Existing fields (`category`, `subtype`, `universe`, `priority`, etc.) continue to apply.

### 7.2 Routing rules at Stage 0

```
if expected_payoff_shape == "convex" AND convexity_track in {trend, vol_expansion, both}:
    route to convexity_pipeline
elif expected_payoff_shape == "linear" AND category == 1:
    route to existing equity_pipeline
elif expected_payoff_shape == "concave":
    flag for review; not in scope of this spec
else:
    research lead manual routing
```

### 7.3 Output to Combiner

Promoted alphas (Stage 6 → Live) join the production combiner pool with:
- Initial weight: 0.25-0.5% NAV.
- Combiner method: IR-weighted minus correlation penalty (placeholder; see Combiner spec — separate doc).
- Re-weighting cadence: monthly.

Cat-2 filters and Cat-3 risk overlays from the registry do not enter the combiner as alphas. They are applied as platform-level layers per the registry companion doc.

### 7.4 Sleeve definitions

For reporting and capital allocation, alphas in production are grouped into **sleeves**:

- **Trend Core** — high-corr-to-trend-beta directional alphas.
- **Trend Diversifiers** — low-corr-to-trend-beta directional alphas (session biases, day-of-week, pair rotations).
- **Vol Expansion** — long-vol / breakout alphas.
- **Tail Hedge** — Cat-4 explicit hedges (ViPar, progressive overlay).
- **Mean-Reversion Sleeve** — small allocation (per registry) of negatively-correlated alphas as a return diversifier.

Total platform sleeve weights and re-weighting rules: separate ops doc.

## 8. Operations

### 8.1 Throughput targets

Per researcher per year (steady-state):
- ~50 candidates registered (Stage 0).
- ~25 reach Stage 1.
- ~6 reach Stage 5.
- ~3-5 promoted to Live.

Steady-state Live pool size: 15-30 alphas after 3 years of pipeline operation.

### 8.2 Cadence

| Activity | Cadence | Owner |
|---|---|---|
| Stage 0 registrations | As discovered | Research |
| Stage 1-2 backtests | Continuous | Research |
| Stage 3-4 batch | Bi-weekly batch run | Research + ops |
| Stage 5 pre-prod | Continuous | Research + ops |
| Stage 6 live shadow | Continuous | Ops |
| Stage 7 monitoring | Daily / Weekly / Quarterly / Annual | Ops |
| Pool re-weighting | Monthly | Combiner ops |
| Universe maintenance | Quarterly | Data ops |
| Cost model recalibration | Quarterly | Execution ops |
| Pipeline spec review | Annually | Research lead |

### 8.3 Retirement policy

Final. Retired alphas cannot be re-added without going through Stage 3 onwards with fresh data.

Reason for finality: a retired alpha that "comes back" is almost always a regime change masquerading as alpha recovery. Force the full pipeline to confirm.

## 9. Implementation roadmap

### Phase 1 — Spec + infra (weeks 1-4)

- This spec doc + hypothesis template + registry field migration.
- `src/convexity_pipeline/` skeleton: stage runner, metric computation, kill-criterion evaluator.
- Standard universes defined and stored.
- Default cost model implemented and calibrated.

### Phase 2 — First cohort (weeks 5-12)

- Pick 10 candidates from existing Alpha Registry T1 picks.
- Run through Stages 0-4. Document any spec issues.
- Calibrate kill criteria from empirical observation (CCS thresholds, etc.).
- Iterate spec to v1.1.

### Phase 3 — Production (weeks 13+)

- First 2-3 alphas reach Stage 5-6.
- Combiner integration.
- Live monitoring dashboard.
- Quarterly review cadence kicks in.

### Phase 4 — Scaling (month 6+)

- Throughput target hit.
- Pool reaches 5+ live alphas.
- First retirement cycle.
- Universe expansion (additional asset classes).

## Appendix A — Default cost model

Stage 1 (fast):
- Equities/ETFs: 1 bp commission + 0.5 × bid-ask (typically 1-3 bps).
- Liquid futures: 0.5 ticks + per-contract commission.
- Crypto futures: 5 bps round-trip.
- FX: 0.5 pips.

Stage 2 (realistic):
- Add linear market impact: `impact_bps = k · sqrt(ADV_participation)` with k calibrated per asset class.
- Add short borrow per asset class.
- Add roll for futures.

Stage 2 calibration: targets to match observed execution on existing live systems within 20%. Recalibrate quarterly.

## Appendix B — Standard universes

**Trend-with-stops standard universe:**

- US broad ETFs: SPY, QQQ, IWM, EFA, EEM, TLT, GLD, USO, DBC, VNQ.
- Liquid futures: @ES, @NQ, @CL, @HO, @RB, @GC, @SI, @HG, @ZB, @ZN, @6E, @6J, @BTC, @KC, @NK.
- FX: EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD, AUD/USD.

**Vol-expansion standard universe:**

- VIX futures: VX, weekly + monthly.
- Vol ETPs: VXX, UVXY, SVXY, VIXY.
- Index options (for options-based vol): SPX, RUT, NDX (weekly + monthly).
- Single stock options: top 20 by volume in liquid names.

Maintained in `data/universes/` with quarterly refresh. Constituents tagged with effective date for historical universe reconstruction.

## Appendix C — Metric reference card

| Metric | Formula | Direction | Used as |
|---|---|---|---|
| CCS | `Calmar · skew_factor · payoff_factor · tail_capture` | higher better | primary screen (all stages) |
| Calmar | `R_ann / |MaxDD|` | higher better | CCS component |
| Skew | `E[(r-mu)^3]/sigma^3` | positive needed | CCS component, gate |
| Payoff ratio | `mean(win) / |mean(loss)|` | >1 needed | CCS component |
| Profit factor | `sum(win) / |sum(loss)|` | >1 needed | reported |
| Tail capture | see 4.1 | ≥0.30 (trend track) | gate (Stage 1) |
| Convexity beta b | `r_alpha ~ a + b·|r_X|` | >0, p<0.05 | gate (Stage 1) |
| Sharpe | `mu/sigma · sqrt(252)` | reported | not gating |
| MaxDD | trough-to-peak | smaller better | input to CCS |
| Pain ratio | `avg_DD / avg_under_water_duration` | smaller better | reported |
| TSA | Sharpe+/Sharpe- | >1 needed | check, not gate |

## Appendix D — Hypothesis template

See `docs/research/alpha_hypothesis_template.md` (companion doc).

## Appendix E — Reviewer sign-off matrix

| Stage transition | Required sign-off |
|---|---|
| S0 → S1 | Research lead (or self-cert for backlog) |
| S1 → S2 | Researcher (auto if all kill criteria pass) |
| S2 → S3 | Researcher (auto) |
| S3 → S4 | Research lead |
| S4 → S5 | Research lead + Ops lead |
| S5 → S6 | Research lead + Ops lead + Risk lead |
| S6 → Live | Research lead + Risk lead |
| Live → Retired | Research lead OR triggered by monitoring rules |

---

**Change log:**

- v1.0 (initial draft) — incorporates dev review feedback on skew gates, OOS/IS comparison, trade duration, paper trading windows.

**Open questions for v1.1:**

- Should we add a "fast cycle" track for high-frequency vol-expansion alphas where Stages 5-6 windows are calendar-bound rather than signal-count-bound?
- Should the combiner integration be specified here or in a separate Combiner Spec?
- Calibration of CCS thresholds against the first cohort — placeholder ranges above are illustrative, not measured.
