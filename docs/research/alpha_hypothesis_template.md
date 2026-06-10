# Alpha Hypothesis Registration Form

**Intended path:** `docs/research/alpha_hypothesis_template.md`
**Companion:** `docs/research/convexity_alpha_pipeline_spec.md`

> **How to use:** Copy this template into a new file under `docs/research/hypotheses/<YYYY>-<NN>-<short-slug>.md`. Fill in every required field *before running any backtest*. Submit the filled-out form to the Alpha Registry; the registry entry's `stage` becomes `S0_complete` once accepted.
>
> **Pre-registration is mandatory.** If you find yourself rewriting the hypothesis after seeing backtest results, the alpha returns to Stage 0 — that's curve-fitting territory and the audit log will reflect it.

---

## Header — identity & routing

| Field | Value |
|---|---|
| **Alpha name** | _short, distinctive, no version numbers_ |
| **Registry ID** | _to be assigned by registry_ |
| **Researcher** | _name + email_ |
| **Registration date** | _YYYY-MM-DD_ |
| **Source / reference** | _TASC V42:04 Ehlers / SSRN paper / internal note_ |
| **Reference link or DOI** | _URL_ |
| **Prior related alphas in pool** | _Registry IDs of similar live alphas, if any_ |

## 1. Hypothesis statement

> Write the hypothesis as a single, falsifiable sentence. The sentence must contain: (a) a signal, (b) a predicted outcome, (c) a horizon.

**Example (good):**
> When @CL closes above a 60-min SuperTrend(ATR=13, mult=3) line after at least 5 bars of declining range, @CL produces a positively-skewed distribution of 1-5 day forward returns.

**Example (bad — too vague):**
> SuperTrend works on crude oil.

**Yours:**
> _____

## 2. Economic / behavioral rationale

> Why should this work? What economic, behavioral, or microstructure mechanism produces the predicted payoff? Be specific.
>
> Acceptable mechanisms: liquidity provisioning, sticky positioning, calendar/flow effects, structural carry, regime persistence, attention-driven herd behavior, dealer-flow imbalances, narrative arc, momentum/trend persistence, vol risk premium asymmetry.
>
> *Not* acceptable: "the indicator works" / "the chart looks good" / "the paper showed it works".

**Yours:**
> _____

## 3. Expected payoff shape

> *This determines pipeline routing.*

- [ ] **Convex** (positive skew expected; large rare wins, small frequent losses)
- [ ] **Linear** (symmetric payoff; standard equity alpha)
- [ ] **Concave** (negative skew; option selling, mean reversion)
- [ ] **Ambiguous** (research lead decides)

**Convexity track (if convex):**

- [ ] **Trend-with-stops**
- [ ] **Vol-expansion**
- [ ] **Both** (rare; defend below)

**Justification — why this payoff shape:**
> _____

**If convex, the alpha must include:**
- [ ] Stops (bounded downside per trade), OR
- [ ] Breakout / compression-expansion logic, OR
- [ ] Long-vol vehicle (options, vol futures, vol ETP)

## 4. Signal definition (math)

> Write the exact signal formula. Use ASCII or LaTeX. If parameters, list them all with defaults.

```
# Pseudocode or formula
signal_at(t) = ...
```

**Parameters and defaults:**

| Param | Default | Range tested |
|---|---|---|
| lookback | 60 | 40-100 |
| threshold | 1.5 | 1.0-2.5 |

**Lookback windows used (audit):**
- Longest backward reference: _____
- Any forward references (label leakage check): _____

## 5. Entry / exit rules

| Rule | Spec |
|---|---|
| **Entry trigger** | _exact condition_ |
| **Entry timing** | _on close / next open / intra-bar with stop_ |
| **Position size at entry** | _formula (vol target, fixed-fractional, etc.)_ |
| **Stop loss** | _yes / no; if yes, formula_ |
| **Re-entry rule** | _yes / no; trigger condition_ |
| **Profit target** | _yes / no; formula_ |
| **Trailing exit** | _yes / no; formula_ |
| **Time exit** | _yes / no; bar count_ |
| **Hard exit** | _e.g., session close, end of week_ |

## 6. Universe

| Field | Value |
|---|---|
| **Primary universe** | _Trend-with-stops standard / Vol-expansion standard / custom_ |
| **Custom universe (if applicable)** | _list of tickers / contracts_ |
| **Universe size** | _N_ |
| **Bar frequency** | _1-min / 5-min / 60-min / daily / weekly_ |
| **History window** | _earliest available; minimum 5 years_ |

**Justify** any deviation from the standard universe (per spec Appendix B).

## 7. Cost assumptions

> Use Appendix A defaults unless this alpha materially differs.

| Cost layer | Stage 1 | Stage 2 |
|---|---|---|
| Commission | _per-asset_ | _same_ |
| Spread | _0.5 × spread_ | _0.5 × spread_ |
| Market impact | _N/A at Stage 1_ | _k · sqrt(participation)_ |
| Slippage extra | _none_ | _0.25 × spread_ |
| Roll cost | _N/A_ | _per curve_ |
| Borrow (if short) | _N/A at Stage 1_ | _per asset_ |

## 8. Pre-registered expected metrics

> *Required.* These are the metric ranges you expect to see at Stage 1. If actual metrics fall *outside* your pre-registered ranges, this is meaningful data — either the hypothesis was wrong, the implementation has a bug, or the universe is mismatched.

| Metric | Expected range | Confidence |
|---|---|---|
| Aggregate CCS | _0.8 - 1.5_ | _Low/Med/High_ |
| Skew (aggregate) | _+0.3 - +1.5_ | _Med_ |
| Payoff ratio | _>1.8_ | _Med_ |
| Hit rate | _35-55%_ | _Med_ |
| Tail capture (trend track) | _0.35 - 0.55_ | _Low_ |
| Convexity beta b (vol track) | _0.5 - 1.5_ | _Med_ |
| Avg trade duration | _matches hypothesis horizon_ | _High_ |
| Sharpe (reported, not gating) | _0.4 - 0.9_ | _Med_ |

**Track this honestly.** Over-confident pre-registration → too many alphas accidentally pass on luck. Under-confident pre-registration → too many real alphas are killed by your own conservatism.

## 9. Falsification

> What would convince you this alpha is *not real*?

- [ ] **Skew check fails:** aggregate skew below the lower bound of your pre-registered range.
- [ ] **Convexity beta non-positive** at Stage 1 (vol track only).
- [ ] **Tail capture < 0.20** at Stage 1 (trend track only).
- [ ] **CCS drops > 60% from Stage 1 to Stage 2** (costs).
- [ ] **OOS CCS < 0.3 × IS CCS** at Stage 3.
- [ ] **2+ regime sub-periods negative** at Stage 4.
- [ ] **Other** (specify): _____

## 10. Risk / blow-up scenarios

> What's the worst-case path-dependent loss? Describe a scenario where this alpha catastrophically underperforms or blows up. Identify counter-measures (additional gates, hard caps, hedges).

**Scenario:**
> _e.g., "Strategy is long-vol via VXX puts; market enters a year-long quiet regime with VIX < 15. Strategy bleeds theta steadily, finishes -25% on the year."_

**Counter-measure:**
> _e.g., "Cap quarterly bleed at 8% of NAV; reduce allocation 50% after 2 quarters under cap; halt if 3 consecutive quarters under cap."_

## 11. Implementation notes

> Anything operational that matters: special data needs, exchange access, vendor dependencies, options chain requirements, etc.

**Yours:**
> _____

## 12. Stage routing

| Stage | Status | Date | Notes |
|---|---|---|---|
| S0 - Registered | _filled-in / submitted_ | YYYY-MM-DD | |
| S1 - Fast screen | _pending_ | | |
| S2 - Realistic costs | | | |
| S3 - Walk-forward OOS | | | |
| S4 - Robustness | | | |
| S5 - Pre-production | | | |
| S6 - Live shadow | | | |
| Live | | | |
| Retired | | | |

## 13. Sign-off

- [ ] Researcher: _name + date_
- [ ] Research lead (required at S3 promotion): _name + date_
- [ ] Ops lead (required at S5 promotion): _name + date_
- [ ] Risk lead (required at S6 promotion): _name + date_

---

## Examples (delete before submitting)

### Example: Trend-with-stops hypothesis

**Hypothesis:** When @ES daily Continuation Index (Ehlers Sep 2025, gamma=0.8, order=8) is +1 AND last 3 bars show monotonically shrinking ranges (Unger 3DC pattern), forward 5-20 day @ES returns are positively skewed with median > 0.3%.

**Mechanism:** Trend persistence post-compression — volatility compression resolves directionally with high probability; CI confirms trend state is supportive.

**Expected payoff shape:** Convex / Trend-with-stops.

**Stops:** Yes — ATR(20) × 2 trailing stop after entry. Reentry on new HH.

### Example: Vol-expansion hypothesis

**Hypothesis:** When VIX/VIX3M ratio drops below 0.85 for ≥ 5 consecutive trading days AND RealizedVol(SPX, 20d) is below 25th percentile of trailing 1y, long VIX March futures held for 20-40 days produces positive expected return with convexity beta > 0.7.

**Mechanism:** Vol-risk-premium imbalance: complacency persists, dealers under-hedged, eventual mean reversion in vol surface produces tail expansion. Long-dated futures less affected by daily roll bleed than VXX.

**Expected payoff shape:** Convex / Vol-expansion.

**Risk:** Vol stays compressed for the full hold; lose theta + roll. Mitigation: cap position at 1% NAV, allow 2 sequential losses before sleeve halt.
