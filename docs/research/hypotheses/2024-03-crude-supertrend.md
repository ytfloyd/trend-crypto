# Alpha Hypothesis Registration — Crude SuperTrend System (Unger)

**Companion:** `docs/research/convexity_alpha_pipeline_spec.md` · **Cohort:** 2024-Q4-cohort-01
**Status:** S0 registered (pre-registered before backtest).

## Header — identity & routing

| Field | Value |
|---|---|
| **Alpha name** | Crude SuperTrend System (Unger) |
| **Registry ID** | 5 |
| **Researcher** | convexity-pipeline (cohort-01) |
| **Registration date** | 2024-12-15 |
| **Source / reference** | TASC V43:08 Aug 2025 — Unger, "Crude SuperTrend System" |
| **Prior related alphas in pool** | none (only intraday-futures candidate in cohort-01) |

## 1. Hypothesis statement
When 60-min crude oil (CL, back-adjusted continuous) trades above a SuperTrend(ATR=13, mult=3)
line, staying long until the line flips produces a convex, positively-skewed trend payoff with a
built-in volatility-scaled trailing stop.

## 2. Economic / behavioral rationale
Energy futures show persistent intraday trends driven by inventory/flow shocks and momentum in
positioning. SuperTrend's ATR-scaled trailing line keeps the position in persistent trends and
exits on volatility-scaled reversals, bounding downside per trade — a classic convex trend engine.

## 3. Expected payoff shape
**Convex** — Trend-with-stops (SuperTrend trailing line is the stop).

## 4. Signal definition
```
st = SuperTrend(high, low, close, atr_len=13, multiplier=3.0)
position = 1.0 if st.direction > 0 else 0.0     # decided at close t, held from t+1
```
| Param | Default | Range to test |
|---|---|---|
| atr_len | 13 | 10–16 |
| multiplier | 3.0 | 2.4–3.6 |

**Lookback audit:** longest backward reference = ATR(13) recursion. No forward references.

## 5. Entry / exit rules
| Rule | Spec |
|---|---|
| Entry trigger | SuperTrend direction = +1 |
| Entry timing | close t, fill next open (lag 1) |
| Trailing exit | SuperTrend direction flips to −1 |
| Re-entry | yes, on next +1 |

## 6. Universe
| Field | Value |
|---|---|
| Custom universe | CL only (back-adjusted institutional continuous) |
| Universe size | 1 |
| Bar frequency | 60-min |
| History window | ~11 months (2025-06 → 2026-05) — **below the 5-year minimum; flagged risk** |

**Deviation:** the plan's "@CL + trending futures" was reduced to CL-only because NG/SI front-month
artifacts are not back-adjusted (roll gaps inject spurious tails) and GC was unavailable (lake
locked). The short CL window is a known robustness risk, pre-registered here.

## 7. Cost assumptions
Stage 1: 5 bps. Stage 2+: 2 bps (futures). Cost-2x: 4 bps.

## 8. Pre-registered expected metrics
| Metric | Expected range | Confidence |
|---|---|---|
| Aggregate CCS | 0.4 – 1.2 | Low (short window) |
| Skew (aggregate) | +0.2 – +1.2 | Med |
| Tail capture | 0.25 – 0.50 | Low |
| Calmar | 0.3 – 1.2 | Low |
| Avg trade duration | ≈ 80 bars | Low |

## 9. Falsification
- Aggregate skew < +0.2 · Convexity beta b ≤ 0 · OOS aggregate skew ≤ 0 · fails Stage-4
  parameter-stability or regime breadth (single-instrument, short history → expected fragility).

## 10. Risk / blow-up scenarios
**Scenario:** a single-instrument, 11-month sample over-fits to one crude regime; a regime change
(e.g. range-bound oil) collapses the edge. **Counter-measure:** treat as data-limited; do not
promote past S4 until a multi-year, multi-contract back-adjusted futures series is available.

## 12. Stage routing
| Stage | Status | Date |
|---|---|---|
| S0 — Registered | submitted | 2024-12-15 |
| S1–S4 | pending | |

## 13. Sign-off
- [x] Researcher: convexity-pipeline (cohort-01), 2024-12-15
