# Alpha Hypothesis Registration — Trading The Channel (Kaufman LinReg)

**Companion:** `docs/research/convexity_alpha_pipeline_spec.md` · **Cohort:** 2024-Q4-cohort-01
**Status:** S0 registered (pre-registered before backtest).

## Header — identity & routing

| Field | Value |
|---|---|
| **Alpha name** | Trading The Channel (Kaufman LinReg) |
| **Registry ID** | 16 |
| **Researcher** | convexity-pipeline (cohort-01) |
| **Registration date** | 2024-12-15 |
| **Source / reference** | TASC V43:05 May 2025 — Kaufman, "Trading The Channel" |
| **Prior related alphas in pool** | ID 8 (MA Black Swan, overlapping equity-ETF universe — expect high correlation) |

## 1. Hypothesis statement
When a daily ETF close breaks above a 40-bar linear-regression channel (slope > 0 and close > +2σ
band), a long held until the close falls below the lower band captures selective trend bursts with
positive skew.

## 2. Economic / behavioral rationale
An up-sloping regression channel confirms an established trend; a breakout above +2σ signals
acceleration/continuation. Low time-in-market is meant to concentrate exposure into convex bursts
and avoid the choppy middle of the channel.

## 3. Expected payoff shape
**Convex** — Trend-with-stops (lower-band trailing exit). *Pre-registered note:* this is the
weakest convexity claim in the cohort — daily equity-ETF long/flat may inherit equities' negative
daily skew. Falsification criteria below are explicit about this risk.

## 4. Signal definition
```
lr = LinRegChannel(close, length=40, width=2.0)   # center, slope, ±width*sigma bands
entry = (lr.slope > 0) and (close > lr.upper)
exit  = close < lr.lower
position = stateful long/flat
```
| Param | Default | Range to test |
|---|---|---|
| length | 40 | 32–48 |
| width | 2.0 | 1.6–2.4 |

**Lookback audit:** longest backward reference = rolling 40-bar regression fit. No forward references.

## 5. Entry / exit rules
| Rule | Spec |
|---|---|
| Entry trigger | up-slope AND close > +2σ band |
| Entry timing | close t, fill next open (lag 1) |
| Trailing exit | close < −2σ band |
| Re-entry | yes, on next qualifying breakout |

## 6. Universe
12 broad/sector ETFs (SPY, QQQ, IWM, XLE, XLK, XLF, XLV, XLI, XLY, XLP, XLU, XLB), daily, 2005–2026.

## 7. Cost assumptions
Stage 1: 5 bps. Stage 2+: 4 bps (ETF). Cost-2x: 8 bps.

## 8. Pre-registered expected metrics
| Metric | Expected range | Confidence |
|---|---|---|
| Aggregate CCS | 0.4 – 1.3 | Low |
| Skew (aggregate) | +0.2 – +1.2 | **Low** |
| Tail capture | 0.20 – 0.45 | Low |
| Calmar | 0.4 – 1.3 | Low |

## 9. Falsification
- **Aggregate skew < +0.2** (explicitly expected to be the binding risk for ETF long/flat).
- **Convexity beta b ≤ 0** at Stage 1.
- Tail capture < 0.20 at Stage 1.

## 10. Risk / blow-up scenarios
**Scenario:** buying +2σ breakouts in equities systematically buys local tops; equity down-moves are
larger/faster than up-moves, so a long-only ETF strategy has negative per-bar skew and negative
convexity beta. **Counter-measure:** if falsified, route the idea to `src/alpha_pipeline/` (linear)
rather than forcing it through the convexity pipeline.

## 12. Stage routing
| Stage | Status | Date |
|---|---|---|
| S0 — Registered | submitted | 2024-12-15 |
| S1–S4 | pending | |

## 13. Sign-off
- [x] Researcher: convexity-pipeline (cohort-01), 2024-12-15
