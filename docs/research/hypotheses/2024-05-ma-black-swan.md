# Alpha Hypothesis Registration — MA Black Swan Filter (P>MA200)

**Companion:** `docs/research/convexity_alpha_pipeline_spec.md` · **Cohort:** 2024-Q4-cohort-01
**Status:** S0 registered (pre-registered before backtest).

## Header — identity & routing

| Field | Value |
|---|---|
| **Alpha name** | MA Black Swan Filter (P>MA200) |
| **Registry ID** | 8 |
| **Researcher** | convexity-pipeline (cohort-01) |
| **Registration date** | 2024-12-15 |
| **Source / reference** | TASC V43:03 Mar 2025 — Metghalchi, "MA Trading Black Swan Filter" |
| **Prior related alphas in pool** | ID 16 (LinReg Channel, overlapping equity-ETF universe — expect high correlation) |

## 1. Hypothesis statement
Holding broad equity ETFs only while price > MA(200), flat otherwise, sidesteps the worst crisis
drawdowns (dot-com, GFC, COVID) and produces a positively-skewed, convex equity curve relative to
buy & hold.

## 2. Economic / behavioral rationale
Bear markets and crashes cluster below the 200-day moving average; exiting there truncates the left
tail while staying long in uptrends preserves the right tail. The convexity is meant to come from
crisis-avoidance, not stock selection.

## 3. Expected payoff shape
**Convex** — Trend-with-stops (the MA200 cross is the regime stop). *Pre-registered note:* like
ID 16, a long/flat equity strategy may keep negative per-bar skew even as it improves drawdowns;
this is the binding risk and is in the falsification criteria.

## 4. Signal definition
```
ma = SMA(close, 200)
position = 1.0 if close > ma else 0.0     # decided at close t, held from t+1
```
| Param | Default | Range to test |
|---|---|---|
| ma_len | 200 | 160–240 |

**Lookback audit:** longest backward reference = SMA(200). No forward references.

## 5. Entry / exit rules
| Rule | Spec |
|---|---|
| Entry trigger | close > MA(200) |
| Entry timing | close t, fill next open (lag 1) |
| Exit (stop) | close < MA(200) → flat |
| Re-entry | yes, on next close > MA(200) |

## 6. Universe
10 equity ETFs (SPY, QQQ, IWM, EFA, XLK, XLV, XLE, XLF, XLI, XLY), daily, 2005–2026.

## 7. Cost assumptions
Stage 1: 5 bps. Stage 2+: 4 bps (ETF). Cost-2x: 8 bps.

## 8. Pre-registered expected metrics
| Metric | Expected range | Confidence |
|---|---|---|
| Aggregate CCS | 0.5 – 1.5 | Low |
| Skew (aggregate) | +0.1 – +1.0 | **Low** |
| Tail capture | 0.25 – 0.50 | Low |
| Calmar | 0.5 – 1.5 | Med |
| Avg trade duration | ≈ 60 bars | **Low** (price oscillates across MA200 → many short crossings) |

## 9. Falsification
- **Aggregate skew < +0.1** (binding risk for equity long/flat).
- **Convexity beta b ≤ 0** at Stage 1.
- **Median trade duration far from horizon** (frequent MA re-crossings).

## 10. Risk / blow-up scenarios
**Scenario:** price chops around MA200 → frequent whipsaw re-crossings (median hold collapses) while
the rare long crisis-avoidance benefit shows up only in tail windows; per-bar skew stays negative.
**Counter-measure:** if the convexity screen rejects it, this is a *risk-overlay / linear* construct,
not a standalone convex alpha — route accordingly.

## 12. Stage routing
| Stage | Status | Date |
|---|---|---|
| S0 — Registered | submitted | 2024-12-15 |
| S1–S4 | pending | |

## 13. Sign-off
- [x] Researcher: convexity-pipeline (cohort-01), 2024-12-15
