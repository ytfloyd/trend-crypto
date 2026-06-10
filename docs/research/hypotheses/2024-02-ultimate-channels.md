# Alpha Hypothesis Registration — Ultimate Channels & Bands (Ehlers)

**Companion:** `docs/research/convexity_alpha_pipeline_spec.md` · **Cohort:** 2024-Q4-cohort-01
**Status:** S0 registered (pre-registered before backtest).

## Header — identity & routing

| Field | Value |
|---|---|
| **Alpha name** | Ultimate Channels & Bands (Ehlers) |
| **Registry ID** | 17 |
| **Researcher** | convexity-pipeline (cohort-01) |
| **Registration date** | 2024-12-15 |
| **Source / reference** | TASC V42:05 May 2024 — Ehlers, "Ultimate Channels & Bands" |
| **Prior related alphas in pool** | ID 4 (Continuation Index, same crypto universe — expect high correlation) |

## 1. Hypothesis statement
When a daily crypto close breaks above an Ehlers no-lag mid-line + 2·ATR upper band, a long held
until the close crosses the lower band (built-in trailing stop) produces a positively-skewed,
convex trend payoff: small frequent stop-outs and rare large trend wins.

## 2. Economic / behavioral rationale
A near-zero-lag channel enters established trends quickly; the ATR-scaled lower band acts as a
volatility-aware trailing stop that bounds downside per trade. The breakout-with-trailing-stop
structure is the canonical convex trend engine — it cuts losers fast and lets winners run.

## 3. Expected payoff shape
**Convex** — Trend-with-stops (explicit ATR trailing stop on the lower band).

## 4. Signal definition
```
mid    = NoLagEMA(close, length=20)          # 2*EMA - EMA(EMA)
band   = ATR(high, low, close, atr_len=20)
upper  = mid + mult*band ; lower = mid - mult*band   (mult=2.0)
entry  = close > upper
exit   = close < lower
position = stateful long/flat: 1.0 from first entry until first exit, then flat
```
| Param | Default | Range to test |
|---|---|---|
| length | 20 | 16–24 |
| mult | 2.0 | 1.6–2.4 |
| atr_len | 20 | 16–24 |

**Lookback audit:** longest backward reference = NoLagEMA(20)+ATR(20). No forward references.

## 5. Entry / exit rules
| Rule | Spec |
|---|---|
| Entry trigger | close > mid + 2·ATR |
| Entry timing | close t, fill next open (lag 1) |
| Position size | unit long, equal-weight |
| Trailing exit | close < mid − 2·ATR |
| Re-entry | yes, on next upper-band break |

## 6. Universe
8 liquid Coinbase crypto (BTC, ETH, LTC, BCH, LINK, ADA, SOL, AVAX), daily, 2015–2026.

## 7. Cost assumptions
Stage 1: 5 bps. Stage 2+: 12 bps (crypto). Cost-2x: 24 bps.

## 8. Pre-registered expected metrics
| Metric | Expected range | Confidence |
|---|---|---|
| Aggregate CCS | 0.7 – 1.5 | Med |
| Skew (aggregate) | +0.3 – +1.5 | Med |
| Tail capture | 0.30 – 0.55 | Med |
| Calmar | 0.4 – 1.3 | Med |
| Avg trade duration | ≈ 30 bars | Med |

## 9. Falsification
- Aggregate skew < +0.3 · Convexity beta b ≤ 0 · Tail capture < 0.20 · OOS aggregate skew ≤ 0.

## 10. Risk / blow-up scenarios
**Scenario:** choppy range where price repeatedly pokes above the upper band then reverses to the
lower band → death by a thousand stop-outs. **Counter-measure:** ATR-scaled bands widen in chop;
equal-weight spread; halve sleeve after sustained underperformance.

## 12. Stage routing
| Stage | Status | Date |
|---|---|---|
| S0 — Registered | submitted | 2024-12-15 |
| S1–S4 | pending | |

## 13. Sign-off
- [x] Researcher: convexity-pipeline (cohort-01), 2024-12-15
