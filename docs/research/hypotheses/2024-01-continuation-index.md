# Alpha Hypothesis Registration — Continuation Index (Ehlers)

**Companion:** `docs/research/convexity_alpha_pipeline_spec.md` · **Cohort:** 2024-Q4-cohort-01
**Status:** S0 registered (pre-registered before backtest).

## Header — identity & routing

| Field | Value |
|---|---|
| **Alpha name** | Continuation Index (Ehlers) |
| **Registry ID** | 4 |
| **Researcher** | convexity-pipeline (cohort-01) |
| **Registration date** | 2024-12-15 |
| **Source / reference** | TASC V43:09 Sep 2025 — Ehlers, "Continuation Index" |
| **Prior related alphas in pool** | ID 17 (Ultimate Channels, same crypto universe — expect high correlation) |

## 1. Hypothesis statement
When the Ehlers Continuation Index (Laguerre γ=0.8, slope-smoothing length=20) is +1 on daily
crypto bars, holding the asset long (flat when −1) produces a positively-skewed distribution of
forward returns whose right tail (large trend runs) dominates the truncated left tail.

## 2. Economic / behavioral rationale
Crypto exhibits strong trend/momentum persistence driven by attention-driven herd flows and
reflexive positioning. The Laguerre-smoothed slope stays positive through sustained uptrends, so a
binary always-long-in-uptrend state rides the fat right tail; going flat in downtrends truncates the
left tail. The convexity comes from asymmetric participation, not from a forecast of direction.

## 3. Expected payoff shape
**Convex** — Trend-with-stops. The "stop" is the regime flip to −1 (exit to flat), which bounds
downside participation per trend episode.

## 4. Signal definition
```
filt   = LaguerreFilter(close, gamma=0.8)
slope  = filt.diff()
smooth = EMA(slope, length=20)
state  = sign(smooth)   # carried through zero/NaN
position(t) = 1.0 if state(t) > 0 else 0.0     # decided at close t, held from t+1
```
| Param | Default | Range to test |
|---|---|---|
| gamma | 0.8 | 0.64–0.96 (±20%) |
| length | 20 | 16–24 (±20%) |

**Lookback audit:** longest backward reference = recursive Laguerre + EMA(20). No forward references.

## 5. Entry / exit rules
| Rule | Spec |
|---|---|
| Entry trigger | state flips to +1 |
| Entry timing | signal at close t, fill next open (execution_lag=1) |
| Position size | unit long (1.0), equal-weight across universe |
| Stop / trailing exit | state flips to −1 → flat |
| Re-entry | yes, on next +1 |

## 6. Universe
| Field | Value |
|---|---|
| Primary universe | 8 liquid Coinbase crypto: BTC, ETH, LTC, BCH, LINK, ADA, SOL, AVAX (all -USD) |
| Universe size | 8 |
| Bar frequency | daily |
| History window | 2015–2026 (BTC/ETH longest) |

## 7. Cost assumptions
Stage 1: 5 bps simplified round-trip. Stage 2+: 12 bps round-trip (crypto). Cost-2x: 24 bps.

## 8. Pre-registered expected metrics
| Metric | Expected range | Confidence |
|---|---|---|
| Aggregate CCS | 0.8 – 1.6 | Med |
| Skew (aggregate) | +0.3 – +1.5 | Med |
| Tail capture | 0.30 – 0.55 | Med |
| Calmar | 0.5 – 1.5 | Med |
| Avg trade duration | ≈ 30 bars (horizon) | Med |

## 9. Falsification
- Aggregate skew < +0.3.
- Convexity beta b ≤ 0 at Stage 1.
- Tail capture < 0.20 at Stage 1.
- OOS aggregate skew ≤ 0 at Stage 3.

## 10. Risk / blow-up scenarios
**Scenario:** prolonged crypto range/chop → repeated whipsaw flips, many small losses without a
compensating tail. **Counter-measure:** cap per-asset allocation; halve sleeve after 2 consecutive
losing quarters; rely on the equal-weight 8-asset spread to diversify whipsaw.

## 12. Stage routing
| Stage | Status | Date |
|---|---|---|
| S0 — Registered | submitted | 2024-12-15 |
| S1–S4 | pending (results in cohort-01-results.md) | |

## 13. Sign-off
- [x] Researcher: convexity-pipeline (cohort-01), 2024-12-15
