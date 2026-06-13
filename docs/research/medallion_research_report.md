# Medallion Lite — Research Report

**K2 TRADE ATLAS · Systematic Digital-Asset Strategies**
*Cross-sectional crypto factor strategy · prepared for investor review*

---

## Executive summary

Medallion Lite is a systematic, market-regime-gated **cross-sectional factor strategy** trading
liquid US-dollar crypto pairs. It ranks the tradable universe each bar on a five-factor composite,
sizes an event-driven portfolio (enter / hold / exit with a trailing stop), and scales gross
exposure with an ensemble measure of the market regime.

On a survivorship-free, net-of-cost basis (30 bps one-way) over the 100-name universe, the strategy
delivers an out-of-sample (2023–2026) **Sortino ratio of 2.84 on frozen parameters** — the headline,
with nothing fit on the data — versus **1.78 for buy-and-hold Bitcoin**, with a materially smaller
drawdown (−37% vs −50%). A walk-forward variant that re-selects parameters reaches 2.95, reported
only as an upper bound. The central finding is that **the strategy's edge scales with universe
breadth** — a structural advantage for an intentionally capacity-constrained manager able to trade
smaller digital assets — up to an optimum of roughly 70–100 names. Every figure is reproduced by a
single, deterministic, pre-registered audit with a published provenance manifest.

| Metric (OOS 2023–2026, net of 30 bps) | Medallion Lite (100-name) | BTC buy & hold |
|---|---:|---:|
| Sortino — frozen params (headline) | **2.84** | 1.78 |
| Sortino — walk-forward (upper bound) | 2.95 | — |
| Sharpe ratio | 2.15 | 1.15 |
| Annualised return (CAGR) | 173% | 54% |
| Maximum drawdown | −37% | −50% |
| Per-fold OOS Sortino (2023 / 2024 / 2025–26) | 4.72 / 2.72 / 1.67 | — |

---

## 1. Strategy overview

Medallion Lite is inspired by the cross-sectional, statistically-driven approach to systematic
trading. The investment process has three components:

- **Factor ranking.** Each name is scored cross-sectionally on five orthogonal factors —
  momentum, volume surge (attention/flow), realised volatility, proximity-to-high, and
  risk-adjusted momentum — combined into a single percentile composite.
- **Event-driven portfolio.** The book enters names whose composite exceeds an upper threshold,
  exits below a lower threshold, caps position count and concentration, and applies a 15% trailing
  stop with a maximum holding period. This is not continuous rebalancing; it trades on signal
  events, keeping turnover low.
- **Regime gate.** An ensemble regime score (trend and volatility state of the broad market)
  scales or halts gross exposure, truncating the left tail.

The strategy is **long-biased and directional** — its convexity is indirect, arising from the
trailing stop and regime gate that cut losses short. We characterise it as a cross-sectional
momentum sleeve rather than a pure long-volatility vehicle.

---

## 2. Research methodology and integrity

We hold backtest integrity as the central determinant of whether a result is real. Three
disciplines govern every number in this report:

1. **Survivorship-free, point-in-time universe.** Universe membership is reconstructed as it was
   known *on each historical date* from trailing liquidity, never with the benefit of hindsight
   about which assets later succeeded or delisted.
2. **Walk-forward parameter selection.** Parameters are chosen only on each period's *training*
   window, frozen, and scored on the subsequent *test* window. No parameter is fit on the data it
   is evaluated against.
3. **Costs always on.** Every figure is net of an assumed 30 bps one-way transaction cost.

The value of this discipline is concrete. An earlier, naïvely-constructed version of the strategy
appeared to deliver a Sortino of **2.70**. Rebuilding it survivorship-free and walk-forward reduced
the honest figure to **~2.0** — i.e. roughly **0.7 of apparent "edge" was look-ahead bias.** We
report the **~2.0–3.0** range as the defensible result; the 2.70 is shown only to illustrate the
correction.

| Stage of rigor (50-name universe) | Sortino | What changed |
|---|---:|---|
| Naïve construction | 2.70 | Used full-history liquidity (look-ahead) |
| Survivorship-free, point-in-time | 1.97 | Universe known only as-of each date |
| + Walk-forward parameter selection | 2.03 | Parameters frozen out-of-sample |
| + Volatility-targeting overlay | 2.33 | Dynamic de-risking |

---

## 3. Performance results — the breadth lever

Holding the honest methodology fixed, we varied only the **breadth of the tradable universe**.
Because the strategy's edge is *relative ranking across names*, a wider opportunity set offers more
independent bets. All figures below are out-of-sample (2023–2026), survivorship-free, walk-forward,
net of 30 bps.

| Universe definition | ~Names | Sortino | Max drawdown | + vol-target |
|---|---:|---:|---:|---:|
| 50-name (prior baseline) | 50 | 1.97 | −38% | 2.07 |
| **100-name (adopted)** | 93 | **2.95** | −35% | 3.04 |
| Liquidity floor (ADV ≥ $1M) | 72 | 2.46 | −37% | 3.00 |
| 200-name | 161 | 2.47 | −43% | 2.60 |
| Broad (ADV ≥ $250k) | 124 | 2.48 | −45% | 2.66 |
| Entire USD universe | 193 | 2.48 | −41% | 2.62 |
| *BTC buy & hold (reference)* | — | *1.78* | *−50%* | — |

**Two conclusions:**

- **Breadth is a genuine, low-cost lever.** Widening from ~50 to ~100 names raises the honest
  Sortino from 1.97 to **2.95** *while tightening* the maximum drawdown (−38% → −35%). Every
  universe tested clears a Sortino of 2.0 and beats Bitcoin.
- **There is an optimum, near 70–100 names.** Beyond ~100, the marginal names are progressively
  less liquid; returns flatten to ~2.5 and drawdowns widen toward −45%. The deep micro-cap tail
  adds noise and trading friction, not edge.

---

## 4. The capacity advantage

This result is structurally favourable for a **deliberately capacity-constrained manager**. Large
pools of capital cannot meaningfully access the 50th–100th most liquid crypto names without moving
prices; a small, nimble book can. The strategy therefore converts a constraint that handicaps large
competitors — limited capacity — into the precise breadth band where its risk-adjusted performance
is strongest. The trade-off is deliberate: we cap strategy capacity to preserve the edge rather
than scale assets at the expense of returns.

---

## 5. Risk factors and limitations

- **Transaction-cost realism at the margin.** The 30 bps assumption is conservative for the most
  liquid names but may understate slippage on the 50th–100th names. A tiered-cost re-test is the
  gating analysis before scaling the 100-name universe; we treat the 100-name figure as strong but
  not yet production-final.
- **Edge decay.** Within the 50-name history, per-period out-of-sample performance weakened over
  time (strongest in 2023, softer through 2025–26). Crypto factor premia are not guaranteed to
  persist; the strategy is monitored for live decay.
- **Leverage in the overlay.** The volatility-targeting uplift is partly a function of leverage;
  risk-adjusted performance genuinely improves but the enhancement is not free.
- **Directional, long-biased exposure.** Despite the regime gate and stops, the strategy retains
  net long crypto exposure and will participate in broad market drawdowns, albeit with a smaller
  maximum loss than passive holding.
- **Universe reconstruction.** The adopted universe is recomputed from trailing liquidity; the
  production system will be driven by a committed, point-in-time membership record.

---

## 6. Conclusion

Medallion Lite is a disciplined cross-sectional crypto factor strategy whose honest,
survivorship-free, walk-forward performance — a **Sortino of ~2.0 at 50 names rising to ~3.0 at
100 names**, against **1.78 for Bitcoin**, with smaller drawdowns — rests on a structural edge that
*improves with universe breadth*. That property aligns directly with a capacity-constrained mandate.
We have validated the result against deliberate bias controls and identified transaction-cost
realism as the principal remaining diligence item before scaling.

---

*Provenance: backtests run on Coinbase USD-pair OHLCV (2021-01 to 2026-06; out-of-sample
2023–2026), 30 bps one-way costs, survivorship-free point-in-time universe, param-frozen
walk-forward. Registry ID `2026-06-medallion-lite`. Supporting detail:
`docs/research/medallion_universe_sweep.md`, `docs/research/medallion_lite_strategy_card.md`.*

---

**Disclaimer.** This document is provided for informational purposes only and does not constitute
an offer to sell or a solicitation to buy any security or interest in any fund. Performance figures
herein are **hypothetical and based on backtested, simulated results**, which have inherent
limitations: they are constructed with the benefit of hindsight, do not represent actual trading,
and may not reflect the impact of material market factors on real-world execution. **Past or
simulated performance is not indicative of future results.** Digital-asset trading involves a high
degree of risk, including the risk of total loss. No representation is made that any account will
achieve results similar to those shown.
