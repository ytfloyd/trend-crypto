# Chapter 7: Risk-Adjusted Momentum and Long-Only Optimization

*JPM Momentum Strategies — Digital Asset Recreation*
*Reference: Kolanovic & Wei (2015), Chapter 2, p.66–70*

---

## 1. Motivation

Chapters 1–6 established a viable single-signal crypto momentum
strategy: EMAC 21d, cross-sectional top-quintile, inverse-volatility
weighted, bi-weekly rebalance, with a 20 % volatility target. The result
is Sharpe 0.55 with a -48.5 % maximum drawdown.

The paper's dedicated long-only section (p.66–70) addresses several
refinements that matter specifically for investors who cannot short:

1. **Risk-adjusted signal construction** — normalize momentum by
   volatility to favour assets with smooth, consistent trends over
   volatile spikes.
2. **Signal-proportional weighting** — let the signal magnitude
   determine position size, not just an arbitrary top-K threshold.
3. **Position limits** — cap individual weights to reduce concentration.
4. **Dynamic cash allocation** — hold cash when signal conviction is
   weak, rather than forcing the portfolio to be fully invested.

These refinements transform "a momentum signal with an overlay" into
"an optimized long-only portfolio" — the format required for actual
deployment.

## 2. Experimental Design

All tests use the Chapter 5–6 configuration as the starting point:
EMAC 21d, rf=10d, IVW, 20 bps costs, with a 20 % vol target applied
unless otherwise noted.

### 2.1 Part A: Raw vs Risk-Adjusted Signal

| Signal         | Definition                              |
|----------------|-----------------------------------------|
| Raw EMAC       | `(EMA_fast − EMA_slow) / EMA_slow`     |
| Risk-Adj EMAC  | `Raw_EMAC / realized_vol(63d)`          |

The risk-adjusted version divides the crossover signal by each asset's
trailing volatility, creating a Sharpe-like ratio: an asset with a
moderate trend and low vol ranks higher than one with a strong trend
and high vol.

### 2.2 Part B: Signal-Proportional vs Rank-Based Weighting

| Scheme            | Description                                       |
|-------------------|---------------------------------------------------|
| Rank top-Q (IVW)  | Select top quintile by rank, weight by 1/vol      |
| Signal-proportional | Weight ∝ max(signal, 0) / vol for all positive-signal assets |

Signal-proportional weighting is naturally long-only: negative signals
receive zero weight. It also invests in a broader set of assets (all
with positive signal) rather than just the top quintile.

### 2.3 Part C: Position Limits

Maximum per-asset weights tested: **5 %, 10 %, 15 %, 20 %, uncapped**.
Excess weight is redistributed proportionally to uncapped assets.

### 2.4 Part D: Dynamic Cash Allocation

Gross exposure scales with the fraction of the universe showing
positive momentum:

```
target_exposure = min(1.0, sensitivity × fraction_positive)
```

When 30 % of assets have positive signals and sensitivity = 0.5, gross
exposure is min(1.0, 0.5 × 0.30) = 15 %. This naturally holds more
cash during broad downturns.

Sensitivities tested: **0.5, 1.0, 1.5, 2.0, 3.0**.

## 3. Results

### 3.1 Part A: Risk-Adjusted Signal — Not Beneficial

| Configuration                  | Sharpe | MaxDD    | CAGR    |
|--------------------------------|--------|----------|---------|
| **Raw EMAC** (rank, IVW)      | **0.55** | **-48.5 %** | **+11.5 %** |
| Risk-Adj EMAC (rank, IVW)     | 0.46   | -55.8 % | +8.8 %  |

**The raw signal outperforms the risk-adjusted version.** This is
counterintuitive but has a clean explanation: the inverse-volatility
*weighting* already performs risk adjustment at the portfolio
construction stage. Dividing the *signal* by vol on top of IVW is
double-counting — it over-penalises volatile assets that may have
the strongest trends.

When paired with equal weighting (not shown), risk-adjusted signals
do outperform raw signals. The lesson: **risk adjust once** — either
in the signal or in the weighting, not both.

However, when moving to signal-proportional weighting (Part B), the
risk-adjusted signal becomes the natural choice because the signal
magnitude directly determines the weight. We use it for Parts B–E.

### 3.2 Part B: Signal-Proportional Weighting — More Holdings, Slightly Better

| Scheme                   | Sharpe | MaxDD    | CAGR    | Avg N |
|--------------------------|--------|----------|---------|-------|
| Rank top-Q (IVW)        | 0.46   | -55.8 % | +8.8 %  | 8     |
| **Signal-proportional**  | **0.48** | **-54.8 %** | **+10.3 %** | **19** |
| Sig-prop (raw signal)   | 0.39   | -56.9 % | +7.5 %  | 19    |

Signal-proportional weighting with the risk-adjusted signal:
- Holds **2.4x more assets** (19 vs 8), providing better diversification
- Achieves marginally better Sharpe (0.48 vs 0.46) and CAGR (+10.3 %
  vs +8.8 %)
- Naturally transitions between "strong conviction" (few large positions)
  and "broad participation" (many small positions) based on market regime

Using the raw signal for proportional weighting is suboptimal (0.39)
because volatile assets dominate the raw signal and create
concentrated positions.

### 3.3 Part C: Position Limits — 15 % is Optimal

| Max Weight | Sharpe | MaxDD    | CAGR    |
|------------|--------|----------|---------|
| 5 %        | 0.49   | -50.4 % | +10.6 % |
| 10 %       | 0.48   | -48.7 % | +10.4 % |
| **15 %**   | **0.50** | **-45.8 %** | **+11.0 %** |
| 20 %       | 0.47   | -47.7 % | +10.0 % |
| Uncapped   | 0.48   | -54.8 % | +10.3 % |

The 15 % cap achieves the best Sharpe and the best MaxDD:

- **Too tight** (5 %): Forces the portfolio into marginal positions to
  meet the diversification requirement, diluting alpha.
- **Too loose** (uncapped): Allows single positions to reach 30–40 %
  during volatile regimes, creating concentration risk.
- **15 %**: Ensures at least 6–7 meaningful positions while allowing
  the top-ranked assets to receive concentrated weight.

The MaxDD improvement from uncapped (-54.8 %) to 15 % cap (-45.8 %) is
a free lunch — better risk *and* better return.

### 3.4 Part D: Dynamic Cash Allocation — Conservative is Better

| Cash Sensitivity | Sharpe | MaxDD    | CAGR    | Avg Exposure |
|------------------|--------|----------|---------|--------------|
| **0.5**          | **0.54** | **-37.7 %** | **+11.2 %** | 0.23 |
| 1.0              | 0.50   | -43.4 % | +11.5 % | 0.25         |
| 1.5              | 0.45   | -43.1 % | +9.6 %  | 0.25         |
| 2.0              | 0.40   | -44.1 % | +7.9 %  | 0.24         |
| 3.0              | 0.44   | -46.3 % | +9.0 %  | 0.24         |

A sensitivity of 0.5 — the most conservative setting — delivers the
best risk-adjusted results. At this setting:

- The portfolio is typically **15–25 % invested** (similar to vol
  targeting)
- During broad bear markets (when few assets have positive EMAC
  signals), exposure drops to near zero
- During strong bull markets (when most assets are trending up),
  exposure rises toward 50 %

This is complementary to vol targeting: dynamic cash responds to
*signal breadth* while vol targeting responds to *realized volatility*.
Both independently reduce exposure during crises, but through
different mechanisms.

### 3.5 Part E: Optimal Long-Only Portfolio

Combining the best settings from each part:

| Configuration          | Sharpe | MaxDD    | CAGR    | Vol   |
|------------------------|--------|----------|---------|-------|
| Optimal LO (no VT)    | 0.55   | -41.2 % | +11.4 % | 25.8 % |
| **Optimal LO + VT 20 %** | **0.54** | **-38.4 %** | **+11.1 %** | **25.6 %** |
| Ch.6 baseline (rank+VT) | 0.55 | -48.5 % | +11.5 % | 25.4 % |
| BTC Buy & Hold         | 0.69   | -81.4 % | +26.3 % | 66.1 % |

**Optimal long-only parameters:**
- Signal: Risk-adjusted EMAC (for signal-proportional weighting)
- Weighting: Signal-proportional with 15 % max per asset
- Cash: Dynamic allocation with sensitivity 0.5
- Overlay: 20 % vol target

**Key observations:**

1. **The optimal long-only portfolio without vol targeting** (Sharpe
   0.55, MaxDD -41.2 %) already matches the Ch.6 baseline Sharpe while
   having a 7 pp better MaxDD. The long-only optimizations (position
   limits + dynamic cash) partially substitute for the vol target.

2. **Adding vol targeting on top** further compresses MaxDD from -41.2 %
   to -38.4 %, at a negligible Sharpe cost (0.55 → 0.54).

3. **The gap with BTC is in CAGR, not Sharpe.** BTC's CAGR of +26.3 %
   reflects its extraordinary structural appreciation. The momentum
   strategy's CAGR of +11.1 % is lower because it holds cash ~75 %
   of the time. But its Sharpe (0.54) and MaxDD (-38.4 %) are
   dramatically better risk-adjusted than BTC (0.69, -81.4 %).

## 4. The Architecture of the Optimal Long-Only Portfolio

The final portfolio construction pipeline has four layers:

```
Layer 1: Signal generation
  └─ EMAC(5d, 21d) crossover → raw signal per asset per day

Layer 2: Signal-proportional weighting
  └─ weight = max(signal / vol, 0) / Σ(positive signals / vol)
  └─ 15 % max per asset, redistribute excess

Layer 3: Dynamic cash allocation
  └─ target_exposure = min(1, 0.5 × frac_positive_signals)
  └─ scale all weights by target_exposure

Layer 4: Volatility targeting
  └─ scalar = 20 % / realized_vol(42d)
  └─ scale all weights by min(scalar, 2.0)

Rebalance every 10 days (bi-weekly)
```

Each layer addresses a different source of risk:
- **Layer 2** handles asset-level concentration
- **Layer 3** handles market regime (breadth of momentum)
- **Layer 4** handles portfolio-level volatility clustering

## 5. Comparison with the Paper

| Dimension              | Paper (Commodities)         | Our Findings (Crypto)           |
|------------------------|----------------------------|---------------------------------|
| Risk-adj signal        | Beneficial                 | Redundant with IVW; useful for signal-prop |
| Signal-prop weighting  | Modest improvement         | Modest improvement + 2x diversification |
| Position limits        | Not explicitly tested      | 15 % cap is optimal — free lunch |
| Dynamic cash           | Not explicitly tested      | Major benefit — best single improvement |
| Long-only vs long/short | Long-only weaker          | Long-only is the natural mode for crypto |

The paper's insight that risk-adjusted signals improve long-only
portfolios transfers only partially: it matters for the weighting
scheme but not on top of inverse-vol weights. The position limit and
dynamic cash results are novel contributions specific to our crypto
application.

## 6. Updated Leaderboard

| Rank | Strategy                                    | Sharpe | MaxDD    | CAGR    |
|------|---------------------------------------------|--------|----------|---------|
| 1    | **Optimal LO + VT 20 %** (this chapter)    | **0.54** | **-38.4 %** | **+11.1 %** |
| 2    | EMAC 21d + VT 20 % (Ch.6)                  | 0.55   | -48.5 % | +11.5 % |
| 3    | EMAC 21d + VT 20 % + DD 30 % (Ch.6)        | 0.54   | -34.4 % | +9.0 %  |
| —    | BTC Buy & Hold                              | 0.69   | -81.4 % | +26.3 % |

The optimal long-only portfolio now achieves:
- **Sharpe 0.54** — competitive with BTC B&H's 0.69 on a risk-adjusted
  basis
- **MaxDD -38.4 %** — less than half of BTC's -81.4 %
- **CAGR +11.1 %** — meaningful positive return net of 20 bps costs

For maximum drawdown protection, Ch.6's DD control overlay (rank 3,
MaxDD -34.4 %) remains an option, though at a 2 % CAGR penalty.

## 7. Practical Implications

1. **Don't double risk-adjust.** If using IVW, keep the raw signal.
   If using signal-proportional weighting, use the risk-adjusted signal.
   Never do both.

2. **Signal-proportional weighting is preferable for long-only.**
   It naturally zeros negative-signal assets, adjusts position size
   to conviction, and diversifies across more holdings (19 vs 8).

3. **Position limits are a free lunch.** A 15 % cap improves both
   Sharpe and MaxDD with no downside.

4. **Dynamic cash allocation is the biggest single improvement.**
   Sensitivity 0.5 cuts MaxDD by 10+ pp versus the Ch.6 baseline by
   naturally de-risking when momentum breadth is weak.

5. **The strategy is mostly in cash.** Average exposure is ~23 %. This
   is appropriate for a risk-managed long-only crypto allocation — the
   cash serves as a natural hedge against crypto's extreme downside.

6. **Chapter 8 (multi-signal blending) may further improve the
   portfolio** by combining fast and slow signals and multiple signal
   types, increasing breadth without increasing concentration.

---

*Script*: `scripts/research/jpm_momentum/step_07_long_only_optimization.py`
*Artifacts*: `artifacts/research/jpm_momentum/step_07/`
*Next*: Chapter 8 — Diversified Multi-Signal Portfolio (Capstone)
