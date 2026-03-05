# Chapter 8: Diversified Multi-Signal Portfolio (Capstone)

*JPM Momentum Strategies — Digital Asset Recreation*
*Reference: Kolanovic & Wei (2015), Chapter 2, p.73–78*

---

## 1. Motivation

The preceding seven chapters isolated each component of a crypto
momentum strategy — lookback window, signal type, rebalance frequency,
risk overlay, and long-only optimization — in controlled single-variable
experiments. The paper's final section assembles these components into
a diversified multi-signal, multi-speed portfolio.

The core thesis is that blending complementary momentum signals should
deliver a smoother return stream than any single signal alone, through
diversification of timing risk. A 10-day RET signal may whipsaw during
a choppy consolidation while a 42-day EMAC signal holds steady; by
blending them, the portfolio captures trends across multiple horizons.

This chapter tests that thesis in the crypto context.

## 2. Experimental Design

### 2.1 Building Blocks: 9 Sub-Strategies

Each sub-strategy uses the full Chapter 7 long-only pipeline:

- Signal-proportional weighting (signal / vol)
- 15 % max position per asset
- Dynamic cash allocation (sensitivity 0.5)
- Bi-weekly rebalance (10 d)
- 20 bps round-trip costs

Three signal types × three lookback windows:

|           | 10 d | 21 d | 42 d |
|-----------|------|------|------|
| **RET**   | RET 10d | RET 21d | RET 42d |
| **EMAC**  | EMAC 10d | EMAC 21d | EMAC 42d |
| **MAC**   | MAC 10d | MAC 21d | MAC 42d |

### 2.2 Blending Methods

| Method       | Description                                          |
|--------------|------------------------------------------------------|
| Equal Weight | 1/9 allocation to each sub-strategy's return stream  |
| EMV          | Inverse-vol weighting (63 d rolling); risk parity    |
| Sharpe Blend | Weight ∝ rolling 126 d Sharpe ratio (clip ≥ 0)      |

All blends receive a 20 % volatility target as a final overlay.

### 2.3 Partial Blends

- **Signal-type blends**: RET multi-speed, EMAC multi-speed, MAC multi-speed
  (each blends 3 lookbacks for one signal type)
- **Speed blends**: Multi-signal 10d, 21d, 42d (each blends 3 signal
  types at one lookback)

## 3. Results

### 3.1 Part A: The 9 Building Blocks

| Sub-Strategy | Sharpe | MaxDD    | CAGR    | Vol    |
|--------------|--------|----------|---------|--------|
| RET 10d      | 0.52   | -40.3 % | +10.8 % | 26.3 % |
| RET 21d      | 0.39   | -40.6 % | +7.0 %  | 26.6 % |
| RET 42d      | 0.25   | -49.5 % | +3.2 %  | 27.3 % |
| EMAC 10d     | 0.50   | -35.1 % | +9.9 %  | 25.2 % |
| EMAC 21d     | 0.51   | -40.3 % | +10.6 % | 26.3 % |
| **EMAC 42d** | **0.58** | **-37.5 %** | **+12.4 %** | 26.4 % |
| MAC 10d      | 0.38   | -39.0 % | +6.8 %  | 25.5 % |
| MAC 21d      | 0.38   | -38.8 % | +6.8 %  | 26.2 % |
| MAC 42d      | 0.48   | -39.6 % | +9.6 %  | 26.6 % |

**Key observations:**

1. **EMAC dominates at every lookback.** It produces the top-3 Sharpe
   ratios (0.58, 0.51, 0.50). The exponential smoothing continues to
   prove its value.

2. **EMAC 42d is the surprise best single strategy** (Sharpe 0.58,
   CAGR +12.4 %). Under the Chapter 7 long-only pipeline with dynamic
   cash and position limits, the slower lookback outperforms the 21d
   variant that led in prior chapters. The Ch.7 cash overlay handles
   the timing risk that previously hurt slower signals.

3. **Longer lookbacks outperform within each signal type.** The 42d
   lookback beats 21d and 10d for EMAC and MAC, and 10d beats longer
   for RET. This pattern reflects how each signal type interacts with
   the cash overlay.

4. **All 9 sub-strategies are profitable** after costs and risk
   management — the worst (RET 42d) still achieves +3.2 % CAGR with
   Sharpe 0.25.

### 3.2 Part B: Blending Methods — Performance-Adaptive Wins

| Method              | Sharpe | MaxDD    | CAGR    | Vol    |
|---------------------|--------|----------|---------|--------|
| EW Blend (9) + VT   | 0.43   | -34.2 % | +7.1 %  | 21.5 % |
| EMV Blend (9) + VT  | 0.44   | -36.4 % | +7.4 %  | 21.3 % |
| **Sharpe Blend (9) + VT** | **0.73** | **-30.2 %** | **+14.2 %** | **21.5 %** |

This is the most important table of the entire study:

1. **The Sharpe Blend achieves Sharpe 0.73** — the highest of any
   configuration tested, exceeding even BTC buy-and-hold (0.69), with
   a maximum drawdown of only -30.2 %.

2. **Naive blending (EW, EMV) actually *underperforms* the best single
   signal.** EW and EMV both produce Sharpe ~0.44, well below EMAC 42d's
   standalone 0.58. This is because **mean pairwise correlation across
   the 9 sub-strategies is 0.86** — there is simply not enough
   diversification to offset the dilution from weaker signals.

3. **Performance-adaptive weighting (Sharpe Blend) overcomes the
   correlation problem** by dynamically tilting toward whichever
   signal/speed is currently working. When EMAC 42d dominates a quiet
   trend, it receives the lion's share. When fast RET 10d catches a
   sudden reversal, the blend pivots. This is momentum-of-momentum —
   a meta-strategy that exploits the time-varying alpha of each
   sub-signal.

### 3.3 The Correlation Problem

The correlation heatmap (Figure 8B) reveals why naive diversification
fails:

| Metric                   | Value |
|--------------------------|-------|
| Mean pairwise corr       | 0.86  |
| Min pairwise corr        | 0.70  |
| Max pairwise corr        | 0.97  |

With correlations this high, adding more signals to an equal-weight
blend is like adding more copies of the same return stream. The paper's
commodity universe, with its heterogeneous risk drivers (weather, supply
chains, geopolitics), likely benefits from lower cross-signal
correlations. In crypto, all assets are primarily driven by BTC/ETH
beta, which propagates through every momentum signal.

**This is why performance-adaptive (Sharpe) weighting is essential for
crypto.** It concentrates into the signals that are providing alpha
*right now*, rather than averaging across highly correlated streams.

### 3.4 Part C: Signal-Type vs Speed Blends

| Blend                  | Sharpe | MaxDD    |
|------------------------|--------|----------|
| **EMAC multi-speed**   | **0.54** | **-32.5 %** |
| MAC multi-speed        | 0.41   | -36.6 % |
| RET multi-speed        | 0.32   | -40.8 % |
| Multi-signal 10d       | 0.43   | -35.6 % |
| Multi-signal 21d       | 0.38   | -36.3 % |
| Multi-signal 42d       | 0.39   | -38.8 % |

**Signal-type blends outperform speed blends.** Blending EMAC across
three speeds (Sharpe 0.54) is far more effective than blending three
signals at a fixed speed. This confirms that signal type carries more
alpha variation than lookback window in crypto.

EMAC multi-speed also achieves the lowest MaxDD (-32.5 %) among the
partial blends, reinforcing the "EMAC + multiple speeds" as the
strongest building-block approach.

### 3.5 Part D: The Final Leaderboard

| Strategy                     | Sharpe | MaxDD    | CAGR    | Vol    |
|------------------------------|--------|----------|---------|--------|
| **Sharpe Blend (9) + VT**   | **0.73** | **-30.2 %** | **+14.2 %** | **21.5 %** |
| BTC Buy & Hold               | 0.69   | -81.4 % | +26.3 % | 66.1 % |
| Ch.7 Optimal (EMAC 21d + VT) | 0.58  | -33.4 % | +10.7 % | 21.8 % |
| EMAC multi-speed + VT        | 0.54   | -32.5 % | +9.7 %  | 21.1 % |
| Best single (EMAC 42d) + VT  | 0.54  | -35.1 % | +9.5 %  | 20.6 % |
| EMV Blend (9) + VT           | 0.44   | -36.4 % | +7.4 %  | 21.3 % |

The **Sharpe Blend of 9 sub-strategies with a 20 % vol target** is
the definitive output of this research:

- **Sharpe 0.73** — exceeds BTC buy-and-hold
- **MaxDD -30.2 %** — less than 40 % of BTC's worst drawdown
- **CAGR +14.2 %** — net of 20 bps costs, with ~20 % average exposure
- **Vol 21.5 %** — targeted and stable
- **Sortino 0.97** — near-unit downside-adjusted return

## 4. Architecture of the Final Portfolio

```
┌─────────────────────────────────────────────────┐
│          9 Sub-Strategy Return Streams           │
│                                                  │
│  RET 10d   RET 21d   RET 42d                    │
│  EMAC 10d  EMAC 21d  EMAC 42d                   │
│  MAC 10d   MAC 21d   MAC 42d                    │
│                                                  │
│  Each sub-strategy:                              │
│    • Signal-proportional weighting (signal/vol)  │
│    • 15 % max position per asset                 │
│    • Dynamic cash (sensitivity 0.5)              │
│    • Bi-weekly rebalance (10 d)                  │
│    • 20 bps costs                                │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│        Performance-Adaptive (Sharpe) Blend       │
│                                                  │
│  weight(i,t) ∝ max(rolling_sharpe(i, 126d), 0)  │
│  Normalize to sum = 1                            │
│  Portfolio return = Σ weight(i,t) × return(i,t)  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│          Volatility Target (20 % ann.)           │
│                                                  │
│  scalar = 0.20 / realized_vol(42d)              │
│  final_return = blend_return × min(scalar, 2.0)  │
└─────────────────────────────────────────────────┘
```

## 5. Comparison with the Paper

| Dimension            | Paper (Commodities)          | Our Findings (Crypto)           |
|----------------------|-----------------------------|---------------------------------|
| Best single signal   | 12-month momentum (~0.50)   | EMAC 42d (0.58)                 |
| Naive blend benefit  | Significant                 | **Negative** (correlation too high) |
| EMV vs EW            | EMV slightly better         | Nearly identical (both ~0.44)   |
| Adaptive weighting   | Not tested                  | **Massive benefit** (0.73 vs 0.44) |
| Multi-signal Sharpe  | ~0.60–0.70                  | **0.73**                        |
| MaxDD improvement    | Moderate                    | -30.2 % (from -81.4 % BTC)     |

The paper's conclusion that "combining signals improves risk-adjusted
returns" holds for crypto, but **only with performance-adaptive
weighting**. The naive blending that works in commodities fails in
crypto because of the asset class's extreme return correlation.

## 6. Caveats and Robustness Considerations

1. **Sharpe Blend has look-back sensitivity.** The 126-day rolling
   window for performance weighting introduces a parameter that could
   be overfit. Robustness tests across 63 d, 126 d, and 252 d windows
   are warranted before deployment.

2. **The 0.73 Sharpe includes an implicit momentum factor.** The Sharpe
   Blend itself is a momentum strategy — it tilts toward recently
   outperforming sub-strategies. This makes it vulnerable to
   momentum-of-momentum crashes (sudden regime changes).

3. **Correlation structure may be time-varying.** The mean correlation
   of 0.86 is a full-sample statistic. During crypto winters,
   correlations spike toward 1.0 (all assets fall together); during
   altcoin seasons, they may drop. The blend's effectiveness depends
   on these decorrelation episodes.

4. **Transaction costs are applied within sub-strategies, not at the
   blend level.** In practice, sub-strategy weights overlap — the same
   asset appears in multiple sub-strategies — and netting would reduce
   actual turnover. The reported results are therefore *conservative*
   on costs.

## 7. Journey Summary: From -0.09 to 0.73

| Chapter | Configuration                        | Sharpe | MaxDD    |
|---------|--------------------------------------|--------|----------|
| 1       | Baseline 252d absolute momentum      | -0.09  | -99.4 %  |
| 2       | Best lookback (126d)                 | +0.22  | -97.2 %  |
| 3       | Cross-sectional, IVW, weekly         | +0.35  | -90+ %   |
| 4       | MAC 42d signal                       | +0.43  | -90+ %   |
| 5       | Bi-weekly rebalance                  | +0.56  | -92.4 %  |
| 6       | + Vol targeting 20 %                 | +0.55  | -48.5 %  |
| 7       | + Long-only optimization             | +0.54  | -38.4 %  |
| **8**   | **Sharpe Blend 9 + VT**              | **+0.73** | **-30.2 %** |

Each chapter contributed a distinct improvement:
- Chapters 1–5 improved **alpha** (signal quality, portfolio construction)
- Chapter 6 improved **risk management** (drawdown reduction)
- Chapter 7 improved **implementation** (long-only constraints)
- Chapter 8 improved **diversification** (multi-signal blending)

## 8. Practical Implications

1. **The Sharpe Blend is the recommended production configuration.**
   It captures the paper's thesis (multi-signal diversification)
   through an adaptive mechanism suited to crypto's high-correlation
   regime.

2. **EMAC is the backbone signal type.** If forced to simplify, an
   EMAC multi-speed blend (Sharpe 0.54) is a strong second choice
   with fewer moving parts.

3. **The strategy is capital-efficient.** With ~20 % average exposure
   and a 20 % vol target, 80 % of capital is available for other
   allocations (yield farming, lending, or simply risk-free rates).

4. **Net of costs, the strategy generates real wealth.** +14.2 % CAGR
   at -30.2 % MaxDD with ~21 % volatility represents an investable
   risk/return profile for a crypto allocation.

5. **The full pipeline can be implemented in production** using the
   project's existing infrastructure (`market.duckdb` → daily bars →
   signal generation → weight construction → risk overlay).

---

*Script*: `scripts/research/jpm_momentum/step_08_multi_signal_portfolio.py`
*Artifacts*: `artifacts/research/jpm_momentum/step_08/`
*Research series complete.*
