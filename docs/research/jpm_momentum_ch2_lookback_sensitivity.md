# Momentum Strategies in Digital Assets: Chapter 2

## Lookback Sensitivity -- Absolute Momentum Prototypes

*Research note -- JPM Momentum (Kolanovic & Wei 2015) adapted for digital assets*
*Branch: `research/jpm-momentum-v0` | Script: `step_02_lookback_sweep.py`*

---

## 1. Introduction

Chapter 1 established that a 252-day (12-month) absolute momentum factor produces a Sharpe ratio of -0.09 in crypto spot markets, in stark contrast to the 0.49 Sharpe the same methodology delivers in commodities. A natural hypothesis is that crypto's faster trend cycles require a shorter lookback window. In this chapter we sweep across seven lookback horizons spanning one week to one year to test this hypothesis.

This corresponds to the paper's Chapter 2 (p.20--24), "Absolute Momentum Prototypes," where Kolanovic and Wei evaluate lookback windows from 1 month to 12 months across equities, bonds, currencies, and commodities.

## 2. Methodology

### 2.1 Lookback Grid

We test seven lookback windows, covering approximately one trading week to one trading year:

| Window | Calendar Equivalent |
|:---|:---|
| 5 days | ~1 week |
| 10 days | ~2 weeks |
| 21 days | ~1 month |
| 42 days | ~2 months |
| 63 days | ~3 months |
| 126 days | ~6 months |
| 252 days | ~12 months |

This extends the paper's grid (which starts at 1 month) to include shorter windows that may better suit crypto's faster dynamics.

### 2.2 Common Setup

All other parameters are held constant across the sweep to isolate the lookback effect:

- **Signal**: Trailing L-day price return, lagged by 1 day
- **Rule**: Absolute momentum (long if trailing return > 0, else cash)
- **Weighting**: Equal-weight across selected assets
- **Rebalance**: Monthly (every 21 trading days)
- **Universe**: Dynamic, ADV > $1M, minimum 90-day history (284 eligible symbols)
- **Costs**: 20 bps round-trip
- **Period**: January 2018 -- December 2025

## 3. Results

### 3.1 Performance Summary

**Table 3: Absolute Momentum Factor by Lookback Window (Jan 2018 -- Dec 2025)**

| Lookback | CAGR | Vol | Sharpe | Sortino | MaxDD | Hit Rate | Skew | Kurtosis |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5d | -28.5% | 79.4% | -0.02 | -0.03 | -99.0% | 45.3% | -0.13 | 5.30 |
| 10d | -29.0% | 84.4% | **+0.02** | +0.03 | -97.1% | 46.2% | -0.10 | 9.04 |
| 21d | -34.0% | 81.6% | -0.10 | -0.13 | -99.0% | 44.7% | +0.09 | 6.53 |
| 42d | -35.1% | 90.7% | -0.02 | -0.02 | -98.5% | 45.2% | +0.35 | 19.19 |
| 63d | -30.1% | 81.0% | -0.03 | -0.04 | -95.9% | 45.2% | -0.38 | 7.55 |
| 126d | -17.4% | 86.6% | **+0.22** | +0.28 | -97.1% | 45.2% | -0.31 | 9.67 |
| 252d | -34.3% | 81.6% | -0.09 | -0.11 | -97.4% | 45.1% | -0.66 | 8.38 |
| *BTC B&H* | *+26.3%* | *66.1%* | *+0.69* | *+0.92* | *-81.4%* | *51.0%* | *-0.31* | *8.55* |

**Table 4: Portfolio Characteristics by Lookback**

| Lookback | Avg Gross | Daily Turnover | Avg Holdings |
|:---|---:|---:|---:|
| 5d | 84.9% | 6.58% | 20.6 |
| 10d | 87.6% | 6.44% | 20.1 |
| 21d | 83.9% | 6.70% | 18.9 |
| 42d | 84.9% | 5.62% | 20.2 |
| 63d | 83.7% | 4.94% | 20.5 |
| 126d | 83.5% | 3.90% | 19.3 |
| 252d | 85.2% | 3.14% | 15.9 |

### 3.2 Visual Analysis

![Step 2 Lookback Sweep](../../artifacts/research/jpm_momentum/step_02/step_02_lookback_sweep.png)

**Top-left (Equity curves):** All seven lookback variants lose money over the sample period. The 126d lookback is the best performer (green, uppermost), tracking BTC during bull markets but decaying during bear markets. The 21d and 252d lookbacks are the worst, with nearly complete capital destruction.

**Top-right (Sharpe by lookback):** Only two lookbacks produce positive Sharpe ratios: 126d (+0.22) and 10d (+0.02). Neither approaches the BTC benchmark (0.69, dashed orange line). The relationship between lookback and Sharpe is not monotonic -- there is no clean "shorter is better" pattern.

**Bottom-left (Risk-return):** All momentum variants cluster in the lower-right quadrant: high drawdown (>95%), negative CAGR. BTC sits alone in the upper-left with the best risk-return profile.

**Bottom-right (Turnover):** Shorter lookbacks generate substantially higher turnover (6.6% daily at 5d vs. 3.1% at 252d), which increases transaction cost drag. The number of holdings is relatively stable at 16--21 across all lookbacks.

## 4. Analysis

### 4.1 Comparison to the Original Paper

The paper (Table on p.22) shows that commodity momentum Sharpe ratios range from approximately 0.3 to 0.6 across lookback windows of 1--12 months, with a relatively flat profile and a mild peak at 6--12 months. In equities, the profile is similar but peaks somewhat earlier.

Our crypto results show a fundamentally different pattern:

| Property | Paper (Commodities) | Crypto |
|:---|:---|:---|
| Sharpe range | +0.3 to +0.6 | -0.10 to +0.22 |
| Best lookback | 6--12 months | 6 months (126d) |
| Worst lookback | 1 month | 2 months (42d) |
| All lookbacks profitable? | Yes | **No** |

The best crypto lookback (126d) coincides with the paper's commodity sweet spot, but at a dramatically lower Sharpe level. More importantly, no lookback window produces a strategy that is investable in isolation.

### 4.2 Why No Lookback Window Works

The results indicate that the problem identified in Chapter 1 is not primarily a lookback issue. Even at the optimal 126d window, the Sharpe of 0.22 comes with a -97% maximum drawdown and a -17% CAGR. Three structural factors appear responsible:

**1. The "buy everything positive" problem persists.** At every lookback, the strategy holds 16--21 assets on average. During bull markets, the majority of crypto assets have positive trailing returns (regardless of lookback), so the strategy holds nearly every eligible coin. When the market reverses, the entire portfolio declines simultaneously. This is a portfolio construction problem, not a signal timing problem.

**2. Equal weighting amplifies tail risk.** The equal-weight scheme allocates the same capital to a $500M-ADV large-cap as to a $1M-ADV micro-cap. The micro-cap tokens have far higher idiosyncratic volatility and are more prone to permanent loss, dragging the portfolio into deep drawdowns.

**3. Monthly rebalancing is too slow for short lookbacks.** The 5d and 10d lookbacks generate signals that change rapidly, but the portfolio only rebalances monthly. By the time the portfolio adjusts, the short-term momentum signal has already reversed. This mismatch is evident in the high turnover (6.5%+ daily) at short lookbacks, which reflects large position changes at each monthly rebalance rather than smooth tracking.

### 4.3 The 126d Anomaly

The 126d lookback produces the least-bad result (Sharpe +0.22), which is consistent with the paper's finding that medium-term lookbacks capture the strongest momentum signal. At 6 months, the signal is long enough to filter out short-term noise but short enough to avoid the "buying at the top" problem of the 252d window. However, even at this optimal horizon, absolute momentum applied to a broad, equal-weighted universe is not viable in crypto.

### 4.4 Turnover and Cost Sensitivity

Shorter lookbacks generate 2x the turnover of longer ones, but the cost drag alone (at 20 bps) is not the primary driver of underperformance. Even at zero cost, the strategies would still be deeply negative. The issue is signal quality, not friction.

## 5. Implications

This chapter establishes that the lookback window is a necessary but not sufficient parameter to optimize. The core problem is the portfolio construction rule -- specifically, the combination of:

1. **Absolute momentum** (binary: in or out) rather than **relative momentum** (rank and select top-K)
2. **Equal weighting** rather than **inverse-volatility** or **risk-parity** weighting
3. **Broad universe** inclusion without concentration limits

The paper's subsequent sections address exactly these issues:

- **Relative momentum (Step 3, Paper p.25--29)**: Ranking assets and selecting only the top quintile eliminates the "buy everything positive" problem and concentrates capital in the strongest trends.
- **Signal type selection (Step 4, Paper p.36--41)**: Moving average crossovers and breakout channels provide smoother signals with better entry/exit timing than raw trailing returns.
- **Risk-adjusted momentum (Step 7, Paper p.66--70)**: Normalizing returns by volatility before ranking may help prevent overallocation to high-vol, low-quality names.

The key takeaway is that crypto momentum research must move beyond lookback optimization to address portfolio construction. The signal may exist at the individual asset level, but it is destroyed by the equal-weighted, all-inclusive portfolio wrapper.

## 6. Conclusion

Sweeping absolute momentum across lookback windows from 5 to 252 days produces no investable strategy in crypto. The best lookback (126d) achieves a Sharpe of 0.22 -- positive but accompanied by a -97% drawdown. The pattern is structurally different from the paper's commodity findings, where all lookbacks are profitable. The failure is attributable to portfolio construction (broad universe, equal weighting, binary inclusion) rather than signal absence. The next step is to test relative/cross-sectional momentum with concentrated selection.

---

**Data sources**: Coinbase 1-minute candles (market.duckdb), 351 USD pairs, Jun 2016 -- Dec 2025.
**Artifacts**: `artifacts/research/jpm_momentum/step_02/`
