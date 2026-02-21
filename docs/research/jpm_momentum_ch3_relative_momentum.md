# Momentum Strategies in Digital Assets: Chapter 3

## Relative (Cross-Sectional) Momentum

*Research note -- JPM Momentum (Kolanovic & Wei 2015) adapted for digital assets*
*Branch: `research/jpm-momentum-v0` | Script: `step_03_relative_momentum.py`*

---

## 1. Introduction

Chapter 2 established that absolute momentum (long if trailing return > 0, else cash) fails across all lookback windows in crypto. We identified the core problem as portfolio construction: the "buy everything positive" rule indiscriminately loads on dozens of assets during bull markets, with equal weighting amplifying exposure to the weakest names.

In this chapter we test the paper's relative momentum framework (p.25--29), which addresses this problem by **ranking** assets cross-sectionally and selecting only the **top-K** performers. This concentrates capital in the strongest trends and eliminates the long tail of marginal holdings. We also test inverse-volatility weighting (the paper's risk-adjusted approach) and weekly rebalancing.

## 2. Methodology

### 2.1 Cross-Sectional Ranking

On each rebalance date, all eligible assets are ranked by their trailing L-day return. The portfolio goes long the top-K (or top quintile) assets, with all others excluded.

This is a pure **relative** signal: an asset is included not because its return is positive, but because its return ranks among the highest in the cross-section. In a bear market where all assets are declining, the portfolio still selects the "least-bad" performers.

### 2.2 Parameter Grid

**Primary sweep** (equal-weight, monthly rebalance):

| Dimension | Values |
|:---|:---|
| Lookback | 10, 21, 42, 63, 126 days |
| Selection | Top-5, Top-10, Top quintile (top 20%) |

**Variant sweep** (at the best lookback/selection from primary):

| Dimension | Values |
|:---|:---|
| Weighting | Equal-weight, Inverse-volatility (63d realized vol) |
| Rebalance | Weekly (5d), Monthly (21d) |

### 2.3 Inverse-Volatility Weighting

For the inverse-vol variant, each selected asset's weight is proportional to the reciprocal of its trailing 63-day realized volatility (annualized), with a floor of 10% to prevent extreme concentration in low-vol names:

```
w_i = (1 / max(vol_i, 0.10)) / sum_j(1 / max(vol_j, 0.10))
```

This naturally tilts the portfolio toward more stable assets -- a form of risk-budgeting that the paper advocates (p.66--70).

## 3. Results

### 3.1 Primary Sweep: Lookback x Selection

**Table 5: Relative Momentum -- Sharpe by Lookback and Selection (EW, Monthly)**

| Lookback | Top-5 | Top-10 | Quintile |
|:---|---:|---:|---:|
| **10d** | **+0.07** | **+0.21** | **+0.24** |
| 21d | +0.01 | +0.04 | +0.08 |
| 42d | -0.15 | -0.04 | -0.02 |
| 63d | -0.13 | +0.01 | -0.03 |
| 126d | +0.04 | +0.03 | -0.03 |

The heatmap reveals a clear gradient: **shorter lookbacks dominate**, with 10-day momentum producing the only consistently positive Sharpe ratios. This is a structural departure from the paper's commodity results, where 6--12 month lookbacks are optimal.

Within each lookback row, **broader selection (quintile or top-10) outperforms concentrated selection (top-5)**. This is counter-intuitive -- one might expect that concentrating in the very best performers would produce higher returns -- but in crypto the top-5 assets have extreme idiosyncratic volatility that overwhelms the momentum signal.

### 3.2 Best Configuration Variants

At the optimal primary setting (10d lookback, quintile selection), we test weighting and rebalance frequency:

**Table 6: 10d Quintile Variants**

| Variant | CAGR | Vol | Sharpe | Sortino | MaxDD | Hit Rate |
|:---|---:|---:|---:|---:|---:|---:|
| EW, monthly | -20.9% | 96.1% | 0.24 | 0.34 | -94.0% | 49.7% |
| EW, weekly | -16.6% | 98.1% | 0.30 | 0.45 | -96.5% | 50.6% |
| **InvVol, weekly** | **-11.4%** | **96.2%** | **0.35** | **0.51** | **-94.8%** | **51.0%** |
| InvVol, monthly | -13.2% | 94.8% | 0.33 | 0.47 | -90.0% | 50.4% |
| *BTC Buy & Hold* | *+26.3%* | *66.1%* | *0.69* | *0.92* | *-81.4%* | *51.0%* |

Both improvements are additive:
- **Weekly rebalance** improves Sharpe by ~0.06 over monthly, as the 10-day signal changes rapidly enough to warrant more frequent portfolio updates.
- **Inverse-volatility weighting** improves Sharpe by ~0.05 over equal-weight, by tilting away from the most volatile (and typically lowest-quality) names.

The best variant (**10d / quintile / inverse-vol / weekly**) achieves a Sharpe of **0.35** and Sortino of **0.51**.

### 3.3 Visual Analysis

![Step 3 Relative Momentum](../../artifacts/research/jpm_momentum/step_03/step_03_relative_momentum.png)

**Top-left (Equity curves, top-5 by lookback):** The 10d lookback (dark blue) is the strongest performer, though all variants still lose money over the full period. The 42d and 63d lookbacks are notably the worst -- the "middle ground" is a dead zone in crypto momentum.

**Top-right (Sharpe heatmap):** The gradient from green (10d) to red (42-63d) is striking. The 10-day row is entirely positive; the 42-63d rows are entirely negative. This confirms that crypto momentum lives at very short horizons.

**Bottom-left (Best variant vs BTC):** Even the best relative momentum strategy (Sharpe 0.35) declines ~90% during bear markets. It participates in bull rallies (matching BTC's peaks in 2021) but gives it all back. The strategy is always ~97% invested -- there is no cash allocation mechanism to protect during drawdowns.

**Bottom-right (Variant comparison):** Inverse-vol + weekly rebalance produces the highest Sharpe (0.35), but all four variants cluster between 0.24 and 0.35. The weighting/rebalance improvements are incremental.

## 4. Analysis

### 4.1 Improvement over Absolute Momentum

| Approach | Best Sharpe | Improvement |
|:---|---:|:---|
| Absolute (Ch.2) | +0.22 (126d) | -- |
| Relative (Ch.3) | +0.35 (10d/Q1/invvol/5d) | +0.13 (+59%) |

Relative momentum delivers a meaningful improvement over absolute momentum, confirming the paper's general finding (p.25) that cross-sectional ranking adds value. However, the improvement comes from a different mechanism than the paper describes: in commodities, relative momentum works because it isolates the momentum *premium* from the broad market beta. In crypto, it works primarily because it **limits the number of positions** to the strongest names, avoiding the altcoin graveyard.

### 4.2 The 10-Day Lookback

The dominance of the 10-day lookback is the most important finding of this chapter. It diverges sharply from the paper's results across all asset classes, where momentum is strongest at 6--12 months:

| Asset Class | Optimal Lookback (Paper) | Crypto |
|:---|:---|:---|
| Equities | 6--12 months | -- |
| Bonds | 3--6 months | -- |
| Commodities | 6--12 months | -- |
| **Crypto** | -- | **10 days** |

This is consistent with recent academic research: Hsiao & Qi (2024) document that cross-sectional crypto momentum is strongest at 30-day lookback with 7-day holding periods. Our 10-day signal with weekly rebalance is in the same regime.

The structural explanation is that crypto trends are driven by attention cascades (social media, exchange listings, narrative shifts) that play out over days to weeks, not months. By the time a monthly or quarterly momentum signal registers, the attention has moved on.

### 4.3 Why the Strategy Still Loses Money

Despite positive Sharpe ratios (0.24--0.35), all relative momentum strategies produce negative CAGRs (-11% to -21%). This apparent paradox occurs because:

1. **The strategies are always fully invested** (~97% gross exposure). There is no cash allocation, no risk-off mechanism. During bear markets, the portfolio simply rotates into the "least-bad" declining assets.

2. **Volatility is extreme** (94--98% annualized). The Sharpe ratio captures the risk-adjusted *direction* of returns, but with volatility this high, even a slightly positive Sharpe ratio produces negative compounded returns due to the volatility drag: `geometric_return ≈ arithmetic_return - vol²/2`.

3. **Drawdowns are catastrophic** (-90% to -99%). A -90% drawdown requires a +900% rally to recover. The strategy never generates enough upside to recover from bear market losses.

### 4.4 The Missing Ingredient: Risk Management

The results point clearly to what's needed: the momentum signal is **directionally correct** (positive Sharpe) but **unmanaged** (no vol targeting, no stop-loss, always fully invested). The paper dedicates all of Chapter 3 to exactly this problem:

- **Volatility targeting** (p.56): Scale exposure to maintain 20% annualized vol instead of the current ~95%.
- **Stop-loss** (p.56--61): Exit positions or reduce exposure during drawdowns.
- **Mean reversion overlay** (p.62--65): Scale down after extreme short-term moves.

At ~95% realized vol, a simple vol-targeting overlay that scales to 20% target would reduce exposure to ~21% of capital on average, preserving ~79% in cash during normal conditions and even less during high-vol episodes. This alone would dramatically alter the return profile.

## 5. Conclusion

Relative momentum with a 10-day lookback, top-quintile selection, inverse-vol weighting, and weekly rebalance produces a Sharpe of 0.35 -- a 59% improvement over the best absolute momentum variant. The signal is present and directionally positive, but the lack of risk management causes catastrophic drawdowns and negative compounded returns. The next step is to test the paper's signal type alternatives (MA crossover, breakout, linear regression), followed by the risk management overlays that should transform the positive Sharpe into positive realized returns.

---

**Data sources**: Coinbase 1-minute candles (market.duckdb), 351 USD pairs, Jun 2016 -- Dec 2025.
**Artifacts**: `artifacts/research/jpm_momentum/step_03/`
