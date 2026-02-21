# Momentum Strategies in Digital Assets: Chapter 4

## Selection of Trend Signal

*Research note -- JPM Momentum (Kolanovic & Wei 2015) adapted for digital assets*
*Branch: `research/jpm-momentum-v0` | Script: `step_04_signal_shootout.py`*

---

## 1. Introduction

Chapters 2 and 3 used a single signal type -- raw trailing price return -- and explored lookback windows and portfolio construction. The best combination (10d lookback, top-quintile selection, inverse-vol weighting, weekly rebalance) achieved a Sharpe of 0.35.

In this chapter we test the paper's full suite of trend signals (p.36--41) to determine whether alternative signal constructions improve upon raw returns in crypto. The paper finds that moving average and linear regression signals can reduce turnover and improve risk-adjusted performance in traditional asset classes. We test whether this holds for digital assets.

## 2. Signal Definitions

Five signal types are tested, each parameterized by a lookback window L:

| Signal | Formula | Interpretation |
|:---|:---|:---|
| **RET** | `close(t-1) / close(t-1-L) - 1` | Raw trailing return (baseline) |
| **MAC** | `(SMA_fast - SMA_slow) / SMA_slow` | SMA crossover. Fast = L/4, slow = L. Continuous signal normalized by slow MA. |
| **EMAC** | `(EMA_fast - EMA_slow) / EMA_slow` | EMA crossover. Same structure but exponentially weighted. |
| **BRK** | `(close - min_L) / (max_L - min_L)` | Donchian breakout channel position. Ranges 0--1. |
| **LREG** | `slope(OLS(log_price, t)) / std_err` | t-statistic of linear regression slope on log-prices. Measures both trend direction and statistical significance. |

All lookbacks use a 1-day shift to prevent same-bar leakage.

## 3. Test Design

Each signal is tested in both absolute and relative momentum modes, at three lookback windows (10d, 21d, 42d):

- **Absolute (ABS)**: Long if signal > threshold (0 for most signals, 0.5 for BRK), else cash
- **Relative (XS)**: Rank by signal, long top quintile

Common parameters: inverse-vol weighting, weekly rebalance, 20 bps costs. This gives a 5 x 3 x 2 = 30-cell grid.

## 4. Results

### 4.1 Primary Comparison: Relative Momentum at 10d Lookback

**Table 7: Signal Comparison at 10d Lookback, Relative Mode**

| Signal | CAGR | Vol | Sharpe | Sortino | MaxDD | Turnover |
|:---|---:|---:|---:|---:|---:|---:|
| **RET** | **-11.4%** | 96.2% | **+0.35** | +0.51 | -94.8% | 21.8% |
| EMAC | -19.4% | 95.8% | +0.25 | +0.37 | -96.3% | 24.1% |
| LREG | -26.4% | 91.9% | +0.13 | +0.18 | -96.3% | 26.7% |
| MAC | -28.3% | 93.8% | +0.12 | +0.17 | -96.8% | 27.2% |
| BRK | -34.9% | 92.7% | +0.00 | +0.00 | -97.9% | 29.4% |

At the 10-day lookback, **raw returns (RET) decisively win**, with a Sharpe of 0.35 versus 0.25 for the next-best signal (EMAC). Breakout channel (BRK) is effectively flat at 0.00. This is the opposite of the paper's finding, where smoother signals like MAC and LREG outperform raw returns.

### 4.2 Full Sharpe Grid

**Table 8: Sharpe Ratio by Signal, Lookback, and Mode**

| Signal | ABS 10d | XS 10d | ABS 21d | XS 21d | ABS 42d | XS 42d |
|:---|---:|---:|---:|---:|---:|---:|
| RET | 0.13 | **0.35** | 0.29 | 0.24 | 0.22 | 0.09 |
| MAC | -0.02 | 0.12 | 0.29 | 0.28 | 0.24 | **0.43** |
| EMAC | 0.15 | 0.25 | 0.19 | **0.40** | 0.17 | 0.31 |
| BRK | 0.03 | 0.00 | 0.26 | 0.23 | -0.10 | **0.42** |
| LREG | -0.06 | 0.13 | 0.33 | 0.37 | 0.20 | 0.12 |

The full grid reveals a striking **interaction between signal type and lookback**:

- **At 10d**: RET dominates (0.35). Smoother signals (MAC, LREG) lag badly.
- **At 21d**: EMAC leads (0.40), followed by LREG (0.37). RET fades to 0.24.
- **At 42d**: MAC (0.43) and BRK (0.42) take the lead. RET collapses to 0.09.

The **overall best cell is MAC 42d XS at Sharpe 0.43**, followed by BRK 42d XS (0.42) and EMAC 21d XS (0.40).

### 4.3 Visual Analysis

![Step 4 Signal Shootout](../../artifacts/research/jpm_momentum/step_04/step_04_signal_shootout.png)

**Top-left (Equity curves at 10d):** RET (blue) clearly outperforms at the 10-day horizon. EMAC (green) is second. BRK (orange) and MAC (teal) track each other poorly. All still lose money over the full period.

**Top-right (Sharpe heatmap):** The diagonal gradient from upper-left (RET/10d) to lower-right (MAC/42d, BRK/42d) reveals that signal type and lookback are not independent. Faster signals work at shorter horizons; smoother signals work at longer horizons.

**Bottom-left (Sharpe vs turnover):** The green triangles (42d) in the upper-left show the best efficiency frontier -- higher Sharpe at lower turnover. The 10d variants (circles) tend to have both higher turnover and variable Sharpe. Critically, the MAC and BRK signals at 42d achieve the highest Sharpe (0.42--0.43) with meaningfully lower turnover than RET at 10d.

**Bottom-right (Absolute vs relative):** Relative momentum (green) outperforms absolute (red) for almost every signal at 10d, confirming Chapter 3's finding. The sole exception is LREG, where absolute is slightly better.

## 5. Analysis

### 5.1 Comparison to the Paper

The paper (p.36--41) finds that:
1. Moving average and linear regression signals outperform raw returns across asset classes.
2. The improvement comes primarily from **lower turnover**, which reduces transaction costs.
3. All signal types produce similar risk-adjusted returns after accounting for their different turnover profiles.

Our findings partially confirm and partially contradict:

| Paper Finding | Crypto Result |
|:---|:---|
| Smooth signals beat raw returns | **Depends on lookback.** At 10d, raw returns win. At 42d, smooth signals win. |
| Lower turnover improves net performance | **Confirmed.** The 42d smooth signals achieve the highest Sharpe at the lowest turnover. |
| Signal choice has modest impact | **Contradicted.** Signal choice matters enormously -- Sharpe ranges from 0.00 (BRK/10d) to 0.43 (MAC/42d). |

### 5.2 Why Signal and Lookback Interact

The interaction is explained by the nature of each signal:

**Raw returns (RET)** are a point-in-time comparison (price now vs. price L days ago). They react instantly to price changes but are noisy. At short horizons (10d) this reactivity is valuable because crypto trends are brief. At longer horizons (42d+) the noise dominates and false signals multiply.

**Moving average crossovers (MAC, EMAC)** are inherently smoothing filters. At short horizons (10d), the fast MA (2.5d) and slow MA (10d) are both too noisy to produce useful crossovers -- the signals whipsaw constantly. At longer horizons (42d), the fast MA (10d) and slow MA (42d) provide genuine trend identification that adapts progressively as prices change.

**Breakout (BRK)** requires a price to reach the top of its L-day range. At 10d, this happens frequently due to crypto's high daily volatility -- practically every asset breaches its 10-day high or low on any given day, making the signal uninformative. At 42d, a breakout to a 42-day high is a more meaningful event.

**Linear regression (LREG)** measures the statistical significance of a trend. At 10d, fitting a regression to 10 noisy data points produces unreliable t-statistics. At 21d+, the regression becomes more meaningful.

### 5.3 The Optimal Signal is Lookback-Dependent

This is a key finding with practical implications. There is no single "best" signal:

| Horizon | Best Signal | Sharpe |
|:---|:---|---:|
| Very short (10d) | RET | 0.35 |
| Short (21d) | EMAC | 0.40 |
| Medium (42d) | MAC | 0.43 |

This suggests that a **multi-signal, multi-speed blend** (the paper's recommendation in Chapter 3, p.73--78) could capture the strengths of each combination. This will be tested in Step 8.

### 5.4 Turnover Implications

The 42d MAC signal achieves the highest Sharpe (0.43) with daily turnover of ~16%, compared to 22% for 10d RET. At 20 bps cost, this turnover difference accounts for approximately 1.2% annualized cost savings. More importantly, lower turnover implies more stable positions, which reduces the risk of whipsaw losses during volatile periods.

## 6. Conclusion

Signal type selection has a large impact on crypto momentum performance, but the optimal signal depends on the lookback horizon. Raw returns dominate at very short horizons (10d), while moving average crossovers and breakout signals dominate at medium horizons (42d). The overall best configuration is **MAC 42d relative momentum (Sharpe 0.43)**, which improves upon the Chapter 3 baseline of 0.35 while reducing turnover. The strong interaction between signal type and lookback motivates a multi-signal blend in later steps.

All strategies continue to exhibit catastrophic drawdowns (-93% to -98%) due to the absence of risk management overlays, which remain the primary focus for future chapters.

---

**Data sources**: Coinbase 1-minute candles (market.duckdb), 351 USD pairs, Jun 2016 -- Dec 2025.
**Artifacts**: `artifacts/research/jpm_momentum/step_04/`
