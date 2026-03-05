# Momentum Strategies in Digital Assets: Chapter 1

## Baseline Prototype Momentum Factor

*Research note -- JPM Momentum (Kolanovic & Wei 2015) adapted for digital assets*
*Branch: `research/jpm-momentum-v0` | Script: `step_01_baseline_momentum.py`*

---

## 1. Introduction

This research note is the first in a series that adapts the framework developed by Kolanovic and Wei (2015) in "Momentum Strategies Across Asset Classes" to digital asset markets. The original report examined momentum strategies across equities, bonds, currencies, and commodities, demonstrating that simple trend-following factors delivered Sharpe ratios in the 0.18--0.57 range over the period 1972--2014, and that a diversified multi-asset momentum portfolio achieved a Sharpe ratio of 0.78.

Our objective is to determine whether the same framework produces comparable results in cryptocurrency spot markets, and where it does not, to identify the structural differences that require adaptation. In this chapter we reproduce the paper's simplest test: a 12-month (252-day) price-return absolute momentum factor (Paper p.9--13, Figure 1, Table 1).

## 2. Methodology

### 2.1 Data

We use 1-minute OHLCV candle data from Coinbase, resampled to daily bars via the `bars_1d` view of `market.duckdb`. The dataset spans January 2017 through December 2025 and comprises 339,271 daily observations across 351 USD-quoted trading pairs. To our knowledge this is a survivorship-bias-free sample: symbols enter and exit the dataset as they are listed and delisted on Coinbase.

### 2.2 Universe Construction

Following the paper's emphasis on tradable instruments, we apply a dynamic universe filter with two criteria:

- **Minimum average daily volume (ADV)**: Rolling 20-day average dollar volume must exceed $1,000,000.
- **Minimum listing age**: A symbol must have at least 90 days of trading history before becoming eligible.

These filters are evaluated daily, producing a time-varying eligible universe. Of the 351 symbols in the raw data, 284 are eligible at some point during the sample. The number of concurrently eligible symbols ranges from approximately 5 in early 2018 to over 130 during the 2024 bull market.

### 2.3 Signal Construction

The signal is a direct adaptation of the paper's prototype commodity momentum factor:

For each symbol *i* on each date *t*, we compute the trailing 252-day (approximately 12-month) price return:

```
signal_i(t) = close_i(t-1) / close_i(t-1-252) - 1
```

All lookbacks are lagged by one day (`shift(1)`) to prevent same-bar leakage.

### 2.4 Portfolio Construction

We apply **absolute momentum** (time-series momentum): each asset is held long if its trailing return is positive, and excluded otherwise. The portfolio is **equal-weighted** across all selected assets and **rebalanced monthly** (every 21 trading days). This matches the paper's prototype construction as closely as possible.

### 2.5 Execution and Costs

- **Timing**: Model-B -- signal computed at close(t), executed at open(t+1).
- **Returns**: Open-to-close daily returns.
- **Transaction costs**: 20 basis points round-trip (10 bps exchange fee + 10 bps estimated slippage).
- **Annualization**: 365 days (crypto markets trade continuously).

### 2.6 Benchmark

BTC buy-and-hold, normalized to start at 1.0 on the strategy inception date (January 2018).

## 3. Results

### 3.1 Performance Summary

**Table 1: Prototype 252d Momentum Factor vs. BTC Benchmark (Jan 2018 -- Dec 2025)**

| Metric | TSMOM 252d (Long-Only) | BTC Buy & Hold |
|:---|---:|---:|
| CAGR | -34.3% | +26.3% |
| Volatility (ann.) | 81.6% | 66.1% |
| Sharpe Ratio | -0.09 | 0.69 |
| Sortino Ratio | -0.11 | 0.92 |
| Calmar Ratio | -0.35 | -- |
| Maximum Drawdown | -97.4% | -81.4% |
| Hit Rate (daily) | 45.1% | 51.0% |
| Skewness | -0.66 | -0.31 |
| Excess Kurtosis | 8.38 | 8.55 |

**Table 2: Portfolio Characteristics**

| Metric | Value |
|:---|---:|
| Average gross exposure | 85.2% |
| Average daily turnover | 3.14% |
| Average number of holdings | 15.9 |
| Lookback window | 252 days |
| Rebalance frequency | 21 days (monthly) |

### 3.2 Equity Curve and Drawdown

![Step 1 Baseline Momentum](../../artifacts/research/jpm_momentum/step_01/step_01_baseline_momentum.png)

The equity curve (top panel, log scale) reveals near-total capital destruction over the sample period, with the strategy losing 96.5% of its starting value. The strategy is in drawdown for substantially the entire sample, reaching a maximum drawdown of -97.4% -- materially worse than BTC's own -81.4% drawdown during the same period.

The drawdown panel (middle) shows that the strategy never meaningfully recovers from the bear markets of 2018--2019 and 2022. While BTC recovers to new highs by late 2020 and again by late 2024, the momentum portfolio's capital base is too depleted to benefit.

### 3.3 Holdings Count

The number of holdings (bottom panel) provides structural insight into the failure. During the 2024 bull market, the universe expansion caused the strategy to hold over 130 long positions simultaneously. Many of these are small-capitalization altcoins in the late stages of speculative runs -- assets with positive trailing 252-day returns that are about to reverse sharply. Equal-weighting across this many assets compounds the problem, as the portfolio is overexposed to the weakest names.

## 4. Analysis

### 4.1 Comparison to the Original Paper

The paper's commodity momentum factor delivered a Sharpe ratio of 0.49 and a CAGR of 7.3% over 1972--2014 using the same 12-month lookback methodology (Paper Table 1, p.10). Our crypto adaptation produced a Sharpe of -0.09 -- a sign reversal. This is the single largest divergence we observe relative to the paper's findings and demands explanation.

### 4.2 Why 252-Day Momentum Fails in Crypto

We identify three structural differences between commodity futures and crypto spot markets that account for this failure:

**1. Faster trend cycles.** Cryptocurrency markets exhibit significantly shorter trend durations than commodities. Bull and bear cycles in crypto are measured in months, not years. A 252-day lookback is too slow: by the time a positive trailing return is established, the trend is often exhausted or reversing. This is precisely the "turning point risk" the paper describes (p.12--13), but compressed into a much tighter timeline.

**2. Fat-tailed altcoin distribution.** Unlike commodities, where the universe is limited to approximately 20 liquid futures contracts, the crypto universe contains hundreds of tokens with extreme return dispersion. Many altcoins experience brief parabolic rallies followed by permanent capital loss. An equal-weighted portfolio of everything with a positive 252-day return indiscriminately loads on these assets.

**3. Absence of natural mean-reversion at the asset level.** Commodity futures prices revert toward production costs; crypto tokens have no equivalent fundamental anchor. Assets that have appreciated 10x in a year may appreciate another 5x or decline 99%. The 12-month return signal contains almost no information about the direction of the *next* month in this environment.

### 4.3 Negative Skewness and Tail Risk

The strategy exhibits a skewness of -0.66, substantially more negative than BTC's -0.31. This confirms the paper's general observation that momentum strategies carry negative skewness (p.12), but at a severity level that eliminates the positive return premium that is supposed to compensate for it. The excess kurtosis of 8.38 (comparable to BTC's 8.55) indicates heavy tails in both directions, but given the negative mean return, the left tail dominates.

## 5. Implications for Subsequent Research

This baseline result establishes that a direct translation of the paper's commodity momentum methodology to crypto produces a strongly negative outcome. However, this does not imply that momentum is absent in digital assets. Recent academic research (Hsiao & Qi 2024; Patel 2024) documents significant time-series momentum at shorter horizons, and our own codebase already implements successful trend-following strategies at the 5--40 day timescale.

The failure is specific to the **252-day lookback** and the **"buy everything positive"** absolute momentum approach applied to a large, unfiltered universe. The paper's subsequent chapters address exactly these issues:

- **Chapter 2 (p.20--24): Lookback sensitivity.** We expect shorter lookback windows (5--63 days) to capture crypto's faster trend cycles before they reverse. This is the subject of Step 2.

- **Chapter 2 (p.25--29): Relative momentum.** Ranking assets and selecting only the top quintile (rather than all positive-return assets) should mitigate the altcoin graveyard problem. This is Step 3.

- **Chapter 2 (p.36--41): Signal type selection.** Moving average crossovers and breakout channels adapt to trends faster than raw trailing returns, and may produce better entry/exit timing. This is Step 4.

- **Chapter 3 (p.56--65): Risk management.** Stop-loss overlays and volatility targeting may prevent the catastrophic drawdowns observed here. This is Step 6.

## 6. Conclusion

The 252-day absolute momentum factor, which generates a Sharpe ratio of 0.49 in commodities, produces a Sharpe of -0.09 in crypto spot markets over 2018--2025. The failure is attributable to crypto's faster trend cycles, extreme altcoin dispersion, and the absence of fundamental mean-reversion anchors. This result motivates the parameter optimization and risk management work that follows in Chapters 2 and 3.

---

**Data sources**: Coinbase 1-minute candles (market.duckdb), 351 USD pairs, Jan 2017 -- Dec 2025.
**Artifacts**: `artifacts/research/jpm_momentum/step_01/`
