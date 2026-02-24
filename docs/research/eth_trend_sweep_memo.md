---

# MEMO: ETH-USD Trend Signal Exhaustive Sweep

### Single-Asset Signal Discovery — Results & Recommended Path Forward

**NRT Research** | February 2026

**To:** Desk Head  
**From:** Russell  
**Re:** Why simple trend signals on individual assets dominate our TSMOM portfolio framework

---

## Context

Following the TSMOM portfolio experiment (VOL_SCALED 21d, Sharpe 0.77, skewness 0.40, MaxDD -27%),
you correctly observed that simple moving average crossovers on individual assets should
outperform the entire framework. We ran an exhaustive validation of that hypothesis.

This memo reports results from **13,294 configurations** tested on ETH-USD across three
frequencies (daily, 4-hour, 1-hour), covering every major trend signal family with full
parameter sweeps and two classes of trailing stop: fixed-percentage and ATR-based (vol-adaptive).

## Methodology

- **Asset:** ETH-USD, daily OHLCV, 2017-01-02 to 2026-02-13 (3,330 trading days)
- **Signal universe:** 494 base signals across 30+ signal families (SMA/EMA/DEMA/Hull crossovers,
  Supertrend, Aroon, ADX, Donchian, Bollinger, Keltner, MACD, RSI, CCI, momentum, linear
  regression, Ichimoku, Heikin-Ashi, Turtle, Kaufman efficiency, VWAP, and composites)
- **Frequencies:** Signals computed on 1d, 4h, 1h bars; P&L evaluated at daily close
- **Position:** Binary long/cash. Signal > 0 = long at next close. Signal ≤ 0 = cash.
- **Stops:** None, fixed trailing (5%, 10%, 20%), vol-adaptive trailing (1.5x, 2x, 2.5x, 3x, 4x
  entry-date 14d ATR from high-water mark)
- **Costs:** 20 bps round-trip
- **No leverage, no vol targeting, no portfolio construction**

## Buy & Hold Baseline

| Sharpe | CAGR | Max DD | Calmar | Skewness | Time in Market |
|---|---|---|---|---|---|
| 1.11 | 82.7% | **-94.0%** | 0.88 | 0.37 | 100% |

ETH has been an exceptional asset — Sharpe above 1 over nine years of daily data. But the
drawdown is disqualifying for any institutional mandate. The question is whether trend signals
can capture the CAGR while compressing the left tail.

## Key Result #1: Simple signals massively outperform the TSMOM framework

| Strategy | Sharpe | CAGR | Max DD | Calmar | Skewness | TIM |
|---|---|---|---|---|---|---|
| TSMOM VT10 (portfolio) | 0.77 | 10.2% | -27.3% | 0.37 | 0.40 | 87% |
| TSMOM VT15 (portfolio) | 0.75 | 12.0% | -38.9% | 0.31 | 0.29 | 87% |
| EMA 5/150, 4h, ATR 3.0 | **1.72** | **146%** | -47.1% | **3.10** | **2.03** | 45% |
| Supertrend 7/2.0, ATR 1.5 | 1.55 | 95% | **-34.8%** | **2.73** | **2.60** | 30% |
| Aroon 20, daily | 1.71 | 157% | -56.5% | 2.77 | 1.46 | 52% |

The TSMOM framework destroyed value. The portfolio construction layer (vol-scaling the signal,
portfolio-level vol targeting, binary sizing with weight caps) smoothed away the edge that
trend signals naturally have in crypto: speed of exit. A raw binary crossover on a single asset
with a trailing stop beats the portfolio framework by a factor of 2x on Sharpe and 5x+ on
skewness.

## Key Result #2: ATR-based stops dominate fixed-percentage stops

This is the most important finding for deployment. Across all signals, vol-adaptive stops
produce better risk-adjusted returns than fixed-percentage stops at equivalent tightness.

**EMA 5/150 on 4h bars — stop comparison:**

| Stop | Type | Sharpe | CAGR | Max DD | Calmar | Skewness | TIM |
|---|---|---|---|---|---|---|---|
| none | — | 1.64 | 146% | -50.3% | 2.91 | 1.43 | 52% |
| 10% | fixed | 1.03 | 48% | -47.1% | 1.01 | 1.41 | 38% |
| 20% | fixed | 1.43 | 103% | -47.1% | 2.19 | 1.24 | 49% |
| **ATR 2.0** | **vol** | **1.62** | **119%** | **-47.1%** | **2.53** | **2.21** | **40%** |
| **ATR 3.0** | **vol** | **1.72** | **146%** | **-47.1%** | **3.10** | **2.03** | **45%** |

At the same time-in-market (~40%), the ATR 2.0 stop produces Sharpe 1.62 vs the fixed 10%
stop at 1.03. The mechanism: in low-vol regimes, a fixed 10% stop is too loose (normal noise
doesn't trigger it, so it adds nothing). In high-vol regimes, it's too tight (normal noise
constantly triggers it, chopping you in and out). The ATR stop adapts — tight when vol is low,
wide when vol is high. It exits on statistically abnormal moves, not arbitrary thresholds.

**Supertrend 7/2.0 on daily bars — the convexity case:**

| Stop | Type | Sharpe | CAGR | Max DD | Calmar | Skewness | TIM |
|---|---|---|---|---|---|---|---|
| none | — | 1.60 | 134% | -50.6% | 2.65 | 1.69 | 48% |
| 5% | fixed | 1.07 | 39% | -51.8% | 0.75 | 3.28 | 18% |
| **ATR 1.5** | **vol** | **1.55** | **95%** | **-34.8%** | **2.73** | **2.60** | **30%** |

The ATR 1.5 stop cuts max drawdown from -51% to **-35%** while maintaining Sharpe above 1.5.
The fixed 5% stop at similar TIM has Sharpe of just 1.07. The vol-adaptive stop preserves the
trend-riding capacity that the fixed stop destroys.

## Key Result #3: Aggregate statistics across 13,294 configurations

| Stop | Type | Med Sharpe | Med MaxDD | Med Skew | Med TIM |
|---|---|---|---|---|---|
| none | — | 0.87 | -74.9% | 1.61 | 51% |
| pct5 | fixed | 0.67 | -68.4% | 2.06 | 30% |
| pct10 | fixed | 0.74 | -72.3% | 1.66 | 39% |
| pct20 | fixed | 0.84 | -72.3% | 1.57 | 49% |
| **atr1.5** | **vol** | **0.78** | **-69.7%** | **1.75** | **35%** |
| **atr2.0** | **vol** | **0.77** | **-71.7%** | **1.65** | **39%** |
| **atr3.0** | **vol** | **0.81** | **-72.0%** | **1.69** | **43%** |
| B&H | — | 1.11 | -94.0% | 0.37 | 100% |

Across all 1,477 base signals, the ATR stops consistently produce higher median Sharpe than
fixed stops at equivalent TIM. The ATR 3.0 stop (median Sharpe 0.81, TIM 43%) dominates the
fixed 10% stop (median Sharpe 0.74, TIM 39%) — more time in market, more Sharpe, comparable
skewness.

576 of 1,477 no-stop configs beat Buy & Hold on Sharpe. The median skewness across all configs
(1.61) is 4x higher than Buy & Hold (0.37). Positive skewness is nearly universal in binary
long/cash trend-following on crypto — the convexity mandate is naturally satisfied by the
structure of the trade.

## Top 5 Candidates for Deployment Review

Ranked by the combination of Sharpe, drawdown, and skewness:

| # | Signal | Stop | Freq | Sharpe | CAGR | MaxDD | Calmar | Skew | TIM |
|---|---|---|---|---|---|---|---|---|---|
| 1 | EMA 5/150 | ATR 3.0 | 4h | 1.72 | 146% | -47% | 3.10 | 2.03 | 45% |
| 2 | Aroon 20 | pct20 | 1d | 1.76 | 158% | -57% | 2.79 | 1.53 | 51% |
| 3 | Supertrend 7/2.0 | ATR 1.5 | 1d | 1.55 | 95% | -35% | 2.73 | 2.60 | 30% |
| 4 | ADX 10/15 | pct20 | 1d | 1.68 | 140% | -49% | 2.89 | 1.69 | 45% |
| 5 | MomThresh 20/10% | pct20 | 1d | 1.66 | 126% | -46% | 2.78 | 2.13 | 33% |

If the mandate is maximum convexity (highest skewness, lowest drawdown, comfortable with lower
CAGR): **Supertrend 7/2.0 with ATR 1.5 stop** — Sharpe 1.55, MaxDD -35%, skewness 2.60,
cash 70% of the time.

If the mandate is maximum risk-adjusted return: **EMA 5/150 with ATR 3.0 stop** on 4h bars —
Sharpe 1.72, Calmar 3.10, highest of any configuration tested.

## What This Means for the Research Program

### What was wrong with the TSMOM framework

1. **Vol-scaling the signal dampened exit speed.** In high-vol crash regimes — exactly when you
   need the fastest possible exit — the vol-normalized signal magnitude shrank, delaying the
   exit by weeks.

2. **Portfolio-level vol targeting further smoothed the signal.** When vol spiked, the portfolio
   overlay reduced exposure. But by then the drawdown had already occurred.

3. **The 21-day lookback with signal-reversal exit was too slow.** It took 34 days to exit the
   May 2021 crash. A simple EMA crossover with an ATR trailing stop exits in days.

4. **Multi-asset diversification added complexity without edge.** The portfolio framework spread
   across many assets, but in crypto, correlations spike to 1.0 in crashes. Diversification
   provided no protection when it mattered.

### What the data says the architecture should be

The evidence points to a simple, per-asset trend engine:

- **Entry:** EMA crossover or Supertrend on individual assets
- **Exit:** Vol-adaptive trailing stop (ATR-based), not signal reversal
- **Sizing:** Binary long or cash
- **Frequency:** Daily or 4h bars
- **Portfolio:** Run independently per asset, aggregate at the book level

The sophistication should go into **which assets to run** (universe selection) and **how to
size across assets** (risk budgeting), not into the signal or exit logic. The signals themselves
should be as simple as possible.

## Caveats & Next Steps

**This is a single-asset, in-sample analysis.** All 13,294 configs were tested on the same
ETH-USD history. The results are maximally overfitted. Before any deployment:

1. **Cross-asset validation.** Run the top 5 candidates on BTC-USD, SOL-USD, and at least 3
   other assets. If the signal family (not the exact parameters) works across assets, we have
   something real.

2. **Walk-forward.** Train on 2017-2022, evaluate on 2023-2025. The 2022 crypto winter is the
   real stress test.

3. **Transaction cost sensitivity.** We used 20 bps. The 4h EMA cross with ATR 3.0 makes 108
   round-trip trades over 9 years (~12/year). At 20 bps that's manageable, but verify against
   actual execution costs.

4. **Regime analysis.** Conditional correlation to BTC in bear markets. The single-asset
   results don't tell us whether the strategy decouples when it matters.

5. **Ensemble.** If multiple signal families (EMA cross, Supertrend, Aroon) independently work,
   an equal-weight ensemble may be more robust than any single signal.

I recommend proceeding to cross-asset validation immediately. If EMA 5/150 with ATR 3.0 and
Supertrend 7/2.0 with ATR 1.5 both work on BTC and SOL, we have a deployable strategy
architecture.

---

*Data: Coinbase Advanced spot OHLCV. 2017–2026. Costs: 20 bps round-trip. No leverage.*
