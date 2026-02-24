# NRT Alternative Thinking 2026 Issue 1

# Follow the Trend? What 13,000 Crypto Strategies Actually Tell Us

**Portfolio Research Group**

---

## Executive Summary

"Follow the Trend" has become the working hypothesis for systematic crypto allocation at this
desk. In this article, we test that hypothesis exhaustively — building **13,293 trend-following
configurations** on ETH-USD and asking a simple question: does trend beat buy-and-hold?

The short answer is: mostly no. **10,935 of 13,293 (82%)** trend
strategies produce *worse* risk-adjusted returns than passive buy-and-hold. The median strategy
has a Sharpe ratio of 0.78, versus buy-and-hold's 1.11 — a -30%
degradation. On CAGR, it's worse: only 894 of 13,293 (6.7%) beat
buy-and-hold, and the median strategy surrenders 53% of annual return.[^1]

But this is only half the story. 11,031 of 13,293 (83%) strategies have
*shallower* max drawdowns than buy-and-hold. And 13,073 of 13,293
(98%) exhibit positive skewness. In an asset that fell 94%
peak-to-trough, drawdown compression is not a trivial benefit. The question is not whether
trend "works" — it is whether the protection it buys is worth the return it costs.

[^1]: A disclaimer is necessary: we are testing 13,293 strategies on a single asset over a
nine-year period. No matter how significant a result appears, it reflects in-sample data mining
until validated out-of-sample and across assets. We would take very little risk on any single
configuration, preferring to diversify across many — and even then, no directional strategy
is anywhere near perfect.

---

## Contents

1. Introduction
2. Part 1: The Baseline — Buy-and-Hold Is Extremely Hard to Beat
3. Part 2: What Trend Buys and What It Costs
4. Part 3: What Actually Drives Performance? Frequency Dominates Everything
5. Part 4: Do Vol-Adaptive Stops Beat Fixed Stops?
6. Part 5: The Multiple Testing Problem
7. Concluding Thoughts
8. Appendix: Parameter Grid and Data Notes

---

## Introduction

Crypto allocators face a unique problem. The asset class has delivered extraordinary long-term
returns — ETH-USD compounded at 83% annualized from 2017 to 2026 — but the path
was brutal: a -94% peak-to-trough drawdown, with multiple drawdowns exceeding 70%.
No investor, institutional or otherwise, can plausibly hold through a -94% drawdown.[^2]
The standard response is to apply trend-following logic: be long when the asset is trending
up, move to cash when it is not.

The premise has theoretical support. Moskowitz, Ooi, and Pedersen (2012) documented time-series
momentum across dozens of futures markets. Hurst, Ooi, and Pedersen (2017) extended the evidence
to a century of data. In our own prior work at this desk, we attempted a portfolio-level TSMOM
framework for crypto, applying vol-scaled signals with portfolio-level vol targeting across
multiple assets. The results were poor: the framework produced a Sharpe of 0.77 with 87%
time-in-market — essentially dampened buy-and-hold at a fraction of the CAGR.[^3]

This led to a natural question: if portfolio-level momentum fails, do simpler per-asset
trend signals do better? And if so, what matters more — the entry signal, the data frequency,
or the exit mechanism?

To answer these questions, we built **13,293 configurations** from the cross-product of:

| Variable | What We Test |
|---|---|
| Base signals | 493 signals from 30+ families (MA crossovers, channel breakouts, momentum indicators, TA oscillators, composite signals) |
| Frequencies | Daily, 4-hour, 1-hour bars |
| Stop variants | None; fixed trailing stops at 5%, 10%, 20%; vol-adaptive trailing stops at 1.5×, 2.0×, 2.5×, 3.0×, 4.0× entry-date ATR |

All configurations use identical backtest rules: binary long or cash, signal applied with
one-bar lag, 20 bps round-trip transaction costs, no leverage, no position sizing.

[^2]: Luna Foundation Guard, Three Arrows Capital, and Alameda Research all failed to maintain
positions through drawdowns of comparable magnitude. The behavioral and institutional constraints
on holding through -90%+ drawdowns are not merely theoretical.

[^3]: See internal memo, "TSMOM Framework Results," January 2026. The framework actively
destroyed value by slowing exit timing through vol-scaling and portfolio construction.

---

## Part 1: The Baseline — Buy-and-Hold Is Extremely Hard to Beat

Before evaluating trend strategies, it is worth establishing how strong the baseline is.
ETH-USD's buy-and-hold performance from January 2017 to February 2026:

| Metric | Value |
|---|---|
| CAGR | 82.7% |
| Sharpe Ratio | 1.11 |
| Sortino Ratio | 1.63 |
| Max Drawdown | -94.0% |
| Calmar Ratio | 0.88 |
| Skewness | 0.36 |

A Sharpe ratio of 1.11 is exceptional by any standard. In equities, a Sharpe above
0.5 is considered good; in crypto, the secular uptrend and the absence of a reliable risk-free
rate produce much higher ratios. Any strategy that sits in cash for part of the period faces
a substantial headwind from missing this strong secular drift.[^4]

**Exhibit 1** shows the distribution of Sharpe ratios across all 13,293 trend configurations.
The median strategy (0.78) falls well below buy-and-hold (1.11). Only
2,358 (18%) of configurations outperform on this metric.

![Exhibit 1](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_1_sharpe_dist.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results.*

This result should not be surprising. The median strategy is invested only 51%
of the time. In an asset with 83% CAGR, sitting in cash half the time is expensive.
Many configurations are also suboptimal by design (very short lookbacks that trade noise, very
tight stops that exit normal volatility) — the purpose of an exhaustive sweep is to map the full
space, including the bad regions.

[^4]: Throughout this piece, "cash" means zero return. We do not model stablecoin yield. To the
extent that cash earns a positive return (e.g., 5% in a DeFi lending protocol), the case for
trend strategies improves modestly.

---

## Part 2: What Trend Buys and What It Costs

If trend-following in crypto mostly underperforms buy-and-hold on a risk-adjusted basis, why
consider it? Because risk-adjusted returns are not the only thing that matters. An investor
who cannot hold through a -94% drawdown does not earn the 83% CAGR.
The relevant question is: what does trend cost, and what does it buy?

**Exhibit 2** shows the tradeoff directly: the left panel shows the CAGR distribution (what you
give up), and the right panel shows the drawdown distribution (what you get).

![Exhibit 2](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_2_tradeoff.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results.*

The numbers are stark:

| Metric | Buy & Hold | Median Strategy | Difference |
|---|---|---|---|
| CAGR | 82.7% | 30.1% | -52.6% |
| Max Drawdown | -94.0% | -72.2% | +21.9% |
| Skewness | 0.36 | 1.67 | +1.31 |

The median strategy gives up roughly 53% of annual return to compress
the max drawdown by 22%. Is that a good trade? It depends on
the investor's utility function. For an allocator who cannot hold through -94% but
can hold through -72%, trend converts an undeployable return stream into a deployable
one — even if the headline CAGR is lower.[^5]

The skewness improvement is real but partly mechanical. A binary long/cash strategy on a
strongly trending asset will exhibit positive skewness by construction: it participates in the
large up-moves (which are persistent and often occur in trend) while exiting before or during
some of the large down-moves. This does not require signal "skill" — 13,073
of 13,293 (98%) strategies exhibit positive skewness regardless
of signal choice. The skewness is a property of the *trade structure*, not the signal.[^6]

[^5]: This framing is consistent with the crypto-specific finding in our prior work that the
binding constraint is not expected return (which is high) but the ability to stay allocated
through drawdowns.

[^6]: Readers should be cautious about attributing positive skewness to signal "alpha."
A randomly-timed long/cash strategy on ETH-USD would also exhibit positive skewness over
this period, simply because the right tail of ETH daily returns is fatter than the left tail.

---

## Part 3: What Actually Drives Performance? Frequency Dominates Everything

Across 13,293 configurations, we vary three dimensions: signal choice (493 base signals),
frequency (daily, 4-hour, 1-hour), and stop type (9 variants). Which dimension matters most?

The answer is unambiguous: **frequency**.

**Exhibit 3** shows the Sharpe ratio distribution at each frequency. Daily signals have a
median Sharpe of 1.00 and 36%
beat buy-and-hold. Four-hour signals drop to 0.84
(16% beat B&H). One-hour signals collapse
to 0.42 (1% beat B&H).

![Exhibit 3](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_3_frequency.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results.*

**Exhibit 3a: Performance by Frequency**

| Frequency | N | Median Sharpe | Median CAGR | Median MaxDD | % Beat B&H (SR) |
|---|---|---|---|---|---|
| Daily | 4,437 | 0.996 | 44.3% | -56.0% | 36% |
| 4-Hour | 4,428 | 0.845 | 38.0% | -73.7% | 16% |
| 1-Hour | 4,428 | 0.423 | 6.7% | -91.1% | 1% |

The mechanism is straightforward. Higher-frequency signals generate more trades, and each trade
incurs transaction costs (20 bps round-trip). In a strong secular uptrend, frequent trading
also increases the probability of being whipsawed out of a trend during intraday noise, then
missing the subsequent continuation. Daily signals trade less often and allow trends to develop;
hourly signals chop in and out of positions, destroying return through friction and missed
upside.[^7]

Within the daily-frequency universe, **Exhibit 4** shows median Sharpe by signal family. The
spread is modest: the best family (EMA crossover, median Sharpe 1.46)
outperforms the worst by roughly 0.5 Sharpe units. But the variation *within* families
(across parameter choices) is often as large as the variation *across* families. This suggests
that signal choice, while not irrelevant, is secondary to frequency and time-in-market.

![Exhibit 4](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_4_family.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results. Shows daily-frequency, no-stop configurations only. Families with fewer than
3 configurations are excluded.*

**Exhibit 5** confirms that time-in-market is the core dial. More time invested means higher
CAGR but deeper drawdowns — there is no free lunch.

![Exhibit 5](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_5_tim.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results.*

**Exhibit 5a: Performance by Time-in-Market Bucket**

| TIM Bucket | N | Median Sharpe | Median MaxDD | Median Skew |
|---|---|---|---|---|
| 0%–20% | 3,424 | 0.470 | -43.0% | 6.38 |
| 20%–30% | 1,291 | 0.925 | -57.9% | 2.51 |
| 30%–40% | 1,569 | 0.996 | -62.9% | 2.07 |
| 40%–50% | 1,916 | 0.969 | -72.8% | 1.57 |
| 50%–60% | 4,756 | 0.781 | -87.0% | 1.53 |
| 60%–100% | 337 | 0.761 | -92.7% | 0.55 |

The "sweet spot" appears to be 30–40% time-in-market: high enough to capture most of the
secular drift, low enough to exit during sustained drawdowns. Strategies in this bucket have
a median Sharpe near 1.00
and a median drawdown of -62.9%
— roughly one-third less severe than buy-and-hold.

[^7]: This finding is consistent with the broader trend-following literature. Moskowitz et al
(2012) found that time-series momentum is strongest at monthly frequencies and degrades at
shorter horizons due to mean reversion and transaction costs.

---

## Part 4: Do Vol-Adaptive Stops Beat Fixed Stops?

A common hypothesis in systematic trading is that vol-adaptive exits (e.g., ATR-based trailing
stops) should dominate fixed-percentage exits because they adapt to the prevailing volatility
regime. In theory, a fixed 10% stop is too tight in a high-volatility environment and too
loose in a low-volatility one; an ATR-based stop calibrates automatically.

We test this by comparing nine stop variants across all base signals: no stop, three fixed-
percentage stops (5%, 10%, 20%), and five ATR-based stops (1.5×, 2.0×, 2.5×, 3.0×, 4.0× ATR).

**Exhibit 7** shows the aggregate medians by stop type.

![Exhibit 7](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_7_stops_agg.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results.*

**Exhibit 7a: Aggregate Performance by Stop Type**

| Stop | Type | Med Sharpe | Med MaxDD | Med Skew | Med TIM |
|---|---|---|---|---|---|
| None | — | 0.869 | -74.9% | 1.61 | 51% |
| 5% fixed | Fixed % | 0.672 | -68.4% | 2.06 | 30% |
| 10% fixed | Fixed % | 0.735 | -72.3% | 1.66 | 39% |
| 20% fixed | Fixed % | 0.835 | -72.3% | 1.57 | 49% |
| 1.5× ATR | Vol-adaptive | 0.776 | -69.7% | 1.75 | 35% |
| 2.0× ATR | Vol-adaptive | 0.774 | -71.7% | 1.65 | 39% |
| 2.5× ATR | Vol-adaptive | 0.792 | -72.4% | 1.65 | 40% |
| 3.0× ATR | Vol-adaptive | 0.814 | -72.0% | 1.69 | 43% |
| 4.0× ATR | Vol-adaptive | 0.814 | -72.6% | 1.66 | 45% |

At the aggregate level, both stop types compress drawdowns relative to no-stop, but at the
cost of lower Sharpe ratios. The tightest stops (5% fixed, 1.5× ATR) produce the most drawdown
compression but also the worst Sharpe ratios, because they trigger exits on normal volatility
and generate excessive trading costs.

The more interesting test is the **matched comparison**: for the same base signal and frequency,
does the best ATR stop outperform the best fixed stop? **Exhibit 8** shows the result.

![Exhibit 8](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_8_matched.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results. Each point represents one base signal × frequency pair. Above the diagonal
line, ATR outperforms fixed %.*

The answer is not what we expected. **ATR stops win only 31% of the time on Sharpe**
and 32% of the time on max drawdown, across 1,477 matched pairs. The
average Sharpe difference is -0.0116
— essentially zero.[^8]

The aggregate medians in Exhibit 7 were misleading because ATR and fixed stops have different
time-in-market profiles (ATR stops tend to keep positions open slightly longer), which confounds
the comparison. Once we match on the same base signal, the stop type is a wash.

This is an honest but uncomfortable finding. The economic intuition for vol-adaptive stops is
compelling, but the data do not support a strong claim of dominance over this sample. Both
stop types achieve roughly the same thing: they compress drawdowns at the cost of CAGR, with
the compression proportional to how tight the stop is.[^9]

[^8]: The matched comparison uses the *best* ATR and *best* fixed stop for each signal. This
is generous to both; a random stop selection would show even less differentiation.

[^9]: One interpretation is that in a single-asset context with binary positioning, the stop
distance is more important than how it is calibrated. An ATR stop that happens to produce a
similar stop distance to a fixed-% stop will produce similar results. The theoretical advantage
of ATR may require a more diverse asset universe or more complex position sizing to manifest.

---

## Part 5: The Multiple Testing Problem

We tested 13,293 configurations. Even if every strategy were generated by a coin flip with zero
true Sharpe, we would expect some to look impressive by chance alone. Any honest assessment of
these results must address the multiple testing problem.[^10]

The strategies are not independent — stop variants of the same base signal are highly correlated
(average pairwise Sharpe correlation: 0.94). We estimate the effective number of independent
tests at approximately 1,479 (493 base signals × 3 frequencies), treating stop variants
as dependent.

At this test count, a Bonferroni-corrected significance threshold requires a z-statistic of
4.15. Over our 9-year sample, this
translates to a Sharpe ratio of **1.38**. Only **534 of 13,293
(4.0%)** strategies survive this threshold.

**Exhibit 9** shows which strategies survive.

![Exhibit 9](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_9_multiple_testing.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results. Bonferroni correction assumes 1,479 effective independent tests (493 base
signals × 3 frequencies). The 9-year sample converts z-thresholds to Sharpe
thresholds via SR = z / √T.*

**Exhibit 9a: Top 10 Strategies Surviving Bonferroni Correction**

| # | Signal | Stop | Freq | Sharpe | CAGR | MaxDD | Calmar | Skew | TIM |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Aroon_20 | pct20 | 1d | 1.75 | 157.7% | -56.5% | 2.79 | 1.53 | 51% |
| 2 | EMA_cross_5_150 | atr3.0 | 4h | 1.72 | 146.0% | -47.1% | 3.10 | 2.03 | 45% |
| 3 | Aroon_20 | none | 1d | 1.71 | 156.7% | -56.5% | 2.77 | 1.46 | 52% |
| 4 | EMA_cross_7_150 | atr3.0 | 4h | 1.68 | 139.6% | -50.3% | 2.78 | 2.02 | 44% |
| 5 | ADX_10_15 | pct20 | 1d | 1.68 | 140.4% | -48.6% | 2.89 | 1.69 | 45% |
| 6 | MomThresh_20_0.1 | pct20 | 1d | 1.66 | 126.4% | -45.5% | 2.78 | 2.12 | 33% |
| 7 | Supertrend_10_2.5 | none | 1d | 1.66 | 143.9% | -52.7% | 2.73 | 1.46 | 49% |
| 8 | MomThresh_20_0.05 | pct20 | 1d | 1.65 | 133.9% | -51.7% | 2.59 | 1.85 | 40% |
| 9 | DEMA_cross_30_50 | pct20 | 1d | 1.65 | 134.7% | -52.1% | 2.59 | 1.58 | 44% |
| 10 | DEMA_cross_5_50 | atr1.5 | 1d | 1.65 | 115.3% | -48.9% | 2.36 | 2.48 | 36% |

The survivors cluster in daily-frequency, medium-lookback MA crossovers (EMA, DEMA, SMA)
and a few channel-based signals (Aroon, Supertrend, ADX). This is reassuring in one sense —
the surviving signal families are well-established in the trend-following literature — but
concerning in another: the specific parameterizations that survive are almost certainly
influenced by the particular path of ETH over this sample.[^11]

[^10]: AQR's analysis of 196 "Buy the Dip" strategies (Cao, Chong, and Villalon, 2025) faces
a similar challenge with far fewer tests. With 13,293 strategies, the concern is proportionally
more severe.

[^11]: For context: if we were to run the same sweep on BTC-USD or SOL-USD, the specific
winning parameterizations would likely differ, even if the winning signal *families* remain
similar. This is the distinction between signal-family robustness (which we hypothesize)
and parameter robustness (which we do not claim).

---

## Part 6: The Convexity Profile

For allocators whose mandate is long convexity — bounded downside with exposure to unbounded
upside — the joint distribution of Sharpe and skewness matters more than Sharpe alone.

**Exhibit 6** maps this space.

![Exhibit 6](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_6_sharpe_skew.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results.*

Nearly all strategies (13,073 of 13,293, or
98%) exhibit positive skewness. As noted earlier, this is
largely a mechanical consequence of binary long/cash positioning on a positively-trending
asset. The practical implication is that trend strategies in crypto are natural convexity
providers regardless of signal choice — a structural property that may justify their inclusion
in a portfolio even at a modestly lower Sharpe than buy-and-hold.

**Exhibit 10** shows the full risk/return map.

![Exhibit 10](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_10_cagr_dd.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results. Dotted lines show Calmar ratio contours.*

**Exhibit 11** shows the drawdown distribution by stop type.

![Exhibit 11](../../artifacts/research/tsmom/eth_trend_sweep/exhibit_11_dd_box.png)

*Source: NRT Research. Coinbase Advanced spot OHLCV, ETH-USD, January 2017 – February 2026. All strategies are hypothetical, binary long/cash, one-bar lag, net of 20 bps round-trip transaction costs, no leverage. Past performance is not a reliable indicator of future results.*

---

## Concluding Thoughts

Four findings emerge from this study:

**First, trend-following in crypto mostly underperforms buy-and-hold on risk-adjusted returns.**
82% of the 13,293 strategies we tested produced lower Sharpe ratios than
passive buy-and-hold. The median strategy has a Sharpe of 0.78 versus buy-and-hold's
1.11. This is not an indictment of trend-following — it is a statement about how
strong the secular uptrend in crypto has been. In a 83% CAGR environment, any time
spent in cash is expensive.

**Second, the value of trend is in drawdown compression, not return enhancement.** 83%
of strategies have shallower drawdowns than buy-and-hold's -94%. The median
strategy compresses the max drawdown by 22%. For allocators
constrained by drawdown tolerance, this converts an undeployable return stream into a
deployable one — even at lower headline CAGR.

**Third, data frequency dominates signal choice and exit mechanism.** Daily signals vastly
outperform intraday signals. Within the daily universe, the choice of signal family and stop
type matters far less than the choice of frequency. Vol-adaptive (ATR-based) stops do not
reliably dominate fixed-percentage stops in a matched comparison.

**Fourth, after multiple-testing correction, only 4.0% of strategies survive.**
The 534 surviving configurations cluster in well-known trend signals at daily
frequency. We hypothesize that the signal *family* result (MA crossovers, channel breakouts)
may be robust across assets, but the specific *parameterizations* are almost certainly
sample-dependent. Cross-asset validation is required before deployment.

A final note on interpretation. The fact that most trend strategies underperform buy-and-hold
in crypto does *not* mean trend-following is useless. It means the secular uptrend is so strong
that the opportunity cost of being in cash — even temporarily — is enormous. In a lower-drift
environment (equities, commodities, or a future crypto regime with lower secular returns),
the calculus shifts in trend's favor. The historical crypto drift is an anomaly, not a steady
state, and strategies should be evaluated against a range of possible futures, not just the
most favorable past.[^12]

[^12]: This is analogous to AQR's observation that "Buy the Dip" strategies appear to work in
recent data primarily because equities have gone up a lot — not because the timing adds value.
Similarly, many crypto trend strategies "work" primarily because ETH has gone up 243×
over this period — not because the trend signal adds timing value.

---

## Appendix: Parameter Grid and Data Notes

**Data**: Coinbase Advanced spot OHLCV, ETH-USD. January 1, 2017 – February 22, 2026.
Daily, 4-hour, and 1-hour bars. Cached locally from DuckDB.

**Base signals (493)**: SMA crossover (7 fast × 7 slow), EMA crossover, DEMA crossover,
Hull MA crossover, price vs SMA/EMA, Donchian channel, Bollinger Bands, Keltner Channel,
Supertrend, raw momentum, vol-scaled momentum, linear regression t-stat, MACD, RSI, ADX,
CCI, Aroon, Stochastic, Parabolic SAR, Williams %R, MFI, TRIX, PPO, APO, MOM, ROC, CMO,
Ichimoku, OBV, Heikin-Ashi, Kaufman Efficiency Ratio, VWAP, dual momentum, triple MA,
Turtle breakout, regime-filter SMA, ATR breakout, close-above-high, mean-reversion band.

**Stop variants (9)**: None; fixed trailing at 5%, 10%, 20%; ATR-based trailing at 1.5×, 2.0×,
2.5×, 3.0×, 4.0× (14-period ATR at entry date).

**Backtest rules**: Binary long/cash. One-bar lag (signal computed on bar t, position taken on
bar t+1). 20 bps round-trip transaction costs. No leverage. No position sizing. Intraday
signals resampled to daily close for P&L computation.

**Multiple testing**: Effective independent tests estimated at 1,479 (493 signals × 3
frequencies). Stop variants treated as dependent (avg pairwise Sharpe correlation = 0.94).
Bonferroni correction applied at 5% family-wise error rate.

---

## References and Further Reading

Cao, Jeffrey, Nathan Chong, and Dan Villalon. "Hold the Dip." *AQR Alternative Thinking*
2025, Issue 4.

Hurst, Brian, Yao Hua Ooi, Lasse Heje Pedersen. "A Century of Evidence on Trend-Following
Investing." *The Journal of Portfolio Management* 44, no. 1 (2017).

Moskowitz, Tobias J., Yao Hua Ooi, Lasse Heje Pedersen. "Time series momentum." *Journal of
Financial Economics* 104, Issue 2 (2012): 228-50.

Babu, Abilash, Brendan Hoffman, Ari Levine, et al. "You Can't Always Trend When You Want."
*The Journal of Portfolio Management* 46, no. 4 (2020).

AQR. "Trend-Following: Why Now? A Macro Perspective." AQR whitepaper, November 16, 2022.

---

*Hypothetical performance results have many inherent limitations. No representation is made that any strategy will achieve similar results.*
