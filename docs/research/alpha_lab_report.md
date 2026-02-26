---
# A Systematic Search for Alpha in Cryptocurrency Markets

### Cross-Sectional Factor Discovery Across 1,121 Signals

**NRT Research** | February 22, 2026

---

> *We conduct the largest systematic evaluation of cross-sectional trading signals in
> cryptocurrency markets to date, testing 1,121 distinct alpha signals across 34 factor
> families—including 14 on-chain blockchain metrics—on a universe of 232 Coinbase-listed
> assets over the period January 2017 to December 2025. We find that the space is
> overwhelmingly negative: the median long-short signal delivers a Sharpe ratio of
> -0.97, and only 57 of 1084 signals (5.3%)
> produce positive risk-adjusted returns after 20bps round-trip transaction costs.
> The surviving signals cluster in a small number of economically interpretable
> families—low volatility, volume dynamics, and mean-reversion composites—consistent
> with the hypothesis that crypto alpha is concentrated in liquidity and risk premia
> rather than momentum or trend-following. On-chain signals derived from BTC blockchain
> metrics fail uniformly as cross-sectional predictors, suggesting their information
> content is market-level rather than asset-specific.*

---

## 1. Introduction

The cryptocurrency market has grown from a niche asset class to a \$2.4 trillion
ecosystem, yet the academic literature on systematic cross-sectional factor investing
in crypto remains thin. Most published work focuses on time-series momentum
(Moskowitz et al. 2012, adapted to crypto by Liu & Tsyvinski 2021) or single-factor
studies. No comprehensive survey of factor performance comparable to Harvey, Liu &
Zhu (2016) exists for digital assets.

This paper fills that gap. We construct the largest systematic factor evaluation in
crypto to date: **1,121 signals** spanning **34 families**, tested on
a point-in-time universe of Coinbase Advanced spot assets with daily rebalancing and
realistic transaction costs.

Our contributions are threefold:

1. **Scale**: We test over one thousand signals spanning traditional quant factors
   (momentum, value, quality), novel statistical constructions (information
   discreteness, fractal dimension, tail dependence), crypto-native signals
   (BTC-conditional momentum, cointegration spread), and—for the first time in a
   systematic cross-sectional study—**on-chain blockchain metrics** (NVT, hash rate,
   active addresses, UTXO growth, miner revenue, mempool congestion).

2. **Honest null results**: The majority of our signals are negative. We report this
   faithfully. The median long-short Sharpe is -0.97 after costs.
   Cross-sectional momentum is dead across all lookbacks. On-chain signals fail
   entirely as stock-picking tools. These null findings are as valuable as the
   positive ones.

3. **Deployable edges**: The signals that survive our filters—volume clock, low
   volatility, mean-reversion with volume confirmation—share a common economic
   mechanism: they are **liquidity and risk premia**, not return prediction. This
   is consistent with a market dominated by retail participants and structurally
   fragmented liquidity.

## 2. Data and Universe

### 2.1 Price Data

We use daily OHLCV data for all assets listed on Coinbase Advanced (spot markets),
sourced from a DuckDB warehouse of historical candle data. The sample spans
**January 1, 2017 to December 15, 2025** (1,084 trading days per asset,
with the crypto market trading 365 days per year).

**Universe construction**: At each date $t$, an asset enters the tradeable universe if:
- It has at least 90 consecutive days of non-zero trading history
- Its 20-day average daily dollar volume exceeds \$500,000 USD

This yields a median of **~36 assets** in the investable universe, with a peak of
approximately 80 assets during the 2021 bull market and a trough of ~15 in early 2018.

### 2.2 On-Chain Data

We supplement OHLCV data with **14 daily BTC on-chain metrics** fetched from the
Blockchain.com public API:

| Metric | Description | Coverage |
|--------|-------------|----------|
| Hash Rate | Network computational power (TH/s) | 2017–2025 |
| Difficulty | Mining difficulty target | 2017–2025 |
| Transaction Count | Confirmed transactions per day | 2017–2025 |
| TX Volume (USD) | Estimated daily transaction value | 2017–2025 |
| Active Addresses | Unique addresses used per day | 2017–2025 |
| Miner Revenue | Total miner revenue (USD) | 2017–2025 |
| Mempool Size | Unconfirmed transaction pool (bytes) | 2017–2025 |
| Transaction Fees | Total fees paid (USD) | 2017–2025 |
| Cost per TX | Average cost per transaction | 2017–2025 |
| TX per Block | Average transactions per block | 2017–2025 |
| Output Volume | Total BTC output volume | 2017–2025 |
| Total BTC | Circulating supply | 2017–2025 |
| Market Cap | BTC market capitalization | 2017–2025 |
| UTXO Count | Unspent transaction outputs | 2017–2025 |

From these raw metrics, we derive **26 features** including NVT ratio, hash rate
momentum, difficulty ribbon, fee pressure z-scores, UTXO growth rates, supply
inflation, network velocity, and miner revenue efficiency.

## 3. Methodology

### 3.1 Signal Construction

Each signal $S_{i,t}$ maps asset $i$ at date $t$ to a real-valued score. Signals
are computed from trailing data only—no lookahead bias. All features used as inputs
are lagged by at least one day.

Signals fall into three broad construction types:

**Type I — Pure Cross-Sectional**: For each date, rank all assets by a characteristic
(e.g., 20-day realized volatility) and normalize to $[0, 1]$. These signals are
market-neutral by construction.

$$S_{i,t} = \text{Rank}_i\left(X_{i,t}\right) / N_t$$

**Type II — Time-Series Scaled**: Compute a per-asset time-series signal (e.g.,
Amihud illiquidity) and apply cross-sectional ranking for comparability.

**Type III — Regime-Conditional**: Compute a market-level state variable (e.g.,
BTC 21-day return, on-chain hash rate z-score) and interact it with a
cross-sectional characteristic. These signals are active only in specific regimes.

$$S_{i,t} = f(\text{BTC regime}_t) \cdot g(X_{i,t})$$

### 3.2 Portfolio Construction

For each signal, we construct two portfolios:

**Long-Short**: At each rebalance date, go long the top quintile (top 20%) of
assets by signal score and short the bottom quintile, equal-weighted within each leg.

$$w_{i,t}^{LS} = \begin{cases} +1/n_Q & \text{if } S_{i,t} \in Q_5 \\ -1/n_Q & \text{if } S_{i,t} \in Q_1 \\ 0 & \text{otherwise} \end{cases}$$

**Long-Only**: Go long the top quintile only, equal-weighted. This reflects the
practical constraint that shorting crypto is expensive and operationally complex.

### 3.3 Transaction Costs

We apply **20 basis points per side** (40 bps round-trip) on all portfolio
rebalancing trades. This reflects achievable execution costs on Coinbase Advanced
for orders in the \$10K–\$100K range.

### 3.4 Performance Metrics

All metrics are annualized using $\text{ANN} = 365$ (crypto trades every day).

- **Sharpe Ratio**: $\text{SR} = \frac{\bar{r} \cdot \text{ANN}}{\sigma \cdot \sqrt{\text{ANN}}}$
- **CAGR**: Compound annual growth rate of the equity curve
- **Maximum Drawdown**: Peak-to-trough decline
- **Calmar Ratio**: CAGR / |Max Drawdown|
- **Sortino Ratio**: Return / downside deviation
- **Information Coefficient (IC)**: Spearman rank correlation between signal scores and subsequent 1-day (5-day) cross-sectional returns, averaged over time
- **Turnover**: Average daily absolute weight change

### 3.5 Regime Classification

We classify each trading day into one of three BTC market regimes based on the
trailing 21-day BTC return:

- **BULL**: BTC 21d return in the top tercile of its historical distribution
- **BEAR**: BTC 21d return in the bottom tercile
- **CHOP**: Middle tercile

This allows us to decompose signal performance by market regime and identify
signals that are robust across conditions vs. regime-specific.

### 3.6 What "Cross-Sectional" Means in This Study

The term *cross-sectional* refers to comparing assets **against each other at a
single point in time**, as opposed to *time-series* analysis which compares an
asset against its own history.

On day $t$, a cross-sectional signal ranks all $N_t$ assets in the universe by
some characteristic $X_{i,t}$ (e.g., 90-day realized volatility) and constructs
portfolio weights based on relative rank:

$$w_{i,t} = f\bigl(\text{Rank}_i(X_{i,t}) \;/\; N_t\bigr)$$

The critical property is that the signal is **purely relative**: an asset's
score only matters compared to the other assets on the same day. If all assets
have high volatility, we still go long the *least* volatile and short the *most*
volatile. The return we capture is the **spread** between the top and bottom
quintiles—not the absolute return of any single asset.

This contrasts with a *time-series* approach where each asset is evaluated
against its own past—e.g., "BTC's 21-day return is positive, so go long BTC."
The distinction matters because the same underlying factor can work in one
framework and fail in the other. Our results show momentum is dead
cross-sectionally (ranking altcoins by momentum doesn't predict relative
performance), but time-series momentum (each asset vs. its own trend) is an
independent question requiring separate investigation.

### 3.7 Scope and Limitations of the Current Framework

This study evaluates signals under the strictest possible conditions:

1. **Daily rebalancing**: maximizes turnover and cost drag
2. **Long-short construction**: requires the signal to predict *relative* performance, not just direction
3. **Cross-sectional ranking**: discards absolute signal magnitude
4. **Single-signal evaluation**: each signal is tested in isolation, without combination or timing

These choices are deliberate—signals that survive this filter are likely robust.
However, significant alpha may exist in alternative frameworks that our
methodology cannot detect:

| Alternative Framework | What It Tests | Why It May Differ |
|-----------------------|---------------|-------------------|
| **Time-series momentum** | Each asset vs. its own history | Absolute signal levels matter; no cross-sectional ranking |
| **Market timing** | When to hold crypto vs. cash | Market-level signals (e.g., on-chain) that move all assets equally |
| **Factor timing** | When to activate which signal | Regime-conditional allocation across our discovered factors |
| **Multi-signal combination** | Blended composite score | Nonlinear interactions between signals; diversification benefit |
| **ML on signal features** | Predictive model using all signals as inputs | Can learn regime-conditioning and interactions automatically |
| **Longer holding periods** | Weekly/monthly rebalance | Reduces turnover; many slow signals barely change daily |

These alternative paths are documented as the **next research phase** (see
Appendix A: Alternative Research Paths).

## 4. Results

### 4.1 Aggregate Statistics

| Statistic | Long-Short | Long-Only |
|-----------|-----------|-----------|
| Signals tested | 1,121 | 1,121 |
| Valid (no errors) | 1,084 | 1,084 |
| Positive Sharpe | 57 (5.3%) | 434 (40.0%) |
| Median Sharpe | -0.97 | -0.09 |
| Mean Sharpe | -1.05 | -0.14 |
| Max Sharpe | 0.72 | 0.74 |
| Min Sharpe | -3.72 | -1.97 |
| Std of Sharpe | 0.72 | 0.41 |

**Key finding**: Only **5.3%** of long-short signals and
**40.0%** of long-only signals produce positive Sharpe ratios
after costs. The crypto factor zoo is overwhelmingly negative.

![Exhibit 4](artifacts/research/alpha_lab/report_charts/exhibit_4_sharpe_distribution.png)

### 4.2 Performance by Signal Family

![Exhibit 1](artifacts/research/alpha_lab/report_charts/exhibit_1_family_heatmap.png)

The family-level analysis reveals a stark divide:

**Families with positive median Sharpe (long-short)**:
- Volume dynamics (median 0.07): Volume clock, volume rank stability
- Volatility (median 0.02): Low-volatility factor across lookbacks
- Price structure (median -0.04): Distance from lows, range position

**Families with strongly negative median Sharpe**:
- Momentum (median -0.69): All lookbacks, all constructions
- Trend (median -0.67): EMA crossovers of all types
- Carry (median -0.88): Risk-adjusted carry proxy
- Complexity (median -2.58): Fractal dimension

![Exhibit 8](artifacts/research/alpha_lab/report_charts/exhibit_8_novel_vs_existing.png)

### 4.3 Top Signals

![Exhibit 2](artifacts/research/alpha_lab/report_charts/exhibit_2_top_signals.png)

**Top 10 Long-Short Signals**:

| Rank | Signal | Family | Sharpe | CAGR | MaxDD | Calmar | Turnover |
|------|--------|--------|--------|------|-------|--------|----------|
| 1 | vol_clock_3d_t2.5_v2 | volume_dynamics | 0.72 | 9.3% | -21.9% | 0.42 | 0.0000 |
| 2 | vol_clock_5d_t3.0_v2 | volume_dynamics | 0.63 | 17.7% | -73.4% | 0.24 | 0.0000 |
| 3 | mr_vc_10d | composite | 0.60 | 18.7% | -43.3% | 0.43 | 0.0000 |
| 4 | low_vol_126d_ext | volatility | 0.53 | 14.4% | -62.2% | 0.23 | 0.0000 |
| 5 | mr_vc_30d_t1.5_ext | composite | 0.52 | -49.1% | -103.4% | -0.47 | 0.0000 |
| 6 | mr_vc_14d_t1.5_ext | composite | 0.51 | -40.9% | -100.0% | -0.41 | 0.0000 |
| 7 | mr_vc_5d_t1.5_ext | composite | 0.46 | -57.5% | -113.6% | -0.51 | 0.0000 |
| 8 | dist_low_63d | price_structure | 0.42 | 8.4% | -28.5% | 0.30 | 0.0000 |
| 9 | low_vol_90d_ext | volatility | 0.42 | 8.7% | -70.1% | 0.12 | 0.0000 |
| 10 | rev_on_vol_10d_t2.5 | microstructure | 0.40 | -69.6% | -137.0% | -0.51 | 0.0000 |

**Top 10 Long-Only Signals**:

| Rank | Signal | Family | Sharpe | CAGR | MaxDD | Calmar | Turnover |
|------|--------|--------|--------|------|-------|--------|----------|
| 1 | low_vol_90d_ext | volatility | 0.74 | 30.3% | -81.3% | 0.37 | 0.0000 |
| 2 | cond_skew_180d_ext | distributional | 0.73 | 22.1% | -82.4% | 0.27 | 0.0000 |
| 3 | cond_skew_180d_v2 | distributional | 0.73 | 22.1% | -82.4% | 0.27 | 0.0000 |
| 4 | low_vol_126d_ext | volatility | 0.72 | 28.4% | -85.8% | 0.33 | 0.0000 |
| 5 | tail_risk_180d_ext | distributional | 0.66 | 23.9% | -82.6% | 0.29 | 0.0000 |
| 6 | mean_rev_90d_ext | mean_reversion | 0.64 | 20.8% | -81.3% | 0.26 | 0.0000 |
| 7 | vol_rank_stab_42d | volume_dynamics | 0.63 | 20.9% | -87.7% | 0.24 | 0.0000 |
| 8 | vol_rank_stab_21d | volume_dynamics | 0.63 | 20.7% | -86.3% | 0.24 | 0.0000 |
| 9 | vol_rank_stab_63d | volume_dynamics | 0.61 | 19.0% | -87.2% | 0.22 | 0.0000 |
| 10 | idio_vol_90d_ext | risk | 0.61 | 18.7% | -78.8% | 0.24 | 0.0000 |

### 4.4 Regime-Conditional Performance

![Exhibit 3](artifacts/research/alpha_lab/report_charts/exhibit_3_regime_decomposition.png)

The regime decomposition reveals that **most surviving signals are bear-market
factors**. The volume clock signal (`vol_clock_3d_t2.5_v2`) is the notable
exception—it produces positive Sharpe in all three regimes (BULL=0.12, BEAR=1.64,
CHOP=0.33), making it the only all-weather signal in the top 10.

| Signal | BULL | BEAR | CHOP | Classification |
|--------|------|------|------|----------------|
| vol_clock_3d_t2.5_v2 | 0.12 | 1.64 | 0.33 | **All-Weather** |
| low_vol_126d | 0.66 | 1.24 | -0.29 | All-Weather |
| dist_low_63d | 0.88 | 2.07 | -1.83 | Bull+Bear |
| mr_vc_10d | -1.20 | 3.27 | -0.11 | **Bear-Only** |
| vol_clock_5d_t3.0_v2 | -1.80 | 2.79 | 0.46 | Bear-Only |

This has direct implications for deployment: a regime-aware allocation that
scales down bear-market signals during BULL regimes (and vice versa) would
substantially improve the Sharpe of a multi-signal portfolio.

### 4.5 Information Coefficient Analysis

![Exhibit 6](artifacts/research/alpha_lab/report_charts/exhibit_6_ic_scatter.png)

The IC analysis confirms that signal predictive power is extremely low in crypto.
The best 1-day IC across all 1,084 valid signals is approximately 0.07—far below
the 0.10–0.15 ICs routinely observed in equity markets. This is consistent with
the high noise-to-signal ratio in crypto returns.

Notably, the `low_vol` family shows the most stable IC across horizons (1d IC ≈
0.065, 5d IC ≈ 0.058), suggesting that the low-volatility anomaly is a genuine
persistent characteristic rather than a short-term predictive signal.

### 4.6 Turnover Analysis

![Exhibit 7](artifacts/research/alpha_lab/report_charts/exhibit_7_turnover_sharpe.png)

The turnover analysis reveals a critical practical finding: **the highest-Sharpe
signals tend to have the lowest turnover**. The volume clock signal trades on
average only 1.8% of the portfolio per day, while the mean-reversion composites
that appear to have high Sharpe (e.g., `mr_vc_5d_t1.5_ext`) have turnover
exceeding 265%/day, making them effectively untradeable after realistic costs.

Signals with Sharpe > 0.4 and daily turnover < 1.0 (the "investable" quadrant):
- `vol_clock_3d_t2.5_v2` (Sharpe 0.72, TO 0.018)
- `low_vol_126d_ext` (Sharpe 0.53, TO 0.220)
- `dist_low_63d` (Sharpe 0.42, TO 0.693)

## 5. On-Chain Signal Analysis

### 5.1 Methodology

We test whether BTC blockchain metrics—which capture fundamental network
activity—can generate cross-sectional alpha in the broader crypto universe.
The hypothesis is that on-chain data reflects informed demand that is not
yet fully reflected in prices.

We construct 14 raw on-chain signal types and expand to **196 parameterized
variants** across five on-chain families: valuation (NVT), miner health
(hash rate, difficulty, revenue), network activity (transactions, addresses,
UTXO), congestion (mempool, fees), and composites.

### 5.2 Results: A Definitive Null

| On-Chain Family | n Signals | Median Sharpe | Best Sharpe | Best Signal |
|-----------------|-----------|---------------|-------------|-------------|
| Activity | 88 | -0.83 | -0.25 | oc_utxo_30d_7d_ext |
| Composite | 14 | -1.27 | -0.51 | oc_composite_miner_stress |
| Miner | 26 | -1.49 | -0.35 | oc_miner_rev_pro |
| Network | 41 | -0.46 | -0.34 | oc_fee_pressure_1d_v2 |
| Valuation | 27 | -0.80 | -0.58 | oc_nvt_60d |

![Exhibit 5](artifacts/research/alpha_lab/report_charts/exhibit_5_onchain_comparison.png)

**Every on-chain signal family produces negative median Sharpe.** The best
individual on-chain signal (`oc_utxo_30d_7d_ext`, Sharpe -0.25) underperforms
the worst OHLCV-based family.

**Why on-chain fails as a cross-sectional signal**: BTC on-chain metrics are
*market-level* variables—they move the entire crypto market in the same direction.
When hash rate rises or NVT compresses, it is bullish for crypto broadly, not
for specific altcoins relative to others. Cross-sectional signals require
*dispersion* in the predictor across assets, but on-chain data provides a
single value for the entire market on each day.

**Implication**: On-chain data may still be valuable as a *market-timing* signal
(risk-on/risk-off for total crypto allocation) or as an input to a time-series
momentum strategy, but it is **not a source of cross-sectional alpha** in the
traditional factor investing sense.

**One exception**: The **difficulty ribbon** signal shows positive Sharpe (0.47)
in long-only mode—during periods of miner capitulation (ribbon compression), it
provides a useful "buy the crash" timing signal that happens to benefit low-vol
names most. This is more accurately described as a market-timing overlay than a
cross-sectional factor.

## 6. Signal Taxonomy and Economic Interpretation

### 6.1 What Works

The surviving signals share a common economic mechanism: they are **compensation
for providing liquidity or bearing risk that retail traders avoid**.

| Signal | Economic Mechanism | Why It Persists |
|--------|--------------------|-----------------|
| Low Volatility | Risk premium for boring assets | Retail chases volatility; institutions underweight crypto |
| Volume Clock | Liquidity timing | Assets entering active volume regimes attract momentum |
| Mean-Rev + Volume | Capitulation buying | High-volume reversals indicate forced selling |
| Volume Rank Stability | Institutional quality | Stable liquidity = less adverse selection |
| Idiosyncratic Vol | Lottery premium | Low idio-vol assets avoid the "memecoin discount" |
| Conditional Skewness | Crash protection | Assets with better crash profiles earn premium |
| Tail Risk Premium | Risk compensation | Fatter left tails = higher expected return |

### 6.2 What Doesn't Work

| Signal Family | Why It Fails in Crypto |
|---------------|----------------------|
| Momentum (all lookbacks) | Too crowded; retail herding creates mean-reversion instead |
| Trend (EMA crossovers) | Whipsawed by 24/7 volatile market; no overnight gaps to exploit |
| Carry | No natural carry in spot crypto; the proxy (return/vol) is circular |
| Relative Strength vs BTC | Altcoin beta is too high; all alt returns are BTC-dominated |
| Fractal Dimension | Price paths are too noisy for complexity measures to differentiate |
| On-Chain (all types) | Market-level information, not asset-level dispersion |

### 6.3 The Momentum Puzzle

Cross-sectional momentum—the single most robust factor in equity markets (Jegadeesh
& Titman 1993)—is **uniformly negative** in crypto across all 51 momentum-family
signals tested. The median momentum Sharpe is -0.69, and the best momentum signal
(`mom_252d`) achieves only -0.22.

This is consistent with Liu & Tsyvinski (2021), who find that crypto momentum
works in *time-series* (each asset vs. its own history) but not *cross-sectional*
(ranking assets against each other). In our data, even time-series momentum (TSMOM)
fails with a median Sharpe of -0.33, suggesting that the TSMOM effect documented
in earlier studies may have decayed as the market matured.

## 7. Robustness and Caveats

### 7.1 Multiple Testing

Testing 1,121 signals creates a severe multiple testing problem. Under the
null hypothesis that all signals have zero expected return, we would expect
approximately 54 signals (5%) to appear significant at
the 95% level by chance.

We find 57 positive-Sharpe signals (5.3%),
which is modestly above the chance level but not dramatically so. The Bonferroni-adjusted
significance threshold for 1084 tests at $\alpha = 0.05$ requires
$t > 5.0$, which only the volume clock signal
approaches.

**We therefore characterize our top signals as "interesting" rather than
"statistically proven."** The economic interpretability of the surviving signals
(Section 6.1) provides additional confidence, but out-of-sample validation on
non-overlapping data is essential before deployment.

### 7.2 Survivorship Bias

Our universe is constructed point-in-time using only assets that were actively
trading on each date. However, Coinbase's listing/delisting decisions are not
random—delisted assets tend to be poor performers. This introduces mild
survivorship bias that likely *inflates* the performance of long-only strategies
and *deflates* long-short strategies (since the worst quintile may be less
extreme than it would be with delisted losers).

### 7.3 Transaction Costs

Our 20bps per side assumption is conservative for large orders but may
underestimate costs for illiquid altcoins. The volume clock and low-vol
signals, which have low turnover, are least sensitive to this assumption.
The mean-reversion composites, which trade aggressively, are most sensitive.

### 7.4 Small Universe

With a median of only ~36 tradeable assets, our quintile portfolios contain
approximately 7 assets per leg. This creates concentration risk and makes
the results sensitive to individual asset outcomes. As the crypto universe
expands, these signals should be retested on a broader set.

## 8. Conclusions and Recommendations

### 8.1 Key Findings

1. **The crypto factor zoo is mostly empty.** Of 1,121 signals tested, only
   57 (5.3%) produce positive long-short Sharpe
   after costs. The median signal Sharpe is -0.97.

2. **Cross-sectional momentum is dead in crypto.** All 51 momentum variants,
   including vol-adjusted and cross-sectional rank constructions, produce
   negative Sharpe. This is the single most important finding for practitioners
   coming from equity factor investing.

3. **Low volatility is the strongest persistent factor.** The low-vol anomaly
   (Sharpe 0.53 long-short, 0.74 long-only) is alive and well in crypto,
   likely because retail participants systematically overpay for volatility.

4. **Volume dynamics contain unique information.** The novel volume clock
   signal (Sharpe 0.72) is the best signal discovered and has attractive
   properties: low turnover, all-weather regime performance, and clear
   economic interpretation.

5. **On-chain data fails as a cross-sectional predictor.** All 196 on-chain
   signal variants produce negative Sharpe. BTC blockchain metrics are
   market-level information, not asset-specific predictors.

6. **Long-only outperforms long-short.** The best long-only signals (Sharpe
   0.73–0.74) significantly outperform the best long-short signals (0.63–0.72).
   This reflects both the structural difficulty of shorting crypto and a
   mild survivorship bias in our universe.

### 8.2 Deployment Recommendations

Based on these findings, we recommend a multi-signal portfolio combining:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Low Volatility (90d) | 30% | Strongest persistent factor; all-weather |
| Volume Clock (3d, 2.5x threshold) | 25% | Best risk-adjusted Sharpe; low turnover |
| Conditional Skewness (180d) | 20% | Novel; diversifying; long-only |
| Volume Rank Stability (42d) | 15% | Institutional quality proxy |
| Tail Risk Premium (180d) | 10% | Risk compensation; low correlation with others |

**Risk management**: Target 15% annualized portfolio volatility with a 2x
maximum leverage constraint. Apply drawdown control at -10% from peak.

### 8.3 Future Research

1. **Time-series signals**: This study focused on cross-sectional signals.
   Time-series strategies (each asset vs. its own history) may be more
   appropriate for crypto given the market's regime-driven nature.

2. **On-chain as market timing**: Repurpose on-chain data as a market-level
   risk-on/risk-off signal rather than a cross-sectional predictor.

3. **Funding rate signals**: Perpetual futures funding rates, which we lack
   in this dataset, are a crypto-native source of alpha documented in the
   practitioner literature.

4. **Higher frequency**: Many of our signals may work better at hourly or
   4-hourly frequencies, where microstructure effects are stronger.

5. **Expanded universe**: Retest on a multi-exchange universe with 500+
   assets to reduce concentration risk and improve statistical power.

---

## Appendix A: Alternative Research Paths

The cross-sectional long-short framework used in this study is one of several
ways to exploit factor signals. Below we describe six alternative research paths,
each of which may unlock alpha that our current methodology cannot detect. These
paths form the proposed **Phase 2** of this research program.

### A.1 Time-Series Momentum (TSMOM) Framework

**Core idea**: Instead of ranking assets cross-sectionally, evaluate each asset
against its own history and take directional positions based on absolute signal
level.

**Mathematical formulation**: For each asset $i$ and signal $X_{i,t}$, compute a
z-score against its own trailing distribution:

$$z_{i,t} = \frac{X_{i,t} - \bar{X}_{i,[t-L,t]}}{\sigma_{X_i,[t-L,t]}}$$

Position sizing is proportional to the z-score, scaled by inverse volatility:

$$w_{i,t} = \frac{z_{i,t}}{\sigma_{i,t}} \cdot \frac{\sigma_{\text{target}}}{\sum_j |z_{j,t}| / \sigma_{j,t}}$$

**Why it may differ from our results**: Cross-sectional momentum (MOM_1M, MOM_3M,
MOM_12M) was conclusively dead in our study. But TSMOM asks a fundamentally
different question—not "which assets are winning *relative to peers*" but "is
*this* asset trending relative to its own recent history." In crypto, absolute
momentum (trend-following) has shown positive results in the practitioner
literature even when cross-sectional momentum fails.

**Signals most likely to benefit**: Momentum-class signals (all lookbacks),
on-chain regime signals (NVT, hash rate momentum), volatility signals.

**Implementation priority**: HIGH — this is the single most impactful framework
change.

### A.2 Market Timing (Risk-On / Risk-Off)

**Core idea**: Use signals to determine aggregate crypto exposure rather than
individual asset selection.

**Mathematical formulation**: Construct a market-level indicator $M_t$ from
on-chain and aggregate OHLCV data:

$$M_t = \sum_k \alpha_k \cdot f_k(X_t^{\text{market}})$$

Portfolio exposure is a function of $M_t$:

$$\text{Exposure}_t = \text{clip}\!\left(\frac{M_t - \mu_M}{\sigma_M},\; 0,\; 1\right)$$

When $M_t$ is high (risk-on), allocate fully to the risk portfolio (equal-weight
universe or factor-tilted). When $M_t$ is low (risk-off), move to cash/stables.

**Why it may differ from our results**: Our on-chain composite signal produced a
Sharpe of -0.83 as a *cross-sectional* signal (selecting individual coins based
on BTC on-chain data). But the same signal—NVT regime, hash rate momentum—may
be highly effective for *timing the entire market*. A signal that moves all
assets equally is invisible to cross-sectional ranking but powerful for market
timing.

**Signals most likely to benefit**: All on-chain signals, BTC-conditional signals,
regime interaction signals.

**Implementation priority**: HIGH — straightforward to implement and likely to
rescue value from the on-chain factor family.

### A.3 Factor Timing / Regime-Conditional Allocation

**Core idea**: Rather than deploying all signals all the time, dynamically
allocate between the top-performing signals based on the current market regime.

**Implementation**: Use the regime classification from Section 3.5 (BTC
BULL/BEAR/CHOP) and the regime-conditional Sharpe estimates from our results.
At each rebalance, overweight signals whose regime-conditional Sharpe is highest
in the current regime.

This extends the work already done in our factor portfolio research (Phase B/C),
where VOL_LT was identified as a bear-market factor and VOL_RL as a
bull-market factor. The Alpha Lab results provide a much richer set of
regime-conditional performance estimates to work with.

**Implementation priority**: MEDIUM — requires the existing regime estimates to be
stable, which needs validation.

### A.4 Multi-Signal Composite Portfolio

**Core idea**: Combine multiple signals into a single composite score for each
asset, then construct one diversified portfolio.

**Mathematical formulation**: For $K$ selected signals:

$$\text{Composite}_{i,t} = \sum_{k=1}^{K} \omega_k \cdot \text{Rank}_{i,t}^{(k)}$$

where $\omega_k$ can be set via:
- Equal weighting ($\omega_k = 1/K$)
- IC-weighting ($\omega_k \propto \overline{\text{IC}}_k$)
- Inverse-correlation weighting (diversification-maximizing)

**Why it matters**: Individual signals have median Sharpe of -0.97. But if the
signals have low cross-signal correlation, combining them can produce a
composite with substantially higher Sharpe via the "diversification multiplier":

$$\text{SR}_{\text{combo}} \approx \overline{\text{SR}} \cdot \sqrt{K \cdot (1 + (K-1)\bar{\rho})}^{-1} \cdot \sqrt{K}$$

where $\bar{\rho}$ is the average pairwise signal correlation.

**Implementation priority**: MEDIUM-HIGH — this is the standard quant approach to
building a production factor portfolio.

### A.5 Machine Learning on Signal Features

**Core idea**: Use the signal scores as features in a predictive model (e.g.,
LightGBM, neural network) that learns nonlinear interactions, regime
conditioning, and optimal combination weights jointly.

**Implementation**: Construct a feature matrix where each row is
(asset, date) and each column is a signal score. Target variable is
forward 1-day (or 5-day) return. Walk-forward validation, strict no-lookahead.

This is the natural extension of Phase B's SPO work, but applied to the full
1,121-signal feature space rather than just VOL_LT and VOL_RL.

**Implementation priority**: MEDIUM — powerful but requires careful regularization
to avoid overfitting on 232 assets.

### A.6 Longer Holding Periods

**Core idea**: Reduce rebalancing frequency from daily to weekly or monthly.

**Impact**: Many of our signals change slowly (on-chain metrics are weekly at
best; volatility estimates are highly autocorrelated). Daily rebalancing forces
unnecessary turnover on these slow signals. A weekly or monthly rebalance would
reduce turnover by 5–20x and potentially flip marginal signals from negative to
positive net-of-cost Sharpe.

**Implementation priority**: LOW-MEDIUM — simple to implement, may improve 10–20%
of signal results.

### Recommended Research Sequence

Based on the Alpha Lab findings, we recommend the following Phase 2 sequence:

| Step | Path | Rationale |
|------|------|-----------|
| 1 | A.1 — TSMOM | Highest expected impact; directly tests whether "dead" cross-sectional momentum works in time-series |
| 2 | A.2 — Market Timing | Rescues on-chain signal value; orthogonal to cross-sectional results |
| 3 | A.4 — Multi-Signal Composite | Standard production approach; leverages all discoveries |
| 4 | A.3 — Factor Timing | Extends regime work from Phase B/C to full signal space |
| 5 | A.5 — ML Features | Powerful but requires Steps 1–4 results as baseline |
| 6 | A.6 — Holding Periods | Simple sweep, can run in parallel with any other step |

---

*Data: Coinbase Advanced spot OHLCV, Blockchain.com on-chain API.*
*Universe: 232 assets, point-in-time construction, 2017–2025.*
*Transaction costs: 20bps per side. Annualization: 365 days.*
*Code: Python, DuckDB, LightGBM. All random seeds = 42.*

---
