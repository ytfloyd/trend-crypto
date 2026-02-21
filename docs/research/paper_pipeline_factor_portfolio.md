# Academic Paper Pipeline: Cross-Sectional Factor Discovery and Portfolio Construction

## From Literature Review to Deployable Two-Factor Portfolio in Digital Assets

*Research note -- Pipeline: Discovery → Decay Analysis → ML Overlay → Correlation-Aware Construction*
*Scripts: `scripts/research/paper_pipeline/` | `scripts/research/paper_strategies/`*

---

## 1. Introduction

This research note documents an end-to-end pipeline that begins with automated academic paper discovery and ends with a deployable two-factor portfolio for cryptocurrency spot markets. The pipeline processes 111 papers from arXiv and SSRN, applies a five-stage quality filter plus a methodology audit, and translates the surviving strategies into the crypto universe using the same Coinbase data infrastructure as our prior JPM momentum research.

The central finding is that two cross-sectional factors — low-volatility (VOL_LT) and volume-relative (VOL_RL) — survive all filters, exhibit strengthening alpha with no signs of decay, and combine into a regime-complementary portfolio with a test-period Sharpe ratio of 2.77 and maximum drawdown of -6.1%. Machine learning overlays add no value over the raw signals. The simplest portfolio construction method that accounts for relative factor volatility (volatility parity) outperforms both naive equal-weighting and more sophisticated regime-switching or correlation-forecasting approaches.

The work is organized in four phases:

- **Paper Pipeline**: Discovery, filtering, methodology audit, strategy extraction
- **Phase A**: Alpha decay analysis across six reference cross-sectional factors
- **Phase B**: Smart Predict-then-Optimize (SPO) ML overlay comparison
- **Phase C**: Correlation-aware portfolio construction

## 2. Paper Pipeline: Discovery and Filtration

### 2.1 Data Sources and Scope

We query arXiv (categories q-fin.TR, q-fin.PM, q-fin.ST) and SSRN (search terms: "trading strategy", "alpha", "return predictability", "momentum", "mean reversion", "factor investing", "market anomaly") for papers published within the last five years. For each paper we extract: title, authors, publication date, abstract, asset class, strategy type classification, data period, and key claimed result.

### 2.2 Filter Stack

Papers pass through five sequential filters. Failure at any stage is terminal.

| Filter | Criterion | Rejection Rate |
|:---|:---|---:|
| Alpha | Claims excess return, not pure risk decomposition | ~15% |
| Economic Rationale | Plausible causal mechanism (behavioral, structural, liquidity) | ~20% |
| Statistical Robustness | OOS testing, multiple markets/periods, TCA, ≥5 years data | ~30% |
| Implementability | Public data, no proprietary order flow, no HFT infrastructure | ~10% |
| Staleness | Known anomaly >10 years without adaptation → flagged, not auto-rejected | ~5% flagged |

### 2.3 Methodology Audit (Phase 2b)

A post-filter gate specifically designed to catch implausible claimed performance:

- **Sharpe > 3.0**: Flagged as suspicious
- **Sharpe > 5.0**: Requires explicit explanation or auto-rejected
- **Circular OOS detection**: In-sample parameter selection leaking into out-of-sample window
- **Selected windows**: Convenient start/end dates that happen to capture favorable regimes
- **Overfitting flags**: Complex ML models without regularization documentation

This gate would have auto-rejected a paper claiming a Sharpe of 13 on a 20-year equity factor — the kind of result that reflects methodology choices rather than genuine edge.

### 2.4 Pipeline Output

| Stage | Count |
|:---|---:|
| Papers discovered | 111 |
| Passed quality filters | 5 (4.5% survival rate) |
| Passed methodology audit | 5 |
| Translated to crypto | 4 |
| Untranslatable (microstructure, requires L2 data) | 1 |
| Produced deployable edge | 2 factors |

The 4.5% survival rate is consistent with the paper's thesis that most published strategies do not survive contact with implementability constraints and honest statistical scrutiny. The surviving papers span alpha decay modeling, decision-focused ML, and correlation forecasting — a coherent research thread that we exploit through the subsequent phases.

## 3. Phase A: Alpha Decay Analysis

### 3.1 Methodology

Based on "Not All Factors Crowd Equally" (Lee, 2025), we compute daily returns for six reference cross-sectional factors on the Coinbase universe of 232 assets over 2017-12-31 to 2025-12-14:

| Factor | Construction | Direction |
|:---|:---|:---|
| MOM_1M | 21-day price return, ranked | Long winners, short losers |
| MOM_3M | 63-day price return, ranked | Long winners, short losers |
| MOM_12M | 252-day return ex last 21 days, ranked | Long winners, short losers |
| REV_1W | 5-day return, ranked, sign-flipped | Long recent losers, short recent winners |
| VOL_LT | 20-day realized volatility, ranked, sign-flipped | Long low-vol, short high-vol |
| VOL_RL | 5-day avg volume / 60-day avg volume, ranked | Long high ratio, short low ratio |

All factors are constructed as quintile-sorted long-short portfolios (long top 20%, short bottom 20%, equal-weighted), with a one-day execution lag to prevent lookahead.

For each factor we measure: full-period annualized Sharpe (ANN_FACTOR = 365), last-90-day Sharpe, rolling 90-day non-overlapping Sharpe windows, linear decay slope, exponential half-life estimate, and a crowding proxy (12-month trend in cross-asset signal rank correlation).

### 3.2 Results

**Table 1: Factor Performance and Decay Analysis (232 assets, 2017–2025)**

| Factor | Full Sharpe | Last 90d Sharpe | Decay | Half-Life | Crowding | Priority |
|:---|---:|---:|:---|:---|:---|:---|
| **VOL_LT** | **0.59** | **3.68** | STRENGTHENING | ∞ | STABLE | **HIGH** |
| **VOL_RL** | **0.55** | **2.15** | STRENGTHENING | ∞ | STABLE | **HIGH** |
| REV_1W | 0.24 | -1.05 | STRENGTHENING | ∞ | STABLE | LOW |
| MOM_1M | -0.46 | 0.53 | DECAYING | 1,938d | STABLE | AVOID |
| MOM_12M | -0.52 | -1.66 | DECAYING | — | STABLE | AVOID |
| MOM_3M | -1.02 | -0.17 | DECAYING | 1,089d | STABLE | AVOID |

### 3.3 Analysis

**Cross-sectional momentum is dead in crypto.** All three momentum lookbacks produce negative full-period Sharpe ratios and are classified DECAYING or AVOID. MOM_3M's Sharpe of -1.02 is the worst of the group — a remarkably strong anti-momentum effect. This is consistent with what our Chapter 1 JPM momentum baseline found: the 252-day absolute momentum factor produced a Sharpe of -0.09. The failure extends across all lookback horizons and persists through the most recent data.

The structural explanation is the same as documented in Chapter 1: crypto trend cycles are too short and altcoin return distributions too fat-tailed for cross-sectional momentum to work. Winners reverse before momentum strategies can capture them, and losers exhibit permanent capital loss rather than mean-reversion.

**Low-volatility and volume-relative factors are alive and strengthening.** VOL_LT (Sharpe 0.59) and VOL_RL (Sharpe 0.55) are the only factors with positive full-period Sharpe, strengthening decay slopes, and stable crowding. The "boring coins outperform" anomaly — well-documented in equities as the low-risk anomaly (Baker, Bradley & Wurgler, 2011; Frazzini & Pedersen, 2014) — has not been arbitraged away in crypto. This is plausible: the retail-dominated market has a structural preference for volatile, lottery-like assets, creating a persistent overpricing of high-volatility coins relative to low-volatility ones.

The VOL_RL factor captures a related but distinct signal: assets whose recent trading volume is elevated relative to their own longer-term average. This is a proxy for attention or liquidity inflow — coins that are "waking up" tend to outperform those that are "going quiet."

## 4. Phase B: ML Overlay — Smart Predict-then-Optimize

### 4.1 Pre-Flight: Sanity Checks

Before training any model, we run three mandatory sanity checks on the two HIGH-priority factors.

**Sanity Check 1 — 90-Day Sharpe Spike Explanation.** VOL_LT's last-90-day Sharpe (3.56) is 6× its full-period Sharpe (0.59). Is this unprecedented? No. Both factors have extensive historical precedent: VOL_LT has had 19 prior episodes with 90-day Sharpe > 2.0 (max: 4.92), and VOL_RL has had 18 episodes (max: 5.82). The current spike is not a structural anomaly — these factors periodically produce intense short-term performance, predominantly in CHOP and BEAR regimes when volatile altcoins collapse and low-vol assets hold value. Universe stability is confirmed: 88 recent assets vs. 41 historical average (the universe has grown, not shrunk).

**Sanity Check 2 — Regime Sensitivity.** This is the most important finding of Phase B.

**Table 2: Factor Sharpe by Market Regime**

| Factor | BULL | BEAR | CHOP | Flag |
|:---|---:|---:|---:|:---|
| VOL_LT | **-0.75** | **2.55** | 0.35 | REGIME_DEPENDENT |
| VOL_RL | **1.56** | -0.03 | -0.17 | — |

VOL_LT is a **bear-market factor** — it bleeds during bull markets (Sharpe -0.75, when low-vol coins underperform speculative high-flyers) and prints money during bear markets (Sharpe 2.55, when the low-vol premium provides genuine downside protection). VOL_RL is the **mirror image**: it works in bull markets (Sharpe 1.56, when rising volume signals new capital inflow) but goes flat in downturns. These two factors are regime complements, not redundant signals.

**Sanity Check 3 — PnL Concentration.** Both factors are well-diversified. The top 5 contributors account for only 10.2% (VOL_LT) and 15.4% (VOL_RL) of total PnL. All top contributors remain active in the current universe. Neither factor is driven by a handful of idiosyncratic coin moves.

### 4.2 Model Design

Based on "Smart Predict-then-Optimize for Portfolio Optimization" (Wang Yi & Hasuike, 2026), we train two models per factor — one with standard MSE loss and one with a decision-focused SPO approximation that penalizes prediction errors more heavily where portfolio decisions are made (in the quintile tails):

```
weights = 1 + 3 × |rank_percentile − 0.5| × 2
SPO_loss = weights × (prediction − actual)²
```

Both models are LightGBM regressors trained on 18 features (returns at multiple horizons, volatility measures, price structure, factor scores, BTC regime variables, cross-asset correlation, and regime × factor interaction terms) with a strict walk-forward split:

- Training: 2017-12-31 to 2022-12-31 (expanding window, retrained every 21 days)
- Validation: 2023-01-01 to 2024-03-31 (hyperparameter tuning via 50-iteration random search)
- Test: 2024-04-01 to 2025-12-14 (frozen, untouched until final evaluation)

### 4.3 Results

**Table 3: ML Overlay vs. Raw Factor — Test Period (2024-04 to 2025-12)**

| Model | VOL_LT Sharpe | VOL_RL Sharpe |
|:---|---:|---:|
| **Raw Factor** | **2.03** | **1.79** |
| MSE Model | 1.41 | 0.39 |
| SPO Model | 1.18 | 1.10 |

**Table 4: Overfit and Differentiation Diagnostics**

| Diagnostic | VOL_LT | VOL_RL |
|:---|:---|:---|
| MSE IS/OOS Sharpe ratio | 2.3× (LIKELY_OVERFIT) | 23.7× (LIKELY_OVERFIT) |
| SPO prediction correlation with MSE | 0.869 (DIFFERENTIATED) | 0.479 (DIFFERENTIATED) |
| SPO improvement over MSE | -16% (DEGRADED) | +184% |

### 4.4 Analysis

**The raw factors win decisively on both signals.** This is the headline finding. Neither MSE nor SPO models beat the raw factor signal on either factor. The MSE model's VOL_RL performance is particularly instructive: an in-sample Sharpe of 9.21 collapses to 0.39 out-of-sample — a 24× overfit ratio. The model memorized noise.

**SPO loss produces genuinely different predictions** — the prediction correlation with MSE is 0.869 for VOL_LT and 0.479 for VOL_RL, well below the 0.98 null-result threshold. The quintile-weighting approach works as designed: it reshapes the loss landscape to focus on the portfolio-decision boundary. For VOL_RL, SPO nearly triples MSE's test Sharpe (1.10 vs. 0.39). But "nearly tripling terrible" is still not good enough to beat the raw factor's 1.79.

**Why ML adds no value here.** The daily portfolio-level target (next-day return of a quintile long-short portfolio) has a signal-to-noise ratio that is inherently low. The raw factor construction — rank, sort, equal-weight — is already close to optimal for extracting the cross-sectional signal. A gradient-boosted tree cannot improve on a well-specified sorting procedure when the underlying signal-to-noise ratio is this low and the feature set is this correlated with the sort criterion itself.

**The feature importance tells a regime story.** For the best-performing model (SPO on VOL_RL), `btc_ret_21d` ranks #6 by gain, and the interaction term `btc_ret_21d × vol_lt_rank` ranks #4. The model has learned that factor strength is regime-conditional — but this information is better exploited at the portfolio construction level (Phase C) than at the signal prediction level.

**Cross-factor correlation: 0.042.** The two factors' daily returns are essentially uncorrelated, confirming their suitability for combination.

## 5. Phase C: Correlation-Aware Portfolio Construction

### 5.1 The Core Question

Can a correlation-aware portfolio construction approach — one that explicitly models the regime-conditional relationship between VOL_LT and VOL_RL — outperform naive equal-weighting?

### 5.2 Pre-Flight: Regime-Conditional Correlation Analysis

The Phase B finding that the factors have an unconditional correlation of 0.042 was computed on ML evaluation returns (sign-of-prediction × actual). The true unconditional correlation between the raw factor return series over the full 2017–2025 history is **-0.196** — the factors are not just uncorrelated, they are *negatively* correlated. They actively hedge each other.

**Table 5: Factor Correlation by Market Regime**

| Regime | Correlation | Interpretation |
|:---|---:|:---|
| BULL | **-0.221** | Active hedge during bull markets |
| BEAR | -0.116 | Mild hedge during bear markets |
| CHOP | **-0.248** | Strongest hedge in range-bound markets |
| **Unconditional** | **-0.196** | Structural negative correlation |

Flags: **NEGATIVE_HEDGE** (correlation < -0.2 in BULL and CHOP) and **REGIME_INVARIANT** (all regimes within ±0.15 of unconditional). This is an exceptionally clean result: the negative correlation is structural, not regime-dependent. The factors hedge each other *all the time*, not just in crises.

**Correlation Dynamics.** The rolling 60-day correlation has lag-5 autocorrelation of 0.933 — **CORRELATION_FORECASTABLE**. The correlation is highly persistent and slow-moving, with an average of 16 days to stabilize after a regime transition. There were 540 days in the full history where the rolling 60-day correlation dropped below -0.3, representing extended periods of active hedging.

### 5.3 Four Construction Methods

We evaluate four methods, all applied to the same underlying VOL_LT and VOL_RL raw factor return series, differing only in how they allocate between the two factors. All methods receive a 15% annualized volatility target (20-day lookback, scale factor bounded [0.25, 2.0]) and 20 bps per side transaction costs.

| Method | Description | Rebalance |
|:---|:---|:---|
| **M1 — Equal Weight** | Fixed 50/50 | Monthly |
| **M2 — Volatility Parity** | Weight ∝ 1 / 20d realized vol | Weekly |
| **M3 — Regime Switching** | BULL: 30/70, BEAR: 80/20, CHOP: 55/45 (5-day persistence filter) | On regime change + monthly |
| **M4 — Shrinkage Correlation** | Min-variance using shrinkage forecast of cross-factor correlation | Weekly, 5-day transition smooth |

Method 3's allocations are derived from Phase B's full-history regime Sharpe analysis and are fixed — they are not optimized on the test set. Method 4's shrinkage factor is set to 0.35 (trusting the sample more when correlation is persistent), determined by the CORRELATION_FORECASTABLE flag from RC3, and is not tuned.

### 5.4 Results

**Table 6: Portfolio Construction Comparison — Test Period (2024-04 to 2025-12)**

| Metric | M1-EqWt | M2-VolPar | M3-Regime | M4-Shrink |
|:---|---:|---:|---:|---:|
| **Net Sharpe** | 2.587 | **2.774** | 2.078 | 2.487 |
| CAGR | 57.8% | **61.7%** | 46.2% | 52.9% |
| Max Drawdown | -5.8% | -6.1% | **-5.0%** | -6.0% |
| Calmar | 9.90 | **10.17** | 9.22 | 8.83 |
| Sortino | 4.01 | **4.32** | 3.45 | 3.83 |
| Avg Daily Turnover | 0.000% | 1.02% | 1.65% | 1.03% |

**Table 7: Regime Breakdown — Net Sharpe by Construction Method**

| Regime | M1-EqWt | M2-VolPar | M3-Regime | M4-Shrink |
|:---|---:|---:|---:|---:|
| BULL | 3.32 | 4.68 | 4.16 | **5.02** |
| BEAR | **4.75** | 3.95 | 3.82 | 3.02 |
| CHOP | -0.09 | -0.14 | **-1.61** | -0.41 |

**Correlation Forecast Quality (Method 4):**

| Estimator | MAE |
|:---|---:|
| Shrinkage (regime-conditional prior) | 0.4350 |
| Historical (60d sample) | 0.4364 |
| Naive mean (always predict -0.196) | 0.4214 |

### 5.5 Analysis

**Volatility parity wins.** M2 achieves the highest net Sharpe (2.774), CAGR (61.7%), and Calmar (10.17). It beats equal-weight by +0.186 Sharpe and +0.27 Calmar. The mechanism is simple and robust: when VOL_LT's realized volatility spikes (typically during bear markets when it is running hot), vol parity automatically underweights it, preserving capital for when the factor mean-reverts. The inverse happens for VOL_RL in bull markets. This is an adaptive allocation that requires no regime classification, no correlation forecasting, and no parameter tuning beyond the lookback window for the volatility estimate.

**Regime switching (M3) is the worst method.** Despite being directionally correct — it overweights VOL_LT in bears and VOL_RL in bulls — M3 produces the lowest Sharpe (2.078) and a devastating CHOP Sharpe of -1.61. The problem is twofold: regime transitions are noisy (the 5-day persistence filter reduces but does not eliminate whipsawing), and the fixed 80/20 allocations create concentration that amplifies losses during the frequent CHOP periods. The additional 5 bps per switch cost compounds the drag.

**Shrinkage correlation forecasting (M4) adds marginal value over its complexity.** The MAE analysis reveals why: the shrinkage estimator (0.4350) barely beats the raw 60-day sample (0.4364) and is actually *worse* than the naive unconditional mean (0.4214). For a two-asset portfolio with a structural negative correlation near -0.20, the minimum-variance optimizer simply doesn't have enough room to improve on simpler methods. The correlation is too stable — and too close to the same value in all regimes — for a time-varying estimator to add value.

**CHOP is the universal weakness.** Every method produces negative or near-zero Sharpe in CHOP regimes. This is the structural cost of running factors that are designed around directional moves: when BTC is range-bound, both VOL_LT and VOL_RL lose their signal. This weakness is regime-invariant across construction methods and represents an open research question rather than a construction-solvable problem.

**Comparison to JPM momentum portfolio.** The Chapter 8 Sharpe Blend achieved Sharpe 0.73 with -30.2% max drawdown. The cross-sectional factor portfolio achieves Sharpe 2.77 with -6.1% max drawdown — nearly 4× the risk-adjusted return with 80% less drawdown. However, these are fundamentally different strategies operating on different return drivers: time-series momentum vs. cross-sectional value/quality factors. They are likely to be complementary rather than substitutable, making a blend of the two an obvious next research step.

## 6. Journey Summary: From 111 Papers to a Deployable Strategy

| Phase | Action | Key Finding | Output |
|:---|:---|:---|:---|
| Pipeline | 111 papers → 5-stage filter + methodology audit | 4.5% survival rate; momentum dead, low-vol alive | 5 papers, 4 translatable |
| Phase A | Six cross-sectional factors, decay analysis | VOL_LT (0.59) and VOL_RL (0.55) strengthening; all momentum AVOID | Research queue |
| Phase B | MSE and SPO LightGBM overlays | Raw factors beat ML on both signals; 24× overfit on MSE VOL_RL | Deploy raw factors |
| Phase B (sanity) | Regime sensitivity test | VOL_LT = bear factor, VOL_RL = bull factor; correlation 0.042 | Regime-complement thesis |
| Phase C (pre) | Regime-conditional correlation | True unconditional ρ = **-0.196**; NEGATIVE_HEDGE; FORECASTABLE | Structural hedge confirmed |
| Phase C | Four construction methods | Vol parity wins: Sharpe 2.77, CAGR 61.7%, MaxDD -6.1% | Deployment spec |

## 7. Deployment Specification

**Strategy:** Combined VOL_LT + VOL_RL Raw Factor Portfolio

**Signals:**
- VOL_LT: Long bottom-quintile 20d-realized-vol assets, short top-quintile, equal-weighted. Rebalance: daily.
- VOL_RL: Long top-quintile (5d-avg-vol / 60d-avg-vol) assets, short bottom-quintile, equal-weighted. Rebalance: daily.

**Portfolio Construction:** Volatility parity (inverse-vol weighted between the two factors), rebalanced weekly.

**Risk Management:**
- Volatility target: 15% annualized (ANN_FACTOR = 365)
- Vol estimation lookback: 20 days
- Scale factor bounds: [0.25, 2.0]
- Max single-asset position: 20% of either leg

**Transaction Costs:** 20 bps per side. Net Sharpe at this cost level: 2.77.

**Test Period Performance (2024-04 to 2025-12):**

| Metric | Value |
|:---|---:|
| Net Sharpe | 2.774 |
| CAGR | 61.7% |
| Max Drawdown | -6.1% |
| Calmar | 10.17 |
| Sortino | 4.32 |

**Kill Switch Conditions:**
- Hard stop: portfolio drawdown > 10% from forward-test peak
- Hard stop: either factor rolling 10d Sharpe < -1.0 for 5 consecutive days
- Soft review: factor correlation (10d rolling) exceeds 0.6
- Soft review: universe drops below 50 assets on either factor

## 8. Caveats and Robustness Considerations

1. **The test-period Sharpe of 2.77 is high.** Full-period Sharpe ratios for the individual factors are 0.59 and 0.55, which are in the range of realistic, implementable edges. The 2.77 test-period Sharpe reflects a favorable 20-month window that included strong bear and bull episodes — exactly the regimes where these factors perform best. Forward performance should be benchmarked against the full-period expectations (combined Sharpe likely in the 0.8–1.5 range over a full cycle).

2. **CHOP regime performance is negative across all methods.** If the market enters a prolonged range-bound period (BTC consolidating for 6+ months), the strategy will underperform. This is the known structural weakness and cannot be construction-engineered away. The recommended response is to scale down via the vol-targeting overlay, which will naturally reduce exposure as factor returns decline.

3. **The low-risk anomaly may eventually be arbitraged.** In equities, the low-vol effect has been partially eroded since its publication (Blitz, 2020). In crypto, the anomaly persists because the market remains retail-dominated and structurally overweights volatile assets. As institutional participation grows, VOL_LT alpha may decay. The Phase A decay monitoring (re-run every 90 days) is designed to detect this.

4. **Short-leg implementation risk.** The long-short factor construction assumes the ability to short crypto assets. In practice, shorting is expensive or unavailable for many altcoins. A long-only adaptation (underweight instead of short) would reduce Sharpe but may be more implementable. This was not tested.

5. **The ML null result is specific to this target definition.** We tested daily portfolio-level factor return prediction — a very noisy target. Asset-level return prediction or signal-strength conditioning may produce different results and warrants investigation in a future pipeline run.

## 9. Open Research Directions

1. **Time-series momentum (TSMOM).** Cross-sectional momentum is dead, but TSMOM (each asset vs. its own history) may behave differently. Our Chapter 8 Sharpe Blend already exploits TSMOM at 10–42 day horizons. A combined TSMOM + cross-sectional factor portfolio is the obvious next integration step.

2. **Microstructure signals.** The one untranslatable paper from the pipeline requires order book data. Coinbase Advanced exposes L2 via WebSocket. Microstructure is the highest-signal alpha vertical in crypto; the infrastructure investment (2–3 days to build a collector) would unlock an entirely new signal class.

3. **Funding rate arbitrage.** Cross-exchange perpetual funding rates are a documented crypto-native edge that requires adding a perp data source — currently absent from our OHLCV-only infrastructure.

4. **Factor portfolio + TSMOM blend.** The cross-sectional factor portfolio (Sharpe 2.77 test, ~0.6 full-cycle) and the Chapter 8 TSMOM Sharpe Blend (0.73 full-period) are driven by orthogonal return sources. A risk-parity blend of the two is likely to improve both Sharpe and drawdown.

5. **Alpha decay re-run cadence.** Factor half-lives in crypto are short. The Phase A decay model should be re-run every 90 days to detect emerging crowding or structural changes. Automated scheduling is a pipeline infrastructure priority.

---

**Data sources**: Coinbase daily bars via `coinbase_daily_121025.duckdb`, 232 USD pairs, 2017–2025.
**Phase A artifacts**: `artifacts/research/alpha_decay/`
**Phase B artifacts**: `artifacts/research/phase_b_spo/`
**Phase C artifacts**: `artifacts/research/phase_c_correlation/`
**Master report**: `artifacts/research/master_report_2026-02-21.md`
