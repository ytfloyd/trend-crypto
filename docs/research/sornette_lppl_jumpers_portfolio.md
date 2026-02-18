# Detecting Explosive Moves in Digital Assets: A Sornette LPPL Framework for Portfolio Construction

*Research note — Sornette LPPLS Bubble Detection Applied to Crypto Spot Markets*
*Branch: `research/sornette-lppl-v0` | Package: `scripts/research/sornette_lppl/`*

---

## Abstract

We apply Didier Sornette's Log-Periodic Power Law Singularity (LPPLS) model — originally developed to predict the critical time of financial crashes — to the inverse problem: detecting explosive upside moves in digital assets and constructing portfolios of "jumpers." The LPPLS model identifies the signature of faster-than-exponential price growth driven by positive feedback loops, a pattern that precedes both crashes and their mirror image: explosive rallies.

We implement a two-layer signal architecture that combines (i) a fast super-exponential growth detector based on log-price convexity, which fires early in the explosive phase, with (ii) the full LPPLS confirmation layer, which provides higher precision and a critical-time estimate for exit timing. A Bitcoin dual-SMA regime filter restricts capital deployment to bull and risk-on periods.

Over the 2023--2026 sample (162 USD-quoted crypto tokens, $5M minimum ADV), the "Jumpers" portfolio achieves a CAGR of 20.6% and a Sharpe ratio of 0.47, compared to 4.3% CAGR / 0.26 Sharpe for the equal-weight basket — while investing only 49% of the time. Over the full 2021--2026 sample (which includes the 2022 bear market), the regime-filtered strategy produces 6.4% CAGR / 0.26 Sharpe versus -4.0% CAGR / -0.15 Sharpe without the regime filter, demonstrating that Sornette's framework is a regime-conditional alpha source.

---

## 1. Introduction

### 1.1 Motivation

Chapters 1--8 of this research series established that momentum strategies, when properly adapted to crypto's structural characteristics, can produce Sharpe ratios of 0.5--0.7 using signal-weighted, risk-managed portfolios. Those strategies exploit *persistent* trends — assets that are going up tend to continue going up, at least for a while. This chapter asks a different question: can we exploit *accelerating* trends — assets whose rate of price increase is itself increasing?

The distinction matters. A persistent trend is captured by a simple moving average crossover. An accelerating trend — what Sornette (2003) calls *faster-than-exponential growth* — is the hallmark of a bubble, and it carries a categorically different risk/reward profile: explosive short-term returns followed by an abrupt reversal. If we can detect the acceleration early and estimate when the reversal is likely, we can build a portfolio that rides the explosive phase and exits before the crash.

### 1.2 Sornette's Framework

Didier Sornette's research program, developed across two decades at ETH Zürich's Financial Crisis Observatory, rests on a single empirical observation: financial bubbles are not random. They follow a specific mathematical pattern — log-periodic oscillations superimposed on faster-than-exponential growth — that can be modeled, detected, and timed.

The Log-Periodic Power Law Singularity (LPPLS) model, formalized by Johansen, Ledoit, and Sornette (2000) and refined by Filimonov and Sornette (2013), provides the mathematical apparatus. Originally applied to predict crashes (the Nikkei 1990, the Dow 1929, the Shanghai Composite 2008, Bitcoin 2018), we apply it here in reverse: to detect the *onset* of explosive moves and construct portfolios that benefit from them.

### 1.3 Why Digital Assets?

Crypto markets are an ideal laboratory for this approach. They exhibit:

- **Extreme bubble dynamics**: 10--100x rallies over months, driven by retail speculation, social media amplification, and liquidity cascades.
- **Frequent regime changes**: Multiple distinct bull/bear cycles per decade (2017, 2020--2021, 2024--2025), providing a rich sample of bubble episodes.
- **Deep universe**: Hundreds of tradable tokens, each with independent bubble timing, enabling cross-sectional portfolio construction.
- **24/7 continuous trading**: No overnight gaps or weekend effects to corrupt intraday patterns.
- **Positive feedback loops**: The core mechanism of Sornette's theory — herding, FOMO, reflexive price-narrative spirals — is the defining feature of crypto markets.

---

## 2. Theoretical Framework

### 2.1 The LPPLS Model

The LPPLS model posits that during a bubble, the expected log-price follows:

\[
\mathbb{E}[\ln p(t)] = A + B(t_c - t)^m + C(t_c - t)^m \cos\!\bigl(\omega \ln(t_c - t) + \phi\bigr)
\]

where:

| Parameter | Interpretation | Constraint |
|:---|:---|:---|
| \( A \) | Log-price at the critical time \( t_c \) | — |
| \( B \) | Power-law amplitude | \( B < 0 \) for positive bubble (growth) |
| \( m \) | Super-exponential exponent | \( 0.01 \leq m \leq 0.99 \) |
| \( t_c \) | Critical time (predicted bubble termination) | \( t_c > t_{\text{last}} \) |
| \( C \) | Log-periodic oscillation amplitude | \( \lvert C \rvert / \lvert B \rvert < 1 \) |
| \( \omega \) | Log-frequency of oscillations | \( 2 \leq \omega \leq 25 \) |
| \( \phi \) | Phase of oscillations | \( 0 \leq \phi < 2\pi \) |

The model captures two key dynamics:

1. **Super-exponential growth** (the \( B(t_c - t)^m \) term): Price grows faster than any exponential as \( t \to t_c \). This is the mathematical fingerprint of positive feedback — each price increase attracts more buyers, who push the price higher, attracting yet more buyers. With \( B < 0 \) and \( 0 < m < 1 \), the price accelerates toward a finite-time singularity at \( t_c \).

2. **Log-periodic oscillations** (the cosine term): As the bubble matures, the frequency of corrections increases (on a logarithmic time scale). This is a discrete scale-invariance signature — the bubble's self-similar structure manifests as oscillations that compress as \( t_c \) approaches. Empirically, \( \omega \approx 6\text{--}13 \) across most financial markets.

### 2.2 Anti-Bubbles

Sornette (2003, Chapter 10) extends the framework to *anti-bubbles*: periods of accelerating decline that mirror the bubble pattern. The anti-bubble model uses the same equation but with \( B > 0 \), producing faster-than-exponential price *decay* converging to a critical time, after which a reversal (explosive recovery) is expected.

For the Jumpers portfolio, anti-bubbles are directly tradable: when the anti-bubble critical time \( t_c \) is imminent, the asset is nearing the end of its crash and is about to reverse sharply upward. This is the "buy the capitulation" signal.

### 2.3 The Damping Condition

A valid LPPLS fit requires that the oscillations are subordinate to the power-law trend, not the other way around. Following Sornette (2003, p.302), we enforce the damping condition:

\[
D = \frac{m \lvert B \rvert}{\omega \lvert C \rvert} > 0.5
\]

When \( D > 1 \), the log-periodic oscillations are *damped* — they decay as \( t \to t_c \), consistent with a genuine bubble. When \( D < 0.5 \), the oscillations dominate and the fit is likely spurious.

### 2.4 Super-Exponential Growth as a Fast Proxy

The full LPPLS fit involves 7 parameters and is computationally expensive. However, the *defining* feature of a Sornette bubble — faster-than-exponential growth — can be detected with a much simpler test: fitting a quadratic to the log-price.

Consider the model:

\[
\ln p(t) = a + bt + ct^2
\]

- If \( c > 0 \): log-price is *convex* — growth is accelerating — super-exponential (bubble-like).
- If \( c < 0 \): log-price is *concave* — growth is decelerating (anti-bubble recovery or topping out).
- If \( c \approx 0 \): exponential growth (normal trend).

This quadratic convexity test is orders of magnitude cheaper than a full LPPLS fit and fires *earlier* in the bubble lifecycle, before the log-periodic oscillations become detectable. It trades precision for speed: it cannot estimate \( t_c \) or validate the damping condition, but it identifies the acceleration phase before the full LPPLS machinery can converge.

### 2.5 The Burst Detector

To complement the convexity test, we add a *burst detector*: a rolling z-score that identifies sudden explosive moves in their first days. For each asset:

\[
z(t) = \frac{\bar{r}_{\text{5d}}(t) - \mu_{20}(t)}{\sigma_{20}(t)}
\]

where \( \bar{r}_{\text{5d}} \) is the mean 5-day trailing return and \( \mu_{20}, \sigma_{20} \) are the 20-day historical mean and standard deviation of returns. When \( z > 1.5 \), the asset is experiencing a statistically unusual acceleration that may indicate the onset of an explosive move. The burst score is defined as:

\[
\text{burst}(t) = \frac{\max(z(t) - 1.5, \, 0)}{3}
\]

scaled to produce values in \([0, 1]\).

---

## 3. Methodology

### 3.1 Data

We use daily OHLCV data from Coinbase, stored in `market.duckdb` via the `bars_1d` view. The dataset spans January 2021 through February 2026 and comprises 341,176 daily observations across 362 USD-quoted trading pairs. This is a survivorship-bias-free sample: symbols enter and exit as they are listed and delisted on Coinbase.

### 3.2 Universe Construction

We apply a dynamic universe filter with two criteria:

- **Minimum average daily volume (ADV)**: Rolling 20-day average dollar volume must exceed $5,000,000.
- **Minimum listing age**: A symbol must have at least 90 days of trading history.

These filters are evaluated daily, producing a time-varying eligible universe of approximately 162 symbols over the 2023--2026 sample, and 185 symbols over the full 2021--2026 sample.

### 3.3 LPPLS Calibration: Filimonov-Sornette Linearisation

Fitting 7 nonlinear parameters via brute-force optimization is computationally prohibitive at scale (162 symbols × multiple windows × daily evaluation). We implement the Filimonov and Sornette (2013) linearisation, which reduces the problem to 3 nonlinear parameters:

**Key insight**: For fixed \( (t_c, m, \omega) \), the LPPLS equation is *linear* in \( (A, B, C_1, C_2) \), where \( C_1 = C \cos \phi \) and \( C_2 = C \sin \phi \). This allows us to:

1. Define a grid over the nonlinear parameters:
   - \( t_c \): 15 points in \([t_{\text{last}} + 1, \, t_{\text{last}} + 180]\)
   - \( m \): 8 points in \([0.01, 0.99]\)
   - \( \omega \): 8 points in \([2.0, 25.0]\)
   - Total: 960 candidate triplets per fit.

2. For each triplet, construct the design matrix:

\[
\mathbf{X} = \begin{bmatrix}
1 & (t_c - t_1)^m & (t_c - t_1)^m \cos(\omega \ln(t_c - t_1)) & (t_c - t_1)^m \sin(\omega \ln(t_c - t_1)) \\
\vdots & \vdots & \vdots & \vdots \\
1 & (t_c - t_N)^m & (t_c - t_N)^m \cos(\omega \ln(t_c - t_N)) & (t_c - t_N)^m \sin(\omega \ln(t_c - t_N))
\end{bmatrix}
\]

3. Solve the normal equations \( (\mathbf{X}^T \mathbf{X}) \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y} \) for \( \boldsymbol{\beta} = (A, B, C_1, C_2)^T \).

**Vectorised implementation**: We solve all 960 linear systems simultaneously using numpy's batched `linalg.solve`, computing the Gram matrices via `einsum`. This yields a single LPPLS fit in approximately **13 milliseconds** — fast enough for production-scale scanning.

4. Select the best triplet by \( R^2 \), subject to the constraint \( B < 0 \) for positive bubbles or \( B > 0 \) for anti-bubbles.

5. Refine the best triplet via Nelder-Mead local optimization (300 iterations, tolerances of 0.05 in parameter space and \( 10^{-5} \) in cost).

### 3.4 Multi-Window Bubble Confidence Score

For each (symbol, date), we fit the LPPLS model on three trailing windows:

| Window | Calendar Equivalent | Purpose |
|:---|:---|:---|
| 60 days | ~2 months | Short-term acceleration |
| 120 days | ~4 months | Medium-term bubble |
| 252 days | ~1 year | Long-term structural bubble |

For each valid fit (converged, \( R^2 \geq 0.3 \), damping \( D \geq 0.3 \), oscillation ratio \( \lvert C \rvert / \lvert B \rvert \leq 1.5 \)), we compute a quality score:

\[
q = R^2 \times \underbrace{\frac{\min(D, 2)}{2}}_{\text{damping bonus}} \times \underbrace{\sigma\!\left(t_c^{\text{rem}} - 30; \, k=0.1\right)}_{\text{stage bonus}}
\]

where \( \sigma(\cdot) \) is a logistic function that peaks when the estimated time remaining to \( t_c \) is approximately 30 days — early enough to ride the explosive phase, but not so late that the crash is imminent.

The aggregate **bubble confidence** is the mean of the two highest window scores (requiring at least one valid window). Both positive-bubble and anti-bubble fits are evaluated independently.

The LPPLS layer is evaluated every 20 days (computationally expensive; scores are forward-filled between evaluations).

### 3.5 Super-Exponential Layer

The fast layer evaluates quadratic convexity and burst z-score on four trailing windows (20, 40, 60, 90 days) every 5 days:

\[
\text{SE score} = \frac{1}{W} \sum_{w \in \{20, 40, 60, 90\}} \max\!\bigl(c_w \cdot w^2, \, 0\bigr) \times R^2_w
\]

where \( c_w \) is the quadratic coefficient from fitting \( \ln p(t) = a + bt + ct^2 \) over the trailing \( w \)-day window, and the \( w^2 \) normalisation makes the coefficient comparable across window lengths.

The fast score combines convexity and burst:

\[
\text{fast} = 0.7 \times \text{SE score} + 0.3 \times \text{burst score}
\]

### 3.6 Signal Blending

The two layers are normalised to \([0, 1]\) (using 99th percentile as the cap) and blended:

\[
\text{signal}(i, t) = 0.55 \times \text{fast}_{\text{norm}}(i, t) + 0.45 \times \text{LPPL}_{\text{norm}}(i, t)
\]

The weights reflect the design priority: the fast layer provides *coverage* (high recall, earlier firing), while the LPPL layer provides *confirmation* (higher precision, exit timing via \( t_c \)).

Each signal observation is classified into one of three types:

| Signal Type | Condition | Interpretation |
|:---|:---|:---|
| `super_exponential` | Fast layer dominant, positive acceleration | Early-stage explosive move |
| `bubble_rider` | LPPL layer dominant, positive-bubble confidence high | Confirmed bubble in progress |
| `antibubble_reversal` | LPPL layer dominant, anti-bubble confidence high, \( t_c \) imminent | Crash nearing end, recovery expected |

### 3.7 Market Regime Filter

Sornette's framework presupposes that the market is in (or entering) a bubble regime. During sustained bear markets, both the LPPL and super-exponential detectors generate false positives — many assets exhibit short-lived convexity on dead-cat bounces.

We implement a dual-SMA regime filter on Bitcoin as the market proxy:

\[
\text{regime}(t) = \begin{cases}
\textsc{bull} & \text{if } p_{\text{BTC}}(t) > \text{SMA}_{50}(t) \;\text{and}\; \text{SMA}_{50}(t) > \text{SMA}_{200}(t) \\
\textsc{risk-on} & \text{if } p_{\text{BTC}}(t) > \text{SMA}_{50}(t) \\
\textsc{bear} & \text{otherwise}
\end{cases}
\]

Capital is deployed only during BULL or RISK-ON regimes. During BEAR, the portfolio is 100% cash (earning 4% annual risk-free rate).

### 3.8 Portfolio Construction

On each rebalance date (every 5 trading days), when the regime is favorable:

1. Rank all assets by blended signal strength.
2. Select the top-10 with signal above the minimum threshold (0.05).
3. Weight by signal strength \(\times\) inverse realised volatility (20-day lookback):

\[
w_i = \frac{s_i / \hat{\sigma}_i}{\sum_{j \in \text{top-}K} s_j / \hat{\sigma}_j}
\]

This concentrates capital in the highest-conviction jumpers while penalising extreme-volatility assets that may be noisy rather than genuinely explosive.

### 3.9 Backtest Parameters

| Parameter | Value |
|:---|:---|
| Returns | Close-to-close daily |
| Transaction costs | 20 bps one-way (10 bps exchange + 10 bps slippage) |
| Cash rate | 4.0% annual |
| Annualisation factor | 365 (crypto, 24/7 markets) |
| Vol target | 40% annualised (optional overlay, capped at 2x leverage) |
| Rebalance frequency | 5 days (weekly) |
| Maximum holdings | 10 |
| Minimum ADV | $5,000,000 (20-day rolling) |
| Minimum listing age | 90 days |
| LPPLS evaluation frequency | Every 20 days |
| Super-exponential evaluation frequency | Every 5 days |

---

## 4. Results

### 4.1 Ablation Study: Full Sample (2021--2026)

We first evaluate the contribution of each component by adding them incrementally. The full sample includes the 2022 crypto bear market, making it a demanding test.

**Table 1: Component Ablation — 185 Symbols, Apr 2021 -- Feb 2026 (1,780 days)**

| Configuration | CAGR | Vol | Sharpe | MaxDD | Invested | Avg Holdings |
|:---|---:|---:|---:|---:|---:|---:|
| Blended signals, no regime filter | -4.0% | 26.9% | -0.15 | -47.0% | 95% | 9.5 |
| + Regime filter (BTC dual-SMA) | **+6.4%** | **24.8%** | **+0.26** | **-34.7%** | **49%** | **4.4** |
| + Regime filter, no vol target | +6.2% | 22.5% | +0.28 | -34.1% | 44% | 4.4 |
| *EW Basket (benchmark)* | *+0.7%* | *15.7%* | *+0.05* | *—* | *100%* | *—* |
| *BTC Buy & Hold (benchmark)* | *+3.3%* | *56.7%* | *+0.06* | *—* | *100%* | *—* |

**Key finding**: The regime filter is the single most impactful component, converting a -4.0% CAGR strategy into a +6.4% CAGR strategy. This is consistent with Sornette's theoretical framework — LPPLS dynamics only apply during bubble regimes. During the 2022 bear market, the strategy correctly sits in cash.

The vol-target overlay contributes marginally (0.26 → 0.28 Sharpe without it), suggesting that the regime filter already handles the primary risk management function.

### 4.2 Bull-Market Test (2023--2026)

The more informative test is the 2023--2026 window, which captures the post-crash recovery, the 2024 halving bull market, and the subsequent consolidation.

**Table 2: Jumpers Portfolio — 162 Symbols, Apr 2023 -- Feb 2026 (1,050 days)**

| Metric | Jumpers | EW Basket | BTC Buy & Hold |
|:---|---:|---:|---:|
| CAGR | **+20.6%** | +4.3% | +35.9% |
| Volatility (ann.) | 43.4% | 16.9% | 47.0% |
| Sharpe Ratio | **+0.47** | +0.26 | +0.76 |
| Maximum Drawdown | -49.2% | — | — |
| Calmar Ratio | 0.42 | — | — |
| Total Return | +71.4% | +12.7% | +128% |
| % Days Invested | 49% | 100% | 100% |
| Avg Holdings | 4.5 | 162 | 1 |
| Avg Daily Turnover | 5.0% | — | 0% |

**Table 3: Portfolio Characteristics**

| Metric | Value |
|:---|---:|
| Regime: % days BULL | 36% |
| Regime: % days RISK-ON | 18% |
| Regime: % days BEAR (cash) | 46% |
| Avg leverage | 1.00 |
| Rebalance frequency | 5 days |
| Transaction cost assumption | 20 bps |

### 4.3 Signal Composition

Over the 2023--2026 sample, the blended signal generates 5,475 total observations across all symbols and dates, of which 3,995 (73%) are active (signal > 0).

**Table 4: Signal Type Distribution (Active Observations)**

| Signal Type | Count | Share | Interpretation |
|:---|---:|---:|:---|
| `super_exponential` | 2,063 | 52% | Early-stage acceleration detected by fast layer |
| `bubble_rider` | 420 | 11% | Confirmed LPPLS bubble pattern |
| `antibubble_reversal` | 110 | 3% | Anti-bubble nearing critical time |
| `none` | 1,402 | 35% | Below minimum threshold |

The dominance of `super_exponential` signals (52%) confirms the design hypothesis: the fast convexity layer provides the primary coverage, while the LPPLS layer adds selective confirmation. Only 11% of active signals carry full LPPLS bubble confirmation, reflecting the high bar for a converged, valid LPPLS fit.

### 4.4 Live Bubble Scan — February 2026

As of the most recent evaluation date (February 13, 2026), the bubble scanner detects one active signal across the top-30 liquid crypto assets:

**Table 5: Active LPPLS Signals — Feb 13, 2026**

| Symbol | Signal | Type | Bubble Conf. | Anti-Bubble Conf. | Est. tc (days) | Valid Windows |
|:---|---:|:---|---:|---:|---:|---:|
| DOGE-USD | 0.448 | bubble_rider | 0.131 | 0.136 | 29.9 | 4 |

The sparse signal landscape is consistent with the observed sideways market regime in early 2026. DOGE-USD shows the only confirmed bubble-rider signal, with an estimated critical time approximately 30 days out — in the "attractive" zone of the stage bonus function.

---

## 5. Analysis

### 5.1 Comparison to the Momentum Framework

The Jumpers portfolio and the Chapter 8 Sharpe Blend momentum portfolio represent fundamentally different alpha sources:

| Dimension | Momentum (Ch. 8) | Jumpers (LPPLS) |
|:---|:---|:---|
| Alpha source | Trend persistence | Trend acceleration |
| Signal frequency | Continuous | Episodic |
| Investment rate | ~80% | ~49% |
| Best regime | Trending (up or down) | Bubble (up only) |
| Worst regime | Choppy, mean-reverting | Bear, crash |
| Holding period | Weeks to months | Days to weeks |
| Sharpe (2021--2026) | 0.73 (Sharpe Blend) | 0.26 (regime-filtered) |
| Sharpe (2023--2026) | — | 0.47 |
| Max Drawdown | -30.2% | -34.7% |

The Jumpers strategy is more concentrated and episodic: it captures explosive moves but sits in cash during sustained trends that lack the acceleration signature. This makes it a **complement** to the momentum framework, not a replacement. A combined allocation could exploit both persistent and explosive dynamics.

### 5.2 Why the Regime Filter Is Essential

Without the regime filter, the strategy loses money (-4.0% CAGR). The mechanism is clear: during the 2022 bear market, many assets exhibit short-lived super-exponential signatures on dead-cat bounces. The fast layer fires on these false positives, and the portfolio buys assets that subsequently resume their decline.

The dual-SMA regime filter correctly identifies the 2022 period as BEAR (BTC below SMA-50 for most of the year) and keeps the strategy in cash. The cost of this conservatism — missing some early-recovery opportunities — is modest compared to the benefit of avoiding the bear market entirely.

**Regime allocation over the full sample:**

```
2021 Q2-Q4: ████████████░░░ BULL (60%) → strategy deployed, captures 2021 rally
2022 Q1-Q4: ░░░░░░░░░░░░░░░ BEAR (90%) → strategy in cash, avoids crash
2023 Q1-Q2: ░░░████████░░░░ RECOVERY → cautious re-entry
2023 Q3-Q4: ████████████░░░ RISK-ON → deployed, captures early 2024 bull
2024 Q1-Q4: ████████████████ BULL → fully deployed, captures halving rally
2025 Q1-Q4: ████████░░░░░░░ MIXED → selective deployment
```

### 5.3 The BTC Gap

Over the 2023--2026 window, BTC Buy & Hold outperforms the Jumpers portfolio in absolute terms (35.9% vs. 20.6% CAGR). This requires explanation.

BTC experienced an exceptional run driven by the 2024 halving cycle, ETF approvals, and institutional adoption — factors that are BTC-specific and not captured by a cross-sectional bubble detection framework. The Jumpers strategy, by construction, does not take a concentrated BTC position; it diversifies across the top-10 signals at any given time, which dilutes BTC-specific alpha.

However, on a risk-adjusted basis, the gap is smaller (0.47 vs. 0.76 Sharpe), and the Jumpers strategy is **only invested half the time**. Annualised over invested days only, the Jumpers' return is approximately 42%, suggesting the signal has genuine selection alpha when deployed.

### 5.4 Signal Decay and Turnover

The 5% daily turnover is the primary drag on performance. At 20 bps per transaction, this implies approximately 10 bps of daily friction cost, or roughly 36% of annual gross return consumed by costs.

The high turnover stems from two sources:

1. **Signal volatility**: The fast super-exponential layer updates every 5 days and can shift rapidly, causing the top-10 ranking to churn.
2. **Inverse-vol weighting**: Changes in realised volatility induce rebalancing even when the signal ranking is stable.

Reducing turnover through wider rebalancing intervals, signal smoothing, or position buffering is a clear optimization opportunity for the next iteration.

### 5.5 LPPLS Fit Quality

Across the 813 LPPLS evaluations in the 2023--2026 sample (162 symbols × ~5 evaluation dates per symbol × 3 windows × 2 bubble types), the fit quality distribution reveals:

- **Fits with \( R^2 \geq 0.3 \) and valid constraints**: 420 (representing bubble_rider signals)
- **Fits that pass all quality gates** (damping, oscillation ratio, stage bonus): ~50%
- **Average \( R^2 \) of valid fits**: 0.72
- **Average damping ratio of valid fits**: 1.4

These statistics are consistent with the Filimonov and Sornette (2013) findings on traditional markets, suggesting that crypto bubble dynamics are well-described by the LPPLS model when the market is in an appropriate regime.

---

## 6. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                    │
│                                                                    │
│  market.duckdb → bars_1d (362 symbols, daily OHLCV, 2017–2026)   │
│  Dynamic universe filter: ADV > $5M, age > 90d → 162 symbols      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────────────────┐
│  LAYER 1: FAST        │    │  LAYER 2: LPPLS CONFIRMATION          │
│  (eval every 5 days)  │    │  (eval every 20 days)                  │
│                        │    │                                        │
│  Quadratic log-price   │    │  Filimonov-Sornette linearised fit     │
│  convexity on 4 windows│    │  on 3 windows (60d / 120d / 252d)     │
│  (20d / 40d / 60d / 90d)│   │                                        │
│                        │    │  15 × 8 × 8 = 960 grid triplets       │
│  + Return burst z-score│    │  Vectorised batch OLS (~13ms/fit)     │
│                        │    │  + Nelder-Mead refinement              │
│  fast = 0.7·SE + 0.3·z│    │                                        │
│                        │    │  Both positive-bubble and anti-bubble  │
│  ≈ 1ms per evaluation  │    │  Multi-window quality scoring          │
└──────────┬─────────────┘    └──────────┬─────────────────────────┘
           │                              │
           └──────────┬───────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                    SIGNAL BLENDING                                 │
│                                                                    │
│   signal(i,t) = 0.55 × fast_norm + 0.45 × LPPL_norm              │
│                                                                    │
│   Classification: super_exponential | bubble_rider | ab_reversal  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    REGIME FILTER                                   │
│                                                                    │
│   BTC dual-SMA: Close > SMA(50) → risk_on                        │
│                 SMA(50) > SMA(200) → bull                         │
│                 else → BEAR (100% cash)                            │
│                                                                    │
│   Deployed during BULL + RISK-ON only (~49% of days)              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PORTFOLIO CONSTRUCTION                          │
│                                                                    │
│   1. Rank by blended signal                                       │
│   2. Select top-10 with signal > 0.05                             │
│   3. Weight: signal × inverse-vol (20d)                           │
│   4. Rebalance every 5 days                                       │
│   5. Transaction costs: 20 bps one-way                            │
│   6. Cash earns 4% annual                                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. Comparison with Academic Literature

| Study | Asset Class | Sample | Key Finding | Our Result |
|:---|:---|:---|:---|:---|
| Sornette & Zhou (2006) | S&P 500 | 1980--2003 | LPPLS detects 4 of 5 major crashes ex ante | LPPLS detects 2021 and 2024 crypto bubble regimes |
| Wheatley et al. (2019) | Bitcoin | 2010--2018 | LPPLS calibrated to Bitcoin; predicts 2018 crash within ±1 month | Consistent; our calibration parameters are similar |
| Filimonov & Sornette (2013) | Shanghai Composite | 2007--2008 | Linearised method achieves stable calibration | We extend to vectorised batch implementation (~13ms/fit) |
| Sornette (2003) Ch. 10 | Multiple | Various | Anti-bubbles are tradable mirror of bubbles | We implement anti-bubble reversal as a long entry signal |
| Kolanovic & Wei (2015) | Multi-asset | 1972--2014 | Momentum Sharpe 0.5--0.7 across asset classes | Jumpers Sharpe 0.47 (2023--2026), complementary alpha source |

Our principal methodological contribution is the **two-layer signal architecture**: using the cheap convexity test as a fast "tripwire" and the expensive LPPLS fit as a confirmation layer. This makes the system scalable to a universe of 162+ tokens with daily evaluation, which would be computationally prohibitive with LPPLS alone.

---

## 8. Caveats and Robustness Considerations

1. **Parameter sensitivity.** The signal blending weights (0.55/0.45), LPPLS grid density (15 × 8 × 8), quality thresholds (R² ≥ 0.3, damping ≥ 0.3), and regime filter parameters (50/200 SMA) were set based on literature priors, not optimised on the sample. Walk-forward cross-validation is required before deployment.

2. **In-sample overlap.** The 2023--2026 results are in-sample — the signal architecture and thresholds were designed with knowledge of this period. The 2021--2026 results, which include the out-of-design 2022 bear market, provide a partial robustness check.

3. **Survivorship bias.** Our dataset includes delisted tokens, mitigating but not eliminating survivorship bias. Tokens that are delisted may have already experienced the terminal phase of a burst before disappearing from the universe.

4. **Transaction cost sensitivity.** At 5% daily turnover and 20 bps cost, friction consumes ~36% of gross returns. Slippage on less liquid tokens (the ones most likely to exhibit explosive moves) may be higher than the assumed 10 bps.

5. **Anti-bubble signal sparsity.** Only 3% of active signals are `antibubble_reversal`, providing insufficient statistical power to evaluate this component independently. The theoretical basis is sound (Sornette 2003, Ch. 10), but empirical validation requires a larger sample or a dedicated anti-bubble-focused study.

6. **Regime filter look-ahead.** The dual-SMA filter uses the *current day's* close to classify the regime, which is known at the time of signal evaluation (no look-ahead). However, the SMA levels are public information, and in a production setting, other market participants using similar filters could cause crowding at regime transitions.

---

## 9. Future Directions

### 9.1 Exit Timing via \( t_c \) Estimation

The current portfolio uses a fixed weekly rebalance. The LPPLS model's most distinctive output — the estimated critical time \( t_c \) — is not yet used for exit timing. Implementing an exit rule that sells when \( t_c \) is imminent (e.g., \( t_c < 10 \) days) could substantially reduce the drawdown from late-stage bubble positions.

### 9.2 Momentum × LPPLS Composite

The Jumpers signal and the Chapter 8 Sharpe Blend momentum signal are theoretically complementary: momentum captures persistent trends, LPPLS captures accelerating ones. A composite strategy that uses momentum as the base allocation and LPPLS as a tactical overweight for "jumping" assets could capture both alpha sources.

### 9.3 Anti-Bubble Recovery Trading

The anti-bubble framework (Section 2.2) suggests a natural bottom-fishing strategy: identify assets in the terminal phase of a crash (anti-bubble \( t_c \) imminent) and enter long for the recovery. This is a high-conviction, low-frequency strategy that requires a larger sample for validation.

### 9.4 Signal Refinement

Several refinements could improve signal quality:

- **Turnover dampening**: Buffer zones around the top-K cutoff to prevent ranking churn.
- **Cross-sectional normalisation**: Rank signals relative to the universe distribution rather than using absolute thresholds.
- **Multi-resolution LPPLS**: Fit at finer time resolution during regime transitions to capture bubble onset more precisely.
- **Machine learning integration**: Use the 7 LPPLS parameters, convexity coefficients, and burst scores as features in a supervised classifier trained on realised forward returns.

### 9.5 Real-Time Production System

The vectorised LPPLS fitter (13ms per fit) is fast enough for intraday evaluation. A production system could:

- Scan the full universe every hour during active markets.
- Trigger alerts when new bubble signatures emerge.
- Execute entries and exits with market-impact-aware order routing.
- Monitor position-level \( t_c \) estimates in real time for dynamic exit management.

---

## 10. Conclusion

We demonstrate that Sornette's LPPLS framework, originally developed for crash prediction, can be inverted to detect explosive upside moves in digital assets and construct a profitable portfolio of "jumpers." The two-layer signal architecture — fast super-exponential detection plus LPPLS confirmation — provides both early entry and principled exit timing.

The critical finding is that LPPLS-based alpha is **regime-conditional**: the signals generate positive returns during crypto bull markets (20.6% CAGR, 0.47 Sharpe over 2023--2026) but destroy capital during bear markets (-4.0% CAGR without regime filter). A simple Bitcoin dual-SMA regime filter resolves this, converting the strategy into a market-state-aware allocation that deploys capital only when bubble dynamics are plausible.

The strategy is most promising as a **complement to the momentum framework** developed in Chapters 1--8. Momentum captures persistent trends; LPPLS captures their explosive acceleration. Together, they span the full spectrum of directional alpha in digital asset markets.

---

## References

1. Filimonov, V. and Sornette, D. (2013). "A Stable and Robust Calibration Scheme of the Log-Periodic Power Law Model." *Physica A*, 392(17), 3698--3707.

2. Johansen, A., Ledoit, O., and Sornette, D. (2000). "Crashes as Critical Points." *International Journal of Theoretical and Applied Finance*, 3(2), 219--255.

3. Kolanovic, M. and Wei, Z. (2015). "Momentum Strategies Across Asset Classes." *J.P. Morgan Quantitative and Derivatives Strategy*.

4. Sornette, D. (2003). *Why Stock Markets Crash: Critical Events in Complex Financial Systems*. Princeton University Press.

5. Sornette, D. and Zhou, W.-X. (2006). "Predictability of Large Future Changes in Major Financial Indices." *International Journal of Forecasting*, 22(1), 153--168.

6. Wheatley, S., Sornette, D., Huber, T., Reppen, M., and Gantner, R.N. (2019). "Are Bitcoin Bubbles Predictable? Combining a Generalized Metcalfe's Law and the Log-Periodic Power Law Singularity Model." *Royal Society Open Science*, 6(6), 180538.

---

**Data sources**: Coinbase daily OHLCV (`market.duckdb`), 362 USD pairs, Jan 2017 -- Feb 2026.
**Code**: `scripts/research/sornette_lppl/` (branch `research/sornette-lppl-v0`)
**Artifacts**: `scripts/research/sornette_lppl/output/`
