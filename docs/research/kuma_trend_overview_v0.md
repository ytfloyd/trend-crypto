# kuma_trend Strategy Overview (v0)
_Last updated: 2026-01-07_

## 1. Strategy Summary
- **Name:** kuma_trend  
- **Type:** Long-only trend-following breakout with trailing ATR stops and inverse-vol sizing  
- **Universe (current):** BTC-USD, ETH-USD, SOL-USD, SUI-USD, BCH-USD  
- **Frequency:** Daily (1d bars)  
- **Cash yield assumption:** 4% annualized on uninvested USD  
- **Implementation status:** Research-only (no production wiring)  

In brief: a 20-day breakout gated by a 5/40 MA trend filter, sized inverse to 20-day realized volatility with a 5% cash buffer. The portfolio is long-only; any non-signal capital remains in cash earning 4%. Risk is controlled with 2×ATR(20) trailing stops on each position.

## 2. Investment Objective
Capture large, sustained uptrends in major crypto assets while:
- Cutting losses with 2×ATR(20) trailing stops,
- Moving to cash when the trend filter is off,
- Controlling risk through inverse-volatility allocations and a 5% cash buffer.

Goal: better downside control versus “always long crypto” (e.g., BTC/ETH buy-and-hold), with improved risk-adjusted returns in trending regimes.

## 3. Universe & Data
- **Symbols (v0):** BTC-USD, ETH-USD, SOL-USD, SUI-USD, BCH-USD.  
- **Data source:** DuckDB `../data/coinbase_daily_121025.duckdb`, view/table `bars_1d_usd_universe_clean` (daily OHLCV, vwap/close).  
- **Sample period (v0 backtest):** 2023-01-01 to 2025-01-01.  
- Liquidity: large-cap/upper-midcap USD spot pairs; no additional ADV filter in v0.

## 4. Signal Definition
### 4.1 Breakout logic
- Breakout buy if close_t ≥ prior 20-day high (excluding today).
- Trend filter: MA(5) > MA(40) on close.

**Pseudo-code:**
```
breakout_t = close_t > max(close_{t-20..t-1})
trend_ok_t = MA5_t > MA40_t
enter long if breakout_t and trend_ok_t and currently flat
```

### 4.2 Risk & sizing inputs
- **Realized vol window:** 20 days (std of daily returns).  
- **ATR window:** 20 days (TR = max(high-low, |high-prev_close|, |low-prev_close|)).  

## 5. Portfolio Construction
### 5.1 Inverse-volatility sizing (20-day)
- Eligible names (breakout + MA filter) get weight ∝ 1 / vol_20, normalized so total gross long = 1 – cash_buffer.
- **Cash buffer:** 5% fixed. If no eligible names, portfolio is 100% cash.

### 5.2 Cash and uninvested capital
- Long-only; no shorts, no leverage (gross ≤ 1.0).  
- Residual capital sits in USD earning 4% annualized (daily).  

## 6. Trade Rules & Risk Management
### 6.1 Entry
- Go long when close breaches 20-day high AND MA(5) > MA(40). Size per inverse-vol scheme with 5% cash buffer applied at the portfolio level.

### 6.2 Exit
- **Primary exit:** 2×ATR(20) trailing stop.
  - Initial stop: entry_close – 2×ATR(20 at entry).
  - Trailing stop: max_close_since_entry – 2×ATR_entry.
  - Exit if close_t ≤ stop_t.
- **Secondary exit:** None in v0; relies on trailing stop (MA/breakout can fail without forced exit until stop triggers).

### 6.3 Rebalancing & turnover
- Rebalanced daily off end-of-day signals.  
- Turnover definition: two-sided equity turnover = 0.5 × Σ |w_t – w_{t-1}|.  
- Observed turnover (v0 sample): mean ≈ 0.048; 25/50/75 pct ≈ 0.000 / 0.000 / 0.0065.

## 7. Backtest Assumptions
- Transaction costs: none applied in v0 (gross results).  
- Slippage: none in v0.  
- Funding: none (spot-only, long-only).  
- Universe stability: assume all five symbols trade across the full sample.

## 8. Performance Summary (Backtest v0)
Source: `artifacts/research/kuma_trend/metrics_kuma_trend_v0.csv`

| period      | n_days | total_return | cagr  | vol   | sharpe | max_dd   |
| ---         | ---    | ---          | ---   | ---   | ---    | ---      |
| full        | 2922   | 2506.39      | 1.658 | 0.589 | 2.817  | -0.9998  |
| pre_2020    | 1094   | 25.69        | 1.992 | 0.650 | 3.063  | -0.9998  |
| 2020_2021   | 730    | 16.20        | 3.148 | 0.612 | 5.147  | -0.3209  |
| 2022        | 364    | -0.091       | -0.091| 0.362 | -0.251 | -0.2729  |
| 2023_plus   | 731    | 5.004        | 1.447 | 0.559 | 2.591  | -0.4074  |

Full-period highlights: CAGR ~166%, vol ~59%, Sharpe ~2.82, maxDD ~-100% (driven by early-period compounding; subperiod DDs are more informative).  
Subperiods: strong performance in 2020–2021 trends (Sharpe ~5.15), weaker/choppy in 2022 (negative Sharpe), solid in 2023+ (Sharpe ~2.59).

## 9. Risk & Drawdown Characteristics
- Max drawdown shows extreme compounding sensitivity; subperiod DDs (e.g., -27% in 2022, -41% in 2023+) are more representative.  
- Trailing 2×ATR helps cut losses but can lag during volatility spikes; ATR(20) may be slow in abrupt shocks.  
- Cash stance when no signals reduces tail risk vs fully invested crypto.

## 10. Turnover, Liquidity & Capacity
- Mean two-sided turnover ~0.048; median ~0; 75th pct ~0.0065 (low turnover).  
- Universe is large/upper-midcap; capacity likely constrained by risk limits before market impact at moderate AUM.  
- Lower turnover than high-churn alpha ensembles → more robust to transaction costs; future cost modeling planned.
- Tear sheet: `artifacts/research/kuma_trend/kuma_trend_tearsheet_v0.pdf` (generated by `scripts/research/kuma_trend_tearsheet_v0.py`).

## 11. Implementation Plan & Next Steps
- Add explicit cost model (e.g., 10–20 bps per side) and re-run metrics.  
- Add BTC/ETH benchmark comparison and/or spread vs B&H.  
- Integrate into broader engine / production backtest if promoted.  
- Consider expanding universe to other majors once stable.

## 12. Disclaimers
- Research-only, backtested results; no guarantee of future performance.  
- Liquidity, fees, and operational frictions may materially reduce returns.  
- Not investment advice.

