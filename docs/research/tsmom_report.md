---
# Time-Series Momentum as a Long Convexity Engine for Crypto

### Pre-Registered TSMOM Experiment

**NRT Research** | February 22, 2026

---

> *We test whether time-series momentum (TSMOM) can serve as the entry/exit timing layer
> for a long convexity compounding engine in cryptocurrency markets.  Unlike our prior
> cross-sectional factor study (1,121 signals, median Sharpe -0.97), TSMOM evaluates
> each asset against its own history — "is BTC trending?" not "is BTC trending more than
> ETH?"  The default position is 100% cash; we enter long only when a trend signal fires
> and exit when it reverses.  The target payoff profile is a portfolio of synthetic call
> options: bounded left tail, fat right tail, positive skewness.*

---

## 1. Pre-Registered Primary Specification

| Parameter | Value |
|---|---|
| Signal | VOL_SCALED |
| Lookback | 21 days |
| Sizing | binary (equal risk per position) |
| Exit | signal_reversal |
| Vol target | 15% annualised |
| Max weight | 20% per asset (excess → cash) |
| Costs | 20 bps round-trip |
| Execution lag | 1 day |

## 2. Primary Specification Results

| Metric | Value |
|---|---|
| **Skewness** | **0.260** |
| Sharpe | 0.764 |
| CAGR | 14.6% |
| Max Drawdown | -38.9% |
| Sortino | 0.886 |
| Calmar | 0.376 |
| Hit Rate | 46.1% |
| Win/Loss Ratio | 1.07 |
| Time in Market | 87.3% |
| Avg Turnover | 0.0526 |
| Participation (portfolio) | 90.9% |
| Participation (per-asset) | 99.1% |

### 2.1 Regime-Conditional Analysis

| Regime | Sharpe | Skewness | Time in Market | BTC Correlation |
|---|---|---|---|---|
| BULL | 4.80 | 1.52 | 96.2% | 0.555 |
| BEAR | -3.41 | -5.51 | 71.5% | 0.344 |
| CHOP | -1.67 | -2.23 | 93.8% | 0.519 |

### 2.2 Pass/Fail Assessment

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| Skewness > 0 | > 0 | 0.260 | PASS |
| Sharpe > 0 | > 0 | 0.764 | PASS |
| Max DD > -30% | > -30% | -38.9% | FAIL |
| BEAR BTC corr < 0.5 | < 0.5 | 0.344 | PASS |
| Participation > 20% | > 20% | 99.1% | PASS |

## 3. Exhibits

![Exhibit 1](../../artifacts/research/tsmom/report_charts/exhibit_1_equity_histogram.png)

![Exhibit 2](../../artifacts/research/tsmom/report_charts/exhibit_2_heatmaps.png)

![Exhibit 3](../../artifacts/research/tsmom/report_charts/exhibit_3_regime_analysis.png)

![Exhibit 4](../../artifacts/research/tsmom/report_charts/exhibit_4_drawdown.png)

![Exhibit 5](../../artifacts/research/tsmom/report_charts/exhibit_5_win_loss.png)

![Exhibit 6](../../artifacts/research/tsmom/report_charts/exhibit_6_crisis_timelines.png)

![Exhibit 7](../../artifacts/research/tsmom/report_charts/exhibit_7_exit_comparison.png)

![Exhibit 8](../../artifacts/research/tsmom/report_charts/exhibit_8_sharpe_skew_pareto.png)

![Exhibit 9](../../artifacts/research/tsmom/report_charts/exhibit_9_vol_target_frontier.png)

## 4. Sensitivity Grid Summary

Total configurations tested: **51**

Median Sharpe: 0.728

Median Skewness: 0.480

### Top 5 by Skewness

| Config | Skewness | Sharpe | CAGR | Max DD |
|---|---|---|---|---|
| RET_63d_binary_signal_reversal_vt15 | 2.80 | 0.80 | 17.1% | -44.0% |
| VOL_SCALED_63d_binary_signal_reversal_vt15 | 2.80 | 0.80 | 17.1% | -44.0% |
| BINARY_63d_binary_signal_reversal_vt15 | 2.80 | 0.80 | 17.1% | -44.0% |
| RET_126d_binary_signal_reversal_vt15 | 2.45 | 0.66 | 11.8% | -42.0% |
| VOL_SCALED_126d_binary_signal_reversal_vt15 | 2.45 | 0.66 | 11.8% | -42.0% |

### Top 5 by Sharpe

| Config | Sharpe | Skewness | CAGR | Max DD |
|---|---|---|---|---|
| MAC_10d_binary_signal_reversal_vt15 | 1.15 | 0.80 | 21.5% | -31.9% |
| LREG_10d_binary_signal_reversal_vt15 | 1.14 | 0.65 | 21.3% | -27.8% |
| EMAC_10d_binary_signal_reversal_vt15 | 1.11 | 0.88 | 21.8% | -36.0% |
| MAC_21d_binary_signal_reversal_vt15 | 1.08 | 0.04 | 22.1% | -36.4% |
| LREG_21d_binary_signal_reversal_vt15 | 1.04 | 0.05 | 21.4% | -33.9% |

---

*Data: Coinbase Advanced spot OHLCV. Universe: point-in-time, $500K ADV filter,
90-day minimum history. Costs: 20 bps round-trip. Execution: 1-day lag.
Annualisation: 365 days. Period: 2017–2025.*
