# 101 Alphas – USD Universe Ensemble (V1 ADV>10M, 2017–2025)

Branch / tag: `research/101-alphas-v1-remediation`, tag `v0.3-101-alphas-custom-201-220`

## Strategy Overview
- Universes  
  - Base: Coinbase USD spot pairs (ex-stablecoins).  
  - V1: ADV > $10M subset (e.g., BTC, ETH, SOL, AVAX, LINK, DOGE, etc.; 82 symbols in the 2023–2025 window after filter).
- Signal set  
  - 101 Kakushadze-style alphas + 20 custom alphas (`alpha_201`–`alpha_220`).  
  - Regime features (`alpha_c16`–`alpha_c20`).
- Construction  
  - Daily cross-sectional ranking of selected alphas; average to a signed signal.  
  - Long-only, long+cash; negative signals map to cash (no shorting).  
  - Danger-regime gating: on “danger” days, weights go to cash.  
  - Target gross ≈ 1.0; per-symbol weights normalized each day.

## Remediation & Data Hygiene
- Ghost-data filters for IC/decay/bias: only bars with `volume > 0` and `close_t != close_{t-1}`. Reduces stale prints and yields conservative IC.  
- Liquidity: ADV > $10M (20-day rolling `close * volume`) to drop thin/ghost names.  
- Turnover: standardized to two-sided equity turnover = `0.5 * Σ_i |w_{i,t} - w_{i,t-1}|`; all metrics/TCA use this.

## Performance Snapshot (V1 ADV>10M, 2023-01-01 to 2025-01-01, 731 days)

Gross (metrics_101_ensemble_filtered_v1.csv)  
- CAGR: 48.5%  
- Vol: 27.2%  
- Sharpe: 1.78  
- Max DD: -22.6%

Net of costs (capacity_sensitivity_v1.csv)  
- 10 bps: Sharpe 1.50, CAGR 40.7%, Max DD -24.1%  
- 20 bps: Sharpe 1.22, CAGR 33.3%, Max DD -25.8%  
- 30 bps: Sharpe 0.97, CAGR 26.3%, Max DD -27.8%  
- 40 bps: Sharpe 0.72, CAGR 19.7%, Max DD -29.8%  
- 50 bps: Sharpe 0.49, CAGR 13.4%, Max DD -31.7%

Interpretation: Strategy remains attractive through ~20–30 bps per-side costs; above that, Sharpe compresses but stays positive.

## Risk & Correlation Profile
- Beta vs BTC (alphas101_beta_vs_btc_v1_adv10m.csv): corr ≈ -0.041, beta ≈ -0.023 (diversifying, small negative beta).  
- Concentration (alphas101_concentration_summary_v1_adv10m.csv): BTC+ETH share ≈ 12.9% of avg gross; top weights spread across BTC, ETH, SOL, LTC, DOGE, LINK, XRP, AVAX, SUI, BONK.

## Signal Quality & Decay
- IC panel (alphas101_ic_panel_v1_adv10m_filtered.csv): horizon-1 mean ICs are low but positive after ghost filtering (e.g., ~0.009–0.03 ranges with modest t-stats); selection kept 2 highest-tstat signals.  
- IC decay (alphas101_ic_decay_filtered_v1.csv): mean IC rises modestly to ~0.03 at horizon 4 then decays; no flat ghost plateau.  
- Bias check (alphas101_alpha008_bias_filtered_v1.csv): baseline IC ~0.0095, lag1 ~0.0081; both positive after filtering → predictive content, not look-ahead.

## Capacity & Turnover
- Average two-sided turnover (ensemble_turnover_v0.csv): ~0.148 per day (median 0.15).  
- Capacity sensitivity (capacity_sensitivity_v1.csv): Sharpe declines smoothly from 1.78 (0 bps) to 0.49 (50 bps); table includes cagr/vol/max_dd per cost tier.

### V1 Concentration Experiment: Base vs Growth (ADV>10M, 2021-11-01+)
Using `alphas101_concentration_compare_v1.py` on:
- Base equity: `ensemble_equity_v1_base.csv`
- Growth equity: `ensemble_equity_v1_growth.csv`
- Start date: 2021-11-01

| Variant   | n_days | Total Return | CAGR  | Vol (ann.) | Sharpe | MaxDD   |
|-----------|--------|--------------|-------|------------|--------|---------|
| V1 Base   |  731   | ~12.4%       | ~6.0% | ~1.7%      | ~3.45  | ~‑0.9% |
| V1 Growth |  731   | ~36.6%       | ~16.8%| ~8.5%      | ~1.98  | ~‑6.1% |

Trade-off (Growth – Base):
- ΔCAGR ≈ +10.8 pts
- ΔMaxDD ≈ ‑5.2 pts (deeper drawdown)

Interpretation:
- Growth sleeve roughly triples total return over the window, at the cost of higher but still modest drawdown.
- Base remains the defensive/yield profile; Growth is the convexity overlay (Top‑10 sleeve at 20% gross, 2% cap).
- Canonical V1 filtered performance (CAGR ~48.5%, Sharpe ~1.78, MaxDD ~‑22.6%) still comes from `ensemble_equity_v0.csv` and `metrics_101_ensemble_filtered_v1.csv`; the concentration experiment is additive and does not change the canonical V1 tear sheet (`alphas101_tearsheet_v1_adv10m.pdf`).

## Symbol-Level Exposure & Turnover (ADV>10M V1)
- Top exposures are balanced across BTC/ETH/SOL/LTC/DOGE/LINK, etc.; avg |weight| for BTC/ETH ~3.3%.  
- Portfolio typically holds ~75% of dates for the top names; BTC+ETH share ~12.9% of total |weight| (matches concentration summary).  
- Turnover contributions: BTC/ETH each ~10% of daily turnover; top names collectively drive most flow while remaining diversified.

Top 10 by avg_abs_weight:

| symbol  | avg_abs_weight | holding_ratio | turnover_share_pct |
| --- | --- | --- | --- |
| ETH-USD | 0.0332 | 75.2% | 10.0% |
| BTC-USD | 0.0332 | 75.2% | 10.4% |
| SOL-USD | 0.0309 | 76.7% | 9.3% |
| LTC-USD | 0.0232 | 66.5% | 7.4% |
| DOGE-USD | 0.0229 | 73.5% | 7.0% |
| LINK-USD | 0.0221 | 66.1% | 6.9% |
| XRP-USD | 0.0160 | 56.8% | 5.3% |
| AVAX-USD | 0.0142 | 57.5% | 4.3% |
| SHIB-USD | 0.0124 | 62.5% | 3.8% |
| ADA-USD | 0.0116 | 45.7% | 3.7% |

Full tables: `alphas101_symbol_stats_v1_adv10m.csv` and `alphas101_symbol_stats_top20_v1_adv10m.csv`.

## Implementation Notes
- Long-only with cash buffer; no funding-rate risk (no perps).  
- Daily rebalance in research; can bucket in implementation if needed.  
- All code and changes are under `scripts/research/`; no engine or deployment config changes.  
- Tear sheet: `artifacts/research/101_alphas/alphas101_tearsheet_v1_adv10m.pdf`.

## Key Artifacts (ADV>10M V1)
- Alphas: `alphas_101_v1_adv10m.parquet`  
- IC panel / selection / regimes: `alphas101_ic_panel_v1_adv10m_filtered.csv`, `alphas101_selected_v1_adv10m.csv`, `alphas101_regimes_v1_adv10m.csv`  
- Ensemble outputs: `ensemble_equity_v0.csv`, `ensemble_turnover_v0.csv`, `ensemble_weights_v0.parquet`  
- Metrics / TCA: `metrics_101_ensemble_filtered_v1.csv`, `metrics_101_ensemble_filtered_v1_costs_bps10/20/30/40/50.csv`  
- Beta / decay / bias / concentration / capacity:  
  - `alphas101_beta_vs_btc_v1_adv10m.csv`  
  - `alphas101_ic_decay_filtered_v1.csv`  
  - `alphas101_alpha008_bias_filtered_v1.csv`  
  - `alphas101_concentration_summary_v1_adv10m.csv`  
  - `capacity_sensitivity_v1.csv`  
- Tear sheet: `alphas101_tearsheet_v1_adv10m.pdf`

