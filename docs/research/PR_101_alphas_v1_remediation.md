# 101 Alphas – V1 Remediation (ADV>10M, Ghost-Filtered IC, Capacity)

## Summary
Adds custom alphas 201–220 to the 101-alphas stack, applies ADV>10M liquidity remediation, ghost-data filtering for IC/decay/bias, standardized two-sided turnover, and refreshed capacity tooling + tear sheet. Both the base USD universe and the ADV>10M V1 stack are recomputed end-to-end.

## What Changed (Code)
- `scripts/research/alphas101_lib_v0.py`: new alphas `alpha_201`–`alpha_220`, custom registry, compute loop wiring.
- `scripts/research/run_101_alphas_compute_v0.py`: ADV10M view flag.
- IC/decay/bias scripts: ghost filters (volume>0, close≠prev_close) and labels.
- `scripts/research/alphas101_metrics_v0.py`: turnover wording; V1 output.
- `scripts/research/alphas101_tca_v0.py`: turnover convention noted.
- `scripts/research/alphas101_capacity_sensitivity_v1.py`: cost/Sharpe capacity table.
- `scripts/research/alphas101_tearsheet_v0.py`: prefers V1 artifacts, adds capacity section.
- `scripts/research/create_usd_universe_adv10m_view.py`: build ADV>10M view.
- Docs: `docs/research/101_alphas_v1_overview.md`.

## Key Results (V1 ADV>10M)
- Gross (2023-01-01–2025-01-01, 731d): CAGR ~48.5%, Vol ~27.2%, Sharpe ~1.78, MaxDD ~-22.6%.
- Net-of-cost Sharpe/CAGR: 10bps ~1.50/40.7%; 20bps ~1.22/33.3%; 30bps ~0.97/26.3%; 40bps ~0.72/19.7%; 50bps ~0.49/13.4%.
- Beta vs BTC: corr ~-0.041, beta ~-0.023 (diversifying).
- IC decay: filtered mean IC rises modestly to ~0.03 by horizon 4 then decays; no ghost plateau.
- Capacity: turnover ~0.148 avg; capacity_sensitivity_v1.csv shows smooth Sharpe decay through 50bps.
- Concentration: BTC+ETH share ~12.9%; top weights spread across BTC, ETH, SOL, LTC, DOGE, LINK, XRP, AVAX, SUI, BONK.

## Risk & Ops Considerations
- Long-only, long+cash; no perps → no funding-rate risk.
- Two-sided equity turnover (0.5 * Σ|w_t - w_{t-1}|) used consistently in metrics/TCA.
- Ghost bars excluded (volume>0, close≠prev_close) in IC/decay/bias.

## Artifacts
- Alphas: `alphas_101_v1_adv10m.parquet`
- IC/selection/regimes: `alphas101_ic_panel_v1_adv10m_filtered.csv`, `alphas101_selected_v1_adv10m.csv`, `alphas101_regimes_v1_adv10m.csv`
- Ensemble: `ensemble_equity_v0.csv`, `ensemble_turnover_v0.csv`, `ensemble_weights_v0.parquet`
- Metrics/TCA: `metrics_101_ensemble_filtered_v1.csv`, `metrics_101_ensemble_filtered_v1_costs_bps10/20/30/40/50.csv`
- Beta/decay/bias/concentration/capacity: `alphas101_beta_vs_btc_v1_adv10m.csv`, `alphas101_ic_decay_filtered_v1.csv`, `alphas101_alpha008_bias_filtered_v1.csv`, `alphas101_concentration_summary_v1_adv10m.csv`, `capacity_sensitivity_v1.csv`
- Tear sheet: `alphas101_tearsheet_v1_adv10m.pdf`

