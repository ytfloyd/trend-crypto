# kuma_trend – v0 Trend-Following Stack (BTC/ETH/SOL/SUI/BCH)

## Summary
Long-only breakout trend strategy: 20-day breakout gated by MA(5) > MA(40), inverse-vol(20) sizing with 5% cash buffer, 2×ATR(20) trailing stops, uninvested cash earns 4% annualized. Universe is BTC, ETH, SOL, SUI, BCH. Research-only; no prod wiring.

## What Changed (Code)
- `scripts/research/kuma_trend_lib_v0.py`: breakout + MA filter, ATR(20) trailing stops, inverse-vol sizing with cash buffer, portfolio/equity calc.
- `scripts/research/run_kuma_trend_backtest_v0.py`: DuckDB loader, runs backtest, writes weights/equity/positions/turnover.
- `scripts/research/kuma_trend_metrics_v0.py`: metrics (full, pre_2020, 2020_2021, 2022, 2023_plus).
- README: added kuma_trend usage.
- Docs: `docs/research/kuma_trend_overview_v0.md`.

## Key Results (v0 backtest, 2017-01-01 to 2025-01-01; equity sample 2922 days)
From `metrics_kuma_trend_v0.csv`:
- Full: CAGR ~165.8%, vol ~58.9%, Sharpe ~2.82, MaxDD ~-100% (compounding-sensitive; subperiod DDs more informative).
- 2020_2021: Sharpe ~5.15, MaxDD ~-32%.
- 2022: negative Sharpe (-0.25), MaxDD ~-27%.
- 2023_plus: Sharpe ~2.59, MaxDD ~-41%.
- Turnover: mean ~0.048, 25/50/75 pct ~0 / 0 / 0.0065 (low).

## Risk & Ops Considerations
- Long-only, no leverage; cash buffer 5% and cash yield 4%.
- Two-sided equity turnover: 0.5 * Σ |w_t - w_{t-1}|.
- Stops: 2×ATR(20) trailing from entry; no secondary exit on MA/BO fail (relies on stop).
- Costs/slippage not applied in v0; future work to add 10–20 bps per side.

## Artifacts
- Equity/weights/positions/turnover: `artifacts/research/kuma_trend/`
  - `kuma_trend_equity_v0.csv`
  - `kuma_trend_weights_v0.parquet`
  - `kuma_trend_positions_v0.parquet`
  - `kuma_trend_turnover_v0.csv`
- Metrics: `metrics_kuma_trend_v0.csv`
- Overview: `docs/research/kuma_trend_overview_v0.md`
- (No tear sheet yet; can be added later if desired.)

## Checklist / Next
- [ ] Add cost model (10–20 bps) and rerun metrics
- [ ] Add BTC/ETH benchmark comparison
- [ ] Risk review (DD depth/length vs benchmarks)
- [ ] Optional: add tear sheet page if needed

