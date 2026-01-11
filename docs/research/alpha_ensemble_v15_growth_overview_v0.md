# Alpha Ensemble – V1.5 Growth Sleeve (Top50 ADV>10M)

## Objective
- Target aspirational profile: CAGR > 70%, Sharpe > 1.5, MaxDD > -35% over 2023–2024.
- Growth-oriented sleeve that gates risk with regime filters and convex exits.

## Universe
- Dynamic Top 50 symbols by rolling 30-day dollar volume.
- ADV filter: >= $10M.
- Views: `bars_1d_usd_universe_clean_top50_adv10m` (daily). 4H not used in v0.

## Regime Filter
- ADX(14) >= 25.
- Price above Ichimoku cloud (9/26/52, disp 26); if in/below cloud => cash.

## Signal Engines
- **Slow (80%)**: Bollinger width breakout > 1.5x 20d mean, price > upper Keltner(20, 2*ATR20), volume > 1.5x ADV30.
- **Fast (20%)**: DEWMA(10) > DEWMA(40) (daily for v0; interface ready for 4H later).
- Exposure multiplier = 0.8*slow_on + 0.2*fast_on; regime must be ON.

## Exits / Convexity
- Chandelier stop: highest high since entry – 3*ATR14.
- PSAR flip exit (long-only).
- Gap exit: if open < stop – 1*ATR14, exit at open with 25 bps slippage penalty.

## Risk & Sizing
- Per-name risk budget: 50 bps of NAV vs stop distance (3*ATR14) → raw weight.
- Volatility parity via ATR stop distance; exposure multiplier applied.
- Cluster cap: corr > 0.7 → connected components capped at 40% aggregate weight.
- Portfolio vol target: 20% annualized; scalar capped at 1.5x.
- Max single-name cap: 8% after scaling; renorm down if breached.
- Cash earns implicit 0% in the backtest (can add yield later).

## Deliverables / Artifacts (v0)
- Equity: `artifacts/research/alpha_ensemble_v15_growth/growth_equity_v0.csv`
- Weights: `artifacts/research/alpha_ensemble_v15_growth/growth_weights_v0.parquet`
- Trades: `artifacts/research/alpha_ensemble_v15_growth/growth_trades_v0.parquet`
- Metrics: `artifacts/research/alpha_ensemble_v15_growth/metrics_growth_v15_v0.csv`
- Benchmark ETH: `artifacts/research/alpha_ensemble_v15_growth/benchmark_eth_usd_equity_v0.csv`
- Tear sheet: `artifacts/research/alpha_ensemble_v15_growth/alpha_ensemble_v15_growth_tearsheet_v0.pdf`
