# Alpha Ensemble – Legacy Full USD Universe (Research Strategy Note)

## 1. Strategy Objective
- Generate attractive, risk-managed absolute returns from a broad USD spot crypto universe (stablecoins excluded).
- Use a cross-sectional Alpha Ensemble (101-style + C-series diagnostics) to allocate to the strongest names and move to cash in fragile regimes.
- Targets: double-digit annualized returns, Sharpe > 1.5 over a full cycle, drawdowns contained to ~25–30%.

## 2. Universe & Data
- Venue: Coinbase USD spot pairs (ex-stablecoins); broader/unfiltered than the ADV>10M V1 stack.
- Frequency: daily bars (open, high, low, close, volume, vwap).
- Backtest window (legacy run): 2023-01-01 to 2024-12-31 (731 observations).

## 3. Signal Construction
- Inputs: classic 101-style price/volume/volatility alphas plus C-series (C1–C20) diagnostics.
- Selection/orientation: per-alpha IC vs forward returns; filter on magnitude, t-stat, stability; reduce multicollinearity.
- Portfolio signal: cross-sectional rank/scale of selected alphas each day; long-only; negative signals map to cash; turnover-aware normalization; cash earns ~4% research yield.

## 4. Regime Gating
- Regimes from C-series diagnostics:
  - Trend: directional, persistent moves with volatility expansion.
  - Mean-reversion: choppy/range-bound behavior.
  - Danger: correlation spikes, range expansion, crash-like skew.
- Policy: trade normally in Trend/Mean-rev; in Danger, weights go to cash (circuit breaker).

## 5. Performance Summary (2023-01-01 to 2024-12-31, legacy full USD ex-stablecoins)
- Gross (pre-cost):
  - CAGR: 67.29%
  - Vol (ann.): 29.88%
  - Sharpe: 2.25
  - Max drawdown: -25.13%
  - Sample length: 731 days
- Net-of-cost (per-side):
  - 10 bps: CAGR 58.39%, Sharpe 1.95, MaxDD -26.75%
  - 10 bps: CAGR 40.71%, Sharpe 1.50, MaxDD -24.11%
  - 20 bps: CAGR 33.33%, Sharpe 1.22, MaxDD -25.76%
  - 20 bps: CAGR 49.96%, Sharpe 1.67, MaxDD -28.35%
  - 30 bps: CAGR 26.33%, Sharpe 0.97, MaxDD -27.81%
  - 30 bps: CAGR 41.98%, Sharpe 1.40, MaxDD -29.92%
  - 40 bps: CAGR 19.70%, Sharpe 0.72, MaxDD -29.80%
  - 50 bps: CAGR 27.27%, Sharpe 0.91, MaxDD -33.07%
  - 50 bps: CAGR 13.42%, Sharpe 0.49, MaxDD -31.74%
- Regime breakdown (daily):
  - danger: 144 days, ann. return 31.82%, ann. vol 13.17%, Sharpe 2.42
  - mean_rev: 343 days, ann. return 60.30%, ann. vol 29.54%, Sharpe 2.04
  - trend: 244 days, ann. return 133.48%, ann. vol 36.72%, Sharpe 3.64
- Beta / correlation vs BTC-USD:
  - Corr=-0.04, Beta=-0.02, R²=0.002, t_beta=-1.10

## 6. Net-of-Cost Sensitivity (capacity sweep reference)
- 0 bps: Sharpe 1.78, CAGR 48.51%, MaxDD -22.63% (research capacity sweep; performance decays with higher costs).

## 7. Risk Profile & Max Drawdown
- Pre-cost MaxDD: -25.13% over the sample.
- 0 bps scenario: MaxDD -22.63% with Sharpe 1.78.
- Long-only plus danger-to-cash gating contains tail risk relative to buy-and-hold crypto beta.

## 8. Capacity & Caveats
- More aggressive, less liquidity-filtered than the ADV>10M V1 stack; capacity lower, turnover costs bite harder.
- ADV/participation constraints are lighter; higher slippage/fees erode performance faster.
- Treat this V0 as a high-octane archival snapshot; prefer V1 ADV>10M for more conservative liquidity hygiene.
