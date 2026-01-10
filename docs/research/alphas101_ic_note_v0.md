# Alpha Ensemble – ADV>10M V1 (Research Strategy Note)

## 1. Strategy Objective
- Deliver attractive, risk-managed absolute returns from liquid USD spot crypto (ADV>10M subset).
- Use a cross-sectional Alpha Ensemble (101-style plus custom alphas) with regime-aware gating.
- Targets: double-digit annualized returns, Sharpe > 1.5 over a full cycle, controlled drawdowns.

## 2. Universe & Data
- Venue: Coinbase USD spot pairs (ex-stablecoins), ADV > $10M filter; ~82 symbols in 2023–2025 window.
- Frequency: daily bars (open, high, low, close, volume, vwap).
- Backtest window: 2023-01-01 to 2025-01-01 (731 days).

## 3. Signal Construction
- Signal set: classic “101 alphas” + 20 custom alphas (alpha_201–alpha_220) + C-series regime features (alpha_c16–alpha_c20).
- Workflow: compute alphas daily, build IC panel vs forward returns, select/orient alphas by IC magnitude/t-stat/stability, reduce multicollinearity.
- Portfolio signal: cross-sectional rank/scale of selected alphas; long-only; negative signals map to cash; target gross ≈ 1.0; turnover-aware normalization; cash earns ~4%.

## 4. Regime Gating
- Regimes (C-series diagnostics):
  - Trend: directional, persistent moves, volatility expansion.
  - Mean-reversion: choppy/range-bound.
  - Danger: correlation spikes, range expansion, crash-like skew.
- Policy: trade normally in Trend/Mean-rev; in Danger, weights go to cash (circuit breaker).

## 5. Performance Snapshot (ADV>10M V1, 2023-01-01 to 2025-01-01, 731 days)
- Gross (metrics_101_ensemble_filtered_v1.csv):
  - CAGR: 48.5%
  - Vol: 27.2%
  - Sharpe: 1.78
  - Max DD: -22.6%
- Net of costs (capacity_sensitivity_v1.csv):
  - 10 bps: Sharpe 1.50, CAGR 40.7%, Max DD -24.1%
  - 20 bps: Sharpe 1.22, CAGR 33.3%, Max DD -25.8%
  - 30 bps: Sharpe 0.97, CAGR 26.3%, Max DD -27.8%
  - 40 bps: Sharpe 0.72, CAGR 19.7%, Max DD -29.8%
  - 50 bps: Sharpe 0.49, CAGR 13.4%, Max DD -31.7%
- Beta / correlation vs BTC (alphas101_beta_vs_btc_v1_adv10m.csv):
  - corr ≈ -0.041, beta ≈ -0.023 (diversifying, small negative beta)

## 6. Regime-Level & Concentration Highlights
- Regime gating: long-only with danger-to-cash reduces drawdowns versus naïve exposure.
- Concentration (alphas101_concentration_summary_v1_adv10m.csv): BTC+ETH share ≈ 12.9% of avg gross; weights spread across BTC, ETH, SOL, LTC, DOGE, LINK, XRP, AVAX, SUI, BONK.

## 7. Signal Quality & Decay
- IC panel (alphas101_ic_panel_v1_adv10m_filtered.csv): horizon-1 mean ICs low but positive after ghost filtering (~0.009–0.03 with modest t-stats); selection keeps top t-stat signals.
- IC decay (alphas101_ic_decay_filtered_v1.csv): mean IC rises to ~0.03 at horizon 4 then decays; no ghost plateau.
- Bias check (alphas101_alpha008_bias_filtered_v1.csv): baseline IC ~0.0095, lag1 ~0.0081; positive after filtering → predictive, not look-ahead.

## 8. Capacity & Turnover
- Two-sided turnover (ensemble_turnover_v0.csv): ~0.148 average per day (median 0.15).
- Capacity sensitivity (capacity_sensitivity_v1.csv): Sharpe decays smoothly from 1.78 (0 bps) to 0.49 (50 bps); provides cagr/vol/max_dd by cost tier.

## 9. Role & Use Case
- Absolute-return, low-beta satellite vs BTC/ETH beta.
- Systematic cross-sectional alpha in liquid names, with explicit regime gating and cash overlay.
- Research-only scope; no engine/deployment changes implied.

## Strategy Overview
- Daily long-only, long+cash ensemble over Coinbase USD spot pairs (stablecoins excluded).
- Objective: attractive absolute returns with controlled drawdowns via a cross-sectional alpha stack plus regime-aware risk management.
- Targets: double-digit annual returns, Sharpe > 1.5 over a full cycle, drawdowns around 25–30%.

## Universe & Data
- Venue: Coinbase USD spot pairs (non-stablecoin bases such as BTC-USD, ETH-USD, large/mid-cap alts).
- Frequency: daily bars (open, high, low, close, volume, vwap), roughly one bar per calendar day.
- Backtest window: 731 trading days (2023-01-01 to 2024-12-31 in the canonical run).

## Signal Construction
- Inputs: classic 101-style price/volume/volatility alphas plus custom C-series features (C1–C20).
- C-series coverage:
  - Mean-reversion signatures (e.g., wick reversion, volume exhaustion).
  - Trend/breakout behavior (e.g., VWAP trend, volatility breakout).
  - Regime diagnostics (Hurst proxy, tail-risk, liquidity fragility, correlation spikes, range expansion).
- Workflow: compute all alphas daily per asset; build per-alpha ICs vs forward returns; select/orient alphas by IC magnitude, t-stat, stability, and multicollinearity checks; average selected signals into a composite per asset/day.

## Portfolio Construction
- Long-only, long+cash; negative signals map to cash (no shorting).
- Cross-sectional rank/scale of composite signals each day; only positive signals receive weight.
- Turnover-aware normalization targets ~40–50% gross long on average in canonical configuration.
- Cash earns a modest yield assumption (~4% annualized).

## Regime Gating (Risk Management)
- Regimes derived from C-series “Oxford-style” features:
  - Trend: directional, persistent moves with volatility expansion in direction of price.
  - Mean-reversion: choppy/range-bound; fade/reversion signals dominate.
  - Danger: correlation spikes, range expansion, or crash-like skewness.
- Gating:
  - Trend/Mean-rev: trade normally (subject to signal strength and liquidity).
  - Danger: go to cash (as close as practical), acting as a circuit breaker.

## Performance Snapshot (Legacy Full-USD Run)
- Period: 2023-01-01 to 2024-12-31, 731 daily observations.
- Gross (pre-cost): CAGR 67.29%, Vol 29.88%, Sharpe 2.25, MaxDD -25.13%.

## Net-of-Cost Sensitivity (per-side costs, indicative)
- 10 bps: net CAGRs in ~40–60% range, Sharpe > 1.5, MaxDD mid-20s.
- 20–50 bps: positive but decaying performance as frictions rise; consistent with non-trivial yet manageable turnover.

## Beta & Correlation vs BTC-USD
- Corr ≈ -0.04, Beta ≈ -0.02, R² ≈ 0.002, t_beta ≈ -1.10.
- Interpretation: behaves like an idiosyncratic cross-sectional alpha book, approximately market-neutral to BTC over the sample.

## Regime-Level Performance (Legacy Run)
- Danger: ann. return ~31.8%, ann. vol ~13.2%, Sharpe ~2.4 (typically low exposure; returns mostly from prior risk-on and cash).
- Mean-reversion: ann. return ~60.3%, ann. vol ~29.5%, Sharpe ~2.0.
- Trend: ann. return ~133.5%, ann. vol ~36.7%, Sharpe ~3.6.
- Takeaway: best risk-adjusted returns in Trend; still attractive Sharpe in other regimes with controlled drawdowns via Danger gating.

## Risk & Capacity Considerations
- Turnover: high but managed; capacity governed by ADV filters, per-name caps, participation limits.
- Tail risk: mitigated via long-only construction + regime gating; historical MaxDD ~25% is consistent with a high-Sharpe systematic crypto strategy.

## Role in Portfolio
- Intended as an absolute-return, low-beta satellite within a broader crypto or multi-asset allocation.
- Offers diversification vs BTC/ETH beta; systematic exposure to cross-sectional alpha in liquid alts; explicit risk controls via regime gating and long+cash design.
