# Spot-Convexity Sleeve — Plan, Status & Index

**Branch:** `research/spot-convexity` · **Universe:** crypto spot (Coinbase USD daily) ·
**Philosophy:** stop-aware right-tail capture, not a generic indicator ensemble. Pre-register, then
run; the transparent baseline must be beaten OOS before any model is trusted (medallion-sleeve
discipline). Routes to the **convexity pipeline** (gates on convexity, not Sharpe).

## Phases

| Phase | Deliverable(s) | Status |
|---|---|---|
| **0. Foundation** | Research design (#1), feature taxonomy (#2), **stop-aware R-multiple labeler (#3)** + unit tests, transparent baseline score (#6), this plan | ✅ **done** |
| 1. Data + labels | Wire labeler to the lake; build point-in-time panel; generate R-multiple labels across the universe; sanity-check the trade-outcome distribution | ⬜ next |
| 2. Baseline empirics | Run `spot_convexity_score` → rank → trades; full distribution metrics (#4) with costs/slippage/gap; sweep stop/trail/horizon sensitivity | ⬜ |
| 3. Hypothesis tests | Brief §5 H1–H10 (compression, trend, accel, stop viability, whipsaw, gap, right-tail, volume, pullback, stop-aware-vs-fwd-return target) | ⬜ |
| 4. Models | Regression (R), classification (>+1/+2/+3R), risk (stop-before-+1R), ranking (#A–D) — must beat baseline OOS | ⬜ |
| 5. Validation | Walk-forward OOS by time + regime; deflated/PBO on any selected config; full execution realism (#5,#7) | ⬜ |
| 6. Portfolio | Sizing by initial risk, portfolio heat, correlation/crowding, correlated stop-outs (#8) | ⬜ |
| 7. Package | Exec summary, results, OOS, feature importance, trade-distribution, failure analysis, portfolio sim, recommendation, open questions (#10) | ⬜ |

## Acceptance criteria → where answered

| # | Criterion | Phase |
|---|---|---|
| 1 | Positive expected R OOS? | 2, 5 |
| 2 | Right-tail winners vs stop-outs + slippage? | 2, 4 (trade-dist) |
| 3 | Which feature groups most predictive? | 4 (importance) |
| 4 | Stop-aware target beats forward-return target? | 3 (H10) |
| 5 | Robust across time / assets / regimes? | 5 |
| 6 | Sensitivity to stop / trail / horizon? | 2 |
| 7 | Performance after costs + realistic stop execution? | 2, 5 |
| 8 | Implementable within liquidity / capacity / drawdown? | 6 |
| 9 | Main false-positive regimes? | 3, 7 (failure analysis) |
| 10 | Risk controls before live? | 6, 7 |

## Failure modes → mapped controls (brief §9)
false breakout → C+K+I (volume + path) · trend exhaustion → B (extreme dist_ma200) · late entry → J
(prefer pullback) · stop-in-noise → F (`support_in_atr`, `stop_beyond_support`) · gap risk → G ·
illiquidity → K · vol-spike-no-follow → D+A direction · momentum-but-choppy → I · compression-down →
E+B sign · crowding/correlated stop-outs → portfolio phase (corr clustering, portfolio heat).

## Foundation artifact index

| Artifact | Path |
|---|---|
| Research design (#1) | `docs/research/spot_convexity/01_research_design.md` |
| Feature taxonomy (#2) | `docs/research/spot_convexity/02_feature_taxonomy.md` |
| R-multiple labeler (#3) | `scripts/research/spot_convexity/labeler.py` |
| Labeler unit tests | `tests/test_spot_convexity_labeler.py` (10 tests, green) |
| Causal features | `scripts/research/spot_convexity/features.py` |
| Baseline score (#6) | `scripts/research/spot_convexity/baseline_score.py` |

## Key pre-registered decisions (locked)
- **Target = realized R-multiple** after ATR stop + upward-only ATR trailing stop; **gap-through-stop
  modeled** (exit at open if it gaps the stop). Default stop_mult=2, trail_mult=3, max_horizon=60d.
- **Entry = next-bar open**; features end at signal date; label starts at entry (disjoint windows).
- **Headline metric = the full R-distribution** (mean/median R, win rate, stop-out rate, % >+1/+2/+3R,
  EV/trade), not mean forward return.
- **Costs/slippage/gap in every reported number**; baseline-beats-or-investigate rule.

## Open questions (carried)
- Crypto "sector/crowding" proxy: correlation clustering vs token tags — decide in phase 6.
- `hist_right_tail` precise construction (lagged forward-max stat) vs the current up-tail proxy —
  refine in phase 1/3.
- Bar frequency: daily for now; intra-day stop fills are approximated on daily OHLC (open-gap +
  intraday-breach model) — revisit if hourly execution detail is needed.
