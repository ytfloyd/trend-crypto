# Medallion Lite — pre-registered validation protocol

**Status:** pre-registered. Committed to git *before* the run; results are reported only as this
protocol specifies. Any deviation is logged as an amendment with its own commit.

## Objective
Establish a single, defensible, auditable performance figure for Medallion Lite suitable for
investor / head-of-trading review. **Frozen-param result is the headline; walk-forward is reported
as a labeled upper bound.** No flip-flops: one protocol, run once, report what survives.

## Strategy (fixed — not selected from a search)
- **Signal:** the 5-factor cross-sectional composite (momentum, volume surge, realised vol,
  proximity-to-high, risk-adjusted momentum), equal-convention weights as in `medallion_lite.factors`.
  *The 100-factor TA-Lib zoo and all ensembling are excluded — they are documented negative results
  (`medallion_factor_count_experiment.md`, `medallion_bagging_experiment.md`): design selection
  fails PBO (0.70–0.77), so we use the simplest pre-specified composite.*
- **Universe:** point-in-time top-100 by 20-day trailing ADV, survivorship-free; within-universe
  cross-sectional ranking.
- **Portfolio:** flagship params, **frozen** (entry 0.65 / exit 0.40 / trailing stop 0.15 /
  max-hold 336h / rebalance 24h / ≤25 positions / ≤10% per name). No per-fold parameter fitting.
- **Costs:** 30 bps one-way (headline); cost sensitivity reported at {0,10,20,30,50} bps.
- **Data:** Coinbase USD-pair OHLCV, 2021-01-01 → 2026-06-01. **OOS = 2023-01-01 onward.**

## What is computed
1. **Headline (frozen params):** full-sample and OOS Sortino, Sharpe, Calmar, CAGR, ann. vol,
   max drawdown, daily hit-rate — strategy vs BTC buy-and-hold.
2. **Upper bound (walk-forward):** param-frozen walk-forward OOS Sortino (select-on-train / freeze /
   score-on-test; folds 2021-22→2023, →2024, →2025-26), reported explicitly as an upper bound that
   includes a parameter-selection layer.
3. **Per-fold OOS:** frozen-param Sortino in each test window (2023, 2024, 2025-26) to expose decay.
4. **Significance:** Probabilistic Sharpe Ratio (PSR vs 0) on the frozen OOS returns
   (`src/afml/backtest_stats`). The headline strategy is pre-specified (1 trial) ⇒ PSR, not a
   multiple-testing haircut; the design-space PBO above is the separate evidence that we did not
   data-mine the construction.
5. **Cost sensitivity:** OOS Sortino at {0,10,20,30,50} bps.

## Acceptance gates (pre-registered, pass/fail)
- **G1** Headline (frozen) OOS Sortino **> BTC OOS Sortino**.
- **G2** Frozen OOS PSR(0) **≥ 0.95**.
- **G3** Not driven by a single fold: **≥ 2 of 3** OOS folds have positive Sortino.
- **G4** Cost-robust: OOS Sortino **> 1.5 at 30 bps** and **> 1.0 at 50 bps**.
A figure is "validated" only if G1–G4 pass. Failing gates are reported as failures, not hidden.

## Auditability
- **Deterministic:** the 5-factor path uses no RNG; folds and params are fixed ⇒ identical results
  on re-run at the same commit + data.
- **Provenance manifest** (`artifacts/medallion_audit/medallion_audit_<commit>.json`): git commit +
  dirty flag, package versions (python/numpy/pandas/duckdb/scipy), data-lake path, hourly row count,
  date range, symbol count, universe definition, params, costs, fold boundaries, and every result.
- **Re-derivable:** daily return series (strategy frozen, strategy WF, BTC) written to CSV.
- **Reproduce:** `PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_audit.py`

## Reporting
On completion, revise the strategy card, registry `validation` block, and investor report to the
**frozen headline (with its uncertainty band and the WF upper bound)** — replacing the prior
walk-forward-selected 2.95, which carried a parameter-selection optimism this protocol removes.

**Pre-registered by:** research · **Harness:** `scripts/research/k2_atlas/run_medallion_audit.py`
