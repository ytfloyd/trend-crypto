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

---

## Amendment A (2026-06-13) — tiered-cost sensitivity (pre-registered)

Tests whether the headline survives realistic, liquidity-dependent costs (the 30 bps flat
assumption may understate slippage on smaller names). Costs are applied per name, per bar, as
one-way turnover × the name's cost, replicating the engine's accounting exactly. **GC0:** the
flat-30 control must reproduce the 2.84 headline, or the harness is wrong and the run is void.

**Liquidity tier** by point-in-time 20-day dollar-ADV: T1 ≥ $50M · T2 $20–50M · T3 $5–20M · T4 < $5M.

**Part A — tiered flat-cost scenarios** (one-way bps, T1/T2/T3/T4):
S0 control 30/30/30/30 · S1 benign 10/20/40/70 · S2 base 20/40/70/120 · S3 punitive 35/70/130/220.

**Part B — participation/impact capacity curve:** per-name cost = spread_tier + c·√(participation)
bps, participation = AUM·(one-way traded fraction)/ADV, c = 100 bps (square-root law). Sweep
AUM ∈ {5, 25, 50, 100, 250} $M; report OOS Sortino vs AUM and the soft capacity (largest AUM with
OOS Sortino > 2.0).

**Gates:** GC0 S0 ≈ 2.84 (reconciliation) · GC1 OOS Sortino > 2.0 under S2 · GC2 > 1.5 under S3 ·
GC3 capacity AUM (informational). **Harness:** `scripts/research/k2_atlas/run_medallion_costs.py`.

---

## Amendment B (2026-06-13) — cost-robust universe search (pre-registered)

The top-100 universe fails the cost gate (Amendment A) because its alpha sits in the illiquid tail.
This searches for a LIQUID-leaning universe that survives realistic costs, trading capacity for
cost-robustness. For each universe — top_10/25/50/100 and ADV floors ≥$50M / ≥$20M, all
point-in-time, survivorship-free, within-universe rank, frozen flagship params — report OOS Sortino
under **S0 flat-30 (reference)** and **S2 realistic-tiered** costs, plus the capacity curve
(S2 spreads + √-impact, c=100bps) over AUM ∈ {5,25,50,100,250} $M.

**Gate GC-B:** a universe is **cost-robust** if its S2-tiered OOS Sortino **> 2.0**; **deployable**
if additionally its **soft capacity** (largest AUM with S2+impact Sortino > 2.0) **≥ $5M**. If no
universe is cost-robust, the concept does not graduate. **Harness:**
`scripts/research/k2_atlas/run_medallion_cost_universe.py`.
