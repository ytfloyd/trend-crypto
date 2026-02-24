---

# PROPOSAL: Cross-Asset Trend Validation

### From Single-Asset Research to Deployable Strategy Architecture

**NRT Research** | Asymmetry Research — Quantitative Strategy Group | February 2026

**To:** Desk Head
**From:** Russell
**Re:** Proposed methodology for cross-asset validation of ETH-USD trend findings

---

## Background

The comprehensive ETH-USD research (Parts 1–11, 38 pages) established the following on a single asset:

- **534 of 13,293 strategies survive Bonferroni correction** (Sharpe ≥ 1.38)
- **Signal families matter:** DEMA, Supertrend, CCI, Aroon cluster at the top
- **ATR stops dominate for noisy-entry signals** (EMA: 88% win rate), fixed stops for regime-sensitive signals (ADX: 0%)
- **The optimal TIM is ~42%**, and a TIM-filtered ensemble achieves Sharpe 1.37 with max drawdown -48% (vs B&H Sharpe 1.11 / -94%)
- **Walk-forward TIM selection works** — ρ = 0.99, zero Sharpe decay, closing the look-ahead gap
- **Bonferroni survivors beat B&H at every drift level tested**, up to 300% annualized

Every one of these findings is in-sample on one asset. The paper says so explicitly. This proposal describes the test that determines which findings replicate and which are ETH path dependence.

---

## What We're Testing

Three hypotheses, ranked by importance:

**H1: Signal family dominance is structural, not asset-specific.**
Do DEMA, Supertrend, CCI, and Aroon produce the highest Bonferroni survival rates across assets? This is the most consequential test. If family rankings are stable cross-asset, we have a deployable signal selection framework. If they're not, the ETH results are overfitted noise.

**H2: The ~42% TIM optimum is structural, not asset-specific.**
Does the Sharpe-vs-TIM curve peak near 40% on other assets, or is that an artifact of ETH's particular trend/chop regime mix? If the optimum shifts materially (e.g., 25% on BTC, 60% on SOL), the TIM-targeting framework needs asset-specific calibration, which weakens the case for a unified strategy architecture.

**H3: Stop-type recommendations are structural, not asset-specific.**
Does ATR-for-EMA / fixed-for-ADX hold on assets with different volatility clustering? This was flagged in the paper as "precisely the kind of finding that could reflect ETH-specific path dependence." The cross-asset test is decisive.

---

## Asset Universe

We propose a two-tier structure:

### Tier 1: Primary validation (full sweep, full extension analysis)

| Asset | History | Bars | Rationale |
|---|---|---|---|
| **BTC-USD** | 2017–2026 | 3,330 | Same coverage as ETH. Largest asset. Mandatory. |
| **SOL-USD** | 2021–2026 | 1,703 | High-growth alt. Different vol structure. Desk interest. |

These receive the full 13,293-configuration sweep and all relevant extension analyses (Bonferroni survival, TIM curve, stop-type decomposition).

**Limitation — SOL history asymmetry:** SOL's 4.7-year sample (June 2021–February 2026) includes only one bear cycle and no 2017-era price action. The per-asset Bonferroni threshold is correctly higher for SOL (z / √T increases with shorter T), but the qualitative comparison of family rankings between ETH (9 years, two full bull/bear cycles) and SOL (4.7 years, one cycle) may not be apples-to-apples. If family rankings differ on SOL, we will not be able to determine whether the difference is structural or simply reflects regime coverage. This is a known limitation of the SOL comparison specifically and will be reported as such.

### Tier 2: Zero-reoptimization replication (daily-frequency survivors only)

| Asset | History | Bars | Rationale |
|---|---|---|---|
| **LTC-USD** | 2017–2026 | 3,330 | Same coverage as ETH/BTC. Older, lower-vol alt. |
| **LINK-USD** | 2019–2026 | 2,424 | Mid-cap DeFi. Different sector exposure. |
| **ATOM-USD** | 2020–2026 | 2,223 | L1 alt. Different ecosystem (Cosmos). |

Tier 2 assets run only the daily-frequency Bonferroni survivors from ETH (~380 strategies) plus Buy & Hold. No re-optimization, no per-asset tuning — the exact strategies that survived on ETH are applied to completely different assets. This is arguably the sharpest test in the entire study: if strategies retain edge on an asset they were never calibrated to, that is stronger evidence of structural signal than anything achievable from a full per-asset sweep. Full parameter sweeps are unnecessary — we're testing family-level replication, not re-optimizing per asset.

### Why these five

- **Coverage span:** Three assets (BTC, LTC, ETH) have 9+ years of history, capturing two full bull/bear cycles. Two (SOL, AVAX/LINK/ATOM) are shorter but represent the newer, higher-vol alt ecosystem.
- **Structural diversity:** BTC has the lowest vol and highest institutional correlation to macro. SOL has the highest vol and most momentum-driven price action. LTC and LINK are mid-tier. ATOM is a different L1 ecosystem.
- **Data availability:** All five are Coinbase Advanced spot pairs already in our daily bars cache. No new data ingestion required.

---

## Methodology

### Phase 1: Full sweep on Tier 1 assets (~5 minutes compute)

For each Tier 1 asset (BTC, SOL):

1. **Run the existing sweep engine** (`run_eth_trend_sweep_v2.py`) with the symbol parameterized. All 493 signals × 3 frequencies × 9 stop variants = 13,293 configurations. Binary long/cash, one-bar lag, 20 bps round-trip. Identical to ETH.

2. **Compute per-asset Bonferroni threshold.** The threshold depends on sample length: SOL has 4.7 years vs ETH's 9.1 years, so the Sharpe threshold is higher (z / √T increases). This is correct — shorter samples should require stronger evidence.

3. **Produce the same results table** (`results_v2.csv`) for each asset.

**Code change required:** Parameterize `SYMBOL` in `run_eth_trend_sweep_v2.py` (currently hardcoded to `"ETH-USD"`). The signal dispatch, backtest engine, stop overlays, and performance computation are fully generic — they take price arrays, not asset-specific logic. This is a ~10-line change.

### Phase 2: Cross-asset comparison (the new analytical work)

**Test H1 — Family survival rates:**
- For each asset, compute family-level Bonferroni survival rate (% of family's configs that survive correction)
- Rank families by survival rate on each asset
- Report rank correlation (Spearman ρ) of family rankings across all asset pairs
- **Decision rule:** If Spearman ρ > 0.6 for the majority of asset pairs, family dominance is structural. If ρ < 0.3, it's path dependence. These thresholds are pre-specified decision points chosen in advance, not derived from theory. The 0.3–0.6 range is deliberately acknowledged as ambiguous and will require judgment — we state this now rather than rationalizing it after the fact.

**Test H2 — TIM optimum portability:**
- For each Tier 1 asset, fit the Sharpe-vs-TIM curve (1% bins, same methodology as Part 10)
- Report the empirical TIM optimum and 90% bootstrap CI per asset
- **Decision rule:** If all assets' CIs overlap, the optimum is portable. If they don't overlap, TIM targeting needs per-asset calibration.

**Test H3 — Stop-type microstructure:**
- For each Tier 1 asset, run the matched-pair ATR vs. fixed stop decomposition by signal family (same methodology as Part 8)
- Report whether ATR-for-EMA and fixed-for-ADX hold on each asset
- **Decision rule:** If the stop-type recommendation is consistent for ≥ 4 of 5 assets, it's structural. If it flips, it's path dependence.

### Phase 3: Tier 2 zero-reoptimization replication (~2 minutes compute) — PRIMARY RESULT

For each Tier 2 asset (LTC, LINK, ATOM):

1. Run the ~380 daily-frequency ETH Bonferroni survivors on that asset's data. Same signal parameters, same stops. No per-asset tuning whatsoever.

2. Report: what fraction of ETH survivors also beat that asset's Buy & Hold on Sharpe? What fraction maintain positive skewness and drawdown compression?

3. This is the sharpest test of structural signal in the entire study. A full per-asset sweep can always find strategies that work — the relevant question is whether strategies selected on one asset transfer to another without re-optimization. If they do, signal family dominance is structural. If they don't, the ETH findings are path dependence. This test will be framed as a primary result in the report, not a secondary replication check.

### Phase 4: Walk-forward ensemble portability

For BTC-USD (the only Tier 1 asset with identical 9-year coverage):

1. Run the walk-forward TIM selection from Part 11: train TIM on 2017–2021, select strategies in the [37%, 47%] band, evaluate the ensemble on 2022–2026.

2. Compare walk-forward ensemble Sharpe, drawdown, and skewness to BTC Buy & Hold.

3. **This is the deployment-readiness test.** If the walk-forward TIM ensemble works on both ETH and BTC without per-asset tuning, the architecture is deployable.

---

## Infrastructure Reuse

| Component | Status | Change Required |
|---|---|---|
| Signal dispatch (493 signals) | Ready | None — takes generic price arrays |
| Trailing stop overlays (9 variants) | Ready | None |
| Backtest engine | Ready | None |
| Performance computation | Ready | None |
| Data pipeline | Ready | Already has all 5 assets in daily bars cache |
| Bonferroni correction | Ready | Adjust threshold for per-asset sample length |
| TIM curve analysis (Task 4) | Ready | Parameterize symbol |
| Walk-forward TIM (Task 5) | Ready | Parameterize symbol |
| Stop-type decomposition (Task 2) | Ready | Parameterize symbol |
| Sweep runner | Needs 10-line edit | Parameterize `SYMBOL` from CLI arg |

Estimated new code: ~200 lines for the cross-asset comparison script (Phase 2). Everything else is re-execution of existing infrastructure.

---

## Outputs

1. **Per-asset results tables** (BTC, SOL, LTC, LINK, ATOM) — same format as `results_v2.csv`

2. **Cross-asset family survival heatmap** — families × assets, color-coded by survival rate. The single most important exhibit.

3. **TIM optimum comparison** — Sharpe-vs-TIM curves overlaid for all Tier 1 assets with confidence bands

4. **Stop-type portability table** — ATR win rate by signal family, per asset

5. **Tier 2 replication table** — % of ETH survivors retaining edge on each Tier 2 asset

6. **Walk-forward ensemble on BTC** — equity curves and comparison table, same format as Part 11

7. **Integrated report** — Part 12 of the comprehensive paper, or standalone companion piece (your preference)

---

## Timeline

| Phase | Work | Estimated Time |
|---|---|---|
| Phase 1: Full sweep (BTC, SOL) | Parameterize + run | 30 min |
| Phase 2: Cross-asset analysis | New comparison script | 2 hr |
| Phase 3: Tier 2 replication | Run ETH survivors on LTC/LINK/ATOM | 30 min |
| Phase 4: Walk-forward on BTC | Adapt Task 5 | 1 hr |
| Report integration | Part 12 or companion paper | 2 hr |
| **Total** | | **~6 hr** |

The binding constraint is analytical interpretation, not compute. The sweep itself runs in under 5 minutes across all assets.

---

## Decision Framework

After this work, we will have one of three outcomes:

**Outcome A: Family rankings replicate (Spearman ρ > 0.6) and TIM optimum is portable.**
→ The ETH findings are structural. Proceed to ensemble construction across assets and deployment planning.

**Outcome B: Family rankings partially replicate but TIM optimum is asset-specific.**
→ Signal families are real but TIM targeting needs per-asset calibration. Strategy architecture is sound but requires asset-level tuning. Proceed with caution and wider parameter bands.

**Outcome C: Family rankings do not replicate (Spearman ρ < 0.3).**
→ The ETH findings are path dependence. The research has negative value for deployment. This is a useful result — it prevents us from deploying a strategy that only works on one asset in-sample. We would pivot to investigating what structural properties (vol regime, autocorrelation profile, market microstructure) determine which signal families work on which assets, which is a harder but more fundamental question.

We commit to reporting whichever outcome we find. The methodology is pre-specified; the interpretation follows the data.

---

## What This Does Not Address

- **Live execution costs.** We continue to use 20 bps round-trip. Actual slippage on large positions in SOL or ATOM may be materially higher than on ETH/BTC. This is a deployment concern, not a signal validation concern.

- **Correlation structure.** This analysis treats each asset independently. The portfolio construction question — how to size across assets when they're all running trend signals — is deliberately deferred. We need to know which signals work before we can ask how to combine them.

- **Intraday frequency portability.** The 4H and 1H results from ETH are not retested on other assets in this phase (Tier 2 is daily only). If daily-frequency family replication succeeds, extending to 4H is straightforward but doubles compute.

- **Regime alignment in walk-forward design.** The walk-forward test (Phase 4) is restricted to BTC-USD by deliberate design, not oversight. The ETH walk-forward trains on 2017–2021 (predominantly bull market including two major cycles) and tests on 2022–2026 (bear market plus recovery). BTC has identical date coverage, producing the same regime alignment — training on similar market conditions, testing on similar stress periods. SOL has no 2017–2021 history, so no comparable walk-forward split is possible. Running a walk-forward on SOL with a different train/test regime would confound the comparison, which is why we restrict Phase 4 to BTC.

---

*Signed off by desk head. Proceeding with execution.*
