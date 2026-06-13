# Strategy Card — Medallion Lite

| Field | Value |
|---|---|
| **Registry ID** | `2026-06-medallion-lite` |
| **Class** | Cross-sectional crypto factor (long-biased, regime-gated, event-driven) |
| **Route** | cross_sectional |
| **Universe** | Coinbase USD pairs, **point-in-time top-100 by 20-day trailing ADV** (survivorship-free; adv ≥ $1M is the liquidity-floor alternative). Was top-50; widened after the universe sweep. |
| **Bar frequency** | Hourly (flagship pipeline); daily for the registry `signal_fn` proxy |
| **Costs** | 30 bps one-way (flagship) |
| **Status** | S3 · validated with caveats (see below) |

## What it is
A Renaissance-inspired crypto **cross-sectional factor** strategy. Each bar it ranks the
liquid universe on a 5-factor composite, gates exposure with an ensemble market-regime
score, and runs an **event-driven** portfolio (enter / hold / exit) rather than continuous
rebalancing.

- **Factors (composite, cross-sectionally ranked → [0,1]):** momentum, volume surge,
  realized vol, proximity-to-high, risk-adjusted momentum.
- **Regime gate:** ensemble score on BTC (trend/vol) scales/halts exposure.
- **Portfolio:** enter when composite > 0.65, exit < 0.40; ≤ 25 positions; ≤ 10% per name;
  trailing stop 15%; max hold 14 days; rebalance every 24h.

**Mandate note (long convexity):** this is a *long-biased directional factor* book, not a
pure long-gamma vehicle. Its convexity comes indirectly — the **trailing stop + regime gate
truncate the left tail** (the −37% max DD vs BTC's −50%/−77% is the tell). It is closer to a
cross-sectional momentum sleeve than to the trend/options convexity core; treat accordingly.

## Rulebook rules applied (K2 TRADE ATLAS)
`QF-01` cross-sectional momentum · `QF-07` vol-targeting (overlay) · `QF-10`/`CV-17`
time-series convexity framing · `QF-21` data-snooping discipline (walk-forward) ·
`MR-09` crypto regime · factor intent per `QF-16/QF-09`.

## Validated performance (HONEST)
From the **pre-registered, auditable** run (`docs/research/medallion_validation_protocol.md`;
deterministic; manifest in `artifacts/medallion_audit/`). Survivorship-free point-in-time top-100,
within-universe rank, 30 bps, OOS = 2023+. **Headline = frozen params (nothing fit on the data);
walk-forward is a labeled upper bound.** All pre-registered gates (G1–G4) pass.

| Measure (OOS 2023–26) | Medallion Lite (top-100) | BTC buy & hold |
|---|---|---|
| **Sortino — frozen params (HEADLINE)** | **2.84** | 1.78 |
| Sortino — walk-forward (upper bound) | 2.95 | — |
| Sharpe / Calmar / CAGR / MaxDD | 2.15 / 4.73 / 173% / −37% | 1.15 / 1.09 / 54% / −50% |
| Per-fold OOS Sortino (frozen) | 4.72 / 2.72 / 1.67 (’23/’24/’25-26) | — |
| Cost sweep (0/10/20/30/50 bps) | 3.50 / 3.27 / 3.05 / 2.84 / 2.42 | — |
| PSR vs 0 (frozen OOS) | 1.00 | — |

**The honest arc** (how rigor moved the number):
1. As-shipped flagship: OOS Sortino **2.70** — universe was top-50 by *full-period* ADV =
   **look-ahead survivorship**.
2. Point-in-time top-50 + walk-forward corrects to **1.97–2.03**.
3. **Universe sweep** (capacity mandate → smaller names, within-universe ranking; membership spec
   replays the 1.97 baseline, reconciliation ✓): widening to **top-100** raised the
   walk-forward number to 2.95.
4. **Pre-registered audit** then separated the two selection layers: the defensible **frozen-param
   headline is 2.84**; 2.95 is the walk-forward upper bound (it includes a parameter-selection
   layer). The **100-factor TA-Lib zoo and all ensembling were tested and rejected** — design
   selection fails PBO (0.70–0.77); the simple 5-factor composite is what we run. See
   `medallion_validation_protocol.md`, `medallion_factor_count_experiment.md`,
   `medallion_bagging_experiment.md`.

## Cost sensitivity (GATING — currently ❌ FAILS)
The 2.84 headline assumes a **uniform 30 bps**. Under **liquidity-tiered costs** (by point-in-time
ADV; pre-registered Amendment A, harness reconciled to the engine) the edge does **not** survive:

| Cost assumption (OOS 2023–26) | Sortino |
|---|---|
| Flat 30 bps (headline) | 2.84 |
| Benign tiered (10/20/40/70) | 2.26 |
| **Realistic tiered (20/40/70/120)** | **1.42** (below BTC 1.78) |
| Punitive tiered (35/70/130/220) | −0.13 |
| + market impact, $5M AUM | 0.99 |
| + market impact, $25M AUM | 0.67 |

The alpha is concentrated in small, expensive names; **soft capacity is < $5M AUM**. As specified
(top-100) the strategy is **cost-fragile and not production-ready**. Gating next step: re-run the
universe sweep under realistic tiered costs to find a cost-robust liquid universe (top-25 /
ADV ≥ $50M). See `medallion_validation_protocol.md` (Amendment A),
`artifacts/medallion_audit/medallion_cost_sensitivity.json`.

## Risks & caveats
- **Cost realism at the margin (new, from widening):** 30 bps flat is reasonable for the top
  names, but rank 50–100 / sub-$1M assets can slip more. Drawdown *improves* with widening and
  turnover stays low (~0.01/bar), but a tiered-cost re-test is the gating follow-up before
  sizing up the top-100 universe.
- **Recent edge decay (observed at top-50):** per-fold OOS Sortino ran 3.49 → 1.97 → 1.11
  (2023 → 2024 → 2025–26). The top-100 aggregate (2.95) has not been decomposed per-fold —
  do so before treating the uplift as permanent.
- **Vol-target uplift is partly leverage** (CAGR rises with the overlay); risk-adjusted
  genuinely improves but it isn't free.
- **Registry `signal_fn` ≠ validated strategy:** `signals.cross_sectional.medallion_lite` is
  a simplified *daily* proxy; the metrics here are from the flagship hourly pipeline. These
  must be reconciled before `python -m research run` executes the validated strategy.
- **Universe is reconstructed, not committed:** the top-100 universe is recomputed from a 20-day
  trailing ADV (no $10M floor). The membership-table reconciliation (1.97 ≈ 2.03) validates the
  method, but production should build top-100 from a committed point-in-time membership table.

## Provenance & reproduce
- Flagship pipeline: `scripts/research/medallion_lite/` (factors, regime_ensemble, portfolio)
- Honest harnesses: `scripts/research/k2_atlas/run_medallion_pit.py` (survivorship test),
  `run_medallion_walkforward.py` (param-frozen WF + overlays),
  `run_medallion_universe.py` (universe sweep + reconciliation → adopted top-100),
  `medallion_tearsheet.py` (this card's tearsheet)
- Universe study: `docs/research/medallion_universe_sweep.md`
- **Tearsheet:** `scripts/research/medallion_lite/output/medallion_lite_pit_tearsheet.html`
  (regenerate: `PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/medallion_tearsheet.py`)
- Registry validation block: `registry/alphas/2026-06-medallion-lite.yaml`
- Data: `coinbase_crypto_ohlcv_lake.duckdb` (`bars_1h` / membership table)
