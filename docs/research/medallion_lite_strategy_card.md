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
Survivorship-free (point-in-time universe), 30 bps costs, daily Sortino. Adopted universe =
**top-100 by point-in-time trailing ADV** (param-frozen walk-forward, within-universe rank).

| Measure | Medallion Lite (top-100) | BTC buy & hold |
|---|---|---|
| **Walk-forward OOS Sortino (2023–26, param-frozen)** | **2.95** | 1.78 |
| + vol-targeting overlay (QF-07) | **3.04** (frozen) / 2.90 (WF) | — |
| Frozen-param OOS Sortino | 2.84 | 1.78 |
| OOS Sharpe / CAGR / MaxDD | 2.19 / 181% / −35% | 1.15 / 54% / −50% |

**The honest arc** (how the universe choice and the rigor got us here):
1. As-shipped flagship: OOS Sortino **2.70** — but the universe was top-50 by *full-period*
   ADV = **look-ahead survivorship**.
2. Point-in-time top-50 membership + param-frozen walk-forward corrects it to **1.97–2.03**
   (the previous validated number).
3. **Universe sweep** (capacity-constrained mandate → trade smaller names): re-built every
   universe point-in-time with **within-universe cross-sectional ranking**. The committed
   membership table reproduces the **1.97** baseline (reconciliation ✓); widening to
   **top-100 (~93 names)** lifts honest WF-OOS Sortino to **2.95** with a *better* drawdown
   (−35%). + vol-targeting → **3.04**. `adv ≥ $1M` (~72 names) is the liquidity-floor
   alternative (WF 2.46, +vol-target 3.00, DD −32%). See
   `docs/research/medallion_universe_sweep.md`.

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
