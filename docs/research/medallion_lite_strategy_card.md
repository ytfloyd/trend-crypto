# Strategy Card — Medallion Lite

| Field | Value |
|---|---|
| **Registry ID** | `2026-06-medallion-lite` |
| **Class** | Cross-sectional crypto factor (long-biased, regime-gated, event-driven) |
| **Route** | cross_sectional |
| **Universe** | Coinbase USD pairs, point-in-time top-50 by ADV (`bars_1d_usd_universe_clean_top50_adv10m`) |
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
Survivorship-free (point-in-time universe), 30 bps costs, daily Sortino.

| Measure | Medallion Lite | BTC buy & hold |
|---|---|---|
| **Walk-forward OOS Sortino (2023–26, param-frozen)** | **2.03** | 1.78 |
| + vol-targeting overlay (QF-07) | **2.33** | — |
| Full-sample Sortino (2021–26, fixed params) | 1.69 | 0.78 |
| OOS Sharpe / CAGR / MaxDD | 1.60 / 101% / −39% | 1.15 / 54% / −50% |

**The honest arc** (why these numbers and not the headline 2.70):
1. As-shipped flagship: OOS Sortino **2.70** — but the universe was top-50 by *full-period*
   ADV = **look-ahead survivorship**.
2. Point-in-time top-50 membership corrects it to **1.97**.
3. Param-frozen **walk-forward** (select-on-train / freeze / score-on-test) = **2.03** — the
   defensible number. + vol-targeting → **2.33**.

## Risks & caveats
- **Recent edge is decaying:** per-fold OOS Sortino runs **3.49 (2023) → 1.97 (2024) → 1.11
  (2025–26)**. The 2.03 aggregate leans on 2023.
- **Vol-target uplift is partly leverage** (CAGR 101%→146%, MaxDD −39%→−42%); risk-adjusted
  genuinely improves but it isn't free.
- **Registry `signal_fn` ≠ validated strategy:** `signals.cross_sectional.medallion_lite` is
  a simplified *daily* proxy; the metrics here are from the flagship hourly pipeline. These
  must be reconciled before `python -m research run` executes the validated strategy.
- **Unverified:** membership-table ADV ranking assumed point-in-time/trailing; no fill/slippage
  modeling beyond 30 bps flat; small (9-config) param grid in the walk-forward.

## Provenance & reproduce
- Flagship pipeline: `scripts/research/medallion_lite/` (factors, regime_ensemble, portfolio)
- Honest harnesses: `scripts/research/k2_atlas/run_medallion_pit.py` (survivorship test),
  `run_medallion_walkforward.py` (param-frozen WF + overlays),
  `medallion_tearsheet.py` (this card's tearsheet)
- **Tearsheet:** `scripts/research/medallion_lite/output/medallion_lite_pit_tearsheet.html`
  (regenerate: `PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/medallion_tearsheet.py`)
- Registry validation block: `registry/alphas/2026-06-medallion-lite.yaml`
- Data: `coinbase_crypto_ohlcv_lake.duckdb` (`bars_1h` / membership table)
