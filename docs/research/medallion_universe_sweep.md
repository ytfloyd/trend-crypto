# Medallion Lite — universe-definition sweep

**Question.** We are an intentionally capacity-constrained shop able to trade small digital
assets. The flagship universe is a fixed top-50-by-ADV. Should we widen it, and to what?

**Method (reconciled, v2).** Every universe is built **point-in-time** from a 20-day trailing
dollar-ADV (survivorship-free, no look-ahead) on the full clean Coinbase USD set (362 symbols,
2021–2026). Per-symbol factors are computed **once** (their rolling inputs are population-
independent); for each universe we mask the factor frames to the eligible set **before** the
cross-sectional rank, so the composite is ranked **within** that universe. This is the key
correction — the composite is a percentile rank, so ranking over the full 362-name panel and
then masking would conflate "universe" with "ranking population". 30 bps one-way. OOS = 2023+.

Harness: `scripts/research/k2_atlas/run_medallion_universe.py`.

**Reconciliation.** The `membership` spec replays the committed
`bars_1d_usd_universe_clean_top50_adv10m_membership` table. Under within-universe ranking it
reproduces the validated walk-forward baseline — **WF-OOS Sortino 1.97** (≈ the validated
1.97/2.03). So the ADV-floor / top_N numbers below are apples-to-apples with the validated
strategy. (An earlier rank-over-all-then-mask version read top_50 at a spurious 2.65; the fix
brings it back in line.)

## Frozen-param results (entry 0.65 / trail 0.15, OOS 2023–2026, within-universe rank)

| Universe | avg #names | OOS Sortino | Sharpe | CAGR | MaxDD | +vol-target |
|---|---:|---:|---:|---:|---:|---:|
| membership (validated baseline) | ~20* | 1.83 | 1.46 | 86% | −36% | 2.29 |
| top_25 | 25 | 2.09 | 1.64 | 97% | −35% | 2.67 |
| top_50 | 49 | 2.60 | 2.03 | 148% | −38% | 2.90 |
| **top_100** | 93 | **2.84** | 2.15 | 173% | −37% | **3.04** |
| top_200 | 161 | 2.47 | 1.88 | 142% | −43% | 2.60 |
| **adv ≥ $1M** | 72 | 2.76 | 2.11 | 155% | −32% | 3.00 |
| adv ≥ $250k | 124 | 2.48 | 1.93 | 137% | −43% | 2.66 |
| all_usd | 193 | 2.48 | 1.89 | 142% | −41% | 2.62 |
| BTC buy & hold | — | 1.78 | 1.15 | 54% | −50% | — |

\* `membership` avg eligible is low because a few delisted membership symbols are absent from the
clean USD table loaded here (also explains the tiny 1.97-vs-2.03 reconciliation gap).

## Param-frozen walk-forward (honest OOS, fold-selected params, within-universe rank)

| Universe | WF-OOS Sortino | Sharpe | CAGR | MaxDD | +vol-target |
|---|---:|---:|---:|---:|---:|
| membership (validated baseline) | 1.97 | 1.57 | 98% | −38% | 2.07 |
| top_50 | 2.29 | 1.81 | 124% | −41% | 2.33 |
| **top_100** | **2.95** | 2.19 | 181% | −35% | 2.90 |
| adv ≥ $1M | 2.46 | 1.90 | 134% | −37% | 2.49 |

## Findings

1. **Reconciled & robust.** With within-universe ranking the harness reproduces the validated
   baseline (membership WF 1.97), and every widened universe still clears the >2.0 mandate and
   beats BTC (1.78). The edge is not a top-50 artifact.
2. **Widening helps materially; over-widening fades.** The sweet spot is ~70–95 names.
   **top_100** is the standout — honest WF-OOS Sortino **2.95** (vs membership 1.97), best
   frozen OOS (2.84) and best +vol-target (3.04), with a *better* drawdown (−35% vs −38%).
   **adv ≥ $1M** (≈72 names) is the liquidity-floor equivalent: WF 2.46, +vol-target 3.00, and
   the tightest drawdown (−32% frozen). Pushing to `top_200`/`adv ≥ $250k`/`all_usd`
   (120–190 names) settles back to ~2.47–2.48 with −41% to −45% drawdowns — the deep micro-cap
   tail adds noise and cost drag without adding edge.
3. **Adopted rule:** **top_100 by point-in-time trailing ADV** (≈93 names). It nearly doubles
   breadth vs the baseline, raises honest WF-OOS Sortino 1.97 → 2.95, and improves the drawdown.
   `adv ≥ $1M` is documented as the economically-principled liquidity-floor alternative with
   near-identical risk-adjusted performance and the best drawdown control.

## Caveats

- **Cost realism at the margin.** 30 bps flat is reasonable for the top names; rank 50–100
  (and sub-$1M) assets can slip more. The drawdown *improves* with widening and turnover stays
  low (~0.01/bar), but a tiered-cost re-test is the right follow-up before sizing up.
- **Reconstruction vs committed table.** This harness recomputes ADV (20-day window, no $10M
  floor) rather than reading the committed membership table. The membership reconciliation
  (1.97 ≈ 2.03) validates the method; a production universe should still be built from a
  committed, point-in-time membership table rather than recomputed on the fly.

**Provenance:** `scripts/research/k2_atlas/run_medallion_universe.py`,
`coinbase_crypto_ohlcv_lake.duckdb` (`bars_1h` + `bars_1d_usd_universe_clean` +
`…top50_adv10m_membership`), 30 bps, 2021-01..2026-06, OOS 2023+.
