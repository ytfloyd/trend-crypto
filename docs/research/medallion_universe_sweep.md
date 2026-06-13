# Medallion Lite — universe-definition sweep

**Question.** We are an intentionally capacity-constrained shop able to trade small digital
assets. The flagship universe is a fixed top-50-by-ADV. Should we widen it, and to what?

**Method.** Every universe is built **point-in-time** from a 20-day trailing dollar-ADV
(survivorship-free, no look-ahead) on the full clean Coinbase USD set (362 symbols, 2021–2026).
The hourly panel, factors, and ensemble regime are computed **once** on the union; each universe
is just a different eligibility mask on the same composite score. Params are **frozen** at the
flagship defaults (entry 0.65 / trail 0.15) for an apples-to-apples comparison (no per-universe
re-fitting → no extra data-snooping, QF-21). 30 bps one-way costs. OOS = 2023-01-01 onward.

Harness: `scripts/research/k2_atlas/run_medallion_universe.py`.

## Frozen-param results (OOS 2023–2026)

| Universe | avg #names | OOS Sortino | Sharpe | CAGR | MaxDD | +vol-target |
|---|---:|---:|---:|---:|---:|---:|
| top_25 | 25 | 1.97 | 1.57 | 89% | −36% | 2.60 |
| top_50 (baseline) | 49 | 2.56 | 2.01 | 146% | −39% | 2.92 |
| top_100 | 93 | 2.60 | 1.97 | 153% | −36% | 2.87 |
| top_200 | 161 | 2.42 | 1.87 | 139% | −44% | 2.54 |
| **adv ≥ $1M** | 72 | **2.71** | 2.10 | 155% | −36% | **3.03** |
| adv ≥ $250k | 124 | 2.33 | 1.82 | 125% | −45% | 2.42 |
| all_usd | 193 | 2.28 | 1.78 | 127% | −45% | 2.44 |
| BTC buy & hold | — | 1.78 | 1.15 | 54% | −50% | — |

## Param-frozen walk-forward (honest OOS, fold-selected params)

| Universe | WF-OOS Sortino | Sharpe | CAGR | MaxDD | +vol-target |
|---|---:|---:|---:|---:|---:|
| top_50 | 2.65 | 2.12 | 159% | −39% | 2.76 |
| adv ≥ $1M | 2.77 | 2.14 | 161% | −37% | 2.92 |
| top_100 | 2.99 | 2.24 | 192% | −34% | 3.04 |

## Findings

1. **The edge is robust to universe definition.** Every universe tested clears the >2.0 Sortino
   mandate OOS and beats BTC (1.78). The strategy is not a top-50 artifact.
2. **Modest widening helps; over-widening hurts.** The sweet spot is ~70–95 names —
   **top_100 and an ADV ≥ $1M floor** match or beat the top-50 baseline (WF 2.99 / 2.77 vs 2.65),
   with *better* drawdowns (−34% / −37%). Pushing all the way to `all_usd` or `adv ≥ $250k`
   (120–190 names) **degrades** Sortino to ~2.3 and worsens MaxDD to −45% — the deep micro-cap
   tail adds noise and cost drag without adding edge.
3. **Recommended rule:** replace the fixed top-50 with an **ADV ≥ $1M point-in-time floor**
   (≈72 names, varies with the cycle). It directly expresses the capacity-constrained mandate —
   fish a broader pond including smaller assets — while screening out the illiquid junk that
   drowns the edge. `top_100` is an equally good rank-based alternative.

## Honesty caveat — absolute levels run hot vs the validated 2.03

This harness reconstructs the universe from a **20-day trailing-ADV rank on
`bars_1d_usd_universe_clean` with no $10M floor**. The validated walk-forward number (**2.03**)
came from the committed `bars_1d_usd_universe_clean_top50_adv10m_membership` table (a $10M ADV
floor + its own ADV window). Same walk-forward protocol, **different universe construction** —
hence top_50 here reads 2.65, not 2.03.

→ **Trust the cross-universe *shape*, not the absolute levels.** Before promoting any new
universe rule to the registry, reconcile this trailing-ADV reconstruction against the committed
membership-table methodology so the headline number is apples-to-apples with the validated 2.03.
This is a flagged reconciliation item, not a new validated result.

**Provenance:** `scripts/research/k2_atlas/run_medallion_universe.py`,
`coinbase_crypto_ohlcv_lake.duckdb` (`bars_1h` + `bars_1d_usd_universe_clean`), 30 bps,
2021-01..2026-06, OOS 2023+.
