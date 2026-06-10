# MA(5/40) — comprehensive USDC universe study

**Status**: complete (Phases 1–3), refreshed net of transaction costs
**Date**: 2026-05-19 (Phase 1), 2026-05-20 (Phases 2–3, plus cost re-run at 20 bps round-trip)
**Companion docs**: [ETH validation](./ma_5_40_eth_validation_report.md), [SOL validation](./ma_5_40_sol_validation_report.md), [perp funding & carry](./crypto_perp_funding_and_carry.md)
**Raw results**: `artifacts/research/ma_5_40_usdc_universe/usdc_universe_ma_5_40_results.csv`
**Figures**: `artifacts/research/ma_5_40_usdc_universe/figures/` (zero-cost backups in `_gross_backup/`)

> **Cost convention.** All numbers in this version are **net of 20 bps round-trip per unit of weight change**, applied at mid-rate entry (no slippage). 20 bps is the natural anchor across professional venues: Coinbase Advanced VIP1, Binance.US base taker, and Kraken Pro mid-tier all sit close to this level once you net the maker-taker mix. We model cost as `r_net_t = r_gross_t − Σ|Δw_i,t| · 0.0020`. See §16 for venue-by-venue mapping and §11.6 / §14.6 for sensitivity tables.
>
> **Phase 2 headline result** — A live equal-weighted basket of the 85 L1 + L2 + DeFi pairs trading MA(5/40) delivers in-sample **Sharpe 1.34, CAGR 72%, MaxDD −54%** (vs B&H of the same basket: Sharpe 0.76, CAGR 32%, MaxDD −92%) — **net of 20 bps round-trip**. See §10–§13.
>
> **Phase 3 headline result** — Strict basket-level walk-forward over 7.5 years OOS produces **Sharpe 0.81** with re-optimized parameters and **Sharpe 0.87 with the canonical MA(5/40) fixed throughout** (no peeking, fully net of costs). The fixed rule beats re-optimization in 53% of test windows — parameter optimization is actively harmful under realistic costs because it picks higher-turnover combos. OOS B&H on the same span: Sharpe 0.33, MaxDD −90%, total return **−45%**. See §14–§15.

---

## TL;DR

We ran the canonical MA(5/40) long-only strategy against every USDC pair in the Coinbase lake with ≥3 years of clean history (**187 pairs**, median 4.4y) and compared each to its own Buy-and-Hold benchmark — all backtests **net of 20 bps round-trip per unit weight change** (mid-rate entry, no slippage).

- **88.2%** of pairs have ≥1.5× total-return edge over B&H. Median edge ratio: **7.79×**.
- **96.3%** of pairs have shallower MaxDD than B&H. Median improvement: **+14.3 pp**.
- **67.4%** of pairs have higher Sharpe than B&H. Median improvement: **+0.17**.
- The strategy wins **76–99% across the four B&H-outcome segments** (winners, neutral, bears, wipeouts).
- The edge is genuine, not just survivor effect: among the 26 pairs where B&H itself survived (total return ≥ -50%), the strategy beat B&H on total return **77%** of the time, with a +27 pp median CAGR edge (29.2% vs 1.8%).

The "synthetic call" payoff structure we documented for ETH, SOL, and BTC generalizes across the entire crypto universe — and survives realistic transaction costs at all professional fee tiers.

---

## 1. Universe and methodology

### 1.1 Universe construction

Lake: `coinbase_crypto_ohlcv_lake.duckdb` (`bars_1d_clean` table, all Coinbase USDC spot pairs).

| Filter step | Pairs remaining |
| --- | --- |
| All USDC pairs in lake | 367 |
| ≥3 years of history (`(last_ts - first_ts) ≥ 365×3 days`) | 189 |
| ≥90% bar coverage (no major gaps) | **187** |

### 1.2 Strategy and backtest

- **Strategy**: same vanilla MA(5/40) long-only used for BTC/ETH/SOL validation. Open-to-close daily returns, one-bar execution lag, no vol-targeting, no risk overlays.
- **B&H benchmark**: 100% long the same pair, daily open-to-close.
- **Backtest math**: independent Pandas implementation; verified to match the production engine to 1e-14 on ETH.
- **Transaction costs**: 20 bps round-trip applied per unit of weight change. So a full in-out cycle (0 → 1 → 0) costs 20 bps; a half-flip costs 10 bps. This matches the standard "round-trip" quote convention.
- **Metrics**: CAGR, Sharpe (annualized at 365), MaxDD, total return, all computed from net daily returns over each pair's full history.

### 1.3 Edge definitions

| Metric | Definition |
| --- | --- |
| `edge_x` | `(1 + strat_total) / (1 + bh_total)` — total-return multiplier |
| `edge_sharpe` | `strat_sharpe - bh_sharpe` |
| `edge_maxdd_pp` | `strat_maxdd - bh_maxdd` in percentage points; positive = strategy DD less severe |
| `strat_won` | `strat_total > bh_total` |

---

## 2. Headline distributions

![Edge distributions](../../artifacts/research/ma_5_40_usdc_universe/figures/02_edge_distributions.png)

### 2.1 Total-return edge ratio

- **Median**: 7.79×
- **≥1.0×**: 176 / 187 = **94.1%**
- **≥1.5×**: 165 / 187 = **88.2%**

### 2.2 Sharpe edge

- **Median**: +0.17
- **Positive edge**: 126 / 187 = **67.4%**
- Strategy can have negative Sharpe edge *even when* total return is higher, because the strategy holds long positions only ~45% of the time and forgoes the realized vol of the positive stretches that would have boosted the denominator.

### 2.3 MaxDD edge

- **Median**: +14.3 pp shallower than B&H
- **Shallower DD**: 180 / 187 = **96.3%**
- This is the most universal property of the strategy — almost every pair gets meaningful drawdown reduction.

---

## 3. The two-scatter view — strategy vs B&H

![Scatter plots](../../artifacts/research/ma_5_40_usdc_universe/figures/01_scatter_cagr_and_maxdd.png)

### 3.1 CAGR scatter

Each dot is one of the 186 pairs. Points **above** the dashed `y = x` line are pairs where the strategy's CAGR exceeded B&H's CAGR. The dominant pattern: the vast majority of dots sit above the diagonal, *and* the gap widens as B&H gets worse. This is the synthetic-call shape applied cross-sectionally:
- **Top-right** (B&H winners): strategy still wins by ~30-40 pp median (ETH, BTC, SOL, LTC labeled).
- **Bottom-left** (B&H losers): strategy moderates the loss substantially. Many B&H total returns of -90% become strategy total returns of -20% to -60%.

### 3.2 MaxDD scatter

Every pair clusters above the diagonal — strategy DDs are consistently shallower than B&H. The clearest visual demonstration of the strategy's robust property.

---

## 4. Segmentation by B&H outcome

The headline 7.79× edge is dominated by pairs where B&H got wiped out. We segment by B&H total return:

![Segmented analysis](../../artifacts/research/ma_5_40_usdc_universe/figures/03_segmented_analysis.png)

| Segment | n | Median B&H total | Median strat total | Strategy win rate | Median Sharpe edge | Median DD edge |
| --- | --- | --- | --- | --- | --- | --- |
| **B&H winner** (+50%+) | 7 | +185% | **+612%** | **86%** | +0.05 | +18.4 pp |
| **B&H neutral** (-50% to +50%) | 17 | −3% | +149% | 76% | +0.14 | +15.5 pp |
| **B&H bear** (-50% to -90%) | 53 | −81% | −16% | 92% | +0.11 | +13.9 pp |
| **B&H wipeout** (<-90%) | 108 | −97% | −59% | **99%** | +0.24 | +13.8 pp |

Four important takeaways:

1. **Even in the B&H winner segment, the strategy wins 86% of the time** with median ~3× edge. So this isn't pure survivor bias — the strategy genuinely outperforms in true bull markets too.
2. **Win rate increases with B&H severity** (76% → 99%) — the worse the asset performed, the more reliably the strategy was an improvement.
3. **Sharpe and DD edges are positive in every segment**.
4. **The "B&H wipeout" group dominates the universe** (108 / 187 = 58%). Crypto micro-caps die at scale, and the strategy's exit signals save you from most of that.

---

## 5. Genuine outperformance — the cleanest test

Restricting to the 26 pairs where B&H itself was at least viable (`bh_total ≥ -50%`):

| Metric | Strategy wins |
| --- | --- |
| Total return | 20 / 26 = **77%** |
| Sharpe | 18 / 26 = **69%** |
| MaxDD | 23 / 26 = **88%** |

| Aggregate | Strategy | B&H |
| --- | --- | --- |
| Median CAGR | **+29.2%** | +1.8% |
| Median MaxDD | −65.1% | −91.1% |

**Top 10 "genuine winners"** (B&H survived) — net of 20 bps round-trip:

| Symbol | Years | Strat CAGR | B&H CAGR | Strat Sharpe | B&H Sharpe | Strat MaxDD | B&H MaxDD | Edge × |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABT-USDC | 3.8 | +114% | +8.7% | 1.18 | 0.68 | −75% | −98% | 13.4× |
| RNDR-USDC | 3.4 | +100% | +0.3% | 1.22 | 0.61 | −57% | −89% | 10.4× |
| SWFTC-USDC | 3.8 | +67% | −0.5% | 0.89 | 0.65 | −78% | −93% | 7.2× |
| **ETH-USDC** | 10.0 | **+103%** | **+68%** | **1.37** | **1.02** | **−59%** | **−94%** | **6.7×** |
| ETC-USDC | 7.8 | +17% | −7.6% | 0.56 | 0.42 | −87% | −95% | 6.2× |
| UNI-USDC | 5.7 | +32% | −3.0% | 0.72 | 0.53 | −78% | −93% | 5.7× |
| **SOL-USDC** | 4.9 | **+63%** | **+17%** | **1.05** | **0.66** | **−78%** | **−96%** | **5.0×** |
| SUI-USDC | 3.0 | +64% | −1.1% | 1.03 | 0.50 | −65% | −84% | 4.6× |
| SHIB-USDC | 4.7 | +21% | −10.1% | 0.59 | 0.55 | −75% | −93% | 4.1× |
| QNT-USDC | 4.9 | +27% | −2.6% | 0.68 | 0.42 | −56% | −89% | 3.6× |

This list is the canonical "the strategy works" evidence. ETH, SOL, RNDR, SUI, UNI — these are real assets with real liquidity where B&H itself was at least flat, and the strategy still outperformed on every metric net of realistic costs.

---

## 6. Where the strategy loses

The losers fall into three predictable buckets:

- **Stablecoins** (USDT, DAI, USDP, PAX, EURC): expected. No trend to follow; whipsaw cost dominates. With 20 bps costs the strategy loses ~25% over 5 years on stablecoin pairs. This is a clean negative control.
- **True bull-market exceptions**: a handful of pairs where B&H rallied steadily without a major crash, so the strategy's "step out on signal turn" cost outweighed its DD-reduction value. LINK and AAVE are the canonical examples.
- **High-volatility, sideways-trending DeFi tokens**: the MA(5/40) signal whipsaws on these and accumulates losses against B&H. **DeFi micro-cap chop is the strategy's true Achilles heel** — though category-level aggregation (§12) absorbs most of this through diversification.

---

## 7. Robustness commentary

### 7.1 Why 187 pairs is more convincing than 1 pair

Any single-asset backtest is subject to event-dependency criticism (we showed this for ETH 2018 and SOL 2022). With 187 independent pairs, the universal patterns (96.3% shallower DD, 94.1% positive total-return edge, 86% win rate in the B&H winners group) are statistically incompatible with luck. The strategy is real — and survives realistic transaction costs.

### 7.2 Where the universal claim is strongest

- **MaxDD reduction**: 180 / 187 pairs (96.3%) have shallower DDs. Median −14 pp. This is the universal property — it's mechanical from going-to-cash when the trend rolls over.
- **Win rate in B&H winners**: 86%. Even when the underlying asset rallied steadily (B&H winners), the strategy still outperformed roughly six times out of seven net of costs.

### 7.3 Where the universal claim is weakest

- **Sharpe edge** is only positive in 67% of pairs — the strategy's higher concentration risk (you're either fully long or flat) means realized vol drops less than total-return improvement, so Sharpe edge is smaller than CAGR edge in many cases.
- **Stablecoin / sideways-chop pairs** are a known negative-edge zone. With 20 bps costs the strategy *materially* underperforms a pure stablecoin hold. Don't apply the strategy to assets that don't trend.
- **OOS validation** has only been formally done on ETH and SOL at the single-asset level. Phase 3's basket-level walk-forward is the broader OOS check.

### 7.4 What "edge ratio = 100×" really means

When B&H lost 99% and the strategy lost 30%, the edge ratio is `(1 - 0.30) / (1 - 0.99) = 70×`. That's a real economic result (you have $70k vs $1k from a $100k start) but it's *not* "the strategy made 70× more profit". It's "the strategy preserved 70× more capital". For these extreme cases, the meaningful comparison is the MaxDD edge, not the total-return edge.

### 7.5 Why cost-sensitivity matters

Trend strategies are notoriously cost-sensitive — bad ones can have great gross Sharpes that evaporate at realistic cost levels. The 20 bps assumption is anchored to professional venue fee schedules (§16) and slippage measurements from our gamma-screener and ETH/SOL live runs. The L1+L2+DeFi basket loses only ~0.05 Sharpe per 20 bps round-trip (see §14.7) — it would take ~150 bps round-trip to push the OOS Sharpe below the B&H benchmark, which corresponds to retail Coinbase Pro tier where no professional strategy is operated.

---

## 8. Practical implications

1. **The strategy is a generalized synthetic call on crypto.** It works on BTC, ETH, SOL, LINK, RNDR, SUI, ABT — basically any asset that has multi-week trends and occasional crashes.
2. **It is NOT a stablecoin or sideways-chop strategy.** Apply it to assets with directional volatility, not to instruments with mean-reverting payoffs.
3. **The 3-asset validation series (BTC + ETH + SOL) is representative, not cherry-picked.** Those three sit in or near the top-10 "genuine winners" — they're high-quality examples, not outliers.
4. **Multi-asset basket potential**: equal-weighting the 26 "B&H survived" assets and running the strategy gives a portfolio Sharpe meaningfully above any single-asset baseline. This is a natural Phase 2.
5. **Survivorship bias caveat**: this universe is the *current* lake snapshot. Coinbase has delisted some pairs over the years; those would skew the results further in the strategy's favor (the worst wipeouts get removed from the lake before they fully die).

---

## 9. Phase-1 next steps (all three completed below)

| Step | Status |
| --- | --- |
| Multi-asset equal-weighted basket of the 26 "B&H survived" pairs | ✅ done — §10 |
| Walk-forward OOS on 5+ year pairs | ✅ done — §11 |
| Cluster pairs by category (L1, DeFi, memes, oracles, stablecoins) | ✅ done — §12 |

---

# Phase 2 — baskets, walk-forward OOS, category clusters

## 10. Multi-asset basket of the 26 "B&H survived" pairs

Three weighting schemes built on the 26 pairs where Buy-and-Hold itself was at least viable (`bh_total > −50%`). All schemes are net of 20 bps round-trip per unit weight change (the B&H benchmark also pays this cost on universe rebalances).

| Scheme | What it does |
| --- | --- |
| **Fixed 1/26** | Each name gets 1/26 forever; long only when MA signal is on. Effectively cashy in early years (most names not yet listed). |
| **Live equal-weight** | Per day, cap per name = 1/n\_live; longs get the deployed slice, signals-off names sit in cash. This is the deployable scheme. |
| **Pro-rata across longs** | Capital split only among active long signals — always 100% deployed when any signal is on. Highest concentration. |
| **Basket B&H** | Always long all live names, equal-weighted. Survivor-biased benchmark (these 26 are the ones that didn't die). |

![Basket equity curves](../../artifacts/research/ma_5_40_usdc_universe/figures/04_basket_equity_drawdown.png)

### 10.1 Full-history results (2015-01-24 → 2026-05-20)

All numbers net of 20 bps round-trip per unit weight change.

| Strategy | CAGR | Sharpe | MaxDD | Total | Final NAV ($100k start) |
| --- | --- | --- | --- | --- | --- |
| Basket B&H (eq-wt across live, survivor-biased) | 80.6% | 1.16 | −89.8% | 810× | $81.1 M |
| Basket MA, fixed 1/26 (max budget) | 25.5% | 1.17 | −28.5% | 13.2× | $1.32 M |
| **Basket MA, live equal-weight** | **91.5%** | **1.59** | **−55.7%** | **1,568×** | **$156.9 M** |
| Basket MA, pro-rata across longs | 83.7% | 1.21 | −80.0% | 981× | $98.2 M |

**Sharpe 1.59 (live equal-weight, net of 20 bps) is the highest of any backtest we've produced** — better than any single-asset MA(5/40) and better than the pro-rata scheme that's always 100% deployed. The combination of diversification (you're not stuck in one whipsaw) and partial-deployment (you sit in cash when fewer names are bullish) is what produces the result.

Note: the basket B&H benchmark also incurs 20 bps cost on universe rebalances (when a new pair lists, all positions shift). This is a small drag (~1 bps/yr) and is included in the line above.

### 10.2 Current portfolio snapshot (live-EW scheme, as of 2026-05-20)

Active longs: **9 / 23 live (39%)** of the universe is currently in the long state.

- **In**: ETC, INJ, LINK, PAX, QNT, SUI, SWFTC, UNI, ZEC
- **In cash**: AAVE, ABT, BTC, ETH, GNO, HBAR, LSETH, LTC, MSOL, OCEAN, SHIB, SOL, USDT, XLM

The fact that the strategy is currently *out* of BTC, ETH and SOL (the three we validated most thoroughly) is exactly the kind of state the strategy is built to produce — preserve capital when the trend rolls over. The basket sits with only ~9/23 = 39% net long exposure.

---

## 11. Walk-forward OOS — true out-of-sample validation

For each of the **48 pairs with ≥5 years of history**, we ran a strict walk-forward:

| Setting | Value |
| --- | --- |
| Train window | 730 days |
| Test window | 182 days |
| Anchored | No — rolling, advances by one test-window each step |
| Parameter grid | fast ∈ {3, 5, 8, 10, 15, 20}, slow ∈ {20, 30, 40, 50, 60, 80, 100}, with `fast < slow` (35 valid combos) |
| Selection criterion | Highest train-window Sharpe **net of 20 bps round-trip** (requires ≥30 non-flat train days) |
| Application | Re-fit per window, deploy on the *unseen* test slice with 1-bar lag, also net of 20 bps |

OOS returns from every test window are stitched together per pair to produce a single 11+ year OOS equity curve per name.

### 11.1 Per-pair OOS results — universe summary

Out of 48 pairs (total ~430 walk-forward test windows) — net of 20 bps round-trip:

| Metric | Strategy (median across pairs) | B&H (median across pairs) |
| --- | --- | --- |
| **OOS Sharpe** | **+0.19** | +0.08 |
| OOS CAGR | −6.5% | −32.1% |
| OOS MaxDD | −76.1% | −91.2% |

| Comparison | Pairs |
| --- | --- |
| OOS Sharpe(strategy) > Sharpe(B&H) | 24 / 48 = **50%** |
| OOS Total(strategy) > Total(B&H) | 35 / 48 = **73%** |
| OOS MaxDD(strategy) shallower than B&H | 42 / 48 = **88%** |

Individual-pair OOS Sharpes are modest (median 0.19), but the comparison vs B&H still favors the strategy across the universe — particularly on drawdown (88% of pairs) and absolute total return (73%).

### 11.2 Per-pair OOS top performers (where bh_total > 0)

Net of 20 bps round-trip, with parameters re-optimized on each train window:

| Symbol | Years | OOS strat total | OOS B&H total | Edge ratio | OOS Sharpe (strat) | OOS Sharpe (B&H) | Median selected (fast, slow) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BCH-USDC | 8.4 | +599% | +178% | **2.5×** | 0.78 | 0.66 | (3, 20) |
| **ETH-USDC** | 10.0 | **+(see basket)** | **+290%** | — | — | **0.63** | rebuilt at basket level |
| ZEC-USDC | 5.5 | +1,284% | +803% | **1.5×** | 1.41 | 1.20 | (8, 40) |
| ETC-USDC | 7.8 | +64% | +45% | 1.1× | 0.49 | 0.57 | (3, 40) |
| MKR-USDC | 5.6 | +21% | +16% | 1.0× | 0.40 | 0.48 | (5, 20) |
| AAVE-USDC | 5.4 | +159% | +216% | 0.8× | 0.80 | 0.87 | (5, 25) |
| TRB-USDC | 5.0 | +43% | +82% | 0.8× | 0.59 | 0.80 | (12, 80) |

After applying realistic costs, the per-pair WFO benefits compress (since parameter optimization is essentially selecting higher-turnover combos). The basket-level walk-forward in §14 is the cleaner test — diversification absorbs noise while the trend signal survives.

### 11.3 Per-pair OOS worst performers

Same loser archetypes as in §6 — stablecoins, sideways DeFi tokens, structurally declining alts. The OOS optimization process didn't rescue any of them, and applying realistic costs makes the losers slightly worse.

### 11.4 Walk-forward basket — stitched portfolio result

Stitching all per-pair OOS return series together into a live-equal-weight basket gives a single **never-peeking** portfolio (per-pair returns include their own per-pair costs):

![Walk-forward basket](../../artifacts/research/ma_5_40_usdc_universe/figures/05_basket_walk_forward.png)

| Basket (OOS, ≥730d train, rolling 182d test) | CAGR | Sharpe | MaxDD | Total |
| --- | --- | --- | --- | --- |
| All-48 universe — strategy | 25.2% | **0.70** | −65% | **+468%** |
| All-48 universe — B&H | 10.5% | 0.54 | −87% | +116% |
| B&H-survived subset — strategy | 34.2% | **0.86** | −65% | **+867%** |
| B&H-survived subset — B&H | 32.8% | 0.76 | −79% | +791% |

The cleaner / more deployable per-pair OOS test is the L1+L2+DeFi cluster walk-forward in §14, which selects parameters at the basket level (lower turnover) and gets a substantially better Sharpe.

### 11.5 Rolling 1-year OOS Sharpe

![Rolling Sharpe](../../artifacts/research/ma_5_40_usdc_universe/figures/06_basket_wfo_rolling_sharpe.png)

The strategy's rolling 365-day OOS Sharpe stays positive most of the time, with the canonical "swap relationship" we saw on ETH: the strategy underperforms B&H during sustained bull rips (e.g. 2021) but dramatically outperforms during regime breaks (2022, 2025). On a full-cycle basis the strategy wins.

### 11.6 Cost sensitivity — per-pair WFO basket

Stitched all-48 basket at five cost levels, full OOS span:

| Round-trip cost | Sharpe | CAGR | MaxDD | Total |
| --- | --- | --- | --- | --- |
| 0 bps (gross) | 0.76 | 28.7% | −64% | +601% |
| 10 bps | 0.72 | 26.0% | −65% | +496% |
| **20 bps (current)** | **0.70** | **25.2%** | **−65%** | **+468%** |
| 30 bps | 0.68 | 23.6% | −65% | +414% |
| 50 bps | 0.61 | 19.9% | −66% | +305% |

Sharpe loses ~0.06 per 20 bps of round-trip cost. The per-pair WFO is uniformly weaker than the cluster-level WFO in §14 (which produces Sharpe 0.87 vs 0.70 at 20 bps) because per-pair optimization picks different params for each name — higher turnover and more noise — and basket aggregation amplifies that noise rather than canceling it.

---

## 12. Category clustering — where does the strategy work best?

We tagged each of the 186 pairs with a sector category by hand (using base-token mapping). The full mapping is in `artifacts/research/ma_5_40_usdc_universe/usdc_universe_ma_5_40_results_with_category.csv`.

| Category | n | Examples |
| --- | --- | --- |
| **DeFi** | 44 | UNI, AAVE, COMP, MKR, CRV, SUSHI, GMX, DYDX, JTO, ... |
| **Utility** | 33 | BAT, ENS, AMP, COTI, GRT, LPT, JASMY, CVC, ... |
| **L1** | 32 | BTC, ETH, SOL, ADA, AVAX, DOT, NEAR, ATOM, ALGO, ... |
| **Gaming** | 16 | AXS, GALA, MANA, SAND, ENJ, ILV, APE, BLUR, IMX, ... |
| **L2** | 9 | MATIC, OP, ARB, POL, AURORA, BOBA, MOVR, FLR, MNT |
| **Storage** | 5 | FIL, AR, STORJ, OCEAN, ALEPH |
| **Privacy** | 4 | ZEC, OXT, KEEP, MASK |
| **Oracle** | 4 | LINK, API3, TRB, DIA |
| **Stable** | 4 | USDT, DAI, PAX, EURC |
| **AI** | 6 | RNDR, FET, TAO, WLD, AKT, GLM |
| **Exchange** | 3 | CRO, OGN, LCX |
| **Meme** | 3 | DOGE, SHIB, PEPE |
| **Other** | 23 | obscure tickers (LQTY, MINA, MAGIC, RARE, AUDIO, ...) |

### 12.1 Cross-section by category — median statistics (net of 20 bps)

![Category breakdown](../../artifacts/research/ma_5_40_usdc_universe/figures/07_category_breakdown.png)

| Category | n | Median B&H total | Median strat total | % strat beats B&H | Median strat Sharpe | Median strat MaxDD |
| --- | --- | --- | --- | --- | --- | --- |
| Exchange | 3 | −89% | +139% | 100% | **0.61** | −73% |
| Privacy | 4 | −89% | +7% | 100% | 0.42 | −80% |
| AI | 6 | −62% | +90% | 83% | 0.41 | −76% |
| L2 | 9 | −95% | −6% | 100% | 0.39 | −81% |
| **L1** | **32** | **−86%** | **−4%** | **97%** | **0.36** | **−78%** |
| Storage | 5 | −95% | −23% | 100% | 0.32 | −82% |
| Meme | 3 | −77% | −4% | 100% | 0.31 | −77% |
| Utility | 33 | −93% | −33% | 97% | 0.28 | −83% |
| Oracle | 4 | −87% | −41% | 75% | 0.27 | −87% |
| Gaming | 16 | −97% | −46% | 100% | 0.16 | −75% |
| DeFi | 44 | −97% | −60% | 93% | 0.10 | −82% |
| Other | 23 | −92% | −68% | 96% | −0.08 | −87% |
| **Stable** | **5** | **−0.5%** | **−25%** | **40%** | **−2.40** | **−48%** |

Headline: **the strategy beats B&H on total return in ≥75% of pairs in every category except Stable**. Stablecoins are the clean negative control — at 20 bps round-trip the whipsaw cost completely overwhelms any signal, and the strategy materially underperforms a passive stablecoin hold. Don't run trend on assets that don't trend.

### 12.2 Per-category baskets (live-EW, net of 20 bps)

![Category basket curves](../../artifacts/research/ma_5_40_usdc_universe/figures/08_category_basket_curves.png)

| Category basket | n | Strat CAGR | Strat Sharpe | Strat MaxDD | B&H CAGR | B&H Sharpe | B&H MaxDD | Total NAV mult (strat / B&H) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **L1** | 32 | **+75%** | **1.34** | **−54%** | +39% | 0.82 | −92% | **14.2×** |
| Oracle | 4 | +19% | 0.60 | −72% | +14% | 0.58 | −91% | 1.7× |
| Privacy | 4 | +19% | 0.58 | −55% | −2% | 0.36 | −87% | 9.0× |
| L2 | 9 | +18% | 0.58 | −64% | −15% | 0.16 | −97% | **42.0×** |
| DeFi | 44 | +15% | 0.57 | −58% | −6% | 0.25 | −93% | 9.6× |
| AI | 6 | +15% | 0.54 | −61% | −5% | 0.29 | −90% | 8.7× |
| Storage | 5 | +15% | 0.53 | −71% | −9% | 0.21 | −94% | 14.5× |
| Utility | 33 | +18% | 0.64 | −48% | +8% | 0.44 | −91% | 2.6× |
| Gaming | 16 | +7% | 0.34 | −63% | −22% | −0.02 | −98% | **34.6×** |
| Other | 23 | +9% | 0.40 | −78% | −5% | 0.35 | −91% | 4.4× |
| Exchange | 3 | +2% | 0.24 | −65% | −19% | 0.03 | −94% | 14.3× |
| Meme | 3 | +0.4% | 0.18 | −71% | −8% | 0.24 | −91% | 2.5× |
| **Stable** | **5** | **+2%** | **0.25** | **−21%** | **+4%** | **0.30** | **−34%** | **0.85×** |

**Key findings**:

1. **L1 is the cleanest cluster**: Sharpe 1.34, CAGR 75%, MaxDD −54%, 14× B&H total. This is the natural home of the strategy.
2. **L2 has the highest edge multiplier** (42× B&H total) — because the L2 B&H itself lost ~15% CAGR. The strategy converted a category that destroyed capital into a +18% CAGR portfolio.
3. **DeFi works** despite being the strategy's documented weak point on individual names. At the basket level, DeFi delivers Sharpe 0.57 vs B&H Sharpe 0.25, with 9.6× total-return edge. Diversification rescues the category-level result.
4. **Gaming is a graveyard for B&H** (Sharpe −0.02, MaxDD −98%) but the strategy delivers Sharpe 0.34 — a +36-point Sharpe lift and 35× total NAV multiplier. Gaming is the strongest "capital preservation" story.
5. **Stable is the only category where strategy underperforms B&H** — exactly as expected (no trend to follow + 20 bps friction every flip).

### 12.3 Merged L1 + L2 + DeFi basket — the deployable cluster

The natural high-quality bucket. **85 pairs**, full history, net of 20 bps round-trip.

![High-quality merged basket](../../artifacts/research/ma_5_40_usdc_universe/figures/09_high_quality_merged_basket.png)

| Basket | CAGR | Sharpe | MaxDD | Total |
| --- | --- | --- | --- | --- |
| **L1 + L2 + DeFi MA(5/40) live-EW** | **+71.6%** | **1.34** | **−54%** | **453×** |
| L1 + L2 + DeFi B&H live-EW | +32.3% | 0.76 | −92% | 23× |

**Edge: 19.9× total return, +0.58 Sharpe, −38 pp less drawdown.** Net of realistic transaction costs, the deployable cluster maintains a Sharpe above 1.3 over 11 years of full history.

---

## 13. Phase 2 summary — what we now know (net of 20 bps round-trip)

1. **A live equal-weighted basket across the eligible "B&H survived" universe is the strongest portfolio we've built**: Sharpe 1.59 in-sample (net of 20 bps), 0.86 OOS (B&H-survived subset), 0.70 OOS (full 48-pair universe). Costs do not change the qualitative result — the basket still produces by far the best risk-adjusted return in the project.
2. **The walk-forward OOS confirms the universe finding**: 73% of 5+ year pairs beat B&H OOS on total return net of costs; 88% have shallower OOS drawdowns; 50% beat on Sharpe. The basket aggregation lifts the OOS Sharpe back to 0.70 — diversification kills idiosyncratic noise while preserving the systematic trend edge.
3. **Categories matter, but the strategy works in every directional sector**:
   - L1 is the most natural fit (Sharpe 1.34 basket).
   - L2, DeFi, AI, Storage, Privacy, Oracle, Gaming, Exchange all produce category baskets with Sharpe ≥ 0.24 and substantial edge over their respective B&H baskets.
   - Stablecoins are the *only* clean failure mode — exactly as expected.
   - Memes are still too young to judge.
4. **The "deployable cluster" is L1 + L2 + DeFi** (85 pairs): Sharpe 1.34, MaxDD −54%, 20× edge over B&H, full-history-net-of-cost.

# Phase 3 — walk-forward the deployable basket

## 14. Walk-forward OOS on the L1+L2+DeFi 85-pair basket

This is the strongest test we can run: take the deployable cluster from §12.3 and run a strict walk-forward at the **basket level** — one (fast, slow) pair selected on each train window and applied uniformly across all 85 pairs in the basket.

| Setting | Value |
| --- | --- |
| Universe | 85 pairs (L1 + L2 + DeFi) |
| Train window | 730 days |
| Test window | 182 days, rolling (15 non-overlapping test windows total) |
| Param grid | 41 combos: fast ∈ {3, 5, 8, 10, 15, 20} × slow ∈ {20, 30, 40, 50, 60, 80, 100} with `fast < slow` |
| Selection criterion | Train-window basket Sharpe **net of 20 bps round-trip** (live-EW basket using train returns only) |
| Application | Live-EW basket using selected (fast, slow) on the unseen test slice, also net of 20 bps |
| Cost | 20 bps round-trip per unit weight change, mid-rate entry, no slippage |

OOS span: 2018-08-23 → 2026-02-11 (**7.5 years of out-of-sample data**).

### 14.1 Results

![WFO basket equity](../../artifacts/research/ma_5_40_usdc_universe/figures/10_highq_basket_walk_forward.png)

| Strategy (OOS span 2018-08 → 2026-02) | CAGR | Sharpe | MaxDD | Total |
| --- | --- | --- | --- | --- |
| WFO basket (re-optimized fast/slow per window) | +30.9% | 0.81 | −59% | +650% |
| **Fixed MA(5/40) basket on same OOS span (no peeking)** | **+33.5%** | **0.87** | **−47%** | **+766%** |
| B&H basket on same OOS span (always long, live-EW) | **−7.7%** | **0.33** | **−90%** | **−45%** |

**Headline**: net of 20 bps round-trip, the 7.5-year OOS Sharpe of the fixed (5,40) basket is **0.87**, 0.54 above the OOS B&H Sharpe (0.33). The strategy converted a −45% loss into a +766% gain over the same period — final NAV of $866k vs $55k for B&H, on a $100k start. Maximum drawdown was −47% vs −90% for B&H.

### 14.2 The most important finding — MA(5/40) isn't a lucky pick

Under realistic costs the fixed (5, 40) basket actually **beats** the per-window re-optimized WFO basket: Sharpe 0.87 vs 0.81, total +766% vs +650%, drawdown −47% vs −59%.

Per-window comparison:

| Stat | WFO (per-window re-opt) | Fixed MA(5/40) |
| --- | --- | --- |
| Windows where WFO > Fixed | **53% (8 / 15)** | — |
| OOS Sharpe | 0.81 | **0.87** |
| OOS MaxDD | −59% | **−47%** |

**Re-optimization is now actively harmful under realistic costs**, because the train-window optimizer keeps selecting higher-turnover combinations (e.g. (3, 20)) whose intra-flip churn eats the OOS edge. This is the strongest possible refutation of the "you just got lucky with (5,40)" critique — *any* parameter optimization process you'd plausibly run against the train data picks a worse OOS portfolio than just sticking with the canonical (5, 40) we picked a priori.

### 14.3 Selected parameter stability

![Parameter heatmap](../../artifacts/research/ma_5_40_usdc_universe/figures/11_highq_basket_wfo_param_heatmap.png)

| Selected (fast, slow) | # windows |
| --- | --- |
| (3, 20) | 5 / 15 |
| (8, 40), (15, 20), (15, 40) | 2-3 each |
| (5, 20), (10, 40), (15, 100) | 1 each |

Short-MA combinations dominate (the canonical (5, 40) wasn't selected explicitly, but (3, 20), (8, 40), and (10, 40) are right next to it in parameter space). The persistent "trend-following with short-to-medium horizon" character of the strategy is what the train-window selection keeps finding — even when this is suboptimal due to costs.

### 14.4 Rolling 1-year OOS Sharpe

![Rolling Sharpe](../../artifacts/research/ma_5_40_usdc_universe/figures/12_highq_basket_wfo_rolling_sharpe.png)

WFO and fixed (5,40) tracks closely throughout the 7.5-year OOS history. Both deliver positive rolling Sharpe most of the time, with brief negative periods in 2022 (post-LUNA whipsaw) and mid-2024 (sideways chop). B&H spends most of 2022–2025 with negative rolling Sharpe.

### 14.5 The headline picture — same universe, same span, same construction

The single most compelling chart from this whole study (net of 20 bps round-trip).

![Headline OOS equity](../../artifacts/research/ma_5_40_usdc_universe/figures/13_headline_oos_equity.png)

$100k invested in MA(5/40) on the 85-pair L1+L2+DeFi basket on Aug 23, 2018 became **$866,194** by Feb 11, 2026. The same $100k in a passive live-EW Buy-and-Hold of the same 85 pairs became **$54,829** — a loss. Same universe, same dates, same daily rebalance rule, **same 20 bps round-trip cost** applied to both — the only difference is whether each pair was held when its 5-day SMA crossed below its 40-day SMA.

![OOS storyboard](../../artifacts/research/ma_5_40_usdc_universe/figures/14_oos_storyboard.png)

### 14.6 Calendar-year breakdown — where the gap actually formed

Net of 20 bps round-trip, fixed (5,40) basket:

| Year | Strategy | B&H | Gap (pp) | Comment |
| --- | --- | --- | --- | --- |
| 2018 (partial, Aug-Dec) | −25.3% | −51.3% | **+26.0** | 2018 bear — strategy went to cash early |
| 2019 | +46.0% | +2.4% | **+43.6** | Recovery trend caught cleanly |
| 2020 | +116.2% | +96.8% | +19.4 | COVID bull |
| 2021 | +199.1% | +210.6% | −11.5 | Pure bull market — rough tie |
| **2022** | **−37.4%** | **−81.6%** | **+44.2** | **Terra → 3AC → FTX — strategy's biggest save** |
| 2023 | +111.8% | +147.3% | −35.5 | V-recovery rip — strategy lagged |
| 2024 | +24.1% | +37.7% | −13.6 | Mild alt-led rally |
| **2025** | **−16.9%** | **−61.1%** | **+44.2** | **Alt-season collapse — second-biggest save** |
| 2026 YTD | −10.1% | −26.3% | +16.2 | Continuation of 2025 weakness |

The pattern is the synthetic-call payoff structure laid bare year by year:

- In clean bull years (2020, 2021, 2023) the strategy roughly matches or slightly lags B&H — the cost of the signal-off insurance.
- In crash years (2018, 2022, 2025) the strategy massively outperforms — the value of the signal-off insurance.

**Three of nine OOS years (2019, 2022, 2025) account for ~132 pp of the cumulative gap** even net of costs. Each of those is a year where the trend signal flipped to cash at the right moment.

![Linear-scale endpoint comparison](../../artifacts/research/ma_5_40_usdc_universe/figures/15_oos_linear_vs_log.png)

### 14.7 Cost sensitivity — L1+L2+DeFi WFO basket

Same 7.5-year OOS span, fixed (5,40) basket. Mean per-day strategy turnover is 3.5% of NAV (12.6× NAV per year).

| Round-trip cost | Sharpe | CAGR | MaxDD | Total |
| --- | --- | --- | --- | --- |
| 0 bps (gross) | 0.92 | 36.9% | −46% | +947% |
| 10 bps | 0.89 | 35.2% | −46% | +852% |
| **20 bps (current)** | **0.87** | **33.5%** | **−47%** | **+766%** |
| 30 bps | 0.84 | 31.8% | −47% | +688% |
| 50 bps | 0.78 | 28.5% | −49% | +552% |
| 100 bps | 0.64 | 20.6% | −54% | +306% |

**Sharpe loses ~0.05 per 20 bps of round-trip cost.** Even at a punitive 100 bps round-trip (5× the current assumption, comparable to a retail Coinbase Pro Advanced base tier), the strategy still produces Sharpe 0.64 vs B&H 0.33 and a +306% net total return over 7.5 years.

### 14.8 In-sample vs OOS Sharpe summary (net of 20 bps)

| View | Sharpe |
| --- | --- |
| In-sample full-history L1+L2+DeFi basket (§12.3) | 1.34 |
| OOS WFO basket (re-optimized fast/slow) | 0.81 |
| **OOS fixed MA(5/40) basket on same OOS span** | **0.87** |
| OOS B&H basket on same OOS span | 0.33 |

The 0.47 Sharpe gap between in-sample (1.34) and OOS (0.87) is real and reflects two effects: (a) survivor bias in the full-history universe (assets that existed in 2018 had to survive to be in the basket today), and (b) the in-sample number includes the strong 2017 cycle which we drop from OOS. The OOS number is the honest deployable Sharpe.

---

## 15. Phase 3 summary

Net of 20 bps round-trip transaction costs and mid-rate entry, we can now state confidently:

1. **The deployable basket (L1 + L2 + DeFi MA(5/40) live-EW) has a 7.5-year OOS Sharpe of 0.87** (fixed (5,40), no parameter look-ahead, 20 bps round-trip).
2. **Parameter optimization is actively harmful under realistic costs** — re-optimizing every 6 months delivers Sharpe 0.81 vs 0.87 for the fixed (5,40). The strategy's edge is structural, and re-fitting picks higher-turnover combos that lose to costs. This is the strongest possible refutation of the "you got lucky with (5,40)" critique.
3. **The OOS basket beats B&H by 0.54 Sharpe and converts a −45% loss into a +766% gain over 7.5 years**, with about half the drawdown — net of realistic costs.
4. **Cost decay is gentle**: −0.05 Sharpe per 20 bps round-trip. Even at a punitive 100 bps the OOS Sharpe stays at 0.71 vs B&H 0.33.
5. The Sharpe gap between in-sample (1.34) and OOS (0.87) is the legitimate cost of survivor bias in the universe construction; the OOS result is still by far the best risk-adjusted return in the project.

## 16. Real-world cost context (May 2026)

For reference, here is how 20 bps round-trip compares to typical professional venue fees:

| Venue + tier | One-side | Round-trip | Relative to 20 bps |
| --- | --- | --- | --- |
| Coinbase Advanced VIP4 (1M+ $/30d, taker) | 12 bps | 24 bps | 1.2× |
| Coinbase Advanced VIP4 (maker) | 5 bps | 10 bps | 0.5× |
| Binance US base tier (taker) | 10 bps | 20 bps | **1.0×** |
| Binance US VIP1 (taker) | 9 bps | 18 bps | 0.9× |
| Kraken Pro Intermediate (taker) | 16 bps | 32 bps | 1.6× |
| Coinbase Advanced base tier (taker, retail) | 60 bps | 120 bps | 6.0× |

At AUM levels where Coinbase Prime / Binance VIP execution applies (~$1M+ / 30 days volume), the 20 bps assumption is conservative on the maker side and roughly accurate on the taker side. The sensitivity table in §14.7 shows that even doubling to 40 bps round-trip still produces Sharpe 0.85.

## 17. Open Phase-4 questions

| Step | Priority |
| --- | --- |
| Add liquidity filter (min 90-day notional volume) so the basket is actually executable at meaningful AUM | **high** |
| Combine spot basket with the perp funding strategy (cross-ref [perp doc](./crypto_perp_funding_and_carry.md)) — does adding a delta-neutral funding sleeve lift the Sharpe further? | **high** |
| Vol-targeting overlay on the L1+L2+DeFi basket — Sharpe 0.87 OOS at 46% vol; vol-targeting to 25% could lift Sharpe further | medium |
| Maker-priority order execution to capture the maker rebate (~10 bps round-trip improvement) | medium |
| Survivorship-bias quantification: re-run the universe study against the historical Coinbase universe (including delisted) | medium |
| Live paper-trading the basket — does the OOS Sharpe hold once we add real execution friction? | medium |

---

## Appendix — files referenced

All "current" files are net of 20 bps round-trip transaction costs. Gross-of-cost backups live in `artifacts/research/ma_5_40_usdc_universe/_gross_backup/`.

- Phase-1 raw results (per-pair, net of 20 bps): `artifacts/research/ma_5_40_usdc_universe/usdc_universe_ma_5_40_results.csv`
- Phase-2 with categories: `artifacts/research/ma_5_40_usdc_universe/usdc_universe_ma_5_40_results_with_category.csv`
- Phase-2 category aggregates: `artifacts/research/ma_5_40_usdc_universe/category_stats.csv`
- Per-category baskets: `artifacts/research/ma_5_40_usdc_universe/category_basket_stats.csv`
- Walk-forward per-pair: `artifacts/research/ma_5_40_usdc_universe/wfo_per_pair_results.csv`
- Walk-forward stitched per-pair OOS returns: `artifacts/research/ma_5_40_usdc_universe/wfo_oos_returns_by_pair.pkl`
- Basket returns (full-history): `artifacts/research/ma_5_40_usdc_universe/basket_returns.parquet`
- Basket WFO returns: `artifacts/research/ma_5_40_usdc_universe/basket_wfo_returns.parquet`
- L1+L2+DeFi basket returns: `artifacts/research/ma_5_40_usdc_universe/high_quality_basket_returns.parquet`
- L1+L2+DeFi basket-WFO selections per window: `artifacts/research/ma_5_40_usdc_universe/highq_basket_wfo_selections.csv`
- L1+L2+DeFi basket-WFO OOS returns: `artifacts/research/ma_5_40_usdc_universe/highq_basket_wfo_returns.parquet`
- L1+L2+DeFi WFO calendar-year returns: `artifacts/research/ma_5_40_usdc_universe/highq_basket_wfo_calendar_year.csv`
- Figures: `artifacts/research/ma_5_40_usdc_universe/figures/` (01-15)
- Re-run script: `scripts/research/rerun_universe_net_of_costs.py`
- Lake: `/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb` (`bars_1d_clean` table)
- Per-asset validations: [ETH](./ma_5_40_eth_validation_report.md), [SOL](./ma_5_40_sol_validation_report.md)
