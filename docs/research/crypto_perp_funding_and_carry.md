# Crypto perpetual funding & cash-and-carry research

**Status**: complete (Phase 1)
**Date**: 2026-05-19
**Companion docs**: [MA(5/40) ETH validation report](./ma_5_40_eth_validation_report.md)
**Source code**: `scripts/binance/fetch_eth_perp.py`
**Raw data**: `data/binance_perp/`
**Figures**: `artifacts/research/crypto_perp_funding/figures/`

---

## TL;DR

1. **Funding is the dominant economic factor on crypto perpetuals.** Average funding on the 12 most-liquid Binance USD-M perps runs **+12-15% APR** for the majors (longs pay shorts), making perps a structurally inferior instrument vs. spot for long-only directional trading.

2. **Trend-following on perps degrades sharply because of funding.** MA(5/40) long-only on ETH-PERP retains the drawdown-reduction property (-55% MaxDD vs B&H's -80%) but loses ~50% of cumulative return to funding (525% vs spot's 1,204%).

3. **Long/short trend on perps is a disaster.** Whipsaws + adverse funding correlation give -87% drawdown and 3% CAGR on ETH-PERP over 6 years.

4. **The clean edge is cash-and-carry**: long spot, short perp, harvest funding. Both BTC and ETH single-asset carry trades produced **Sharpe 5.1-5.7, MaxDD ≤3%, 7-14% CAGR** depending on leverage. The strategy was profitable through 2022's -67% ETH crash (delta-neutral).

5. **The basket version is worse than the single-name version.** XRP alone destroyed the basket Sharpe (-74% per leg from SEC-era basis blowouts). The cleanest basket is just **BTC + ETH equal-weighted** — Sharpe ~5.4 at 5x perp leverage.

---

## 1. Data infrastructure

### 1.1 Source: Binance public archive

`fapi.binance.com` is geofenced from US IPs (HTTP 451), so we ingest from Binance's official public archive at `data.binance.vision`, which is hosted on AWS CloudFront and is US-accessible. The archive contains the exact same data as the live API:

- Monthly ZIP files for completed months: `data/futures/um/monthly/{klines,fundingRate}/SYMBOL/...`
- Daily ZIP files for the current partial month

The fetcher script `scripts/binance/fetch_eth_perp.py` walks both monthly and daily archives, falling back to daily for the trailing partial month. Output: parquet files in `data/binance_perp/`:

- `{sym}_1d.parquet` — daily OHLCV
- `{sym}_funding_8h.parquet` — raw 8h funding rates
- `{sym}_funding_1d.parquet` — daily-aggregated funding (sum of 3 payments per day)

### 1.2 Universe and coverage

12 of the most-liquid Binance USD-M perpetuals from contract launch through 2026-05-18:

| Symbol | Start date | End date | Days |
| --- | --- | --- | --- |
| BTCUSDT | 2020-01-01 | 2026-05-18 | 2,330 |
| ETHUSDT | 2020-01-01 | 2026-05-18 | 2,330 |
| XRPUSDT | 2020-01-06 | 2026-05-18 | 2,320 |
| LTCUSDT | 2020-01-09 | 2026-05-18 | 2,317 |
| LINKUSDT | 2020-01-17 | 2026-05-18 | 2,314 |
| ADAUSDT | 2020-01-31 | 2026-05-18 | 2,300 |
| BNBUSDT | 2020-02-10 | 2026-05-18 | 2,290 |
| DOGEUSDT | 2020-07-10 | 2026-05-18 | 2,139 |
| DOTUSDT | 2020-08-22 | 2026-05-18 | 2,096 |
| SOLUSDT | 2020-09-14 | 2026-05-18 | 2,068 |
| AVAXUSDT | 2020-09-23 | 2026-05-18 | 2,064 |
| MATICUSDT | 2020-10-22 | **2024-09-11** | 1,421 (delisted — rebranded to POL) |

### 1.3 Data quality cross-check

Binance perp close vs. Coinbase spot close, daily:

| Pair | Common days | Correlation | Mean basis | Max abs basis |
| --- | --- | --- | --- | --- |
| Binance ETHUSDT perp vs Coinbase ETH-USD | 2,329 | **0.999998** | -3 bps | 3.25% |
| Binance BTCUSDT perp vs Coinbase BTC-USD | 2,329 | **0.999998** | -4 bps | 1.90% |

Same instrument economically. Data is trustworthy.

---

## 2. Funding rate landscape

![Funding rate landscape](../../artifacts/research/crypto_perp_funding/figures/00_funding_rate_landscape.png)

| Symbol | Avg funding APR | Regime |
| --- | --- | --- |
| LTCUSDT | +15.5% | high |
| XRPUSDT | +15.5% | high |
| MATICUSDT | +14.8% | high |
| ETHUSDT | +14.5% | high |
| LINKUSDT | +14.3% | high |
| ADAUSDT | +14.2% | high |
| DOGEUSDT | +12.9% | high |
| BTCUSDT | +12.2% | high |
| DOTUSDT | +8.2% | mid |
| AVAXUSDT | +7.2% | mid |
| **SOLUSDT** | **+0.1%** | flat (two-sided positioning) |
| **BNBUSDT** | **-0.3%** | negative (shorts pay longs!) |

**Positive funding = longs pay shorts.** Most majors run +12-15% APR funding *as a baseline cost of being long perp instead of spot*. SOL and BNB are anomalies — heavy two-sided positioning produces near-zero or slightly-negative average funding.

Funding is heavily **regime-dependent**:

- **2020-2021 (euphoric bull)**: ETH +27% then +38% APR; all majors red-hot
- **2022 (deep bear)**: ETH +0.8%; some names dipped negative
- **2023-2024**: normalization back to +8-13% APR
- **2025-2026 YTD**: modest, with several names briefly negative

The drag on a perpetual long position is **most punishing exactly when you most want to be long**. This is the core problem with trend-following on perps.

---

## 3. Long-only MA(5/40) on perps: the funding tax

Same MA(5/40) strategy as in the [ETH validation report](./ma_5_40_eth_validation_report.md), now run on Binance perps with funding cost applied to long positions:

### 3.1 ETH

| Strategy (2020-01 → 2026-05) | CAGR | Sharpe | MaxDD | Total | NAV (from $100k) |
| --- | --- | --- | --- | --- | --- |
| ETH Spot B&H (no funding) | 55.0% | 0.95 | -79% | +1,542% | $1,642,354 |
| ETH Perp B&H (with funding) | 34.2% | 0.78 | -80% | +552% | $651,869 |
| ETH MA(5/40) spot long-only | 50.6% | 0.99 | -53% | +1,204% | $1,304,171 |
| **ETH MA(5/40) perp long-only** | **33.9%** | **0.79** | **-55%** | **+525%** | **$624,741** |
| ETH MA(5/40) perp long/short | 3.1% | 0.45 | **-87%** | +21% | $121,412 |

### 3.2 BTC

| Strategy (2020-01 → 2026-05) | CAGR | Sharpe | MaxDD | Total |
| --- | --- | --- | --- | --- |
| BTC Spot B&H (no funding) | 45.3% | 0.92 | -77% | +986% |
| BTC Perp B&H (with funding) | 28.5% | 0.72 | -79% | +395% |
| BTC MA(5/40) spot long-only | 50.1% | 1.17 | -64% | +1,179% |
| **BTC MA(5/40) perp long-only** | **36.1%** | **0.94** | **-65%** | **+591%** |
| BTC MA(5/40) perp long/short | 22.9% | 0.64 | -71% | +265% |

### 3.3 Key observations

1. **Funding compounded to a -60% drag on B&H** and a -52% drag on MA long-only ETH. The drawdown asymmetry survives, but the total return edge is gutted.
2. **BTC is more viable than ETH on perp.** BTC has 200 bps lower funding (12.2% vs 14.5% APR), and the trend signal is sharper. BTC MA perp long-only still beats BTC perp B&H (1.49× total return); ETH does not.
3. **Long/short is a disaster on both.** Trend on the short side gets whipsawed in chop, *and* funding correlation works against you (positions tend to be long when funding is most positive). 2022 ETH dropped -67% but MA L/S still lost 29% gross from whipsaws; 2024 was even worse (-44%).
4. **The "synthetic call" property carries over.** The MA drawdown is -55% on perp vs B&H's -80% — same 25 percentage-point improvement we saw on spot.

![Perp vs Spot economics](../../artifacts/research/crypto_perp_funding/figures/01_perp_vs_spot.png)
![BTC perp + carry](../../artifacts/research/crypto_perp_funding/figures/04_btc_perp_and_carry.png)

---

## 4. Funding-aware MA gating: modest improvement

We tried gating the long signal by funding rate to skip days when funding is too expensive:

| Rule | Sharpe | CAGR | MaxDD | % long |
| --- | --- | --- | --- | --- |
| Baseline: MA(5/40) long-only (no gate) | 0.79 | 33.9% | -55.0% | 52.9% |
| R1 hard gate: long iff 1d funding < 30% APR | 0.80 | 30.8% | -52.4% | 42.3% |
| **R2 30d-MA gate: long iff 30d funding < 20% APR** | **0.84** | 30.4% | -47.8% | 37.6% |
| R3 continuous decay: w = max(0, 1 - funding/100%) | 0.79 | 29.5% | -49.2% | 50.6% |

Best Sharpe improvement: 0.79 → 0.84. DD improves -55% → -48%. CAGR drops 3-4pp.

The problem is structural: funding is highest in strong bull trends, which is exactly when the MA signal wants to be long. Gating filters out a lot of the strategy's good days. **Not a meaningful unlock.**

![Funding-aware MA gating](../../artifacts/research/crypto_perp_funding/figures/03_eth_perp_funding_aware.png)

---

## 5. Cash-and-carry: the clean edge

If funding is the dominant force, the natural trade is to **harvest** it rather than fight it:

- Long $1 of ETH (or BTC) spot on Coinbase
- Short $1 of the matching perp on Binance
- Short leg receives positive funding each 8h cycle
- Net price exposure: zero (basis P&L only, which is mean-reverting around zero)

### 5.1 Per-asset results

| Asset | Sizing | CAGR | Sharpe | MaxDD | Total |
| --- | --- | --- | --- | --- | --- |
| ETH | $2 cap (unencumbered: $1 spot + $1 perp) | 7.6% | **5.13** | -2.0% | +60% |
| ETH | $1.20 cap (5x perp leverage) | 13.0% | **5.13** | -3.4% | +118% |
| ETH | $1.10 cap (10x perp leverage) | 14.2% | **5.13** | -3.7% | +134% |
| BTC | $2 cap (unencumbered) | 6.3% | **5.69** | -1.7% | +48% |
| BTC | $1.20 cap (5x perp leverage) | 10.7% | **5.69** | -2.8% | +92% |
| BTC | $1.10 cap (10x perp leverage) | 11.8% | **5.69** | -3.0% | +103% |

### 5.2 Where the return comes from

Decomposition for ETH carry over the full 2020-2026 window, per $1 of leg notional:

| Source | Cumulative % |
| --- | --- |
| Funding harvested (short leg collects) | **+150.4%** |
| Basis P&L (spot - perp price drift) | +1.9% |
| **Total per leg** | **+155.2%** |

It's essentially **pure funding harvest**; basis is noise.

### 5.3 Behavior through regimes (ETH unencumbered, $2 cap)

| Year | Funding APR | Carry return (per leg) |
| --- | --- | --- |
| 2020 | +27.4% | +31% |
| 2021 | +37.5% | +45% |
| 2022 | +0.8% | +1% |
| 2023 | +8.3% | +9% |
| 2024 | +13.0% | +14% |
| 2025 | +4.9% | +5% |
| 2026 YTD | -0.3% | +2% (basis tailwind) |

The 2022 result is the validator: a brutal ETH bear (-67% spot drawdown), and the cash-and-carry trade **still made +1%**. Pure delta-neutrality works.

![ETH cash-and-carry](../../artifacts/research/crypto_perp_funding/figures/02_eth_cash_and_carry.png)

---

## 6. Multi-asset basket: diversification doesn't help

### 6.1 Basket MA(5/40) long-only with funding

| Strategy (2021-01 → 2026-05) | CAGR | Sharpe | MaxDD |
| --- | --- | --- | --- |
| Basket B&H (eq-wt 12 perps, with funding) | 41.2% | 0.83 | -80% |
| BTC-PERP MA(5/40) only (same window) | 10.2% | 0.45 | -65% |
| ETH-PERP MA(5/40) only (same window) | 4.4% | 0.34 | -55% |
| **Basket MA(5/40), fixed 1/12 budget** | **24.3%** | **0.71** | **-54%** |
| Basket MA(5/40), pro-rata across actives | 1.1% | 0.38 | -83% |

Diversification across the 12-name basket does help vs single-name (Sharpe 0.71 fixed-1/12 vs 0.34-0.45 for BTC/ETH alone), but it doesn't beat just holding the basket (0.83). The fundamental issue is unchanged: funding eats the trend edge.

![Basket MA perps](../../artifacts/research/crypto_perp_funding/figures/05_perp_basket_ma_5_40.png)

### 6.2 Per-symbol cash-and-carry — the basket falls apart

| Symbol | Funding APR | Carry Sharpe | Total/leg |
| --- | --- | --- | --- |
| **BTCUSDT** | +11.1% | **6.04** | +84% |
| **ETHUSDT** | +12.0% | **5.40** | +95% |
| **LINKUSDT** | +14.7% | **5.05** | +112% |
| **LTCUSDT** | +13.6% | **2.59** | +121% |
| DOGEUSDT | +13.1% | 1.21 | +46% |
| AVAXUSDT | +8.1% | 0.92 | +16% |
| ADAUSDT | +12.0% | 0.83 | +34% |
| **XRPUSDT** | +13.6% | **-0.35** | **-74%** |

Two failure modes:

1. **XRP completely blows up the carry** (-74% per leg despite +13.6% funding) — the SEC lawsuit era 2021-2023 produced catastrophic basis blowouts (Coinbase delisted XRP and the perp price diverged from spot price).
2. **Low-funding names (SOL, BNB) have no edge to harvest.**

### 6.3 Best-4 basket: BTC + ETH + LINK + LTC, equal-weighted

| Sizing | CAGR | Sharpe | MaxDD |
| --- | --- | --- | --- |
| Best-4 carry unencumbered | 7.4% | 4.90 | -2.5% |
| Best-4 carry 5x perp | 12.7% | **4.90** | -4.1% |
| Best-4 carry 10x perp | 13.9% | **4.90** | -4.5% |

The basket *slightly underperforms* the BTC and ETH singles (Sharpe 5.69 / 5.13). Diversification across crypto carries doesn't reduce variance much because all funding regimes are highly correlated (bull → all high, bear → all low). The basket just adds idiosyncratic basis risk on names like LINK without lowering portfolio vol.

**Conclusion**: the cleanest expression is **BTC + ETH equal-weighted, daily rebalanced**, sized to 5x perp leverage. Expected ~12% CAGR / Sharpe ~5.4 / ≤4% DD over the next several years if funding rates stay in the historical 8-15% APR range.

![Best-4 carry basket](../../artifacts/research/crypto_perp_funding/figures/07_best4_cash_and_carry.png)

---

## 7. Implementation risks and unknowns

The backtest is clean, but real-world execution requires caveats:

### 7.1 Venue accessibility

**Binance is not legally accessible to US persons.** US-accessible alternatives for the short-perp leg:

| Venue | Funding mechanic | History | Notes |
| --- | --- | --- | --- |
| CME ether/bitcoin futures | Basis curve (not funding) — different beast | 8+ yrs | Regulated. Contango/backwardation roll cost replaces funding. |
| Coinbase International Exchange | Perpetual w/ funding | Since 2023 | Younger but a legit perp venue. |
| Hyperliquid | Perpetual w/ funding | Since 2022 | Decentralized, US-accessible, growing volumes. |
| dYdX | Perpetual w/ funding | Since 2021 | Decentralized, US-accessible. |
| Bitnomial | Futures (regulated) | Since 2024 | US-regulated perp/futures venue. |

The cash-and-carry idea is sound; the venue-specific funding curve will differ from Binance's. CME basis trade is a separate research project (different mechanic).

### 7.2 Counterparty risk

- 2 venues per trade (spot + perp). Either side failing (FTX-style) loses that leg entirely.
- Mitigations: split spot leg across exchanges, use cold-storage where possible, post minimum perp margin (no excess collateral).

### 7.3 Liquidation risk on perp leg

- Max single-day basis loss in the data: ETH 2.5%, BTC 1.9% (both during March 2020 COVID crash).
- At 10x perp leverage, a 9% adverse perp move = liquidation. Spot leg doesn't unwind fast enough.
- **5x perp leverage is the prudent ceiling.** Same Sharpe, lower blow-up risk.

### 7.4 Funding regime risk

- The historical +12-15% APR funding is largely a function of crypto's persistent bull bias and retail leverage demand.
- If retail leverage demand dries up structurally (e.g., post-spot-ETF era), funding could compress toward zero.
- 2026 YTD has been the first stretch of *negative* average funding for several names. Worth monitoring.

---

## 8. Bottom line and next steps

### Bottom line

- **MA(5/40) on perps is structurally worse than on spot.** Funding eats ~50% of cumulative returns. Long/short doesn't help — it makes things worse.
- **Cash-and-carry is the real find of this investigation.** BTC and ETH carry trades produced Sharpe 5+ with bond-like volatility and survived 2022 with no drawdown. It is **the only crypto strategy we have tested** that delivered consistent positive returns through both bull and bear regimes.
- **Trend and carry are complementary.** Trend gives upside convexity on spot ETF / spot exposure. Carry gives steady compounding through bears. **Both belong in a portfolio.**

### Open research questions

1. **CME basis trade**: equivalent of cash-and-carry using CME ether/bitcoin futures contango. Different mechanic (term-structure roll), US-regulated venue. *Highest-value follow-up.*
2. **Funding regime forecasting**: can we predict funding rate decay over the next 3-6 months from on-chain/positioning data? If so, position-size carry dynamically.
3. **Negative-funding capture**: should we ever go *short* spot + *long* perp when funding turns negative? Data is thin (only 12% of days in our sample), but the implied edge is symmetric.
4. **Multi-venue spread**: arbitrage funding differentials between Binance, Bybit, OKX. If venue A pays +20% and venue B pays +10%, short on A, long on B = +10% spread less basis risk.
5. **Combining trend + carry**: portfolio with both trend (synthetic call) and carry (synthetic credit) sleeves. What's the optimal allocation? Markowitz-style, or vol-targeted?

---

## Appendix — files referenced

- **Ingest script**: `scripts/binance/fetch_eth_perp.py`
- **Raw data**: `data/binance_perp/{symbol}_{1d,funding_1d,funding_8h}.parquet`
- **Figures**: `artifacts/research/crypto_perp_funding/figures/`
- **Companion strategy doc**: `docs/research/ma_5_40_eth_validation_report.md`
