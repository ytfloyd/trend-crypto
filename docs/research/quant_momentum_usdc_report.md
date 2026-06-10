# Full-USDC Quantitative Momentum

**Status:** internal research note  
**Date:** 2026-05-23  
**Script:** `scripts/research/quant_momentum_usdc.py`  
**Artifacts:** `artifacts/research/quant_momentum_usdc/`

## Summary

We replicated the Gray/Vogel *Quantitative Momentum* recipe on the full Coinbase
USDC spot universe: rank by generic momentum, keep the strongest momentum pool,
then select the smoothest "frog-in-the-pan" paths by positive-day frequency.
The faithful crypto adaptation uses 365-day momentum skipping the most recent
30 days. We also tested faster 180/30 and 90/14 variants.

The result is negative across the board. The faithful 365/30 version is the
least bad, losing 64-69% from first exposure. Shorter lookbacks amplify the
same pump-and-fade problem documented in the weekly-breakout full-universe
study, with 90/14 losing 91-93%.

![Equity and drawdown](../../artifacts/research/quant_momentum_usdc/figures/01_equity_drawdown.png)

## Method

- Universe: all Coinbase `*-USDC` pairs in `bars_1d_clean`, excluding stable,
  wrapped, and liquid-staking bases, with at least 365 days history and at
  least 90% coverage.
- Point-in-time trading eligibility: trailing 90-day median dollar volume of at
  least $500k.
- Ranking: generic momentum over the formation window, excluding the skip
  window; take the top 100 momentum names, then select the top 50 by FIP path
  quality.
- FIP path quality: share of positive daily close-to-close returns in the
  formation window.
- Rebalance: monthly at month-end signal, one-bar execution lag.
- Costs: 30 bps per side on turnover.
- Sizing: equal weight primary, inverse-volatility sensitivity.
- Benchmark: BTC-USDC buy-and-hold, aligned to the earliest active strategy
  date in the variant set.

## Headline Results

| Variant | Active window | Final equity | CAGR | Vol | Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| 365/30 equal | 2023-04-01 to 2026-05-23 | $31,217 | -31.0% | 68.0% | -0.20 | -82.9% |
| 365/30 inv-vol | 2023-04-01 to 2026-05-23 | $35,795 | -27.9% | 66.0% | -0.16 | -81.3% |
| 180/30 equal | 2022-02-01 to 2026-05-23 | $12,040 | -39.1% | 70.6% | -0.34 | -92.0% |
| 180/30 inv-vol | 2022-02-01 to 2026-05-23 | $15,442 | -35.4% | 68.5% | -0.29 | -89.6% |
| 90/14 equal | 2021-10-01 to 2026-05-23 | $7,644 | -43.6% | 74.0% | -0.40 | -95.9% |
| 90/14 inv-vol | 2021-10-01 to 2026-05-23 | $9,908 | -40.3% | 71.5% | -0.36 | -94.4% |
| BTC-USDC B&H | 2021-10-01 to 2026-05-23 | $154,983 | +9.9% | 52.3% | +0.44 | -76.7% |

![Sharpe comparison](../../artifacts/research/quant_momentum_usdc/figures/02_sharpe_comparison.png)

## Interpretation

The FIP quality filter does not rescue full-universe crypto momentum. It prefers
smooth paths within the already-high-momentum pool, but the broad USDC universe
still contains many reflexive altcoin runs whose forward returns mean-revert
after selection. The shorter 90/14 and 180/30 variants are worse because they
respond more quickly to those short-cycle pumps.

Inverse-volatility sizing helps modestly in every lookback, but not enough to
change the conclusion. The issue is signal direction and universe composition,
not position sizing.

![Most selected symbols](../../artifacts/research/quant_momentum_usdc/figures/03_top_selected_symbols.png)

## Caveats

- The Coinbase lake appears to represent currently listed pairs, so delisted
  pairs are likely missing. This survivorship bias should improve the reported
  full-universe results relative to a truly point-in-time universe.
- The book strategy was designed for equities after a large/liquid universe
  screen. Applying it to every structurally eligible USDC pair is intentionally
  stress-testing the rule on a much noisier asset set.
- The BTC benchmark is aligned to the earliest active strategy date
  (2021-10-01), while each strategy row reports its own active window.

## Artifacts

- Metrics: `artifacts/research/quant_momentum_usdc/metrics.csv`
- NAVs: `artifacts/research/quant_momentum_usdc/navs.csv`
- Holdings: `artifacts/research/quant_momentum_usdc/holdings.csv`
- Universe: `artifacts/research/quant_momentum_usdc/universe.csv`
- Config: `artifacts/research/quant_momentum_usdc/config.json`
