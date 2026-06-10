# Weekly Breakout Strategy — Universe Sensitivity

**Status:** internal research note (not for publication)
**Date:** 2026-05-23
**Author:** trend_crypto research desk
**Companion to:** `docs/blog/trailing_stops_substack.{md,pdf}` (the published
stop-design study)

## Summary

The weekly breakout trend-following strategy delivers Sharpe 0.71 / CAGR 17.8%
on a 10-name bluechip universe (BC-10: BTC, ETH, SOL, XRP, ADA, DOGE, AVAX,
LINK, DOT, LTC, all quoted in USDC) over a 9-year window. Extending the
universe to all 273 USDC-quoted spot pairs that pass the standard structural
filters (≥365 days history, ≥90% coverage, ≥$500k median 90-day $-volume,
non-stablecoin) converts the strategy into one that loses 75–95% of capital
in absolute terms over the matched 6-year window, regardless of which of the
three stop variants (no-stop, fixed 3×ATR, trailing 3×ATR) is used.

The bluechip restriction is not a curated cherry-pick. It defines the only
universe on which this strategy has positive edge over the periods we have
data for.

## Method

We re-ran the three exit specifications studied in the published note on the
full 273-pair USDC universe with all other parameters held constant
(`scripts/research/v4_full_usdc_universe.py`). The wider universe's
backtest naturally starts 2020-05-24 because the default `min_eligible_at_start = 15`
threshold is not met before that date in the wide universe (only BC-10
names existed earlier with sufficient history). For apples-to-apples
comparison we re-ran the BC-10 over the same 2020-05-24 → 2026-05-23 window.

Caveat: pairs that were listed and subsequently delisted before 2026 are
not represented in the lake, so the 273-pair set carries survivorship bias.
This bias *improves* the wide-universe absolute return relative to a fully
point-in-time-correct equivalent.

## Headline numbers

### 2020-05-24 → 2026-05-23 (6.0 years, matched window)

| Variant         | BC-10 (10 pairs) | Full USDC (273 pairs) | Universe effect (CAGR) |
|---|---:|---:|---:|
| No stop         | Sh 0.61, CAGR +16.1%, MaxDD -55.9%, Total +145% | Sh 0.03, CAGR -20.4%, MaxDD -95.8%, Total **-75%** | **-36 pp** |
| Fixed 3×ATR     | Sh 0.66, CAGR +17.5%, MaxDD -52.5%, Total +163% | Sh -0.08, CAGR -24.2%, MaxDD -96.9%, Total **-81%** | **-42 pp** |
| Trailing 3×ATR  | Sh 0.17, CAGR +0.7%,  MaxDD -50.4%, Total +4%   | Sh -0.54, CAGR -40.1%, MaxDD -98.5%, Total **-95%** | **-41 pp** |
| BTC HODL (ref)  | Sh 0.91, CAGR +43.1%, MaxDD -76.7%, Total +756% | (same as BC-10)                                  | —    |

### Observations

1. **The stop-cost hierarchy holds direction-wise on the wide universe.**
   Trailing < fixed < no-stop on every risk-adjusted measure. The
   article's thesis on stop geometry survives universe expansion as an
   ordering.

2. **Absolute returns collapse uniformly across stop choices.** All three
   variants destroy 75–95% of capital. Stops are a third-order concern
   once the underlying signal is destructive.

3. **Stops do not even reduce drawdown on the wide universe.** Fixed
   stops produce a *worse* max DD (-96.9%) than no-stop (-95.8%); trailing
   is worse still (-98.5%). The "trade CAGR for DD protection" relationship
   documented on BC-10 disappears entirely. Stops force locked-in losses
   on alts that subsequently bounce, then continue down.

4. **Universe effect (~40 pp of CAGR) dwarfs stop-choice effect (~20 pp).**
   For the strategy as specified, choice of universe is the dominant
   design lever; choice of stop is the secondary lever within the chosen
   universe.

## Mechanism

Trade-level diagnostics on the wide-universe fixed-stop variant show:

- 2,496 trades across **155 distinct symbols** (out of 273 candidates) over 6 years
- BC-10 names account for only **17% of trades** despite carrying the
  majority of dollar exposure (because of inverse-vol sizing favoring
  lower-vol names)
- The most-traded non-BC names include ARB, BLZ, ZETA, BIGTIME, TRUMP,
  DRIFT, PROMPT, WIF, PEPE, MOG — alts whose price action is characterized
  by short-cycle pump-and-fade dynamics rather than persistent trend

The 5-day breakout filter combined with cross-sectional momentum ranking
selects on *just-realized* upside on whatever the top of the alt-coin
distribution happens to be in any given week. On BTC/ETH/SOL, that filter
identifies persistent trend regimes that last weeks-to-months. On the long
tail of alts, that same filter identifies pump tops — moments at which
realized momentum is high but expected forward return is negative because
the pump is mean-reverting.

The momentum-rotation logic, in other words, is a directional sort. On
assets that trend (bluechips), it sorts toward winners. On assets that
pump-and-fade (the long tail), it sorts toward eventual losers.

## Implications

For the published stop-design piece:

- Add one paragraph in the methodology section noting that the bluechip
  restriction is foundational (now done in revision 4 of the article).
- No change to the stop-design conclusions: the hierarchy holds on both
  universes; only the absolute magnitudes change.

For the strategy as a deployable product:

- BC-10 restriction is the operative spec. Do not deploy any version of
  this signal on a wider universe without a substantial redesign of either
  (a) the entry filter (less responsive to short-cycle pumps), or (b) the
  universe definition (deterministic top-N-by-trailing-volume with a much
  higher liquidity floor).

For future research (logical next pieces):

- **Liquidity-floor sweep.** Re-run at LIQ_MIN ∈ {$1M, $5M, $10M, $25M,
  $100M} to characterize the knee point where universe expansion stops
  destroying the strategy. Likely between $5M and $25M.
- **Deterministic top-N-by-DV.** Replace the universe definition with
  "top-20 USDC pairs by trailing-90-day median $-volume, refreshed
  monthly." This was already studied in `weekly_breakout_v2.py` (the
  dynamic-universe experiment) and showed marginal improvement vs BC-10
  with the trade-off that constituents drift over time.
- **Entry filter sensitivity on the wide universe.** The 5-day breakout
  may be too tight for alts. Test whether a 20-day breakout filter (which
  was strictly worse than 5-day on BC-10) is *better* than 5-day on the
  wide universe, since the longer window would filter out short-cycle
  pump tops.
- **Composite momentum redesign.** A momentum score that includes a
  reversal penalty (long-term up minus short-term up) might neutralize
  the pump-top selection bias.

## Artifacts

- Script: `scripts/research/v4_full_usdc_universe.py`
- Metrics JSON: `artifacts/research/weekly_breakout_v4/v4_full_usdc.json`
- NAV series: `artifacts/research/weekly_breakout_v4/v4_navs.csv`
