# Volbook Regime Sanity Check

Date: 2026-04-25

This is a one-time behavioral inspection of the current `data/volbook/bundle.json`. It is not a backtest and does not measure trading edge.

## Assumptions

- Assumption: each bundle series is treated as an instrument/timeframe observation for the regime count.
- Assumption: day-over-day flip rate is approximated as adjacent-bar flip rate within each series, because the bundle mixes daily and hourly timeframes.
- Assumption: the current bundle may contain setups generated before the latest regime-gating code was added, so the filter-stage counts reapply the current regime rule to existing setup rows.

## Regime Mix

| Regime State | Series | Share |
|---|---:|---:|
| Mean reversion | 593 | 41.2% |
| Trend | 317 | 22.0% |
| No-trade transition | 298 | 20.7% |
| Insufficient history | 232 | 16.1% |
| Total | 1,440 | 100.0% |

Finding: the regime split is not obviously pathological. The no-trade band is material but not dominant, and the screen is not forcing 90%+ of the universe into no-trade.

## Regime Flip Rate

- Average adjacent-bar flip rate: `5.36%`.
- Median adjacent-bar flip rate: `5.26%`.

Finding: this is not daily flipping on most series. It implies roughly one state change per 19 adjacent bars at the median. If trader review finds the gate still too twitchy, the next adjustment should be hysteresis around the ADX thresholds rather than removing the gate.

## Filter Stage Counts

| Stage | Setups |
|---|---:|
| Raw setup rows in bundle | 726 |
| Survive current regime gate | 333 |
| Survive liquidity floor | 50 |
| After conflict rule | 50 |
| Conflicted rows flagged | 2 |
| After curve collapse | 18 |

Finding: the hard liquidity floor and curve collapse are doing most of the cleanup. The final valid watchlist is sparse but readable. The current surviving set is all `trend` class after applying current regime and liquidity filters to the existing bundle.

## Curve Collapse Routing

- Collapsed views inspected: `18`.
- Cases where the chosen most-liquid contract had materially worse data quality than another source row: `0`.

Finding: curve collapse did not route to a materially worse data-quality contract in this bundle.

## Pathology Check

No blocker was found. The main watch item is that current filtered output is trend-heavy. That appears to come from the current setup inventory and liquidity floor, not from a pathological regime split. Proposed adjustment if this remains undesirable after trader review: loosen mean-reversion setup trigger thresholds before changing the regime gate.
