# Volbook Discretionary Screen Spec

Volbook is a discretionary technical screen, not a systematic strategy. The output is a ranked watchlist for human review. Sizing, execution, and final trade selection live with the trader.

Good behavior means the screen surfaces technically interesting, liquid, readable setups without burying them in duplicate curve rows or cross-strategy noise. The screen does not claim measured edge, hit rate, or production sizing approval.

## Scope And Assumptions

- Assumption: this sleeve is separate from NRT's long-convexity work; trend, breakout, and mean-reversion are all allowed when their regime gate admits them.
- Assumption: the current liquidity floors are one-contract readability filters, not a portfolio capacity model.
- Assumption: `Setup Grade` is a fixed setup-strength tier derived from the existing setup heuristic. It is not a probability, hit rate, or empirical confidence estimate.
- Assumption: the dashboard is a watchlist-generation tool. A trader may override any row after reviewing contract, chart, volume, curve position, and market context.

## Output Design

The screen is filter-then-group:

1. Apply upstream regime gating before signal generation.
2. Apply hard liquidity floors before output.
3. Apply direction conflict handling.
4. Collapse duplicate curve-month rows into one view by `(underlying x direction x strategy x timeframe)`.
5. Group valid setups by `Strategy Class`.
6. Within each group, sort by `Setup Grade` descending, then `Avg Vol20` descending.
7. Cap each group at 10 rows, with a 30-row total output cap.

The deprecated single cross-strategy ranking is intentionally removed. Trend continuation, range breakout, and mean reversion are different setup types and should not be forced into one universal score.

## Regime Gate

Design: classify each instrument/timeframe before evaluating setups. Ineligible strategy classes are never emitted.

Parameters:

- `ADX(14) <= 20`: mean-reversion eligible.
- `ADX(14) >= 25`: trend continuation and range breakout eligible.
- `20 < ADX(14) < 25`: no-trade transition band.

Expected behavioral impact: fewer direct trend-versus-fade conflicts and cleaner strategy sections.

Review guardrails:

- If a one-time sanity check shows most instruments trapped in the no-trade band, widen or move the thresholds.
- If regime state flips frequently day-over-day, add hysteresis or consecutive-bar confirmation before relying on the gate.

## Setup Grade

Design: replace probability-like `Conf` or `Quality` naming with a fixed `Setup Grade` tier.

Current tiering:

- `A`: existing setup strength is at least `0.75`.
- `B`: existing setup strength is at least `0.60`.
- `C`: all other emitted setups.

The grade is a human-readable sorting aid only. It reflects setup-type strength and confluence, not measured future performance.

## Liquidity Floor

Design: remove illiquid setups before grouping, rather than gently penalizing them.

Parameters:

- Daily floor: `Avg Vol20 >= 10`, `Med Vol20 >= 3`, `Nonzero Vol20 >= 25%`, `Stale Bars <= 35%`.
- Hourly floor: `Avg Vol20 >= 2`, `Med Vol20 >= 1`, `Nonzero Vol20 >= 12%`, `Stale Bars <= 35%`.
- Additional screen floor: rows must pass the current one-contract tradability flag and `Nonzero Ratio >= 50%`.

Expected behavioral impact: fewer thin deferred contracts and fewer charts that look technically clean only because prints are sparse.

## Curve Collapse

Design: the watchlist unit is the view, not the contract month. Duplicate curve points are collapsed by `(underlying x direction x strategy x timeframe)`.

Default vehicle selection:

- Choose the row with highest `Avg Vol20`.
- Tie-break with `Med Vol20`.
- Tie-break with the former composite score only if liquidity is tied.

Visible field:

- `View Contracts` shows how many curve points were collapsed into the visible row.

Expected behavioral impact: one HG long view appears as one row, not many rows across the strip.

## Direction Conflict Rule

Design: keep conflicts visible but demote them.

Current rule:

- Conflict key: `(underlying, strategy class, timeframe)`.
- If both long and short survive for the same key, apply a one-grade demotion and flag `direction conflict`.

Expected behavioral impact: conflicted rows are still visible for trader review, but are less likely to crowd out cleaner signals.

## Data Quality Flags

Design: hard filters remove obviously bad rows, but questionable data that survives should be visible to the trader.

Current flags:

- `ok`: no watch flag.
- `missing bars`: missing price history.
- `missing volume`: volume fields are not usable.
- `stale watch`: stale OHLC bars are elevated but below the hard exclusion threshold.
- `sparse watch`: nonzero volume is elevated but still passed the floor.
- `direction conflict`: row survived but was demoted by conflict handling.

## Score Transparency

The dashboard keeps component fields visible:

- `Setup Grade`: discretionary setup-strength tier.
- `RR`: displayed trade geometry.
- `Liq`: liquidity component from recent volume.
- `Data Q Score`: data quality component from stale/nonzero bars.
- `Curve`: curve-position component.
- `Tech Score` and `Former Composite`: retained as diagnostic fields only, not as the primary sort.

These fields help the trader understand why a row appears and override sensibly.

## Changelog Requirement

Every change to scoring logic, filter thresholds, regime gate parameters, or strategy class definitions must be recorded in `docs/volbook_changelog.md` with:

- date,
- one-line change description,
- rationale,
- expected behavioral impact,
- rollback note if relevant.
