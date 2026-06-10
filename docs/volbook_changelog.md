# Volbook Changelog

## 2026-04-25

- Change: pivoted Volbook from systematic scoring validation to a discretionary technical watchlist.
  Rationale: the desired output is a human-review screen, not an automated sized strategy.
  Expected behavioral impact: fewer claims about measured edge; more emphasis on readable setup grouping and trader override.

- Change: replaced the single cross-strategy ranking with filter-then-group output by `Strategy Class`, capped at 10 rows per group and 30 rows total.
  Rationale: trend continuation, range breakout, and mean reversion are heterogeneous setup types and should not be forced into one universal rank.
  Expected behavioral impact: the dashboard should be easier to scan and less likely to bury one strategy class behind another.

- Change: renamed `Quality` to `Setup Grade` and removed empirical calibration language.
  Rationale: without a systematic backtest, the grade is a fixed setup-strength tier, not a probability or hit rate.
  Expected behavioral impact: downstream users should be less likely to mistake the field for measured confidence.

- Change: added `Data Quality` flags while retaining hard liquidity floors.
  Rationale: questionable rows that pass the hard screen should be visible rather than silently scored.
  Expected behavioral impact: traders can identify stale, sparse, or conflicted setups before opening the chart.

- Change: retained ADX regime gating, hard liquidity exclusion, curve collapse, and direction conflict demotion.
  Rationale: these make the discretionary screen more readable by reducing incompatible strategy rows, thin contracts, duplicate curve views, and direct long/short conflicts.
  Expected behavioral impact: fewer noisy rows and cleaner grouped watchlist sections.
