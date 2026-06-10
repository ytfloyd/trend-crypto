# K2 Systematic Macro First Slice

`k2_systematic_macro` is the first narrow slice of an institutional macro
futures convexity/regime framework. It reuses the existing `volbook` futures
minute lake instead of creating a parallel database.

## What This Builds

- A `volbook` research adapter that loads CL 1-minute bars from either the
  DuckDB lake at `../data/futures_market.duckdb` or the Parquet export at
  `../data/futures_market_parquet/bars_1m/`.
- Session-aware resampling to `1h`, `4h`, and daily bars. The primary CL
  timeframe is `4h`; timestamps are UTC bar-end timestamps, with exchange-local
  `session_date` and `session_start_ts` metadata.
- Canonical CL dataset generation with lineage metadata for source system,
  source type, contract source, roll rule, source expiries, adjustment method,
  and bar coverage.
- A first feature layer: ATR, realized volatility, vol-of-vol, ATR/range
  compression, multi-horizon returns, breakout distance, and trend persistence.
- A markdown diagnostics report for CL data quality and exploratory regime
  state checks before any trading logic.
- Leakage-safe forward research targets on the primary `4h` frame:
  normalized future move, future volatility expansion, `>2 ATR` tail event, and
  conditional tail direction for `4h`, `1d`, `3d`, and `5d` horizons.
- A first unsupervised regime layer using sklearn Gaussian Mixture Models over
  volatility, compression, trend persistence, and range features. If sklearn is
  unavailable at runtime, the engine falls back to a deterministic quantile
  assignment and marks the method in `regime_method`.
- Regime evaluation artifacts: transition matrix, persistence, conditional
  returns, conditional volatility, skew, drawdown, and tail participation.
- A baseline probabilistic volatility-expansion model using logistic regression
  with walk-forward validation and calibration diagnostics. XGBoost/LightGBM are
  detected and included only when already installed.

## Usage

Build from the default discovered `volbook` source. The adapter prefers a
Parquet export when present and otherwise falls back to the DuckDB lake:

```python
from k2_systematic_macro.data import CanonicalCLDatasetBuilder

dataset = CanonicalCLDatasetBuilder().build()
```

Build directly from the DuckDB lake:

```python
from k2_systematic_macro.configs.cl import CLResearchConfig
from k2_systematic_macro.data import CanonicalCLDatasetBuilder

config = CLResearchConfig(source="duckdb", lake_path="../data/futures_market.duckdb")
dataset = CanonicalCLDatasetBuilder(config).build()
```

Build with the institutional CL continuous constructor instead of the legacy
dated-front stitch:

```python
from k2_systematic_macro.configs.cl import CLResearchConfig
from k2_systematic_macro.data import CanonicalCLDatasetBuilder

config = CLResearchConfig(
    source="duckdb",
    lake_path="../data/futures_market.duckdb",
    contract_source="institutional_continuous",
    continuous_adjustment="additive",
    continuous_roll_policy="volume_crossover_with_calendar_guard",
    continuous_roll_window_business_days=10,
    continuous_forced_roll_business_days_before_last_trade=3,
    continuous_volume_crossover_sessions=2,
)
dataset = CanonicalCLDatasetBuilder(config).build()
```

`contract_source="institutional_continuous"` reads dated `YYYYMM` rows from
`volbook`, excludes opaque IBKR `expiry="continuous"` rows, builds a CL roll
schedule, then resamples the adjusted active-contract series. The default roll
policy looks for next-contract volume greater than front-contract volume for
two consecutive sessions inside the pre-expiry window and falls back to a
forced roll three business days before CL last trade. CL last trade is modeled
as the third business day before the 25th calendar day of the month preceding
delivery, after moving that 25th to the prior business day if needed.

The default calendar is weekend-only with an injectable holiday extension point.
It is deterministic and testable, but holiday-sensitive production runs should
inject a CME/NYMEX holiday calendar. The current minute lake has volume but no
open interest, so OI crossover is not faked or used for production rolls.

Write a versioned local research dataset:

```python
builder = CanonicalCLDatasetBuilder()
dataset = builder.build()
output_dir = builder.write(dataset)
```

Create a diagnostics report:

```python
from k2_systematic_macro.research import build_cl_quality_report

report = build_cl_quality_report(dataset)
```

Run the full local CL research artifact pipeline:

```bash
python scripts/run_k2_cl_research.py \
  --source duckdb \
  --contract-source institutional_continuous \
  --continuous-adjustment additive
```

By default this writes reproducible local artifacts under
`artifacts/research/k2_systematic_macro/CL/<dataset_id>/`, including:

- canonical `bars_1h.csv`, `bars_4h.csv`, and `bars_1d.csv`
- feature panels for each timeframe
- the primary `panel_4h.csv` with targets and regime assignments
- `regime_transition_matrix.csv`, `regime_summary.csv`, and
  `regime_persistence.csv`
- `expansion_predictions_<model>.csv`, fold metrics, calibration diagnostics,
  `model_metrics.json`, `metadata.json`, and `report.md`

Use `--output-root` to redirect generated artifacts, or
`--no-optional-boosters` to force the logistic-regression baseline only.

## Local Data Status

The local runtime currently has a DuckDB lake at `../data/futures_market.duckdb`
with `bars_1m` and `ingest_state` tables. No Parquet export was found at
`../data/futures_market_parquet/bars_1m/`, so the default `source="auto"` path
falls back to DuckDB. Override discovery with `VOLBOOK_LAKE_PATH`,
`VOLBOOK_PARQUET_ROOT`, `K2_RESEARCH_ROOT`, or explicit `CLResearchConfig`
values. Tests use synthetic CL bars in a temporary `MinuteLake`, so they do not
require IBKR, network access, or large local data.

## Continuous Construction Choices

- `contract_source="dated_front"` remains available for backwards-compatible
  diagnostics. It uses the older first-calendar-day-minus-N-days convention and
  is unadjusted.
- `contract_source="continuous"` still reads IBKR's opaque continuous rows when
  requested explicitly; those rows are limited by IBKR's continuous-history
  behavior.
- `contract_source="institutional_continuous"` is now preferred for CL research
  when dated contracts exist. It supports raw active-contract output,
  additive/Panama back-adjustment, and ratio adjustment when roll factors are
  positive and stable. Roll gaps, offsets/factors, policy version, and fallback
  flags are preserved in volbook lineage metadata.

## Important Limitations

- The default institutional calendar excludes weekends and explicitly supplied
  holidays only. Exact CME/NYMEX holiday handling remains the next validation
  step before production roll dates.
- Targets, regimes, and probabilistic expansion forecasts are research
  diagnostics only. No trading rules, portfolio construction, or backtests are
  included in this slice.
- Open interest crossover remains scaffolded only at the policy-design level
  because `bars_1m` has no OI column. Volume crossover is the implemented
  liquidity trigger.
- Carry decomposition and richer roll-return diagnostics remain future gates
  before portfolio modeling.
- HMM regimes are not enabled by default because `hmmlearn` is not a project
  dependency. The current first-pass regime engine is GMM plus an explicit
  fallback.

## Next Gates

1. Inject and validate a CME/NYMEX holiday calendar for CL rolls.
2. Compare additive vs ratio adjusted CL features and roll-gap diagnostics.
3. Expand the same data/feature path to HO, RB, NG, GC, HG, ZL, BTC-like MBT,
   ETH-like MET, and broader macro futures.
4. Compare GMM with HMM/change-point baselines and convexity forecast evaluation
   after the data-adjustment gate.
