# Volbook — Futures OHLCV Tool

The `volbook` section of the repo hosts trading-desk tooling for the
volatility book. The first tool connects to Interactive Brokers, pulls
historical OHLCV for a user-selected futures contract, and regenerates
both a lightweight Cursor canvas and a standalone HTML dashboard.

## Layout

```
src/volbook/
  contracts.py         FuturesSpec + curated alias registry (CL_JUN26, …)
  ibkr_client.py       ib_insync wrapper: connect + reqHistoricalData
  bundle.py            OhlcvBundle / OhlcvSeries + JSON persistence
  indicators.py        Runs every TA-Lib function, grouped by category
  signals.py           Ranked trend / mean-reversion / breakout setups
  canvas_writer.py     Emits volbook-futures-ohlcv.canvas.tsx
  html_writer.py       Emits standalone browser HTML dashboard
  cli.py               argparse entry point
scripts/volbook/
  fetch_futures_ohlcv.py    python -m shim for src/volbook/cli.py
  refresh_core_futures.py   batch refresh for options-underlying/core universes
  backfill_futures_minute.py     trailing-window seed for the 1-min DuckDB lake
  refresh_futures_minute.py      daily forward refresh for ContFuture continuous bars
  walk_dated_futures_minute.py   deep-history backward walk over dated expiries
  refresh_dated_futures_minute.py forward-walk for active dated contracts
  export_lake.py            snapshot the lake and dump partitioned Parquet for outside readers
data/volbook/
  bundle.json          Persistent cache; upsert target (git-ignored)
```

## Prerequisites

- Interactive Brokers TWS or IB Gateway running locally with the API
  enabled (`Configuration → API → Settings → Enable ActiveX and Socket
  Clients`). Default socket port for paper trading is **7497**, live is
  **7496** — pass the right one with `--port`.
- `pip install -e .[options]` to get `ib_insync`.
- `pip install -e .[ta]` (or `brew install ta-lib && pip install TA-Lib`)
  to enable the technical-indicator tables. Without it the CLI still
  fetches data and the canvas renders; the indicator section simply
  shows an install hint.
- A market-data subscription for NYMEX (CL/NG) or whichever exchange
  the contract you're asking for trades on.

## Typical runs

Fetch the default (CL Jun'26, 1-day bars, 1 year):

```bash
python -m scripts.volbook.fetch_futures_ohlcv
```

Add an hourly view of the same contract without losing the daily view:

```bash
python -m scripts.volbook.fetch_futures_ohlcv \
    --alias CL_JUN26 --bar-size "1 hour" --duration "30 D" --append
```

Fetch a different expiry alongside:

```bash
python -m scripts.volbook.fetch_futures_ohlcv --alias CL_JUL26 --append
```

Ad-hoc contract (no alias):

```bash
python -m scripts.volbook.fetch_futures_ohlcv \
    --symbol NG --expiry 202608 --append
```

After each run the standalone browser dashboard at
`artifacts/volbook/volbook-futures-ohlcv.html` and the canvas at
`~/.cursor/projects/<workspace>/canvases/volbook-futures-ohlcv.canvas.tsx`
are rewritten with the full bundle embedded. Prefer the HTML dashboard
for long TA-Lib tables and risk/reward scans because it uses normal
browser scrolling and native disclosure controls. The canvas remains a
compact IDE-side preview.

Refresh the full 42-market options-underlying universe used by the
trend/convexity research screen:

```bash
python -m scripts.volbook.refresh_core_futures --replace --no-canvas
```

Run the same universe faster when the hourly view is not needed:

```bash
python -m scripts.volbook.refresh_core_futures_daily --replace --no-canvas
```

Equivalently, pass `--daily-only` to the main batch refresh command.

## Minute-bar futures lake

The volbook persists 1-minute IBKR futures bars in a DuckDB lake at
`../data/futures_market.duckdb`. Tables:

- `bars_1m(symbol, expiry, ts, o, h, l, c, v, fetched_at)` with primary
  key `(symbol, expiry, ts)`. Continuous front-month bars use
  `expiry='continuous'`; dated bars use `expiry='YYYYMM'`.
- `ingest_state(symbol, expiry, earliest_ts, latest_ts, head_ts, last_run_at, notes)`
  with primary key `(symbol, expiry)`.

### Continuous trailing-window backfill (Option 2)

IBKR rejects `endDateTime` on `ContFuture` historical requests
(error 10339), so the continuous backfill cannot chunk backward through
history. Instead it issues a single trailing-window request per symbol
and writes whatever IB returns:

```bash
python -m scripts.volbook.backfill_futures_minute
```

The default `--duration 30 D` is the practical IB max for 1-minute bars
on a continuous contract. Daily refresh keeps the lake current:

```bash
python -m scripts.volbook.refresh_futures_minute
```

Both scripts pace requests at ~11s to stay under IB's 60-requests-per-
10-minutes small-bar limit. Useful flags: `--universe core-macro`,
`--aliases ES_JUN26 CL_JUN26`, `--lake-path /custom/path.duckdb`,
`--duration "30 D"` (backfill), `--max-days 30` (refresh). Run the
backfill once to seed and then the refresh on a daily cron; over time
the lake naturally accumulates several years of forward history per
product.

### Deep-history dated walk (Option 1)

For history older than IBKR's continuous trailing window, dated
`Future` contracts *do* accept `endDateTime` and can be walked backward
in 30-day chunks. The dated walker enumerates each product's historical
expiries client-side using known **listed-month patterns** in
`MONTH_PATTERN_BY_ROOT` (HMUZ for ES/NQ/ZN, HKNUZ for grains, GJMQVZ
for metals, etc.) — no `reqContractDetails(includeExpired=True)` calls,
which IBKR queues unreliably for popular roots. For each candidate
`(symbol, YYYYMM)` the walker calls `qualifyContracts` to check
existence and then chunks 1-minute bars backward to IB's head
timestamp, storing rows under `expiry='YYYYMM'`:

```bash
python -m scripts.volbook.walk_dated_futures_minute --aliases ES_JUN26 --years 5
```

Useful flags: `--years N` (default 5), `--min-expiry YYYYMM` /
`--max-expiry YYYYMM` (overrides `--years`), `--chunk-days 30`,
`--max-chunks-per-contract N`, `--max-expiries-per-symbol N`,
`--oldest-first` (default newest-first so failures don't block recent
depth), `--skip-active`. Resumable: each run consults `(symbol, expiry)`
ingest state and picks up at the existing `earliest_ts`.

### Daily refresh (continuous + dated)

The dated walker is for *deep-history* backfills; it isn't meant for
daily updates. Two refresh CLIs handle the rolling forward edge:

```bash
# Trailing-window pull for IBKR's `ContFuture` (continuous expiry).
python -m scripts.volbook.refresh_futures_minute

# Forward-walk every active dated `(symbol, expiry)` from latest_ts to now.
python -m scripts.volbook.refresh_dated_futures_minute
```

Run both as a daily pair (different `--client-id` defaults — 29 for
continuous, 31 for dated — so they can run back-to-back without
colliding with the deep walker on 30). The dated refresh:

- Reads every `(symbol, expiry)` row from `ingest_state`.
- Filters to "active" via `--keep-active-days` (default 14): a contract
  is active if its expiry month + grace window is on or after today.
  Dead expiries are fixed history and silently skipped.
- Batch-qualifies the active expiries in one IB round trip via
  `IBHistoricalClient.qualify_dated_futures`.
- For each, walks forward in 30-day chunks from `latest_ts` to `now`,
  upserting into the lake. Quiet mid-walk windows (zero bars) are
  treated as quiet — the walk only stops on a real timeout failure or
  reaching `now`.
- Optionally seeds brand-new upcoming expiries via
  `--max-new-expiries N`: enumerates the next N quarterlies not yet in
  `ingest_state`, batch-qualifies them, and pulls a single trailing
  30-day chunk per qualified contract. Off by default; turn it on once
  per quarter when a new front-month is added at IB.

Both the deep walker and the daily refresh share a single retry helper
on the IB client — `IBHistoricalClient.fetch_dated_minute_bars_with_retry` —
which distinguishes a genuinely empty IB response (Error 162: HMDS
query returned no data, returns instantly) from a stuck request that
ib_insync silently exits with `[]` after the configured timeout. The
helper raises `HistoricalDataTimeout` on the latter so callers can
retry rather than mistake a timeout for the head/tail of data.

Once the dated bars are in place, build a continuous series on demand:

```python
from volbook.datalake import MinuteLake

with MinuteLake() as lake:
    df = lake.stitch_continuous_series(
        "ES",
        roll_days_before_expiry=8,  # ES rolls ~3rd Friday; pull back 8d
    )
```

`stitch_continuous_series` picks, for each timestamp, the contract with
the smallest `expiry` whose roll point (first of expiry month minus
`roll_days_before_expiry` days) hasn't yet been reached. Continuous-
source rows (`expiry='continuous'`) are excluded so the dated walk and
the trailing-window store don't double-count.

For CL research that needs an institutional-style dated-contract
continuous series, use the newer `institutional_continuous_series` path:

```python
from volbook.datalake import MinuteLake

with MinuteLake("../data/futures_market.duckdb") as lake:
    result = lake.institutional_continuous_series(
        "CL",
        adjustment="additive",  # "raw", "additive", or "ratio"
    )

bars = result.bars
schedule = result.schedule
metadata = result.metadata
```

This builds from dated `YYYYMM` rows only. For CL, last trade is modeled
as the third business day before the 25th calendar day of the month
preceding delivery, with the 25th first adjusted to the previous
business day when it falls on a weekend/non-business day. The default
calendar excludes weekends and supports an injectable holiday set; pass
a full exchange holiday calendar before treating holiday-sensitive
rolls as production exact.

Available roll policies are:

- `last_trade_minus_n_business_days`: forced calendar roll a configured
  number of business days before CL last trade.
- `volume_crossover`: roll when next-contract volume exceeds front
  volume for the configured number of consecutive sessions inside the
  pre-expiry window; falls back to the forced date if no trigger appears.
- `volume_crossover_with_calendar_guard`: the preferred default; uses
  the same volume trigger and explicitly records a calendar-guard
  fallback when forced.

The output includes raw active-contract prices or additive/Panama and
ratio-adjusted prices, plus lineage columns such as `active_expiry`,
`roll_date`, `next_expiry`, `roll_gap`, `adjustment_offset`,
`adjustment_factor`, `adjustment_method`, `roll_policy`,
`roll_policy_version`, and `roll_fallback`. Open interest crossover is
not inferred because the current minute lake has no OI column.

### External access via Parquet

DuckDB enforces a single-writer file lock on `futures_market.duckdb`,
so while the walker or daily refresh is connected, other processes
cannot open the lake — even read-only. The exporter snapshots the
lake (point-in-time `cp` of the data file plus its WAL) and dumps
`bars_1m` into a hive-partitioned Parquet dataset that any tool with
a Parquet reader can hit concurrently:

```bash
python -m scripts.volbook.export_lake --include-state
```

Output layout (default `../data/futures_market_parquet/`):

```
futures_market_parquet/
  bars_1m/symbol=ES/expiry=202606/data_0.parquet
  bars_1m/symbol=NQ/expiry=202606/data_0.parquet
  ...
  ingest_state.parquet            # one row per (symbol, expiry); only with --include-state
```

Useful flags: `--symbols ES NQ CL` (filter), `--include-continuous`
(off by default — dated bars are usually what consumers want),
`--no-snapshot` (read the lake in place; only safe when no writer is
up), `--keep-snapshot` (leave `/tmp/futures_lake_snapshot.duckdb`
behind for inspection), `--output-dir /custom/path`. The `bars_1m/`
directory is wiped before each run so partitions for symbols you no
longer export don't linger.

Wire it into the daily refresh by running it after the dated and
continuous refreshes have closed their connections; downstream tools
then read with no Python required:

```python
import polars as pl
df = pl.scan_parquet("../data/futures_market_parquet/bars_1m/**/*.parquet",
                     hive_partitioning=True).collect()
```

```sql
-- DuckDB CLI, Athena, BigQuery external table, etc.
SELECT * FROM read_parquet(
    '../data/futures_market_parquet/bars_1m/**/*.parquet',
    hive_partitioning=1
)
WHERE symbol='ES' AND expiry='202606';
```

## Front/continuous-only HTML refresh

Refresh only one front/continuous contract per product:

```bash
python -m scripts.volbook.refresh_front_futures --replace --no-canvas
```

That uses IBKR's continuous futures contract (`CONTFUT`) for each alias,
so the dashboard has one rolling series per product instead of separate
dated curve points. Add `--daily-only` to this command for the fastest
front-only daily update.

That default universe is sourced from the CME options-underlying screen
and includes the futures roots behind the tradable options markets:
`SR3`, `ES`, `ZN`, `CL`, `NG`, `ZF`, `ZC`, `ZB`, `ZT`, `ZS`, `GC`,
`ZL`, `SDA`, `6E`, `LE`, `ZW`, `HE`, `ZM`, `NQ`, `KE`, `6J`, `GF`,
`SME`, `SI`, `DC`, `HG`, `6A`, `6B`, `RTY`, `HO`, `6C`, `GDK`,
`CSC`, `MET`, `GNF`, `CB`, `6S`, `PL`, `MBT`, `RB`, `DY`, and `PA`.
Use `--universe core-macro` to refresh only the smaller original macro
set, or pass `--aliases` to request a specific subset.

## Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--alias` | — | Shortcut into `KNOWN_FUTURES` (e.g. `CL_JUN26`) |
| `--symbol` | `CL` | Ignored when `--alias` is set |
| `--expiry` | `202606` | IB `lastTradeDateOrContractMonth` |
| `--exchange` | auto | Defaulted per symbol (CL→NYMEX, ES→CME, …) |
| `--bar-size` | `1 day` | Any IB bar size (`1 hour`, `5 mins`, …) |
| `--duration` | `1 Y` | IB duration string (`6 M`, `30 D`, …) |
| `--what-to-show` | `TRADES` | Or `MIDPOINT`, `BID_ASK`, … |
| `--use-rth` | off | Regular trading hours only |
| `--append` | off | Upsert into existing bundle instead of replacing |
| `--no-canvas` | off | Skip canvas regeneration (bundle only) |
| `--html-path` | `artifacts/volbook/volbook-futures-ohlcv.html` | Standalone HTML dashboard destination |
| `--no-html` | off | Skip standalone HTML dashboard regeneration |
| `--no-indicators` | off | Skip TA-Lib indicator computation |
| `--indicator-tail` | `20` | Recent bars of each indicator to persist |
| `--bundle-path` | `data/volbook/bundle.json` | |
| `--canvas-path` | workspace canvases dir | |
| `--host` / `--port` / `--client-id` | 127.0.0.1 / 7497 / 17 | IB socket |

Batch refresh flags in `scripts.volbook.refresh_core_futures`:

| Flag | Default | Notes |
| --- | --- | --- |
| `--universe` | `options-underlyings` | Use `core-macro` for the smaller original desk set |
| `--aliases` | selected universe | Explicit alias subset, e.g. `ES_JUN26 CL_JUN26 GC_JUN26` |
| `--curve-points` | `5` | Active futures expiries discovered per root |
| `--fixed-alias-expiry` | off | Fetch only the alias expiry instead of the curve |
| `--continuous-only` | off | Fetch one IBKR continuous/front futures series per alias instead of dated curve points |
| `--hourly-curve-points` | `5` | Fetch hourly bars only for the front N curve points; daily bars still cover all selected points |
| `--daily-only` | off | Skip hourly bars and fetch only the daily timeframe |
