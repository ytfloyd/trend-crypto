# silver_replicator

Research scaffold for fitting a TA-Lib-based systematic model to the silver
trades in a real IBKR account. **This pass is data preparation only — no
modeling code lives here yet.**

## Goal

1. Reproduce the silver-complex trade ledger from an IBKR activity report
   (and, separately, from a Flex Web Service Trades query) so we have both
   period aggregates and per-fill granularity.
2. Build a clean multi-timeframe (1H / 4H / 8H / 1D) OHLCV series for the
   silver front-month series.
3. Pre-compute a wide TA-Lib feature matrix per timeframe to be consumed by
   future modeling work.

## Data sources

- **Activity report CSV** (period aggregates, performance attribution,
  monthly account returns): the user-supplied IBKR Flex-style activity CSV.
- **Flex Web Service Trades query** (per-fill): pulled live via
  `ib_insync.FlexReport` once credentials are available.
- **Stitched silver front-month 1m parquet** (price truth):
  `trend_crypto/artifacts/research/si_quicklook/si_front_month_1m.parquet`.
  This is the canonical front-month series stitched out of
  `data/futures_market.duckdb` (`bars_1m`, `symbol='SI'`).

The silver complex is treated as `COMEX MINY SILVER` (QI* futures) plus
`NYMEX SILVER INDEX` (SO* future-options on full-size SI).

## Layout

```
silver_replicator/
  src/
    ledger.py            # IBKR activity-report -> silver ledger frames
    flex_puller.py       # ib_insync Flex Trades fetch + parse
    bars.py              # front-month load + multi-TF resample
    talib_features.py    # wide TA-Lib feature matrix
  scripts/
    build_ledger.py      # CLI -> artifacts/silver_ledger.parquet etc.
    pull_flex.py         # CLI -> artifacts/flex_trades.parquet
    build_bars.py        # CLI -> artifacts/si_front_month_{tf}.parquet
    build_features.py    # CLI -> artifacts/features_{tf}.parquet
  artifacts/             # parquet/csv outputs (gitignored in practice)
  figures/               # plots produced by future EDA passes
  notebooks/             # ad-hoc exploration (no notebook code committed)
```

## Run order

Activate the project venv first:

```bash
source /Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto/.venv/bin/activate
cd /Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto/research/silver_replicator
```

1. **Ledger from activity-report CSV** (no creds needed):

   ```bash
   python scripts/build_ledger.py \
       --csv /path/to/russell_floyd_October_01_2025_May_27_2026.csv
   ```

   Produces `artifacts/silver_ledger.parquet`,
   `artifacts/silver_perf_by_underlying.csv`,
   `artifacts/silver_monthly_account_returns.csv`.

2. **Multi-timeframe bars**:

   ```bash
   python scripts/build_bars.py --start 2025-10-01 --end 2026-05-05
   ```

   Produces `artifacts/si_front_month_{1H,4H,8H,1D}.parquet`.

3. **TA-Lib features**:

   ```bash
   python scripts/build_features.py
   ```

   Produces `artifacts/features_{1H,4H,8H,1D}.parquet`.

4. **Flex pull (per-fill)** — requires env vars:

   ```bash
   export IBKR_FLEX_QUERY_ID=...
   export IBKR_FLEX_TOKEN=...
   python scripts/pull_flex.py
   ```

   The script exits with a clear message if either variable is missing.

## Required environment variables

| Variable              | Used by                | Notes                                      |
| --------------------- | ---------------------- | ------------------------------------------ |
| `IBKR_FLEX_QUERY_ID`  | `scripts/pull_flex.py` | Integer query id of a Trades Flex query.   |
| `IBKR_FLEX_TOKEN`     | `scripts/pull_flex.py` | TWS → Settings → Reporting → Flex Web Service → Configure → Generate Token. |

`ib_insync` must also be installed in the venv before running the Flex
puller (`pip install ib_insync`). It is imported lazily, so the rest of
the pipeline runs without it.

## Date caveats

- The price datalake (`bars_1m`, stitched front-month parquet) currently
  ends **2026-05-05**.
- The IBKR activity report covers **2025-10-01 → 2026-05-27**.
- There is a **22-day gap (2026-05-06 → 2026-05-27)** where we have
  trade-level ledger rows but no price bars in the lake. Any modeling work
  that joins fills against bars must either (a) refresh the lake forward
  to 2026-05-27, or (b) explicitly restrict to fills on or before
  2026-05-05.
- `silver_monthly_account_returns.csv` is **account-level** (all
  strategies), not silver-specific. Use `silver_perf_by_underlying.csv`
  for silver-isolated P&L.
