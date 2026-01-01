# Trend Crypto Backtest

Single-asset (BTC-USD) hourly backtesting pipeline with Polars and DuckDB. The engine enforces decision-at-close and execution-at-next-open timing, with reproducible artifacts for every run.

Incubation deployment: see `deployments/v2_5_incubation/DEPLOYMENT.md`.

## Setup

```bash
python -m venv venv_trend_crypto
source venv_trend_crypto/bin/activate
pip install -e .
# or, for tests/linting:
# pip install -e .[dev]
```

## Run a backtest

```bash
python scripts/run_backtest.py --config configs/runs/btc_hourly_ma_vol_target.yaml
```

Run buy & hold baseline:

```bash
python scripts/run_backtest.py --config configs/runs/btc_hourly_buy_and_hold.yaml
```

Compare two runs:

```bash
python scripts/compare_runs.py --run_a <strategy_run_dir> --run_b <bh_run_dir> --out artifacts/compare/btc_hourly
```

Equal-risk BTC+ETH portfolio from existing runs:

```bash
python scripts/build_equal_risk_portfolio.py \
  --run_btc artifacts/runs/<btc_run_dir> \
  --run_eth artifacts/runs/<eth_run_dir> \
  --out artifacts/compare/portfolio_btc_eth_equal_risk
```

Diagnostics:

```bash
python scripts/diagnose_sleeve_correlation.py \
  --run_btc artifacts/runs/<btc_run_dir> \
  --run_eth artifacts/runs/<eth_run_dir> \
  --crisis_quantile 0.2
```

Strategy diagnostics vs benchmark:

```bash
python scripts/diagnose_strategy.py \
  --run_strategy artifacts/runs/v12_daily_portfolio_20251223 \
  --run_benchmark artifacts/runs/btc_daily_buy_and_hold_20251223T164253Z \
  --out_dir artifacts/diagnostics/v12_vs_btc \
  --name_strategy "V1.2 Daily (Net)" \
  --name_benchmark "BTC Buy & Hold" \
  --top_weeks 10 \
  --target_vol_annual 0.80
```

Timeframe sensitivity (1h/4h/1d) with slippage sweep:

```bash
python scripts/run_timeframe_sensitivity.py \
  --timeframes 1h,4h,1d \
  --slippage_grid_bps 1,3,5,10,15,20 \
  --fee_bps 10.0 \
  --base_config_btc configs/runs/btc_hourly_ma_vol_target.yaml \
  --base_config_eth configs/runs/eth_hourly_ma_vol_target.yaml \
  --out_csv artifacts/compare/timeframe_sensitivity.csv
```

Look-ahead lag diagnostic:

```bash
python scripts/check_lookahead_lag.py --config configs/runs/btc_hourly_ma_vol_target.yaml --lags 1,2
```

## Research (midcap momentum)

Create/refresh deduped daily view for midcaps:

    python scripts/research/create_midcap_daily_clean_view.py

Run daily MA 5/40 batch across midcaps:

    python scripts/research/run_midcap_momentum_v0.py --config configs/research/midcap_daily_ma_5_40_v0.yaml

Usage:
- Create/refresh view: `python scripts/research/create_midcap_daily_clean_view.py`
- Run batch: `python scripts/research/run_midcap_momentum_v0.py --config configs/research/midcap_daily_ma_5_40_v0.yaml [--start ... --end ...]`

Validation (should return 0 rows):

    SELECT symbol, ts, COUNT(*) AS n
    FROM bars_1d_midcap_clean
    GROUP BY 1,2
    HAVING COUNT(*) > 1;

Outputs:
- Backtest artifacts are written to `artifacts/runs/<run_id>/`
- Per-run files include:
  - `equity.parquet`
  - `positions.parquet`
  - `trades.parquet`
  - `summary.json`

Why this exists:
- The backtest engine enforces strictly unique `ts` values per run.
- Daily bars derived from upstream data may contain duplicate timestamps.
- `bars_1d_midcap_clean` guarantees exactly one row per `(symbol, ts)` for research runs.

Notes:
- Research-only workflow; does not affect V2.5 deployment or production configs.
- No tests were run (not requested).

## Tests

```bash
pytest
```

Install (dev):

```bash
pip install -e ".[dev]"
```

## Notes

- Bars are read from DuckDB (Parquet-backed). The example config expects a table or view containing hourly BTC-USD bars.
- Strategies decide at bar close; orders fill at the next bar open with slippage and fees.
- Strict validation can be enabled to verify temporal integrity and fill timing.

# Days with missing hours (BTCUSD):
#(datetime.date(2025, 10, 25), 19, datetime.datetime(2025, 10, 25, 0, 0), datetime.datetime(2025, 10, 25, 23, 0))
#(datetime.date(2025, 11, 30), 16, datetime.datetime(2025, 11, 30, 0, 0), datetime.datetime(2025, 11, 30, 15, 0))