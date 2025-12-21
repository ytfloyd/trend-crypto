# Trend Crypto Backtest

Single-asset (BTC-USD) hourly backtesting pipeline with Polars and DuckDB. The engine enforces decision-at-close and execution-at-next-open timing, with reproducible artifacts for every run.

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

Canonical PnL (close-to-close, lagged weights):

```
gross_ret = signal.shift(lag) * close.pct_change()
turnover = abs(signal.shift(lag) - signal.shift(lag+1))
cost_ret = turnover * (fee_bps + slippage_bps)/10000
net_ret = gross_ret - cost_ret
```

Artifacts are written under `artifacts/runs/<run_id>/`.

## Tests

```bash
pytest
```

## Notes

- Bars are read from DuckDB (Parquet-backed). The example config expects a table or view containing hourly BTC-USD bars.
- Strategies decide at bar close; orders fill at the next bar open with slippage and fees.
- Strict validation can be enabled to verify temporal integrity and fill timing.

# Days with missing hours (BTCUSD):
#(datetime.date(2025, 10, 25), 19, datetime.datetime(2025, 10, 25, 0, 0), datetime.datetime(2025, 10, 25, 23, 0))
#(datetime.date(2025, 11, 30), 16, datetime.datetime(2025, 11, 30, 0, 0), datetime.datetime(2025, 11, 30, 15, 0))