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

Canonical PnL (close-to-close, lagged weights):

```
gross_ret = signal.shift(lag) * close.pct_change()
turnover = abs(signal.shift(lag) - signal.shift(lag+1))
cost_ret = turnover * (fee_bps + slippage_bps)/10000
net_ret = gross_ret - cost_ret
```

Incubation deployment:
- See `deployments/v2_5_incubation/DEPLOYMENT.md` for pinned tag, configs, and operational checklist.

Combined 50/50 BTC/ETH portfolio and tear sheet:

```bash
python scripts/build_combined_portfolio_50_50.py \
  --run_a artifacts/runs/<btc_run_dir> \
  --run_b artifacts/runs/<eth_run_dir> \
  --out_dir artifacts/compare/combined_example

python scripts/generate_tearsheet_pdf.py \
  --run_btc artifacts/runs/<btc_run_dir> \
  --run_eth artifacts/runs/<eth_run_dir> \
  --combined_dir artifacts/compare/combined_example \
  --out_pdf artifacts/compare/combined_example/tearsheet.pdf \
  --benchmark_btc_bh artifacts/runs/<btc_bh_run_dir> \
  --rf_apy 0.04 \
  --roll_corr_days 90
```

Artifacts are written under `artifacts/runs/<run_id>/`.

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