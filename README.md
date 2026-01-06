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

## Research (USD universe)

Create/refresh USD spot universe (ex-stablecoin bases) from bars_1d_clean:

    python scripts/research/create_usd_universe_daily_clean_view.py --db ../data/market.duckdb

## Research (101_alphas)

Create/refresh USD universe view:

    python scripts/research/create_usd_universe_daily_clean_view.py --db ../data/market.duckdb

Compute alphas for USD universe:

    python scripts/research/run_101_alphas_compute_v0.py \
      --db ../data/market.duckdb \
      --table bars_1d_usd_universe_clean \
      --start 2017-01-01 \
      --end 2025-01-01 \
      --out artifacts/research/101_alphas/alphas_101_v0.parquet

Run ensemble backtest:

    python scripts/research/run_101_alphas_ensemble_v0.py \
      --alphas artifacts/research/101_alphas/alphas_101_v0.parquet \
      --db ../data/market.duckdb \
      --price_table bars_1d_usd_universe_clean \
      --out_dir artifacts/research/101_alphas

Compute ensemble metrics:

    python scripts/research/alphas101_metrics_v0.py

### Phase 3: Beta / IC decay / capacity checks (101_alphas)

From repo root:

- Beta vs BTC (or any benchmark in `bars_1d_usd_universe_clean`):

```bash
python scripts/research/alphas101_beta_analysis_v0.py \
  --equity artifacts/research/101_alphas/ensemble_equity_v0.csv \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean \
  --benchmark_symbol BTC-USD \
  --out artifacts/research/101_alphas/alphas101_beta_vs_btc_v0.csv
```

- IC decay for alpha_008 (horizons 1–5 days):

```bash
python scripts/research/alphas101_ic_decay_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v0.parquet \
  --alpha_name alpha_008 \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean \
  --max_horizon 5 \
  --out_csv artifacts/research/101_alphas/alphas101_ic_decay_alpha008_v0.csv \
  --out_png artifacts/research/101_alphas/alphas101_ic_decay_alpha008_v0.png
```

- TCA: 10 bps vs 20 bps cost sensitivity:

```bash
python scripts/research/alphas101_tca_v0.py \
  --equity artifacts/research/101_alphas/ensemble_equity_v0.csv \
  --turnover artifacts/research/101_alphas/ensemble_turnover_v0.csv \
  --cost_bps 10 \
  --out artifacts/research/101_alphas/metrics_101_alphas_ensemble_v0_costs_bps10.csv

python scripts/research/alphas101_tca_v0.py \
  --equity artifacts/research/101_alphas/ensemble_equity_v0.csv \
  --turnover artifacts/research/101_alphas/ensemble_turnover_v0.csv \
  --cost_bps 20 \
  --out artifacts/research/101_alphas/metrics_101_alphas_ensemble_v0_costs_bps20.csv
```

### Phase 3: IC panel, regimes, gating (101_alphas)

IC panel (per-alpha cross-sectional IC):

```bash
python scripts/research/alphas101_ic_panel_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v0.parquet \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean \
  --horizon 1 \
  --out_csv artifacts/research/101_alphas/alphas101_ic_panel_v0_h1.csv
```

Regime labels from C-features:

```bash
python scripts/research/alphas101_regime_labels_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v0.parquet \
  --out artifacts/research/101_alphas/alphas101_regimes_v0.csv
```

Ensemble with danger-only gating:

```bash
python scripts/research/run_101_alphas_ensemble_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v0.parquet \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean \
  --out_dir artifacts/research/101_alphas \
  --target_gross 1.0 \
  --cash_yield_annual 0.04 \
  --regime_csv artifacts/research/101_alphas/alphas101_regimes_v0.csv \
  --regime_mode danger_cash
```

TCA stress (10/20/30/50 bps):

```bash
for bps in 10 20 30 50; do
  python scripts/research/alphas101_tca_v0.py \
    --equity artifacts/research/101_alphas/ensemble_equity_v0.csv \
    --turnover artifacts/research/101_alphas/ensemble_turnover_v0.csv \
    --cost_bps ${bps} \
    --out artifacts/research/101_alphas/metrics_101_alphas_ensemble_v0_costs_bps${bps}.csv
done
```

Beta vs BTC:

```bash
python scripts/research/alphas101_beta_analysis_v0.py \
  --equity artifacts/research/101_alphas/ensemble_equity_v0.csv \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean \
  --benchmark_symbol BTC-USD \
  --out artifacts/research/101_alphas/alphas101_beta_vs_btc_v0.csv
```

IC decay for alpha_008:

```bash
python scripts/research/alphas101_ic_decay_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v0.parquet \
  --alpha_name alpha_008 \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean \
  --max_horizon 5 \
  --out_csv artifacts/research/101_alphas/alphas101_ic_decay_alpha008_v0.csv \
  --out_png artifacts/research/101_alphas/alphas101_ic_decay_alpha008_v0.png
```

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