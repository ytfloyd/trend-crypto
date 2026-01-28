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

## CI

GitHub Actions runs registry validation and `pytest -q` on pull requests and pushes to `main`.
Optional dependencies (e.g., duckdb/polars) are skipped when missing in CI.

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

## Research (MA 5/40 BTC/ETH baseline)

Run the MA(5/40) BTC/ETH baseline using the portable template config:

```bash
export DUCKDB_PATH=/path/to/market.duckdb
python scripts/research/run_btc_eth_daily_ma_5_40_v0.py \
  --base_config configs/research/btc_eth_daily_ma_5_40_v0.template.yaml \
  --symbols BTC-USD ETH-USD \
  --run_prefix ma_5_40_btc_eth_baseline_v0
```

Notes:
- ADX defaults OFF unless explicitly enabled in config/overrides.

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

## Research (Formulaic Alpha Engine v0)

Run the formulaic alpha factory from `alphas.txt`:

```bash
python scripts/run_alpha_factory.py --db ../data/market.duckdb --table bars_1d_usd_universe_clean_adv10m
```

If the ADV10m view is missing, `run_alpha_factory.py` can fall back to `bars_1d_clean`.
Use `--allow-fallback` or create the view via:
`python scripts/research/create_usd_universe_adv10m_view.py --db <db>`.

Generate an institutional tearsheet for a single alpha:

```bash
python scripts/generate_alpha_tearsheet.py \
  --alphas artifacts/research/formulaic_alphas/alphas_formulaic_v0.parquet \
  --alpha alpha_001 \
  --db ../data/market.duckdb \
  --price_table bars_1d_clean \
  --output artifacts/research/formulaic_alphas/tearsheets/alpha_001
```

## Research (Survivor Protocol — full universe)

One-command run against `bars_1d_clean`:

```bash
python scripts/run_survivor_protocol_universe.py \
  --db /path/to/market.duckdb \
  --price_table bars_1d_clean \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --out_dir artifacts/research/101_alphas/tearsheets_v0
```

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

### Phase 4: Alpha selection & orientation (101_alphas)

Build IC panel (example horizon 1):

```bash
python scripts/research/alphas101_ic_panel_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v0.parquet \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean \
  --horizon 1 \
  --out_csv artifacts/research/101_alphas/alphas101_ic_panel_v0_h1.csv
```

Select and orient alphas (flip negatives unless --no_flip):

```bash
python scripts/research/alphas101_select_v0.py \
  --ic_panel artifacts/research/101_alphas/alphas101_ic_panel_v0_h1.csv \
  --min_tstat 3.0 \
  --min_mean_ic 0.01 \
  --min_n_days 400 \
  --max_alphas 40 \
  --out artifacts/research/101_alphas/alphas101_selected_v0.csv
```

Run ensemble with selection (and optional danger gating):

```bash
python scripts/research/run_101_alphas_ensemble_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v0.parquet \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean \
  --out_dir artifacts/research/101_alphas \
  --target_gross 1.0 \
  --cash_yield_annual 0.04 \
  --alpha_selection_csv artifacts/research/101_alphas/alphas101_selected_v0.csv \
  --regime_csv artifacts/research/101_alphas/alphas101_regimes_v0.csv \
  --regime_mode danger_cash
```

Tear sheet (multi-page PDF using existing artifacts):

```bash
python scripts/research/alphas101_tearsheet_v0.py \
  --research_dir artifacts/research/101_alphas \
  --out_pdf artifacts/research/101_alphas/alphas101_tearsheet_v0.pdf \
  --strategy_note_md docs/research/alphas101_v0_full_usd_legacy_ic_note.md
```

V1 ADV>10M remediation & tear sheet:
- Overview: `docs/research/101_alphas_v1_overview.md`
- Tear sheet: `artifacts/research/101_alphas/alphas101_tearsheet_v1_adv10m.pdf` (use `--strategy_note_md docs/research/alphas101_ic_note_v0.md` if regenerating)
- BTC benchmark overlay (optional):
  ```bash
  python scripts/research/benchmark_btc_hold_v0.py \
    --db ../data/coinbase_daily_121025.duckdb \
    --price_table bars_1d_usd_universe_clean_adv10m \
    --symbol BTC-USD \
    --equity_csv artifacts/research/101_alphas/ensemble_equity_v0.csv \
    --out_csv artifacts/research/101_alphas/benchmark_btc_usd_equity_v0.csv

  python scripts/research/alphas101_tearsheet_v0.py \
    --research_dir artifacts/research/101_alphas \
    --metrics_csv artifacts/research/101_alphas/metrics_101_ensemble_filtered_v1.csv \
    --capacity_csv artifacts/research/101_alphas/capacity_sensitivity_v1.csv \
    --strategy_note_md docs/research/alphas101_ic_note_v0.md \
    --benchmark_equity_csv artifacts/research/101_alphas/benchmark_btc_usd_equity_v0.csv \
    --out_pdf artifacts/research/101_alphas/alphas101_tearsheet_v1_adv10m.pdf
  ```

Symbol-level stats (ADV>10M V1):

```bash
python scripts/research/alphas101_symbol_stats_v1.py \
  --weights artifacts/research/101_alphas/ensemble_weights_v0.parquet \
  --turnover artifacts/research/101_alphas/ensemble_turnover_v0.csv \
  --equity artifacts/research/101_alphas/ensemble_equity_v0.csv \
  --out_symbol artifacts/research/101_alphas/alphas101_symbol_stats_v1_adv10m.csv \
  --out_top artifacts/research/101_alphas/alphas101_symbol_stats_top20_v1_adv10m.csv \
  --top_n 20
```

### Phase 5: V1 remediation (ADV>10M, ghost-filtered IC, standardized turnover)

Turnover definition: two-sided equity turnover = 0.5 * sum_s |w_t - w_{t-1}|.

Build ADV>10M view:

```bash
python scripts/research/create_usd_universe_adv10m_view.py \
  --db ../data/coinbase_daily_121025.duckdb \
  --source_view bars_1d_usd_universe_clean \
  --adv_window 20 \
  --adv_threshold_usd 10000000 \
  --out_view bars_1d_usd_universe_clean_adv10m
```

## Research (Alpha Ensemble V1.5 Growth Sleeve)

- Build Top50 ADV>10M universe views:
  ```
  venv_trend_crypto/bin/python scripts/research/create_usd_universe_top50_adv10m_views_v15.py --db ../data/coinbase_daily_121025.duckdb
  ```
- Run Growth Sleeve backtest (daily v0):
  ```
  venv_trend_crypto/bin/python scripts/research/run_alpha_ensemble_v15_growth_backtest_v0.py --db ../data/coinbase_daily_121025.duckdb --price_table bars_1d_usd_universe_clean_top50_adv10m --start 2023-01-01 --end 2024-12-31 --out_dir artifacts/research/alpha_ensemble_v15_growth --config_name v0
  ```
- Compute metrics (includes Sortino/Calmar/AvgDD/Hit/Expectancy and trade stats):
  ```
  venv_trend_crypto/bin/python scripts/research/alpha_ensemble_v15_growth_metrics_v0.py --equity artifacts/research/alpha_ensemble_v15_growth/growth_equity_v0.csv --trades artifacts/research/alpha_ensemble_v15_growth/growth_trades_v0.parquet --out artifacts/research/alpha_ensemble_v15_growth/metrics_growth_v15_v0.csv
  ```
- Generate ETH benchmark aligned to growth equity:
  ```
  venv_trend_crypto/bin/python scripts/research/benchmark_btc_hold_v0.py --db ../data/coinbase_daily_121025.duckdb --price_table bars_1d_usd_universe_clean_top50_adv10m --symbol ETH-USD --equity_csv artifacts/research/alpha_ensemble_v15_growth/growth_equity_v0.csv --out_csv artifacts/research/alpha_ensemble_v15_growth/benchmark_eth_usd_equity_v0.csv
  ```
- Tear sheet (uses tearsheet_common overlays; renders note markdown):
  ```
  venv_trend_crypto/bin/python scripts/research/alpha_ensemble_v15_growth_tearsheet_v0.py --research_dir artifacts/research/alpha_ensemble_v15_growth --metrics_csv artifacts/research/alpha_ensemble_v15_growth/metrics_growth_v15_v0.csv --benchmark_equity_csv artifacts/research/alpha_ensemble_v15_growth/benchmark_eth_usd_equity_v0.csv --benchmark_label "ETH-USD Buy & Hold" --strategy_note_md docs/research/alpha_ensemble_v15_growth_overview_v0.md --out_pdf artifacts/research/alpha_ensemble_v15_growth/alpha_ensemble_v15_growth_tearsheet_v0.pdf
  ```

Compute alphas on ADV>10M:

```bash
python scripts/research/run_101_alphas_compute_v0.py \
  --db ../data/coinbase_daily_121025.duckdb \
  --table bars_1d_usd_universe_clean_adv10m \
  --start 2023-01-01 \
  --end 2025-01-01 \
  --out artifacts/research/101_alphas/alphas_101_v1_adv10m.parquet \
  --use_adv10m_view
```

IC panel (ghost-filtered; volume>0 & close != prev_close):

```bash
python scripts/research/alphas101_ic_panel_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v1_adv10m.parquet \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean_adv10m \
  --horizon 1 \
  --out_csv artifacts/research/101_alphas/alphas101_ic_panel_v1_adv10m_filtered.csv \
  --filtered_label filtered_v1
```

Selection/orientation:

```bash
python scripts/research/alphas101_select_v0.py \
  --ic_panel artifacts/research/101_alphas/alphas101_ic_panel_v1_adv10m_filtered.csv \
  --min_tstat 3.0 \
  --min_mean_ic 0.01 \
  --min_n_days 400 \
  --max_alphas 40 \
  --out artifacts/research/101_alphas/alphas101_selected_v1_adv10m.csv
```

Regimes (unchanged logic, new parquet):

```bash
python scripts/research/alphas101_regime_labels_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v1_adv10m.parquet \
  --out artifacts/research/101_alphas/alphas101_regimes_v1_adv10m.csv
```

Ensemble (selection + danger->cash gating, ADV>10M):

```bash
python scripts/research/run_101_alphas_ensemble_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v1_adv10m.parquet \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean_adv10m \
  --out_dir artifacts/research/101_alphas \
  --target_gross 1.0 \
  --cash_yield_annual 0.04 \
  --alpha_selection_csv artifacts/research/101_alphas/alphas101_selected_v1_adv10m.csv \
  --regime_csv artifacts/research/101_alphas/alphas101_regimes_v1_adv10m.csv \
  --regime_mode danger_cash
```

Metrics (writes both v0 and V1 filenames):

```bash
python scripts/research/alphas101_metrics_v0.py
```

TCA (10/20/30/40/50 bps, two-sided turnover assumed):

```bash
for bps in 10 20 30 40 50; do
  python scripts/research/alphas101_tca_v0.py \
    --equity artifacts/research/101_alphas/ensemble_equity_v0.csv \
    --turnover artifacts/research/101_alphas/ensemble_turnover_v0.csv \
    --cost_bps ${bps} \
    --out artifacts/research/101_alphas/metrics_101_ensemble_filtered_v1_costs_bps${bps}.csv
done
```

Beta vs BTC (ADV>10M):

```bash
python scripts/research/alphas101_beta_analysis_v0.py \
  --equity artifacts/research/101_alphas/ensemble_equity_v0.csv \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean_adv10m \
  --benchmark_symbol BTC-USD \
  --out artifacts/research/101_alphas/alphas101_beta_vs_btc_v1_adv10m.csv
```

IC decay (filtered V1):

```bash
python scripts/research/alphas101_ic_decay_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v1_adv10m.parquet \
  --alpha_name alpha_008 \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean_adv10m \
  --max_horizon 5 \
  --out_csv artifacts/research/101_alphas/alphas101_ic_decay_filtered_v1.csv \
  --out_png artifacts/research/101_alphas/alphas101_ic_decay_filtered_v1.png
```

Alpha_008 bias check (filtered V1):

```bash
python scripts/research/alphas101_alpha008_bias_check_v0.py \
  --alphas artifacts/research/101_alphas/alphas_101_v1_adv10m.parquet \
  --db ../data/coinbase_daily_121025.duckdb \
  --price_table bars_1d_usd_universe_clean_adv10m \
  --out artifacts/research/101_alphas/alphas101_alpha008_bias_filtered_v1.csv
```

Concentration (ADV>10M ensemble):

```bash
python scripts/research/alphas101_concentration_v0.py \
  --weights artifacts/research/101_alphas/ensemble_weights_v0.parquet \
  --out artifacts/research/101_alphas/alphas101_concentration_summary_v1_adv10m.csv \
  --top_n 10
```

Capacity sensitivity table:

```bash
python scripts/research/alphas101_capacity_sensitivity_v1.py \
  --metrics_dir artifacts/research/101_alphas \
  --base_metrics metrics_101_ensemble_filtered_v1.csv \
  --cost_metrics_glob "metrics_101_ensemble_filtered_v1_costs_bps*.csv" \
  --out capacity_sensitivity_v1.csv
```

Tear sheet (prefers V1 artifacts if present):

```bash
python scripts/research/alphas101_tearsheet_v0.py \
  --research_dir artifacts/research/101_alphas \
  --out_pdf artifacts/research/101_alphas/alphas101_tearsheet_v1_adv10m.pdf \
  --strategy_note_md docs/research/alphas101_ic_note_v0.md \
  --benchmark_equity_csv artifacts/research/101_alphas/benchmark_btc_usd_equity_v1_adv10m.csv \
  --benchmark_label "BTC-USD Buy & Hold"
```

All research tear sheets use `scripts/research/tearsheet_common_v0.py` for equity loading, BTC benchmark overlay, and BTC vs Strategy summary tables.

### Growth Sleeve v1.5 Universe (Top50 by 30D volume, ADV >= $10M)

Build the daily/4H universe views:

```bash
python scripts/research/create_usd_universe_top50_adv10m_views_v15.py \
  --db ../data/coinbase_daily_121025.duckdb
```

Outputs:
- Daily: `bars_1d_usd_universe_clean_top50_adv10m`
- 4H: `bars_4h_usd_universe_clean_top50_adv10m` (if a 4H source exists)

## Research (kuma_trend)

Tear sheet with strategy note:

```bash
python scripts/research/kuma_trend_tearsheet_v0.py \
  --research_dir artifacts/research/kuma_trend \
  --out_pdf artifacts/research/kuma_trend/kuma_trend_tearsheet_v0.pdf \
  --strategy_note_md docs/research/kuma_trend_overview_v0.md \
  --benchmark_equity_csv artifacts/research/kuma_trend/benchmark_btc_usd_equity_v0.csv \
  --benchmark_label "BTC-USD Buy & Hold"
```

## Strategy Registry (research)

For quick visibility into research strategies and their canonical performance:

```bash
cd /Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto
source venv_trend_crypto/bin/activate

# List all strategies with full-period CAGR / Sharpe / MaxDD
venv_trend_crypto/bin/python scripts/research/strategy_registry_v0.py list

# Inspect a specific strategy (e.g., 101_alphas V1 ADV>10M)
venv_trend_crypto/bin/python scripts/research/strategy_registry_v0.py show --id alphas101_v1_adv10m

# (Optional) Re-run the full pipeline for a strategy from the current HEAD
venv_trend_crypto/bin/python scripts/research/strategy_registry_v0.py run --id alphas101_v1_adv10m
```

Registry config: `docs/research/strategy_registry_v0.json`, which defines:
- `metrics_csv` + `metrics_period` (canonical metrics source)
- `equity_csv` / `tearsheet_pdf` (equity curves and tear sheets)
- `run_recipe` (canonical recompute commands)
- `git_tag` (tag capturing the original run, especially for legacy/archived strategies)

## MA Sweep (20–200h)

Parameter robustness check for MA crossover strategies. Sweeps fast MA (20–200 hours) and slow MA (40–400 hours) over full period and subperiods (2021/2022/2023).

### Setup

The sweep runner requires a DuckDB path, provided via `--db` flag or `TREND_CRYPTO_DB` environment variable:

```bash
# Option 1: Use --db flag
python scripts/sweep_ma.py --db ../data/market.duckdb --symbol BTC-USD

# Option 2: Set environment variable (recommended)
export TREND_CRYPTO_DB=../data/market.duckdb
python scripts/sweep_ma.py --symbol BTC-USD
```

**Auto-Discovery**: The sweep runner automatically:
- Selects the appropriate `bars_*` table based on `--timeframe` (e.g., `bars_1h` for `--timeframe 1h`)
- Discovers start/end dates from the data (MIN/MAX of timestamp column)
- Validates required OHLCV columns
- Detects and reports missing funding data gracefully

**Manual Overrides** (optional):
```bash
# Explicit table and date range
python scripts/sweep_ma.py --table bars_1d --start 2020-01-01 --end 2024-01-01
```

### Run Sweep

```bash
# Basic sweep (BTC-USD, 1h bars)
python scripts/sweep_ma.py \
  --db ../data/market.duckdb \
  --symbol BTC-USD \
  --timeframe 1h \
  --fee-bps 10 \
  --slippage-bps 2 \
  --funding-mode none

# Using environment variable (recommended for multiple sweeps)
export TREND_CRYPTO_DB=../data/market.duckdb
python scripts/sweep_ma.py --symbol BTC-USD --timeframe 1h

# Custom parameter ranges
python scripts/sweep_ma.py \
  --symbol BTC-USD \
  --fast-start 30 \
  --fast-end 100 \
  --fast-step 5 \
  --slow-start 60 \
  --slow-end 200 \
  --slow-step 10
```

### Outputs

Results written to `artifacts/sweeps/ma_sweep_{timestamp}/results.csv` with columns:
- `fast_hours`, `slow_hours` — MA window sizes in hours
- `subperiod_name` — `full`, `2021`, `2022`, `2023`
- `sharpe`, `max_drawdown`, `total_return` — core metrics
- `return_mode` — `open_to_close` or `close_to_close_fallback`
- `used_close_to_close_fallback` — True if `open` column missing
- `total_funding_cost`, `funding_cost_as_pct_of_gross` — funding diagnostics (0.0 if funding disabled)

### Selection Guidance

**Favor parameters that survive 2022 (bear) and 2023 (chop), not just 2021.** 

Look for:
- Consistent positive Sharpe across all subperiods
- Drawdowns < 30% in 2022/2023
- Stable entry/exit frequency (avoid over-trading)

### Troubleshooting

**Problem**: "Multiple bars_* tables found" error

**Solution**: The database contains multiple timeframes. Specify which to use:
```bash
# For hourly data
python scripts/sweep_ma.py --table bars_1h --timeframe 1h

# For daily data
python scripts/sweep_ma.py --table bars_1d --timeframe 1d
```

**Problem**: "Expected timeframe=1h → table=bars_1h, but it does not exist"

**Solution**: Either load hourly bars into DuckDB, or run sweep on daily timeframe:
```bash
# Run daily sweep instead
python scripts/sweep_ma.py --timeframe 1d --fast-start 2 --fast-end 20 --fast-step 1 --slow-start 5 --slow-end 50 --slow-step 2
```

**Problem**: "funding_mode=column but column 'funding_rate' not found"

**Solution**: Funding data not available. Switch to `--funding-mode none`:
```bash
python scripts/sweep_ma.py --funding-mode none
```

## Tests

```bash
pytest
```

Install (dev):

```bash
pip install -e ".[dev]"
```

## Timing & Returns

The backtest engine enforces clean-room execution timing to prevent lookahead bias:

- **Signals** are decided at `Close(t)` using only data available through that bar.
- **Trades** execute at `Open(t + execution_lag_bars)` (default lag = 1 bar).
- **PnL attribution** uses **Model B** (open-to-close returns):
  - When `open` column exists: `asset_ret[t] = close[t] / open[t] - 1`
  - Position held at `t` earns the intraday return from `open[t]` → `close[t]`
  - This correctly reflects that execution happens at open, not at the prior close.
- **Fallback**: If `open` is missing, the engine falls back to close-to-close returns (`close[t] / close[t-1] - 1`). This is recorded in `summary["used_close_to_close_fallback"]`.
- **No double-shift pitfall**: Positions are lagged exactly once (`target.shift(execution_lag_bars)`), and returns are computed directly without an additional shift.

This model avoids the common "double shift" bug where strategies would incorrectly appear to execute instantly at the decision bar close.

**Return metrics in `summary.json` are explicit**:
- `total_return_decimal`: decimal return (e.g., 1.23 = +123%)
- `total_return_pct`: percent return (decimal * 100)
- `total_return_multiple`: multiple (1 + decimal), e.g., 2.23x
- `total_return`: retained for backward compatibility and equals `total_return_decimal`

## Data Resampling

- `data.timeframe` is the **requested** timeframe; `data.native_timeframe` (optional) is the **native** source cadence.
- Requested timeframe must be an **integer multiple** of native (e.g., 1m→1h, 1m→1d).
- OHLCV aggregation: open=first, high=max, low=min, close=last, volume=sum.
- Funding aggregation: `funding_rate` uses mean; `funding_cost` uses sum.
- Incomplete bucket policy: drop first/last bucket when coverage < `min_bucket_coverage_frac` (default 0.8).
- Manifest records requested/native timeframe, resampling rule, coverage, and drop flags.

## Funding

Perpetual futures funding rates are applied as a per-bar carry cost:

```
funding_costs[t] = position[t] * funding_rate[t]
```

**Convention**: `positive funding_rate` means **longs pay shorts** (Binance/Bybit style).

**Outputs**:
- `equity_df["funding_costs"]` — per-bar funding cost (positive = cost to longs)
- `equity_df["cum_funding_costs"]` — cumulative funding paid/received
- `summary["total_funding_cost"]` — total funding over the backtest period
- `summary["avg_funding_cost_per_bar"]` — average per-bar funding
- `summary["funding_cost_as_pct_of_gross"]` — funding as % of gross PnL (magnitude)
- `summary["funding_convention"]` — always `"positive_means_longs_pay"`

**Data integration guidance**: When adding funding rates from a new venue, verify the sign convention by checking that `cum_funding_costs` behaves as expected (e.g., longs pay when funding is positive during bull runs). If the curve is inverted, the venue may use the opposite convention.

## Notes

- Bars are read from DuckDB (Parquet-backed). The example config expects a table or view containing hourly BTC-USD bars.
- Strategies decide at bar close; orders fill at the next bar open with slippage and fees.
- Strict validation can be enabled to verify temporal integrity and fill timing.

## Release / Tagging

To tag the Model B engine baseline after merge:

```bash
git tag -a engine-v1.0-model-b -m "Engine v1.0: Model B open-to-close timing + funding diagnostics"
git push origin engine-v1.0-model-b
```

# Days with missing hours (BTCUSD):
#(datetime.date(2025, 10, 25), 19, datetime.datetime(2025, 10, 25, 0, 0), datetime.datetime(2025, 10, 25, 23, 0))
#(datetime.date(2025, 11, 30), 16, datetime.datetime(2025, 11, 30, 0, 0), datetime.datetime(2025, 11, 30, 15, 0))