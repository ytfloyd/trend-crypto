# Trend Crypto — Project Status Breakdown

*March 2026*

## Project Identity

**trend-crypto-backtest** is a crypto trend-following research and backtesting
platform. It supports single-asset and multi-asset backtesting (BTC, ETH,
mid-caps), systematic alpha research, and has a live/paper trading path. The
codebase is Python 3.12, uses Polars + DuckDB for data, Pydantic for config,
and runs CI via GitHub Actions (mypy, ruff, pytest).

### Repository layout

```
src/                  Core library — backtest engine, strategies, risk, data,
                      execution, alphas, pricing, volatility, live trading
scripts/              Entry points (run_backtest, diagnostics, sweeps)
scripts/research/     Research modules — JPM momentum, TSMOM, logreg filter,
                      talib scanner, alpha lab, Sornette, ML4T experiments
configs/              YAML configs — backtest runs and research experiments
notebooks/alpha/      Jupyter notebooks (00–06) for interactive research
tests/                67 pytest files covering the core path
docs/                 Strategy memos, alpha framework docs, validation guides
deployments/          Deployment specs (v2.5 incubation)
artifacts/            Run outputs — equity curves, trades, tearsheets, sweeps
```

---

## 1. What Is Working Well

### Core Backtest Engine — Production-grade, heavily tested

The backtest engine (`src/backtest/engine.py`) implements strict Model B timing
(signal at close, fill at open+1) with realistic cost modeling (fees, slippage,
funding). It has 67 tests covering timing, PnL, costs, rebalancing, impact,
sell constraints, cash buffers, and deadbands. This is the strongest part of
the codebase.

Key files:

| File | Role |
|------|------|
| `src/backtest/engine.py` | Single-asset BacktestEngine |
| `src/backtest/portfolio_engine.py` | Multi-asset PortfolioEngine |
| `src/backtest/portfolio.py` | Cash/position tracking |
| `src/backtest/rebalance.py` | Target-weight rebalancing |
| `src/backtest/impact.py` | Dynamic slippage model |
| `src/backtest/metrics.py` | Performance statistics |

### Data Pipeline — Reliable

- **Coinbase collector** (`src/data/collector.py`) ingests 1m OHLCV into DuckDB.
- **DataPortal** (`src/data/portal.py`) loads and resamples bars (1m → 1h, 4h, 1d).
- Both are tested. Schema is consistent across crypto and ETF databases.

### Risk Framework — Comprehensive

`src/risk/` covers:

- Vol targeting (`vol_target.py`, `risk_manager.py`)
- Kelly criterion (`kelly.py`)
- VaR — historical + parametric (`var.py`)
- Stress testing (`stress.py`)
- Factor attribution (`attribution.py`)
- Correlation regime detection (`correlation.py`, `regime.py`)
- Carver-style position sizing (`position_sizing.py`, `handcraft.py`, `diversification.py`)

Tested via `test_risk_framework.py`, `test_kelly.py`, `test_vol_target_scales_down.py`,
`test_portfolio_engine.py`.

### Strategy Layer — Solid

| Strategy | File |
|----------|------|
| MA crossover + vol hysteresis | `src/strategy/ma_cross_vol_hysteresis.py` |
| MA crossover long-only | `src/strategy/ma_crossover_long_only.py` |
| Buy-and-hold | `src/strategy/buy_and_hold.py` |
| Carver-style forecasts (EWMAC, breakout, carry) | `src/strategy/forecast.py` |
| Forecast combiner with FDM | `src/strategy/forecast_combiner.py` |

Tested and cataloged via a strategy registry (`scripts/research/strategy_registry_v0.py`).

### Alpha Factory — Well-architected

`src/alphas/` implements a formulaic alpha DSL:

- **Parser** → AST from string expressions
- **Compiler** → Polars execution plans
- **Primitives** → Time-series and cross-sectional operators
- **Signal processor** → z-score normalization, EMA smoothing, winsorization

Six test files cover parser, two-stage pipeline, warmup, signal processing,
and table resolution.

### Live/Paper Trading Path — Tested

`src/live/runner.py` (LiveRunner), `src/execution/` (PaperBroker, OMS), and
`src/monitoring/` (alerts, reconciliation, dashboard) form a complete
paper-trading loop. Tested in `test_live_trading.py`.

### Research Infrastructure — Productive

- **Common utilities** (`scripts/research/common/`) — shared data loading,
  simple backtest, metrics, Bayesian evaluation, risk overlays, cost analysis.
- **7 research notebooks** (`notebooks/alpha/00` through `06`) — data
  explorer, signal sandbox, turtle trader, AVAX deep-dive, logreg filter,
  Bayesian evaluation, vol estimators.
- **ExperimentTracker** and **ParameterOptimizer** (`src/research/`).

### Documentation — Good coverage for strategies

35 markdown files in `docs/research/` covering Transtrend (v0/v1/v2), JPM
momentum (chapters 1–8), 101 alphas, Kuma trend, alpha framework, purged CV,
and the options/vol program setup guide.

### CI/CD — Working

GitHub Actions runs on every push/PR to `main`:

1. mypy (strict mode)
2. ruff lint
3. Strategy registry validation
4. pytest with coverage

---

## 2. What Is Not Working or Problematic

### Options/Volatility Modules — Zero tests

| Module | Contents | Test coverage |
|--------|----------|---------------|
| `src/pricing/` | Black-Scholes, Black-76, Bachelier | **None** |
| `src/volatility/` | 5 realized vol estimators, VolSurface | **None** |
| `src/data/options/` | IB chain fetcher, vol surface collector | **None** |

The IB integration requires the `ib_insync` optional dependency and has never
been validated against a live or paper TWS connection.

### Notebook 04 — Broken imports

`notebooks/alpha/04_logreg_probability_filter.ipynb` fails with
`ModuleNotFoundError: No module named 'scripts'` because
`jpm_bigdata_ai/helpers.py` uses absolute imports from
`scripts.research.common.data` that don't resolve via the notebook's
`_setup.py` path setup.

### Mypy Overrides — Suppressing rather than fixing

`pyproject.toml` has `ignore_errors = true` for:
`data.*`, `alphas.*`, `analysis.*`, `portfolio.*`, `volatility.*`, `pricing.*`,
`strategy.*`, `risk.*`.

This masks real type errors rather than addressing them. The strict mypy config
loses most of its value when the majority of modules opt out.

### Stale GitHub Actions Run

A zombie "queued" run (ID 22727555100) is permanently stuck and cannot be
cancelled or deleted (GitHub returns HTTP 500). It may block or delay future CI
runs. Monitor and contact GitHub support if it persists.

### Inconsistent Import Conventions

- `test_ma_crossover_adx_default.py` uses `src.common.config` while all other
  tests use `common.config`.
- `src/data/options/snapshot.py` and `src/volatility/surface.py` use absolute
  imports (`from volatility.surface`) after being patched from broken relative
  imports — correct given the `sys.path` setup, but a fragility point.

### Missing Developer Documentation

No architecture overview, no module dependency diagram, no developer setup
guide, no contribution guide. The README covers usage but not internals.

### Stray Files

- `snx_usd_timeseries.png` — untracked PNG in repo root, referenced nowhere.
- `scripts/research/common/_cache/*.parquet` — cache files that should never
  be committed (gitignored via `**/_cache/` but show in some views).

---

## 3. Highest-Leverage Ideas for Pushing Forward

### A. Test the options/vol stack

The pricing and volatility modules are mathematically dense and critical if the
options program is to be trusted. Unit tests for Black-Scholes put/call parity,
Greeks symmetries, IV round-tripping, and vol estimator sanity checks would
take roughly a day and dramatically increase confidence.

### B. Integrate Carver forecasts into the main backtest path

`forecast.py` and `forecast_combiner.py` already implement EWMAC, breakout,
and carry forecasts with proper scaling and combination. These are currently
only used by the Carver position-sizing research path — wiring them into
`BacktestEngine` as a first-class strategy type would unify the single-signal
and multi-signal paths.

### C. Multi-asset portfolio backtesting as the default

`PortfolioEngine` exists and is tested, but most configs and workflows still
target single-asset runs. Shifting the default workflow to multi-asset (with
HRP or handcrafted weights) would better represent the actual portfolio being
traded.

### D. Standardize Bayesian evaluation

The Bayesian toolkit in `scripts/research/common/bayesian.py` (posterior
Sharpe, credible intervals, P(A beats B), Bayes factors) is powerful. Making it
a mandatory step in every strategy evaluation — not just notebook 05 — would
raise the research bar across the board.

### E. Options data pipeline — validate with a real IB connection

The IB integration code exists but has never been tested against a live or
paper TWS connection. A single end-to-end test (connect → fetch chain →
snapshot surface → store in DuckDB → reconstruct VolSurface) would validate the
entire path.

### F. Clean up mypy overrides

Replace blanket `ignore_errors = true` overrides with targeted `type: ignore`
comments or proper type annotations. This would catch real bugs that the strict
mypy config was designed to surface.

### G. Architecture documentation

A one-page diagram showing the data flow and module dependency map would help
onboarding and team communication:

```
Coinbase API → Collector → DuckDB
                              ↓
                          DataPortal → bars (1m/1h/4h/1d)
                              ↓
                          Strategy → target weights / forecasts
                              ↓
                          RiskManager → vol-targeted weights
                              ↓
                          BacktestEngine → fills, PnL, equity curve
                              ↓
                          Artifacts → parquet, JSON, tearsheets
```

---

## 4. Candidates for Removal

### Likely removable

| Item | Reason |
|------|--------|
| `snx_usd_timeseries.png` | Stray chart in repo root, referenced nowhere. Delete and add `*.png` to `.gitignore`. |
| `scripts/research/ml4t_autoencoder/` | ML4T textbook exercise. If not feeding production strategies, it adds maintenance weight. |
| `scripts/research/ml4t_garch/` | Same — pedagogical GARCH experiment. |
| `scripts/research/ml4t_pairs/` | Same — pairs trading experiment from ML4T. |
| `scripts/research/sornette_lppl/` | LPPL bubble detection. Interesting but niche; if not informing position sizing or risk overlays, it is dead weight. |
| `scripts/research/crowding/` | Crowding overlay. Check if used downstream; if not, archive. |

### Review for consolidation

| Item | Action |
|------|--------|
| `scripts/research/paper_strategies/` | If superseded by `jpm_momentum`, `alpha_lab`, or `tsmom`, archive. |
| `scripts/research/paper_pipeline/` | Same — check for overlap with active research modules. |
| `scripts/research/multifreq/` | Multi-frequency momentum. If findings are incorporated into the main strategy, archive the standalone module. |
| `configs/runs/` (24 YAML files) | Many near-duplicates (e.g. `btc_daily_ma_5_40_v2` through `v25`). Consider parameterizing or consolidating. |

### Do not remove

| Item | Reason |
|------|--------|
| All core `src/` modules | Production or actively used |
| `scripts/research/jpm_momentum/` | Active research, complete with runners |
| `scripts/research/logreg_filter/` | Active research, full pipeline |
| `scripts/research/talib_scanner/` | Active research, IC scanning |
| `scripts/research/tsmom/` | Active research, time-series momentum |
| `scripts/research/alpha_lab/` | Active research, forward simulation |
| `scripts/research/common/` | Shared utilities — everything depends on it |
| `scripts/research/etf_data/` | Needed for cross-asset research |

---

## 5. Module Health Summary

```
MODULE                STATUS    NOTES
─────────────────────────────────────────────────────────────────
src/backtest/         STRONG    15+ tests, core path, production-ready
src/data/             STRONG    Collector + portal tested, clean schema
src/risk/             STRONG    Comprehensive, tested
src/strategy/         STRONG    Tested, forecast framework ready
src/alphas/           STRONG    6 test files, DSL well-designed
src/execution/        SOLID     Tested via live trading tests
src/live/             SOLID     Tested
src/monitoring/       SOLID     Tested
src/portfolio/        SOLID     HRP tested
src/research/         SOLID     ExperimentTracker, optimizer tested
src/validation/       SOLID     Purged CV tested
src/common/           SOLID     Shared config/utils, tested
src/utils/            SOLID     DuckDB inspect, tested
src/pricing/          GAP       No tests, used by options research only
src/volatility/       GAP       No tests, used by options research only
src/data/options/     GAP       No tests, requires ib_insync, experimental
```

---

## 6. Research Module Inventory

| Module | Status | Entry point | Notes |
|--------|--------|-------------|-------|
| `jpm_momentum/` | Complete | `run_ch2_*.py`, `run_ch3_*.py`, `run_etf_*.py` | Crypto + ETF, 6 signal types, Ch2–Ch8 |
| `tsmom/` | Complete | Multiple runners | Time-series momentum, ETH sweeps, cross-asset |
| `logreg_filter/` | Complete | `python -m scripts.research.logreg_filter` | Walk-forward logistic regression overlay |
| `talib_scanner/` | Complete | `python -m talib_scanner --phase 1\|2\|all` | IC scan of ~95 TA-Lib features |
| `alpha_lab/` | Complete | Multiple runners | Turtle portfolio, Clenow momentum, forward sim |
| `etf_data/` | Complete | `python -m scripts.research.etf_data.ingest` | Tiingo API → DuckDB, ~64 ETFs |
| `common/` | Complete | Imported by other modules | Data, backtest, metrics, Bayesian, overlays |
| `sornette_lppl/` | Complete | Standalone | LPPL bubble detection — review for removal |
| `ml4t_autoencoder/` | Experiment | Standalone | Pedagogical — review for removal |
| `ml4t_garch/` | Experiment | Standalone | Pedagogical — review for removal |
| `ml4t_pairs/` | Experiment | Standalone | Pedagogical — review for removal |
| `crowding/` | Experiment | Standalone | Crowding overlay — review for removal |
| `paper_strategies/` | Unclear | Multiple runners | May overlap with jpm_momentum/tsmom |
| `paper_pipeline/` | Unclear | Multiple runners | May overlap with alpha_lab |
| `multifreq/` | Unclear | Standalone | Check if findings incorporated elsewhere |

---

## 7. Notebook Inventory

| Notebook | Topic | Status |
|----------|-------|--------|
| `00_data_explorer.ipynb` | OHLCV coverage, universe, ADV | Working |
| `01_signal_sandbox.ipynb` | Prototype signals, quick backtest | Working |
| `02_turtle_trader.ipynb` | Turtle rules on crypto (20/10, 55/20) | Working |
| `03_avax_turtle_deep_dive.ipynb` | AVAX-USD 8h vs 1d mechanics | Working |
| `04_logreg_probability_filter.ipynb` | LogReg probability engine | **Broken** — import error |
| `05_bayesian_strategy_evaluation.ipynb` | Bayesian credible intervals, Bayes factors | Working |
| `06_vol_estimators.ipynb` | Realized vol estimators, vol cone, pricing | Working |

---

## 8. External Dependencies

| Dependency | Type | Used by |
|------------|------|---------|
| Coinbase Advanced Trade API | External API | `src/data/collector.py` |
| Tiingo API | External API | `scripts/research/etf_data/` (requires `TIINGO_API_KEY`) |
| Interactive Brokers TWS/Gateway | External API | `src/data/options/` (requires `ib_insync`) |
| DuckDB (`market.duckdb`) | Local database | Core data path, all research |
| DuckDB (`etf_market.duckdb`) | Local database | ETF research (JPM momentum, cross-asset) |
| TA-Lib | C library | `talib_scanner`, `logreg_filter`, `jpm_bigdata_ai` |
| scikit-learn | Library | `logreg_filter`, `jpm_bigdata_ai`, HRP covariance |
| scipy | Library | HRP, Kelly, vol estimators, pricing |
