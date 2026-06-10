# CLAUDE.md — Operating contract for trend_crypto

This file tells humans and AI agents how to work in this repo: how to load data, run a
backtest, evaluate an alpha, and the rules that keep results honest. Read it before designing
a strategy or running a backtest.

> **Direction of travel.** The repo is mid-consolidation (see
> [docs/RESEARCH_PIPELINE_REORGANIZATION.md](docs/RESEARCH_PIPELINE_REORGANIZATION.md)). Where
> "Today" and "Target" differ below, **prefer the Target convention for new work** so you
> don't entrench the fragmentation the reorg is removing.

---

## What this repo is

Systematic crypto (and increasingly futures) trend/alpha research and trading. Python 3.12,
Polars + DuckDB for data, Pydantic for config. Two things live here: a **typed production
core** (`src/`) and a **research layer** (`scripts/research/`, `notebooks/`) that designs and
validates alphas before they graduate toward live.

---

## Setup, test, lint

```bash
python -m venv venv_trend_crypto && source venv_trend_crypto/bin/activate
pip install -e ".[dev]"        # core + tests/linting

make check                     # lint + typecheck + test (run before declaring done)
make test                      # python -m pytest -q   (pythonpath=src; tests/ + src/convexity_pipeline/tests)
make lint                      # ruff check .          (line-length 100)
make typecheck                 # mypy src/ --ignore-missing-imports (strict; many modules waived in pyproject)
```

**Running scripts:** modules under `src/` import as top-level packages (e.g. `convexity_pipeline`,
`alpha_pipeline`) because `src/` is on the path. The established convention is:

```bash
PYTHONPATH=src python scripts/research/<area>/<script>.py [args]
```

Research scripts also import shared utilities as `from common.X import ...`
(`common` = `scripts/research/common/`); they put that dir on `sys.path` themselves.

---

## The core stack (`src/`) — the typed path

This is the authoritative load → signal → risk → backtest → report chain. Use it for anything
heading toward production.

| Layer | Entry point | Contract |
|---|---|---|
| Data | `data.portal.DataPortal(cfg).load_bars()` | `DataConfig` → `pl.DataFrame[ts, symbol, OHLCV, (funding)]`, UTC, validated, no incomplete bars |
| Strategy | `strategy.base.TargetWeightStrategy` | `on_bar_close(ctx) -> float`; `ctx.history` is sliced to `decision_ts` (no look-ahead). `PortfolioStrategy` protocol for multi-asset |
| Risk | `risk.risk_manager.RiskManager(cfg).apply(weight, history)` | vol targeting + leverage/concentration limits |
| Backtest | `backtest.engine.BacktestEngine(cfg, strategy, risk, portal).run()` | **Model B: decide@close[t], fill@open[t+1]**; returns `(Portfolio, summary)` |
| Config | `common.config`: `RunConfigRaw` → `compile_config()` → `RunConfigResolved` | YAML in `configs/`; resolved config carries a hash + manifest |
| Report | `analysis.tearsheet.generate_tearsheet(df, out)` | quantile/IC tearsheet (PDF + JSON) |
| Notebook helpers | `research.api`: `quick_backtest()`, `quick_sweep()` | fastest path for exploratory work |

Configs live in `configs/runs/` (production) and `configs/research/`. Naming:
`{symbol}_{timeframe}_{params}_{version}.yaml`.

---

## The research layer (`scripts/research/`)

**Today** there is a *second*, simpler stack used by most research scripts:

- `common.data.load_daily_bars()` / `load_bars()` — wide-panel OHLCV from the DuckDB lake (`ANN_FACTOR = 365`, crypto trades daily)
- `common.backtest.simple_backtest(weights, returns, cost_bps=20.0, execution_lag=1)`
- `common.metrics.compute_metrics()`, `common.risk_overlays.{apply_vol_targeting, apply_dd_control, apply_position_limit_wide}`
- `tearsheet_common_v0.build_standard_html_tearsheet()` for HTML tearsheets

> **Target:** this layer collapses into `src/core` (one backtest, one metrics module, one cost
> model). When adding research, prefer the core stack; if you must use `common.*`, don't fork
> new copies of backtest/metrics logic.

---

## The three evaluation pipelines (keep separate)

A signal is screened by the pipeline matching its **payoff shape** — they have different input
contracts, backtest models, and gates. Do **not** merge them.

| Pipeline | Module | Use when the signal is… | Backtest model |
|---|---|---|---|
| Cross-sectional | `alpha_pipeline/` | a **rank score** across symbols `(ts, symbol) → float` | inverse-vol L/S quintiles |
| Time-series | `ts_pipeline/` | a **per-asset directional forecast** (+long / −short) | vol-targeted portfolio (Carver) |
| Convexity | `convexity_pipeline/` | an **options/vol or trend-with-stops payoff** | caller-provided via adapter, routed by payoff shape |

Each runs a sequential fast-fail gate stack ending in walk-forward (CPCV + PBO) and a deflated
Sharpe / multiple-testing correction. The convexity pipeline gates on the **Composite
Convexity Score (CCS)**, not Sharpe — see the spec below.

---

## The research loop (how to design an alpha)

This is the loop both humans and agents follow. **Pre-register before you backtest.**

1. **Hypothesis.** Fill [docs/research/alpha_hypothesis_template.md](docs/research/alpha_hypothesis_template.md):
   falsifiable statement, economic rationale (a *mechanism*, not "the indicator works"),
   expected **payoff shape** (this routes the pipeline), signal definition, universe, cost
   assumptions, **pre-registered expected metrics**, and falsification criteria. Save it under
   `docs/research/hypotheses/`.
2. **Register.** Add the alpha to the registry (Today: `data/alpha_registry/alpha_registry_v2.xlsx`
   + in-code `Hypothesis`/`Candidate` objects, e.g. in `scripts/research/convexity_alpha_runner.py`.
   Target: one `registry/alphas/<registry_id>.yaml` per the reorg plan).
3. **Implement the signal** as a *pure, causal function* — backward-looking windows only; the
   engine holds bar-`t`'s decision from `t+1` (`execution_lag=1`). No future data in `ctx.history`.
4. **Run it through its pipeline** (e.g. `PYTHONPATH=src python scripts/research/convexity_alpha_runner.py --calibrate`).
   Read the stage scorecard.
5. **Report.** Produce a validation report following the worked examples in
   `docs/research/ma_5_40_*_validation_report.md` (setup, parameter sweep, cost sensitivity,
   yearly returns, walk-forward OOS, bootstrap significance, risk decomposition).
6. **Promote** through stages only when gates pass; record provenance (config hash, git tag).

Reference docs:
- Pipeline spec & stage gates: `docs/research/convexity_alpha_pipeline_spec.md`
- IC framework & pre-built alphas: `docs/research/ALPHA_FRAMEWORK_README.md`, `docs/research/alpha_testing_guide.md`
- Worked hypotheses & cohorts: `docs/research/hypotheses/`, `docs/research/cohorts/`

---

## Hard rules (these keep results honest)

1. **Pre-register before backtesting.** If a hypothesis is rewritten *after* seeing results, it
   returns to S0 — that's curve-fitting, and it's visible in git. The spec is the contract.
2. **Costs are not optional.** Every reported result includes a cost assumption. Pre-cost
   numbers must be labeled pre-cost.
3. **No look-ahead.** Signals use only backward-looking windows; `StrategyContext.history` is
   sliced to `decision_ts`; execution is lagged (`execution_lag=1`, Model B). If you compute a
   signal on the full panel, you must justify causality.
4. **One backtest, one metrics module per result.** Don't mix the core engine and
   `simple_backtest` in the same comparison without reconciling cost/annualization conventions —
   they can disagree. State which produced a given Sharpe.
5. **Convexity track gates on CCS, not Sharpe.** Sharpe is reported but not the screen for
   trend/convex alphas.
6. **Provenance on every promoted result:** config hash, git tag, data range, and the registry id.
7. **Archive, don't delete.** Old research scripts move to `research/_archive/` (git-recoverable);
   don't mass-delete.

---

## Where things live

```
src/
  data/ strategy/ risk/ backtest/ common/   ← typed core stack (Model B)
  analysis/ research/                        ← tearsheets + notebook API
  alpha_pipeline/ ts_pipeline/ convexity_pipeline/   ← the three evaluation pipelines
  alphas/                                    ← formulaic-alpha DSL (parse/compile to Polars)
  k2_systematic_macro/ volbook/              ← futures domain sandbox + data infra (not pipelines)
  volatility/ pricing/ portfolio/ live/ execution/ monitoring/ validation/
scripts/research/        ← research scripts; common/ holds shared backtest/data/metrics
  common/                  load_daily_bars, simple_backtest, compute_metrics, risk_overlays
configs/runs|research/   ← Pydantic-validated YAML backtest configs
notebooks/alpha/         ← exploratory notebooks 00–09
data/                    ← DuckDB lakes + data/alpha_registry/ (xlsx ledger + run scorecards)
docs/research/           ← hypothesis template, pipeline spec, hypotheses, cohorts, reports
artifacts/               ← run outputs (equity curves, trades, tearsheets)
```

---

## For AI agents specifically

- **To propose an alpha:** write the hypothesis doc + a pure causal `signal_fn`, register it,
  run it through the matching pipeline, then summarize the stage scorecard. You do **not** need
  to write a bespoke `run_*.py` — reuse the runner for that payoff shape.
- **Route by payoff shape:** rank-across-symbols → `alpha_pipeline`; per-asset direction →
  `ts_pipeline`; options/vol or trend-with-stops → `convexity_pipeline`.
- **Always run `make check`** before declaring a code change done; note honestly if tests fail.
- **When unsure which Sharpe/cost convention applies, say so** rather than picking silently —
  the two stacks differ and reconciling them is an open item in the reorg plan.
- **Don't introduce a fourth pipeline or a second backtest engine.** Extend an existing one or
  raise it as a design question.
