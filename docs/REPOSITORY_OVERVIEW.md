# Repository Overview — `trend_crypto`

A practitioner's map of what this repository is, how it is organized, how to use it, and an honest
assessment of its strengths and weaknesses **as a trading-research pipeline**. For the day-to-day
operating rules (load data, run a backtest, the honesty rules) see [`CLAUDE.md`](../CLAUDE.md); this
document is the higher-level architecture + evaluation.

*Last reviewed: 2026-06-14.*

---

## 1. What it does

`trend_crypto` is a **systematic trading research platform** for digital assets (and, increasingly,
futures/equities). Its job is to take a trading idea from hypothesis → causal signal → honest
backtest → multi-stage statistical validation → a registered, reproducible result that can graduate
toward live trading. Python 3.12; Polars + DuckDB for data; Pydantic for typed config.

It contains two layers that should not be confused:

- **A typed production core (`src/`)** — the authoritative load → signal → risk → backtest → report
  chain, plus three formal evaluation pipelines and a machine-readable alpha registry.
- **A research layer (`scripts/research/`, `notebooks/`)** — where alphas are designed and stress-
  tested before they graduate. Faster-moving, less typed, larger surface area.

The design intent (per `CLAUDE.md` and `docs/RESEARCH_PIPELINE_REORGANIZATION.md`) is that research
graduates *into* the core, and that the core enforces the discipline (no look-ahead, costs-on,
pre-registration) that keeps results honest.

---

## 2. High-level architecture

The core chain (the "typed path"):

```
DataPortal ──► TargetWeightStrategy ──► RiskManager ──► BacktestEngine ──► tearsheet/registry
 (load_bars)    (on_bar_close, causal)   (vol target,     (Model B:          (PDF/JSON, provenance)
                                          limits)          decide@close[t],
                                                           fill@open[t+1])
```

Around that chain sit:

- **Three evaluation pipelines**, routed by a signal's *payoff shape* (see §5).
- **A shared validation toolkit** (`src/afml/`) — purged/combinatorial cross-validation, the
  Deflated/Probabilistic Sharpe Ratio, and Probability-of-Backtest-Overfitting (PBO).
- **A machine-readable registry** (`registry/alphas/*.yaml`) — the single source of truth for each
  alpha's hypothesis, routing, provenance, and realized validation metrics.
- **DuckDB data lakes** (§4) with a survivorship-free, point-in-time universe convention.

---

## 3. Directory map

| Path | Role |
|---|---|
| `src/core/` | Unified data loader, metrics, backtest, **registry schema** (the consolidation target) |
| `src/data/ strategy/ risk/ backtest/ common/` | Typed core stack (Model B: decide@close, fill@next-open) |
| `src/pipelines/{cross_sectional,time_series,convexity,common}/` | **New** unified pipelines + shared gate code |
| `src/alpha_pipeline/ ts_pipeline/ convexity_pipeline/` | **Legacy** pipeline locations (pre-reorg; still present — see §9) |
| `src/afml/` | Backtest-overfitting statistics: DSR/PSR, expected-max-Sharpe, PBO, CPCV |
| `src/signals/` | Signal libraries wired for registry-driven execution |
| `src/alphas/` | Formulaic-alpha DSL (parse/compile to Polars) |
| `src/analysis/ research/` | Tearsheets + notebook API (`quick_backtest`, `quick_sweep`), parameter optimizer w/ DSR |
| `src/risk/ portfolio/ execution/ live/ monitoring/ validation/` | Risk, sizing, execution, live, monitoring, CV |
| `src/k2_systematic_macro/ volbook/ domains/` | Futures domain sandbox + data infra |
| `scripts/research/` | **312 research scripts** — sleeves (`k2_atlas/`, `spot_convexity/`), `common/` utilities, one-offs |
| `registry/alphas/` | Per-alpha YAML specs (3 registered: continuation-index, ma-5-40-trend, medallion-lite) |
| `configs/runs/ research/` | Pydantic-validated YAML backtest configs |
| `docs/research/` | 66 docs: hypothesis template, pipeline spec, sleeve writeups, research log, whitepaper |
| `data/` (and `../data/`) | DuckDB lakes + `alpha_registry/` ledger |
| `artifacts/` | Run outputs (equity curves, tearsheets, validation manifests) — gitignored |
| `tests/` | 112 test files (CI runs `pytest --cov=src`) |
| `notebooks/alpha/` | Exploratory notebooks 00–09 |

Scale: ~**189** `src/` modules, ~**312** research scripts, ~**112** test files.

---

## 4. Data layer

DuckDB lakes (multi-asset; crypto is the most mature):

| Lake | Size | Contents |
|---|--:|---|
| `coinbase_crypto_ohlcv_lake.duckdb` | 43 GB | Crypto spot 1m/1h/4h/1d + clean/universe/membership tables |
| `futures_market.duckdb` | 6.3 GB | Futures |
| `stocks_market.duckdb` | 278 MB | US equities |
| `etf_market.duckdb` | 85 MB | ETFs |
| `indices_market.duckdb` / `coinbase_daily_*.duckdb` | 19 / 37 MB | Indices, daily snapshots |

**Universe convention (important).** The validated work uses a **point-in-time, survivorship-free**
universe: membership (e.g. top-N by 20-day trailing dollar-ADV) is reconstructed as it was known on
each historical date. Cross-sectional features are ranked **within** the eligible set per date. This
convention is what separates honest results from look-ahead-inflated ones (see §8).

---

## 5. The three evaluation pipelines

A signal is screened by the pipeline matching its **payoff shape** — different input contracts,
backtest models, and gates. They are deliberately kept separate.

| Pipeline | Use when the signal is… | Backtest model | Primary gate |
|---|---|---|---|
| **Cross-sectional** (`pipelines/cross_sectional`, legacy `alpha_pipeline`) | a rank score across symbols | inverse-vol L/S quantiles | IC / deflated Sharpe |
| **Time-series** (`pipelines/time_series`, legacy `ts_pipeline`) | a per-asset directional forecast | vol-targeted portfolio (Carver) | deflated Sharpe |
| **Convexity** (`pipelines/convexity`, legacy `convexity_pipeline`) | options/vol or trend-with-stops | caller-provided adapter | **Composite Convexity Score** (not Sharpe) |

Each runs a sequential fast-fail gate stack ending in walk-forward (CPCV + PBO) and a deflated-Sharpe
multiple-testing correction (`src/afml/`).

---

## 6. The research workflow

1. **Hypothesis** — fill the falsifiable template (`docs/research/alpha_hypothesis_template.md`):
   mechanism, payoff shape (routes the pipeline), universe, **pre-registered expected metrics**,
   falsification criteria.
2. **Register** — add the alpha to `registry/alphas/<id>.yaml` (validated by `core.registry`).
3. **Implement** the signal as a *pure, causal* function (backward-looking windows only;
   `execution_lag=1`; no future data in `ctx.history`).
4. **Run it through its pipeline** → read the stage scorecard.
5. **Validate** — walk-forward OOS, DSR/PSR, PBO, and (the standard set this session) a
   **pre-registered, deterministic audit** that emits a provenance manifest.
6. **Promote** through stages only when gates pass; record provenance (config hash, git tag,
   data range, registry id) in the registry's `validation` block.

**Worked examples** (this is the best way to learn the platform): the `medallion_lite` sleeve
(`scripts/research/k2_atlas/`, `docs/research/medallion_lite_whitepaper.md`) and the `spot_convexity`
sleeve (`scripts/research/spot_convexity/`) — both carry their full design → labels → experiments →
validation → cost analysis trail and a `RESEARCH_LOG.md`.

---

## 7. How to use it

```bash
python -m venv venv_trend_crypto && source venv_trend_crypto/bin/activate
pip install -e ".[dev]"

make check        # lint + typecheck + test (run before declaring done)
make test         # pytest -q (pythonpath=src)
make lint         # ruff check . (line-length 100)
make typecheck    # mypy src/ --ignore-missing-imports
```

- **Run a research script:** `PYTHONPATH=src python scripts/research/<area>/<script>.py`
  (research scripts also put `scripts/research/common/` on the path and import `from common.X`).
- **Fastest exploratory path:** `src/research/api` → `quick_backtest()`, `quick_sweep()`.
- **Registry-driven execution:** validated alphas execute from their registry spec.
- **CI gates (must pass to merge):** `mypy src/`, `ruff check src/ tests/`, registry validate,
  `pytest --cov=src`. Merges are **rebase-only**; branch protection requires up-to-date + green.

---

## 8. Strengths

1. **Honesty is engineered in, not aspirational.** Survivorship-free point-in-time universes,
   `execution_lag=1` (decide@close / fill@next-open), pre-registration before backtesting, and a
   "beat the transparent baseline OOS or investigate" rule are *enforced conventions*, not slogans.
2. **A real multiple-testing toolkit.** `src/afml/` provides DSR/PSR, expected-max-Sharpe haircut,
   and PBO/CSCV — and recent sleeves actually *use* them to kill false positives (e.g. a 100-factor
   zoo and a bagged ensemble were both rejected on PBO grounds, not vibes).
3. **Reproducibility + provenance.** Deterministic harnesses emit JSON manifests (git commit,
   package versions, data fingerprint, gates, results) plus return CSVs for independent
   re-derivation; the registry stores realized metrics with provenance.
4. **Payoff-shape routing.** Forcing each signal through the pipeline that matches its payoff
   (rank / directional / convex) prevents the classic error of judging a trend-with-stops book by a
   cross-sectional Sharpe.
5. **Typed core + Pydantic configs + a registry as single source of truth** make the "graduation"
   path auditable.
6. **Multi-asset data** is in place (crypto mature; futures/equities/ETF lakes present).

---

## 9. Weaknesses, risks & technical debt (honest)

1. **Mid-consolidation fragmentation.** Old and new pipeline locations coexist
   (`src/alpha_pipeline` vs `src/pipelines/cross_sectional`, etc.), and there are historically **two
   backtest stacks** (the typed `backtest.engine` Model B vs the research `common.simple_backtest`)
   and overlapping metrics modules. They can disagree on cost/annualization conventions; results must
   state which produced a given number. Finishing `docs/RESEARCH_PIPELINE_REORGANIZATION.md` is the
   single highest-leverage cleanup.
2. **Research-layer sprawl.** ~312 scripts under `scripts/research/`; many are one-offs pending
   archive. Discoverability and "which script is canonical" suffer. (Tracked: "archive the one-off
   research scripts.")
3. **Cost realism has been applied late, historically.** The medallion sleeve reached a validated
   ~2.84 Sortino headline and only then discovered the edge does not survive realistic
   liquidity-tiered costs (→ ~1.42, sub-$5M capacity). The platform now has the tooling
   (tiered-cost + square-root impact harnesses), but **cost realism is not yet a mandatory *early*
   gate** in the pipeline stack — it should be, especially for small-cap/illiquid universes.
4. **Convention foot-guns.** The backtest charges `tc_bps` as a **round-trip** cost (≈half per
   side); this was briefly mislabeled "one-way" in docs. Such convention ambiguities (cost units,
   ANN_FACTOR 365 vs 8760, daily vs hourly) are a recurring source of error and deserve a single
   documented conventions reference.
5. **Validation-vs-deployment gap.** The platform is strong at *idea → validated signal* but thinner
   on *validated signal → live*: portfolio construction (correlation/crowding, portfolio heat,
   capacity) and live execution/monitoring are less battle-tested than the research path. Only **3
   alphas** are formally registered.
6. **Single-researcher provenance gaps.** Stray uncommitted WIP files in the tree (e.g. equity-data
   ingestion) indicate work happening outside the branch/PR discipline; easy to lose or accidentally
   merge.
7. **Heavy local data dependency.** A 43 GB crypto lake (and a 6.3 GB futures lake) live outside the
   repo; runs aren't trivially reproducible without that data, and there's no lightweight fixture
   for CI-level integration tests of the data path.

---

## 10. Maturity assessment as a trading-research pipeline

| Capability | Maturity | Note |
|---|---|---|
| Data ingestion & point-in-time universe | ●●●●○ | crypto mature; multi-asset present, less exercised |
| Causal signal authoring | ●●●●○ | clear contract; DSL + libraries |
| Backtest honesty (look-ahead, survivorship) | ●●●●● | genuinely strong |
| Statistical validation (DSR/PBO/CPCV) | ●●●●○ | real toolkit, actively used |
| Cost & capacity realism | ●●●○○ | tooling exists; not yet an *early/mandatory* gate |
| Reproducibility/provenance | ●●●●○ | manifests + registry; hampered by data-size + WIP-in-tree |
| Portfolio construction | ●●○○○ | framed, not deeply exercised |
| Live execution / monitoring | ●●○○○ | scaffolding present; least mature |
| Codebase coherence | ●●○○○ | mid-consolidation; dual stacks + script sprawl |

**Net:** an unusually *honest* research engine — better than most at not fooling itself — whose main
gaps are (a) finishing the consolidation, (b) promoting cost/capacity realism to an early gate, and
(c) maturing the path from validated signal to live portfolio.

---

## 11. Recommendations / roadmap

1. **Finish the reorg** — collapse legacy pipelines/backtests/metrics into `src/core` + `src/pipelines`;
   one backtest, one metrics module, one cost model. Delete or archive duplicates.
2. **Make cost+capacity an early, mandatory gate** (a `stage_cost_realism`), with liquidity-tiered
   costs and a square-root impact model, *before* the expensive validation stages — so cost-fragile
   ideas die cheaply (the medallion lesson, codified).
3. **Write a single conventions reference** (cost units round-trip vs one-way, annualization,
   bar-frequency, R-multiple semantics) and assert it in tests.
4. **Archive the one-off research scripts**; keep sleeves (`k2_atlas/`, `spot_convexity/`) and
   `common/` as the canonical structure.
5. **Mature portfolio + live**: portfolio heat / correlation-aware sizing, capacity-aware position
   sizing, and a paper-trading harness with realized-vs-planned slippage tracking.
6. **CI data fixtures** — a tiny committed sample lake so the data→backtest path is integration-tested
   without the 43 GB dependency.
7. **Enforce branch/PR discipline** for all work (no long-lived uncommitted WIP in the tree).

---

## Appendix — entry points cheat-sheet

| I want to… | Start here |
|---|---|
| Understand the rules | `CLAUDE.md` |
| See a full validated study | `docs/research/medallion_lite_whitepaper.md` (+ `RESEARCH_LOG.md`) |
| See a clean sleeve being built | `docs/research/spot_convexity/00_sleeve_plan.md` |
| Author an alpha | `docs/research/alpha_hypothesis_template.md` → `registry/alphas/` |
| Validate overfitting | `src/afml/backtest_stats.py`, `src/afml/cross_validation.py` |
| Quick backtest in a notebook | `src/research/api` |
| Reproduce a result | the `reproduce` field in the relevant `artifacts/**/*.json` manifest |
