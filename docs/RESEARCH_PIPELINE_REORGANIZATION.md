# Research Pipeline Reorganization Plan

**Status:** Proposal / draft
**Author:** Research engineering
**Date:** 2026-06-10
**Goal:** Consolidate `trend_crypto` into a single, legible core research pipeline that both humans and AI agents use to design strategies and find alpha.

---

## 0. TL;DR

This repo is **not** a greenfield. It already contains a sophisticated research operating
system. The problem is **fragmentation**: two parallel core stacks doing the same job, 293
one-off research scripts with no registry, and a strong process-on-paper (hypothesis
templates, stage gates, convexity metrics) that is not wired to code.

The fix is **consolidation and contract-definition, not a rebuild**:

1. **One core stack** — merge the typed `src/` production path and the `scripts/research/common/` research path into a single `src/core`.
2. **Three evaluation pipelines, kept separate** — cross-sectional, time-series, convexity. They are distinct by design; only their shared gate utilities get unified.
3. **One machine-readable registry** — a single YAML schema that is the source of truth for every alpha (metadata + payoff shape + signal function + universe + costs + stage), replacing the split JSON ledger / markdown hypotheses / hardcoded `Candidate` objects.
4. **One templated runner** — `python -m research run <registry_id>` replaces 293 bespoke scripts; routes to the right pipeline by payoff shape; emits a standard tearsheet.
5. **One contract doc** — `CLAUDE.md` defines the loop both humans and agents follow.

Each phase is independently valuable and low-risk. Nothing is mass-deleted; old scripts are
archived and ported as they earn their keep.

---

## 1. Current-state diagnosis

### 1.1 The strong bones (keep, do not rewrite)

**Typed production core** (`src/`):

| Layer | Location | Contract |
|---|---|---|
| Data | `src/data/portal.py` | `DataPortal(cfg: DataConfig).load_bars() -> pl.DataFrame` (ts, symbol, OHLCV [+funding]) |
| Strategy | `src/strategy/base.py` | `TargetWeightStrategy.on_bar_close(ctx) -> float`; `PortfolioStrategy` protocol for multi-asset |
| Risk | `src/risk/risk_manager.py` | `RiskManager.apply(weight, history) -> float`; vol targeting + leverage/concentration limits |
| Execution | `src/backtest/engine.py` | `BacktestEngine(cfg, strategy, risk, portal).run() -> (Portfolio, summary)` — Model B (decide@close, fill@open+1) |
| Config | `src/common/config.py` | Pydantic `RunConfigRaw` → `compile_config()` → `RunConfigResolved` (+ hash, manifest) |
| Reporting | `src/analysis/tearsheet.py` | `generate_tearsheet(df, out)` — quantile/IC PDF + JSON |
| Research API | `src/research/api.py` | `quick_backtest()`, `quick_sweep()` notebook helpers |

**Research process on paper** (`docs/research/`):

- `alpha_hypothesis_template.md` — 13-section pre-registration template (identity, hypothesis, rationale, payoff shape, signal def, entry/exit, universe, costs, pre-registered metrics, falsification, blow-up scenarios, stage routing, sign-off).
- `convexity_alpha_pipeline_spec.md` — 8-stage gated pipeline (S0 registration → S7 retirement), convexity-first metrics (CCS), Trend-with-stops vs Vol-expansion tracks.
- `hypotheses/`, `cohorts/` — worked examples of registered hypotheses and cohort selection/calibration/results.
- `strategy_registry_v0.json` — strategy ledger (metrics_csv, equity_csv, tearsheet_pdf, run_recipe, git_tag).

**Three distinct evaluation pipelines** (`src/`) — **correctly separate**:

| Pipeline | Module | Input contract | Backtest model | Gates |
|---|---|---|---|---|
| Cross-sectional | `src/alpha_pipeline/` | rank scores `(ts, symbol) → float` | inverse-vol L/S quintiles | IC → IC decay → redundancy → turnover → WF/CPCV → deflated SR |
| Time-series | `src/ts_pipeline/` | directional forecasts `(ts, symbol) → float` | vol-targeted portfolio (Carver) | TS IC → persistence → horizon → vol portfolio → WF → DSR → blend |
| Convexity | `src/convexity_pipeline/` | hypothesis + `BacktestFn` | caller-provided, routed by payoff shape | S0 registration → S1 screen → S2 costs → S3 regime → S4 rigor |

`ts_pipeline` already reuses `alpha_pipeline`'s `StageResult`/`StageVerdict` and embargo logic —
evidence the shared-gate extraction (Phase 4) is natural, not forced.

### 1.2 The fragmentation (the real target)

**Problem 1 — Two competing core stacks.**
The typed `src/` path (`DataPortal` + `BacktestEngine` + Pydantic) and the
`scripts/research/common/` path (`load_daily_bars` + `simple_backtest` + `compute_metrics`)
are parallel reimplementations of load → signal → backtest → metrics. **All 293 research
scripts use the latter; almost nothing in research touches the typed core.** Consequences:
divergent cost assumptions (`DEFAULT_COST_BPS = 20.0` in research vs `ExecutionConfig` in
core), two annualization conventions, and ambiguity over "which Sharpe is the real Sharpe."

```python
# scripts/research/common/backtest.py  — the path research actually uses
def simple_backtest(weights, returns, cost_bps=20.0, execution_lag=1) -> pd.DataFrame: ...

# src/backtest/engine.py  — the typed path production uses
class BacktestEngine:
    def __init__(self, cfg, strategy, risk_manager, data_portal, ...): ...
    def run(self) -> tuple[Portfolio, dict]: ...
```

**Problem 2 — 293 one-off scripts, no execution registry.**
21 research subdirs, each with ~5 `run_X.py` scripts repeating argparse + load + output-dir
boilerplate. Two entry-point conventions coexist (`__main__.py` orchestration vs ad-hoc
`run_*.py`). `strategy_registry_v0.json` is a *reporting* ledger, not a dispatch mechanism —
nothing maps a strategy name to runnable code.

**Problem 3 — Process docs not wired to code.**
The hypothesis template, stage gates, and CCS metric live in markdown; the pipelines live in
`src/`; the `Candidate` objects live hardcoded in pipeline demos. Nothing enforces that a
backtest originated from a pre-registered hypothesis.

**Problem 4 — No agent-facing contract.**
No `CLAUDE.md`, no machine-readable "how to propose / run / evaluate an alpha." An agent must
reverse-engineer the repo before it can contribute.

---

## 2. Target architecture

Organizing principle: **one core stack, three evaluation pipelines, one registry, one contract.**

```
src/
  core/                  ← THE single stack. Merge src/backtest + src/data + src/risk +
                            src/common/config + scripts/research/common into one home.
                            One DataPortal, one backtest, one metrics module, one cost model.
                            Wide-panel and single-asset adapters both live here.
    data.py                  load_bars / DataPortal (DuckDB, timeframe handling, provenance)
    backtest.py              unified engine (Model B) + wide-panel adapter
    metrics.py               compute_metrics, CCS, IC, deflated Sharpe — ONE module
    costs.py                 single cost model (bps + dynamic impact + funding)
    config.py                Pydantic RunConfig (raw → resolved)
    tearsheet.py             standard tearsheet (absorbs src/analysis + tearsheet_common_v0)

  pipelines/
    cross_sectional/     ← was src/alpha_pipeline/
    time_series/         ← was src/ts_pipeline/
    convexity/           ← was src/convexity_pipeline/
    common/              ← shared gates: CPCV/embargo, PBO, deflated Sharpe,
                            StageResult / StageVerdict / GateConfig base types

  signals/               ← pure signal functions, NO I/O. Absorbs src/alphas (DSL) +
                            indicator libraries. Everything a registry signal_fn can point to.

  domains/
    k2_macro/            ← was src/k2_systematic_macro (single-product CL sandbox; leave as-is)
    volbook/             ← futures data infra + dashboards (leave as-is)

  utils/                 ← misc helpers
  volatility/            ← estimator library (leave as-is)
  live/  execution/      ← unchanged
  monitoring/ portfolio/ pricing/ validation/ screeners/  ← unchanged for now

research/                ← ONE templated runner replaces 293 bespoke scripts.
  __main__.py              `python -m research run <registry_id>`
  runner.py                resolve candidate → route by payoff_shape → backtest → tearsheet
  _archive/                old scripts land here; ported as they earn their keep

registry/                ← the keystone (machine-readable source of truth)
  alphas/<registry_id>.yaml
  results/<registry_id>/   canonical equity.csv / metrics.csv / tearsheet.pdf / manifest.json

CLAUDE.md                ← agent + human operating contract
docs/research/           ← templates stay; become the human-readable spec the registry mirrors
```

### 2.1 What explicitly does NOT change
- **The three pipelines stay separate.** Different input contracts, backtest models, gates.
- **`k2_macro` / `volbook` internals untouched** — domain sandbox + data infra, not pipeline candidates. They keep feeding research as today.
- **`live/`, `execution/`** — out of scope for this reorg.
- **No mass deletion.** Scripts are archived, not removed.

---

## 3. The keystone: a unified alpha registry

Today the "registry" is split three ways: a JSON ledger (`strategy_registry_v0.json`),
markdown hypotheses (`docs/research/hypotheses/*.md`), and `Candidate` objects hardcoded in
pipeline demos. Unify into **one YAML schema** that is the single source of truth and mirrors
the fields the hypothesis template already defines.

### 3.1 Schema (`registry/alphas/<registry_id>.yaml`)

```yaml
registry_id: 2026-06-continuation-index      # unique, date-prefixed
name: "Ehlers Continuation Index"
researcher: rfloyd
registered: 2026-06-10
source: "Ehlers (2001); docs/research/hypotheses/2026-06-continuation-index.md"

# --- Routing (decides which pipeline runs, automatically) ---
payoff_shape: convex          # convex | linear | concave | ambiguous
track: trend_with_stops       # trend_with_stops | vol_expansion | cross_sectional | null
horizon_bars: 20

# --- Hypothesis (pre-registration; runner refuses backtest if incomplete) ---
hypothesis: "A rising continuation index predicts positive forward returns over ~20 bars."
rationale: "Persistence of trend regimes; not 'the indicator works'."
falsification:                # any TRUE at the gated stage kills the alpha
  - "Aggregate skew < 0 at S1"
  - "CCS_OOS / CCS_IS < 0.5 at S3"

# --- Implementation ---
signal_fn: signals.trend.continuation_index   # dotted path into src/signals
signal_params: {fast: 5, slow: 40}
universe: midcap_usd                            # named universe
cost_profile: crypto_default                    # named cost model in core/costs.py

# --- Pre-registered expectations (filled BEFORE backtest) ---
pre_registered_metrics:
  ccs: {expected: 1.2, confidence: medium}
  skew: {expected: 0.4}
  tail_capture: {expected: 0.35}

# --- Lifecycle (advanced only by the runner, with provenance) ---
stage: S1                     # S0..S7 | live | retired
status: queued                # queued | running | passed | killed | live | retired
git_tag: null
```

### 3.2 Why this is the highest-leverage move

- **Routing for free.** `payoff_shape` + `track` already determine cross-sectional vs time-series vs convexity. The runner dispatches with no per-strategy glue.
- **Pre-registration becomes a code check.** The runner refuses to backtest a candidate whose hypothesis/falsification fields are empty. The existing "rewrite after backtest → back to S0" rule stops being an honor system.
- **One agent surface.** An AI agent proposes an alpha by writing **one YAML + one `signal_fn`** and calling the runner. No bespoke script. The schema *is* the prompt template.
- **Results provenance is uniform.** Every run writes to `registry/results/<registry_id>/` with a manifest hash — replacing 21 subdirs of ad-hoc `output/` paths.

---

## 4. The runner

```
python -m research run <registry_id> [--stage S1] [--start ... --end ...]
python -m research sweep <registry_id> --grid fast=5,10 slow=20,40
python -m research promote <registry_id>     # advance stage iff gates pass; writes provenance
python -m research list [--stage S1] [--track trend_with_stops]
```

Internally: load registry YAML → validate pre-registration → resolve `signal_fn` and
`universe` → load bars via `core.data` → route to the pipeline implied by `payoff_shape` →
run gates → write standard tearsheet + metrics + manifest to `registry/results/`. One code
path; all 293 scripts collapse into "a registry entry + a signal function."

---

## 5. CLAUDE.md — the operating contract

The doc that makes the repo agent-usable (and disciplines humans). It states, unambiguously:

- **The one true way to load data** (`core.data.load_bars`), the **one backtest entry point**, the **one metrics module**.
- **The registry schema** and where signal functions live (`src/signals`).
- **The loop:** propose hypothesis → register YAML → implement `signal_fn` → `research run` → read tearsheet → `research promote` through stages.
- **The gate definitions** (CCS thresholds, IC screens, deflated Sharpe) per pipeline.
- **The hard rules:** pre-register before backtest; costs are non-optional; no look-ahead (history is sliced to `decision_ts`); promotion only via the runner.

Both humans and agents follow the same loop. The agent's job becomes "write a YAML and a pure
function," which is exactly the surface area an LLM handles reliably.

---

## 6. Phased migration

Each phase is independently valuable and shippable. Ordering minimizes breakage.

### Phase 1 — Unify the core stack *(highest impact, kills the worst drift)*
- Choose canonical implementations: one backtest, one data loader, one metrics module, one cost model.
- Make `scripts/research/common/{backtest,data,metrics,risk_overlays}.py` **thin shims that re-export from `src/core`** → zero breakage for the 293 scripts on day one.
- Reconcile the two cost/annualization conventions explicitly; document the decision.
- Migrate callers off the shims incrementally.
- **Exit criteria:** research and production share one backtest + one metrics module; shims emit a deprecation note.

### Phase 2 — CLAUDE.md + registry schema *(pure addition, zero breakage)*
- Write `CLAUDE.md` (the contract) and `registry/alphas/` schema + a JSON-schema validator.
- Backfill 2–3 existing strategies (Medallion Lite, an MA 5/40, a convexity candidate) as reference registrations.
- **Exit criteria:** repo is legible to an agent; `research list` shows the backfilled alphas.

### Phase 3 — Templated runner + registry dispatch
- Build `research/runner.py` and `python -m research run/sweep/promote/list`.
- Route by `payoff_shape` to the existing three pipelines.
- Port the flagship strategies to run end-to-end through the runner.
- **Exit criteria:** a new alpha is runnable from a YAML + signal function alone.

### Phase 4 — Consolidate shared pipeline gates
- Extract `pipelines/common`: CPCV/embargo, PBO, deflated Sharpe, `StageResult`/`Verdict`, `GateConfig` base.
- Point all three pipelines at it (`ts_pipeline` already borrows from `alpha_pipeline` — formalize this).
- **Exit criteria:** one implementation of each shared statistic; pipelines import from `common`.

### Phase 5 — Archive and port
- Move the 293 scripts to `research/_archive/`.
- Port the ones with ongoing value into registry entries; leave the rest archived (git-recoverable).
- **Exit criteria:** `scripts/research/` top level holds only the runner and actively-maintained tooling.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Merging core stacks changes a backtest number | Phase 1 reconciles conventions *explicitly* and diffs a reference strategy's equity curve before/after; document any intended change. |
| Re-export shims hide subtle behavior differences | Add characterization tests on `simple_backtest` vs unified engine for a fixed input before swapping. |
| Registry schema churn | Version the schema (`schema_version`) and validate in CI; treat changes as migrations. |
| 293-script archive loses something live | Archive (not delete); grep `cron`/CI for any script still scheduled before moving it. |
| Reorg stalls the active convexity branch | Phases 2–3 are additive; do the `convexity_pipeline` → `pipelines/convexity` move only when that branch merges. |

---

## 8. Open questions for the team

1. **Cost/annualization reconciliation** — when research (`365`, `20bps`) and core
   (`ExecutionConfig`, timeframe-derived annualization) disagree, which wins, and do we accept
   re-stated historical Sharpes?
2. **Registry storage** — flat YAML files (git-diffable, agent-friendly) vs a DuckDB table
   (queryable). Proposal favors YAML as source of truth with an optional generated DuckDB view.
3. **Universe definitions** — centralize named universes (`midcap_usd`, etc.) in
   `core` so both registry and `DataPortal` resolve the same membership.
4. **Where K2 macro plugs in** — does it adopt the registry/runner for its CL signals, or stay
   a standalone domain sandbox indefinitely?

---

## 9. Concrete first PRs (if approved)

1. `docs/` — this plan (done) + `CLAUDE.md` draft.
2. `registry/` — schema + validator + 1 backfilled reference alpha.
3. `src/core/metrics.py` — unify metrics, with characterization tests vs both existing impls.
4. Shim `scripts/research/common/metrics.py` → `src/core/metrics.py` and verify the 293 scripts still import cleanly.
```

(Steps 3–4 are the smallest safe slice of Phase 1 and validate the whole approach.)
