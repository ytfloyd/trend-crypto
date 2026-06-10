# Convexity Pipeline — Revised Implementation Plan

**Intended path:** `docs/research/convexity_pipeline_implementation_plan.md`
**Status:** v1.0 — supersedes prior "Alpha Registry routing fields" plan
**Companion docs:**
- `docs/research/convexity_alpha_pipeline_spec.md` — what we're building
- `docs/research/alpha_hypothesis_template.md` — Stage 0 form
- `data/alpha_registry/alpha_registry_v2.xlsx` — seed ledger

---

## 1. Scope change (decision)

**Old scope:** Add routing fields to the Alpha Registry. Treat the registry as the deliverable.

**New scope:** Build the convexity research operating system. The registry is one component (the research ledger); the spec is the operating doctrine; `src/convexity_pipeline/` is the engine; the hypothesis template is the Stage 0 input format.

This supersedes the earlier plan. The earlier plan was useful but too narrow — it would have given us a sortable index of ideas without a way to actually evaluate them under our own framework.

## 2. Layout

```
docs/research/
├── convexity_alpha_pipeline_spec.md          # operating doctrine
├── alpha_hypothesis_template.md              # Stage 0 form
├── convexity_pipeline_implementation_plan.md # this doc
└── hypotheses/                                # one .md per candidate
    └── 2025-001-ehlers-continuation-index.md  # example

data/alpha_registry/
└── alpha_registry_v2.xlsx                    # seed research ledger

src/
├── alpha_pipeline/                           # EXISTING - cross-sectional / linear
│   └── ...                                   # untouched
└── convexity_pipeline/                       # NEW - trend / vol / convex
    ├── __init__.py
    ├── types.py                              # enums + dataclasses
    ├── metrics.py                            # CCS, tail capture, convexity beta, etc.
    ├── stages.py                             # Stage 0-4 evaluators
    ├── runner.py                             # orchestrator
    ├── thresholds.py                         # configurable kill criteria
    └── tests/
        ├── test_metrics.py
        └── test_stages.py

scripts/research/
└── convexity_alpha_runner.py                 # CLI for batch cohorts
```

## 3. Sequencing (per dev recommendation)

Six steps in order. Do not skip ahead. Calibration on real candidates happens *before* scaling and before any Stage 5+ ops work.

### Step 1 — Drop in static artifacts

- `docs/research/convexity_alpha_pipeline_spec.md`
- `docs/research/alpha_hypothesis_template.md`
- `docs/research/convexity_pipeline_implementation_plan.md`
- `data/alpha_registry/alpha_registry_v2.xlsx`

**Deliverable:** Files in repo, CI lints markdown, registry opens in Excel and recalculates cleanly.

**Exit criteria:** Research lead can read the spec end-to-end without flagging undefined terms.

### Step 2 — `convexity_pipeline/metrics.py`

All pure functions. Each takes pandas/numpy primitives and returns scalar (or tuple). No state.

Required functions:

- `skew(returns: pd.Series) -> float`
- `calmar(returns: pd.Series, periods_per_year: int = 252) -> float`
- `max_drawdown(equity: pd.Series) -> float`
- `sharpe(returns: pd.Series, periods_per_year: int = 252) -> float`
- `payoff_ratio(trade_pnls: pd.Series) -> float`
- `profit_factor(trade_pnls: pd.Series) -> float`
- `hit_rate(trade_pnls: pd.Series) -> float`
- `max_consecutive_losses(trade_pnls: pd.Series) -> int`
- `tail_capture(alpha_returns, underlying_returns, decile=0.10) -> float`
- `convexity_beta(alpha_returns, underlying_returns) -> Tuple[float, float]`  # (b, p_value)
- `tail_sharpe_asymmetry(returns) -> float`
- `pain_ratio(equity) -> float`
- `composite_convexity_score(...) -> float`
- `trade_duration_stats(durations: pd.Series) -> Dict[str, float]`
- `calculate_all(...) -> Dict[str, Any]` — convenience

**Test coverage:** unit tests covering edge cases (empty, all-zero, all-same, NaN handling, division-by-zero).

**Deliverable:** module + tests passing.

### Step 3 — `convexity_pipeline/types.py`

Dataclasses and enums. No business logic. This module defines the interfaces between Stage evaluators, the runner, and the registry.

- `Track` (enum: TREND, VOL_EXPANSION, BOTH, NA)
- `PayoffShape` (enum: CONVEX, LINEAR, CONCAVE, AMBIGUOUS, NA)
- `Stage` (enum: S0..S7, LIVE, RETIRED, KILLED)
- `Hypothesis` (dataclass: all template fields)
- `Candidate` (dataclass: id, name, hypothesis, signal_fn, universe, params, costs)
- `BacktestResult` (dataclass: alpha_returns, underlying_returns, trades, equity, per_instrument)
- `StageResult` (dataclass: stage, passed, metrics, kill_reasons, notes, timestamp)
- `KillDecision` (dataclass: stage, criteria_violated, details)
- `PipelineConfig` (dataclass: thresholds for each stage, cost model, universes)

**Deliverable:** module loadable, no logic — purely declarative.

### Step 4 — `convexity_pipeline/stages.py`

Stage 0–4 evaluators. Each implements a single `evaluate(candidate, backtest) -> StageResult` method. Kill criteria are read from `PipelineConfig`, not hardcoded.

- `Stage0Evaluator` — validates hypothesis completeness, routing fields, no duplicate alphas in pool.
- `Stage1Evaluator` — fast metrics + dev-modified kill criteria (pooled skew, per-instrument catastrophic-skew, CCS_agg, % universe positive, convexity beta, trade-duration-vs-horizon, max consecutive losses).
- `Stage2Evaluator` — realistic cost adjustment + CCS-drop check + per-instrument net-positive check.
- `Stage3Evaluator` — walk-forward OOS evaluation; CCS_OOS / CCS_IS ratio gate (not Sharpe); median + aggregate OOS skew check; catastrophic-fold check.
- `Stage4Evaluator` — robustness battery (parameter perturbation, regime decomposition, universe substitution, cost sensitivity, look-ahead audit, curve-fit detection).

**Stages 5-7 are deferred** — represented as states in `Stage` enum but no evaluator code. Status transitions to S5+ happen via the registry, gated by manual ops sign-off until the live-trading infrastructure is wired.

**Deliverable:** Stages 0-4 with tests using synthetic candidates. Stages 5-7 documented as TODO and tracked in the registry status model only.

### Step 5 — `convexity_pipeline/runner.py` + scorecard

Orchestrator that:
- Takes a list of `Candidate`s and a backtest engine (Protocol).
- For each candidate, runs Stages 0-4 in order, stopping at first kill.
- Emits a `pd.DataFrame` scorecard: one row per candidate, columns for each stage's pass/fail + key metrics.
- Persists results to a parquet file in `data/alpha_registry/runs/<timestamp>/`.

CLI in `scripts/research/convexity_alpha_runner.py` for batch invocation.

**Deliverable:** Runner can process 5-10 registry candidates in one shot and produce a sortable scorecard.

### Step 6 — Calibrate thresholds on first cohort

Before promoting *any* candidate to Stage 5, run the first cohort and **calibrate the kill thresholds against observed distributions**. The spec's numeric thresholds are illustrative; they need to be tuned to our cost model, universe, and data.

Process:
1. Pick 10 candidates from the registry's T1 list (mix of Cat-1 alphas, covering both tracks).
2. Run all 10 through Stages 0-4 with the default thresholds from `PipelineConfig`.
3. Plot distributions of every metric *across all candidates*.
4. For each kill criterion, choose a threshold at the 25th percentile (kill the worst quartile) rather than relying on the spec's illustrative number.
5. Document calibrated thresholds in `convexity_pipeline/thresholds.py` and check into git.
6. Re-run cohort with calibrated thresholds.

**Deliverable:** Calibrated `thresholds.py`, written changelog in implementation plan, first cohort scorecard.

**Exit criteria for Phase 1:** 5-10 candidates run through 0-4; thresholds calibrated; at least 2 candidates pass through to Stage 4 with full robustness scorecards; team has confidence in metric definitions and kill criteria.

## 4. What we explicitly defer to Phase 2+

Per dev caution: do not pretend we have live-ops infrastructure yet.

| Component | Phase | Why deferred |
|---|---|---|
| Stage 5 paper-trading executor | Phase 2 | Needs broker connectivity + execution logger |
| Stage 6 live shadow allocator | Phase 2 | Needs position management + reconciliation |
| Stage 7 daily/weekly/quarterly monitoring | Phase 2 | Needs dashboard + alerting infrastructure |
| Decay detection | Phase 2 | Needs live PnL feed |
| Combiner (production weighting) | Phase 3 | After Phase 2 has run live shadow for ≥1 quarter |
| Capacity model (quantitative) | Phase 2 | Needs execution logs to calibrate impact |

For all of these, the registry's `stage` field captures the status (S5, S6, etc.) but advancement is a manual ops decision in Phase 1.

## 5. Interfaces with existing systems

### 5.1 With `src/alpha_pipeline/`

The two pipelines do not call each other. They share:
- Universe definitions in `data/universes/` (single source of truth).
- Cost model defaults (eventually shared module).
- Combiner (Phase 3 — one combiner, both pipeline outputs feed in).

Routing happens at registration time via the registry. A research lead can re-route a candidate if Stage 0 reveals it's actually a different alpha class.

### 5.2 With the existing backtest engine

`BacktestEngine` is defined as a `Protocol` in `types.py`. Our existing backtest engine implements the protocol via an adapter. We do not build a new backtest engine inside the convexity pipeline.

```python
class BacktestEngine(Protocol):
    def run(self, signal_fn: Callable, universe: List[str],
            costs: CostModel, **kwargs) -> BacktestResult: ...
```

Adapter responsibility: take our existing engine's output, populate `BacktestResult` fields (alpha_returns, underlying_returns, trade-level pnls, equity, per_instrument breakdown).

### 5.3 With the Alpha Registry

The registry is the **research ledger**. Each candidate has a registry row. Pipeline status writes back to the registry:

- `stage` updates as candidate progresses.
- `kill_reason` populated on kill.
- `ccs_is`, `ccs_oos`, `tail_capture`, `convexity_beta`, `convexity_beta_p` populated after Stage 1 and Stage 3.
- `promotion_decision` text logs each stage transition.

The runner emits a "registry diff" file that the registry maintainer applies (manual review). Phase 2 automates this write-back.

## 6. Risk and mitigation

| Risk | Mitigation |
|---|---|
| Threshold calibration on first cohort is biased by sample | Use 25th percentile of observed; re-calibrate after every 10 candidates |
| `BacktestResult` schema misses fields some alphas need | Add fields as needed in Phase 1; freeze for Phase 2 |
| Stage 4 robustness battery becomes a curve-fit detector that lets curve-fit alphas through | Periodic blind audit: run randomly-generated signals through pipeline, confirm ~100% kill rate |
| Researchers post-hoc edit hypothesis templates after seeing backtest | Hypotheses git-tracked; any edit after S0 visible in PR review |
| Convexity beta hard to interpret in OOS folds with small samples | Require minimum sample size (e.g., 252 bars) for convexity beta to be reported; otherwise N/A |
| Walk-forward inappropriate for very low-frequency alphas | Block-bootstrap fallback for alphas with < 200 round-trips total |

## 7. Decision log

- **2024-XX-XX:** Initial scope was "Alpha Registry routing fields." Dev review escalated to full convexity research OS. This plan supersedes.
- **2024-XX-XX:** Stages 5-7 deferred to Phase 2 (live ops dependencies). Status model only for now.
- **2024-XX-XX:** Calibration step inserted before any production use — illustrative thresholds in spec are not used as gates.

## 8. Phase 1 success criteria (single statement)

> When 5 candidates from the registry's T1 list can be processed through Stages 0-4 with the calibrated `thresholds.py`, with scorecards persisted to disk, robustness batteries documented per candidate, and the research lead can articulate why each pass-or-kill decision is the right one — Phase 1 is done.

## 9. Owners

| Component | Owner |
|---|---|
| Spec doc maintenance | Research lead |
| `convexity_pipeline/` code | Engineering |
| Registry maintenance | Research ops |
| Calibration cohort | Research lead + Engineering |
| Phase 2 planning | Research lead + Ops lead (kicks off month 3) |
