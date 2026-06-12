# Alpha Registry

The **single source of truth** for every alpha. One YAML file per alpha under
`registry/alphas/<registry_id>.yaml`. This converges what used to live in three
places — `data/alpha_registry/*.xlsx`, the markdown hypotheses under
`docs/research/hypotheses/`, and the in-code `Hypothesis`/`Candidate` objects.

Schema, loader, and validator: [`src/core/registry.py`](../src/core/registry.py).
Background: [`docs/RESEARCH_PIPELINE_REORGANIZATION.md`](../docs/RESEARCH_PIPELINE_REORGANIZATION.md).

## Why a registry

- **Routing for free** — `payoff_shape` + `track` decide which pipeline runs:
  - `linear` + `cross_sectional` → cross-sectional pipeline
  - `convex` + `trend` / `vol_expansion` → convexity pipeline
  - per-asset directional forecasts → time-series pipeline
- **Pre-registration as a code check** — an alpha cannot be run/promoted past S0
  until `hypothesis`, `rationale`, and `falsification` are filled in. Rewriting
  the hypothesis after seeing results sends it back to S0 (curve-fit guard).
- **One agent surface** — proposing an alpha = write one YAML + one `signal_fn`.

## Fields

| Field | Required | Notes |
|---|---|---|
| `registry_id` | yes | date-prefixed kebab-case; must equal the filename stem |
| `name`, `researcher`, `registered`, `source` | name only | provenance |
| `payoff_shape` | yes | `convex` / `linear` / `concave` / `ambiguous` / `N_A` |
| `track` | for convex | `trend` / `vol_expansion` / `both` / `cross_sectional` / `N_A` |
| `horizon_bars` | yes | expected trade horizon (> 0) |
| `hypothesis`, `rationale`, `falsification` | for promotion | pre-registration (gates past S0) |
| `signal_fn` | yes | dotted import path, e.g. `signals.trend.ma_crossover` |
| `signal_params` | no | kwargs passed to the signal function |
| `universe` | yes | named universe (str) or explicit symbol list |
| `bar_frequency` | no | default `1d` |
| `cost_profile` | no | named cost model, default `crypto_default` |
| `pre_registered_metrics` | no | `{metric: {expected, confidence}}`, filled BEFORE backtest |
| `stage` | no | `S0..S6` / `Live` / `Retired` / `Killed`; default `S0` |
| `status` | no | `queued` / `running` / `passed` / `killed` / `live` / `retired` |

## Validating

```python
from core.registry import load_registry, load_alpha_spec
specs = load_registry()                 # validates every registry/alphas/*.yaml
spec = load_alpha_spec("registry/alphas/2026-06-ma-5-40-trend.yaml")
spec.require_preregistration()          # raises if hypothesis/rationale/falsification missing
```

## Note on `signal_fn`

The validator checks `signal_fn` is a well-formed dotted path, not that it
imports today. Several targets point at the `signals.*` package, which lands
when signal libraries are consolidated (reorg task #5); the runner (task #6)
resolves and calls them. Until then these entries validate and route correctly
but are not yet end-to-end runnable.
