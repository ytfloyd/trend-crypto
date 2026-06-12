"""Registry-driven research runner.

Resolves a registry entry (core.registry.AlphaSpec) into a runnable plan:
validates pre-registration, routes by payoff_shape/track to the right pipeline,
resolves the signal function, and records a run plan / promotes stages.

This is the one entry point that replaces bespoke run_*.py scripts. CLI surface
lives in research.cli (`python -m research`). See
docs/RESEARCH_PIPELINE_REORGANIZATION.md and registry/README.md.

Note on execution: full backtest execution requires the signal_fn to be
importable (the signals.* package lands in reorg task #5) and market data to be
present. Until then `resolve_run` still validates, routes, and resolves
everything it can, and `run` writes a run plan; it reports clearly when the
signal_fn is not yet importable rather than faking a result.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import yaml

from core.backtest import DEFAULT_COST_BPS, simple_backtest
from core.metrics import compute_metrics
from core.registry import AlphaSpec, PayoffShape, Stage, Track, load_alpha_spec

# repo-root/registry/results/<registry_id>/
RESULTS_DIR = Path(__file__).resolve().parents[2] / "registry" / "results"

# pipeline name -> importable package
PIPELINE_MODULES: dict[str, str] = {
    "cross_sectional": "pipelines.cross_sectional",
    "time_series": "pipelines.time_series",
    "convexity": "pipelines.convexity",
}

# linear pipeline progression for promotion
_STAGE_ORDER: tuple[Stage, ...] = (
    Stage.S0, Stage.S1, Stage.S2, Stage.S3, Stage.S4, Stage.S5, Stage.S6, Stage.LIVE
)


def route_for(spec: AlphaSpec) -> str:
    """Decide which pipeline evaluates this alpha, from payoff_shape/track.

    - cross_sectional track          -> cross_sectional pipeline (rank L/S)
    - convex payoff                  -> convexity pipeline (trend / vol_expansion)
    - linear / ambiguous / concave   -> time_series pipeline (directional)
    """
    if spec.track == Track.CROSS_SECTIONAL:
        return "cross_sectional"
    if spec.payoff_shape == PayoffShape.CONVEX:
        return "convexity"
    return "time_series"


def resolve_signal_fn(spec: AlphaSpec) -> tuple[Optional[Callable[..., Any]], str]:
    """Import the dotted ``signal_fn`` path. Returns (callable_or_None, reason)."""
    module_path, _, attr = spec.signal_fn.rpartition(".")
    if not module_path:
        return None, f"signal_fn {spec.signal_fn!r} is not a dotted path"
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        return None, f"signal_fn module {module_path!r} not importable yet ({exc})"
    fn = getattr(mod, attr, None)
    if fn is None:
        return None, f"{module_path!r} has no attribute {attr!r}"
    if not callable(fn):
        return None, f"signal_fn {spec.signal_fn!r} is not callable"
    return fn, "ok"


@dataclass
class ResolvedRun:
    """Everything needed to (attempt to) execute an alpha, fully resolved."""
    spec: AlphaSpec
    route: str
    pipeline_module: str
    signal_fn: Optional[Callable[..., Any]]
    signal_reason: str

    @property
    def is_runnable(self) -> bool:
        return self.signal_fn is not None

    @property
    def blockers(self) -> list[str]:
        out: list[str] = []
        if not self.spec.is_preregistration_complete():
            out.append("pre-registration incomplete (hypothesis/rationale/falsification)")
        if self.signal_fn is None:
            out.append(self.signal_reason)
        return out


def resolve_run(spec: AlphaSpec) -> ResolvedRun:
    """Resolve a spec to a route + pipeline + signal callable (no execution)."""
    route = route_for(spec)
    fn, reason = resolve_signal_fn(spec)
    return ResolvedRun(
        spec=spec,
        route=route,
        pipeline_module=PIPELINE_MODULES[route],
        signal_fn=fn,
        signal_reason=reason,
    )


def results_dir_for(registry_id: str) -> Path:
    return RESULTS_DIR / registry_id


def write_run_plan(resolved: ResolvedRun) -> Path:
    """Persist a run plan to registry/results/<id>/plan.json. Returns the path."""
    out_dir = results_dir_for(resolved.spec.registry_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "registry_id": resolved.spec.registry_id,
        "name": resolved.spec.name,
        "route": resolved.route,
        "pipeline_module": resolved.pipeline_module,
        "signal_fn": resolved.spec.signal_fn,
        "stage": resolved.spec.stage.value,
        "status": resolved.spec.status.value,
        "runnable": resolved.is_runnable,
        "blockers": resolved.blockers,
    }
    path = out_dir / "plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path


def load_bars_for(spec: AlphaSpec) -> pd.DataFrame:
    """Load market bars for a spec's universe at its bar frequency.

    Lazy-imports core.data (duckdb) so the runner stays importable without a DB.
    Raises a clear error for named universes (no universe registry yet) or a
    missing data lake.
    """
    from core.data import load_bars  # lazy: pulls in duckdb

    bars = load_bars(spec.bar_frequency)
    if isinstance(spec.universe, list):
        return bars[bars["symbol"].isin(spec.universe)].copy()
    raise ValueError(
        f"named universe {spec.universe!r} not resolvable yet (no universe registry); "
        "use an explicit symbol list in the registry entry"
    )


def execute_screen(
    resolved: ResolvedRun, bars: pd.DataFrame, cost_bps: Optional[float] = None
) -> dict[str, Any]:
    """Fast vectorized screen: signal_fn -> simple_backtest -> equity metrics.

    This is the S1-style quick screen, not the full routed pipeline (which adds
    the per-pipeline stage gates). Requires a resolved signal_fn and complete
    pre-registration. Returns {"metrics", "equity", "backtest"}.
    """
    if resolved.signal_fn is None:
        raise ValueError(f"{resolved.spec.registry_id}: not runnable — {resolved.signal_reason}")
    resolved.spec.require_preregistration()

    weights = resolved.signal_fn(bars, **resolved.spec.signal_params)
    close = bars.pivot(index="ts", columns="symbol", values="close").sort_index()
    returns = close.pct_change().fillna(0.0)
    bt = simple_backtest(
        weights, returns, cost_bps=DEFAULT_COST_BPS if cost_bps is None else cost_bps
    )
    equity = bt.set_index("ts")["portfolio_equity"]
    return {"metrics": compute_metrics(equity), "equity": equity, "backtest": bt}


def write_results(registry_id: str, result: dict[str, Any]) -> Path:
    """Persist screen outputs to registry/results/<id>/{metrics.json,equity.csv}."""
    out_dir = results_dir_for(registry_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, default=float) + "\n", encoding="utf-8"
    )
    result["backtest"].to_csv(out_dir / "equity.csv", index=False)
    return out_dir


def next_stage(stage: Stage) -> Optional[Stage]:
    """The stage that follows ``stage`` in the linear pipeline, or None at the end."""
    if stage not in _STAGE_ORDER:
        return None
    idx = _STAGE_ORDER.index(stage)
    return _STAGE_ORDER[idx + 1] if idx + 1 < len(_STAGE_ORDER) else None


def promote(spec: AlphaSpec) -> Stage:
    """Advance the alpha to the next stage, enforcing the pre-registration gate.

    Raises ValueError if pre-registration is incomplete or the alpha is already
    at the terminal stage. Returns the new stage. Does not persist; callers use
    ``write_spec`` to save.
    """
    spec.require_preregistration()
    nxt = next_stage(spec.stage)
    if nxt is None:
        raise ValueError(f"{spec.registry_id}: already at terminal stage {spec.stage.value}")
    spec.stage = nxt
    return nxt


def write_spec(spec: AlphaSpec, path: str | Path) -> None:
    """Persist a spec back to YAML (canonical formatting; enums as their values)."""
    data = spec.model_dump(mode="json", exclude_none=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def load_and_resolve(path: str | Path) -> ResolvedRun:
    """Convenience: load a single registry YAML and resolve it."""
    return resolve_run(load_alpha_spec(path))


__all__ = [
    "RESULTS_DIR",
    "PIPELINE_MODULES",
    "ResolvedRun",
    "route_for",
    "resolve_signal_fn",
    "resolve_run",
    "results_dir_for",
    "write_run_plan",
    "load_bars_for",
    "execute_screen",
    "write_results",
    "next_stage",
    "promote",
    "write_spec",
    "load_and_resolve",
]
