"""`python -m research` — the registry-driven research CLI.

Subcommands:
    list      [--stage S1] [--track trend]   list registered alphas + routing
    validate                                 validate every registry entry
    run       <registry_id>                  resolve + route + write a run plan
    promote   <registry_id>                  advance one stage (pre-registration gated)

See research.runner for the underlying logic and registry/README.md for the schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.registry import DEFAULT_REGISTRY_DIR, load_alpha_spec, load_registry

from . import runner


def _spec_path(args: argparse.Namespace) -> Path:
    return Path(args.registry_dir) / f"{args.registry_id}.yaml"


def _cmd_list(args: argparse.Namespace) -> int:
    specs = load_registry(args.registry_dir)
    rows = sorted(specs.values(), key=lambda s: s.registry_id)
    if args.stage:
        rows = [s for s in rows if s.stage.value == args.stage]
    if args.track:
        rows = [s for s in rows if s.track.value == args.track]
    if not rows:
        print("(no matching alphas)")
        return 0
    print(f"{'registry_id':<32} {'route':<15} {'stage':<6} {'status':<8} {'prereg':<6} name")
    print("-" * 100)
    for s in rows:
        prereg = "yes" if s.is_preregistration_complete() else "NO"
        print(
            f"{s.registry_id:<32} {runner.route_for(s):<15} {s.stage.value:<6} "
            f"{s.status.value:<8} {prereg:<6} {s.name}"
        )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        specs = load_registry(args.registry_dir)
    except Exception as exc:  # noqa: BLE001 — report any validation failure
        print(f"INVALID: {exc}")
        return 1
    print(f"OK: {len(specs)} registry entries valid.")
    for s in specs.values():
        if not s.is_preregistration_complete():
            print(f"  warn: {s.registry_id} pre-registration incomplete (cannot run/promote)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_alpha_spec(_spec_path(args))
    resolved = runner.resolve_run(spec)
    path = runner.write_run_plan(resolved)
    print(f"alpha    : {spec.registry_id}  ({spec.name})")
    print(f"route    : {resolved.route}  ->  {resolved.pipeline_module}")
    print(f"signal_fn: {spec.signal_fn}  [{resolved.signal_reason}]")
    print(f"stage    : {spec.stage.value}   status: {spec.status.value}")
    print(f"plan     : {path}")
    if resolved.blockers:
        print("blockers :")
        for b in resolved.blockers:
            print(f"  - {b}")
        print("not runnable yet — see blockers above.")
        return 1
    print("runnable : yes")

    if not args.execute:
        print("(use --execute to run the fast screen)")
        return 0
    try:
        bars = runner.load_bars_for(spec)
    except Exception as exc:  # noqa: BLE001 — surface any data/universe issue
        # Exit 2 (not 1): data/infra unavailable is distinct from a blocked alpha,
        # so automation can tell "DB not mounted" from "alpha not runnable".
        print(f"execute  : data unavailable — {exc}")
        return 2
    try:
        result = runner.execute_screen(resolved, bars)
    except ValueError as exc:
        print(f"execute  : {exc}")
        return 1
    out = runner.write_results(spec.registry_id, result)
    m = result["metrics"]
    print(f"executed : sharpe={m['sharpe']:.2f}  cagr={m['cagr']:.1%}  max_dd={m['max_dd']:.1%}")
    print(f"results  : {out}")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    path = _spec_path(args)
    spec = load_alpha_spec(path)
    prev = spec.stage
    try:
        new = runner.promote(spec)
    except ValueError as exc:
        print(f"cannot promote: {exc}")
        return 1
    runner.write_spec(spec, path)
    print(f"promoted {spec.registry_id}: {prev.value} -> {new.value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research", description="Registry-driven research runner.")
    p.add_argument(
        "--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR,
        help="directory of registry/alphas/*.yaml (default: repo registry)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list", help="list registered alphas + routing")
    pl.add_argument("--stage", help="filter by stage (e.g. S1)")
    pl.add_argument("--track", help="filter by track (e.g. trend, cross_sectional)")
    pl.set_defaults(func=_cmd_list)

    pv = sub.add_parser("validate", help="validate every registry entry")
    pv.set_defaults(func=_cmd_validate)

    pr = sub.add_parser("run", help="resolve + route + write a run plan")
    pr.add_argument("registry_id")
    pr.add_argument(
        "--execute", action="store_true",
        help="run the fast screen end-to-end (requires signal_fn + market data)",
    )
    pr.set_defaults(func=_cmd_run)

    pp = sub.add_parser("promote", help="advance one stage (pre-registration gated)")
    pp.add_argument("registry_id")
    pp.set_defaults(func=_cmd_promote)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func  # set by each subparser
    result: int = func(args)
    return result


__all__ = ["build_parser", "main"]
