#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

REGISTRY_PATH = "docs/research/strategy_registry_v0.json"
REPO_ROOT = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto"
SCHEMA_PATH = "docs/research/strategy_registry_v0.schema.json"


def load_registry(path: Optional[str] = None) -> List[Dict[str, Any]]:
    use_path = path or REGISTRY_PATH
    with open(use_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("strategies", [])


def _coerce_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_metric(row: Dict[str, str], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            val = _coerce_float(row.get(key))
            if val is not None:
                return val
    return None


def _select_metrics_row(rows: List[Dict[str, str]], period_key: str) -> Dict[str, str]:
    if not rows:
        return {}
    if len(rows) == 1:
        return rows[0]
    for col in ("period", "label"):
        if col in rows[0]:
            for row in rows:
                if row.get(col) == period_key:
                    return row
    return rows[0]


def load_metrics(strategy: Dict[str, Any]) -> Dict[str, Optional[float]]:
    metrics_csv = strategy.get("metrics_csv")
    period_key = strategy.get("metrics_period", "full")
    if not metrics_csv or not os.path.exists(metrics_csv):
        return {"cagr": None, "sharpe": None, "max_dd": None}

    with open(metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    row = _select_metrics_row(rows, period_key)

    return {
        "cagr": _get_metric(row, ["cagr", "CAGR"]),
        "sharpe": _get_metric(row, ["sharpe", "Sharpe"]),
        "max_dd": _get_metric(row, ["max_dd", "MaxDD", "max_drawdown"]),
    }


def fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.2f}%"


def fmt_num(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.2f}"


def resolve_period(strategy: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    if strategy.get("start_date") or strategy.get("end_date"):
        return strategy.get("start_date"), strategy.get("end_date")

    period = strategy.get("canonical_period")
    if isinstance(period, dict):
        start = period.get("start")
        end = period.get("end")
        if start or end:
            return start, end

    period = strategy.get("period")
    if isinstance(period, dict):
        start = period.get("start")
        end = period.get("end")
        if start or end:
            return start, end
    if isinstance(period, str) and "→" in period:
        start, end = [part.strip() for part in period.split("→", 1)]
        return start or None, end or None

    return None, None


def build_list_row(strategy: Dict[str, Any]) -> Dict[str, Any]:
    m = load_metrics(strategy)
    start, end = resolve_period(strategy)
    return {
        "id": strategy.get("id"),
        "name": strategy.get("name"),
        "family": strategy.get("family"),
        "version": strategy.get("version"),
        "status": strategy.get("status"),
        "start": start,
        "end": end,
        "CAGR": fmt_pct(m["cagr"]),
        "Sharpe": fmt_num(m["sharpe"]),
        "MaxDD": fmt_pct(m["max_dd"]),
    }


def _is_tearsheet_command(line: str, tearsheet_pdf: Optional[str] = None) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    lower = stripped.lower()
    if "tearsheet" not in lower:
        return False
    if tearsheet_pdf and tearsheet_pdf in stripped:
        return True
    return "--out_pdf" in lower or "tearsheet" in lower


def _should_skip_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith("NOTE:") or stripped.startswith("NOTE "):
        return True
    if stripped.startswith("source "):
        return True
    return False


def _rewrite_python_command(line: str, python_path: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("python "):
        return f"{python_path} {stripped[len('python '):]}"
    if stripped.startswith("venv_trend_crypto/bin/python "):
        return f"{python_path} {stripped[len('venv_trend_crypto/bin/python '):]}"
    return line


def _is_python_command(line: str, python_path: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith("python ") or stripped.startswith("venv_trend_crypto/bin/python "):
        return True
    cmd = stripped.split()[0] if stripped.split() else ""
    if python_path and cmd == python_path:
        return True
    return "python" in cmd


def _override_tearsheet_output(line: str, tearsheet_pdf: Optional[str], tearsheet_dir: Optional[str]) -> str:
    if not tearsheet_dir or not tearsheet_pdf:
        return line
    if tearsheet_pdf not in line:
        return line
    new_pdf = str(Path(tearsheet_dir) / Path(tearsheet_pdf).name)
    return line.replace(tearsheet_pdf, new_pdf)


def _recipe_has_tearsheet(recipe: List[str], tearsheet_pdf: Optional[str]) -> bool:
    return any(_is_tearsheet_command(line, tearsheet_pdf) for line in recipe)


def build_run_plan(
    recipe: List[str],
    *,
    tearsheet_pdf: Optional[str],
    no_tearsheet: bool,
    tearsheet_only_top: Optional[int],
    tearsheet_dir: Optional[str],
    python_path: str,
) -> List[str]:
    plan: List[str] = []
    tearsheet_count = 0
    for line in recipe:
        if _should_skip_line(line):
            continue
        line = _rewrite_python_command(line, python_path)
        if _is_tearsheet_command(line, tearsheet_pdf):
            if no_tearsheet:
                continue
            if tearsheet_only_top is not None and tearsheet_count >= tearsheet_only_top:
                continue
            tearsheet_count += 1
            plan.append(_override_tearsheet_output(line, tearsheet_pdf, tearsheet_dir))
            continue
        plan.append(line)
    return plan


def cmd_list(_: argparse.Namespace) -> None:
    strategies = load_registry()
    rows: List[Dict[str, Any]] = []
    for s in strategies:
        rows.append(build_list_row(s))

    def status_rank(status: Optional[str]) -> int:
        return {"canonical": 0, "experiment": 1, "archived": 2}.get(status, 3)

    rows = sorted(rows, key=lambda r: (status_rank(r.get("status")), r.get("family"), r.get("id")))

    columns = ["id", "name", "family", "version", "status", "start", "end", "CAGR", "Sharpe", "MaxDD"]
    display_rows: List[List[str]] = []
    for row in rows:
        display_rows.append([str(row.get(col)) if row.get(col) is not None else "None" for col in columns])

    widths = [len(col) for col in columns]
    for row in display_rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    header = " ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    print(header)
    for row in display_rows:
        print(" ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def find_strategy(strategy_id: str) -> Dict[str, Any]:
    strategies = load_registry()
    for s in strategies:
        if s.get("id") == strategy_id:
            return s
    raise SystemExit(f"Strategy id not found: {strategy_id}")


def cmd_show(args: argparse.Namespace) -> None:
    s = find_strategy(args.id)
    m = load_metrics(s)
    start, end = resolve_period(s)
    print(f"id         : {s.get('id')}")
    print(f"name       : {s.get('name')}")
    print(f"family     : {s.get('family')}")
    print(f"version    : {s.get('version')}")
    print(f"status     : {s.get('status')}")
    print(f"universe   : {s.get('universe')}")
    print(f"equity_csv : {s.get('equity_csv')}")
    print(f"metrics_csv: {s.get('metrics_csv')}")
    print(f"tearsheet  : {s.get('tearsheet_pdf')}")
    print(f"period     : {start} → {end}")
    print(f"CAGR       : {fmt_pct(m['cagr'])}")
    print(f"Sharpe     : {fmt_num(m['sharpe'])}")
    print(f"MaxDD      : {fmt_pct(m['max_dd'])}")
    print(f"git_tag    : {s.get('git_tag')}")

    recipe = s.get("run_recipe", [])
    if recipe:
        print("\nRun recipe (canonical rerun):\n")
        print(f"  # from repo root")
        print(f"  cd {REPO_ROOT}")
        for line in recipe:
            print(f"  {line}")


def cmd_run(args: argparse.Namespace) -> None:
    s = find_strategy(args.id)
    recipe = s.get("run_recipe", [])
    if not recipe:
        raise SystemExit(f"No run_recipe defined for strategy {args.id}")

    no_tearsheet = args.no_tearsheet and not args.tearsheet
    plan = build_run_plan(
        recipe,
        tearsheet_pdf=s.get("tearsheet_pdf"),
        no_tearsheet=no_tearsheet,
        tearsheet_only_top=args.tearsheet_only_top,
        tearsheet_dir=args.tearsheet_dir,
        python_path=args.python,
    )

    print(f"Running strategy pipeline for: {args.id}")
    for line in plan:
        env = None
        if not args.no_pythonpath and _is_python_command(line, args.python):
            env = os.environ.copy()
            prefix = f".{os.pathsep}src"
            if env.get("PYTHONPATH"):
                env["PYTHONPATH"] = f"{prefix}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = prefix
        print(f"\n[RUN] {line}")
        result = subprocess.run(line, shell=True, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            raise SystemExit(f"Command failed with code {result.returncode}: {line}")
    print("\nDone.")


def _load_schema(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_status_fields(s: Dict[str, Any]) -> List[str]:
    status = s.get("status")
    errs: List[str] = []
    required_base = ["id", "family", "version", "status"]
    for f in required_base:
        if not s.get(f):
            errs.append(f"missing required field '{f}'")
    if status == "canonical":
        needed = ["equity_csv", "metrics_csv", "tearsheet_pdf", "run_recipe"]
    elif status == "incubation":
        needed = ["equity_csv", "metrics_csv", "run_recipe"]
    else:
        needed = []
    for f in needed:
        if not s.get(f):
            errs.append(f"missing {f} (required for status={status})")
    if status in {"canonical", "incubation"}:
        rr = s.get("run_recipe")
        if not isinstance(rr, list) or not all(isinstance(x, str) and x.strip() for x in rr) or len(rr) == 0:
            errs.append("run_recipe must be a non-empty list of commands")
        period = s.get("period") or s.get("canonical_period")
        if not period or not period.get("start") or not period.get("end"):
            errs.append(f"period.start/end required for status={status}")
        if status == "canonical":
            tearsheet_pdf = s.get("tearsheet_pdf")
            if not isinstance(tearsheet_pdf, str) or not tearsheet_pdf.strip():
                errs.append("tearsheet_pdf required for status=canonical")
            elif not _recipe_has_tearsheet(rr, tearsheet_pdf):
                errs.append("run_recipe must include a tearsheet generation step for canonical strategies")
    return errs


def cmd_validate(_: argparse.Namespace) -> None:
    strategies = load_registry()
    schema = _load_schema(SCHEMA_PATH)
    errors: List[str] = []

    # Optional jsonschema validation if available
    if schema:
        try:
            import jsonschema

            jsonschema.validate({"strategies": strategies}, schema)
        except ImportError:
            print("[registry] jsonschema not installed; skipping schema validation.")
        except Exception as e:
            errors.append(f"Schema validation failed: {e}")

    for s in strategies:
        sid = s.get("id", "<unknown>")
        errs = _validate_status_fields(s)
        for e in errs:
            errors.append(f"{sid}: {e}")

    if errors:
        print("Registry validation failed:")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)
    else:
        print("Registry validation OK.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy registry CLI (list/show/run/validate).")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all registered strategies with key metrics.")

    p_show = sub.add_parser("show", help="Show details & run recipe for a strategy.")
    p_show.add_argument("--id", required=True, help="Strategy id from registry.")

    p_run = sub.add_parser("run", help="Execute the run_recipe for a strategy.")
    p_run.add_argument("--id", required=True, help="Strategy id from registry.")
    p_run.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for run_recipe steps (default: current interpreter).",
    )
    p_run.add_argument(
        "--no-pythonpath",
        action="store_true",
        help="Do not inject PYTHONPATH=.",
    )
    g = p_run.add_mutually_exclusive_group()
    g.add_argument("--no-tearsheet", action="store_true", help="Skip tearsheet generation.")
    g.add_argument("--tearsheet", action="store_true", help="Force tearsheet generation.")
    p_run.add_argument(
        "--tearsheet-only-top",
        type=int,
        default=None,
        help="If multiple tearsheets are present in the recipe, run only the first N.",
    )
    p_run.add_argument(
        "--tearsheet-dir",
        type=str,
        default=None,
        help="Optional override directory for tearsheet output PDF filenames.",
    )

    sub.add_parser("validate", help="Validate registry against schema and status rules.")

    args = parser.parse_args()
    if args.cmd is None:
        args.cmd = "list"

    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "show":
        cmd_show(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

