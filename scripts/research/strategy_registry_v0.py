#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any, Dict, List, Optional
import sys

import pandas as pd

REGISTRY_PATH = "docs/research/strategy_registry_v0.json"
REPO_ROOT = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto"
SCHEMA_PATH = "docs/research/strategy_registry_v0.schema.json"


def load_registry(path: Optional[str] = None) -> List[Dict[str, Any]]:
    use_path = path or REGISTRY_PATH
    with open(use_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("strategies", [])


def load_metrics(strategy: Dict[str, Any]) -> Dict[str, float]:
    metrics_csv = strategy.get("metrics_csv")
    period_key = strategy.get("metrics_period", "full")
    if not metrics_csv or not os.path.exists(metrics_csv):
        return {"cagr": None, "sharpe": None, "max_dd": None}

    df = pd.read_csv(metrics_csv)
    # If no period column, try matching on label (e.g., base_vs_growth), else first row.
    if "period" in df.columns:
        if period_key in df["period"].values:
            row = df.loc[df["period"] == period_key].iloc[0]
        else:
            row = df.iloc[0]
    elif "label" in df.columns and period_key in df["label"].values:
        row = df.loc[df["label"] == period_key].iloc[0]
    else:
        row = df.iloc[0]

    return {
        "cagr": float(row.get("cagr", float("nan"))),
        "sharpe": float(row.get("sharpe", float("nan"))),
        "max_dd": float(row.get("max_dd", float("nan"))),
    }


def fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x * 100:.2f}%"


def fmt_num(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2f}"


def cmd_list(_: argparse.Namespace) -> None:
    strategies = load_registry()
    rows = []
    for s in strategies:
        m = load_metrics(s)
        period = s.get("canonical_period", {})
        rows.append(
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "family": s.get("family"),
                "version": s.get("version"),
                "status": s.get("status"),
                "start": period.get("start"),
                "end": period.get("end"),
                "CAGR": fmt_pct(m["cagr"]),
                "Sharpe": fmt_num(m["sharpe"]),
                "MaxDD": fmt_pct(m["max_dd"]),
            }
        )

    df = pd.DataFrame(rows)
    df["status_rank"] = df["status"].map({"canonical": 0, "experiment": 1, "archived": 2}).fillna(3)
    df = df.sort_values(["status_rank", "family", "id"])
    df = df.drop(columns=["status_rank"])
    print(df.to_string(index=False))


def find_strategy(strategy_id: str) -> Dict[str, Any]:
    strategies = load_registry()
    for s in strategies:
        if s.get("id") == strategy_id:
            return s
    raise SystemExit(f"Strategy id not found: {strategy_id}")


def cmd_show(args: argparse.Namespace) -> None:
    s = find_strategy(args.id)
    m = load_metrics(s)
    period = s.get("canonical_period", {})
    print(f"id         : {s.get('id')}")
    print(f"name       : {s.get('name')}")
    print(f"family     : {s.get('family')}")
    print(f"version    : {s.get('version')}")
    print(f"status     : {s.get('status')}")
    print(f"universe   : {s.get('universe')}")
    print(f"equity_csv : {s.get('equity_csv')}")
    print(f"metrics_csv: {s.get('metrics_csv')}")
    print(f"tearsheet  : {s.get('tearsheet_pdf')}")
    print(f"period     : {period.get('start')} → {period.get('end')}")
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

    print(f"Running strategy pipeline for: {args.id}")
    for line in recipe:
        if not line.strip() or line.strip().startswith("#"):
            continue
        print(f"\n[RUN] {line}")
        result = subprocess.run(line, shell=True, cwd=REPO_ROOT)
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
        needed = ["equity_csv", "metrics_csv", "tearsheet_pdf", "run_recipe"]
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

