#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from alphas.adapters import to_alphas_panel
from alphas.compiler import compile_formulas
from alphas.factory import build_alpha_panel, write_outputs
from alphas.parser import load_alphas_file
from utils.duckdb_inspect import list_tables


DEFAULT_ALPHA_FILE = "alphas.txt"
DEFAULT_OUT_PARQUET = "artifacts/research/formulaic_alphas/alphas_formulaic_v0.parquet"
DEFAULT_META_JSON = "artifacts/research/formulaic_alphas/alphas_formulaic_v0.meta.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Formulaic Alpha Engine v0 runner")
    p.add_argument("--db", required=True, help="DuckDB path")
    p.add_argument("--table", default=None, help="Price table/view")
    p.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow fallback table when requested table is missing",
    )
    p.add_argument(
        "--fallback-table",
        default="bars_1d_clean",
        help="Fallback table when requested table is missing",
    )
    p.add_argument("--alphas-file", default=DEFAULT_ALPHA_FILE, help="Alpha formula file")
    p.add_argument("--start", default=None, help="Start timestamp")
    p.add_argument("--end", default=None, help="End timestamp")
    p.add_argument("--symbols", default=None, help="Comma-separated symbol list")
    p.add_argument("--out-parquet", default=DEFAULT_OUT_PARQUET, help="Output parquet path")
    p.add_argument("--out-meta", default=DEFAULT_META_JSON, help="Output metadata JSON path")
    p.add_argument("--run-ic", action="store_true", help="Run IC panel script on output parquet")
    return p.parse_args()


def resolve_price_table(
    db_path: str,
    requested: str,
    fallback: str = "bars_1d_clean",
    allow_fallback: bool = False,
) -> str:
    tables = list_tables(db_path)
    if requested in tables:
        return requested

    candidates = [
        t
        for t in tables
        if t.startswith(("bars_1d", "bars_4h", "hourly_bars", "bars_"))
    ]
    candidates_display = ", ".join(candidates[:10]) if candidates else "none"
    msg = (
        f"Requested table '{requested}' not found in {db_path}.\n"
        f"Candidate tables (first 10): {candidates_display}\n"
        "If you need ADV10m, run:\n"
        f"  python scripts/research/create_usd_universe_adv10m_view.py --db {db_path}\n"
    )
    if allow_fallback:
        print(msg.strip())
        print(f"[run_alpha_factory] Falling back to '{fallback}'.")
        return fallback
    raise SystemExit(msg)


def resolve_table(db_path: str, requested: str | None) -> str:
    if requested:
        return requested
    tables = list_tables(db_path)
    if "bars_1d_usd_universe_clean_adv10m" in tables:
        return "bars_1d_usd_universe_clean_adv10m"
    if "bars_1d_clean" in tables:
        return "bars_1d_clean"
    for candidate in tables:
        if candidate.startswith("bars_"):
            return candidate
    raise ValueError("No bars_* tables found; pass --table explicitly.")


def main() -> None:
    args = parse_args()
    db_path = args.db
    if args.table:
        table = resolve_price_table(
            db_path,
            args.table,
            fallback=args.fallback_table,
            allow_fallback=args.allow_fallback,
        )
    else:
        table = resolve_table(db_path, args.table)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    formulas = load_alphas_file(args.alphas_file)
    plan = compile_formulas(formulas)

    df, meta = build_alpha_panel(
        db_path=db_path,
        table=table,
        plan=plan,
        start=args.start,
        end=args.end,
        symbols=symbols,
    )

    panel = to_alphas_panel(df)
    write_outputs(panel, meta, args.out_parquet, args.out_meta)

    print("=" * 72)
    print("Formulaic Alpha Engine v0")
    print("=" * 72)
    print(f"n_alphas: {len(formulas)}")
    print(f"n_rows: {panel.height}")
    print(f"warmup_max: {max(plan.warmup_bars.values()) if plan.warmup_bars else 0}")
    print(f"out_parquet: {args.out_parquet}")
    print(f"out_meta: {args.out_meta}")

    if args.run_ic:
        out_csv = Path(args.out_parquet).with_suffix("").with_name("alphas_formulaic_v0_ic_panel.csv")
        cmd = [
            sys.executable,
            "scripts/research/alphas101_ic_panel_v0.py",
            "--alphas",
            args.out_parquet,
            "--db",
            db_path,
            "--price_table",
            table,
            "--out_csv",
            str(out_csv),
        ]
        subprocess.run(cmd, check=True)
        print(f"IC panel written: {out_csv}")


if __name__ == "__main__":
    main()
