from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl

from alphas.compiler import ExecutionPlan


@dataclass(frozen=True)
class AlphaMeta:
    formulas: dict[str, str]
    warmup_bars: dict[str, int]
    price_table: str
    start: str | None
    end: str | None
    n_rows: int
    dropped_rows: dict[str, int]

    def to_json(self) -> dict:
        return {
            "formulas": self.formulas,
            "warmup_bars": self.warmup_bars,
            "price_table": self.price_table,
            "start": self.start,
            "end": self.end,
            "n_rows": self.n_rows,
            "dropped_rows": self.dropped_rows,
        }


def load_price_data(
    db_path: str,
    table: str,
    start: str | None = None,
    end: str | None = None,
    symbols: Iterable[str] | None = None,
) -> pl.DataFrame:
    filters = []
    if start:
        filters.append(f"ts >= '{start}'")
    if end:
        filters.append(f"ts <= '{end}'")
    if symbols:
        symbols_list = ",".join(f"'{s}'" for s in symbols)
        filters.append(f"symbol IN ({symbols_list})")
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    query = f"SELECT * FROM {table} {where_clause}"
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = pl.read_database(query, con)
    finally:
        con.close()

    return df


def validate_input_df(df: pl.DataFrame) -> None:
    required = {"ts", "symbol", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dupes = (
        df.group_by(["ts", "symbol"]).len().filter(pl.col("len") > 1)
    )
    if dupes.height > 0:
        raise ValueError("Duplicate (ts, symbol) rows detected.")

    counts = df.group_by("ts").len()
    if counts.filter(pl.col("len") < 2).height > 0:
        raise ValueError("Each timestamp must have at least 2 symbols for cross-sectional ops.")


def execute_plan(df: pl.DataFrame, plan: ExecutionPlan) -> pl.DataFrame:
    df = df.with_columns(plan.stage1_exprs)
    df = df.with_columns(plan.stage2_exprs)
    return df


def apply_warmup_policy(df: pl.DataFrame, warmup_bars: dict[str, int]) -> tuple[pl.DataFrame, dict[str, int]]:
    row_idx = (pl.col("ts").cum_count().over("symbol") - 1).alias("_row_in_symbol")
    df = df.with_columns(row_idx)

    dropped_counts: dict[str, int] = {}
    for name, warmup in warmup_bars.items():
        if name not in df.columns:
            continue
        mask = pl.col("_row_in_symbol") < warmup
        dropped_counts[name] = int(df.select(mask.cast(pl.Int64).sum()).item())
        df = df.with_columns(
            pl.when(mask)
            .then(None)
            .otherwise(pl.col(name))
            .alias(name)
        )

    for name in warmup_bars.keys():
        if name not in df.columns:
            continue
        df = df.with_columns(
            pl.when(pl.col(name).is_finite())
            .then(pl.col(name))
            .otherwise(0.0)
            .fill_null(0.0)
            .alias(name)
        )

    df = df.drop(["_row_in_symbol"])
    return df, dropped_counts


def build_alpha_panel(
    db_path: str,
    table: str,
    plan: ExecutionPlan,
    start: str | None = None,
    end: str | None = None,
    symbols: Iterable[str] | None = None,
) -> tuple[pl.DataFrame, AlphaMeta]:
    df = load_price_data(db_path, table, start=start, end=end, symbols=symbols)
    validate_input_df(df)

    df = execute_plan(df, plan)
    df, dropped = apply_warmup_policy(df, plan.warmup_bars)

    df = df.sort(["ts", "symbol"])
    meta = AlphaMeta(
        formulas=plan.formulas,
        warmup_bars=plan.warmup_bars,
        price_table=table,
        start=start,
        end=end,
        n_rows=df.height,
        dropped_rows=dropped,
    )
    return df, meta


def write_outputs(df: pl.DataFrame, meta: AlphaMeta, out_parquet: str, out_meta: str) -> None:
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    meta_path = Path(out_meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta.to_json(), f, indent=2, sort_keys=True)
