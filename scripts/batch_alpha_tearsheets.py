#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from analysis.tearsheet import generate_tearsheet


@dataclass(frozen=True)
class GatekeeperResult:
    alpha: str
    mean_ic: float
    spread_sharpe: float
    monotonic_pass: bool
    mean_turnover: float
    verdict: str
    reason: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch gatekeeper + tearsheets for alpha panels")
    p.add_argument("--alphas", required=True, help="Alpha panel parquet (ts, symbol, alpha_*)")
    p.add_argument("--db", required=True, help="DuckDB path")
    p.add_argument("--price_table", default="bars_1d_clean", help="Price table/view")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--horizon", type=int, default=1, help="Forward return horizon in days")
    p.add_argument("--max_alphas", type=int, default=None, help="Optional cap for debugging")
    p.add_argument("--top_k", type=int, default=None, help="Optional top-k pre-screen")
    p.add_argument("--emit-returns", action="store_true", help="Emit spread returns parquet")
    return p.parse_args()


def _normalize_ts(df: pl.DataFrame) -> pl.DataFrame:
    dtype = df.schema.get("ts")
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return df.with_columns(
            pl.col("ts").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    return df.with_columns(pl.col("ts").dt.replace_time_zone(None))


def load_alpha_panel(path: str) -> tuple[pl.DataFrame, list[str]]:
    df = pl.read_parquet(path)
    if "ts" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Alpha panel must include 'ts' and 'symbol'.")
    alpha_cols = sorted([c for c in df.columns if c.startswith("alpha_")])
    if not alpha_cols:
        raise ValueError("No alpha_* columns found in panel.")
    df = _normalize_ts(df.select(["ts", "symbol"] + alpha_cols))
    return df, alpha_cols


def load_forward_returns(
    db_path: str,
    table: str,
    start_ts,
    end_ts,
) -> pl.DataFrame:
    price_sql = f"""
        SELECT
            symbol,
            ts,
            LEAD(close) OVER (PARTITION BY symbol ORDER BY ts) / close - 1.0 AS forward_ret
        FROM {table}
        WHERE ts BETWEEN ? AND ?
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        arrow = con.execute(price_sql, [start_ts, end_ts]).fetch_arrow_table()
    finally:
        con.close()
    df = pl.from_arrow(arrow)
    return _normalize_ts(df)


def merge_alpha_returns(alpha_df: pl.DataFrame, returns_df: pl.DataFrame) -> pl.DataFrame:
    merged = alpha_df.join(returns_df, on=["ts", "symbol"], how="inner")
    merged = merged.drop_nulls(subset=["forward_ret"])
    if merged.is_empty():
        raise ValueError("No overlap between alphas and price data after join.")
    return merged


def infer_periods_per_year(ts: pl.Series) -> float:
    if ts.len() < 2:
        return 365.0
    diffs = ts.sort().diff().dt.total_seconds().drop_nulls()
    if diffs.len() == 0:
        return 365.0
    dt_seconds = diffs.median()
    if not dt_seconds or dt_seconds <= 0:
        return 365.0
    return float(365 * 24 * 3600 / dt_seconds)


def _quantile_frame(
    df: pl.DataFrame,
    n_quantiles: int,
) -> tuple[pl.DataFrame | None, dict[str, float], int, str]:
    counts = df.group_by("ts").len().rename({"len": "n_symbols"})
    min_symbols = counts.select(pl.col("n_symbols").min()).item()
    if min_symbols < n_quantiles:
        return None, {}, int(min_symbols or 0), "insufficient_symbols"

    df = df.join(counts, on="ts", how="left")
    rank = pl.col("signal").rank(method="average").over("ts")
    rank_pct = pl.when(pl.col("n_symbols") > 1).then(
        (rank - 1) / (pl.col("n_symbols") - 1)
    ).otherwise(0.0)
    quantile = (
        (rank_pct * n_quantiles)
        .floor()
        .clip(0, n_quantiles - 1)
        .cast(pl.Int64)
        .alias("quantile")
    )

    df = df.with_columns([rank_pct.alias("rank_pct"), quantile])
    quantile_returns = (
        df.group_by(["ts", "quantile"])
        .agg(pl.col("forward_ret").mean().alias("ret"))
        .sort(["ts", "quantile"])
        .with_columns(pl.col("quantile").cast(pl.Utf8).alias("q_label"))
    )

    quantile_wide = quantile_returns.pivot(
        values="ret", index="ts", on="q_label", aggregate_function="first"
    ).sort("ts")

    quantile_cols = []
    for q in range(n_quantiles):
        col = str(q)
        if col in quantile_wide.columns:
            quantile_wide = quantile_wide.rename({col: f"Q{q + 1}"})
        quantile_cols.append(f"Q{q + 1}")

    for col in quantile_cols:
        if col not in quantile_wide.columns:
            quantile_wide = quantile_wide.with_columns(pl.lit(0.0).alias(col))

    quantile_wide = quantile_wide.with_columns(
        (pl.col(f"Q{n_quantiles}") - pl.col("Q1")).alias("Spread")
    )

    avg_ret_by_quantile = {
        col: float(quantile_wide.select(pl.col(col).mean()).item())
        for col in quantile_cols
    }

    return quantile_wide, avg_ret_by_quantile, int(min_symbols or 0), "ok"


def compute_gatekeeper_metrics(
    df: pl.DataFrame,
    alpha: str,
    n_quantiles: int,
) -> tuple[GatekeeperResult, dict[str, float]]:
    sub = df.select(["ts", "symbol", pl.col(alpha).alias("signal"), "forward_ret"])

    quantile_wide, avg_ret_by_quantile, min_symbols, status = _quantile_frame(
        sub, n_quantiles
    )
    if status != "ok":
        result = GatekeeperResult(
            alpha=alpha,
            mean_ic=0.0,
            spread_sharpe=0.0,
            monotonic_pass=False,
            mean_turnover=0.0,
            verdict="FAIL",
            reason=f"insufficient_symbols_min={min_symbols}",
        )
        return result, avg_ret_by_quantile

    periods_per_year = infer_periods_per_year(quantile_wide["ts"])

    spread = quantile_wide["Spread"].drop_nulls()
    spread_mean = spread.mean() if spread.len() > 0 else 0.0
    spread_std = spread.std(ddof=1) if spread.len() > 1 else 0.0
    spread_sharpe = (
        (spread_mean / spread_std) * (periods_per_year ** 0.5)
        if spread_std and spread_std > 0
        else 0.0
    )

    monotonic_pass = (
        avg_ret_by_quantile.get("Q5", 0.0)
        > avg_ret_by_quantile.get("Q3", 0.0)
        > avg_ret_by_quantile.get("Q1", 0.0)
    )

    # Turnover
    counts = sub.group_by("ts").len().rename({"len": "n_symbols"})
    sub = sub.join(counts, on="ts", how="left")
    rank = pl.col("signal").rank(method="average").over("ts")
    rank_pct = pl.when(pl.col("n_symbols") > 1).then(
        (rank - 1) / (pl.col("n_symbols") - 1)
    ).otherwise(0.0)
    quantile = (
        (rank_pct * n_quantiles)
        .floor()
        .clip(0, n_quantiles - 1)
        .cast(pl.Int64)
        .alias("quantile")
    )
    sub = sub.with_columns([rank_pct.alias("rank_pct"), quantile])

    n_long = (
        sub.filter(pl.col("quantile") == n_quantiles - 1)
        .group_by("ts")
        .len()
        .rename({"len": "n_long"})
    )
    n_short = (
        sub.filter(pl.col("quantile") == 0)
        .group_by("ts")
        .len()
        .rename({"len": "n_short"})
    )
    sub = sub.join(n_long, on="ts", how="left").join(n_short, on="ts", how="left")
    sub = sub.with_columns(
        pl.when(pl.col("quantile") == n_quantiles - 1)
        .then(1.0 / pl.col("n_long"))
        .when(pl.col("quantile") == 0)
        .then(-1.0 / pl.col("n_short"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("weight")
    )
    sub = sub.sort(["symbol", "ts"]).with_columns(
        pl.col("weight").shift(1).over("symbol").fill_null(0.0).alias("weight_prev")
    )
    turnover = (
        sub.with_columns((pl.col("weight") - pl.col("weight_prev")).abs().alias("dw"))
        .group_by("ts")
        .agg((0.5 * pl.col("dw").sum()).alias("turnover"))
        .sort("ts")
    )
    mean_turnover = turnover["turnover"].mean() if turnover.height > 0 else 0.0

    # Spearman IC
    sub = sub.with_columns(
        [
            pl.col("signal").rank(method="average").over("ts").alias("signal_rank"),
            pl.col("forward_ret").rank(method="average").over("ts").alias("ret_rank"),
        ]
    )
    ic = (
        sub.group_by("ts")
        .agg(pl.corr(pl.col("signal_rank"), pl.col("ret_rank")).alias("ic"))
        .sort("ts")
    )
    ic_series = ic["ic"].drop_nulls()
    mean_ic = ic_series.mean() if ic_series.len() > 0 else 0.0

    verdict = "PASS"
    reason = "ok"
    if mean_ic <= 0.02:
        verdict = "FAIL"
        reason = "mean_ic"
    if spread_sharpe <= 1.0:
        verdict = "FAIL"
        reason = "spread_sharpe"
    if not monotonic_pass:
        verdict = "FAIL"
        reason = "monotonicity"
    if mean_turnover >= 0.40 and spread_sharpe <= 2.5:
        verdict = "FAIL"
        reason = "turnover"

    return (
        GatekeeperResult(
            alpha=alpha,
            mean_ic=float(mean_ic),
            spread_sharpe=float(spread_sharpe),
            monotonic_pass=bool(monotonic_pass),
            mean_turnover=float(mean_turnover),
            verdict=verdict,
            reason=reason,
        ),
        avg_ret_by_quantile,
    )


def write_gatekeeper_csv(path: Path, rows: list[GatekeeperResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha",
            "mean_ic",
            "spread_sharpe",
            "monotonic_pass",
            "mean_daily_turnover",
            "verdict",
            "reason",
        ])
        for r in rows:
            writer.writerow(
                [
                    r.alpha,
                    f"{r.mean_ic:.6f}",
                    f"{r.spread_sharpe:.6f}",
                    str(r.monotonic_pass),
                    f"{r.mean_turnover:.6f}",
                    r.verdict,
                    r.reason,
                ]
            )


def write_survivor_corr(df: pl.DataFrame, survivors: list[str], out_path: Path) -> None:
    if len(survivors) < 2:
        return
    counts = df.group_by("ts").len().rename({"len": "n_symbols"})
    df = df.join(counts, on="ts", how="left")
    norm_cols = []
    for alpha in survivors:
        rank = pl.col(alpha).rank(method="average").over("ts")
        rank_pct = pl.when(pl.col("n_symbols") > 1).then(
            (rank - 1) / (pl.col("n_symbols") - 1)
        ).otherwise(0.0)
        norm_cols.append(rank_pct.alias(alpha))
    norm = df.select(["ts", "symbol"] + norm_cols)

    rows = []
    for a in survivors:
        row = {"alpha": a}
        for b in survivors:
            corr = norm.select(pl.corr(pl.col(a), pl.col(b)).alias("corr")).item()
            row[b] = f"{corr:.6f}" if corr is not None else ""
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha"] + survivors)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    alpha_df, alpha_cols = load_alpha_panel(args.alphas)
    if args.max_alphas is not None:
        alpha_cols = alpha_cols[: args.max_alphas]

    ts_min = alpha_df.select(pl.col("ts").min()).item()
    ts_max = alpha_df.select(pl.col("ts").max()).item()
    end_ts = ts_max + timedelta(days=args.horizon)

    returns_df = load_forward_returns(args.db, args.price_table, ts_min, end_ts)
    merged = merge_alpha_returns(alpha_df, returns_df)

    results: list[GatekeeperResult] = []
    avg_quantiles: dict[str, dict[str, float]] = {}

    for alpha in alpha_cols:
        metrics, avg_ret = compute_gatekeeper_metrics(merged, alpha, n_quantiles=5)
        results.append(metrics)
        avg_quantiles[alpha] = avg_ret

    if args.top_k:
        ranked = sorted(results, key=lambda r: r.mean_ic, reverse=True)
        top_set = {r.alpha for r in ranked[: args.top_k]}
    else:
        top_set = set(a.alpha for a in results)

    survivors = [r.alpha for r in results if r.verdict == "PASS" and r.alpha in top_set]

    out_dir = Path(args.out_dir)
    write_gatekeeper_csv(out_dir / "gatekeeper_all.csv", results)
    write_gatekeeper_csv(out_dir / "gatekeeper_survivors.csv", [r for r in results if r.verdict == "PASS"])

    for alpha in survivors:
        df_alpha = merged.select(["ts", "symbol", pl.col(alpha).alias("signal"), "forward_ret"])
        out_path = out_dir / "survivors" / alpha / alpha
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generate_tearsheet(
            df_alpha,
            str(out_path),
            alpha_name=alpha,
            n_quantiles=5,
            emit_returns=args.emit_returns,
        )

    if len(survivors) >= 2:
        corr_path = out_dir / "survivors" / "survivor_corr.csv"
        write_survivor_corr(merged.select(["ts", "symbol"] + survivors), survivors, corr_path)

    print(f"n_total={len(results)}")
    print(f"n_pass={len([r for r in results if r.verdict == 'PASS'])}")
    print(f"n_fail={len([r for r in results if r.verdict == 'FAIL'])}")
    if survivors:
        print("passing alphas:")
        for name in survivors:
            print(f"  {name}")


if __name__ == "__main__":
    main()
