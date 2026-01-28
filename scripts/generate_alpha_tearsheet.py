#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import duckdb
import polars as pl

from analysis.tearsheet import generate_tearsheet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate institutional alpha tearsheet")
    p.add_argument("--alphas", required=True, help="Alpha panel parquet (ts, symbol, alpha_*)")
    p.add_argument("--alpha", required=True, help="Alpha column to analyze")
    p.add_argument("--db", default=None, help="DuckDB path (required unless --joined)")
    p.add_argument("--price_table", default="bars_1d_clean", help="Price table/view")
    p.add_argument("--horizon", type=int, default=1, help="Forward return horizon")
    p.add_argument("--symbols", default=None, help="Comma-separated symbol list")
    p.add_argument("--n-quantiles", type=int, default=5, help="Requested quantiles")
    p.add_argument("--emit-returns", action="store_true", help="Emit spread returns parquet")
    p.add_argument("--joined", default=None, help="Optional pre-joined parquet with signal/forward_ret")
    p.add_argument(
        "--output",
        default=None,
        help="Output path prefix (default: artifacts/.../tearsheets/{alpha})",
    )
    return p.parse_args()


def _normalize_ts(df: pl.DataFrame) -> pl.DataFrame:
    dtype = df.schema.get("ts")
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return df.with_columns(
            pl.col("ts").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    return df.with_columns(pl.col("ts").dt.replace_time_zone(None))


def _alpha_panel_signal(alphas_path: str, alpha_name: str, symbols: list[str] | None) -> pl.DataFrame:
    panel = pl.read_parquet(alphas_path)
    if "ts" not in panel.columns or "symbol" not in panel.columns:
        raise ValueError("Alpha panel must include 'ts' and 'symbol'.")
    if alpha_name not in panel.columns:
        raise ValueError(f"Alpha column {alpha_name} not found in {alphas_path}")
    panel = panel.select(["ts", "symbol", pl.col(alpha_name).alias("signal")])
    if symbols:
        panel = panel.filter(pl.col("symbol").is_in(symbols))
    return _normalize_ts(panel)


def _forward_returns_from_duckdb(
    db_path: str,
    table: str,
    start_ts,
    end_ts,
    symbols: list[str] | None,
) -> pl.DataFrame:
    symbol_clause = ""
    params = [start_ts, end_ts]
    if symbols:
        symbol_list = ", ".join([f"'{s}'" for s in symbols])
        symbol_clause = f" AND symbol IN ({symbol_list})"
    price_sql = f"""
        SELECT
            symbol,
            ts,
            LEAD(close) OVER (PARTITION BY symbol ORDER BY ts) / close - 1.0 AS forward_ret
        FROM {table}
        WHERE ts BETWEEN ? AND ?{symbol_clause}
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        arrow = con.execute(price_sql, params).fetch_arrow_table()
    finally:
        con.close()
    df = pl.from_arrow(arrow)
    return _normalize_ts(df)


def build_alpha_tearsheet_frame(
    *,
    alphas_path: str,
    alpha_name: str,
    db_path: str,
    price_table: str,
    symbols: list[str] | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if not Path(db_path).exists():
        raise ValueError(f"DuckDB file does not exist: {db_path}")

    alpha_df = _alpha_panel_signal(alphas_path, alpha_name, symbols)
    ts_min = alpha_df.select(pl.col("ts").min()).item()
    ts_max = alpha_df.select(pl.col("ts").max()).item()
    end_ts = ts_max + timedelta(days=1)

    prices = _forward_returns_from_duckdb(db_path, price_table, ts_min, end_ts, symbols)

    merged = alpha_df.join(prices, on=["ts", "symbol"], how="inner").drop_nulls(
        subset=["forward_ret"]
    )
    return alpha_df, prices, merged


def main() -> None:
    args = parse_args()

    if args.joined:
        joined = pl.read_parquet(args.joined)
        if not {"ts", "symbol", "signal", "forward_ret"}.issubset(set(joined.columns)):
            raise ValueError("Joined parquet must include ts, symbol, signal, forward_ret")
        df = _normalize_ts(joined)
    else:
        if not args.db:
            raise ValueError("--db is required unless --joined is provided")
        symbol_list = (
            [s.strip() for s in args.symbols.split(",") if s.strip()]
            if args.symbols
            else None
        )
        alpha_df, prices_df, df = build_alpha_tearsheet_frame(
            alphas_path=args.alphas,
            alpha_name=args.alpha,
            db_path=args.db,
            price_table=args.price_table,
            symbols=symbol_list,
        )

        def _range_info(frame: pl.DataFrame) -> tuple[int, str, str]:
            if frame.is_empty():
                return 0, "n/a", "n/a"
            start = frame.select(pl.col("ts").min()).item()
            end = frame.select(pl.col("ts").max()).item()
            return frame.height, str(start), str(end)

        alpha_rows, alpha_min, alpha_max = _range_info(alpha_df)
        price_rows, price_min, price_max = _range_info(prices_df)
        merged_rows, merged_min, merged_max = _range_info(df)

        null_signal = alpha_df.select(pl.col("signal").is_null().sum()).item()
        null_fwd = prices_df.select(pl.col("forward_ret").is_null().sum()).item()

        print("[generate_alpha_tearsheet] Inputs")
        print(f"  alpha parquet: {args.alphas}")
        print(f"  db: {args.db}")
        print(f"  table: {args.price_table}")
        print(f"  alpha: {args.alpha}")
        print(f"  alpha df: rows={alpha_rows}, ts_min={alpha_min}, ts_max={alpha_max}")
        print(f"  prices df: rows={price_rows}, ts_min={price_min}, ts_max={price_max}")
        print(f"  merged df: rows={merged_rows}, ts_min={merged_min}, ts_max={merged_max}")
        print(f"  nulls: signal={null_signal}, forward_ret={null_fwd}")

        if df.is_empty():
            raise ValueError(
                "No overlap between alpha panel and forward returns. "
                "Check db path/table/date range/timezone alignment."
            )

    output = args.output
    if output is None:
        output = f"artifacts/research/formulaic_alphas/tearsheets/{args.alpha}"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    n_quantiles = args.n_quantiles
    if args.symbols:
        symbol_count = len(symbol_list or [])
        if symbol_count > 0 and n_quantiles > symbol_count:
            n_quantiles = max(2, symbol_count)
            print(
                f"[generate_alpha_tearsheet] Only {symbol_count} symbols provided; "
                f"using effective_quantiles={n_quantiles}."
            )

    generate_tearsheet(
        df,
        output,
        alpha_name=args.alpha,
        n_quantiles=n_quantiles,
        emit_returns=args.emit_returns,
    )
    print(f"Wrote tearsheet to {output}.pdf/.png/.json")


if __name__ == "__main__":
    main()
