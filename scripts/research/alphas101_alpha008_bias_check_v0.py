#!/usr/bin/env python
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bias check for alpha_008 (baseline vs lag1 IC).")
    p.add_argument(
        "--alphas",
        required=True,
        help="Path to alphas parquet (e.g. artifacts/research/101_alphas/alphas_101_v0.parquet)",
    )
    p.add_argument(
        "--db",
        required=True,
        help="Path to DuckDB file (e.g. ../data/coinbase_daily_121025.duckdb)",
    )
    p.add_argument(
        "--price_table",
        default="bars_1d_usd_universe_clean",
        help="Daily bars table/view in DuckDB (default: bars_1d_usd_universe_clean)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output CSV for IC summary (e.g. artifacts/research/101_alphas/alphas101_alpha008_bias_check_v0.csv)",
    )
    return p.parse_args()


def daily_ic(merged: pd.DataFrame, sig_col: str) -> pd.Series:
    by_ts = merged.groupby("ts")

    def _cs_corr(g: pd.DataFrame) -> float:
        x = g[sig_col]
        y = g["fwd_ret"]
        if x.notna().sum() < 2 or y.notna().sum() < 2:
            return np.nan
        return x.corr(y)

    ic = by_ts.apply(_cs_corr).dropna()
    return ic


def main() -> None:
    args = parse_args()
    alphas_path = Path(args.alphas)
    db_path = Path(args.db)
    out_path = Path(args.out)

    print(f"[alpha008_bias_check] Loading alphas from {alphas_path}")
    alphas = pd.read_parquet(alphas_path)
    if "ts" not in alphas.columns or "symbol" not in alphas.columns:
        raise ValueError("Expected 'ts' and 'symbol' columns in alphas parquet.")
    if "alpha_008" not in alphas.columns:
        raise ValueError("Expected 'alpha_008' in alphas parquet.")

    alphas["ts"] = pd.to_datetime(alphas["ts"])

    start_ts = alphas["ts"].min()
    end_ts = alphas["ts"].max() + pd.Timedelta(days=1)

    print(f"[alpha008_bias_check] Fetching prices from {db_path}")
    con = duckdb.connect(str(db_path))
    prices = con.execute(
        f"""
        SELECT
            symbol,
            ts,
            LEAD(close) OVER (PARTITION BY symbol ORDER BY ts) / close - 1.0 AS fwd_ret
        FROM {args.price_table}
        WHERE ts BETWEEN ? AND ?
        """,
        [start_ts, end_ts],
    ).df()
    prices["ts"] = pd.to_datetime(prices["ts"])

    merged = alphas.merge(prices, on=["symbol", "ts"], how="inner")
    merged = merged.dropna(subset=["fwd_ret"])
    if merged.empty:
        raise ValueError("No overlap between alphas and prices after merge.")

    merged["sig_base"] = merged["alpha_008"]
    ic_base = daily_ic(merged, "sig_base")

    merged["sig_lag1"] = (
        merged.sort_values(["symbol", "ts"])
        .groupby("symbol")["alpha_008"]
        .shift(1)
    )
    ic_lag1 = daily_ic(merged.dropna(subset=["sig_lag1"]), "sig_lag1")

    def summarize(ic: pd.Series, label: str) -> dict:
        n = len(ic)
        mean_ic = ic.mean() if n > 0 else np.nan
        std_ic = ic.std(ddof=0) if n > 1 else np.nan
        tstat = mean_ic / (std_ic / np.sqrt(n)) if (std_ic is not None and std_ic > 0 and n > 1) else np.nan
        return dict(label=label, n_days=n, mean_ic=mean_ic, std_ic=std_ic, tstat_ic=tstat)

    rows = [
        summarize(ic_base, "alpha_008_baseline"),
        summarize(ic_lag1, "alpha_008_lag1"),
    ]
    out_df = pd.DataFrame.from_records(rows)
    out_df.to_csv(out_path, index=False)
    print(f"[alpha008_bias_check] Wrote IC summary to {out_path}")


if __name__ == "__main__":
    main()

