#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-alpha IC panel (cross-sectional IC vs forward returns)."
    )
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
        help="Daily bars table/view (default: bars_1d_usd_universe_clean)",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forward return horizon in days (default: 1)",
    )
    p.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path for IC panel summary.",
    )
    p.add_argument(
        "--filtered_label",
        default="filtered_v1",
        help="Label to indicate filtered IC panel (e.g., ghost-data filtered).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    alphas_path = Path(args.alphas)
    db_path = Path(args.db)
    out_path = Path(args.out_csv)

    # Load alphas
    df = pd.read_parquet(alphas_path)
    if "ts" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Alphas parquet must contain 'ts' and 'symbol' columns.")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["ts", "symbol"])

    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    if not alpha_cols:
        raise ValueError("No alpha_* columns found in alphas parquet.")

    # Load prices and compute fwd returns (ghost-data filter)
    con = duckdb.connect(str(db_path))
    prices = con.execute(
        f"""
        SELECT ts, symbol, close, volume
        FROM {args.price_table}
        """
    ).df()
    con.close()

    prices["ts"] = pd.to_datetime(prices["ts"])
    prices = prices.sort_values(["symbol", "ts"])
    prices["prev_close"] = prices.groupby("symbol")["close"].shift(1)
    valid = (prices["volume"] > 0) & (prices["close"] != prices["prev_close"])
    prices_clean = prices.loc[valid].copy()
    prices_clean["fwd_ret"] = (
        prices_clean.groupby("symbol")["close"]
        .pct_change(periods=args.horizon)
        .shift(-args.horizon)
    )
    prices_clean = prices_clean.dropna(subset=["fwd_ret"])[["ts", "symbol", "fwd_ret"]]

    merged = df.merge(prices_clean, on=["ts", "symbol"], how="inner")
    if merged.empty:
        raise ValueError("No overlap between alphas and prices after merge.")

    rows = []
    for name in alpha_cols:
        sub = merged[["ts", name, "fwd_ret"]].dropna(subset=[name, "fwd_ret"])
        if sub.empty:
            continue

        def _cs_ic(g: pd.DataFrame) -> float:
            x = g[name]
            y = g["fwd_ret"]
            if x.count() < 5 or y.count() < 5:
                return np.nan
            return x.corr(y)

        ic_series = sub.groupby("ts").apply(_cs_ic).dropna()
        n_days = ic_series.shape[0]
        if n_days == 0:
            continue
        mean_ic = ic_series.mean()
        std_ic = ic_series.std(ddof=1)
        tstat_ic = mean_ic / (std_ic / np.sqrt(n_days)) if std_ic and std_ic > 0 and n_days > 1 else np.nan

        rows.append(
            dict(
                alpha_name=name,
                n_days=n_days,
                mean_ic=mean_ic,
                std_ic=std_ic,
                tstat_ic=tstat_ic,
            )
        )

    out_df = pd.DataFrame(rows)
    if "alpha" not in out_df.columns and "alpha_name" in out_df.columns:
        out_df["alpha"] = out_df["alpha_name"]
    out_df["label"] = args.filtered_label
    out_df = out_df.sort_values("tstat_ic", key=lambda s: s.abs(), ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["alpha", "n_days", "mean_ic", "std_ic", "tstat_ic", "label"]
    missing = [c for c in out_cols if c not in out_df.columns]
    if missing:
        raise ValueError(f"IC panel is missing required columns: {missing}")
    out_df[out_cols].to_csv(out_path, index=False)
    print(f"[alphas101_ic_panel_v0] Wrote IC panel for {len(out_df)} alphas to {out_path}")


if __name__ == "__main__":
    main()

