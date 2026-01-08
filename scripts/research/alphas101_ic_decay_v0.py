#!/usr/bin/env python
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IC decay for a single alpha (e.g. alpha_008).")
    p.add_argument(
        "--alphas",
        required=True,
        help="Alphas parquet (e.g. artifacts/research/101_alphas/alphas_101_v0.parquet)",
    )
    p.add_argument(
        "--alpha_name",
        default="alpha_008",
        help="Alpha column to analyze (default: alpha_008)",
    )
    p.add_argument(
        "--db",
        required=True,
        help="DuckDB path (e.g. ../data/coinbase_daily_121025.duckdb)",
    )
    p.add_argument(
        "--price_table",
        default="bars_1d_usd_universe_clean",
        help="Daily bars table/view (default: bars_1d_usd_universe_clean)",
    )
    p.add_argument(
        "--max_horizon",
        type=int,
        default=5,
        help="Max forward horizon in days (default: 5)",
    )
    p.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV for IC decay summary.",
    )
    p.add_argument(
        "--out_png",
        required=True,
        help="Output PNG for IC decay plot.",
    )
    return p.parse_args()


def daily_ic(merged: pd.DataFrame, sig_col: str, ret_col: str) -> pd.Series:
    by_ts = merged.groupby("ts")

    def _cs_corr(g: pd.DataFrame) -> float:
        x = g[sig_col]
        y = g[ret_col]
        if x.notna().sum() < 2 or y.notna().sum() < 2:
            return np.nan
        return x.corr(y)

    ic = by_ts.apply(_cs_corr).dropna()
    return ic


def main() -> None:
    args = parse_args()

    alphas_path = Path(args.alphas)
    db_path = Path(args.db)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)

    print(f"[alphas101_ic_decay_v0] Loading alphas from {alphas_path}")
    df = pd.read_parquet(alphas_path)
    if "ts" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Expected 'ts' and 'symbol' columns in alphas parquet.")
    if args.alpha_name not in df.columns:
        raise ValueError(f"Alpha column {args.alpha_name} not found in alphas parquet.")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df[["symbol", "ts", args.alpha_name]].rename(columns={args.alpha_name: "alpha"})
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()

    print(f"[alphas101_ic_decay_v0] Fetching prices from {db_path}")
    con = duckdb.connect(str(db_path))
    prices = con.execute(
        f"""
        SELECT symbol, ts, close, volume
        FROM {args.price_table}
        WHERE ts BETWEEN ? AND ?
        """,
        [min_ts, max_ts + pd.Timedelta(days=args.max_horizon)],
    ).df()
    prices["ts"] = pd.to_datetime(prices["ts"])
    prices = prices.sort_values(["symbol", "ts"])
    prices["prev_close"] = prices.groupby("symbol")["close"].shift(1)
    valid = (prices["volume"] > 0) & (prices["close"] != prices["prev_close"])
    prices = prices.loc[valid].copy()

    # Compute forward returns for horizons 1..max_horizon on cleaned prices
    prices = prices.set_index(["symbol", "ts"]).sort_index()
    for h in range(1, args.max_horizon + 1):
        prices[f"fwd_ret_{h}d"] = (
            prices.groupby(level="symbol")["close"].shift(-h) / prices["close"] - 1.0
        )

    prices = prices.reset_index()

    rows = []

    for h in range(1, args.max_horizon + 1):
        col = f"fwd_ret_{h}d"
        merged = df.merge(
            prices[["symbol", "ts", col]],
            on=["symbol", "ts"],
            how="inner",
        ).dropna(subset=["alpha", col])
        if merged.empty:
            print(f"[alphas101_ic_decay_v0] Horizon {h}d has no overlap, skipping.")
            continue

        ic_series = daily_ic(merged, "alpha", col)
        n_days = len(ic_series)
        mean_ic = ic_series.mean()
        std_ic = ic_series.std(ddof=0) if n_days > 1 else np.nan
        tstat = (
            mean_ic / (std_ic / np.sqrt(n_days))
            if (std_ic and std_ic > 0 and n_days > 1)
            else np.nan
        )

        rows.append(
            dict(
                horizon_days=h,
                n_days=n_days,
                mean_ic=mean_ic,
                std_ic=std_ic,
                tstat_ic=tstat,
            )
        )

    out_df = pd.DataFrame.from_records(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"[alphas101_ic_decay_v0] Wrote IC decay summary to {out_csv}")

    # Plot mean IC vs horizon
    plt.figure()
    plt.plot(out_df["horizon_days"], out_df["mean_ic"], marker="o")
    plt.xlabel("Horizon (days)")
    plt.ylabel("Mean daily cross-sectional IC")
    plt.title(f"IC Decay for {args.alpha_name}")
    plt.grid(True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"[alphas101_ic_decay_v0] Wrote IC decay plot to {out_png}")


if __name__ == "__main__":
    main()

