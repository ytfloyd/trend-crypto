from __future__ import annotations

import argparse
import numpy as np
import polars as pl


def load_equity(run_dir: str) -> pl.DataFrame:
    path = f"{run_dir}/equity.parquet"
    df = pl.read_parquet(path).select(["ts", "nav"]).sort("ts")
    if df.is_empty():
        raise ValueError(f"Empty equity data in {run_dir}")
    return df


def align_and_returns(run_btc: str, run_eth: str) -> pl.DataFrame:
    btc = load_equity(run_btc).rename({"nav": "nav_btc"})
    eth = load_equity(run_eth).rename({"nav": "nav_eth"})
    joined = btc.join(eth, on="ts", how="inner").sort("ts")
    if joined.height < 2:
        raise ValueError("Not enough overlapping samples between BTC and ETH runs.")
    joined = joined.with_columns(
        [
            (pl.col("nav_btc") / pl.col("nav_btc").shift(1) - 1).alias("r_btc"),
            (pl.col("nav_eth") / pl.col("nav_eth").shift(1) - 1).alias("r_eth"),
        ]
    ).drop_nulls(subset=["r_btc", "r_eth"])
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose BTC/ETH sleeve correlation.")
    parser.add_argument("--run_btc", required=True, help="Path to BTC strategy run dir")
    parser.add_argument("--run_eth", required=True, help="Path to ETH strategy run dir")
    parser.add_argument("--crisis_quantile", type=float, default=0.2, help="Drawdown quantile for crisis regime")
    parser.add_argument("--hours_per_year", type=int, default=8760, help="Hours per year (consistency)")
    parser.add_argument("--print_head", type=int, default=0, help="Print first N rows of aligned returns")
    args = parser.parse_args()

    df = align_and_returns(args.run_btc, args.run_eth)
    if args.print_head > 0:
        print(df.head(args.print_head))

    r_btc = df["r_btc"].to_numpy()
    r_eth = df["r_eth"].to_numpy()
    corr_full = np.corrcoef(r_btc, r_eth)[0, 1] if len(r_btc) > 1 else float("nan")

    df = df.with_columns(
        [
            ((pl.col("r_btc") + pl.col("r_eth")) / 2).alias("r_port"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("r_port") + 1).cum_prod().alias("nav_port"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("nav_port") / pl.col("nav_port").cum_max() - 1).alias("dd_port"),
        ]
    )

    q = df["dd_port"].quantile(args.crisis_quantile)
    crisis_df = df.filter(pl.col("dd_port") <= q)
    corr_crisis = None
    if crisis_df.height >= 2:
        rc_btc = crisis_df["r_btc"].to_numpy()
        rc_eth = crisis_df["r_eth"].to_numpy()
        corr_crisis = np.corrcoef(rc_btc, rc_eth)[0, 1]
    else:
        print("Warning: crisis subset too small for correlation.")

    print("BTC/ETH Sleeve Correlation Diagnostic")
    print(f"Aligned samples: {df.height}")
    print(f"Full-sample correlation: {corr_full:.4f}")
    print(f"Crisis quantile (dd_port): {q:.4f}")
    print(f"Crisis samples: {crisis_df.height}")
    if corr_crisis is not None:
        print(f"Crisis correlation: {corr_crisis:.4f}")
    else:
        print("Crisis correlation: n/a (insufficient samples)")


if __name__ == "__main__":
    main()

