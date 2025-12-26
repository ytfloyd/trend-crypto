from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def load_nav(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path).select(["ts", "nav"]).sort("ts")
    return df


def metrics(df: pl.DataFrame) -> tuple[float, float]:
    df = df.sort("ts")
    nav = df["nav"]
    returns = nav.pct_change().fill_null(0.0)
    diffs = df.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
    dt_seconds = diffs.median() if diffs.len() > 0 else 0
    periods_per_year = (365 * 24 * 3600 / dt_seconds) if dt_seconds and dt_seconds > 0 else 8760
    mean = returns.mean()
    std = returns.std(ddof=1)
    sharpe = (mean / std) * (periods_per_year ** 0.5) if std and std > 0 else 0.0
    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_dd = drawdowns.min()
    return sharpe, max_dd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V1.2 vs BTC Buy & Hold")
    parser.add_argument("--v12_parquet", required=True, help="Path to V1.2 portfolio_equity.parquet")
    parser.add_argument("--btc_bh_parquet", required=True, help="Path to BTC buy&hold equity.parquet")
    parser.add_argument("--out_png", required=True, help="Output PNG path")
    args = parser.parse_args()

    v12 = load_nav(Path(args.v12_parquet))
    btc = load_nav(Path(args.btc_bh_parquet))

    joined = (
        v12.rename({"nav": "nav_v12"})
        .join(btc.rename({"nav": "nav_btc"}), on="ts", how="inner")
        .sort("ts")
    )
    if joined.is_empty():
        print("No overlapping timestamps between V1.2 and BTC buy&hold.")
        sys.exit(1)

    joined = joined.with_columns(
        [
            (pl.col("nav_v12") / pl.col("nav_v12").first()).alias("v12_norm"),
            (pl.col("nav_btc") / pl.col("nav_btc").first()).alias("btc_norm"),
        ]
    )

    sharpe_v12, maxdd_v12 = metrics(v12)
    sharpe_btc, maxdd_btc = metrics(btc)

    if maxdd_v12 < -0.25:
        print(f"V1.2 MaxDD {maxdd_v12:.4f} < -0.25 threshold; failing.")
        sys.exit(1)

    plt.figure(figsize=(10, 5))
    plt.plot(joined["ts"].to_list(), joined["v12_norm"].to_list(), label="V1.2 Net")
    plt.plot(joined["ts"].to_list(), joined["btc_norm"].to_list(), label="BTC Buy & Hold")
    plt.yscale("log")
    plt.title("V1.2 vs BTC Buy & Hold (normalized)")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Normalized NAV (log scale)")
    plt.legend()
    plt.tight_layout()
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved chart to {out_path}")
    print(f"V1.2 Sharpe={sharpe_v12:.4f} MaxDD={maxdd_v12:.4f}")
    print(f"BTC B&H Sharpe={sharpe_btc:.4f} MaxDD={maxdd_btc:.4f}")


if __name__ == "__main__":
    main()

