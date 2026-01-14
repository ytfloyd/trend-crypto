#!/usr/bin/env python
import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-alpha IC vs next-day returns for 101_alphas v0"
    )
    p.add_argument(
        "--alphas",
        required=True,
        help=(
            "Path to alphas parquet "
            "(e.g. artifacts/research/101_alphas/alphas_101_v0.parquet)"
        ),
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
        "--out_dir",
        required=True,
        help="Output directory for IC results",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alphas_path = Path(args.alphas)
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load alphas
    print(f"[alphas101_ic_v0] Loading alphas from {alphas_path}")
    df = pd.read_parquet(alphas_path)
    if "ts" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Expected columns 'ts' and 'symbol' in alphas parquet.")

    df["ts"] = pd.to_datetime(df["ts"])

    # Identify alpha columns
    alpha_cols = [c for c in df.columns if c not in ("ts", "symbol")]
    alpha_cols = sorted(alpha_cols)
    print(
        f"[alphas101_ic_v0] Found {len(alpha_cols)} alpha columns: "
        + ", ".join(alpha_cols)
    )

    # Determine date range for returns; need one extra day for forward return
    start_ts = df["ts"].min()
    end_ts = df["ts"].max()
    end_ts_plus = end_ts + pd.Timedelta(days=1)

    print(
        f"[alphas101_ic_v0] Fetching prices from {db_path} "
        f"(table/view: {args.price_table})"
    )
    con = duckdb.connect(str(db_path))

    price_sql = f"""
        SELECT
            symbol,
            ts,
            LEAD(close) OVER (PARTITION BY symbol ORDER BY ts) / close - 1.0 AS fwd_ret
        FROM {args.price_table}
        WHERE ts BETWEEN ? AND ?
    """
    prices = con.execute(price_sql, [start_ts, end_ts_plus]).df()
    prices["ts"] = pd.to_datetime(prices["ts"])

    print(f"[alphas101_ic_v0] Loaded {len(prices)} rows of forward returns")

    # Merge alphas with forward returns
    merged = df.merge(prices, on=["symbol", "ts"], how="inner")
    merged = merged.dropna(subset=["fwd_ret"])
    if merged.empty:
        raise ValueError("No overlap between alphas and price data after join.")

    print(f"[alphas101_ic_v0] Merged dataset: {len(merged)} rows")

    # Compute daily cross-sectional IC for each alpha
    daily_ic_frames = {}
    summary_rows = []

    by_ts = merged.groupby("ts")

    for col in alpha_cols:
        print(f"[alphas101_ic_v0] Computing IC for {col} ...")

        # Cross-sectional corr per ts; Series indexed by ts
        def _cs_corr(g: pd.DataFrame) -> float:
            if g[col].notna().sum() < 2 or g["fwd_ret"].notna().sum() < 2:
                return np.nan
            return g[col].corr(g["fwd_ret"])

        daily_ic = by_ts.apply(_cs_corr).dropna()

        n = len(daily_ic)
        mean_ic = daily_ic.mean() if n > 0 else np.nan
        std_ic = daily_ic.std(ddof=0) if n > 1 else np.nan
        if std_ic is not None and std_ic > 0 and n > 1:
            tstat = mean_ic / (std_ic / np.sqrt(n))
        else:
            tstat = np.nan

        daily_ic_frames[col] = daily_ic

        summary_rows.append(
            {
                "alpha": col,
                "n_days": n,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "tstat_ic": tstat,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("mean_ic", ascending=False)

    summary_path = out_dir / "alphas101_ic_summary_v0.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[alphas101_ic_v0] Wrote per-alpha IC summary to {summary_path}")

    # Wide daily IC panel (ts x alpha)
    ic_panel = pd.DataFrame(daily_ic_frames)
    ic_panel.index.name = "ts"
    ic_panel_path = out_dir / "alphas101_ic_daily_v0.parquet"
    ic_panel.to_parquet(ic_panel_path)
    print(f"[alphas101_ic_v0] Wrote daily IC panel to {ic_panel_path}")

    # Bar chart of mean IC per alpha
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary["alpha"], summary["mean_ic"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel("Mean daily cross-sectional IC")
    ax.set_title("101_alphas v0 – per-alpha IC vs next-day returns")
    ax.set_xticks(range(len(summary["alpha"])))
    ax.set_xticklabels(summary["alpha"], rotation=90)
    fig.tight_layout()

    bar_path = out_dir / "alphas101_ic_mean_bar_v0.png"
    fig.savefig(bar_path)
    plt.close(fig)
    print(f"[alphas101_ic_v0] Wrote IC bar chart to {bar_path}")


if __name__ == "__main__":
    main()

