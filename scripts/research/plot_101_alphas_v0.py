#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot 101-alphas v0 diagnostics.")
    p.add_argument(
        "--alphas",
        required=True,
        help=(
            "Path to alphas parquet "
            "(e.g. artifacts/research/101_alphas/alphas_101_v0.parquet)"
        ),
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for plots",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alphas_path = Path(args.alphas)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot_101_alphas_v0] Loading alphas from {alphas_path}")
    df = pd.read_parquet(alphas_path)

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])

    # Identify alpha columns: everything except ts/symbol
    alpha_cols = [c for c in df.columns if c not in ("ts", "symbol")]
    alpha_cols = sorted(alpha_cols)
    print(
        f"[plot_101_alphas_v0] Found {len(alpha_cols)} alphas: "
        + ", ".join(alpha_cols)
    )

    # Per-alpha time-series (cross-sectional mean/std) and histogram
    for col in alpha_cols:
        print(f"[plot_101_alphas_v0] Plotting {col} ...")
        s = df[col]

        # Time-series: cross-sectional mean/std over symbols per ts
        if "ts" in df.columns:
            g = df.groupby("ts")[col]
            mean_ts = g.mean()
            std_ts = g.std(ddof=0)

            fig, ax = plt.subplots()
            ax.plot(mean_ts.index, mean_ts.values, label="mean")
            ax.fill_between(
                mean_ts.index,
                (mean_ts - std_ts).values,
                (mean_ts + std_ts).values,
                alpha=0.3,
                label="±1 std",
            )
            ax.axhline(0.0, linewidth=0.8)
            ax.set_title(f"{col} – cross-sectional mean ±1σ")
            ax.set_xlabel("ts")
            ax.set_ylabel(col)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{col}_timeseries.png")
            plt.close(fig)

        # Histogram over all (ts, symbol)
        s_clean = s.replace([np.inf, -np.inf], np.nan).dropna()
        if not s_clean.empty:
            fig, ax = plt.subplots()
            ax.hist(s_clean.values, bins=100)
            ax.set_title(f"{col} – histogram")
            ax.set_xlabel(col)
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(out_dir / f"{col}_hist.png")
            plt.close(fig)

    # Correlation heatmap across alphas (flattened over ts,symbol)
    print("[plot_101_alphas_v0] Computing alpha correlation heatmap ...")
    flat = df[alpha_cols].dropna()
    corr = flat.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, origin="lower")
    ax.set_xticks(np.arange(len(alpha_cols)))
    ax.set_yticks(np.arange(len(alpha_cols)))
    ax.set_xticklabels(alpha_cols, rotation=90)
    ax.set_yticklabels(alpha_cols)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Alpha correlation heatmap (flattened over ts,symbol)")
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_corr_heatmap.png")
    plt.close(fig)

    print(f"[plot_101_alphas_v0] Done. Plots written to {out_dir}")


if __name__ == "__main__":
    main()

