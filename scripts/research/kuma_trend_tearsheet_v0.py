#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PDF tear sheet for kuma_trend v0.")
    p.add_argument(
        "--research_dir",
        type=str,
        default="artifacts/research/kuma_trend",
        help="Root directory containing kuma_trend artifacts.",
    )
    p.add_argument(
        "--out_pdf",
        type=str,
        default="artifacts/research/kuma_trend/kuma_trend_tearsheet_v0.pdf",
        help="Output PDF path.",
    )
    return p.parse_args()


def _load_csv(path: Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def _load_csv_optional(path: Path, parse_dates: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[kuma_trend_tearsheet_v0] Optional file missing, skipping: {path}")
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def make_tearsheet(research_dir: Path, out_pdf: Path) -> None:
    equity_path = research_dir / "kuma_trend_equity_v0.csv"
    turnover_path = research_dir / "kuma_trend_turnover_v0.csv"
    weights_path = research_dir / "kuma_trend_weights_v0.parquet"
    metrics_path = research_dir / "metrics_kuma_trend_v0.csv"
    positions_path = research_dir / "kuma_trend_positions_v0.parquet"

    equity = _load_csv(equity_path, parse_dates=["ts"]).sort_values("ts")
    metrics = _load_csv(metrics_path)
    turnover = _load_csv_optional(turnover_path, parse_dates=["ts"])
    weights = None
    if weights_path.exists():
        weights = pd.read_parquet(weights_path)
    positions = None
    if positions_path.exists():
        positions = pd.read_parquet(positions_path)

    equity["drawdown"] = equity["portfolio_equity"] / equity["portfolio_equity"].cummax() - 1.0
    equity["rolling_vol_63"] = (
        equity["portfolio_ret"]
        .rolling(63, min_periods=63)
        .std()
        * np.sqrt(252.0)
    )
    roll_std = equity["portfolio_ret"].rolling(63, min_periods=63).std()
    roll_mean = equity["portfolio_ret"].rolling(63, min_periods=63).mean()
    equity["rolling_sharpe_63"] = (roll_mean / roll_std) * np.sqrt(252.0)

    full_row = metrics.loc[metrics["period"] == "full"].iloc[0] if not metrics.empty else None

    turnover_stats = {}
    if turnover is not None:
        t = turnover["turnover"].dropna()
        turnover_stats = {
            "mean": t.mean(),
            "median": t.median(),
            "pct25": t.quantile(0.25),
            "pct75": t.quantile(0.75),
            "max": t.max(),
        }

    # Prepare weight stats if available
    weight_stats = None
    weight_pivot = None
    if weights is not None and {"ts", "symbol", "weight"}.issubset(weights.columns):
        weights["ts"] = pd.to_datetime(weights["ts"])
        weight_pivot = (
            weights.pivot(index="ts", columns="symbol", values="weight")
            .sort_index()
            .fillna(0.0)
        )
        avg_abs = weight_pivot.abs().mean()
        holding_ratio = (weight_pivot.abs() > 0).mean()
        weight_stats = pd.DataFrame(
            {
                "symbol": avg_abs.index,
                "avg_abs_weight": avg_abs.values,
                "holding_ratio": holding_ratio.values,
            }
        ).sort_values("avg_abs_weight", ascending=False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # Page 1: Summary & equity
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.plot(equity["ts"], equity["portfolio_equity"])
        ax.set_yscale("log")
        ax.set_title("kuma_trend v0 – 20D Breakout + MA(5/40), ATR(20) Trailing Stop")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Equity (log)")
        ax.grid(True, alpha=0.3)

        text_lines = []
        if full_row is not None:
            text_lines.append(f"CAGR: {full_row['cagr']:.2%}")
            text_lines.append(f"Vol: {full_row['vol']:.2%}")
            text_lines.append(f"Sharpe: {full_row['sharpe']:.2f}")
            text_lines.append(f"MaxDD: {full_row['max_dd']:.2%}")
            text_lines.append(f"n_days: {int(full_row['n_days'])}")
        if turnover_stats:
            text_lines.append(f"Mean turnover: {turnover_stats['mean']:.3f}")
        text_lines.append("Cash yield: 4% annual (research assumption)")
        anchored_text = "\n".join(text_lines)
        ax.text(
            0.02,
            0.98,
            anchored_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Drawdowns
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.plot(equity["ts"], equity["drawdown"], color="tab:red")
        ax.axhline(0.0, color="black", linewidth=0.5)
        for lvl in [-0.1, -0.2, -0.3]:
            ax.axhline(lvl, color="gray", linestyle="--", linewidth=0.7)
        ax.set_title("Drawdowns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Rolling risk & turnover
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
        axes[0].plot(equity["ts"], equity["rolling_vol_63"])
        axes[0].set_ylabel("Vol (ann.)")
        axes[0].set_title("Rolling 63-day Volatility")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(equity["ts"], equity["rolling_sharpe_63"])
        axes[1].set_ylabel("Sharpe (63d)")
        axes[1].set_title("Rolling 63-day Sharpe")
        axes[1].grid(True, alpha=0.3)

        if turnover is not None:
            axes[2].plot(turnover["ts"], turnover["turnover"])
            if turnover_stats:
                axes[2].axhline(turnover_stats["mean"], color="gray", linestyle="--", linewidth=0.8, label="Mean")
                axes[2].legend()
        axes[2].set_ylabel("Turnover")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Daily Turnover (two-sided)")
        axes[2].grid(True, alpha=0.3)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Return distribution
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ret = equity["portfolio_ret"].dropna()
        ax.hist(ret, bins=50, alpha=0.8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title("Daily Return Distribution")
        ax.set_xlabel("Daily return")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        mu = ret.mean()
        sd = ret.std()
        ax.text(
            0.98,
            0.95,
            f"mean={mu:.4f}\nstd={sd:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Symbol exposure (if weights available)
        if weight_stats is not None and not weight_stats.empty:
            top = weight_stats.head(10)
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
            axes[0].bar(top["symbol"], top["avg_abs_weight"])
            axes[0].set_ylabel("Avg |weight|")
            axes[0].set_title("Top Symbols by Avg |Weight|")
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].grid(True, axis="y", alpha=0.3)

            axes[1].bar(top["symbol"], top["holding_ratio"])
            axes[1].set_ylabel("Holding ratio")
            axes[1].set_title("Holding Ratio (fraction of days in position)")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].grid(True, axis="y", alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print("[kuma_trend_tearsheet_v0] Weights not found or empty; skipping symbol exposure page.")

        # Optional Page 6: Position diagnostics (skip if missing)
        if positions is not None and {"ts", "symbol"}.issubset(positions.columns):
            # placeholder simple count by symbol
            fig, ax = plt.subplots(figsize=(11, 8.5))
            counts = positions["symbol"].value_counts()
            counts.head(20).plot(kind="bar", ax=ax)
            ax.set_title("Positions count by symbol (top 20)")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print("[kuma_trend_tearsheet_v0] Positions not found or incomplete; skipping position page.")

    print(f"[kuma_trend_tearsheet_v0] Wrote tear sheet to {out_pdf}")


def main() -> None:
    args = parse_args()
    research_dir = Path(args.research_dir)
    out_pdf = Path(args.out_pdf)
    make_tearsheet(research_dir, out_pdf)


if __name__ == "__main__":
    main()

