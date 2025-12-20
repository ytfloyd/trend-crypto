from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import polars as pl
import numpy as np


def load_equity(run_dir: Path) -> pl.DataFrame:
    path = run_dir / "equity.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity.parquet in {run_dir}")
    return pl.read_parquet(path).select(["ts", "nav"]).sort("ts")


def compute_equal_risk_portfolio(
    eq_btc: pl.DataFrame,
    eq_eth: pl.DataFrame,
    vol_window: int,
    initial_nav: float,
    target_portfolio_vol_annual: float,
    port_vol_window: int,
    max_gross_leverage: float,
    min_gross_leverage: float,
    port_vol_method: str,
    downside_threshold: float,
) -> pl.DataFrame:
    joined = (
        eq_btc.rename({"nav": "nav_btc"})
        .join(eq_eth.rename({"nav": "nav_eth"}), on="ts", how="inner")
        .sort("ts")
    )
    if joined.height < 2:
        raise ValueError("Not enough overlapping data between BTC and ETH runs.")

    df = joined.with_columns(
        [
            (pl.col("nav_btc") / pl.col("nav_btc").shift(1) - 1).alias("r_btc"),
            (pl.col("nav_eth") / pl.col("nav_eth").shift(1) - 1).alias("r_eth"),
        ]
    ).drop_nulls(subset=["r_btc", "r_eth"])

    df = df.with_columns(
        [
            pl.col("r_btc")
            .rolling_std(window_size=vol_window, min_samples=vol_window)
            .shift(1)
            .alias("vol_btc"),
            pl.col("r_eth")
            .rolling_std(window_size=vol_window, min_samples=vol_window)
            .shift(1)
            .alias("vol_eth"),
        ]
    )

    inv_btc = pl.when(pl.col("vol_btc") > 0).then(1.0 / pl.col("vol_btc")).otherwise(None)
    inv_eth = pl.when(pl.col("vol_eth") > 0).then(1.0 / pl.col("vol_eth")).otherwise(None)
    sum_inv = inv_btc + inv_eth
    w_btc = pl.when(sum_inv.is_not_null() & (sum_inv > 0)).then(inv_btc / sum_inv).otherwise(0.5)
    w_eth = 1 - w_btc

    df = df.with_columns(
        [
            w_btc.alias("w_btc"),
            w_eth.alias("w_eth"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("w_btc").shift(1) * pl.col("r_btc") + pl.col("w_eth").shift(1) * pl.col("r_eth")).alias(
                "r_port_base"
            ),
        ]
    )

    df = df.drop_nulls(subset=["r_port_base"])

    target_sigma_hourly = (
        target_portfolio_vol_annual / (8760 ** 0.5)
        if target_portfolio_vol_annual is not None and target_portfolio_vol_annual > 0
        else None
    )

    def downside_std(vals: list[float]) -> float | None:
        downs = [x for x in vals if x < downside_threshold]
        min_required = max(10, int(port_vol_window * 0.05))
        if len(downs) < min_required:
            return None
        return float(np.std(downs, ddof=1))

    if port_vol_method == "downside":
        df = df.with_columns(
            [
                pl.col("r_port_base")
                .rolling_map(downside_std, window_size=port_vol_window)
                .shift(1)
                .alias("sigma_port"),
            ]
        )
    else:
        df = df.with_columns(
            [
                pl.col("r_port_base")
                .rolling_std(window_size=port_vol_window, min_samples=port_vol_window)
                .shift(1)
                .alias("sigma_port"),
            ]
        )

    if target_sigma_hourly is not None:
        leverage_expr = (
            (target_sigma_hourly / pl.col("sigma_port"))
            .fill_null(0.0)
            .clip(min_gross_leverage, max_gross_leverage)
        )
    else:
        leverage_expr = pl.lit(1.0)

    df = df.with_columns(leverage_expr.shift(1).fill_null(min_gross_leverage).alias("leverage_scalar"))

    df = df.with_columns(
        [
            (pl.col("w_btc") * pl.col("leverage_scalar")).alias("w_btc_scaled"),
            (pl.col("w_eth") * pl.col("leverage_scalar")).alias("w_eth_scaled"),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("w_btc_scaled").shift(1) * pl.col("r_btc")
                + pl.col("w_eth_scaled").shift(1) * pl.col("r_eth")
            ).alias("r_port"),
        ]
    )

    df = df.drop_nulls(subset=["r_port"])
    df = df.with_columns(
        [
            ((1 + pl.col("r_port")).cum_prod() * initial_nav).alias("nav"),
        ]
    )
    return df


def metrics(df: pl.DataFrame) -> Dict[str, float]:
    df = df.sort("ts")
    nav = df["nav"]
    start = nav.item(0)
    end = nav.item(nav.len() - 1)
    total_return = (end / start) - 1 if start else 0.0
    returns = df["r_port"]
    mean = returns.mean()
    std = returns.std(ddof=1)
    sharpe = (mean / std) * (8760 ** 0.5) if std and std > 0 else 0.0
    ts_min = df["ts"].min()
    ts_max = df["ts"].max()
    total_hours = (ts_max - ts_min).total_seconds() / 3600 if ts_max and ts_min else 0
    cagr = (end / start) ** (8760 / total_hours) - 1 if start and total_hours > 0 else 0.0
    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_dd = drawdowns.min()
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def plot_equity(df: pl.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["ts"].to_list(), df["nav"].to_list(), label="Equal-Risk Portfolio")
    plt.yscale("log")
    plt.title("Equal-Risk BTC+ETH Portfolio (log NAV)")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("NAV (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve_log.png", dpi=150)
    plt.close()


def plot_weights(df: pl.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["ts"].to_list(), df["w_btc"].to_list(), label="w_btc (base)")
    plt.plot(df["ts"].to_list(), df["w_eth"].to_list(), label="w_eth (base)")
    plt.plot(df["ts"].to_list(), df["w_btc_scaled"].to_list(), label="w_btc_scaled")
    plt.plot(df["ts"].to_list(), df["w_eth_scaled"].to_list(), label="w_eth_scaled")
    plt.title("Equal-Risk Weights (base vs scaled)")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "weights.png", dpi=150)
    plt.close()


def plot_leverage(df: pl.DataFrame, out_dir: Path, method: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["ts"].to_list(), df["leverage_scalar"].to_list(), label="leverage_scalar")
    plt.title(f"Portfolio Leverage Scalar ({method})")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Leverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "leverage.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build equal-risk BTC+ETH portfolio from run outputs.")
    parser.add_argument("--run_btc", required=True, help="Path to BTC strategy run dir")
    parser.add_argument("--run_eth", required=True, help="Path to ETH strategy run dir")
    parser.add_argument(
        "--out", default="artifacts/compare/portfolio_btc_eth_equal_risk", help="Output directory"
    )
    parser.add_argument("--vol_window_hours", type=int, default=720, help="Rolling vol window in hours")
    parser.add_argument(
        "--target_portfolio_vol_annual",
        type=float,
        default=0.30,
        help="Target annualized portfolio vol (set <=0 to disable)",
    )
    parser.add_argument(
        "--port_vol_window_hours",
        type=int,
        default=None,
        help="Rolling window for portfolio vol targeting (defaults to vol_window_hours)",
    )
    parser.add_argument(
        "--max_gross_leverage", type=float, default=1.0, help="Max gross leverage clamp"
    )
    parser.add_argument(
        "--min_gross_leverage", type=float, default=0.0, help="Min gross leverage clamp"
    )
    parser.add_argument(
        "--port_vol_method",
        type=str,
        default="total",
        choices=["total", "downside"],
        help="Portfolio vol method: total (std) or downside (semi-vol)",
    )
    parser.add_argument(
        "--downside_threshold",
        type=float,
        default=0.0,
        help="Threshold for downside vol (returns below this count as downside)",
    )
    parser.add_argument("--initial_nav", type=float, default=100000.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_btc = load_equity(Path(args.run_btc))
    eq_eth = load_equity(Path(args.run_eth))
    port_window = args.port_vol_window_hours or args.vol_window_hours

    portfolio = compute_equal_risk_portfolio(
        eq_btc,
        eq_eth,
        args.vol_window_hours,
        args.initial_nav,
        args.target_portfolio_vol_annual,
        port_window,
        args.max_gross_leverage,
        args.min_gross_leverage,
        args.port_vol_method,
        args.downside_threshold,
    )
    portfolio.write_parquet(out_dir / "portfolio_equity.parquet")

    m = metrics(portfolio)
    summary = {
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "vol_window_hours": args.vol_window_hours,
        "target_portfolio_vol_annual": args.target_portfolio_vol_annual,
        "port_vol_window_hours": port_window,
        "max_gross_leverage": args.max_gross_leverage,
        "min_gross_leverage": args.min_gross_leverage,
        "port_vol_method": args.port_vol_method,
        "downside_threshold": args.downside_threshold,
        "run_btc": str(Path(args.run_btc).resolve()),
        "run_eth": str(Path(args.run_eth).resolve()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Equal-Risk Portfolio Summary")
    print(f"Total Return : {m['total_return']:.4f}")
    print(f"CAGR         : {m['cagr']:.4f}")
    print(f"Sharpe       : {m['sharpe']:.4f}")
    print(f"Max Drawdown : {m['max_drawdown']:.4f}")

    plot_equity(portfolio, out_dir)
    plot_weights(portfolio, out_dir)
    plot_leverage(portfolio, out_dir, args.port_vol_method)
    print(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()

