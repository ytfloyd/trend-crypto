from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from math import exp, log, sqrt


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
    panic_fast_halflife_hours: float,
    panic_vol_threshold_annual: float,
    panic_min_bars: int,
    corr_window_hours: int,
    corr_threshold: float,
    corr_leverage_mult: float,
    portfolio_fee_bps: float,
    portfolio_slippage_bps: float,
    enable_panic_switch: int,
    enable_corr_override: int,
    corr_window_slow_hours: int,
    corr_threshold_slow: float,
    corr_window_fast_hours: int,
    corr_threshold_fast: float,
    max_allowed_drawdown: float,
    dd_vol_floor: float,
    enable_dd_targetvol_scaling: int,
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

    def compute_leverage_scalar(target_sigma_col: pl.Expr) -> pl.Expr:
        return (
            (target_sigma_col / pl.col("sigma_port"))
            .fill_null(0.0)
            .clip(min_gross_leverage, min(max_gross_leverage, 1.0))
            .shift(1)
            .fill_null(min_gross_leverage)
        )

    # base target sigma
    target_sigma_hourly_base = (
        target_portfolio_vol_annual / (8760 ** 0.5)
        if target_portfolio_vol_annual is not None and target_portfolio_vol_annual > 0
        else None
    )
    base_sigma_col = (
        pl.lit(target_sigma_hourly_base) if target_sigma_hourly_base is not None else pl.lit(0.0)
    )

    # first pass leverage (for provisional nav)
    df = df.with_columns(compute_leverage_scalar(base_sigma_col).alias("leverage_scalar_base"))

    # Correlation calculation (slow/fast)
    def rolling_corr_expr(window: int) -> pl.Expr:
        mean_btc = pl.col("r_btc").rolling_mean(window_size=window, min_samples=window)
        mean_eth = pl.col("r_eth").rolling_mean(window_size=window, min_samples=window)
        mean_prod = (
            (pl.col("r_btc") * pl.col("r_eth")).rolling_mean(window_size=window, min_samples=window)
        )
        var_btc = pl.col("r_btc").rolling_var(window_size=window, min_samples=window)
        var_eth = pl.col("r_eth").rolling_var(window_size=window, min_samples=window)
        cov = mean_prod - mean_btc * mean_eth
        return (
            pl.when((var_btc > 0) & (var_eth > 0))
            .then(cov / (var_btc.sqrt() * var_eth.sqrt()))
            .otherwise(None)
        )

    corr_slow_expr = rolling_corr_expr(corr_window_slow_hours).shift(1).alias("corr_slow")
    corr_fast_expr = rolling_corr_expr(corr_window_fast_hours).shift(1).alias("corr_fast")
    df = df.with_columns([corr_slow_expr, corr_fast_expr])

    override_cond = (
        ((pl.col("corr_slow").is_not_null()) & (pl.col("corr_slow") > corr_threshold_slow))
        | ((pl.col("corr_fast").is_not_null()) & (pl.col("corr_fast") > corr_threshold_fast))
    )
    corr_reason_expr = pl.when(
        (pl.col("corr_slow").is_not_null()) & (pl.col("corr_slow") > corr_threshold_slow)
    ).then(pl.lit("slow")).otherwise(None)
    corr_reason_expr = pl.when(
        (pl.col("corr_fast").is_not_null()) & (pl.col("corr_fast") > corr_threshold_fast)
    ).then(
        pl.when(corr_reason_expr.is_not_null()).then(pl.lit("both")).otherwise(pl.lit("fast"))
    ).otherwise(corr_reason_expr)
    corr_reason_expr = corr_reason_expr.alias("corr_override_reason")

    if enable_corr_override:
        leverage_scalar_corr_base = pl.when(override_cond).then(
            pl.col("leverage_scalar_base") * corr_leverage_mult
        ).otherwise(pl.col("leverage_scalar_base"))
        corr_override_triggered = override_cond.fill_null(False)
    else:
        leverage_scalar_corr_base = pl.col("leverage_scalar_base")
        corr_override_triggered = pl.lit(False)

    df = df.with_columns(
        [
            leverage_scalar_corr_base.alias("leverage_scalar_corr_base"),
            corr_override_triggered.alias("corr_override_triggered"),
            corr_reason_expr,
        ]
    )

    # Panic kill-switch using EWMA fast vol
    alpha = 1 - exp(log(0.5) / max(panic_fast_halflife_hours, 1e-6))
    r_base_np = df["r_port_base"].to_list()
    var_list = []
    var_prev = 0.0
    for i, r in enumerate(r_base_np):
        if i == 0:
            var_prev = r * r
        else:
            var_prev = (1 - alpha) * var_prev + alpha * (r_base_np[i - 1] ** 2)
        var_list.append(var_prev)
    fast_vol_ann = [sqrt(v) * sqrt(8760) if v >= 0 else None for v in var_list]
    df = df.with_columns(
        [
            pl.Series("fast_vol_ann", fast_vol_ann),
        ]
    )
    if enable_panic_switch:
        df = df.with_columns(
            (
                pl.when(
                    (pl.col("fast_vol_ann") > panic_vol_threshold_annual)
                    & (pl.arange(0, pl.len()) >= panic_min_bars)
                )
                .then(pl.lit(0.0))
                .otherwise(pl.col("leverage_scalar_corr_base"))
            ).alias("leverage_scalar_panic_base")
        )
        df = df.with_columns(
            (
                (pl.col("fast_vol_ann") > panic_vol_threshold_annual)
                & (pl.arange(0, pl.len()) >= panic_min_bars)
            ).alias("panic_triggered_base")
        )
    else:
        df = df.with_columns(pl.col("leverage_scalar_corr_base").alias("leverage_scalar_panic_base"))
        df = df.with_columns(pl.lit(False).alias("panic_triggered_base"))

    cost_bps_total = (portfolio_fee_bps + portfolio_slippage_bps) / 10000.0

    # Provisional NAV (pre drawdown scaling)
    df = df.with_columns(
        [
            (pl.col("w_btc") * pl.col("leverage_scalar_panic_base")).alias("w_btc_scaled_pre"),
            (pl.col("w_eth") * pl.col("leverage_scalar_panic_base")).alias("w_eth_scaled_pre"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("w_btc_scaled_pre").shift(1) * pl.col("r_btc")
                + pl.col("w_eth_scaled_pre").shift(1) * pl.col("r_eth")
            ).alias("r_port_pre"),
        ]
    )
    df = df.drop_nulls(subset=["r_port_pre"])
    df = df.with_columns(
        (
            (pl.col("w_btc_scaled_pre") - pl.col("w_btc_scaled_pre").shift(1)).abs()
            + (pl.col("w_eth_scaled_pre") - pl.col("w_eth_scaled_pre").shift(1)).abs()
        )
        .fill_null(0.0)
        .alias("turnover_pre")
    )
    nav_pre_list = []
    nav_prev = initial_nav
    r_port_pre_list = df["r_port_pre"].to_list()
    turnover_pre_list = df["turnover_pre"].to_list()
    for i in range(len(r_port_pre_list)):
        cost_cash = turnover_pre_list[i] * cost_bps_total * nav_prev
        r_net = r_port_pre_list[i] - (cost_cash / nav_prev if nav_prev != 0 else 0.0)
        nav_curr = nav_prev * (1 + r_net)
        nav_pre_list.append(nav_curr)
        nav_prev = nav_curr
    df = df.with_columns(pl.Series("nav_pre", nav_pre_list))

    # Drawdown scaling
    df = df.with_columns((pl.col("nav_pre") / pl.col("nav_pre").cum_max() - 1).alias("dd"))
    if enable_dd_targetvol_scaling:
        df = df.with_columns(
            (
                (1 - (pl.col("dd").abs() / max_allowed_drawdown))
                .clip(lower_bound=dd_vol_floor)
            ).alias("scale_dd")
        )
    else:
        df = df.with_columns(pl.lit(1.0).alias("scale_dd"))
    df = df.with_columns(
        (pl.lit(target_portfolio_vol_annual) * pl.col("scale_dd")).alias("target_vol_new_annual")
    )
    target_sigma_dynamic = pl.col("target_vol_new_annual") / (8760 ** 0.5)
    df = df.with_columns(compute_leverage_scalar(target_sigma_dynamic).alias("leverage_scalar_base_scaled"))

    # Final corr override and panic on scaled leverage
    if enable_corr_override:
        leverage_scalar_corr_final = pl.when(override_cond).then(
            pl.col("leverage_scalar_base_scaled") * corr_leverage_mult
        ).otherwise(pl.col("leverage_scalar_base_scaled"))
    else:
        leverage_scalar_corr_final = pl.col("leverage_scalar_base_scaled")

    if enable_panic_switch:
        df = df.with_columns(
            (
                pl.when(
                    (pl.col("fast_vol_ann") > panic_vol_threshold_annual)
                    & (pl.arange(0, pl.len()) >= panic_min_bars)
                )
                .then(pl.lit(0.0))
                .otherwise(leverage_scalar_corr_final)
            ).alias("leverage_scalar_final")
        )
        df = df.with_columns(
            (
                (pl.col("fast_vol_ann") > panic_vol_threshold_annual)
                & (pl.arange(0, pl.len()) >= panic_min_bars)
            ).alias("panic_triggered")
        )
    else:
        df = df.with_columns(leverage_scalar_corr_final.alias("leverage_scalar_final"))
        df = df.with_columns(pl.lit(False).alias("panic_triggered"))

    df = df.with_columns(
        [
            (pl.col("w_btc") * pl.col("leverage_scalar_final")).alias("w_btc_scaled"),
            (pl.col("w_eth") * pl.col("leverage_scalar_final")).alias("w_eth_scaled"),
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
        (
            (pl.col("w_btc_scaled") - pl.col("w_btc_scaled").shift(1)).abs()
            + (pl.col("w_eth_scaled") - pl.col("w_eth_scaled").shift(1)).abs()
        )
        .fill_null(0.0)
        .alias("turnover")
    )

    r_port_list = df["r_port"].to_list()
    turnover_list = df["turnover"].to_list()

    nav_list = []
    r_port_net_list = []
    cost_cash_list = []
    nav_prev = initial_nav
    for i in range(len(r_port_list)):
        cost_cash = turnover_list[i] * cost_bps_total * nav_prev
        r_net = r_port_list[i] - (cost_cash / nav_prev if nav_prev != 0 else 0.0)
        nav_curr = nav_prev * (1 + r_net)
        nav_list.append(nav_curr)
        r_port_net_list.append(r_net)
        cost_cash_list.append(cost_cash)
        nav_prev = nav_curr

    df = df.with_columns(
        [
            pl.Series("r_port_net", r_port_net_list),
            pl.Series("cost_cash", cost_cash_list),
            pl.Series("nav", nav_list),
        ]
    )

    df = df.select(
        [
            "ts",
            "nav_btc",
            "nav_eth",
            "r_btc",
            "r_eth",
            "vol_btc",
            "vol_eth",
            "w_btc",
            "w_eth",
            "r_port_base",
            "sigma_port",
            "corr_slow",
            "corr_fast",
            "corr_override_triggered",
            "corr_override_reason",
            "fast_vol_ann",
            "panic_triggered",
            "leverage_scalar_final",
            "w_btc_scaled",
            "w_eth_scaled",
            "turnover",
            "cost_cash",
            "r_port",
            "r_port_net",
            "nav",
            "dd",
            "scale_dd",
            "target_vol_new_annual",
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
    diffs = df.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
    dt_seconds = diffs.median() if diffs.len() > 0 else 0
    periods_per_year = (365 * 24 * 3600 / dt_seconds) if dt_seconds and dt_seconds > 0 else 8760
    sharpe = (mean / std) * (periods_per_year ** 0.5) if std and std > 0 else 0.0
    n_periods = returns.len()
    cagr = (end / start) ** (periods_per_year / n_periods) - 1 if start and n_periods > 0 else 0.0
    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_dd = drawdowns.min()
    max_dd_idx = int(drawdowns.arg_min()) if drawdowns.len() > 0 else 0
    max_dd_ts = df["ts"].item(max_dd_idx) if drawdowns.len() > 0 else None
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_ts": max_dd_ts,
        "scale_dd_min": df["scale_dd"].min() if "scale_dd" in df.columns else None,
        "scale_dd_median": df["scale_dd"].median() if "scale_dd" in df.columns else None,
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
    parser.add_argument(
        "--panic_fast_halflife_hours",
        type=float,
        default=72.0,
        help="Half-life (hours) for fast EWMA vol kill-switch",
    )
    parser.add_argument(
        "--panic_vol_threshold_annual",
        type=float,
        default=0.80,
        help="Annualized vol threshold for panic kill-switch",
    )
    parser.add_argument(
        "--panic_min_bars",
        type=int,
        default=24,
        help="Minimum bars before panic vol can trigger",
    )
    parser.add_argument(
        "--enable_panic_switch",
        type=int,
        default=1,
        help="Set to 0 to disable panic kill-switch; 1 to enable",
    )
    parser.add_argument(
        "--corr_window_hours",
        type=int,
        default=336,
        help="Rolling correlation window (hours) for correlation override",
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.80,
        help="Correlation threshold to apply override",
    )
    parser.add_argument(
        "--corr_window_slow_hours",
        type=int,
        default=336,
        help="Rolling correlation window (hours) for slow correlation override",
    )
    parser.add_argument(
        "--corr_threshold_slow",
        type=float,
        default=0.80,
        help="Correlation threshold for slow window",
    )
    parser.add_argument(
        "--corr_window_fast_hours",
        type=int,
        default=24,
        help="Rolling correlation window (hours) for fast correlation override",
    )
    parser.add_argument(
        "--corr_threshold_fast",
        type=float,
        default=0.90,
        help="Correlation threshold for fast window",
    )
    parser.add_argument(
        "--corr_leverage_mult",
        type=float,
        default=0.50,
        help="Leverage multiplier when correlation override triggers",
    )
    parser.add_argument(
        "--enable_corr_override",
        type=int,
        default=1,
        help="Set to 0 to disable correlation override; 1 to enable",
    )
    parser.add_argument(
        "--max_allowed_drawdown",
        type=float,
        default=0.25,
        help="Drawdown level at which target vol scales down to floor",
    )
    parser.add_argument(
        "--dd_vol_floor",
        type=float,
        default=0.10,
        help="Minimum fraction of base target vol under deep drawdown",
    )
    parser.add_argument(
        "--enable_dd_targetvol_scaling",
        type=int,
        default=1,
        help="Set to 0 to disable drawdown-based target vol scaling",
    )
    parser.add_argument(
        "--portfolio_fee_bps",
        type=float,
        default=0.0,
        help="Additional portfolio-level fee bps applied on turnover (spot-only)",
    )
    parser.add_argument(
        "--portfolio_slippage_bps",
        type=float,
        default=0.0,
        help="Additional portfolio-level slippage bps applied on turnover",
    )
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
        args.panic_fast_halflife_hours,
        args.panic_vol_threshold_annual,
        args.panic_min_bars,
        args.corr_window_hours,
        args.corr_threshold,
        args.corr_leverage_mult,
        args.portfolio_fee_bps,
        args.portfolio_slippage_bps,
        args.enable_panic_switch,
        args.enable_corr_override,
        args.corr_window_slow_hours,
        args.corr_threshold_slow,
        args.corr_window_fast_hours,
        args.corr_threshold_fast,
        args.max_allowed_drawdown,
        args.dd_vol_floor,
        args.enable_dd_targetvol_scaling,
    )
    portfolio.write_parquet(out_dir / "portfolio_equity.parquet")

    m = metrics(portfolio)
    summary = {
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "max_drawdown_ts": m["max_drawdown_ts"].isoformat() if m["max_drawdown_ts"] else None,
        "vol_window_hours": args.vol_window_hours,
        "target_portfolio_vol_annual": args.target_portfolio_vol_annual,
        "port_vol_window_hours": port_window,
        "max_gross_leverage": args.max_gross_leverage,
        "min_gross_leverage": args.min_gross_leverage,
        "port_vol_method": args.port_vol_method,
        "downside_threshold": args.downside_threshold,
        "portfolio_fee_bps": args.portfolio_fee_bps,
        "portfolio_slippage_bps": args.portfolio_slippage_bps,
        "panic_triggers": int(portfolio["panic_triggered"].sum()) if "panic_triggered" in portfolio.columns else 0,
        "corr_overrides": int(portfolio["corr_override_triggered"].sum()) if "corr_override_triggered" in portfolio.columns else 0,
        "corr_slow_triggers": int(
            ((portfolio["corr_slow"] > args.corr_threshold_slow) & portfolio["corr_slow"].is_not_null()).sum()
        )
        if "corr_slow" in portfolio.columns
        else 0,
        "corr_fast_triggers": int(
            ((portfolio["corr_fast"] > args.corr_threshold_fast) & portfolio["corr_fast"].is_not_null()).sum()
        )
        if "corr_fast" in portfolio.columns
        else 0,
        "max_allowed_drawdown": args.max_allowed_drawdown,
        "dd_vol_floor": args.dd_vol_floor,
        "enable_dd_targetvol_scaling": args.enable_dd_targetvol_scaling,
        "scale_dd_min": m.get("scale_dd_min"),
        "scale_dd_median": m.get("scale_dd_median"),
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

