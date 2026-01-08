#!/usr/bin/env python

"""
Alphas 101 Ensemble Tearsheet v0

Reads the standard 101_alphas research artifacts and produces a
multi-page PDF tearsheet with:

- Performance summary and net-of-cost metrics
- Equity curve and drawdown
- Rolling Sharpe and turnover
- Regime breakdown (trend/mean_rev/danger)
- Alpha selection / IC panel summary
- Cross-sectional concentration

Assumes you have already run the Phase 4 pipeline to generate:
- ensemble_equity_v0.csv
- ensemble_turnover_v0.csv
- metrics_101_alphas_ensemble_v0.csv
- metrics_101_alphas_ensemble_v0_costs_bps*.csv (optional but recommended)
- alphas101_beta_vs_btc_v0.csv
- alphas101_ic_panel_v0_h1.csv
- alphas101_selected_v0.csv
- alphas101_concentration_summary_v0_sel.csv
- alphas101_regimes_v0.csv
"""

import argparse
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _load_csv(path: str, *, parse_dates: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)


def _load_csv_optional(path: str, *, parse_dates: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    cummax = equity.cummax().replace(0, np.nan)
    dd = equity / cummax - 1.0
    return dd.fillna(0.0)


def _rolling_sharpe(returns: pd.Series, window: int = 90, ann_factor: float = 365.0) -> pd.Series:
    roll = returns.rolling(window=window)
    mean = roll.mean()
    std = roll.std()
    sharpe = (mean * ann_factor) / (std * np.sqrt(ann_factor))
    return sharpe


def _auto_discover_tca(research_dir: str) -> pd.DataFrame:
    """
    Look for V1/V0 cost metrics files in research_dir and return a combined DataFrame sorted by cost_bps.
    """
    rows = []
    patterns = [
        "metrics_101_ensemble_filtered_v1_costs_bps",
        "metrics_101_alphas_ensemble_v0_costs_bps",
    ]
    for fn in os.listdir(research_dir):
        if not fn.endswith(".csv"):
            continue
        if not any(fn.startswith(p) for p in patterns):
            continue
        try:
            bps_str = fn.split("bps", 1)[1].split(".csv", 1)[0]
            cost_bps = int(bps_str)
        except Exception:
            continue
        df = pd.read_csv(os.path.join(research_dir, fn))
        if df.empty:
            continue
        row = df.iloc[0].copy()
        row["cost_bps"] = cost_bps
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    tca = pd.DataFrame(rows)
    tca = tca.sort_values("cost_bps").reset_index(drop=True)
    return tca


def make_tearsheet(
    research_dir: str,
    out_pdf: str,
    equity_path: Optional[str] = None,
    turnover_path: Optional[str] = None,
    base_metrics_path: Optional[str] = None,
    beta_path: Optional[str] = None,
    ic_panel_path: Optional[str] = None,
    selection_path: Optional[str] = None,
    concentration_path: Optional[str] = None,
    regimes_path: Optional[str] = None,
    capacity_path: Optional[str] = None,
    symbol_stats_top_path: Optional[str] = None,
) -> None:
    # Resolve paths (with sensible defaults)
    rd = research_dir
    equity_path = equity_path or os.path.join(rd, "ensemble_equity_v0.csv")
    turnover_path = turnover_path or os.path.join(rd, "ensemble_turnover_v0.csv")

    # Prefer V1 metrics if present
    base_metrics_path = base_metrics_path or (
        os.path.join(rd, "metrics_101_ensemble_filtered_v1.csv")
        if os.path.exists(os.path.join(rd, "metrics_101_ensemble_filtered_v1.csv"))
        else os.path.join(rd, "metrics_101_alphas_ensemble_v0.csv")
    )
    beta_path = beta_path or (
        os.path.join(rd, "alphas101_beta_vs_btc_v1_adv10m.csv")
        if os.path.exists(os.path.join(rd, "alphas101_beta_vs_btc_v1_adv10m.csv"))
        else os.path.join(rd, "alphas101_beta_vs_btc_v0.csv")
    )
    ic_panel_path = ic_panel_path or os.path.join(rd, "alphas101_ic_panel_v0_h1.csv")
    selection_path = selection_path or (
        os.path.join(rd, "alphas101_selected_v1_adv10m.csv")
        if os.path.exists(os.path.join(rd, "alphas101_selected_v1_adv10m.csv"))
        else os.path.join(rd, "alphas101_selected_v0.csv")
    )
    # Prefer selection-aware V1 concentration summary if present
    conc_default = os.path.join(rd, "alphas101_concentration_summary_v1_adv10m.csv")
    if not os.path.exists(conc_default):
        conc_default = os.path.join(rd, "alphas101_concentration_summary_v0_sel.csv")
    if os.path.exists(conc_default):
        concentration_path = concentration_path or conc_default
    else:
        concentration_path = concentration_path or os.path.join(rd, "alphas101_concentration_summary_v0.csv")

    regimes_path = regimes_path or (
        os.path.join(rd, "alphas101_regimes_v1_adv10m.csv")
        if os.path.exists(os.path.join(rd, "alphas101_regimes_v1_adv10m.csv"))
        else os.path.join(rd, "alphas101_regimes_v0.csv")
    )

    # Core series
    equity = _load_csv(equity_path, parse_dates=["ts"])
    turnover = _load_csv(turnover_path, parse_dates=["ts"])
    base_metrics = _load_csv(base_metrics_path)
    beta_df = _load_csv_optional(beta_path)
    ic_panel = _load_csv(ic_panel_path)
    selection = _load_csv(selection_path)
    concentration = _load_csv(concentration_path)
    regimes = _load_csv(regimes_path, parse_dates=["ts"])
    tca = _auto_discover_tca(rd)
    if capacity_path is None:
        cap_default = os.path.join(rd, "capacity_sensitivity_v1.csv")
        if os.path.exists(cap_default):
            capacity_path = cap_default
    capacity = _load_csv_optional(capacity_path) if capacity_path else None
    if symbol_stats_top_path is None:
        sym_default = os.path.join(rd, "alphas101_symbol_stats_top20_v1_adv10m.csv")
        if os.path.exists(sym_default):
            symbol_stats_top_path = sym_default
    symbol_stats_top = _load_csv_optional(symbol_stats_top_path) if symbol_stats_top_path else None

    # Join regimes into equity
    equity = equity.sort_values("ts").reset_index(drop=True)
    regimes = regimes.sort_values("ts").reset_index(drop=True)
    equity = equity.merge(regimes[["ts", "regime"]], on="ts", how="left")

    # Derived series
    equity["drawdown"] = _compute_drawdown(equity["portfolio_equity"])
    equity["rolling_sharpe_90d"] = _rolling_sharpe(equity["portfolio_ret"], window=90)
    equity["rolling_vol_90d"] = equity["portfolio_ret"].rolling(window=90).std() * np.sqrt(365.0)
    # Merge turnover
    turnover = turnover.sort_values("ts")
    equity = equity.merge(turnover[["ts", "turnover"]], on="ts", how="left")

    # Regime stats
    regime_stats = (
        equity.groupby("regime")["portfolio_ret"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n_days"})
        .reset_index()
    )
    regime_stats["ann_return"] = (1.0 + regime_stats["mean"]) ** 365 - 1.0
    regime_stats["ann_vol"] = regime_stats["std"] * np.sqrt(365.0)
    regime_stats["sharpe"] = regime_stats["ann_return"] / regime_stats["ann_vol"]

    # Selection / IC
    # selection: ['alpha','sign','n_days','mean_ic','std_ic','tstat_ic','mean_ic_oriented']
    selected_sorted = selection.sort_values("mean_ic_oriented", ascending=False).reset_index(drop=True)

    # Merge IC panel + selection to show where selection sits in the universe
    ic_full = ic_panel.merge(
        selection[["alpha", "sign", "mean_ic_oriented"]],
        on="alpha",
        how="left",
        suffixes=("", "_sel"),
    )
    ic_full["selected"] = ~ic_full["sign"].isna()

    # Start PDF
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # Page 1: Title + summary tables
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        title = "Alpha 101 Ensemble – USD Spot Universe (ex-stablecoins)"
        subtitle = "Daily long+cash ensemble over Coinbase USD pairs | selection + regime gating"

        # Base metrics row 'full'
        full_row = base_metrics.loc[base_metrics["period"] == "full"].iloc[0]
        summary_lines = [
            f"CAGR: {full_row['cagr']:.2%}",
            f"Vol (ann.): {full_row['vol']:.2%}",
            f"Sharpe: {full_row['sharpe']:.2f}",
            f"Max drawdown: {full_row['max_dd']:.2%}",
            f"Sample length: {int(full_row['n_days'])} days",
        ]

        text_y = 0.9
        ax.text(0.02, text_y, title, fontsize=18, fontweight="bold", transform=ax.transAxes)
        text_y -= 0.05
        ax.text(0.02, text_y, subtitle, fontsize=11, transform=ax.transAxes)

        text_y -= 0.08
        ax.text(
            0.02,
            text_y,
            "Performance summary (before transaction costs):",
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )
        text_y -= 0.04
        for line in summary_lines:
            ax.text(0.04, text_y, f"• {line}", fontsize=11, transform=ax.transAxes)
            text_y -= 0.035

        # Net-of-cost metrics if present
        if not tca.empty:
            text_y -= 0.03
            ax.text(
                0.02,
                text_y,
                "Net-of-cost performance (per-side cost assumption):",
                fontsize=12,
                fontweight="bold",
                transform=ax.transAxes,
            )
            text_y -= 0.04
            for _, row in tca.iterrows():
                line = (
                    f"{int(row['cost_bps'])} bps: "
                    f"CAGR {row['cagr']:.2%}, "
                    f"Sharpe {row['sharpe']:.2f}, "
                    f"MaxDD {row['max_dd']:.2%}"
                )
                ax.text(0.04, text_y, f"• {line}", fontsize=11, transform=ax.transAxes)
                text_y -= 0.032

        # Beta vs BTC
        if beta_df is not None and not beta_df.empty:
            row = beta_df.iloc[0]
            text_y -= 0.03
            ax.text(0.02, text_y, "Beta / correlation vs BTC-USD:", fontsize=12, fontweight="bold", transform=ax.transAxes)
            text_y -= 0.04
            beta_line = (
                f"Corr={row['corr']:.2f}, Beta={row['beta']:.2f}, "
                f"R²={row['r2']:.3f}, t_beta={row['t_beta']:.2f}"
            )
            ax.text(0.04, text_y, f"• {beta_line}", fontsize=11, transform=ax.transAxes)

        # Regime stats overview
        text_y -= 0.06
        ax.text(0.02, text_y, "Regime breakdown (daily):", fontsize=12, fontweight="bold", transform=ax.transAxes)
        text_y -= 0.04
        for _, row in regime_stats.iterrows():
            line = (
                f"{row['regime']}: "
                f"{int(row['n_days'])} days, "
                f"ann. return {row['ann_return']:.2%}, "
                f"ann. vol {row['ann_vol']:.2%}, "
                f"Sharpe {row['sharpe']:.2f}"
            )
            ax.text(0.04, text_y, f"• {line}", fontsize=11, transform=ax.transAxes)
            text_y -= 0.032

        # Capacity sensitivity (optional)
        if capacity is not None and not capacity.empty:
            text_y -= 0.03
            ax.text(0.02, text_y, "Capacity sensitivity (per-side cost sweep):", fontsize=12, fontweight="bold", transform=ax.transAxes)
            text_y -= 0.04
            for _, row in capacity.iterrows():
                line = (
                    f"{int(row['cost_bps'])} bps: Sharpe {row['sharpe']:.2f}, "
                    f"CAGR {row['cagr']:.2%}, MaxDD {row['max_dd']:.2%}"
                )
                ax.text(0.04, text_y, f"• {line}", fontsize=11, transform=ax.transAxes)
                text_y -= 0.032

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Equity & drawdown
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        axes[0].plot(equity["ts"], equity["portfolio_equity"])
        axes[0].set_ylabel("Equity (nav)")
        axes[0].set_title("Equity curve")

        axes[1].plot(equity["ts"], equity["drawdown"])
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Date")
        axes[1].set_title("Drawdown (from peak)")

        for ax_ in axes:
            ax_.grid(True, alpha=0.3)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Rolling risk & turnover
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

        axes[0].plot(equity["ts"], equity["rolling_sharpe_90d"])
        axes[0].set_ylabel("Sharpe (90d)")
        axes[0].set_title("Rolling 90-day Sharpe")

        axes[1].plot(equity["ts"], equity["rolling_vol_90d"])
        axes[1].set_ylabel("Ann. vol (90d)")
        axes[1].set_title("Rolling 90-day annualized volatility")

        axes[2].plot(equity["ts"], equity["turnover"])
        axes[2].set_ylabel("Turnover")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Daily turnover")

        for ax_ in axes:
            ax_.grid(True, alpha=0.3)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Regimes & exposures
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

        # Regime-colored returns
        colors = {"trend": "tab:green", "mean_rev": "tab:blue", "danger": "tab:red"}
        for regime_name, group in equity.groupby("regime"):
            axes[0].scatter(
                group["ts"],
                group["portfolio_ret"],
                s=8,
                label=regime_name,
                alpha=0.7,
                color=colors.get(regime_name, "gray"),
            )
        axes[0].axhline(0.0, color="black", linewidth=0.5)
        axes[0].set_ylabel("Daily return")
        axes[0].set_title("Daily returns by regime")
        axes[0].legend()

        # Gross long & cash weight
        axes[1].plot(equity["ts"], equity["gross_long"], label="Gross long")
        axes[1].plot(equity["ts"], equity["cash_weight"], label="Cash weight")
        axes[1].set_ylabel("Weight")
        axes[1].set_title("Gross exposure vs cash")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Regime occupancy time series (stacked area, 30d rolling)
        regime_dummies = pd.get_dummies(equity["regime"])
        regime_share = regime_dummies.rolling(window=30, min_periods=1).mean()
        axes[2].stackplot(
            equity["ts"],
            *[regime_share[col] for col in regime_share.columns],
            labels=list(regime_share.columns),
        )
        axes[2].set_ylabel("Share (30d rolling)")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Regime occupancy (30-day rolling share)")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Alpha IC / selection
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))

        # Top oriented IC alphas (selected)
        top_sel = selected_sorted.head(20)
        axes[0].bar(top_sel["alpha"], top_sel["mean_ic_oriented"])
        axes[0].set_ylabel("Mean IC (oriented)")
        axes[0].set_title("Top selected alphas by oriented mean IC (horizon 1)")
        axes[0].tick_params(axis="x", rotation=90)
        axes[0].grid(True, axis="y", alpha=0.3)

        # IC distribution (all alphas)
        axes[1].hist(ic_full["mean_ic"], bins=30, alpha=0.7, label="All alphas")
        axes[1].hist(
            ic_full.loc[ic_full["selected"], "mean_ic"],
            bins=30,
            alpha=0.7,
            label="Selected",
        )
        axes[1].set_xlabel("Mean IC (horizon 1)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of mean IC (all vs selected alphas)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: Concentration (top names)
        fig, ax = plt.subplots(figsize=(11, 8.5))

        top_names = concentration.head(20).copy()
        ax.bar(top_names["symbol"], top_names["avg_abs_weight"])
        ax.set_ylabel("Avg |weight|")
        ax.set_title("Top symbols by average absolute weight")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(True, axis="y", alpha=0.3)

        # BTC/ETH share (single number from concentration table)
        if "btc_eth_share_total" in concentration.columns and not concentration["btc_eth_share_total"].isna().all():
            btc_eth_share = concentration["btc_eth_share_total"].iloc[0]
            ax.text(
                0.02,
                0.95,
                f"BTC+ETH share of total |weight|: {btc_eth_share:.1%}",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 7: Symbol exposure & turnover (top 20) if available
        if symbol_stats_top is not None and not symbol_stats_top.empty:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")

            cols = ["symbol", "avg_abs_weight", "holding_ratio", "turnover_share_pct"]
            tbl = symbol_stats_top[cols].copy()
            tbl["avg_abs_weight"] = tbl["avg_abs_weight"].map(lambda x: f"{x:.4f}")
            tbl["holding_ratio"] = tbl["holding_ratio"].map(lambda x: f"{x:.2%}")
            tbl["turnover_share_pct"] = tbl["turnover_share_pct"].map(lambda x: f"{x*100:.2f}%")

            table = ax.table(
                cellText=tbl.values,
                colLabels=tbl.columns,
                loc="center",
                cellLoc="center",
            )
            table.scale(1, 1.5)
            ax.set_title("Symbol Exposure & Turnover (Top 20)", fontsize=14, pad=10)
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print("[alphas101_tearsheet_v0] Symbol stats not found; skipping symbol exposure page.")

        # Page 8: Return distribution
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.hist(equity["portfolio_ret"], bins=50, alpha=0.8)
        ax.set_xlabel("Daily return")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of daily returns")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PDF tearsheet for 101_alphas ensemble.")
    parser.add_argument(
        "--research_dir",
        type=str,
        default="artifacts/research/101_alphas",
        help="Root directory containing 101_alphas artifacts.",
    )
    parser.add_argument(
        "--out_pdf",
        type=str,
        default=None,
        help="Output PDF path (default: <research_dir>/alphas101_tearsheet_v0.pdf)",
    )
    parser.add_argument(
        "--equity",
        type=str,
        default=None,
        help="Optional override path for ensemble_equity_v0.csv",
    )
    parser.add_argument(
        "--turnover",
        type=str,
        default=None,
        help="Optional override path for ensemble_turnover_v0.csv",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Optional override path for metrics_101_alphas_ensemble_v0.csv",
    )
    parser.add_argument(
        "--beta",
        type=str,
        default=None,
        help="Optional override path for alphas101_beta_vs_btc_v0.csv",
    )
    parser.add_argument(
        "--ic_panel",
        type=str,
        default=None,
        help="Optional override path for alphas101_ic_panel_v0_h1.csv",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default=None,
        help="Optional override path for alphas101_selected_v0.csv",
    )
    parser.add_argument(
        "--concentration",
        type=str,
        default=None,
        help="Optional override path for alphas101_concentration_summary_v0_sel.csv",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default=None,
        help="Optional override path for alphas101_regimes_v0.csv",
    )
    parser.add_argument(
        "--capacity",
        type=str,
        default=None,
        help="Optional override path for capacity_sensitivity_v1.csv",
    )
    args = parser.parse_args()

    out_pdf = args.out_pdf or os.path.join(args.research_dir, "alphas101_tearsheet_v0.pdf")

    make_tearsheet(
        research_dir=args.research_dir,
        out_pdf=out_pdf,
        equity_path=args.equity,
        turnover_path=args.turnover,
        base_metrics_path=args.metrics,
        beta_path=args.beta,
        ic_panel_path=args.ic_panel,
        selection_path=args.selection,
        concentration_path=args.concentration,
        regimes_path=args.regimes,
        capacity_path=args.capacity,
    )


if __name__ == "__main__":
    main()

