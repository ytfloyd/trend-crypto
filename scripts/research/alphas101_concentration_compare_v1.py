#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if eq.empty:
        return {
            "n_days": 0,
            "total_return": np.nan,
            "cagr": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
        }

    n_days = len(eq)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = n_days / 365.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    daily_ret = eq.pct_change().dropna()
    if daily_ret.empty:
        vol = np.nan
        sharpe = np.nan
    else:
        vol = daily_ret.std() * np.sqrt(365.0)
        if daily_ret.std() > 0:
            sharpe = (daily_ret.mean() * 365.0) / (daily_ret.std() * np.sqrt(365.0))
        else:
            sharpe = np.nan

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = dd.min()

    return {
        "n_days": n_days,
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "nan%"
    return f"{x * 100:.2f}%"


def format_ratio(x: float) -> str:
    if pd.isna(x):
        return "nan"
    return f"{x:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V1 Base vs Growth concentration sleeves."
    )
    parser.add_argument(
        "--base_equity",
        default="artifacts/research/101_alphas/ensemble_equity_v1_base.csv",
        help="Path to base equity CSV (ts, portfolio_equity).",
    )
    parser.add_argument(
        "--growth_equity",
        default="artifacts/research/101_alphas/ensemble_equity_v1_growth.csv",
        help="Path to growth equity CSV (ts, portfolio_equity).",
    )
    parser.add_argument(
        "--start_date",
        default="2021-11-01",
        help="Start date (inclusive) for comparison window.",
    )
    parser.add_argument(
        "--rf_annual",
        type=float,
        default=0.04,
        help="Annual risk-free (not used in stats; kept for symmetry).",
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts/research/101_alphas/metrics_101_ensemble_v1_base_vs_growth.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    base = pd.read_csv(args.base_equity, parse_dates=["ts"]).set_index("ts")[
        "portfolio_equity"
    ]
    growth = pd.read_csv(args.growth_equity, parse_dates=["ts"]).set_index("ts")[
        "portfolio_equity"
    ]

    base = base.loc[pd.to_datetime(args.start_date) :]
    growth = growth.loc[pd.to_datetime(args.start_date) :]

    stats_base = compute_stats(base)
    stats_growth = compute_stats(growth)

    rows = [
        {"label": "v1_base", **stats_base},
        {"label": "v1_growth", **stats_growth},
    ]
    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"V1 Base (from {args.start_date}):")
    print(f"  n_days      : {stats_base['n_days']}")
    print(f"  total_return:  {format_pct(stats_base['total_return'])}")
    print(f"  CAGR        :  {format_pct(stats_base['cagr'])}")
    print(f"  Vol         :  {format_pct(stats_base['vol'])}")
    print(f"  Sharpe      :   {format_ratio(stats_base['sharpe'])}")
    print(f"  MaxDD       : {format_pct(stats_base['max_dd'])}")
    print()
    print(f"V1 Growth (from {args.start_date}):")
    print(f"  n_days      : {stats_growth['n_days']}")
    print(f"  total_return:  {format_pct(stats_growth['total_return'])}")
    print(f"  CAGR        :  {format_pct(stats_growth['cagr'])}")
    print(f"  Vol         :  {format_pct(stats_growth['vol'])}")
    print(f"  Sharpe      :   {format_ratio(stats_growth['sharpe'])}")
    print(f"  MaxDD       : {format_pct(stats_growth['max_dd'])}")
    print()
    delta_cagr = stats_growth["cagr"] - stats_base["cagr"]
    delta_maxdd = stats_growth["max_dd"] - stats_base["max_dd"]
    print("Trade-off (Growth - Base):")
    print(f"  ΔCAGR  :  {format_pct(delta_cagr)} pts")
    print(f"  ΔMaxDD :  {format_pct(delta_maxdd)} pts")


if __name__ == "__main__":
    main()

