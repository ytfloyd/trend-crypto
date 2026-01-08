#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Symbol-level exposure and turnover stats for 101-alphas V1.")
    p.add_argument(
        "--weights",
        default="artifacts/research/101_alphas/ensemble_weights_v0.parquet",
        help="Path to weights parquet (ts, symbol, weight).",
    )
    p.add_argument(
        "--turnover",
        default="artifacts/research/101_alphas/ensemble_turnover_v0.csv",
        help="Path to turnover CSV (ts, turnover).",
    )
    p.add_argument(
        "--equity",
        default="artifacts/research/101_alphas/ensemble_equity_v0.csv",
        help="Path to equity CSV (ts, portfolio_equity) for date grid reference.",
    )
    p.add_argument(
        "--out_symbol",
        default="artifacts/research/101_alphas/alphas101_symbol_stats_v1_adv10m.csv",
        help="Output CSV for symbol-level stats.",
    )
    p.add_argument(
        "--out_top",
        default="artifacts/research/101_alphas/alphas101_symbol_stats_top20_v1_adv10m.csv",
        help="Output CSV for top-N symbols by avg_abs_weight.",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of symbols to include in top-N table (default: 20).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    turnover_path = Path(args.turnover)
    equity_path = Path(args.equity)
    out_symbol = Path(args.out_symbol)
    out_top = Path(args.out_top)

    weights = pd.read_parquet(weights_path)
    turnover = pd.read_csv(turnover_path, parse_dates=["ts"])
    equity = pd.read_csv(equity_path, parse_dates=["ts"])

    if not {"ts", "symbol", "weight"}.issubset(weights.columns):
        raise ValueError("Weights file must contain ts, symbol, weight.")
    if "turnover" not in turnover.columns:
        raise ValueError("Turnover file must contain 'turnover'.")
    if "ts" not in equity.columns:
        raise ValueError("Equity file must contain 'ts'.")

    # Prepare date grid
    dates = equity["ts"].sort_values().unique()

    # Pivot weights to wide (dates x symbols), reindex to equity dates and fill missing with 0
    w_wide = (
        weights.assign(ts=pd.to_datetime(weights["ts"]))
        .pivot(index="ts", columns="symbol", values="weight")
        .reindex(dates)
        .fillna(0.0)
    )
    w_wide = w_wide.sort_index()

    # Compute per-symbol flows and daily turnover (should align with provided turnover)
    delta = w_wide.diff().fillna(0.0).abs()
    symbol_flow = 0.5 * delta  # per-day contribution per symbol
    daily_turnover_calc = symbol_flow.sum(axis=1)

    mean_turnover_reported = turnover["turnover"].mean()
    mean_turnover_calc = daily_turnover_calc.mean()
    ratio = mean_turnover_calc / mean_turnover_reported if mean_turnover_reported else np.nan

    if not (0.98 <= ratio <= 1.02):
        print(
            f"[alphas101_symbol_stats_v1] WARNING: computed mean turnover {mean_turnover_calc:.6f} "
            f"differs from reported {mean_turnover_reported:.6f} (ratio {ratio:.4f})"
        )
    else:
        print(
            f"[alphas101_symbol_stats_v1] Turnover check OK: computed {mean_turnover_calc:.6f}, "
            f"reported {mean_turnover_reported:.6f}, ratio {ratio:.4f}"
        )

    n_days = w_wide.shape[0]

    avg_weight = w_wide.mean()
    avg_abs_weight = w_wide.abs().mean()
    max_abs_weight = w_wide.abs().max()
    days_held = (w_wide.abs() > 0).sum()
    holding_ratio = days_held / n_days
    avg_turnover_contribution = symbol_flow.mean()
    turnover_share_pct = avg_turnover_contribution / mean_turnover_reported

    stats = pd.DataFrame(
        {
            "symbol": avg_weight.index,
            "avg_weight": avg_weight.values,
            "avg_abs_weight": avg_abs_weight.values,
            "max_abs_weight": max_abs_weight.values,
            "days_held": days_held.values.astype(int),
            "holding_ratio": holding_ratio.values,
            "avg_turnover_contribution": avg_turnover_contribution.values,
            "turnover_share_pct": turnover_share_pct.values,
        }
    )

    stats = stats.sort_values("avg_abs_weight", ascending=False)
    out_symbol.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_symbol, index=False)
    print(f"[alphas101_symbol_stats_v1] Wrote symbol stats for {len(stats)} symbols to {out_symbol}")

    top_n = max(1, args.top_n)
    top = stats.head(top_n).copy()
    top.insert(0, "rank_by_abs_weight", range(1, len(top) + 1))
    top.to_csv(out_top, index=False)
    print(f"[alphas101_symbol_stats_v1] Wrote top-{top_n} symbols to {out_top}")
    print(f"[alphas101_symbol_stats_v1] Unique symbols: {stats['symbol'].nunique()}")
    print(f"[alphas101_symbol_stats_v1] Mean daily turnover (reported): {mean_turnover_reported:.6f}")
    print(
        f"[alphas101_symbol_stats_v1] Sum avg_turnover_contribution: "
        f"{avg_turnover_contribution.sum():.6f} (ratio to reported: {ratio:.4f})"
    )


if __name__ == "__main__":
    main()

