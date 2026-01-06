#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply per-turnover transaction costs to ensemble returns and recompute metrics."
    )
    p.add_argument(
        "--equity",
        required=True,
        help="Path to ensemble equity CSV (e.g. artifacts/research/101_alphas/ensemble_equity_v0.csv)",
    )
    p.add_argument(
        "--turnover",
        required=True,
        help="Path to ensemble turnover CSV (e.g. artifacts/research/101_alphas/ensemble_turnover_v0.csv)",
    )
    p.add_argument(
        "--cost_bps",
        type=float,
        required=True,
        help="Cost in basis points per unit turnover (e.g. 10 for 10 bps).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output CSV path for net-of-cost metrics.",
    )
    return p.parse_args()


def compute_metrics(ret: pd.Series, ann_factor: int = 365) -> dict:
    ret = pd.Series(ret).dropna()
    n = len(ret)
    if n == 0:
        raise ValueError("No returns to compute metrics on.")

    ann = 365
    total_ret = (1.0 + ret).prod() - 1.0
    cagr = (1.0 + total_ret) ** (ann / n) - 1.0

    vol = ret.std() * np.sqrt(ann)
    sharpe = cagr / vol if vol and vol > 0 else np.nan

    eq = (1.0 + ret).cumprod()
    max_dd = (eq / eq.cummax() - 1.0).min()

    return dict(
        n_days=n,
        total_return=total_ret,
        cagr=cagr,
        vol=vol,
        sharpe=sharpe,
        max_dd=max_dd,
    )


def main() -> None:
    args = parse_args()
    eq_path = Path(args.equity)
    to_path = Path(args.turnover)
    out_path = Path(args.out)

    equity = pd.read_csv(eq_path, parse_dates=["ts"])
    if "portfolio_ret" not in equity.columns:
        raise ValueError("Expected 'portfolio_ret' in equity file.")
    turnover = pd.read_csv(to_path, parse_dates=["ts"])
    if "turnover" not in turnover.columns:
        raise ValueError("Expected 'turnover' in turnover file.")

    df = pd.merge(
        equity[["ts", "portfolio_ret"]],
        turnover[["ts", "turnover"]],
        on="ts",
        how="inner",
    ).sort_values("ts")
    if df.empty:
        raise ValueError("No overlapping dates between equity and turnover.")

    cost_per_unit = args.cost_bps / 1e4  # bps -> decimal
    df["cost_ret"] = df["turnover"] * cost_per_unit
    df["net_ret"] = df["portfolio_ret"] - df["cost_ret"]

    metrics = compute_metrics(df["net_ret"])
    period_label = f"full_net_cost{int(args.cost_bps)}bps"

    out = pd.DataFrame.from_records(
        [
            dict(
                period=period_label,
                **metrics,
            )
        ]
    )
    out.to_csv(out_path, index=False)
    print(f"[alphas101_tca_v0] Wrote net-of-cost metrics to {out_path}")


if __name__ == "__main__":
    main()

