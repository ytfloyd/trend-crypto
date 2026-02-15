#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metrics for kuma_01")
    p.add_argument("--equity", required=True, help="Path to equity.csv")
    p.add_argument("--out", required=True, help="Output metrics CSV path")
    return p.parse_args()


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.sort_values("ts")

    returns = df["portfolio_ret"].fillna(0.0)
    equity = df["portfolio_equity"]

    n_days = len(df)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if n_days > 0 else 0.0
    cagr = (1 + total_return) ** (365.0 / n_days) - 1.0 if n_days > 0 else 0.0
    vol = returns.std(ddof=1) * np.sqrt(365.0) if n_days > 1 else 0.0
    sharpe = (returns.mean() / returns.std(ddof=1)) * np.sqrt(365.0) if returns.std(ddof=1) > 0 else 0.0

    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std(ddof=1)) * np.sqrt(365.0) if downside.std(ddof=1) > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min() if n_days > 0 else 0.0
    avg_dd = drawdown.mean() if n_days > 0 else 0.0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    hit_ratio = float((returns > 0).mean()) if n_days > 0 else 0.0
    expectancy = float(returns.mean()) if n_days > 0 else 0.0

    avg_turnover_one = df["turnover_one_sided"].mean() if "turnover_one_sided" in df.columns else 0.0
    avg_gross = df["gross_exposure"].mean() if "gross_exposure" in df.columns else 0.0

    out = pd.DataFrame(
        [
            {
                "period": "full",
                "start": df["ts"].min(),
                "end": df["ts"].max(),
                "n_days": n_days,
                "total_return": total_return,
                "cagr": cagr,
                "vol": vol,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "avg_dd": avg_dd,
                "max_dd": max_dd,
                "hit_ratio": hit_ratio,
                "expectancy": expectancy,
                "avg_turnover_one_sided": avg_turnover_one,
                "avg_gross_exposure": avg_gross,
            }
        ]
    )
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.equity)
    metrics = compute_metrics(df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)
    print(f"[kuma_01_metrics] Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
