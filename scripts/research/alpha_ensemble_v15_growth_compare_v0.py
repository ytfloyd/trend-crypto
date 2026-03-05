#!/usr/bin/env python
from __future__ import annotations

"""
Comparison helper: Growth V1.5 vs V1 base vs ETH buy & hold.

Reads:
- Growth metrics CSV (full period row)
- V1 base metrics CSV (full period row)
- ETH benchmark equity CSV (ts,equity) to compute metrics

Writes a comparison CSV and prints a brief console summary.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ANN_FACTOR = 365.0


def compute_metrics_from_returns(
    s: pd.Series,
    equity: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
) -> Dict[str, float]:
    s = s.dropna()
    if s.empty:
        return {k: float("nan") for k in ["cagr", "vol", "sharpe", "sortino", "calmar", "avg_dd", "hit_ratio", "expectancy", "max_dd"]}

    n = int(s.shape[0])
    total_ret = float((1.0 + s).prod() - 1.0)

    try:
        cagr = (1.0 + total_ret) ** (ANN_FACTOR / n) - 1.0
    except Exception:
        cagr = float("nan")

    vol = float(s.std() * math.sqrt(ANN_FACTOR))
    daily_std = float(s.std())
    daily_mean = float(s.mean())
    rf_daily = (1.0 + rf_annual) ** (1.0 / ANN_FACTOR) - 1.0
    sharpe = float((daily_mean - rf_daily) / daily_std * math.sqrt(ANN_FACTOR)) if daily_std > 1e-12 else float("nan")

    if equity is None:
        equity = (1.0 + s).cumprod()
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min()) if not dd.empty else float("nan")
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    downside = s[s < 0]
    down_vol = downside.std()
    down_vol_ann = down_vol * math.sqrt(ANN_FACTOR) if down_vol and not math.isnan(down_vol) else float("nan")
    downside_std = float(down_vol) if down_vol and not math.isnan(down_vol) else float("nan")
    sortino = float((daily_mean - rf_daily) / downside_std * math.sqrt(ANN_FACTOR)) if downside_std > 1e-12 else float("nan")
    calmar = float(cagr / abs(max_dd)) if max_dd and max_dd < 0 else float("nan")

    hit_ratio = float((s > 0).mean()) if not s.empty else float("nan")
    wins = s[s > 0]
    losses = s[s < 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    p_win = hit_ratio if not math.isnan(hit_ratio) else 0.0
    p_loss = 1.0 - p_win
    expectancy = float(p_win * avg_win + p_loss * avg_loss)

    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "avg_dd": avg_dd,
        "hit_ratio": hit_ratio,
        "expectancy": expectancy,
        "max_dd": max_dd,
    }


def pick_full_row(df: pd.DataFrame) -> pd.Series:
    if "period" in df.columns and (df["period"] == "full").any():
        return df[df["period"] == "full"].iloc[0]
    return df.iloc[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Growth V1.5 vs V1 base vs ETH buy & hold.")
    p.add_argument(
        "--growth_metrics",
        default="artifacts/research/alpha_ensemble_v15_growth/metrics_growth_v15_v0.csv",
        help="Growth V1.5 metrics CSV.",
    )
    p.add_argument(
        "--v1_metrics",
        default="artifacts/research/101_alphas/metrics_101_ensemble_filtered_v1.csv",
        help="V1 base metrics CSV (period column expected).",
    )
    p.add_argument(
        "--eth_equity",
        default="artifacts/research/alpha_ensemble_v15_growth/benchmark_eth_usd_equity_v0.csv",
        help="ETH buy & hold equity CSV (ts,equity).",
    )
    p.add_argument("--out", required=True, help="Output comparison CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    growth_df = pd.read_csv(args.growth_metrics, parse_dates=["start", "end"])
    v1_df = pd.read_csv(args.v1_metrics, parse_dates=["start", "end"])
    eth_eq = pd.read_csv(args.eth_equity, parse_dates=["ts"])

    growth_row = pick_full_row(growth_df)
    v1_row = pick_full_row(v1_df)

    eth_eq = eth_eq.sort_values("ts")
    eth_ret = eth_eq["equity"].pct_change().dropna()
    eth_metrics = compute_metrics_from_returns(eth_ret, equity=eth_eq["equity"].iloc[1:])

    rows = [
        {
            "label": "Growth V1.5",
            "cagr": growth_row.get("cagr"),
            "vol": growth_row.get("vol"),
            "sharpe": growth_row.get("sharpe"),
            "sortino": growth_row.get("sortino"),
            "calmar": growth_row.get("calmar"),
            "avg_dd": growth_row.get("avg_dd"),
            "hit_ratio": growth_row.get("hit_ratio"),
            "expectancy": growth_row.get("expectancy"),
            "max_dd": growth_row.get("max_dd"),
            "trade_win_rate": growth_row.get("trade_win_rate", float("nan")),
            "trade_reward_risk": growth_row.get("trade_reward_risk", float("nan")),
        },
        {
            "label": "Alpha Ensemble V1",
            "cagr": v1_row.get("cagr"),
            "vol": v1_row.get("vol"),
            "sharpe": v1_row.get("sharpe"),
            "sortino": v1_row.get("sortino"),
            "calmar": v1_row.get("calmar"),
            "avg_dd": v1_row.get("avg_dd"),
            "hit_ratio": v1_row.get("hit_ratio"),
            "expectancy": v1_row.get("expectancy"),
            "max_dd": v1_row.get("max_dd"),
            "trade_win_rate": v1_row.get("trade_win_rate", float("nan")),
            "trade_reward_risk": v1_row.get("trade_reward_risk", float("nan")),
        },
        {
            "label": "ETH Buy & Hold",
            **eth_metrics,
            "trade_win_rate": float("nan"),
            "trade_reward_risk": float("nan"),
        },
    ]

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("\nComparison (CAGR/Vol/Sharpe/Sortino/Calmar/MaxDD):")
    for _, row in out_df.iterrows():
        print(
            f"{row['label']}: "
            f"CAGR {row['cagr']:.2%} | Vol {row['vol']:.2%} | Sharpe {row['sharpe']:.2f} | "
            f"Sortino {row['sortino']:.2f} | Calmar {row['calmar']:.2f} | MaxDD {row['max_dd']:.2%}"
        )
    print(f"\nWrote comparison CSV to {out_path}")


if __name__ == "__main__":
    main()
