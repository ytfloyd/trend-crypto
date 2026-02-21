#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ANN_FACTOR = 365.0


def compute_metrics_from_returns(
    df: pd.DataFrame,
    ret_col: str,
    equity_col: Optional[str] = None,
    rf_annual: float = 0.0,
) -> Dict[str, float]:
    if df.empty:
        return {
            "n_days": 0,
            "total_return": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "sortino": float("nan"),
            "calmar": float("nan"),
            "avg_dd": float("nan"),
            "hit_ratio": float("nan"),
            "expectancy": float("nan"),
        }

    s = df[ret_col].astype(float)
    n = int(s.shape[0])
    total_ret = float((1.0 + s).prod() - 1.0)

    try:
        cagr = (1.0 + total_ret) ** (ANN_FACTOR / n) - 1.0
    except Exception:
        cagr = float("nan")

    daily_std = float(s.std())
    vol = daily_std * math.sqrt(ANN_FACTOR)
    daily_mean = float(s.mean())
    rf_bar = (1 + rf_annual) ** (1 / ANN_FACTOR) - 1 if rf_annual > 0 else 0.0
    sharpe = float(((daily_mean - rf_bar) / daily_std) * math.sqrt(ANN_FACTOR)) if daily_std > 1e-12 else float("nan")

    if equity_col is not None and equity_col in df.columns:
        equity = df[equity_col].astype(float)
    else:
        equity = (1.0 + s).cumprod()

    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min()) if not dd.empty else float("nan")
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    downside = s[s < 0]
    downside_vol_daily = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = float(((daily_mean - rf_bar) / downside_vol_daily) * math.sqrt(ANN_FACTOR)) if downside_vol_daily > 1e-12 else float("nan")

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
        "n_days": n,
        "total_return": total_ret,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "sortino": sortino,
        "calmar": calmar,
        "avg_dd": avg_dd,
        "hit_ratio": hit_ratio,
        "expectancy": expectancy,
    }


def compute_portfolio_metrics(equity_path: Path) -> pd.DataFrame:
    df = pd.read_csv(equity_path, parse_dates=["ts"])
    df = df.sort_values("ts")
    if df.empty:
        raise RuntimeError(f"Ensemble equity is empty: {equity_path}")

    min_ts = df["ts"].min()
    max_ts = df["ts"].max()

    periods: List[Tuple[str, pd.Timestamp, Optional[pd.Timestamp]]] = [
        ("full", min_ts, max_ts),
        ("pre_2020", pd.Timestamp("1900-01-01"), pd.Timestamp("2019-12-31")),
        ("2020_2021", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
        ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        ("2023_plus", pd.Timestamp("2023-01-01"), None),
    ]

    rows: List[Dict[str, float]] = []
    for name, start, end in periods:
        start_clipped = max(start, min_ts)
        end_clipped = max_ts if end is None else min(end, max_ts)
        if start_clipped > end_clipped:
            continue
        mask = (df["ts"] >= start_clipped) & (df["ts"] <= end_clipped)
        slice_df = df.loc[mask].copy()
        if slice_df.empty:
            continue
        metrics = compute_metrics_from_returns(
            slice_df, ret_col="portfolio_ret", equity_col="portfolio_equity"
        )
        metrics["period"] = name
        metrics["start"] = start_clipped
        metrics["end"] = end_clipped
        rows.append(metrics)

    return pd.DataFrame(rows)[
        [
            "period",
            "start",
            "end",
            "n_days",
            "total_return",
            "cagr",
            "vol",
            "sharpe",
            "sortino",
            "calmar",
            "avg_dd",
            "hit_ratio",
            "expectancy",
            "max_dd",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metrics for 101 Alphas ensemble."
    )
    parser.add_argument(
        "--equity",
        type=str,
        default="artifacts/research/101_alphas/ensemble_equity_v0.csv",
        help="Path to ensemble equity CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/research/101_alphas/metrics_101_alphas_ensemble_v0.csv",
        help="Output metrics CSV.",
    )

    args = parser.parse_args()
    equity_path = Path(args.equity)
    if not equity_path.exists():
        raise FileNotFoundError(f"Equity file not found: {equity_path}")

    metrics_df = compute_portfolio_metrics(equity_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path, index=False)
    print(f"[alphas101_metrics_v0] Wrote ensemble metrics to {out_path}")

    # Also write explicit V1 remediation metrics filename alongside the main output.
    v1_path = out_path.parent / "metrics_101_ensemble_filtered_v1.csv"
    metrics_df.to_csv(v1_path, index=False)
    print(f"[alphas101_metrics_v0] Wrote V1 metrics to {v1_path}")


if __name__ == "__main__":
    main()

