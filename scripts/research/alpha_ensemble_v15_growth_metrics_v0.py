#!/usr/bin/env python
from __future__ import annotations

"""
Metrics for Growth Sleeve V1.5 (daily-only v0).

Computes portfolio stats and trade stats:
- CAGR, vol, Sharpe, Sortino, Calmar, avg drawdown, max drawdown
- Daily hit ratio, daily expectancy
- Trade win rate, avg win/avg loss, reward-to-risk
Includes sample start/end.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Optional

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

    vol = float(s.std() * math.sqrt(ANN_FACTOR))
    sharpe = float((cagr - rf_annual) / vol) if vol and not math.isnan(vol) and vol != 0.0 else float("nan")

    if equity_col is not None and equity_col in df.columns:
        equity = df[equity_col].astype(float)
    else:
        equity = (1.0 + s).cumprod()

    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min()) if not dd.empty else float("nan")
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    downside = s[s < 0]
    downside_vol_daily = downside.std()
    downside_vol_ann = downside_vol_daily * math.sqrt(ANN_FACTOR) if downside_vol_daily and not math.isnan(downside_vol_daily) else float("nan")
    sortino = float((cagr - rf_annual) / downside_vol_ann) if downside_vol_ann and not math.isnan(downside_vol_ann) else float("nan")

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


def compute_trade_stats(trades_path: Optional[Path]) -> Dict[str, float]:
    if trades_path is None or not trades_path.exists():
        return {
            "trade_win_rate": float("nan"),
            "trade_avg_win": float("nan"),
            "trade_avg_loss": float("nan"),
            "trade_reward_risk": float("nan"),
        }
    trades = pd.read_parquet(trades_path)
    if trades.empty or "ret" not in trades.columns:
        return {
            "trade_win_rate": float("nan"),
            "trade_avg_win": float("nan"),
            "trade_avg_loss": float("nan"),
            "trade_reward_risk": float("nan"),
        }
    ret = trades["ret"].dropna().astype(float)
    if ret.empty:
        return {
            "trade_win_rate": float("nan"),
            "trade_avg_win": float("nan"),
            "trade_avg_loss": float("nan"),
            "trade_reward_risk": float("nan"),
        }
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    win_rate = float((ret > 0).mean())
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    reward_risk = float(avg_win / abs(avg_loss)) if avg_loss < 0 else float("nan")
    return {
        "trade_win_rate": win_rate,
        "trade_avg_win": avg_win,
        "trade_avg_loss": avg_loss,
        "trade_reward_risk": reward_risk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Metrics for Growth Sleeve V1.5.")
    parser.add_argument(
        "--equity",
        required=True,
        help="Equity CSV from run_alpha_ensemble_v15_growth_backtest_v0.py.",
    )
    parser.add_argument(
        "--trades",
        default=None,
        help="Optional trades parquet (for trade stats).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output metrics CSV path.",
    )
    args = parser.parse_args()

    equity_path = Path(args.equity)
    trades_path = Path(args.trades) if args.trades else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    equity_df = pd.read_csv(equity_path, parse_dates=["ts"])
    equity_df = equity_df.sort_values("ts")
    if equity_df.empty:
        raise SystemExit(f"equity file empty: {equity_path}")

    metrics = compute_metrics_from_returns(equity_df, ret_col="portfolio_ret", equity_col="portfolio_equity")
    metrics["period"] = "full"
    metrics["start"] = equity_df["ts"].min()
    metrics["end"] = equity_df["ts"].max()

    trade_stats = compute_trade_stats(trades_path)
    metrics.update(trade_stats)

    cols = [
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
        "trade_win_rate",
        "trade_avg_win",
        "trade_avg_loss",
        "trade_reward_risk",
    ]

    out_df = pd.DataFrame([metrics])[cols]
    out_df.to_csv(out_path, index=False)
    print(f"[growth_metrics] Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
