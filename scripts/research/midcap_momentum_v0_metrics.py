#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import pandas as pd

ANN_FACTOR = 365.0


def compute_metrics_from_returns(
    df: pd.DataFrame,
    ret_col: str,
    equity_col: Optional[str] = None,
) -> Dict[str, float]:
    """Compute basic performance metrics from a daily returns series."""
    if df.empty:
        return {
            "n_days": 0,
            "total_return": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
        }

    s = df[ret_col].astype(float)
    n = int(s.shape[0])

    total_ret = float((1.0 + s).prod() - 1.0)

    # CAGR
    try:
        cagr = (1.0 + total_ret) ** (ANN_FACTOR / n) - 1.0
    except Exception:
        cagr = float("nan")

    # Vol and Sharpe
    vol = float(s.std() * math.sqrt(ANN_FACTOR))
    daily_std = float(s.std())
    sharpe = float((s.mean() / daily_std) * math.sqrt(ANN_FACTOR)) if daily_std > 1e-12 else float("nan")

    # Equity curve and max drawdown
    if equity_col is not None and equity_col in df.columns:
        equity = df[equity_col].astype(float)
    else:
        equity = (1.0 + s).cumprod()

    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min()) if not dd.empty else float("nan")

    return {
        "n_days": n,
        "total_return": total_ret,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def apply_vol_target(
    df: pd.DataFrame,
    ret_col: str,
    target_vol: float = 0.30,
    window: int = 30,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """Apply a simple rolling vol-target overlay to a return series."""
    out = df.copy()
    s = out[ret_col].astype(float)

    rolling_vol = s.rolling(window).std() * math.sqrt(ANN_FACTOR)
    scale = target_vol / rolling_vol
    scale = scale.clip(upper=max_leverage)
    scale = scale.fillna(1.0)

    scaled_ret = s * scale
    out[ret_col + "_vt"] = scaled_ret
    out["portfolio_equity_vt"] = (1.0 + scaled_ret).cumprod()
    return out


def compute_portfolio_metrics(
    portfolio_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute full-sample and sub-period metrics, plus vol-targeted metrics."""
    df = pd.read_csv(portfolio_path, parse_dates=["ts"])
    df = df.sort_values("ts")

    if df.empty:
        raise RuntimeError(f"Portfolio file is empty: {portfolio_path}")

    # Full-sample metrics
    full_metrics = compute_metrics_from_returns(
        df, ret_col="portfolio_ret", equity_col="portfolio_equity"
    )
    full_metrics["period"] = "full"

    # Sub-period metrics
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()

    # Define desired sub-periods; we will clip to [min_ts, max_ts].
    raw_periods: List[Tuple[str, pd.Timestamp, Optional[pd.Timestamp]]] = [
        ("pre_2020", pd.Timestamp("1900-01-01"), pd.Timestamp("2019-12-31")),
        ("2020_2021", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
        ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        ("2023_plus", pd.Timestamp("2023-01-01"), None),
    ]

    rows: List[Dict[str, float]] = [full_metrics]

    for name, start, end in raw_periods:
        # Clip to actual data range
        start_clipped = max(start, min_ts)
        end_clipped = max_ts if end is None else min(end, max_ts)
        if start_clipped > end_clipped:
            continue

        mask = (df["ts"] >= start_clipped) & (df["ts"] <= end_clipped)
        df_slice = df.loc[mask].copy()
        if df_slice.empty:
            continue

        m = compute_metrics_from_returns(
            df_slice, ret_col="portfolio_ret", equity_col="portfolio_equity"
        )
        m["period"] = name
        rows.append(m)

    metrics_df = pd.DataFrame(rows)[
        ["period", "n_days", "total_return", "cagr", "vol", "sharpe", "max_dd"]
    ]

    # Vol-targeted series: apply to full df, then compute metrics.
    df_vt = apply_vol_target(df, ret_col="portfolio_ret")
    vt_metrics = compute_metrics_from_returns(
        df_vt, ret_col="portfolio_ret_vt", equity_col="portfolio_equity_vt"
    )
    vt_metrics["period"] = "full_vt"

    vt_df = pd.DataFrame([vt_metrics])[
        ["period", "n_days", "total_return", "cagr", "vol", "sharpe", "max_dd"]
    ]

    return metrics_df, vt_df


def summarize_midcaps_cross_section(summary_path: Path) -> pd.DataFrame:
    """Load midcap summary.csv and add ranking columns."""
    df = pd.read_csv(summary_path)

    required_cols = ["symbol", "total_return", "sharpe", "max_drawdown"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Midcap summary at {summary_path} is missing columns: {missing}"
        )

    df["rank_sharpe"] = df["sharpe"].rank(ascending=False, method="min").astype(int)
    df["rank_total_return"] = df["total_return"].rank(
        ascending=False, method="min"
    ).astype(int)
    # Less negative max_dd is better -> ascending rank
    df["rank_max_dd"] = df["max_drawdown"].rank(ascending=True, method="min").astype(int)

    df = df.sort_values(["rank_sharpe", "rank_total_return"])
    return df


def build_benchmark_comparison(
    benchmark_summary_path: Path,
    portfolio_full_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a comparison table for BTC-USD / ETH-USD vs midcap portfolio.
    Expects benchmark_summary_path CSV with columns:
      symbol,total_return,sharpe,max_drawdown,...
    """
    df_bench = pd.read_csv(benchmark_summary_path)

    required_cols = ["symbol", "total_return", "sharpe", "max_drawdown"]
    missing = [c for c in required_cols if c not in df_bench.columns]
    if missing:
        raise ValueError(
            f"Benchmark summary at {benchmark_summary_path} is missing columns: {missing}"
        )

    df_bench = df_bench[df_bench["symbol"].isin(["BTC-USD", "ETH-USD"])].copy()
    if df_bench.empty:
        raise RuntimeError(
            f"No BTC-USD / ETH-USD rows found in benchmark summary at {benchmark_summary_path}"
        )

    # Build portfolio row from full-sample metrics
    full_row = portfolio_full_metrics.loc[
        portfolio_full_metrics["period"] == "full"
    ].iloc[0]
    portfolio_row = {
        "symbol": "MIDCAP-PORTFOLIO",
        "total_return": full_row["total_return"],
        "sharpe": full_row["sharpe"],
        "max_drawdown": full_row["max_dd"],
    }

    df_portfolio = pd.DataFrame([portfolio_row])
    df_out = pd.concat([df_bench[["symbol", "total_return", "sharpe", "max_drawdown"]], df_portfolio], ignore_index=True)

    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute midcap momentum v0 metrics: cross-section, portfolio, vol-target, and optional BTC/ETH comparison."
    )
    parser.add_argument(
        "--summary_midcap",
        type=str,
        default="artifacts/research/midcap_momentum_v0/summary.csv",
        help="Path to midcap summary.csv.",
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default="artifacts/research/midcap_momentum_v0/portfolio.csv",
        help="Path to midcap portfolio.csv.",
    )
    parser.add_argument(
        "--benchmark_summary",
        type=str,
        default=None,
        help="Optional path to benchmark summary CSV with BTC-USD and ETH-USD rows.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/research/midcap_momentum_v0",
        help="Output directory for metrics CSVs.",
    )

    args = parser.parse_args()
    summary_midcap_path = Path(args.summary_midcap)
    portfolio_path = Path(args.portfolio)
    out_dir = Path(args.out_dir)

    if not summary_midcap_path.exists():
        raise FileNotFoundError(f"Midcap summary not found: {summary_midcap_path}")
    if not portfolio_path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {portfolio_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Cross-sectional midcap ranking
    midcap_cross = summarize_midcaps_cross_section(summary_midcap_path)
    cross_path = out_dir / "metrics_midcap_cross_section.csv"
    midcap_cross.to_csv(cross_path, index=False)
    print(f"[midcap_momentum_v0_metrics] Wrote midcap cross-section metrics to {cross_path}")

    # 2) Portfolio metrics (full + subperiods) and vol-target metrics
    portfolio_metrics, vt_metrics = compute_portfolio_metrics(portfolio_path)

    portfolio_metrics_path = out_dir / "metrics_portfolio_full_and_subperiods.csv"
    portfolio_metrics.to_csv(portfolio_metrics_path, index=False)
    print(
        f"[midcap_momentum_v0_metrics] Wrote portfolio metrics to {portfolio_metrics_path}"
    )

    vt_metrics_path = out_dir / "metrics_portfolio_vol_target.csv"
    vt_metrics.to_csv(vt_metrics_path, index=False)
    print(
        f"[midcap_momentum_v0_metrics] Wrote vol-target portfolio metrics to {vt_metrics_path}"
    )

    # 3) Optional BTC/ETH benchmark comparison
    if args.benchmark_summary:
        benchmark_summary_path = Path(args.benchmark_summary)
        if not benchmark_summary_path.exists():
            raise FileNotFoundError(
                f"Benchmark summary not found: {benchmark_summary_path}"
            )

        bench_df = build_benchmark_comparison(
            benchmark_summary_path=benchmark_summary_path,
            portfolio_full_metrics=portfolio_metrics,
        )
        bench_out_path = out_dir / "metrics_benchmark_vs_midcap.csv"
        bench_df.to_csv(bench_out_path, index=False)
        print(
            f"[midcap_momentum_v0_metrics] Wrote benchmark vs midcap comparison to {bench_out_path}"
        )


if __name__ == "__main__":
    main()

