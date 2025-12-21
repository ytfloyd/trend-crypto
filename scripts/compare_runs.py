from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import polars as pl


def load_equity(run_dir: Path) -> pl.DataFrame:
    path = run_dir / "equity.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity.parquet in {run_dir}")
    return pl.read_parquet(path).sort("ts")


def load_trade_count(run_dir: Path) -> Optional[int]:
    path = run_dir / "trades.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    return df.height


def metrics(df: pl.DataFrame) -> Dict[str, float]:
    df = df.sort("ts")
    nav = df["nav"]
    start = nav.item(0)
    end = nav.item(nav.len() - 1)
    total_return = (end / start) - 1 if start else 0.0
    returns = nav.pct_change().fill_null(0.0)
    mean = returns.mean()
    std = returns.std(ddof=1)
    diffs = df.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
    dt_seconds = diffs.median() if diffs.len() > 0 else 0
    periods_per_year = (365 * 24 * 3600 / dt_seconds) if dt_seconds and dt_seconds > 0 else 8760
    sharpe = (mean / std) * (periods_per_year ** 0.5) if std and std > 0 else 0.0
    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_dd = drawdowns.min()
    n_periods = returns.len()
    cagr = (end / start) ** (periods_per_year / n_periods) - 1 if start and n_periods > 0 else 0.0
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "cagr": cagr,
    }


def plot_equity(run_a: pl.DataFrame, run_b: pl.DataFrame, name_a: str, name_b: str, out_dir: Path) -> None:
    joined = (
        run_a.rename({"nav": "nav_a"})
        .join(run_b.rename({"nav": "nav_b"}), on="ts", how="inner")
        .sort("ts")
    )
    if joined.is_empty():
        print("No overlapping timestamps for equity plot; skipping.")
        return
    joined = joined.with_columns(
        [
            (pl.col("nav_a") / pl.col("nav_a").first()).alias("eq_a_norm"),
            (pl.col("nav_b") / pl.col("nav_b").first()).alias("eq_b_norm"),
        ]
    )
    x = joined["ts"].to_list()
    y_a = joined["eq_a_norm"].to_list()
    y_b = joined["eq_b_norm"].to_list()
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_a, label=name_a)
    plt.plot(x, y_b, label=name_b)
    plt.title("Normalized Equity")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("NAV (normalized)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve_normalized.png", dpi=150)
    plt.close()


def plot_rolling_cagr(run_a: pl.DataFrame, run_b: pl.DataFrame, name_a: str, name_b: str, out_dir: Path) -> None:
    window = 8760
    joined = (
        run_a.rename({"nav": "nav_a"})
        .join(run_b.rename({"nav": "nav_b"}), on="ts", how="inner")
        .sort("ts")
    )
    if joined.height < window + 1:
        print("Not enough overlapping data for rolling 1Y CAGR chart; skipping.")
        return
    joined = joined.with_columns(
        [
            (pl.col("nav_a") / pl.col("nav_a").shift(window) - 1).alias("cagr_a"),
            (pl.col("nav_b") / pl.col("nav_b").shift(window) - 1).alias("cagr_b"),
        ]
    ).drop_nulls()
    if joined.is_empty():
        print("Rolling CAGR window produced no data; skipping.")
        return
    x = joined["ts"].to_list()
    y_a = joined["cagr_a"].to_list()
    y_b = joined["cagr_b"].to_list()
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_a, label=name_a)
    plt.plot(x, y_b, label=name_b)
    plt.title("Rolling 1Y CAGR (8760 hours)")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("CAGR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rolling_1y_cagr.png", dpi=150)
    plt.close()


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {run_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_dt(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two backtest runs.")
    parser.add_argument("--run_a", required=True, help="Path to run A directory")
    parser.add_argument("--name_a", default="Strategy", help="Label for run A")
    parser.add_argument("--run_b", required=True, help="Path to run B directory")
    parser.add_argument("--name_b", default="Buy & Hold", help="Label for run B")
    parser.add_argument("--out", default="artifacts/compare", help="Output directory for charts")
    args = parser.parse_args()

    run_a_dir = Path(args.run_a)
    run_b_dir = Path(args.run_b)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_a = load_equity(run_a_dir)
    eq_b = load_equity(run_b_dir)

    man_a = load_manifest(run_a_dir)
    man_b = load_manifest(run_b_dir)
    a_tr = man_a.get("time_range", {})
    b_tr = man_b.get("time_range", {})
    a_start = parse_dt(a_tr.get("start")) if a_tr.get("start") else None
    a_end = parse_dt(a_tr.get("end")) if a_tr.get("end") else None
    b_start = parse_dt(b_tr.get("start")) if b_tr.get("start") else None
    b_end = parse_dt(b_tr.get("end")) if b_tr.get("end") else None

    a_cfg_data = man_a.get("params", {}).get("data", {})
    b_cfg_data = man_b.get("params", {}).get("data", {})

    print(f"Run A time_range: {a_start} -> {a_end}")
    print(f"Run A params.data: {a_cfg_data.get('start')} -> {a_cfg_data.get('end')}")
    print(f"Run B time_range: {b_start} -> {b_end}")
    print(f"Run B params.data: {b_cfg_data.get('start')} -> {b_cfg_data.get('end')}")

    overlap_hours = None
    if a_start and a_end and b_start and b_end:
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        if overlap_end > overlap_start:
            overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
        else:
            overlap_hours = 0
        if overlap_hours < 8760:
            print(
                f"Warning: overlap is short ({overlap_hours:.1f} hours). "
                f"Run A covers {a_start}->{a_end}; Run B covers {b_start}->{b_end}; comparison may be misleading."
            )

    m_a = metrics(eq_a)
    m_b = metrics(eq_b)
    trades_a = load_trade_count(run_a_dir)
    trades_b = load_trade_count(run_b_dir)
    trades_a_str = str(trades_a) if trades_a is not None else "n/a"
    trades_b_str = str(trades_b) if trades_b is not None else "n/a"

    print(f"{'Metric':<18}{args.name_a:<20}{args.name_b:<20}")
    print(f"{'Total Return':<18}{m_a['total_return']:<20.4f}{m_b['total_return']:<20.4f}")
    print(f"{'CAGR':<18}{m_a['cagr']:<20.4f}{m_b['cagr']:<20.4f}")
    print(f"{'Sharpe':<18}{m_a['sharpe']:<20.4f}{m_b['sharpe']:<20.4f}")
    print(f"{'Max Drawdown':<18}{m_a['max_drawdown']:<20.4f}{m_b['max_drawdown']:<20.4f}")
    print(f"{'# Trades':<18}{trades_a_str:<20}{trades_b_str:<20}")

    plot_equity(eq_a, eq_b, args.name_a, args.name_b, out_dir)
    plot_rolling_cagr(eq_a, eq_b, args.name_a, args.name_b, out_dir)
    print(f"Charts written to {out_dir}")


if __name__ == "__main__":
    main()

