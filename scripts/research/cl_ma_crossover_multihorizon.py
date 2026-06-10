#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TIMEFRAMES: dict[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CL institutional continuous 5/40 MA crossover by timeframe.")
    parser.add_argument("--continuous_dir", default="artifacts/research/cl_institutional_continuous")
    parser.add_argument("--out_dir", default="artifacts/research/cl_ma_5_40_multihorizon")
    parser.add_argument("--fast", type=int, default=5)
    parser.add_argument("--slow", type=int, default=40)
    parser.add_argument("--cost_bps", type=float, default=0.0)
    parser.add_argument("--timeframes", default="5m,15m,30m,1h,4h,1d")
    return parser.parse_args()


def load_valid_1m(continuous_dir: Path) -> pd.DataFrame:
    bars = pd.read_parquet(continuous_dir / "bars_1m.parquet")
    bars["ts"] = pd.to_datetime(bars["ts"])
    bars = bars[bars["front_month_valid"]].sort_values("ts").copy()
    if bars.empty:
        raise RuntimeError("No front-month-valid institutional CL rows found.")
    return bars


def resample_bars(bars_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = TIMEFRAMES[timeframe]
    bars = bars_1m.set_index("ts").resample(rule).agg(
        {
            "o": "first",
            "h": "max",
            "l": "min",
            "c": "last",
            "v": "sum",
            "active_expiry": "last",
        }
    )
    bars = bars.dropna(subset=["o", "c"]).reset_index()
    bars["timeframe"] = timeframe
    return bars


def backtest_ma_crossover(
    bars: pd.DataFrame,
    *,
    fast: int,
    slow: int,
    cost_bps: float,
) -> pd.DataFrame:
    df = bars.copy()
    close = df["c"].astype(float)
    fast_ma = close.shift(1).rolling(fast, min_periods=fast).mean()
    slow_ma = close.shift(1).rolling(slow, min_periods=slow).mean()
    signal = pd.Series(0.0, index=df.index)
    signal = signal.mask(fast_ma > slow_ma, 1.0)
    signal = signal.mask(fast_ma < slow_ma, -1.0)

    df["fast_ma"] = fast_ma
    df["slow_ma"] = slow_ma
    df["w_signal"] = signal.fillna(0.0)
    df["w_held"] = df["w_signal"].shift(1).fillna(0.0)
    df["ret_open_to_next_open"] = df["o"].shift(-1) / df["o"] - 1.0
    df["ret_open_to_next_open"] = df["ret_open_to_next_open"].fillna(0.0)
    df["turnover_one_sided"] = (df["w_held"] - df["w_held"].shift(1).fillna(0.0)).abs()
    df["cost_ret"] = df["turnover_one_sided"] * (cost_bps / 10_000.0)
    df["portfolio_ret"] = df["w_held"] * df["ret_open_to_next_open"] - df["cost_ret"]
    df["portfolio_equity"] = (1.0 + df["portfolio_ret"]).cumprod()
    df["drawdown"] = df["portfolio_equity"] / df["portfolio_equity"].cummax() - 1.0
    return df


def compute_metrics(df: pd.DataFrame) -> dict[str, object]:
    ret = df["portfolio_ret"].astype(float)
    eq = df["portfolio_equity"].astype(float)
    elapsed_years = (df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() / (365.0 * 24 * 3600)
    bars_per_year = len(df) / elapsed_years if elapsed_years > 0 else 0.0
    ret_std = ret.std(ddof=0)
    active = df["w_held"].abs() > 0
    trades = int((df["w_signal"] != df["w_signal"].shift(1)).sum())
    return {
        "timeframe": str(df["timeframe"].iloc[0]),
        "start": str(df["ts"].iloc[0]),
        "end": str(df["ts"].iloc[-1]),
        "n_bars": int(len(df)),
        "elapsed_years": float(elapsed_years),
        "bars_per_year_observed": float(bars_per_year),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": float(eq.iloc[-1] ** (1.0 / elapsed_years) - 1.0) if elapsed_years > 0 and eq.iloc[-1] > 0 else None,
        "vol": float(ret_std * math.sqrt(bars_per_year)) if bars_per_year > 0 else 0.0,
        "sharpe": float(ret.mean() / ret_std * math.sqrt(bars_per_year)) if ret_std > 0 and bars_per_year > 0 else 0.0,
        "max_dd": float(df["drawdown"].min()),
        "avg_abs_exposure": float(df["w_held"].abs().mean()),
        "long_pct": float((df["w_held"] > 0).mean()),
        "short_pct": float((df["w_held"] < 0).mean()),
        "active_pct": float(active.mean()),
        "avg_turnover_one_sided": float(df["turnover_one_sided"].mean()),
        "signal_changes": trades,
    }


def write_figures(equity: pd.DataFrame, metrics: pd.DataFrame, roll_schedule: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    for timeframe, group in equity.groupby("timeframe", sort=False):
        ax.plot(group["ts"], group["portfolio_equity"], linewidth=1.1, label=timeframe)
    ax.set_yscale("log")
    ax.set_title("CL Institutional Continuous 5/40 MA Crossover by Timeframe")
    ax.set_ylabel("Strategy equity (log)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "01_equity_by_timeframe.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = metrics.sort_values("sharpe", ascending=True)
    ax.barh(ordered["timeframe"], ordered["sharpe"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Sharpe by timeframe")
    ax.set_xlabel("Annualized Sharpe")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_sharpe_by_timeframe.png", dpi=160)
    plt.close(fig)

    # Overlay best equity with CL close and roll markers.
    best_tf = str(metrics.sort_values("sharpe", ascending=False).iloc[0]["timeframe"])
    best = equity[equity["timeframe"] == best_tf]
    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax1.plot(best["ts"], best["portfolio_equity"], linewidth=1.2, label=f"{best_tf} equity")
    ax1.set_yscale("log")
    ax1.set_ylabel("Strategy equity (log)")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(best["ts"], best["c"], color="black", alpha=0.35, linewidth=1.0, label="CL close")
    start = best["ts"].min()
    end = best["ts"].max()
    for row in roll_schedule.itertuples(index=False):
        roll_ts = pd.Timestamp(row.roll_date)
        if start.tzinfo is not None and roll_ts.tzinfo is None:
            roll_ts = roll_ts.tz_localize(start.tzinfo)
        if start <= roll_ts <= end:
            px = best[best["ts"] >= roll_ts].head(1)
            if not px.empty:
                ax2.scatter(px["ts"].iloc[0], px["c"].iloc[0], color="red", s=35, zorder=5)
                ax2.axvline(px["ts"].iloc[0], color="red", alpha=0.15, linewidth=0.8)
    ax2.set_ylabel("CL continuous close")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title(f"Best 5/40 MA timeframe ({best_tf}) with CL overlay and rolls")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_best_timeframe_with_cl_overlay.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    continuous_dir = Path(args.continuous_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_1m = load_valid_1m(continuous_dir)
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    unknown = sorted(set(timeframes) - set(TIMEFRAMES))
    if unknown:
        raise ValueError(f"Unknown timeframes: {unknown}")

    frames = []
    metric_rows = []
    for timeframe in timeframes:
        bars = resample_bars(bars_1m, timeframe)
        result = backtest_ma_crossover(bars, fast=args.fast, slow=args.slow, cost_bps=args.cost_bps)
        frames.append(result)
        metric_rows.append(compute_metrics(result))
        result.to_csv(out_dir / f"equity_{timeframe}.csv", index=False)

    equity = pd.concat(frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("sharpe", ascending=False)
    roll_schedule = pd.read_csv(continuous_dir / "roll_schedule.csv", parse_dates=["roll_date"])
    equity.to_csv(out_dir / "equity_all_timeframes.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "continuous_dir": str(continuous_dir),
                "fast": args.fast,
                "slow": args.slow,
                "cost_bps": args.cost_bps,
                "timeframes": timeframes,
                "rule": "long fast_ma > slow_ma, short fast_ma < slow_ma, one-bar execution lag",
                "return_model": "signal_at_close_t_fill_next_open_hold_open_to_next_open",
            },
            indent=2,
        )
    )
    write_figures(equity, metrics, roll_schedule, out_dir)
    print(metrics.to_string(index=False))
    print(f"[cl_ma_crossover_multihorizon] wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
