#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5m mean-reversion and trend on institutional CL continuous bars."
    )
    parser.add_argument(
        "--continuous_dir",
        default="artifacts/research/cl_institutional_continuous",
        help="Directory containing bars_1m.parquet and roll_schedule.csv.",
    )
    parser.add_argument("--out_dir", default="artifacts/research/cl_5m_institutional_mr_trend")
    parser.add_argument("--lookback_bars", type=int, default=12 * 12)
    parser.add_argument("--entry_z", type=float, default=2.0)
    parser.add_argument("--exit_z", type=float, default=0.0)
    parser.add_argument("--cost_bps", type=float, default=0.0)
    return parser.parse_args()


def load_institutional_5m(continuous_dir: Path) -> pd.DataFrame:
    bars = pd.read_parquet(continuous_dir / "bars_1m.parquet")
    bars["ts"] = pd.to_datetime(bars["ts"])
    bars = bars[bars["front_month_valid"]].sort_values("ts").copy()
    if bars.empty:
        raise RuntimeError("No front-month-valid institutional continuous CL rows found.")

    bars = bars.set_index("ts")
    out = bars.resample("5min").agg(
        {
            "o": "first",
            "h": "max",
            "l": "min",
            "c": "last",
            "v": "sum",
            "active_expiry": "last",
            "front_month_valid": "last",
        }
    )
    out = out.dropna(subset=["o", "c"]).reset_index()
    return out


def build_strategy(
    bars: pd.DataFrame,
    *,
    strategy: str,
    lookback: int,
    entry_z: float,
    exit_z: float,
    cost_bps: float,
) -> pd.DataFrame:
    df = bars.copy()
    close = df["c"].astype(float)
    mean = close.shift(1).rolling(lookback, min_periods=lookback).mean()
    std = close.shift(1).rolling(lookback, min_periods=lookback).std()
    zscore = (close - mean) / std.replace(0.0, pd.NA)

    signal: list[float] = []
    position = 0.0
    long_entries = short_entries = exits = 0
    for z in zscore:
        if pd.notna(z):
            if position == 0.0:
                if strategy == "mean_reversion":
                    if z <= -entry_z:
                        position = 1.0
                        long_entries += 1
                    elif z >= entry_z:
                        position = -1.0
                        short_entries += 1
                elif strategy == "trend":
                    if z >= entry_z:
                        position = 1.0
                        long_entries += 1
                    elif z <= -entry_z:
                        position = -1.0
                        short_entries += 1
                else:
                    raise ValueError(f"unknown strategy {strategy}")
            elif position > 0.0 and z <= exit_z:
                position = 0.0
                exits += 1
            elif position < 0.0 and z >= -exit_z:
                position = 0.0
                exits += 1
        signal.append(position)

    df["strategy"] = strategy
    df["zscore"] = zscore
    df["w_signal"] = signal
    df["w_held"] = df["w_signal"].shift(1).fillna(0.0)
    df["ret_open_to_next_open"] = df["o"].shift(-1) / df["o"] - 1.0
    df["ret_open_to_next_open"] = df["ret_open_to_next_open"].fillna(0.0)
    df["turnover_one_sided"] = (df["w_held"] - df["w_held"].shift(1).fillna(0.0)).abs()
    df["cost_ret"] = df["turnover_one_sided"] * (cost_bps / 10_000.0)
    df["portfolio_ret"] = df["w_held"] * df["ret_open_to_next_open"] - df["cost_ret"]
    df["portfolio_equity"] = (1.0 + df["portfolio_ret"]).cumprod()
    df["drawdown"] = df["portfolio_equity"] / df["portfolio_equity"].cummax() - 1.0
    df.attrs["long_entries"] = long_entries
    df.attrs["short_entries"] = short_entries
    df.attrs["exits"] = exits
    return df


def compute_metrics(frame: pd.DataFrame) -> dict[str, object]:
    ret = frame["portfolio_ret"].astype(float)
    eq = frame["portfolio_equity"].astype(float)
    elapsed_years = (frame["ts"].iloc[-1] - frame["ts"].iloc[0]).total_seconds() / (365.0 * 24 * 3600)
    bars_per_year = len(frame) / elapsed_years if elapsed_years > 0 else 0.0
    ret_std = ret.std(ddof=0)
    active = frame["w_held"].abs() > 0
    return {
        "strategy": str(frame["strategy"].iloc[0]),
        "start": str(frame["ts"].iloc[0]),
        "end": str(frame["ts"].iloc[-1]),
        "n_bars_5m": int(len(frame)),
        "elapsed_years": float(elapsed_years),
        "bars_per_year_observed": float(bars_per_year),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": float(eq.iloc[-1] ** (1.0 / elapsed_years) - 1.0) if elapsed_years > 0 and eq.iloc[-1] > 0 else None,
        "vol": float(ret_std * math.sqrt(bars_per_year)) if bars_per_year > 0 else 0.0,
        "sharpe": float(ret.mean() / ret_std * math.sqrt(bars_per_year)) if ret_std > 0 and bars_per_year > 0 else 0.0,
        "max_dd": float(frame["drawdown"].min()),
        "avg_abs_exposure": float(frame["w_held"].abs().mean()),
        "active_pct": float(active.mean()),
        "long_bars": int((frame["w_held"] > 0).sum()),
        "short_bars": int((frame["w_held"] < 0).sum()),
        "avg_turnover_one_sided": float(frame["turnover_one_sided"].mean()),
        "long_entries": int(frame.attrs.get("long_entries", 0)),
        "short_entries": int(frame.attrs.get("short_entries", 0)),
        "exits": int(frame.attrs.get("exits", 0)),
    }


def _roll_marker_points(price: pd.DataFrame, roll_schedule: pd.DataFrame) -> pd.DataFrame:
    if roll_schedule.empty:
        return pd.DataFrame(columns=["roll_date", "ts", "c", "front_expiry", "next_expiry"])
    rows = []
    px = price.sort_values("ts")
    start = px["ts"].min()
    end = px["ts"].max()
    for row in roll_schedule.itertuples(index=False):
        roll_ts = pd.Timestamp(row.roll_date)
        if roll_ts.tzinfo is None and start.tzinfo is not None:
            roll_ts = roll_ts.tz_localize(start.tzinfo)
        if roll_ts < start or roll_ts > end:
            continue
        after = px[px["ts"] >= roll_ts]
        if after.empty:
            continue
        first = after.iloc[0]
        rows.append(
            {
                "roll_date": str(row.roll_date),
                "ts": first["ts"],
                "c": float(first["c"]),
                "front_expiry": row.front_expiry,
                "next_expiry": row.next_expiry,
            }
        )
    return pd.DataFrame(rows)


def write_figures(
    bars_5m: pd.DataFrame,
    equity: pd.DataFrame,
    roll_schedule: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    price_plot = bars_5m.set_index("ts").resample("30min").agg({"c": "last", "v": "sum"}).dropna().reset_index()
    roll_points = _roll_marker_points(price_plot, roll_schedule)

    fig, ax1 = plt.subplots(figsize=(13, 7))
    for strategy, group in equity.groupby("strategy", sort=True):
        ax1.plot(group["ts"], group["portfolio_equity"], linewidth=1.15, label=f"{strategy} equity")
    ax1.set_ylabel("Strategy equity")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(price_plot["ts"], price_plot["c"], color="black", alpha=0.35, linewidth=1.0, label="CL continuous close")
    if not roll_points.empty:
        ax2.scatter(roll_points["ts"], roll_points["c"], color="red", s=35, zorder=5, label="roll date")
        for rp in roll_points.itertuples(index=False):
            ax2.axvline(rp.ts, color="red", alpha=0.15, linewidth=0.8)
    ax2.set_ylabel("CL institutional continuous close")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("CL Institutional Continuous: 5m Mean Reversion vs Trend")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_equity_with_continuous_overlay_and_rolls.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for strategy, group in equity.groupby("strategy", sort=True):
        axes[0].plot(group["ts"], group["portfolio_equity"], linewidth=1.1, label=strategy)
        axes[1].plot(group["ts"], group["drawdown"], linewidth=0.9, label=strategy)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Equity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_equity_drawdown_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    continuous_dir = Path(args.continuous_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_5m = load_institutional_5m(continuous_dir)
    strategies = [
        build_strategy(
            bars_5m,
            strategy="mean_reversion",
            lookback=args.lookback_bars,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            cost_bps=args.cost_bps,
        ),
        build_strategy(
            bars_5m,
            strategy="trend",
            lookback=args.lookback_bars,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            cost_bps=args.cost_bps,
        ),
    ]
    metrics = pd.DataFrame([compute_metrics(s) for s in strategies])
    equity = pd.concat(strategies, ignore_index=True)
    roll_schedule = pd.read_csv(continuous_dir / "roll_schedule.csv", parse_dates=["roll_date"])

    bars_5m.to_parquet(out_dir / "bars_5m.parquet", index=False)
    equity.to_csv(out_dir / "equity.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "continuous_dir": str(continuous_dir),
                "lookback_bars": args.lookback_bars,
                "entry_z": args.entry_z,
                "exit_z": args.exit_z,
                "cost_bps": args.cost_bps,
                "return_model": "signal_at_close_t_fill_next_open_hold_open_to_next_open",
            },
            indent=2,
        )
    )
    write_figures(bars_5m, equity, roll_schedule, out_dir)
    print(metrics.to_string(index=False))
    print(f"[cl_mr_trend_5m_institutional_continuous] wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
