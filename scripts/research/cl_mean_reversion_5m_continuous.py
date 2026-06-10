#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5m long/short mean reversion on CL continuous futures.")
    parser.add_argument("--db", default="/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/futures_market.duckdb")
    parser.add_argument("--symbol", default="CL")
    parser.add_argument("--expiry", default="continuous")
    parser.add_argument("--out_dir", default="artifacts/research/cl_mean_reversion_5m_continuous")
    parser.add_argument("--lookback_bars", type=int, default=12 * 12)
    parser.add_argument("--entry_z", type=float, default=2.0)
    parser.add_argument("--exit_z", type=float, default=0.0)
    parser.add_argument("--cost_bps", type=float, default=0.0)
    return parser.parse_args()


def load_5m_bars(db_path: str, symbol: str, expiry: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        bars = con.execute(
            """
            SELECT
                time_bucket(INTERVAL '5 minutes', ts) AS ts,
                FIRST(o ORDER BY ts) AS open,
                MAX(h) AS high,
                MIN(l) AS low,
                LAST(c ORDER BY ts) AS close,
                SUM(v) AS volume
            FROM bars_1m
            WHERE symbol = ? AND expiry = ?
            GROUP BY 1
            HAVING FIRST(o ORDER BY ts) > 0
               AND LAST(c ORDER BY ts) > 0
            ORDER BY 1
            """,
            [symbol, expiry],
        ).fetch_df()
    finally:
        con.close()
    if bars.empty:
        return bars
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    return bars


def build_strategy(bars: pd.DataFrame, lookback: int, entry_z: float, exit_z: float, cost_bps: float) -> pd.DataFrame:
    df = bars.copy()
    close = df["close"].astype(float)
    mean = close.shift(1).rolling(lookback, min_periods=lookback).mean()
    std = close.shift(1).rolling(lookback, min_periods=lookback).std()
    zscore = (close - mean) / std.replace(0.0, pd.NA)

    signal: list[float] = []
    position = 0.0
    long_entries = short_entries = exits = flips = 0
    for z in zscore:
        previous = position
        if pd.notna(z):
            if position == 0.0:
                if z <= -entry_z:
                    position = 1.0
                    long_entries += 1
                elif z >= entry_z:
                    position = -1.0
                    short_entries += 1
            elif position > 0.0 and z >= exit_z:
                position = 0.0
                exits += 1
            elif position < 0.0 and z <= -exit_z:
                position = 0.0
                exits += 1
        if previous != 0.0 and position != 0.0 and previous != position:
            flips += 1
        signal.append(position)

    # Signal is observed at close t, traded from next bar open to following bar open.
    df["zscore"] = zscore
    df["w_signal"] = signal
    df["w_held"] = df["w_signal"].shift(1).fillna(0.0)
    df["ret_open_to_next_open"] = df["open"].shift(-1) / df["open"] - 1.0
    df["ret_open_to_next_open"] = df["ret_open_to_next_open"].fillna(0.0)
    df["turnover_one_sided"] = (df["w_held"] - df["w_held"].shift(1).fillna(0.0)).abs()
    df["cost_ret"] = df["turnover_one_sided"] * (cost_bps / 10_000.0)
    df["portfolio_ret"] = df["w_held"] * df["ret_open_to_next_open"] - df["cost_ret"]
    df["portfolio_equity"] = (1.0 + df["portfolio_ret"]).cumprod()
    df["drawdown"] = df["portfolio_equity"] / df["portfolio_equity"].cummax() - 1.0
    df.attrs["long_entries"] = long_entries
    df.attrs["short_entries"] = short_entries
    df.attrs["exits"] = exits
    df.attrs["flips"] = flips
    return df


def compute_metrics(df: pd.DataFrame) -> dict[str, object]:
    ret = df["portfolio_ret"].astype(float)
    eq = df["portfolio_equity"].astype(float)
    elapsed_years = (df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() / (365.0 * 24 * 3600)
    bars_per_year = len(df) / elapsed_years if elapsed_years > 0 else 0.0
    ret_std = ret.std(ddof=0)
    active = df["w_held"].abs() > 0
    return {
        "start": str(df["ts"].iloc[0]),
        "end": str(df["ts"].iloc[-1]),
        "n_bars_5m": int(len(df)),
        "elapsed_years": float(elapsed_years),
        "bars_per_year_observed": float(bars_per_year),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": float(eq.iloc[-1] ** (1.0 / elapsed_years) - 1.0) if elapsed_years > 0 and eq.iloc[-1] > 0 else None,
        "vol": float(ret_std * math.sqrt(bars_per_year)) if bars_per_year > 0 else 0.0,
        "sharpe": float(ret.mean() / ret_std * math.sqrt(bars_per_year)) if ret_std > 0 and bars_per_year > 0 else 0.0,
        "max_dd": float(df["drawdown"].min()),
        "avg_abs_exposure": float(df["w_held"].abs().mean()),
        "active_bars": int(active.sum()),
        "active_pct": float(active.mean()),
        "long_bars": int((df["w_held"] > 0).sum()),
        "short_bars": int((df["w_held"] < 0).sum()),
        "avg_turnover_one_sided": float(df["turnover_one_sided"].mean()),
        "long_entries": int(df.attrs.get("long_entries", 0)),
        "short_entries": int(df.attrs.get("short_entries", 0)),
        "exits": int(df.attrs.get("exits", 0)),
        "flips": int(df.attrs.get("flips", 0)),
    }


def write_figures(df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    axes[0].plot(df["ts"], df["portfolio_equity"], linewidth=1.1)
    axes[0].set_title("CL Continuous 5m Long/Short Mean Reversion")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(df["ts"], df["drawdown"], 0.0, alpha=0.35)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(df["ts"], df["w_held"], linewidth=0.8)
    axes[2].set_ylabel("Position")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_equity_drawdown_position.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_5m_bars(args.db, args.symbol, args.expiry)
    if bars.empty:
        raise RuntimeError(f"No bars found for {args.symbol} {args.expiry}")

    result = build_strategy(bars, args.lookback_bars, args.entry_z, args.exit_z, args.cost_bps)
    metrics = compute_metrics(result)
    config = {
        "db": args.db,
        "symbol": args.symbol,
        "expiry": args.expiry,
        "bar_interval": "5m",
        "lookback_bars": args.lookback_bars,
        "entry_z": args.entry_z,
        "exit_z": args.exit_z,
        "cost_bps": args.cost_bps,
        "return_model": "signal_at_close_t_fill_next_open_hold_open_to_next_open",
    }
    result.to_csv(out_dir / "equity.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    write_figures(result, out_dir)
    print(json.dumps(metrics, indent=2, default=str))
    print(f"[cl_mean_reversion_5m_continuous] wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
