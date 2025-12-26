from __future__ import annotations

import argparse
import json
from pathlib import Path
from math import sqrt

import polars as pl
import matplotlib.pyplot as plt


def load_equity(run_dir: Path) -> pl.DataFrame:
    path = run_dir / "equity.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity.parquet in {run_dir}")
    df = pl.read_parquet(path)
    cols = [c for c in ["ts", "nav", "net_ret", "turnover", "cost_ret"] if c in df.columns]
    df = df.select(cols).sort("ts")
    if "net_ret" not in df.columns:
        df = df.with_columns((pl.col("nav") / pl.col("nav").shift(1) - 1).alias("net_ret"))
    if "turnover" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("turnover"))
    if "cost_ret" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("cost_ret"))
    return df


def metrics(df: pl.DataFrame) -> dict:
    df = df.sort("ts")
    nav = df["nav"]
    start = nav.item(0)
    end = nav.item(nav.len() - 1)
    total_return = (end / start) - 1 if start else 0.0
    returns = df["r_port"]
    mean = returns.mean()
    std = returns.std(ddof=1)
    diffs = df.select(pl.col("ts").diff().dt.total_seconds()).to_series().drop_nulls()
    dt_seconds = diffs.median() if diffs.len() > 0 else 0
    periods_per_year = (365 * 24 * 3600 / dt_seconds) if dt_seconds and dt_seconds > 0 else 8760
    sharpe = (mean / std) * (periods_per_year ** 0.5) if std and std > 0 else 0.0
    n_periods = returns.len()
    cagr = (end / start) ** (periods_per_year / n_periods) - 1 if start and n_periods > 0 else 0.0
    running_max = nav.cum_max()
    drawdowns = (nav / running_max) - 1
    max_dd = drawdowns.min()
    max_dd_idx = int(drawdowns.arg_min()) if drawdowns.len() > 0 else 0
    max_dd_ts = df["ts"].item(max_dd_idx) if drawdowns.len() > 0 else None
    vol_ann = std * sqrt(periods_per_year) if std is not None else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_ts": max_dd_ts.isoformat() if max_dd_ts else None,
        "vol_annual": vol_ann,
        "calmar": calmar,
    }


def plot_equity(df: pl.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["ts"].to_list(), df["nav"].to_list(), label="Combined 50/50")
    plt.yscale("log")
    plt.title("Combined 50/50 Portfolio (log NAV)")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("NAV (log scale)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined 50/50 portfolio from two runs.")
    parser.add_argument("--run_a", required=True, help="Path to first run dir (e.g., BTC)")
    parser.add_argument("--run_b", required=True, help="Path to second run dir (e.g., ETH)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--initial_nav", type=float, default=100000.0)
    args = parser.parse_args()

    eq_a = load_equity(Path(args.run_a)).rename({"net_ret": "r_a"})
    eq_b = load_equity(Path(args.run_b)).rename({"net_ret": "r_b"})

    joined = (
        eq_a.rename({"nav": "nav_a"})
        .join(eq_b.rename({"nav": "nav_b"}), on="ts", how="inner", suffix="_b")
        .sort("ts")
    )
    # normalize column names
    rename_map = {}
    if "turnover" in joined.columns:
        rename_map["turnover"] = "turnover_a"
    if "turnover_b" in joined.columns:
        rename_map["turnover_b"] = "turnover_b"
    elif "turnover_right" in joined.columns:
        rename_map["turnover_right"] = "turnover_b"
    if "cost_ret" in joined.columns:
        rename_map["cost_ret"] = "cost_ret_a"
    if "cost_ret_b" in joined.columns:
        rename_map["cost_ret_b"] = "cost_ret_b"
    elif "cost_ret_right" in joined.columns:
        rename_map["cost_ret_right"] = "cost_ret_b"
    joined = joined.rename(rename_map)

    # ensure missing turnover/cost become 0.0 where absent
    if "turnover_a" not in joined.columns:
        joined = joined.with_columns(pl.lit(0.0).alias("turnover_a"))
    if "turnover_b" not in joined.columns:
        joined = joined.with_columns(pl.lit(0.0).alias("turnover_b"))
    if "cost_ret_a" not in joined.columns:
        joined = joined.with_columns(pl.lit(0.0).alias("cost_ret_a"))
    if "cost_ret_b" not in joined.columns:
        joined = joined.with_columns(pl.lit(0.0).alias("cost_ret_b"))

    joined = joined.drop_nulls(subset=["r_a", "r_b"])
    if joined.is_empty():
        raise ValueError("No overlapping timestamps between runs.")

    combined = joined.with_columns(
        [
            (0.5 * pl.col("r_a") + 0.5 * pl.col("r_b")).alias("r_port"),
            (0.5 * pl.col("turnover_a") + 0.5 * pl.col("turnover_b")).alias("turnover_combined"),
            (0.5 * pl.col("cost_ret_a") + 0.5 * pl.col("cost_ret_b")).alias("cost_ret_combined"),
        ]
    )
    nav_list = []
    nav_prev = args.initial_nav
    nav_list.append(nav_prev)
    for r in combined["r_port"].to_list():
        nav_curr = nav_prev * (1 + r)
        nav_list.append(nav_curr)
        nav_prev = nav_curr
    # Align nav length to ts length
    nav_series = nav_list[1:]
    combined = combined.with_columns(pl.Series("nav", nav_series))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.select(["ts", "nav", "r_port", "turnover_combined", "cost_ret_combined"]).write_parquet(out_dir / "equity.parquet")

    m = metrics(combined)
    turn_series = combined["turnover_combined"]
    turn_total = turn_series.sum()
    turn_events = int((turn_series.fill_null(0) > 0).sum())
    m["turnover_total"] = turn_total
    m["turnover_events"] = turn_events
    m["turnover_p50"] = float(turn_series.quantile(0.5, interpolation="nearest")) if turn_series.len() > 0 else 0.0
    m["turnover_p90"] = float(turn_series.quantile(0.9, interpolation="nearest")) if turn_series.len() > 0 else 0.0
    m["turnover_p99"] = float(turn_series.quantile(0.99, interpolation="nearest")) if turn_series.len() > 0 else 0.0
    m["turnover_max"] = float(turn_series.max()) if turn_series.len() > 0 else 0.0
    total_net_ret = combined["r_port"].sum()
    m["edge_per_turnover_bps"] = (total_net_ret / turn_total) * 10000 if turn_total and turn_total != 0 else None
    if "cost_ret_combined" in combined.columns:
        m["cost_bps_paid"] = float(combined["cost_ret_combined"].sum() * 10000)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    plot_equity(combined, out_dir / "equity_curve_log.png")

    print("Combined 50/50 Portfolio Summary")
    print(f"Total Return : {m['total_return']:.4f}")
    print(f"CAGR         : {m['cagr']:.4f}")
    print(f"Sharpe       : {m['sharpe']:.4f}")
    print(f"Vol Ann      : {m['vol_annual']:.4f}")
    print(f"Max Drawdown : {m['max_drawdown']:.4f}")
    print(f"Max DD TS    : {m['max_drawdown_ts']}")
    print(f"Calmar       : {m.get('calmar', 0):.4f}")
    print(
        f"Turnover     : total={turn_total:.4f} events={turn_events} "
        f"p50={m['turnover_p50']:.6f} p90={m['turnover_p90']:.6f} "
        f"p99={m['turnover_p99']:.6f} max={m['turnover_max']:.6f} "
        f"edge/turn_bps={m['edge_per_turnover_bps'] if m['edge_per_turnover_bps'] is not None else 'n/a'}"
    )
    print(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()

