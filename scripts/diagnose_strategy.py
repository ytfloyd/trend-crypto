from __future__ import annotations

import argparse
import json
from math import sqrt, isnan
from pathlib import Path
from typing import Dict, Tuple

import polars as pl


def load_equity(run_dir: Path) -> pl.DataFrame:
    path = run_dir / "equity.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity.parquet in {run_dir}")
    df = pl.read_parquet(path)
    cols = [c for c in ["ts", "nav", "net_ret", "turnover"] if c in df.columns]
    df = df.select(cols).sort("ts")
    if "net_ret" not in df.columns:
        df = df.with_columns((pl.col("nav") / pl.col("nav").shift(1) - 1).alias("net_ret"))
    if "turnover" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("turnover"))
    df = df.drop_nulls(subset=["ts", "nav", "net_ret"])
    return df


def infer_periods_per_year(ts: pl.Series) -> float:
    diffs = ts.diff().dt.total_seconds().drop_nulls()
    if diffs.is_empty():
        return 8760.0
    dt_seconds = diffs.median()
    return (365 * 24 * 3600) / dt_seconds if dt_seconds and dt_seconds > 0 else 8760.0


def sharpe(returns: pl.Series, ppy: float) -> float:
    from common.metrics import compute_sharpe
    if not isinstance(returns, pl.Series):
        raise TypeError("sharpe() expects a polars Series, got Expr; ensure you pass df.get_column(...)")
    return compute_sharpe(returns, ppy)


def max_drawdown(nav: pl.Series) -> Tuple[float, pl.Series]:
    running_max = nav.cum_max()
    dd = (nav / running_max) - 1
    return dd.min(), dd


def compound_group(df: pl.DataFrame, bucket: str, ret_col: str, turn_col: str | None = None) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("ts").dt.truncate(bucket).alias("bucket"))
        .group_by("bucket")
        .agg(
            [
                ((1 + pl.col(ret_col)).product() - 1).alias("ret"),
                (pl.col(turn_col).sum() if turn_col else pl.lit(None)).alias("turnover")
                if turn_col
                else pl.lit(None).alias("turnover"),
            ]
        )
        .sort("bucket")
    )


def beta_alpha_ir(strat: pl.Series, bench: pl.Series, ppy: float) -> Dict[str, float]:
    assert isinstance(strat, pl.Series)
    assert isinstance(bench, pl.Series)
    strat = strat.drop_nulls()
    bench = bench.drop_nulls()
    mean_s = float(strat.mean())
    mean_b = float(bench.mean())
    var_b = float(bench.var(ddof=1))
    cov = float(((strat - mean_s) * (bench - mean_b)).mean())
    beta = cov / var_b if var_b > 0 else float("nan")
    alpha_ann = (mean_s - beta * mean_b) * ppy if not isnan(beta) else float("nan")
    active = strat - (beta * bench)
    ir = sharpe(active, ppy) if not isnan(beta) else 0.0
    return {"beta": beta, "alpha_ann": alpha_ann, "ir": ir}


def capture_ratios(strat: pl.Series, bench: pl.Series) -> Dict[str, float]:
    up_mask = bench > 0
    down_mask = bench < 0
    up_strat = strat.filter(up_mask)
    up_bench = bench.filter(up_mask)
    down_strat = strat.filter(down_mask)
    down_bench = bench.filter(down_mask)
    up_cap = up_strat.sum() / up_bench.sum() if up_bench.sum() != 0 else 0.0
    down_cap = down_strat.sum() / down_bench.sum() if down_bench.sum() != 0 else 0.0
    up_part = (strat.filter(up_mask) > 0).sum() / up_mask.sum() if up_mask.sum() > 0 else 0.0
    return {"up_capture": up_cap, "down_capture": down_cap, "up_participation": up_part}


def leverage_thought_experiment(strat: pl.Series, ppy: float, target_vol_annual: float) -> Dict[str, float]:
    vol_ann = strat.std(ddof=1) * sqrt(ppy) if strat.std(ddof=1) is not None else 0.0
    if not vol_ann or vol_ann <= 0:
        return {"leverage": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    lev = target_vol_annual / vol_ann
    lev_ret = strat * lev
    nav = (1 + lev_ret).cum_prod()
    total_ret = nav[-1] - 1 if len(nav) > 0 else 0.0
    n = len(lev_ret)
    cagr = (1 + total_ret) ** (ppy / n) - 1 if n > 0 else 0.0
    mdd, _ = max_drawdown(nav)
    return {
        "leverage": lev,
        "cagr": cagr,
        "sharpe": sharpe(lev_ret, ppy),
        "max_dd": mdd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy diagnostics vs benchmark (Engine v2 outputs).")
    parser.add_argument("--run_strategy", required=True)
    parser.add_argument("--run_benchmark", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name_strategy", default="Strategy")
    parser.add_argument("--name_benchmark", default="Benchmark")
    parser.add_argument("--top_weeks", type=int, default=10)
    parser.add_argument("--weekly_bucket", default="1w")
    parser.add_argument("--target_vol_annual", type=float, default=0.80)
    parser.add_argument("--min_overlap_days", type=int, default=365)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strat = load_equity(Path(args.run_strategy)).rename({"net_ret": "r_s"})
    bench = load_equity(Path(args.run_benchmark)).rename({"net_ret": "r_b"})
    df = strat.join(bench, on="ts", how="inner", suffix="_bench")
    if df.height < 2:
        raise ValueError("Insufficient overlap between strategy and benchmark.")

    df = df.drop_nulls(subset=["r_s", "r_b"]).select(
        ["ts", "nav", "r_s", "turnover", "nav_bench", "r_b", "turnover_bench"]
    ).sort("ts")
    ts_span_days = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400
    ppy = infer_periods_per_year(df["ts"])

    nav = df["nav"]
    strat_ret = df.get_column("r_s")
    bench_ret = df.get_column("r_b")
    assert isinstance(strat_ret, pl.Series)
    assert isinstance(bench_ret, pl.Series)

    total_return = nav[-1] / nav[0] - 1
    n = len(strat_ret)
    cagr = (1 + total_return) ** (ppy / n) - 1 if n > 0 else 0.0
    shp = sharpe(strat_ret, ppy)
    mdd, dd_series = max_drawdown(nav)
    vol_ann = strat_ret.std(ddof=1) * sqrt(ppy) if strat_ret.std(ddof=1) is not None else 0.0

    cap = capture_ratios(strat_ret, bench_ret)
    beta_stats = beta_alpha_ir(strat_ret, bench_ret, ppy)

    weekly = compound_group(df, args.weekly_bucket, "r_b", None)
    weekly = df.with_columns(pl.col("ts").dt.truncate(args.weekly_bucket).alias("bucket")).group_by("bucket").agg(
        [
            ((1 + pl.col("r_b")).product() - 1).alias("bench_week_ret"),
            ((1 + pl.col("r_s")).product() - 1).alias("strat_week_ret"),
            pl.col("turnover").sum().alias("strat_week_turnover"),
        ]
    ).sort("bucket")

    top_weeks = weekly.sort("bench_week_ret", descending=True).head(args.top_weeks)
    rest_weeks = weekly.filter(~pl.col("bucket").is_in(top_weeks["bucket"].implode()))
    top_capture = {
        "bench_sum": top_weeks["bench_week_ret"].sum(),
        "strat_sum": top_weeks["strat_week_ret"].sum(),
        "capture": top_weeks["strat_week_ret"].sum() / top_weeks["bench_week_ret"].sum()
        if top_weeks["bench_week_ret"].sum() != 0
        else 0.0,
    }
    rest_capture = {
        "bench_sum": rest_weeks["bench_week_ret"].sum(),
        "strat_sum": rest_weeks["strat_week_ret"].sum(),
        "capture": rest_weeks["strat_week_ret"].sum() / rest_weeks["bench_week_ret"].sum()
        if rest_weeks["bench_week_ret"].sum() != 0
        else 0.0,
    }

    turn_stats = {}
    if "turnover" in df.columns:
        total_turn = df["turnover"].sum()
        turn_events = (df["turnover"] > 0).sum()
        edge_per_turn = strat_ret.sum() / total_turn if total_turn != 0 else 0.0
        turn_stats = {
            "total_turnover": total_turn,
            "turnover_events": int(turn_events),
            "edge_per_turnover": edge_per_turn,
            "bps_per_turnover": edge_per_turn * 10000,
        }

    lev_stats = leverage_thought_experiment(strat_ret, ppy, args.target_vol_annual)

    diagnostics = {
        "name_strategy": args.name_strategy,
        "name_benchmark": args.name_benchmark,
        "periods_per_year": ppy,
        "span_days": ts_span_days,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": shp,
        "realized_vol_annual": vol_ann,
        "max_drawdown": mdd,
        "beta": beta_stats["beta"],
        "alpha_ann": beta_stats["alpha_ann"],
        "information_ratio": beta_stats["ir"],
        "upside_capture": cap["up_capture"],
        "downside_capture": cap["down_capture"],
        "upside_participation": cap["up_participation"],
        "top_weeks": top_capture,
        "rest_weeks": rest_capture,
        "turnover": turn_stats,
        "leverage_thought_experiment": lev_stats,
        "top_weeks_capture": top_capture["capture"],
        "rest_weeks_capture": rest_capture["capture"],
        "turnover_events": turn_stats.get("turnover_events"),
        "total_turnover": turn_stats.get("total_turnover"),
        "bps_per_unit_turnover": turn_stats.get("bps_per_turnover"),
        "lev_to_target_vol": lev_stats.get("leverage"),
        "levered_cagr": lev_stats.get("cagr"),
        "levered_sharpe": lev_stats.get("sharpe"),
        "levered_maxdd": lev_stats.get("max_dd"),
    }

    weekly.write_csv(out_dir / "top_weeks.csv")
    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Strategy: {args.name_strategy} vs Benchmark: {args.name_benchmark}")
    print(f"Overlap days: {ts_span_days:.1f}, periods/year: {ppy:.2f}")
    print(f"Total Return: {total_return:.4f}, CAGR: {cagr:.4f}, Sharpe: {shp:.4f}, VolAnn: {vol_ann:.4f}")
    print(f"MaxDD: {mdd:.4f}, Beta: {beta_stats['beta']:.4f}, Alpha_ann: {beta_stats['alpha_ann']:.4f}, IR: {beta_stats['ir']:.4f}")
    print(f"Upside Capture: {cap['up_capture']:.4f}, Downside Capture: {cap['down_capture']:.4f}, Upside Participation: {cap['up_participation']:.4f}")
    print(f"Top-{args.top_weeks} weeks capture: {top_capture['capture']:.4f}, Rest weeks capture: {rest_capture['capture']:.4f}")
    if turn_stats:
        print(
            f"Turnover: total={turn_stats['total_turnover']:.4f}, events={turn_stats['turnover_events']}, "
            f"edge/turn={turn_stats['edge_per_turnover']:.6f} ({turn_stats['bps_per_turnover']:.2f} bps)"
        )
    print(
        f"Lever to target vol {args.target_vol_annual:.2f}: lev={lev_stats.get('leverage',0):.3f}, "
        f"CAGR={lev_stats.get('cagr',0):.4f}, Sharpe={lev_stats.get('sharpe',0):.4f}, MaxDD={lev_stats.get('max_dd',0):.4f}"
    )


if __name__ == "__main__":
    main()

