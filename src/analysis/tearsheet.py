from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TearsheetArtifacts:
    summary: dict[str, Any]
    quantile_equity: pl.DataFrame
    ic_series: pl.DataFrame
    turnover: pl.DataFrame
    net_exposure: pl.DataFrame


def generate_tearsheet(
    df: pl.DataFrame,
    output_path: str,
    *,
    alpha_name: str | None = None,
    n_quantiles: int = 5,
    rolling_window: int = 30,
    emit_returns: bool = False,
) -> dict:
    required = {"ts", "symbol", "signal", "forward_ret"}
    missing = required - set(df.columns)
    if missing:
        if "forward_ret" in missing and "fwd_ret" in df.columns:
            raise ValueError(
                "Expected column 'forward_ret'. Did you mean 'fwd_ret'? "
                "Rename in the runner."
            )
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    n_rows_input = df.height
    ts_min_input = df.select(pl.col("ts").min()).item()
    ts_max_input = df.select(pl.col("ts").max()).item()
    null_signal = df.select(pl.col("signal").is_null().sum()).item()
    null_forward = df.select(pl.col("forward_ret").is_null().sum()).item()
    nonfinite_signal = df.select((~pl.col("signal").is_finite()).sum()).item()
    nonfinite_forward = df.select((~pl.col("forward_ret").is_finite()).sum()).item()

    df = df.filter(~pl.col("signal").is_null() & ~pl.col("forward_ret").is_null())
    n_after_null = df.height

    df = df.filter(
        pl.col("signal").is_finite() & pl.col("forward_ret").is_finite()
    )
    n_after_finite = df.height
    if df.is_empty():
        raise ValueError(
            "Empty after filtering. "
            f"n_rows_input={n_rows_input}, n_after_null={n_after_null}, "
            f"n_after_finite={n_after_finite}, "
            f"null_signal={null_signal}, null_forward_ret={null_forward}, "
            f"nonfinite_signal={nonfinite_signal}, nonfinite_forward_ret={nonfinite_forward}, "
            f"ts_min={ts_min_input}, ts_max={ts_max_input}"
        )

    dupes = df.group_by(["ts", "symbol"]).len().filter(pl.col("len") > 1)
    if dupes.height > 0:
        raise ValueError("Duplicate (ts, symbol) rows detected in input.")

    df = df.sort(["ts", "symbol"])

    counts = df.group_by("ts").len().rename({"len": "n_symbols"})
    min_symbols = counts.select(pl.col("n_symbols").min()).item()
    unique_ts = counts.height
    unique_symbols = df.select(pl.col("symbol").n_unique()).item()

    effective_quantiles = min(n_quantiles, max(2, int(min_symbols or 0)))
    if effective_quantiles < 2:
        raise ValueError(
            "Insufficient symbols per timestamp for quantile tearsheet. "
            f"min_symbols_per_ts={min_symbols}, requested_quantiles={n_quantiles}"
        )

    df = df.join(counts, on="ts", how="left")
    skipped = counts.filter(pl.col("n_symbols") < effective_quantiles).height
    df = df.filter(pl.col("n_symbols") >= effective_quantiles)
    n_after_gate = df.height
    if df.is_empty():
        ts_min_post = df.select(pl.col("ts").min()).item() if df.height else "n/a"
        ts_max_post = df.select(pl.col("ts").max()).item() if df.height else "n/a"
        raise ValueError(
            "Empty after min-symbols gate. "
            f"n_rows_input={n_rows_input}, n_after_null={n_after_null}, "
            f"n_after_finite={n_after_finite}, n_after_gate={n_after_gate}, "
            f"unique_ts={unique_ts}, unique_symbols={unique_symbols}, "
            f"min_symbols_per_ts={min_symbols}, requested_quantiles={n_quantiles}, "
            f"effective_quantiles={effective_quantiles}, "
            f"ts_min={ts_min_input}, ts_max={ts_max_input}, "
            f"ts_min_post={ts_min_post}, ts_max_post={ts_max_post}"
        )

    rank = pl.col("signal").rank(method="average").over("ts")
    rank_pct = pl.when(pl.col("n_symbols") > 1).then(
        (rank - 1) / (pl.col("n_symbols") - 1)
    ).otherwise(0.0)
    quantile = (
        (rank_pct * effective_quantiles)
        .floor()
        .clip(0, effective_quantiles - 1)
        .cast(pl.Int64)
        .alias("quantile")
    )

    df = df.with_columns([rank_pct.alias("rank_pct"), quantile])

    quantile_returns = (
        df.group_by(["ts", "quantile"])
        .agg(pl.col("forward_ret").mean().alias("ret"))
        .sort(["ts", "quantile"])
        .with_columns(pl.col("quantile").cast(pl.Utf8).alias("quantile_label"))
    )

    quantile_wide = quantile_returns.pivot(
        values="ret", index="ts", on="quantile_label", aggregate_function="first"
    ).sort("ts")

    quantile_cols = []
    for q in range(effective_quantiles):
        col = str(q)
        if col in quantile_wide.columns:
            quantile_wide = quantile_wide.rename({col: f"Q{q + 1}"})
        quantile_cols.append(f"Q{q + 1}")

    for col in quantile_cols:
        if col not in quantile_wide.columns:
            quantile_wide = quantile_wide.with_columns(pl.lit(0.0).alias(col))

    quantile_wide = quantile_wide.with_columns(
        (pl.col(f"Q{effective_quantiles}") - pl.col("Q1")).alias("Spread")
    )

    equity_cols = [f"Q{q}" for q in range(1, effective_quantiles + 1)] + ["Spread"]
    equity = quantile_wide.select(
        ["ts"]
        + [
            (pl.col(col) + 1.0).cum_prod().alias(col)
            for col in equity_cols
        ]
    )

    # Turnover
    n_long = (
        df.filter(pl.col("quantile") == effective_quantiles - 1)
        .group_by("ts")
        .len()
        .rename({"len": "n_long"})
    )
    n_short = (
        df.filter(pl.col("quantile") == 0)
        .group_by("ts")
        .len()
        .rename({"len": "n_short"})
    )
    df = df.join(n_long, on="ts", how="left").join(n_short, on="ts", how="left")
    df = df.with_columns(
        pl.when(pl.col("quantile") == effective_quantiles - 1)
        .then(1.0 / pl.col("n_long"))
        .when(pl.col("quantile") == 0)
        .then(-1.0 / pl.col("n_short"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("weight")
    )

    df = df.sort(["symbol", "ts"]).with_columns(
        pl.col("weight").shift(1).over("symbol").fill_null(0.0).alias("weight_prev")
    )
    turnover = (
        df.with_columns((pl.col("weight") - pl.col("weight_prev")).abs().alias("dw"))
        .group_by("ts")
        .agg((0.5 * pl.col("dw").sum()).alias("turnover"))
        .sort("ts")
    )

    # Daily IC (Spearman)
    df = df.with_columns(
        [
            pl.col("signal").rank(method="average").over("ts").alias("signal_rank"),
            pl.col("forward_ret").rank(method="average").over("ts").alias("ret_rank"),
        ]
    )
    ic = (
        df.group_by("ts")
        .agg(pl.corr(pl.col("signal_rank"), pl.col("ret_rank")).alias("ic"))
        .sort("ts")
    )
    ic = ic.with_columns(
        pl.col("ic").rolling_mean(window_size=rolling_window, min_samples=1).alias("ic_roll")
    )

    # Net exposure (bias detector)
    net_exposure = (
        df.with_columns((pl.col("rank_pct") - 0.5).alias("rank_centered"))
        .group_by("ts")
        .agg(pl.col("rank_centered").sum().alias("net_exposure"))
        .sort("ts")
    )

    summary = _build_summary(
        df=df,
        quantile_returns=quantile_wide,
        ic=ic,
        turnover=turnover,
        n_quantiles=effective_quantiles,
        skipped_timestamps=skipped,
        alpha_name=alpha_name,
        requested_quantiles=n_quantiles,
        unique_ts=unique_ts,
        unique_symbols=unique_symbols,
        min_symbols_per_ts=min_symbols,
    )

    _plot_dashboard(
        equity=equity,
        ic=ic,
        turnover=turnover,
        net_exposure=net_exposure,
        rolling_window=rolling_window,
        n_quantiles=effective_quantiles,
        alpha_name=alpha_name,
        output_path=output_path,
        mean_turnover=summary["mean_daily_turnover"],
    )

    if emit_returns:
        spread_returns = quantile_wide.select(
            [pl.col("ts"), pl.col("Spread").alias("spread_ret")]
        ).sort("ts")
        spread_path = Path(output_path).with_suffix(".spread_returns.parquet")
        spread_returns.write_parquet(spread_path)
        summary["spread_returns_path"] = str(spread_path)
        summary["n_spread_points"] = int(spread_returns.height)

    _write_summary(output_path, summary)
    return summary


def _infer_periods_per_year(ts: pl.Series) -> float:
    if ts.len() < 2:
        return 365.0
    diffs = ts.sort().diff().dt.total_seconds().drop_nulls()
    if diffs.len() == 0:
        return 365.0
    dt_seconds = diffs.median()
    if not dt_seconds or dt_seconds <= 0:
        return 365.0
    return float(365 * 24 * 3600 / dt_seconds)


def _build_summary(
    *,
    df: pl.DataFrame,
    quantile_returns: pl.DataFrame,
    ic: pl.DataFrame,
    turnover: pl.DataFrame,
    n_quantiles: int,
    skipped_timestamps: int,
    alpha_name: str | None,
    requested_quantiles: int,
    unique_ts: int,
    unique_symbols: int,
    min_symbols_per_ts: int,
) -> dict[str, Any]:
    n_rows = df.height
    n_symbols = df.select(pl.col("symbol").n_unique()).item()
    ts_min = df.select(pl.col("ts").min()).item()
    ts_max = df.select(pl.col("ts").max()).item()

    periods_per_year = _infer_periods_per_year(quantile_returns["ts"])

    ic_series = ic["ic"].drop_nulls()
    mean_ic = ic_series.mean() if ic_series.len() > 0 else 0.0
    std_ic = ic_series.std(ddof=1) if ic_series.len() > 1 else 0.0
    tstat_ic = (
        mean_ic / (std_ic / (ic_series.len() ** 0.5))
        if std_ic and ic_series.len() > 1
        else 0.0
    )

    mean_turnover = turnover["turnover"].mean() if turnover.height > 0 else 0.0

    sharpe_by_quantile: dict[str, float] = {}
    avg_ret_by_quantile: dict[str, float] = {}
    for q in range(1, n_quantiles + 1):
        col = f"Q{q}"
        series = quantile_returns[col].drop_nulls()
        avg_ret = series.mean() if series.len() > 0 else 0.0
        std_ret = series.std(ddof=1) if series.len() > 1 else 0.0
        sharpe = (
            (avg_ret / std_ret) * (periods_per_year ** 0.5)
            if std_ret and std_ret > 0
            else 0.0
        )
        sharpe_by_quantile[col] = float(sharpe)
        avg_ret_by_quantile[col] = float(avg_ret)

    spread_series = quantile_returns["Spread"].drop_nulls()
    spread_mean = spread_series.mean() if spread_series.len() > 0 else 0.0
    spread_std = spread_series.std(ddof=1) if spread_series.len() > 1 else 0.0
    spread_sharpe = (
        (spread_mean / spread_std) * (periods_per_year ** 0.5)
        if spread_std and spread_std > 0
        else 0.0
    )

    return {
        "alpha_name": alpha_name,
        "n_rows": int(n_rows),
        "n_symbols": int(n_symbols),
        "date_range": [str(ts_min), str(ts_max)],
        "requested_quantiles": int(requested_quantiles),
        "effective_quantiles": int(n_quantiles),
        "mean_ic": float(mean_ic),
        "std_ic": float(std_ic),
        "tstat_ic": float(tstat_ic),
        "mean_daily_turnover": float(mean_turnover),
        "sharpe_by_quantile": sharpe_by_quantile,
        "spread_sharpe": float(spread_sharpe),
        "avg_forward_ret_by_quantile": avg_ret_by_quantile,
        "periods_per_year": float(periods_per_year),
        "skipped_timestamps": int(skipped_timestamps),
        "unique_ts": int(unique_ts),
        "unique_symbols": int(unique_symbols),
        "min_symbols_per_ts": int(min_symbols_per_ts),
    }


def _plot_dashboard(
    *,
    equity: pl.DataFrame,
    ic: pl.DataFrame,
    turnover: pl.DataFrame,
    net_exposure: pl.DataFrame,
    rolling_window: int,
    n_quantiles: int,
    alpha_name: str | None,
    output_path: str,
    mean_turnover: float,
) -> None:
    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))

    title_suffix = f" — {alpha_name}" if alpha_name else ""

    # Layer Cake
    ax = axes[0, 0]
    equity_pd = equity.to_pandas()
    colors = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4", "#313695"]
    for i in range(n_quantiles):
        col = f"Q{i + 1}"
        ax.plot(equity_pd["ts"], equity_pd[col], label=col, color=colors[i % len(colors)])
    ax.plot(equity_pd["ts"], equity_pd["Spread"], label="Spread", color="black")
    ax.set_title(f"Layer Cake: Cumulative Returns by Quantile{title_suffix}")
    ax.legend(loc="best", fontsize=8)

    # Rolling IC
    ax = axes[0, 1]
    ic_pd = ic.to_pandas()
    ax.plot(ic_pd["ts"], ic_pd["ic_roll"], color="black", linewidth=1.2)
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.fill_between(
        ic_pd["ts"],
        ic_pd["ic_roll"],
        0.0,
        where=ic_pd["ic_roll"] >= 0,
        color="green",
        alpha=0.2,
    )
    ax.fill_between(
        ic_pd["ts"],
        ic_pd["ic_roll"],
        0.0,
        where=ic_pd["ic_roll"] < 0,
        color="red",
        alpha=0.2,
    )
    ax.set_title(f"Heartbeat: Rolling IC ({rolling_window}D)")

    # Turnover
    ax = axes[1, 0]
    turnover_pd = turnover.to_pandas()
    ax.plot(turnover_pd["ts"], turnover_pd["turnover"], color="purple", linewidth=1)
    ax.axhline(mean_turnover, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Cost Barrier: Turnover (Mean: {mean_turnover:.2%})")

    # Net Exposure
    ax = axes[1, 1]
    net_pd = net_exposure.to_pandas()
    ax.plot(net_pd["ts"], net_pd["net_exposure"], color="teal", linewidth=1)
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-0.1, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Bias Detector: Net Rank Exposure")

    fig.tight_layout()

    out_base = Path(output_path)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    plt.close(fig)


def _write_summary(output_path: str, summary: dict[str, Any]) -> None:
    out_base = Path(output_path)
    with open(out_base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
