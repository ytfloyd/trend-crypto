from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_tearsheet(
    df: pl.DataFrame,
    output_path: str,
    *,
    alpha_name: str | None = None,
    n_quantiles: int = 5,
    rolling_window: int = 30,
    emit_returns: bool = False,
    fed_cycles_path: str | Path | None = None,
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

    net_exposure = (
        df.with_columns((pl.col("rank_pct") - 0.5).alias("rank_centered"))
        .group_by("ts")
        .agg(pl.col("rank_centered").sum().alias("net_exposure"))
        .sort("ts")
    )

    fed_cycles = _load_fed_cycles(fed_cycles_path)
    periods_per_year = _infer_periods_per_year(quantile_wide["ts"])
    crowding_window = max(60, int(round(5.0 * periods_per_year)))
    crowding = _crowding_series(df, effective_quantiles, crowding_window)

    summary = _build_summary(
        df=df,
        quantile_returns=quantile_wide,
        equity=equity,
        ic=ic,
        turnover=turnover,
        crowding=crowding,
        fed_cycles=fed_cycles,
        n_quantiles=effective_quantiles,
        skipped_timestamps=skipped,
        alpha_name=alpha_name,
        requested_quantiles=n_quantiles,
        unique_ts=unique_ts,
        unique_symbols=unique_symbols,
        min_symbols_per_ts=min_symbols,
        periods_per_year=periods_per_year,
        crowding_window=crowding_window,
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
        spread_returns=quantile_wide.select(["ts", "Spread"]),
        annual_breakdown=summary["annual_breakdown"],
        regime_breakdown=summary["regime_breakdown"],
        drawdown=summary["drawdown"],
        crowding=crowding,
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


def _hit_rate(spread_arr: np.ndarray) -> float:
    """Fraction of periods where spread > 0. Primer's "Probability of Top Quintile
    Outperforming Bottom Quintile". Ignores NaNs."""
    arr = np.asarray(spread_arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0
    return float((arr > 0).sum()) / float(arr.size)


def _nw_tstat(spread_arr: np.ndarray, lags: int | None = None) -> tuple[float, int]:
    """Newey-West autocorrelation-corrected t-stat of the mean.

    Default lag selection: floor(4 * (n/100)^(2/9)) (Newey-West 1994 rule of thumb).
    Returns (tstat, lags_used). NaNs are dropped.
    """
    arr = np.asarray(spread_arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 3:
        return (0.0, 0)
    if lags is None:
        lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))

    x = arr - arr.mean()
    gamma_0 = float(np.dot(x, x) / n)
    nw_var = gamma_0
    for k in range(1, lags + 1):
        gk = float(np.dot(x[k:], x[:-k]) / n)
        weight = 1.0 - k / (lags + 1.0)
        nw_var += 2.0 * weight * gk
    if nw_var <= 0.0:
        return (0.0, lags)
    se = math.sqrt(nw_var / n)
    if se == 0.0:
        return (0.0, lags)
    return (float(arr.mean() / se), int(lags))


def _drawdown_stats(equity_arr: np.ndarray, ts_arr: Iterable) -> dict[str, Any]:
    """Compute peak / trough / recovery and drawdown depth + lengths."""
    eq = np.asarray(equity_arr, dtype=float)
    ts = list(ts_arr)
    n = eq.size
    if n == 0:
        return {
            "max_dd": 0.0,
            "peak_ts": None,
            "trough_ts": None,
            "recovery_ts": None,
            "drawdown_periods": 0,
            "recovery_periods": None,
        }
    running_max = np.maximum.accumulate(eq)
    drawdown = eq / running_max - 1.0
    trough_idx = int(np.argmin(drawdown))
    max_dd = float(drawdown[trough_idx])
    peak_eq = float(running_max[trough_idx])
    peak_idx = int(np.where(eq[: trough_idx + 1] >= peak_eq)[0][0]) if trough_idx >= 0 else 0
    recovery_idx: int | None = None
    if trough_idx + 1 < n:
        post = eq[trough_idx + 1 :]
        rec_rel = np.where(post >= peak_eq)[0]
        if rec_rel.size > 0:
            recovery_idx = int(trough_idx + 1 + rec_rel[0])
    return {
        "max_dd": max_dd,
        "peak_ts": str(ts[peak_idx]) if peak_idx < len(ts) else None,
        "trough_ts": str(ts[trough_idx]) if trough_idx < len(ts) else None,
        "recovery_ts": str(ts[recovery_idx]) if recovery_idx is not None else None,
        "drawdown_periods": int(trough_idx - peak_idx),
        "recovery_periods": int(recovery_idx - trough_idx) if recovery_idx is not None else None,
    }


def _annual_breakdown(spread_df: pl.DataFrame, periods_per_year: float) -> list[dict[str, Any]]:
    """Per calendar year: spread mean, std, Sharpe, hit rate, n."""
    if spread_df.height == 0:
        return []
    df = spread_df.with_columns(pl.col("ts").dt.year().alias("year"))
    rows: list[dict[str, Any]] = []
    for year_val in sorted(df.select("year").unique().to_series().to_list()):
        seg = df.filter(pl.col("year") == year_val).select("Spread").to_series()
        arr = np.asarray(seg.to_list(), dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        sharpe = float(mean / std * math.sqrt(periods_per_year)) if std > 0 else 0.0
        rows.append(
            {
                "year": int(year_val),
                "mean": mean,
                "std": std,
                "sharpe": sharpe,
                "hit_rate": _hit_rate(arr),
                "n": int(arr.size),
            }
        )
    return rows


def _load_fed_cycles(path: str | Path | None) -> pl.DataFrame | None:
    """Load fed_cycles.csv into a polars DataFrame with parsed timestamps.

    Returns None if path is None or file is missing.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    cycles = pl.read_csv(p)
    needed = {"start_ts", "end_ts", "phase"}
    if not needed.issubset(set(cycles.columns)):
        raise ValueError(
            f"fed_cycles file {p} missing columns; expected {sorted(needed)}, got {cycles.columns}"
        )
    cycles = cycles.with_columns(
        [
            pl.col("start_ts").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False),
            pl.col("end_ts").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False),
        ]
    ).sort("start_ts")
    return cycles


def _attach_fed_phase(
    spread_df: pl.DataFrame, fed_cycles: pl.DataFrame
) -> pl.DataFrame:
    """Asof-join Fed phase onto a spread DataFrame, gating by end_ts."""
    spread = spread_df.sort("ts").with_columns(pl.col("ts").cast(pl.Datetime))
    cycles = fed_cycles.with_columns(
        [
            pl.col("start_ts").cast(pl.Datetime),
            pl.col("end_ts").cast(pl.Datetime),
        ]
    ).sort("start_ts")
    joined = spread.join_asof(
        cycles,
        left_on="ts",
        right_on="start_ts",
        strategy="backward",
    )
    joined = joined.with_columns(
        pl.when(pl.col("ts") <= pl.col("end_ts"))
        .then(pl.col("phase"))
        .otherwise(None)
        .alias("phase")
    )
    return joined


def _regime_breakdown(
    spread_df: pl.DataFrame,
    fed_cycles: pl.DataFrame | None,
    periods_per_year: float,
) -> list[dict[str, Any]]:
    """Per Fed-cycle phase: mean, std, Sharpe, hit rate, n."""
    if fed_cycles is None or spread_df.height == 0:
        return []
    joined = _attach_fed_phase(spread_df, fed_cycles)
    rows: list[dict[str, Any]] = []
    for phase in ["early_hike", "late_hike", "easing", "neutral"]:
        seg = joined.filter(pl.col("phase") == phase).select("Spread").to_series()
        arr = np.asarray(seg.to_list(), dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        sharpe = float(mean / std * math.sqrt(periods_per_year)) if std > 0 else 0.0
        rows.append(
            {
                "phase": phase,
                "mean": mean,
                "std": std,
                "sharpe": sharpe,
                "hit_rate": _hit_rate(arr),
                "n": int(arr.size),
            }
        )
    return rows


def _crowding_series(df: pl.DataFrame, n_quantiles: int, window: int) -> pl.DataFrame:
    """Long-basket signal level + trailing percentile vs. its own history.

    Acts as a price-based stand-in for the BofA primer's "Long Only Funds'
    Relative Wt." crowding measure (Exhibit 372). Without fundamentals we use
    the long-basket's average raw signal value and its rolling percentile
    relative to a trailing window of itself.
    """
    long_basket = (
        df.filter(pl.col("quantile") == n_quantiles - 1)
        .group_by("ts")
        .agg(pl.col("signal").mean().alias("long_signal"))
        .sort("ts")
    )
    if long_basket.height == 0:
        return pl.DataFrame(
            {"ts": [], "long_signal": [], "pct_history": []},
            schema={"ts": pl.Datetime, "long_signal": pl.Float64, "pct_history": pl.Float64},
        )
    arr = np.asarray(long_basket["long_signal"].to_list(), dtype=float)
    pct = _rolling_percentile(arr, window)
    out = long_basket.with_columns(pl.Series("pct_history", pct))
    return out


def _rolling_percentile(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing percentile rank of each value within its prior `window` history.

    pct[i] = fraction of values strictly less than arr[i] within the trailing
    window ending at i. Returns 0.5 for the very first observation. NaN values
    in arr are skipped within each window.
    """
    n = arr.size
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = arr[lo : i + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size <= 1:
            out[i] = 0.5 if seg.size == 1 else np.nan
        else:
            out[i] = float((seg < arr[i]).sum()) / float(seg.size - 1)
    return out


def _build_summary(
    *,
    df: pl.DataFrame,
    quantile_returns: pl.DataFrame,
    equity: pl.DataFrame,
    ic: pl.DataFrame,
    turnover: pl.DataFrame,
    crowding: pl.DataFrame,
    fed_cycles: pl.DataFrame | None,
    n_quantiles: int,
    skipped_timestamps: int,
    alpha_name: str | None,
    requested_quantiles: int,
    unique_ts: int,
    unique_symbols: int,
    min_symbols_per_ts: int,
    periods_per_year: float,
    crowding_window: int,
) -> dict[str, Any]:
    n_rows = df.height
    n_symbols = df.select(pl.col("symbol").n_unique()).item()
    ts_min = df.select(pl.col("ts").min()).item()
    ts_max = df.select(pl.col("ts").max()).item()

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

    spread_arr = np.asarray(spread_series.to_list(), dtype=float)
    hit_rate = _hit_rate(spread_arr)
    nw_t, nw_lags = _nw_tstat(spread_arr)

    spread_pairs = quantile_returns.select(["ts", "Spread"]).sort("ts")
    annual = _annual_breakdown(spread_pairs, periods_per_year)
    regime = _regime_breakdown(spread_pairs, fed_cycles, periods_per_year)

    spread_equity = equity.select(["ts", "Spread"]).sort("ts")
    dd = _drawdown_stats(
        np.asarray(spread_equity["Spread"].to_list(), dtype=float),
        spread_equity["ts"].to_list(),
    )

    if crowding.height > 0:
        last_signal = float(crowding["long_signal"].to_list()[-1])
        last_pct_arr = np.asarray(crowding["pct_history"].to_list(), dtype=float)
        last_pct_arr = last_pct_arr[~np.isnan(last_pct_arr)]
        last_pct = float(last_pct_arr[-1]) if last_pct_arr.size else float("nan")
        mean_pct = float(last_pct_arr.mean()) if last_pct_arr.size else float("nan")
        crowd_summary = {
            "long_signal_last": last_signal,
            "pct_history_last": last_pct,
            "pct_history_mean": mean_pct,
            "window_periods": int(crowding_window),
        }
    else:
        crowd_summary = {
            "long_signal_last": float("nan"),
            "pct_history_last": float("nan"),
            "pct_history_mean": float("nan"),
            "window_periods": int(crowding_window),
        }

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
        "hit_rate": float(hit_rate),
        "spread_nw_tstat": float(nw_t),
        "spread_nw_lags": int(nw_lags),
        "annual_breakdown": annual,
        "regime_breakdown": regime,
        "drawdown": dd,
        "crowding": crowd_summary,
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
    spread_returns: pl.DataFrame,
    annual_breakdown: list[dict[str, Any]],
    regime_breakdown: list[dict[str, Any]],
    drawdown: dict[str, Any],
    crowding: pl.DataFrame,
) -> None:
    plt.style.use("ggplot")

    title_suffix = f" — {alpha_name}" if alpha_name else ""

    fig1, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))

    ax = axes[0, 0]
    equity_pd = equity.to_pandas()
    colors = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4", "#313695"]
    for i in range(n_quantiles):
        col = f"Q{i + 1}"
        ax.plot(equity_pd["ts"], equity_pd[col], label=col, color=colors[i % len(colors)])
    ax.plot(equity_pd["ts"], equity_pd["Spread"], label="Spread", color="black")
    ax.set_title(f"Layer Cake: Cumulative Returns by Quantile{title_suffix}")
    ax.legend(loc="best", fontsize=8)

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

    ax = axes[1, 0]
    turnover_pd = turnover.to_pandas()
    ax.plot(turnover_pd["ts"], turnover_pd["turnover"], color="purple", linewidth=1)
    ax.axhline(mean_turnover, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Cost Barrier: Turnover (Mean: {mean_turnover:.2%})")

    ax = axes[1, 1]
    net_pd = net_exposure.to_pandas()
    ax.plot(net_pd["ts"], net_pd["net_exposure"], color="teal", linewidth=1)
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-0.1, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Bias Detector: Net Rank Exposure")

    fig1.tight_layout()

    fig2, axes2 = plt.subplots(2, 2, figsize=(11.7, 8.3))

    ax = axes2[0, 0]
    if annual_breakdown:
        years = [r["year"] for r in annual_breakdown]
        sharpes = [r["sharpe"] for r in annual_breakdown]
        hits = [r["hit_rate"] for r in annual_breakdown]
        bar_colors = ["#2c7bb6" if s >= 0 else "#d7191c" for s in sharpes]
        ax.bar(years, sharpes, color=bar_colors)
        ax.axhline(0, color="black", linewidth=1)
        for x, y, h in zip(years, sharpes, hits):
            ax.text(
                x,
                y,
                f"{h:.0%}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=7,
            )
        ax.set_title("Annual Spread Sharpe (label = hit rate)")
    else:
        ax.set_title("Annual Spread Sharpe (no data)")
        ax.axis("off")

    ax = axes2[0, 1]
    if regime_breakdown:
        phases = [r["phase"] for r in regime_breakdown]
        sharpes = [r["sharpe"] for r in regime_breakdown]
        hits = [r["hit_rate"] for r in regime_breakdown]
        ns = [r["n"] for r in regime_breakdown]
        bar_colors = ["#2c7bb6" if s >= 0 else "#d7191c" for s in sharpes]
        ax.bar(phases, sharpes, color=bar_colors)
        ax.axhline(0, color="black", linewidth=1)
        for x, y, h, nval in zip(phases, sharpes, hits, ns):
            ax.text(
                x,
                y,
                f"{h:.0%}\nn={nval}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=7,
            )
        ax.set_title("Spread Sharpe by Fed Regime")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.set_title("Spread Sharpe by Fed Regime (no fed_cycles)")
        ax.axis("off")

    ax = axes2[1, 0]
    spread_pd = spread_returns.to_pandas().sort_values("ts").reset_index(drop=True)
    eq = (1.0 + spread_pd["Spread"].fillna(0.0)).cumprod()
    running_max = eq.cummax()
    underwater = eq / running_max - 1.0
    ax.fill_between(spread_pd["ts"], underwater, 0.0, color="#d7191c", alpha=0.4)
    ax.plot(spread_pd["ts"], underwater, color="#d7191c", linewidth=1)
    ax.set_title(
        f"Spread Underwater  (Max DD: {drawdown.get('max_dd', 0.0):.1%},"
        f" Recovery: {drawdown.get('recovery_periods')})"
    )

    ax = axes2[1, 1]
    if crowding.height > 0:
        crowd_pd = crowding.to_pandas()
        ax.plot(
            crowd_pd["ts"],
            crowd_pd["pct_history"],
            color="#762a83",
            linewidth=1.0,
        )
        ax.axhline(0.5, color="gray", linewidth=1, linestyle="--")
        ax.fill_between(
            crowd_pd["ts"],
            crowd_pd["pct_history"],
            0.8,
            where=crowd_pd["pct_history"] >= 0.8,
            color="#762a83",
            alpha=0.25,
        )
        ax.set_ylim(0, 1)
        ax.set_title("Crowding Proxy: Long-Basket Signal Pct vs Trailing History")
    else:
        ax.set_title("Crowding Proxy (no data)")
        ax.axis("off")

    fig2.tight_layout()

    out_base = Path(output_path)
    pdf_path = out_base.with_suffix(".pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
    fig1.savefig(out_base.with_suffix(".png"), dpi=150)
    fig2.savefig(out_base.with_name(out_base.stem + "_p2").with_suffix(".png"), dpi=150)

    plt.close(fig1)
    plt.close(fig2)


def _write_summary(output_path: str, summary: dict[str, Any]) -> None:
    out_base = Path(output_path)
    with open(out_base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
