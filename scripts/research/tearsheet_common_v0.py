#!/usr/bin/env python
"""
Common tear sheet helpers for research:
- Load strategy equity and compute stats (CAGR, vol, Sharpe, MaxDD).
- Load BTC buy-and-hold benchmark equity (ts, equity), align, normalize.
- Plot strategy vs benchmark overlays (equity, drawdown, rolling risk).
- Build BTC vs Strategy summary tables (strategy stats must come from metrics CSV).

All new research tear sheets should consume these helpers for consistency.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANN_FACTOR = 365.0


def resolve_tearsheet_inputs(
    research_dir: Optional[str] = None,
    equity_csv: Optional[str] = None,
    metrics_csv: Optional[str] = None,
    equity_patterns: Optional[List[str]] = None,
    metrics_patterns: Optional[List[str]] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    Resolve equity/metrics (and optional manifest) with strict single-choice semantics.
    - If explicit paths are provided, validate and return them.
    - Else, search research_dir with patterns and require exactly one candidate each.
    """
    equity_patterns = equity_patterns or ["*equity*.csv"]
    metrics_patterns = metrics_patterns or ["*metrics*.csv"]

    def _validate_one(path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
        return str(p)

    if equity_csv and metrics_csv:
        return _validate_one(equity_csv), _validate_one(metrics_csv), None

    if not research_dir:
        raise ValueError("Must provide either explicit equity_csv + metrics_csv or a research_dir.")

    rd = Path(research_dir)
    if not rd.exists():
        raise FileNotFoundError(f"research_dir not found: {rd}")

    def _find_one(patterns: List[str], label: str) -> str:
        candidates: List[str] = []
        for pat in patterns:
            candidates.extend([str(Path(p)) for p in glob.glob(str(rd / pat))])
        uniq = sorted(set(candidates))
        if len(uniq) != 1:
            raise ValueError(f"Expected exactly 1 {label} in {rd}, found {len(uniq)}: {uniq}")
        return uniq[0]

    eq = _find_one(equity_patterns, "equity CSV")
    met = _find_one(metrics_patterns, "metrics CSV")
    manifest = rd / "run_manifest.json"
    manifest_path = str(manifest) if manifest.exists() else None
    return eq, met, manifest_path


def build_provenance_text(equity_csv: str, metrics_csv: str, manifest_path: Optional[str] = None) -> str:
    lines = [
        "Provenance",
        f"equity_csv: {equity_csv}",
        f"metrics_csv: {metrics_csv}",
    ]
    if manifest_path and Path(manifest_path).exists():
        try:
            m = json.loads(Path(manifest_path).read_text())
            lines.append(f"manifest: {manifest_path}")
            for k in ["strategy_id", "git_branch", "git_sha", "timestamp_utc", "config_hash"]:
                if k in m:
                    lines.append(f"{k}: {m[k]}")
            ds = m.get("data_sources", {})
            if "duckdb" in ds:
                lines.append(f"duckdb: {ds['duckdb']}")
        except Exception:
            lines.append(f"manifest: {manifest_path} (unreadable)")
    return "\n".join(lines)


def load_equity_csv(path: str, ts_col: str = "ts", equity_col: str = "portfolio_equity") -> pd.Series:
    """Load equity CSV and return a Series indexed by ts named 'equity'."""
    df = pd.read_csv(path)
    if ts_col not in df.columns:
        raise ValueError(f"Equity CSV missing '{ts_col}' column: {path}")
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Equity CSV has invalid timestamps: {path}")
    df[ts_col] = _normalize_datetime_series(ts)
    candidates = [c for c in [equity_col, "equity"] if c in df.columns]
    if not candidates:
        value_cols = [c for c in df.columns if c != ts_col]
        if not value_cols:
            raise ValueError(f"No equity column found in {path}")
        equity_col = value_cols[0]
    else:
        equity_col = candidates[0]
    ser = df.set_index(ts_col)[equity_col].rename("equity")
    return ser.sort_index()


def compute_drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def compute_stats(eq: pd.Series, rf_annual: float = 0.0) -> Dict[str, float]:
    ret = eq.pct_change().dropna()
    n_days = len(eq)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0 if n_days > 0 else np.nan
    cagr = (1 + total_return) ** (ANN_FACTOR / n_days) - 1.0 if n_days > 0 else np.nan
    vol = ret.std() * np.sqrt(ANN_FACTOR) if not ret.empty else np.nan
    sharpe = (cagr - rf_annual) / vol if vol and not np.isnan(vol) else np.nan
    max_dd = compute_drawdown(eq).min() if n_days > 0 else np.nan
    return {
        "n_days": n_days,
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def load_benchmark_equity(path: str, strategy_index: pd.DatetimeIndex) -> pd.Series:
    """Load benchmark CSV (ts, equity), align to strategy_index, ffill/bfill, normalize to 1.0."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError(f"Benchmark CSV missing ts column: {path}")
    if "equity" not in df.columns:
        value_cols = [c for c in df.columns if c != "ts"]
        if value_cols:
            df = df.rename(columns={value_cols[0]: "equity"})
        else:
            raise ValueError(f"Benchmark CSV missing equity column: {path}")
    bench_ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if bench_ts.isna().any():
        raise ValueError(f"Benchmark CSV has invalid timestamps: {path}")
    bench = df.assign(ts=_normalize_datetime_series(bench_ts)).set_index("ts")["equity"].astype(float).sort_index()
    bench = _align_benchmark_series(bench, strategy_index)
    bench = bench / bench.iloc[0]
    bench.name = "benchmark_equity"
    return bench


def _normalize_datetime_series(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series, utc=True, errors="coerce")
    if series.isna().any():
        raise ValueError("Series contains invalid timestamps.")
    series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    if (series.dt.hour == 0).all() and (series.dt.minute == 0).all() and (series.dt.second == 0).all():
        series = series.dt.normalize()
    return series


def _normalize_datetime_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index, utc=True, errors="coerce")
    else:
        index = pd.to_datetime(index, utc=True, errors="coerce")
    if index.isna().any():
        raise ValueError("Index contains invalid timestamps.")
    index = index.tz_convert("UTC").tz_localize(None)
    if (index.hour == 0).all() and (index.minute == 0).all() and (index.second == 0).all():
        index = index.normalize()
    return index


def _align_benchmark_series(
    bench: pd.Series,
    strategy_index: pd.DatetimeIndex,
) -> pd.Series:
    strategy_index_norm = _normalize_datetime_index(strategy_index)
    bench_index_norm = _normalize_datetime_index(bench.index)
    bench = bench.copy()
    bench.index = bench_index_norm
    aligned = bench.reindex(strategy_index_norm).ffill().bfill()
    if aligned.isna().all() or np.isclose(aligned.fillna(0.0).values, 0.0).all():
        bench_info = {
            "bench_min": bench_index_norm.min(),
            "bench_max": bench_index_norm.max(),
            "bench_count": len(bench_index_norm),
            "bench_head": bench_index_norm[:3].tolist(),
            "bench_tail": bench_index_norm[-3:].tolist(),
            "strategy_min": strategy_index_norm.min(),
            "strategy_max": strategy_index_norm.max(),
            "strategy_count": len(strategy_index_norm),
            "strategy_head": strategy_index_norm[:3].tolist(),
            "strategy_tail": strategy_index_norm[-3:].tolist(),
        }
        raise ValueError(
            "Benchmark alignment yielded empty or zero series. "
            f"strategy_index dtype={strategy_index.dtype} tz={strategy_index.tz} "
            f"benchmark_index dtype={bench_index_norm.dtype} tz=None "
            f"details={bench_info}"
        )
    return aligned


def get_default_benchmark_equity(
    strategy_index: pd.DatetimeIndex,
    research_dir: Optional[str] = None,
    benchmark_equity_csv: Optional[str] = None,
    benchmark_label: Optional[str] = None,
    default_symbol: str = "BTC-USD",
) -> tuple[Optional[pd.Series], str]:
    label = benchmark_label or "BTC Buy & Hold"
    strategy_index = _normalize_datetime_index(strategy_index)
    if benchmark_equity_csv:
        return load_benchmark_equity(benchmark_equity_csv, strategy_index), label

    cache_path = Path("artifacts/research/benchmarks/btc_usd_buy_and_hold_equity.csv")
    if cache_path.exists():
        return load_benchmark_equity(str(cache_path), strategy_index), label

    if not research_dir:
        raise FileNotFoundError(
            "No benchmark CSV provided and no research_dir supplied to locate manifest.json."
        )

    manifest_path = Path(research_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Benchmark cache missing and manifest.json not found: {manifest_path}"
        )

    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "duckdb is required to auto-generate benchmark equity. "
            "Install duckdb or provide --benchmark_equity_csv."
        ) from exc

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("params", {})
    data_cfg = params.get("data", {})
    db_path = data_cfg.get("db_path")
    table = data_cfg.get("table")
    time_range = manifest.get("time_range", {})
    start = time_range.get("start")
    end = time_range.get("end")
    if not db_path or not table or not start or not end:
        raise ValueError("manifest.json missing db_path/table/start/end for benchmark generation.")

    con = duckdb.connect(db_path)
    query = f"""
        SELECT ts, close
        FROM {table}
        WHERE symbol = ? AND ts >= ? AND ts <= ?
        ORDER BY ts
    """
    df = con.execute(query, [default_symbol, start, end]).fetchdf()
    if df.empty:
        raise ValueError("Benchmark query returned no rows.")
    df["ts"] = _normalize_datetime_series(pd.to_datetime(df["ts"], utc=True, errors="coerce"))
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df["equity"] = df["close"] / df["close"].iloc[0]
    bench = df.set_index("ts")["equity"].astype(float)
    bench = _align_benchmark_series(bench, strategy_index)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df[["ts", "equity"]].to_csv(cache_path, index=False)
    bench.name = "benchmark_equity"
    return bench, label


def scale_equity_to_start(series: pd.Series, target_start: float) -> pd.Series:
    """Scale a normalized equity series to start at target_start, preserving relative movements."""
    if series.empty:
        raise ValueError("Cannot scale empty series.")
    first_val = float(series.iloc[0])
    if first_val == 0.0 or np.isnan(first_val):
        raise ValueError(f"Cannot scale series with first value {first_val}")
    scaled = series * (target_start / first_val)
    return scaled


def plot_equity_with_benchmark(
    ax: plt.Axes,
    strat_eq: pd.Series,
    bench_eq: Optional[pd.Series],
    strat_label: str,
    bench_label: str,
) -> None:
    ax.plot(strat_eq.index, strat_eq.values, label=strat_label)
    if bench_eq is not None:
        ax.plot(bench_eq.index, bench_eq.values, label=bench_label, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_drawdown_with_benchmark(
    ax: plt.Axes,
    strat_eq: pd.Series,
    bench_eq: Optional[pd.Series],
    strat_label: str,
    bench_label: str,
) -> None:
    strat_dd = compute_drawdown(strat_eq)
    ax.plot(strat_dd.index, strat_dd.values, label=f"{strat_label} drawdown")
    if bench_eq is not None:
        bench_dd = compute_drawdown(bench_eq)
        ax.plot(bench_dd.index, bench_dd.values, label=f"{bench_label} drawdown", linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)


def add_benchmark_summary_table(
    fig: plt.Figure,
    comparison_df: pd.DataFrame,
    anchor: Tuple[float, float] = (0.7, 0.05),
) -> None:
    if comparison_df is None or comparison_df.empty:
        return
    cell_text = comparison_df.values.tolist()
    col_labels = comparison_df.columns.tolist()
    table = plt.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[anchor[0], anchor[1], 0.28, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)


def load_strategy_stats_from_metrics(metrics_csv: str) -> Dict[str, float]:
    """
    Load strategy performance stats from metrics CSV.
    
    Handles two cases:
    1. Multi-period CSV with 'period' column: selects row where period='full'
    2. Single-row CSV without 'period': injects period='full' and uses that row
    """
    df = pd.read_csv(metrics_csv)
    
    if "period" not in df.columns:
        # MA baseline and other single-period strategies don't have a period column
        if len(df) != 1:
            raise ValueError(
                f"metrics CSV missing 'period' column and has {len(df)} rows: {metrics_csv}"
            )
        df = df.copy()
        df["period"] = "full"
    
    row = df.loc[df["period"] == "full"].iloc[0]
    return {
        "start": pd.to_datetime(row.get("start")) if "start" in row else None,
        "end": pd.to_datetime(row.get("end")) if "end" in row else None,
        "n_days": float(row.get("n_days", np.nan)),
        "cagr": float(row["cagr"]),
        "vol": float(row["vol"]),
        "sharpe": float(row["sharpe"]),
        "sortino": float(row.get("sortino", np.nan)),
        "calmar": float(row.get("calmar", np.nan)),
        "avg_dd": float(row.get("avg_dd", np.nan)),
        "hit_ratio": float(row.get("hit_ratio", np.nan)),
        "expectancy": float(row.get("expectancy", np.nan)),
        "max_dd": float(row["max_dd"]),
    }


def build_benchmark_comparison_table(
    strategy_label: str,
    strategy_stats: Dict[str, float],
    benchmark_label: Optional[str] = None,
    benchmark_eq: Optional[pd.Series] = None,
    risk_free: float = 0.0,
) -> pd.DataFrame:
    bench_stats: Dict[str, float] = {}
    if benchmark_eq is not None:
        bench_stats = compute_stats(benchmark_eq, rf_annual=risk_free)
        bench_stats["avg_dd"] = compute_drawdown(benchmark_eq).loc[lambda x: x < 0].mean()
        bench_stats["hit_ratio"] = float((benchmark_eq.pct_change().dropna() > 0).mean())
        br = benchmark_eq.pct_change().dropna()
        wins = br[br > 0]
        losses = br[br < 0]
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        p_win = bench_stats["hit_ratio"] if not np.isnan(bench_stats["hit_ratio"]) else 0.0
        p_loss = 1.0 - p_win
        bench_stats["expectancy"] = float(p_win * avg_win + p_loss * avg_loss)

    def fmt_pct(x: float) -> str:
        return f"{x:.2%}" if pd.notna(x) else ""

    def fmt_ratio(x: float) -> str:
        return f"{x:.2f}" if pd.notna(x) else ""

    def fmt_hit(x: float) -> str:
        return f"{x*100:.1f}%" if pd.notna(x) else ""

    def fmt_expect(x: float) -> str:
        return f"{x*100:.2f}%" if pd.notna(x) else ""

    rows = [
        [
            "CAGR",
            fmt_pct(strategy_stats.get("cagr")),
            fmt_pct(bench_stats.get("cagr")) if benchmark_label else "",
        ],
        [
            "Vol",
            fmt_pct(strategy_stats.get("vol")),
            fmt_pct(bench_stats.get("vol")) if benchmark_label else "",
        ],
        [
            "Sharpe",
            fmt_ratio(strategy_stats.get("sharpe")),
            fmt_ratio(bench_stats.get("sharpe")) if benchmark_label else "",
        ],
        [
            "Sortino",
            fmt_ratio(strategy_stats.get("sortino")),
            fmt_ratio(bench_stats.get("sortino")) if benchmark_label else "",
        ],
        [
            "Calmar",
            fmt_ratio(strategy_stats.get("calmar")),
            fmt_ratio(bench_stats.get("calmar")) if benchmark_label else "",
        ],
        [
            "Avg DD",
            fmt_pct(strategy_stats.get("avg_dd")),
            fmt_pct(bench_stats.get("avg_dd")) if benchmark_label else "",
        ],
        [
            "Hit %",
            fmt_hit(strategy_stats.get("hit_ratio")),
            fmt_hit(bench_stats.get("hit_ratio")) if benchmark_label else "",
        ],
        [
            "Exp %",
            fmt_expect(strategy_stats.get("expectancy")),
            fmt_expect(bench_stats.get("expectancy")) if benchmark_label else "",
        ],
        [
            "MaxDD",
            fmt_pct(strategy_stats.get("max_dd")),
            fmt_pct(bench_stats.get("max_dd")) if benchmark_label else "",
        ],
    ]
    if not benchmark_label:
        rows = [r[:2] for r in rows]
    cols = ["Metric", strategy_label] + ([benchmark_label] if benchmark_label else [])
    return pd.DataFrame(rows, columns=cols)


def _rolling_sharpe(returns: pd.Series, window: int = 90, ann_factor: float = 365.0) -> pd.Series:
    roll = returns.rolling(window=window)
    mean = roll.mean()
    std = roll.std()
    sharpe = (mean * ann_factor) / (std * np.sqrt(ann_factor))
    return sharpe


def build_standard_tearsheet(
    *,
    out_pdf: str | Path,
    strategy_label: str,
    strategy_equity: pd.Series,
    strategy_stats: Dict[str, float],
    benchmark_equity: Optional[pd.Series] = None,
    benchmark_label: Optional[str] = None,
    equity_csv_path: str,
    metrics_csv_path: str,
    manifest_path: Optional[str] = None,
    subtitle: Optional[str] = None,
    turnover: Optional[pd.Series] = None,
    extra_summary_lines: Optional[List[str]] = None,
) -> None:
    """
    Build the canonical multi-page tear sheet used by all strategies.
    
    Core pages (always included):
    1. Title + performance summary + benchmark comparison table
    2. Equity curve + drawdown (with benchmark overlay if present)
    3. Rolling risk (Sharpe, Vol) + turnover (if provided)
    4. Return distribution histogram
    5. Provenance
    
    Args:
        out_pdf: Output PDF path
        strategy_label: Name for the strategy (e.g., "MA(5/40) baseline")
        strategy_equity: Strategy equity series (indexed by datetime)
        strategy_stats: Dict with keys: cagr, vol, sharpe, sortino, calmar, max_dd, avg_dd, hit_ratio, expectancy, n_days, start, end
        benchmark_equity: Optional benchmark equity series (normalized, same index as strategy)
        benchmark_label: Label for benchmark (e.g., "BTC Buy & Hold")
        equity_csv_path: Path to equity CSV for provenance
        metrics_csv_path: Path to metrics CSV for provenance
        manifest_path: Optional path to run manifest for provenance
        subtitle: Optional subtitle for title page
        turnover: Optional daily turnover series
        extra_summary_lines: Optional additional summary lines for title page
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    
    strat_eq = strategy_equity
    strat_ret = strat_eq.pct_change().dropna()
    
    # Scale benchmark for plotting if present
    bench_eq_plot = None
    if benchmark_equity is not None:
        bench_eq_plot = scale_equity_to_start(benchmark_equity, float(strat_eq.iloc[0]))
    
    # Benchmark comparison table
    comparison_df = build_benchmark_comparison_table(
        strategy_label=strategy_label,
        strategy_stats=strategy_stats,
        benchmark_label=benchmark_label if benchmark_equity is not None else None,
        benchmark_eq=benchmark_equity,
    )
    
    with PdfPages(out_pdf) as pdf:
        # ========== Page 1: Title + Summary + Benchmark Comparison ==========
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        
        title_text = strategy_label
        subtitle_text = subtitle or "Single-asset trend baseline"
        
        summary_lines = [
            f"Sample: {strategy_stats.get('start')} – {strategy_stats.get('end')}",
            f"CAGR: {strategy_stats.get('cagr'):.2%}",
            f"Vol (ann.): {strategy_stats.get('vol'):.2%}",
            f"Sharpe: {strategy_stats.get('sharpe'):.2f}",
            f"Sortino: {strategy_stats.get('sortino', float('nan')):.2f}",
            f"Calmar: {strategy_stats.get('calmar', float('nan')):.2f}",
            f"Max drawdown: {strategy_stats.get('max_dd'):.2%}",
            f"Avg drawdown: {strategy_stats.get('avg_dd', float('nan')):.2%}",
            f"Hit ratio: {strategy_stats.get('hit_ratio', float('nan'))*100:.1f}%",
            f"Expectancy: {strategy_stats.get('expectancy', float('nan'))*100:.2f}%",
            f"Sample length: {int(strategy_stats.get('n_days', 0))} days",
        ]
        
        if extra_summary_lines:
            summary_lines.extend(extra_summary_lines)
        
        text_y = 0.9
        ax.text(0.02, text_y, title_text, fontsize=18, fontweight="bold", transform=ax.transAxes)
        text_y -= 0.05
        ax.text(0.02, text_y, subtitle_text, fontsize=11, transform=ax.transAxes)
        
        text_y -= 0.08
        ax.text(
            0.02,
            text_y,
            "Performance summary:",
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )
        text_y -= 0.04
        for line in summary_lines:
            ax.text(0.04, text_y, f"• {line}", fontsize=11, transform=ax.transAxes)
            text_y -= 0.035
        
        # Add benchmark comparison table (right side)
        if comparison_df is not None and not comparison_df.empty:
            add_benchmark_summary_table(fig, comparison_df, anchor=(0.65, 0.02))
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # ========== Page 2: Equity & Drawdown ==========
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        
        plot_equity_with_benchmark(
            axes[0],
            strat_eq if bench_eq_plot is None else strat_eq,
            bench_eq_plot,
            strat_label=strategy_label,
            bench_label=benchmark_label or "Benchmark",
        )
        axes[0].set_ylabel("Equity (nav)")
        axes[0].set_title("Equity curve")
        
        plot_drawdown_with_benchmark(
            axes[1],
            strat_eq,
            benchmark_equity,  # Use normalized for drawdown
            strat_label=strategy_label,
            bench_label=benchmark_label or "Benchmark",
        )
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Date")
        axes[1].set_title("Drawdown (from peak)")
        
        for ax_ in axes:
            ax_.grid(True, alpha=0.3)
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # ========== Page 3: Rolling Risk + Turnover ==========
        num_subplots = 3 if turnover is not None else 2
        fig, axes = plt.subplots(num_subplots, 1, figsize=(11, 8.5), sharex=True)
        
        # Rolling Sharpe
        roll_sharpe = _rolling_sharpe(strat_ret, window=90)
        axes[0].plot(roll_sharpe.index, roll_sharpe.values, label=strategy_label)
        if benchmark_equity is not None:
            bench_ret = benchmark_equity.pct_change().dropna()
            bench_roll_sharpe = _rolling_sharpe(bench_ret, window=90)
            axes[0].plot(bench_roll_sharpe.index, bench_roll_sharpe.values, label=benchmark_label, linestyle="--")
        axes[0].legend()
        axes[0].set_ylabel("Sharpe (90d)")
        axes[0].set_title("Rolling 90-day Sharpe")
        axes[0].grid(True, alpha=0.3)
        
        # Rolling Vol
        roll_vol = strat_ret.rolling(window=90).std() * np.sqrt(365.0)
        axes[1].plot(roll_vol.index, roll_vol.values, label=strategy_label)
        if benchmark_equity is not None:
            bench_roll_vol = bench_ret.rolling(window=90).std() * np.sqrt(365.0)
            axes[1].plot(bench_roll_vol.index, bench_roll_vol.values, label=benchmark_label, linestyle="--")
        axes[1].legend()
        axes[1].set_ylabel("Ann. vol (90d)")
        axes[1].set_title("Rolling 90-day annualized volatility")
        axes[1].grid(True, alpha=0.3)
        
        # Turnover (optional)
        if turnover is not None:
            axes[2].plot(turnover.index, turnover.values)
            axes[2].set_ylabel("Turnover")
            axes[2].set_xlabel("Date")
            axes[2].set_title("Daily turnover")
            axes[2].grid(True, alpha=0.3)
        else:
            axes[1].set_xlabel("Date")
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # ========== Page 4: Return Distribution ==========
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.hist(strat_ret, bins=50, alpha=0.8, label=strategy_label)
        ax.set_xlabel("Daily return")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of daily returns")
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)
        
        # ========== Final Page: Provenance ==========
        prov_text = build_provenance_text(equity_csv_path, metrics_csv_path, manifest_path)
        fig_p, ax_p = plt.subplots(figsize=(11, 8.5))
        ax_p.axis("off")
        ax_p.text(0.02, 0.98, prov_text, ha="left", va="top", fontsize=10)
        pdf.savefig(fig_p, bbox_inches="tight")
        plt.close(fig_p)


# ============================================================
# HTML Tearsheet  (L2 institutional style)
# ============================================================

_HTML_TEARSHEET_CSS = """
:root{--bg:#0f172a;--bg-card:#1e293b;--bg-alt:#162032;--tx:#f1f5f9;--tx2:#94a3b8;--tx3:#64748b;
--accent:#3b82f6;--green:#22c55e;--red:#ef4444;--border:#334155}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif}
.page{max-width:1200px;margin:0 auto;padding:32px 24px}
.hdr{text-align:center;margin-bottom:28px}
.hdr h1{font-size:26px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px}
.hdr .sub{font-size:15px;color:var(--tx2);margin-bottom:4px}
.hdr .dates{font-size:13px;color:var(--tx3)}
.hero{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:28px}
.hero .cd{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:18px 12px;text-align:center}
.hero .cd .lb{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--tx3);margin-bottom:6px}
.hero .cd .vl{font-size:22px;font-weight:700}
.pos{color:var(--green)}.neg{color:var(--red)}
.sec{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;margin-bottom:22px;overflow:hidden}
.sec .st{font-size:13px;text-transform:uppercase;letter-spacing:1.5px;padding:14px 22px;border-bottom:1px solid var(--border);color:var(--tx2);font-weight:600}
.sec .sb{padding:22px}
.sg{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}
.sg h3{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--accent);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.sr{display:flex;justify-content:space-between;padding:5px 0;font-size:13px;border-bottom:1px solid rgba(51,65,85,.3)}
.sr .sl{color:var(--tx2)}.sr .sv{font-weight:600;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:var(--bg-alt);color:var(--tx2);font-weight:600;text-transform:uppercase;letter-spacing:.5px;font-size:11px;padding:10px 12px;text-align:left;border-bottom:1px solid var(--border)}
td{padding:8px 12px;border-bottom:1px solid rgba(51,65,85,.3)}
tr:hover td{background:rgba(59,130,246,.04)}
.tg{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:20px}
.tc{background:var(--bg-alt);border-radius:6px;padding:14px;text-align:center}
.tc .lb{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--tx3);margin-bottom:4px}
.tc .vl{font-size:17px;font-weight:700}
.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.hl{list-style:disc;padding-left:20px}.hl li{color:var(--tx2);margin-bottom:8px;font-size:14px}
.prov{font-family:'SF Mono','Fira Code',monospace;font-size:12px;color:var(--tx3);line-height:1.8}
.overview{color:var(--tx2);font-size:14px;line-height:1.6;margin-bottom:16px}
.ft{text-align:center;padding:20px;color:var(--tx3);font-size:11px;letter-spacing:.5px;border-top:1px solid var(--border);margin-top:16px}
.sh3{font-size:14px;color:var(--tx2);margin:20px 0 10px;text-transform:uppercase;letter-spacing:1px}
@media(max-width:768px){.hero{grid-template-columns:repeat(3,1fr)}.sg{grid-template-columns:repeat(2,1fr)}.tg{grid-template-columns:repeat(3,1fr)}.two{grid-template-columns:1fr}}
@media(max-width:480px){.hero{grid-template-columns:repeat(2,1fr)}.sg{grid-template-columns:1fr}.tg{grid-template-columns:repeat(2,1fr)}}
"""


def _top_n_drawdown_periods(equity: pd.Series, n: int = 5) -> List[Dict[str, Any]]:
    """Find the worst *n* drawdown periods: start (peak), valley, end, drawdown, duration."""
    dd = compute_drawdown(equity)
    periods: List[Dict[str, Any]] = []
    in_dd = False
    peak_dt = dd.index[0]
    valley_dt = dd.index[0]
    valley_dd = 0.0
    prev_dt = dd.index[0]

    for dt, val in dd.items():
        if val < 0:
            if not in_dd:
                in_dd = True
                peak_dt = prev_dt
                valley_dt, valley_dd = dt, val
            elif val < valley_dd:
                valley_dt, valley_dd = dt, val
        else:
            if in_dd:
                periods.append(dict(start=peak_dt, valley=valley_dt, end=dt,
                                    drawdown=valley_dd, duration_days=(dt - peak_dt).days))
                in_dd = False
        prev_dt = dt

    if in_dd:
        periods.append(dict(start=peak_dt, valley=valley_dt, end=dd.index[-1],
                            drawdown=valley_dd, duration_days=(dd.index[-1] - peak_dt).days))

    periods.sort(key=lambda p: p["drawdown"])
    return periods[:n]


def compute_comprehensive_stats(
    equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
    turnover: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Compute the full L2-style metric dictionary from an equity curve."""
    ret = equity.pct_change().dropna()
    n_days = len(equity)
    if n_days < 2:
        return {"n_days": n_days, "error": "insufficient data"}

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((1 + total_return) ** (ANN_FACTOR / n_days) - 1.0)

    # Best / worst periods
    yr = (1 + ret).groupby(ret.index.year).prod() - 1.0
    mo = (1 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1.0
    best_year, worst_year = (float(yr.max()), float(yr.min())) if not yr.empty else (np.nan, np.nan)
    best_month, worst_month = (float(mo.max()), float(mo.min())) if not mo.empty else (np.nan, np.nan)
    best_day, worst_day = float(ret.max()), float(ret.min())

    # Risk
    vol = float(ret.std() * np.sqrt(ANN_FACTOR))
    neg = ret[ret < 0]
    downside_vol = float(neg.std() * np.sqrt(ANN_FACTOR)) if len(neg) > 1 else 0.0
    dd = compute_drawdown(equity)
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    dd_periods = _top_n_drawdown_periods(equity, n=5)
    max_dd_duration = dd_periods[0]["duration_days"] if dd_periods else 0

    sharpe = float((cagr - rf_annual) / vol) if vol > 1e-10 else np.nan
    sortino = float((cagr - rf_annual) / downside_vol) if downside_vol > 1e-10 else np.nan
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-10 else np.nan

    # Risk-adjusted
    excess = ret - 0.0
    pos_sum = float(excess[excess > 0].sum())
    neg_sum = float(abs(excess[excess <= 0].sum()))
    omega = pos_sum / neg_sum if neg_sum > 1e-10 else np.nan

    log_eq = np.log(equity.values.astype(float))
    x = np.arange(len(log_eq))
    if len(log_eq) > 1:
        coeffs = np.polyfit(x, log_eq, 1)
        ss_res = float(np.sum((log_eq - np.polyval(coeffs, x)) ** 2))
        ss_tot = float(np.sum((log_eq - log_eq.mean()) ** 2))
        stability = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan
    else:
        stability = np.nan

    p95, p05 = float(np.percentile(ret, 95)), float(np.percentile(ret, 5))
    tail_ratio = abs(p95 / p05) if abs(p05) > 1e-10 else np.nan
    var_95 = p05
    cvar_mask = ret <= var_95
    cvar_95 = float(ret[cvar_mask].mean()) if cvar_mask.any() else var_95
    skewness = float(ret.skew())
    kurtosis = float(ret.kurtosis())

    # Trade stats
    win_rate = float((ret > 0).mean())
    wins, losses = ret[ret > 0], ret[ret < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 1e-10 else np.nan
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-10 else np.nan
    expectancy = float(win_rate * avg_win + (1 - win_rate) * avg_loss)
    autocorrelation = float(ret.autocorr(lag=1)) if len(ret) > 1 else np.nan
    turnover_proxy = float(turnover.mean()) if turnover is not None and not turnover.empty else float(ret.abs().mean())

    # Factor analysis (requires benchmark)
    alpha_val = beta_val = correlation = information_ratio = np.nan
    bench_stats_dict: Optional[Dict[str, Any]] = None

    if benchmark_equity is not None:
        br = benchmark_equity.pct_change().dropna()
        common = ret.index.intersection(br.index)
        if len(common) > 1:
            sr_c, br_c = ret.loc[common], br.loc[common]
            cov = np.cov(sr_c.values, br_c.values)
            beta_val = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 1e-10 else np.nan
            alpha_val = float((sr_c.mean() - beta_val * br_c.mean()) * ANN_FACTOR) if not np.isnan(beta_val) else np.nan
            correlation = float(sr_c.corr(br_c))
            ex = sr_c - br_c
            information_ratio = float(ex.mean() / ex.std() * np.sqrt(ANN_FACTOR)) if ex.std() > 1e-10 else np.nan

        bn = len(benchmark_equity)
        bt = float(benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1.0) if bn > 0 else np.nan
        bcagr = float((1 + bt) ** (ANN_FACTOR / bn) - 1.0) if bn > 0 else np.nan
        bvol = float(br.std() * np.sqrt(ANN_FACTOR)) if not br.empty else np.nan
        bsharpe = float((bcagr - rf_annual) / bvol) if bvol and bvol > 1e-10 else np.nan
        bneg = br[br < 0]
        bdsvol = float(bneg.std() * np.sqrt(ANN_FACTOR)) if len(bneg) > 1 else 0.0
        bsortino = float((bcagr - rf_annual) / bdsvol) if bdsvol > 1e-10 else np.nan
        bdd = compute_drawdown(benchmark_equity)
        bench_stats_dict = dict(cagr=bcagr, vol=bvol, sharpe=bsharpe, sortino=bsortino,
                                max_dd=float(bdd.min()), win_rate=float((br > 0).mean()))

    return dict(
        total_return=total_return, cagr=cagr,
        best_year=best_year, worst_year=worst_year,
        best_month=best_month, worst_month=worst_month,
        best_day=best_day, worst_day=worst_day,
        vol=vol, downside_vol=downside_vol,
        max_dd=max_dd, avg_dd=avg_dd, max_dd_duration=max_dd_duration,
        sharpe=sharpe, sortino=sortino, calmar=calmar,
        omega=omega, stability=stability, tail_ratio=tail_ratio,
        var_95=var_95, cvar_95=cvar_95, skewness=skewness, kurtosis=kurtosis,
        alpha=alpha_val, beta=beta_val, correlation=correlation,
        information_ratio=information_ratio, win_rate=win_rate,
        profit_factor=profit_factor, win_loss_ratio=win_loss_ratio,
        expectancy=expectancy, autocorrelation=autocorrelation,
        turnover_proxy=turnover_proxy,
        n_days=n_days, start=equity.index[0], end=equity.index[-1],
        drawdown_periods=dd_periods, benchmark_stats=bench_stats_dict,
    )


def build_standard_html_tearsheet(
    *,
    out_html: str | Path,
    strategy_label: str,
    strategy_equity: pd.Series,
    strategy_stats: Optional[Dict[str, Any]] = None,
    benchmark_equity: Optional[pd.Series] = None,
    benchmark_label: Optional[str] = None,
    equity_csv_path: str = "",
    metrics_csv_path: str = "",
    manifest_path: Optional[str] = None,
    subtitle: Optional[str] = None,
    turnover: Optional[pd.Series] = None,
    auto_open: bool = False,
    confidential_footer: bool = True,
) -> Path:
    """Build a comprehensive, self-contained HTML tearsheet (L2 institutional style).

    Uses Plotly via CDN for interactive charts.  No ``plotly`` Python package required.
    """
    import html as _h
    from datetime import datetime, timezone

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = strategy_stats or compute_comprehensive_stats(
        equity=strategy_equity, benchmark_equity=benchmark_equity, turnover=turnover,
    )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _dates(idx):
        return [d.strftime("%Y-%m-%d") for d in idx]

    def _vals(arr, dec=4):
        out = []
        for v in arr:
            fv = float(v)
            out.append(None if (np.isnan(fv) or np.isinf(fv)) else round(fv, dec))
        return out

    def fpct(v, d=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "\u2014"
        return f"{v:.{d}%}"

    def frat(v, d=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "\u2014"
        return f"{v:.{d}f}"

    esc = _h.escape

    # ------------------------------------------------------------------
    # chart data
    # ------------------------------------------------------------------
    strat_eq = strategy_equity
    strat_ret = strat_eq.pct_change().dropna()
    dd = compute_drawdown(strat_eq)
    eq_dates = _dates(strat_eq.index)
    eq_vals = _vals(strat_eq.values)
    dd_vals = _vals(dd.values)

    has_bench = benchmark_equity is not None
    blabel = benchmark_label or "BTC B&H"

    if has_bench:
        b_eq_s = scale_equity_to_start(benchmark_equity, float(strat_eq.iloc[0]))
        b_eq_dates = _dates(b_eq_s.index)
        b_eq_vals = _vals(b_eq_s.values)
        b_dd = compute_drawdown(benchmark_equity)
        b_dd_dates = _dates(b_dd.index)
        b_dd_vals = _vals(b_dd.values)
        b_ret = benchmark_equity.pct_change().dropna()

    # rolling series
    rs = _rolling_sharpe(strat_ret, window=90)
    rv = strat_ret.rolling(90).std() * np.sqrt(365.0)
    rs_d, rs_v = _dates(rs.index), _vals(rs.values, 2)
    rv_d, rv_v = _dates(rv.index), _vals(rv.values)

    if has_bench:
        brs = _rolling_sharpe(b_ret, window=90)
        brv = b_ret.rolling(90).std() * np.sqrt(365.0)
        brs_d, brs_v = _dates(brs.index), _vals(brs.values, 2)
        brv_d, brv_v = _dates(brv.index), _vals(brv.values)
        ci = strat_ret.index.intersection(b_ret.index)
        if len(ci) > 90:
            sr_a, br_a = strat_ret.loc[ci], b_ret.loc[ci]
            rb = sr_a.rolling(90).cov(br_a) / br_a.rolling(90).var()
            rb_d, rb_v = _dates(rb.index), _vals(rb.values, 3)
        else:
            rb_d, rb_v = [], []
    else:
        rb_d, rb_v = [], []

    # monthly heatmap
    mo_ret = (1 + strat_ret).groupby([strat_ret.index.year, strat_ret.index.month]).prod() - 1.0
    mo_years = sorted(set(strat_ret.index.year))
    mo_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    z_data, z_text = [], []
    for y in mo_years:
        row, trow = [], []
        for m in range(1, 13):
            if (y, m) in mo_ret.index:
                v = round(float(mo_ret.loc[(y, m)]) * 100, 2)
                row.append(v)
                trow.append(f"{v:+.1f}%")
            else:
                row.append(None)
                trow.append("")
        z_data.append(row)
        z_text.append(trow)

    # yearly returns
    yr_ret = (1 + strat_ret).groupby(strat_ret.index.year).prod() - 1.0
    yr_years = [str(int(y)) for y in yr_ret.index]
    yr_vals = [round(float(v) * 100, 2) for v in yr_ret.values]
    byr_years, byr_vals = [], []
    if has_bench:
        byr = (1 + b_ret).groupby(b_ret.index.year).prod() - 1.0
        byr_years = [str(int(y)) for y in byr.index]
        byr_vals = [round(float(v) * 100, 2) for v in byr.values]

    ret_hist = [round(float(v), 6) for v in strat_ret.values]

    # ------------------------------------------------------------------
    # Plotly chart configs  (list of (div_id, traces_json, layout_json))
    # ------------------------------------------------------------------
    LB = dict(paper_bgcolor="transparent", plot_bgcolor="#1e293b",
              font=dict(color="#e2e8f0", family="-apple-system,BlinkMacSystemFont,'Inter',sans-serif", size=12),
              margin=dict(l=60, r=20, t=40, b=40),
              xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
              yaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
              legend=dict(bgcolor="transparent", font=dict(size=11)),
              hovermode="x unified")

    def _jd(obj):
        return json.dumps(obj, default=str)

    charts: List[Tuple[str, str, str]] = []

    # 1 equity
    et = [dict(x=eq_dates, y=eq_vals, type="scatter", mode="lines", name=strategy_label,
               line=dict(color="#3b82f6", width=1.5))]
    if has_bench:
        et.append(dict(x=b_eq_dates, y=b_eq_vals, type="scatter", mode="lines", name=blabel,
                        line=dict(color="#94a3b8", width=1.2, dash="dash")))
    el = {**LB, "title": dict(text="Equity Curve", font=dict(size=14)),
          "yaxis": {**LB["yaxis"], "title": "Equity"}, "height": 350}
    charts.append(("eq-chart", _jd(et), _jd(el)))

    # 2 drawdown
    dt_ = [dict(x=eq_dates, y=dd_vals, type="scatter", mode="lines", name=strategy_label,
                fill="tozeroy", fillcolor="rgba(220,38,38,.15)", line=dict(color="#dc2626", width=1.2))]
    if has_bench:
        dt_.append(dict(x=b_dd_dates, y=b_dd_vals, type="scatter", mode="lines", name=blabel,
                        line=dict(color="#94a3b8", width=1, dash="dash")))
    dl = {**LB, "title": dict(text="Drawdown", font=dict(size=14)),
          "yaxis": {**LB["yaxis"], "title": "Drawdown", "tickformat": ".0%"}, "height": 280}
    charts.append(("dd-chart", _jd(dt_), _jd(dl)))

    # 3 rolling sharpe
    rst = [dict(x=rs_d, y=rs_v, type="scatter", mode="lines", name=strategy_label,
                line=dict(color="#3b82f6", width=1.2))]
    if has_bench:
        rst.append(dict(x=brs_d, y=brs_v, type="scatter", mode="lines", name=blabel,
                        line=dict(color="#94a3b8", width=1, dash="dash")))
    rsl = {**LB, "title": dict(text="Rolling 90-Day Sharpe", font=dict(size=14)),
           "yaxis": {**LB["yaxis"], "title": "Sharpe"}, "height": 280}
    charts.append(("rs-chart", _jd(rst), _jd(rsl)))

    # 4 rolling vol
    rvt = [dict(x=rv_d, y=rv_v, type="scatter", mode="lines", name=strategy_label,
                line=dict(color="#3b82f6", width=1.2))]
    if has_bench:
        rvt.append(dict(x=brv_d, y=brv_v, type="scatter", mode="lines", name=blabel,
                        line=dict(color="#94a3b8", width=1, dash="dash")))
    rvl = {**LB, "title": dict(text="Rolling 90-Day Volatility", font=dict(size=14)),
           "yaxis": {**LB["yaxis"], "title": "Ann. Vol", "tickformat": ".0%"}, "height": 280}
    charts.append(("rv-chart", _jd(rvt), _jd(rvl)))

    # 5 rolling beta (optional)
    if has_bench and rb_d:
        rbt = [dict(x=rb_d, y=rb_v, type="scatter", mode="lines", name="Beta",
                    line=dict(color="#a78bfa", width=1.2))]
        rbl = {**LB, "title": dict(text="Rolling 90-Day Beta", font=dict(size=14)),
               "yaxis": {**LB["yaxis"], "title": "Beta"}, "height": 280,
               "shapes": [dict(type="line", y0=1, y1=1, x0=rb_d[0], x1=rb_d[-1],
                               line=dict(color="#475569", width=1, dash="dot"))]}
        charts.append(("rb-chart", _jd(rbt), _jd(rbl)))

    # 6 return distribution
    ht = [dict(x=ret_hist, type="histogram", nbinsx=60, name="Daily Returns",
               marker=dict(color="#3b82f6", line=dict(color="#2563eb", width=.5)))]
    hl = {**LB, "title": dict(text="Distribution of Daily Returns", font=dict(size=14)),
          "xaxis": {**LB["xaxis"], "title": "Daily Return", "tickformat": ".1%"},
          "yaxis": {**LB["yaxis"], "title": "Count"}, "height": 300, "bargap": .05}
    charts.append(("hist-chart", _jd(ht), _jd(hl)))

    # 7 monthly heatmap
    if z_data:
        hmt = [dict(z=z_data, x=mo_names, y=[str(y) for y in mo_years], type="heatmap",
                    colorscale=[[0, "#991b1b"], [.35, "#dc2626"], [.5, "#1e293b"],
                                [.65, "#16a34a"], [1, "#15803d"]],
                    text=z_text, texttemplate="%{text}",
                    hovertemplate="%{y} %{x}: %{text}<extra></extra>",
                    showscale=True, colorbar=dict(title="%", ticksuffix="%"), zmid=0)]
        hml = {**LB, "title": dict(text="Monthly Returns (%)", font=dict(size=14)),
               "height": max(220, 40 * len(mo_years) + 80),
               "yaxis": {**LB["yaxis"], "autorange": "reversed"}}
        charts.append(("hm-chart", _jd(hmt), _jd(hml)))

    # 8 yearly bar
    ybt = [dict(x=yr_years, y=yr_vals, type="bar", name=strategy_label,
                marker=dict(color="#3b82f6"))]
    if has_bench and byr_vals:
        ybt.append(dict(x=byr_years, y=byr_vals, type="bar", name=blabel,
                        marker=dict(color="#94a3b8")))
    ybl = {**LB, "title": dict(text="Annual Returns (%)", font=dict(size=14)),
           "yaxis": {**LB["yaxis"], "title": "Return %", "ticksuffix": "%"},
           "height": 300, "barmode": "group", "bargap": .15}
    charts.append(("yr-chart", _jd(ybt), _jd(ybl)))

    # ------------------------------------------------------------------
    # HTML assembly  (list of parts, joined at end)
    # ------------------------------------------------------------------
    P: List[str] = []

    # date range
    sd = stats.get("start", strat_eq.index[0])
    ed = stats.get("end", strat_eq.index[-1])
    dr = f"{sd.strftime('%b %Y') if hasattr(sd, 'strftime') else sd} \u2013 {ed.strftime('%b %Y') if hasattr(ed, 'strftime') else ed}"

    # head
    P.append("<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>")
    P.append(f"<title>{esc(strategy_label)}</title>")
    P.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    P.append(f"<style>{_HTML_TEARSHEET_CSS}</style>")
    P.append("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>")
    P.append("</head>\n<body>\n<div class='page'>")

    # header
    P.append("<div class='hdr'>")
    P.append(f"<h1>{esc(strategy_label)}</h1>")
    if subtitle:
        P.append(f"<div class='sub'>{esc(subtitle)}</div>")
    P.append(f"<div class='dates'>{esc(dr)}</div>")
    P.append("</div>")

    # hero cards
    hero = [
        ("CAGR", fpct(stats.get("cagr")), stats.get("cagr", 0)),
        ("VOLATILITY", fpct(stats.get("vol")), None),
        ("SHARPE", frat(stats.get("sharpe")), stats.get("sharpe", 0)),
        ("SORTINO", frat(stats.get("sortino")), stats.get("sortino", 0)),
        ("MAX DD", fpct(stats.get("max_dd")), stats.get("max_dd", 0)),
        ("CALMAR", frat(stats.get("calmar")), stats.get("calmar", 0)),
    ]
    P.append("<div class='hero'>")
    for lbl, val, raw in hero:
        if raw is None:
            cls = ""
        elif isinstance(raw, float) and np.isnan(raw):
            cls = ""
        else:
            cls = "pos" if raw >= 0 else "neg"
        # MAX DD is always negative in meaning
        if lbl == "MAX DD":
            cls = "neg"
        P.append(f"<div class='cd'><div class='lb'>{lbl}</div><div class='vl {cls}'>{val}</div></div>")
    P.append("</div>")

    # ===== Section 1: Charts + Drawdown Periods =====
    P.append("<div class='sec'><div class='st'>Performance Overview</div><div class='sb'>")
    P.append("<div id='eq-chart'></div>")
    P.append("<div id='dd-chart'></div>")

    # worst drawdown periods table
    ddp = stats.get("drawdown_periods", [])
    if ddp:
        P.append("<div class='sh3'>Worst Drawdown Periods</div>")
        P.append("<table><tr><th>Start</th><th>Valley</th><th>End</th><th>Drawdown</th><th>Duration</th></tr>")
        for p in ddp:
            s = p["start"].strftime("%Y-%m-%d") if hasattr(p["start"], "strftime") else str(p["start"])
            v = p["valley"].strftime("%Y-%m-%d") if hasattr(p["valley"], "strftime") else str(p["valley"])
            e = p["end"].strftime("%Y-%m-%d") if hasattr(p["end"], "strftime") else str(p["end"])
            P.append(f"<tr><td>{s}</td><td>{v}</td><td>{e}</td><td class='neg'>{p['drawdown']:.1%}</td>"
                     f"<td>{p['duration_days']} days</td></tr>")
        P.append("</table>")
    P.append("</div></div>")

    # ===== Section 2: Performance Statistics =====
    def sr(label, value):
        return f"<div class='sr'><span class='sl'>{label}</span><span class='sv'>{value}</span></div>"

    P.append("<div class='sec'><div class='st'>Performance Statistics</div><div class='sb'>")
    P.append("<div class='sg'>")

    # Returns column
    P.append("<div><h3>Returns</h3>")
    for l, k, d in [("Total Return", "total_return", 2), ("CAGR", "cagr", 2),
                     ("Best Year", "best_year", 1), ("Worst Year", "worst_year", 1),
                     ("Best Month", "best_month", 1), ("Worst Month", "worst_month", 1),
                     ("Best Day", "best_day", 2), ("Worst Day", "worst_day", 2)]:
        P.append(sr(l, fpct(stats.get(k), d)))
    P.append("</div>")

    # Risk column
    P.append("<div><h3>Risk</h3>")
    for l, k in [("Volatility (Ann.)", "vol"), ("Downside Vol", "downside_vol"),
                  ("Max Drawdown", "max_dd"), ("Avg Drawdown", "avg_dd")]:
        P.append(sr(l, fpct(stats.get(k))))
    P.append(sr("Drawdown Duration", f"{stats.get('max_dd_duration', 0)} days"))
    for l, k in [("Sharpe Ratio", "sharpe"), ("Sortino Ratio", "sortino"), ("Calmar Ratio", "calmar")]:
        P.append(sr(l, frat(stats.get(k))))
    P.append("</div>")

    # Risk-Adjusted column
    P.append("<div><h3>Risk-Adjusted</h3>")
    for l, k in [("Omega Ratio", "omega"), ("Stability", "stability"), ("Tail Ratio", "tail_ratio")]:
        P.append(sr(l, frat(stats.get(k))))
    for l, k in [("VaR (95%)", "var_95"), ("CVaR (95%)", "cvar_95")]:
        P.append(sr(l, fpct(stats.get(k))))
    for l, k in [("Skewness", "skewness"), ("Kurtosis", "kurtosis")]:
        P.append(sr(l, frat(stats.get(k))))
    P.append(sr("Alpha", fpct(stats.get("alpha"))))
    P.append("</div>")

    # Trade & Factor column
    P.append("<div><h3>Trade &amp; Factor</h3>")
    for l, k in [("Beta", "beta"), ("Correlation", "correlation"), ("Information Ratio", "information_ratio")]:
        P.append(sr(l, frat(stats.get(k))))
    wr = stats.get("win_rate", 0)
    P.append(sr("Win Rate", f"{wr * 100:.1f}%"))
    for l, k in [("Profit Factor", "profit_factor"), ("Win/Loss Ratio", "win_loss_ratio")]:
        P.append(sr(l, frat(stats.get(k))))
    P.append(sr("Expectancy", f"{stats.get('expectancy', 0) * 100:.3f}%"))
    P.append(sr("Autocorrelation", frat(stats.get("autocorrelation"))))
    P.append(sr("Turnover Proxy", fpct(stats.get("turnover_proxy"))))
    P.append("</div>")

    P.append("</div>")  # close stats-grid

    # benchmark comparison table
    bs = stats.get("benchmark_stats")
    if bs and has_bench:
        P.append(f"<div class='sh3'>Strategy vs Benchmark</div>")
        P.append(f"<table><tr><th>Metric</th><th>Strategy</th><th>{esc(blabel)}</th></tr>")
        for l, k, fn in [("CAGR", "cagr", fpct), ("Volatility", "vol", fpct),
                          ("Sharpe", "sharpe", frat), ("Sortino", "sortino", frat),
                          ("Max Drawdown", "max_dd", fpct)]:
            P.append(f"<tr><td>{l}</td><td>{fn(stats.get(k))}</td><td>{fn(bs.get(k))}</td></tr>")
        P.append(f"<tr><td>Win Rate</td><td>{stats.get('win_rate', 0) * 100:.1f}%</td>"
                 f"<td>{bs.get('win_rate', 0) * 100:.1f}%</td></tr>")
        P.append("</table>")
    P.append("</div></div>")

    # ===== Section 3: Rolling Analytics & Risk =====
    P.append("<div class='sec'><div class='st'>Rolling Analytics &amp; Risk Analysis</div><div class='sb'>")
    P.append("<div id='rs-chart'></div>")
    P.append("<div id='rv-chart'></div>")
    if has_bench and rb_d:
        P.append("<div id='rb-chart'></div>")
    P.append("<div id='hist-chart'></div>")
    P.append("<div id='hm-chart'></div>")

    # tail risk cards
    P.append("<div class='sh3'>Tail Risk Metrics</div><div class='tg'>")
    for lbl, k, fn in [("VaR (95%)", "var_95", fpct), ("CVaR (95%)", "cvar_95", fpct),
                        ("Skewness", "skewness", frat), ("Kurtosis", "kurtosis", frat),
                        ("Tail Ratio", "tail_ratio", frat), ("Stability", "stability", frat)]:
        P.append(f"<div class='tc'><div class='lb'>{lbl}</div><div class='vl'>{fn(stats.get(k))}</div></div>")
    P.append("</div>")
    P.append("</div></div>")

    # ===== Section 4: Factor & Trade Analysis =====
    P.append("<div class='sec'><div class='st'>Factor &amp; Trade Analysis</div><div class='sb'>")
    P.append("<div id='yr-chart'></div>")

    P.append("<div class='two'>")
    # trade stats table
    P.append("<table><tr><th colspan='2'>Trade Statistics</th></tr>")
    P.append(f"<tr><td>Win Rate</td><td>{stats.get('win_rate', 0) * 100:.1f}%</td></tr>")
    P.append(f"<tr><td>Profit Factor</td><td>{frat(stats.get('profit_factor'))}</td></tr>")
    P.append(f"<tr><td>Win/Loss Ratio</td><td>{frat(stats.get('win_loss_ratio'))}</td></tr>")
    P.append(f"<tr><td>Expectancy</td><td>{stats.get('expectancy', 0) * 100:.3f}%</td></tr>")
    P.append(f"<tr><td>Best Day</td><td>{fpct(stats.get('best_day'))}</td></tr>")
    P.append(f"<tr><td>Worst Day</td><td>{fpct(stats.get('worst_day'))}</td></tr>")
    P.append(f"<tr><td>Autocorrelation</td><td>{frat(stats.get('autocorrelation'))}</td></tr>")
    P.append(f"<tr><td>Turnover Proxy</td><td>{fpct(stats.get('turnover_proxy'))}</td></tr>")
    P.append("</table>")

    # benchmark analysis table
    if has_bench:
        P.append("<table><tr><th colspan='2'>Benchmark Analysis</th></tr>")
        P.append(f"<tr><td>Alpha</td><td>{fpct(stats.get('alpha'))}</td></tr>")
        P.append(f"<tr><td>Beta</td><td>{frat(stats.get('beta'))}</td></tr>")
        P.append(f"<tr><td>Correlation</td><td>{frat(stats.get('correlation'))}</td></tr>")
        P.append(f"<tr><td>Information Ratio</td><td>{frat(stats.get('information_ratio'))}</td></tr>")
        P.append(f"<tr><td>Omega Ratio</td><td>{frat(stats.get('omega'))}</td></tr>")
        P.append(f"<tr><td>Stability</td><td>{frat(stats.get('stability'))}</td></tr>")
        P.append("</table>")
    else:
        P.append("<div></div>")  # empty cell for grid
    P.append("</div>")  # close two-col

    # strategy overview
    if subtitle:
        P.append(f"<div class='sh3'>Strategy Overview</div><p class='overview'>{esc(subtitle)}</p>")

    # key highlights
    nd = stats.get("n_days", 0)
    hls = [
        f"Total Return of {fpct(stats.get('total_return'))} over {nd} trading days",
        f"Risk-adjusted Sharpe of {frat(stats.get('sharpe'))} with Sortino of {frat(stats.get('sortino'))}",
        f"Maximum drawdown of {fpct(stats.get('max_dd'))} with average drawdown of {fpct(stats.get('avg_dd'))}",
        f"Win rate of {stats.get('win_rate', 0) * 100:.1f}% with profit factor of {frat(stats.get('profit_factor'))}",
    ]
    if has_bench and not np.isnan(stats.get("beta", np.nan)):
        hls.append(f"Beta to {blabel} of {frat(stats.get('beta'))} with correlation of {frat(stats.get('correlation'))}")
    P.append("<div class='sh3'>Key Highlights</div><ul class='hl'>")
    for h in hls:
        P.append(f"<li>{h}</li>")
    P.append("</ul>")

    # provenance
    prov_lines: List[str] = []
    if manifest_path and Path(manifest_path).exists():
        try:
            m = json.loads(Path(manifest_path).read_text())
            if "strategy_id" in m:
                prov_lines.append(f"Strategy ID: {m['strategy_id']}")
            prov_lines.append(f"Sample Period: {nd} days ({dr})")
            if "git_sha" in m:
                prov_lines.append(f"Git SHA: {m['git_sha'][:12]}")
            if "config_hash" in m:
                prov_lines.append(f"Config Hash: {m['config_hash'][:12]}")
        except Exception:
            pass
    if not prov_lines:
        prov_lines.append(f"Sample Period: {nd} days ({dr})")
    if equity_csv_path:
        prov_lines.append(f"Equity: {equity_csv_path}")
    if metrics_csv_path:
        prov_lines.append(f"Metrics: {metrics_csv_path}")
    P.append("<div class='sh3'>Data Provenance</div><div class='prov'>")
    P.append("<br>".join(esc(l) for l in prov_lines))
    P.append("</div>")

    P.append("</div></div>")  # close section

    # footer
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ft = "CONFIDENTIAL | For Institutional Investors Only | " if confidential_footer else ""
    P.append(f"<div class='ft'>{ft}Generated {gen}</div>")

    # close page & body
    P.append("</div>")  # page

    # Plotly init scripts
    P.append("<script>")
    plotly_cfg = "{responsive:true,displayModeBar:'hover',displaylogo:false}"
    for div_id, tr_json, la_json in charts:
        P.append(f"Plotly.newPlot('{div_id}',{tr_json},{la_json},{plotly_cfg});")
    P.append("</script>")

    P.append("</body>\n</html>")

    html_content = "\n".join(P)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"[html_tearsheet] Wrote {out_path}")

    if auto_open:
        import webbrowser
        webbrowser.open(out_path.as_uri())

    return out_path
