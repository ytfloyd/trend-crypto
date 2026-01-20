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
from typing import Dict, Optional, Tuple, List
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
