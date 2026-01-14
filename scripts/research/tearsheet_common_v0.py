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
    df = pd.read_csv(path, parse_dates=[ts_col])
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
    """Load benchmark CSV (ts, equity), align to strategy_index, ffill, normalize to 1.0."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["ts"])
    if "equity" not in df.columns:
        value_cols = [c for c in df.columns if c != "ts"]
        if value_cols:
            df = df.rename(columns={value_cols[0]: "equity"})
        else:
            raise ValueError(f"Benchmark CSV missing equity column: {path}")
    bench = df.set_index("ts")["equity"].astype(float).sort_index()
    bench = bench.reindex(strategy_index).ffill().dropna()
    if bench.empty:
        raise ValueError("Benchmark has no overlap with reference dates.")
    bench = bench / bench.iloc[0]
    bench.name = "benchmark_equity"
    return bench


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
    df = pd.read_csv(metrics_csv)
    if "period" not in df.columns:
        raise ValueError(f"metrics CSV missing 'period' column: {metrics_csv}")
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
    cols = ["Metric", strategy_label] + ([benchmark_label] if benchmark_label else [])
    return pd.DataFrame(rows, columns=cols)
