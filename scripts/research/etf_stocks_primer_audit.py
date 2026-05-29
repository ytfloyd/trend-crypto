#!/usr/bin/env python
"""BofA-primer-style cross-sectional audit of MA-crossover signals on ETF +
US-stock daily lakes.

For every ``(asset_class, fast, slow)`` pair we build a tidy alpha panel where:

  - signal at close[t] = ``(fast_ma - slow_ma) / slow_ma`` per symbol
    (continuous version of the +1 / -1 crossover, suitable for cross-sectional
    ranking into Q1..Q5)
  - forward_ret at close[t] = ``open[t+1] / open[t] - 1`` per symbol (matches
    the close-driven, next-open-execution convention of
    ``scripts/research/etf_stocks_ma_crossover.py``)

Each panel is then routed through ``analysis.tearsheet.generate_tearsheet``
which already produces quintile equity, Spearman IC, turnover, and now (after
the Tier 4 retrofit) the BofA primer's robustness anchors: hit rate, NW
t-stat, drawdown stats, annual + Fed-regime breakdowns, and a price-based
crowding proxy. The summary JSONs are aggregated into ``metrics_primer.csv``
and an asset-class-aware README ranks pairs by ``hit_rate * spread_nw_tstat``.

This script does not regenerate the ``metrics.csv`` produced by the original
sweep; it sits alongside under ``artifacts/research/etf_stocks_primer_audit/``
and answers the BofA primer-style questions the per-symbol metrics did not.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ETF_DB = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/etf_market.duckdb"
DEFAULT_STOCKS_DB = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/stocks_market.duckdb"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "research" / "etf_stocks_primer_audit"
DEFAULT_FED_CYCLES = REPO_ROOT / "data" / "macro" / "fed_cycles.csv"

DEFAULT_PAIRS: list[tuple[int, int]] = [
    (5, 20),
    (5, 40),
    (10, 50),
    (20, 100),
    (50, 200),
]


@dataclass(frozen=True)
class AuditCombo:
    asset_class: str
    fast: int
    slow: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BofA primer-style cross-sectional audit of MA crossover signals."
    )
    parser.add_argument("--etf_db", default=DEFAULT_ETF_DB)
    parser.add_argument("--stocks_db", default=DEFAULT_STOCKS_DB)
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--fed_cycles",
        default=str(DEFAULT_FED_CYCLES),
        help="Path to fed_cycles.csv for primer-style regime breakdown.",
    )
    parser.add_argument(
        "--pairs",
        default=",".join(f"{f}:{s}" for f, s in DEFAULT_PAIRS),
        help="Comma-separated fast:slow MA pairs (e.g. 5:40,10:50).",
    )
    parser.add_argument(
        "--max_symbols",
        type=int,
        default=0,
        help="Optional cap per asset class (0 = no cap, useful for smoke tests).",
    )
    parser.add_argument(
        "--asset_classes",
        default="etf,stocks",
        help="Comma-separated asset classes to run; subset of {etf,stocks}.",
    )
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=5,
        help="Number of quantile buckets for the Q1..QN spread.",
    )
    return parser.parse_args()


def parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        f_str, s_str = chunk.split(":")
        f, s = int(f_str), int(s_str)
        if f >= s:
            raise ValueError(f"fast must be < slow, got {f}:{s}")
        pairs.append((f, s))
    return pairs


def load_panel(db_path: str, max_symbols: int = 0) -> pl.DataFrame:
    """Load (symbol, ts, open, close) panel as a polars DataFrame."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DuckDB not found: {db_path}")
    con = duckdb.connect(db_path, read_only=True)
    try:
        if max_symbols and max_symbols > 0:
            sql = f"""
                WITH ranked AS (
                    SELECT symbol, MIN(ts) AS first_ts
                    FROM bars_1d
                    GROUP BY symbol
                    ORDER BY symbol
                    LIMIT {max_symbols}
                )
                SELECT b.symbol, b.ts, b.open, b.close
                FROM bars_1d b
                JOIN ranked r USING (symbol)
                ORDER BY b.symbol, b.ts
            """
        else:
            sql = "SELECT symbol, ts, open, close FROM bars_1d ORDER BY symbol, ts"
        arrow = con.execute(sql).fetch_arrow_table()
    finally:
        con.close()
    df = pl.from_arrow(arrow)
    df = df.with_columns(pl.col("ts").cast(pl.Datetime))
    return df


def build_alpha_panel(
    panel: pl.DataFrame,
    *,
    fast: int,
    slow: int,
) -> pl.DataFrame:
    """Build a tidy ``ts, symbol, signal, forward_ret`` panel for one MA pair.

    Notes
    -----
    - ``signal`` = ``(fast_ma - slow_ma) / slow_ma`` computed on the *shifted*
      close so that signal at index ``t`` only uses information through
      ``t-1``. This matches the +1 / -1 crossover used in
      ``scripts/research/etf_stocks_ma_crossover.py``.
    - ``forward_ret`` at index ``t`` = ``open[t+1] / open[t] - 1`` per symbol.
    """
    panel = panel.sort(["symbol", "ts"]).with_columns(
        pl.col("close").shift(1).over("symbol").alias("close_shift")
    )
    panel = panel.with_columns(
        [
            pl.col("close_shift")
            .rolling_mean(window_size=fast, min_samples=fast)
            .over("symbol")
            .alias("fast_ma"),
            pl.col("close_shift")
            .rolling_mean(window_size=slow, min_samples=slow)
            .over("symbol")
            .alias("slow_ma"),
            pl.col("open").shift(-1).over("symbol").alias("next_open"),
        ]
    )
    panel = panel.with_columns(
        [
            ((pl.col("fast_ma") - pl.col("slow_ma")) / pl.col("slow_ma")).alias("signal"),
            (pl.col("next_open") / pl.col("open") - 1.0).alias("forward_ret"),
        ]
    )
    panel = panel.filter(
        pl.col("signal").is_not_null()
        & pl.col("forward_ret").is_not_null()
        & pl.col("signal").is_finite()
        & pl.col("forward_ret").is_finite()
    )
    return panel.select(["ts", "symbol", "signal", "forward_ret"])


def run_combo(
    panel: pl.DataFrame,
    *,
    combo: AuditCombo,
    out_root: Path,
    fed_cycles_path: str | None,
    n_quantiles: int,
) -> dict[str, object] | None:
    """Run primer-style tearsheet on one ``(asset_class, fast, slow)`` combo.

    Returns the summary dict (with combo identifiers added) or ``None`` if
    the panel was empty after rolling-mean warmup.
    """
    from analysis.tearsheet import generate_tearsheet

    alpha_panel = build_alpha_panel(panel, fast=combo.fast, slow=combo.slow)
    if alpha_panel.is_empty():
        return None

    counts = alpha_panel.group_by("ts").len().rename({"len": "n_symbols"})
    alpha_panel = alpha_panel.join(counts, on="ts", how="left").filter(
        pl.col("n_symbols") >= n_quantiles
    ).drop("n_symbols")
    if alpha_panel.is_empty():
        return None
    if alpha_panel.select(pl.col("ts").n_unique()).item() < 2:
        return None

    out_dir = out_root / combo.asset_class / f"{combo.fast}_{combo.slow}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "tearsheet"

    summary = generate_tearsheet(
        alpha_panel,
        str(output),
        alpha_name=f"{combo.asset_class}_{combo.fast}_{combo.slow}",
        n_quantiles=n_quantiles,
        emit_returns=True,
        fed_cycles_path=fed_cycles_path,
    )
    summary["asset_class"] = combo.asset_class
    summary["fast"] = combo.fast
    summary["slow"] = combo.slow
    return summary


def flatten_summary(summary: dict[str, object]) -> dict[str, object]:
    """Convert the nested summary into a flat row for ``metrics_primer.csv``."""
    dd = summary.get("drawdown", {}) or {}
    crowding = summary.get("crowding", {}) or {}
    regime = {r["phase"]: r for r in summary.get("regime_breakdown", [])}

    def _phase(name: str, key: str) -> object:
        r = regime.get(name)
        return r[key] if r else ""

    flat: dict[str, object] = {
        "asset_class": summary.get("asset_class"),
        "fast": summary.get("fast"),
        "slow": summary.get("slow"),
        "n_rows": summary.get("n_rows"),
        "n_symbols": summary.get("n_symbols"),
        "date_start": summary.get("date_range", ["", ""])[0],
        "date_end": summary.get("date_range", ["", ""])[1],
        "effective_quantiles": summary.get("effective_quantiles"),
        "mean_ic": summary.get("mean_ic"),
        "tstat_ic": summary.get("tstat_ic"),
        "spread_sharpe": summary.get("spread_sharpe"),
        "spread_nw_tstat": summary.get("spread_nw_tstat"),
        "spread_nw_lags": summary.get("spread_nw_lags"),
        "hit_rate": summary.get("hit_rate"),
        "mean_daily_turnover": summary.get("mean_daily_turnover"),
        "max_dd": dd.get("max_dd"),
        "drawdown_periods": dd.get("drawdown_periods"),
        "recovery_periods": dd.get("recovery_periods"),
        "trough_ts": dd.get("trough_ts"),
        "recovery_ts": dd.get("recovery_ts"),
        "crowding_pct_history_last": crowding.get("pct_history_last"),
        "crowding_pct_history_mean": crowding.get("pct_history_mean"),
        "early_hike_sharpe": _phase("early_hike", "sharpe"),
        "early_hike_hit": _phase("early_hike", "hit_rate"),
        "early_hike_n": _phase("early_hike", "n"),
        "late_hike_sharpe": _phase("late_hike", "sharpe"),
        "late_hike_hit": _phase("late_hike", "hit_rate"),
        "late_hike_n": _phase("late_hike", "n"),
        "easing_sharpe": _phase("easing", "sharpe"),
        "easing_hit": _phase("easing", "hit_rate"),
        "easing_n": _phase("easing", "n"),
        "neutral_sharpe": _phase("neutral", "sharpe"),
        "neutral_hit": _phase("neutral", "hit_rate"),
        "neutral_n": _phase("neutral", "n"),
    }
    return flat


def write_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    pl.from_dicts(rows).write_csv(path)


def write_readme(rows: list[dict[str, object]], out_path: Path) -> None:
    """Markdown summary ranked by ``hit_rate * spread_nw_tstat`` per asset class.

    The primer's emphasis is on robustness, so we anchor the leaderboard on
    the pair of metrics it consistently treats as anchors: probability the
    long basket beats the short (hit rate) and the autocorrelation-corrected
    t-stat of the spread.
    """
    if not rows:
        out_path.write_text("# ETF + Stocks Primer Audit\n\n_No combos completed._\n")
        return

    df = pl.from_dicts(rows)
    df = df.with_columns(
        (
            pl.col("hit_rate").cast(pl.Float64, strict=False)
            * pl.col("spread_nw_tstat").cast(pl.Float64, strict=False)
        ).alias("primer_robust_score")
    )
    df = df.sort("primer_robust_score", descending=True, nulls_last=True)

    lines: list[str] = []
    lines.append("# ETF + US Stocks BofA Primer-Style Audit\n")
    lines.append("Cross-sectional MA-crossover signals (continuous "
                 "`(fast_ma - slow_ma) / slow_ma`) ranked into Q1..Q5 with"
                 " open-to-open forward returns. Outputs follow the BofA"
                 " *Quantitative Primer* (June 2022) conventions: spread"
                 " Sharpe and IC are paired with hit rate, Newey-West"
                 " autocorrelation-corrected t-stat, drawdown depth and"
                 " recovery, a Fed-regime breakdown, and a price-based"
                 " crowding proxy.\n")
    lines.append(
        "Robustness score = `hit_rate * spread_nw_tstat`. "
        "This deliberately punishes pairs whose Sharpe is built from a "
        "few large months (high simple Sharpe but low hit rate or low NW "
        "t-stat).\n"
    )

    for asset_class in ["etf", "stocks"]:
        sub = df.filter(pl.col("asset_class") == asset_class)
        if sub.height == 0:
            continue
        lines.append(f"\n## {asset_class.upper()}\n")
        lines.append(
            "| pair | spread Sharpe | NW t-stat | hit rate | max DD | "
            "recovery | crowd %ile | robust score |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for r in sub.iter_rows(named=True):
            pair = f"{r['fast']}/{r['slow']}"
            sharpe = r.get("spread_sharpe") or 0.0
            nwt = r.get("spread_nw_tstat") or 0.0
            hit = r.get("hit_rate") or 0.0
            dd = r.get("max_dd") or 0.0
            rec = r.get("recovery_periods")
            crowd = r.get("crowding_pct_history_last")
            score = r.get("primer_robust_score") or 0.0
            rec_str = f"{int(rec)}" if rec not in (None, "") else "-"
            crowd_str = f"{float(crowd):.0%}" if crowd not in (None, "") and not (
                isinstance(crowd, float) and math.isnan(crowd)
            ) else "-"
            lines.append(
                f"| {pair} | {float(sharpe):+.2f} | {float(nwt):+.2f} | "
                f"{float(hit):.0%} | {float(dd):.1%} | {rec_str} | "
                f"{crowd_str} | {float(score):+.2f} |\n"
            )

        lines.append("\n### Fed-regime spread Sharpe (sorted by robust score)\n")
        lines.append(
            "| pair | early_hike | late_hike | easing | neutral |\n"
            "|---|---:|---:|---:|---:|\n"
        )
        for r in sub.iter_rows(named=True):
            def _fmt(key: str) -> str:
                v = r.get(key)
                if v in (None, ""):
                    return "-"
                try:
                    return f"{float(v):+.2f}"
                except (TypeError, ValueError):
                    return "-"
            lines.append(
                f"| {r['fast']}/{r['slow']} | {_fmt('early_hike_sharpe')} | "
                f"{_fmt('late_hike_sharpe')} | {_fmt('easing_sharpe')} | "
                f"{_fmt('neutral_sharpe')} |\n"
            )

    lines.append(
        "\n## Reading the dashboards\n"
        "Each `{asset_class}/{fast}_{slow}/tearsheet.pdf` is a 2-page document.\n"
        "**Page 1** retains the original layer-cake equity, rolling IC, turnover,"
        " and net-rank exposure panels.\n"
        "**Page 2** adds the BofA primer-style anchors: annual spread Sharpe with"
        " hit rates, spread Sharpe by Fed regime, the spread underwater curve, and"
        " the long-basket crowding percentile vs trailing 5-year history.\n"
    )

    out_path.write_text("".join(lines))


def main() -> None:
    args = parse_args()
    pairs = parse_pairs(args.pairs)
    asset_classes = [
        a.strip() for a in args.asset_classes.split(",") if a.strip()
    ]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fed_cycles_path = (
        args.fed_cycles
        if args.fed_cycles and Path(args.fed_cycles).exists()
        else None
    )

    rows: list[dict[str, object]] = []

    if "etf" in asset_classes:
        print(f"[primer-audit] loading ETF panel from {args.etf_db}")
        etf_panel = load_panel(args.etf_db, max_symbols=args.max_symbols)
        print(
            f"[primer-audit] etf rows={etf_panel.height:,} "
            f"symbols={etf_panel['symbol'].n_unique()}"
        )
        for fast, slow in pairs:
            combo = AuditCombo(asset_class="etf", fast=fast, slow=slow)
            print(f"[primer-audit] running etf {fast}/{slow}")
            summary = run_combo(
                etf_panel,
                combo=combo,
                out_root=out_root,
                fed_cycles_path=fed_cycles_path,
                n_quantiles=args.n_quantiles,
            )
            if summary is None:
                print(f"[primer-audit]   skipped {combo}: insufficient data")
                continue
            rows.append(flatten_summary(summary))

    if "stocks" in asset_classes:
        print(f"[primer-audit] loading STOCKS panel from {args.stocks_db}")
        stocks_panel = load_panel(args.stocks_db, max_symbols=args.max_symbols)
        print(
            f"[primer-audit] stocks rows={stocks_panel.height:,} "
            f"symbols={stocks_panel['symbol'].n_unique()}"
        )
        for fast, slow in pairs:
            combo = AuditCombo(asset_class="stocks", fast=fast, slow=slow)
            print(f"[primer-audit] running stocks {fast}/{slow}")
            summary = run_combo(
                stocks_panel,
                combo=combo,
                out_root=out_root,
                fed_cycles_path=fed_cycles_path,
                n_quantiles=args.n_quantiles,
            )
            if summary is None:
                print(f"[primer-audit]   skipped {combo}: insufficient data")
                continue
            rows.append(flatten_summary(summary))

    metrics_path = out_root / "metrics_primer.csv"
    write_metrics_csv(rows, metrics_path)
    print(f"[primer-audit] wrote {metrics_path}")

    readme_path = out_root / "README.md"
    write_readme(rows, readme_path)
    print(f"[primer-audit] wrote {readme_path}")

    summary_index = out_root / "summary_index.json"
    summary_index.write_text(json.dumps(rows, indent=2, default=str))
    print(f"[primer-audit] wrote {summary_index}")


if __name__ == "__main__":
    main()
