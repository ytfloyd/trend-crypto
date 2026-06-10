#!/usr/bin/env python
"""BofA-primer-style cross-sectional audit of MA-crossover signals on the
Coinbase USDC spot universe.

Mirrors ``scripts/research/etf_stocks_primer_audit.py`` but on the
``coinbase_crypto_ohlcv_lake.duckdb / bars_1d_clean`` table with
crypto-appropriate universe filters:

  * USDC-quoted pairs only (USD pairs are deprecated on Coinbase Advanced).
  * Stablecoin / LST bases excluded (set mirrors ``weekly_breakout_v1.py``).
  * Minimum history (>= 365 days) and coverage (>= 90 %).
  * Liquidity floor: trailing 90-day median dollar-volume >= ``--min_dv_usd``
    (default $500 k, matches the institutional floor in
    ``weekly_breakout_v1.py``). The floor is applied per-symbol-and-day so
    pairs only count on dates where they actually trade enough.

The primer engine (``analysis.tearsheet.generate_tearsheet``) handles the
robustness anchors: hit rate, Newey-West t-stat on the long-short spread,
drawdown depth + recovery, Fed-regime breakdown, and a price-based crowding
proxy. Outputs land under ``artifacts/research/crypto_primer_audit/``
alongside ``metrics_primer.csv`` and a ranked ``README.md``.

Run example:

    python scripts/research/crypto_primer_audit.py \
        --pairs 5:20,5:40,10:50,20:100,50:200 \
        --min_dv_usd 1000000

"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

# Reuse the ETF/stocks helpers so the crypto audit stays in lock-step with
# the published BofA primer methodology.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from etf_stocks_primer_audit import (  # noqa: E402
    AuditCombo,
    build_alpha_panel,
    flatten_summary,
    parse_pairs,
    write_metrics_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAKE = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "research" / "crypto_primer_audit"
DEFAULT_FED_CYCLES = REPO_ROOT / "data" / "macro" / "fed_cycles.csv"

DEFAULT_PAIRS: list[tuple[int, int]] = [
    (5, 20),
    (5, 40),
    (10, 50),
    (20, 100),
    (50, 200),
]

# Bases excluded from the USDC universe — same set as weekly_breakout_v1.py
# (stablecoins + liquid staking tokens that just track an underlying asset).
STABLE_BASES = {
    "USDT", "DAI", "USDP", "PAX", "EURC", "TUSD", "PYUSD", "USDS", "FRAX",
    "USDD", "GUSD",
    "CBETH", "MSOL", "LSETH", "OETH", "WSTETH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BofA primer-style cross-sectional audit on Coinbase USDC."
    )
    parser.add_argument("--lake", default=DEFAULT_LAKE)
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
        "--min_history_days",
        type=int,
        default=365,
        help="Minimum span between first and last bar to keep a symbol.",
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.90,
        help="Minimum (n_days / span_days) to keep a symbol.",
    )
    parser.add_argument(
        "--min_dv_usd",
        type=float,
        default=500_000.0,
        help=(
            "Trailing 90-day median dollar-volume floor (USD); applied "
            "per-bar so a symbol only contributes on liquid days."
        ),
    )
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=5,
        help="Number of quantile buckets for the Q1..QN spread.",
    )
    parser.add_argument(
        "--max_symbols",
        type=int,
        default=0,
        help="Optional cap on USDC symbols (0 = no cap, useful for smoke tests).",
    )
    parser.add_argument(
        "--start_date",
        default="2018-01-01",
        help=(
            "Hard floor on bar date — early lake history is sparse so we "
            "default to 2018-01-01 to keep the cross-sectional ranking "
            "well-defined. Set to e.g. 2015-01-01 to use everything."
        ),
    )
    return parser.parse_args()


def load_crypto_panel(
    lake_path: str,
    *,
    min_history_days: int,
    min_coverage: float,
    min_dv_usd: float,
    start_date: str,
    max_symbols: int,
) -> pl.DataFrame:
    """Load (symbol, ts, open, close) panel for the filtered USDC universe.

    The liquidity filter is applied per-bar: a symbol is dropped on dates
    where its trailing 90-day median dollar-volume is below ``min_dv_usd``.
    This is point-in-time correct (no look-ahead) and matches the
    weekly_breakout_v1.py convention.
    """
    if not Path(lake_path).exists():
        raise FileNotFoundError(f"DuckDB not found: {lake_path}")

    con = duckdb.connect(lake_path, read_only=True)
    try:
        # Stage 1: per-symbol stats so we can apply the structural filters.
        stables_csv = ",".join(f"'{s}'" for s in sorted(STABLE_BASES))
        stats_sql = f"""
            WITH s AS (
                SELECT
                    symbol,
                    split_part(symbol, '-', 1) AS base,
                    MIN(ts) AS first_ts,
                    MAX(ts) AS last_ts,
                    COUNT(*) AS n_days
                FROM bars_1d_clean
                WHERE symbol LIKE '%-USDC'
                GROUP BY symbol
            )
            SELECT symbol, first_ts, last_ts, n_days, base
            FROM s
            WHERE base NOT IN ({stables_csv})
        """
        stats = con.execute(stats_sql).fetch_arrow_table()
        stats_df = pl.from_arrow(stats)
        stats_df = stats_df.with_columns(
            [
                pl.col("first_ts").cast(pl.Datetime("us")),
                pl.col("last_ts").cast(pl.Datetime("us")),
            ]
        )
        stats_df = stats_df.with_columns(
            [
                ((pl.col("last_ts") - pl.col("first_ts")).dt.total_days())
                .alias("span_days"),
            ]
        )
        stats_df = stats_df.with_columns(
            (pl.col("n_days") / pl.col("span_days").cast(pl.Float64)).alias(
                "coverage"
            )
        )
        keep = stats_df.filter(
            (pl.col("span_days") >= min_history_days)
            & (pl.col("coverage") >= min_coverage)
        ).sort("symbol")
        if max_symbols and max_symbols > 0:
            keep = keep.head(max_symbols)
        symbols = keep["symbol"].to_list()
        if not symbols:
            raise RuntimeError(
                "No USDC symbols passed structural filters "
                f"(min_history_days={min_history_days}, min_coverage={min_coverage})"
            )

        # Stage 2: pull bars for the kept symbols.
        symlist = ",".join(f"'{s}'" for s in symbols)
        bars_sql = f"""
            SELECT
                symbol,
                ts,
                open,
                high,
                low,
                close,
                volume
            FROM bars_1d_clean
            WHERE symbol IN ({symlist})
              AND ts >= TIMESTAMPTZ '{start_date}'
            ORDER BY symbol, ts
        """
        bars = con.execute(bars_sql).fetch_arrow_table()
    finally:
        con.close()

    df = pl.from_arrow(bars)
    df = df.with_columns(
        [
            # Strip the timezone so it matches the rest of the primer pipeline
            # (which works in tz-naive UTC like the ETF/stocks audit).
            pl.col("ts").cast(pl.Datetime("us", time_zone=None)).alias("ts"),
            (pl.col("close") * pl.col("volume")).alias("dv_usd"),
        ]
    ).sort(["symbol", "ts"])

    # Per-symbol trailing 90d median dollar volume (point-in-time).
    df = df.with_columns(
        pl.col("dv_usd")
        .rolling_median(window_size=90, min_samples=20)
        .over("symbol")
        .alias("dv90_med")
    )
    df = df.filter(
        pl.col("dv90_med").is_not_null() & (pl.col("dv90_med") >= min_dv_usd)
    )
    return df.select(["symbol", "ts", "open", "close"])


def run_crypto_combo(
    panel: pl.DataFrame,
    *,
    fast: int,
    slow: int,
    out_root: Path,
    fed_cycles_path: str | None,
    n_quantiles: int,
) -> dict[str, object] | None:
    """Run the primer-style tearsheet for one crypto MA pair."""
    from analysis.tearsheet import generate_tearsheet

    combo = AuditCombo(asset_class="crypto", fast=fast, slow=slow)
    alpha_panel = build_alpha_panel(panel, fast=fast, slow=slow)
    if alpha_panel.is_empty():
        return None

    counts = alpha_panel.group_by("ts").len().rename({"len": "n_symbols"})
    alpha_panel = (
        alpha_panel.join(counts, on="ts", how="left")
        .filter(pl.col("n_symbols") >= n_quantiles)
        .drop("n_symbols")
    )
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


def write_readme_crypto(rows: list[dict[str, object]], out_path: Path) -> None:
    """Crypto-specific README mirroring the ETF/stocks layout."""
    if not rows:
        out_path.write_text(
            "# Coinbase USDC Primer Audit\n\n_No combos completed._\n"
        )
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
    lines.append("# Coinbase USDC BofA Primer-Style Audit\n")
    lines.append(
        "Cross-sectional MA-crossover signals (continuous "
        "`(fast_ma - slow_ma) / slow_ma`) ranked into Q1..Q5 with"
        " open-to-open forward returns on Coinbase USDC spot pairs."
        " Applied filters: stablecoin/LST bases excluded, >= 365d history,"
        " >= 90% coverage, trailing-90d median dollar-volume floor."
        " Outputs follow the BofA *Quantitative Primer* (June 2022)"
        " conventions: spread Sharpe + IC paired with hit rate, Newey-West"
        " autocorrelation-corrected t-stat, drawdown depth and recovery,"
        " a Fed-regime breakdown, and a price-based crowding proxy.\n"
    )
    lines.append(
        "Robustness score = `hit_rate * spread_nw_tstat`. "
        "This deliberately punishes pairs whose Sharpe is built from a "
        "few large months (high simple Sharpe but low hit rate or low NW "
        "t-stat).\n"
    )

    lines.append("\n## Crypto (Coinbase USDC)\n")
    lines.append(
        "| pair | spread Sharpe | NW t-stat | hit rate | max DD | "
        "recovery | crowd %ile | robust score |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in df.iter_rows(named=True):
        pair = f"{r['fast']}/{r['slow']}"
        sharpe = r.get("spread_sharpe") or 0.0
        nwt = r.get("spread_nw_tstat") or 0.0
        hit = r.get("hit_rate") or 0.0
        dd = r.get("max_dd") or 0.0
        rec = r.get("recovery_periods")
        crowd = r.get("crowding_pct_history_last")
        score = r.get("primer_robust_score") or 0.0
        rec_str = f"{int(rec)}" if rec not in (None, "") else "-"
        crowd_str = (
            f"{float(crowd):.0%}"
            if crowd not in (None, "") and not (
                isinstance(crowd, float) and math.isnan(crowd)
            )
            else "-"
        )
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
    for r in df.iter_rows(named=True):
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
        "Each `crypto/{fast}_{slow}/tearsheet.pdf` is a 2-page document.\n"
        "**Page 1** retains the original layer-cake equity, rolling IC,"
        " turnover, and net-rank exposure panels.\n"
        "**Page 2** adds the BofA primer-style anchors: annual spread Sharpe"
        " with hit rates, spread Sharpe by Fed regime, the spread underwater"
        " curve, and the long-basket crowding percentile vs trailing 5-year"
        " history.\n"
    )

    out_path.write_text("".join(lines))


def main() -> None:
    args = parse_args()
    pairs = parse_pairs(args.pairs)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fed_cycles_path = (
        args.fed_cycles
        if args.fed_cycles and Path(args.fed_cycles).exists()
        else None
    )

    print(f"[crypto-primer-audit] loading USDC panel from {args.lake}")
    panel = load_crypto_panel(
        args.lake,
        min_history_days=args.min_history_days,
        min_coverage=args.min_coverage,
        min_dv_usd=args.min_dv_usd,
        start_date=args.start_date,
        max_symbols=args.max_symbols,
    )
    n_sym = panel["symbol"].n_unique()
    print(
        f"[crypto-primer-audit] panel rows={panel.height:,} symbols={n_sym} "
        f"date_min={panel['ts'].min()} date_max={panel['ts'].max()}"
    )

    rows: list[dict[str, object]] = []
    for fast, slow in pairs:
        print(f"[crypto-primer-audit] running {fast}/{slow}")
        try:
            summary = run_crypto_combo(
                panel,
                fast=fast,
                slow=slow,
                out_root=out_root,
                fed_cycles_path=fed_cycles_path,
                n_quantiles=args.n_quantiles,
            )
        except Exception as exc:
            print(f"[crypto-primer-audit]   FAILED {fast}/{slow}: {exc!r}")
            continue
        if summary is None:
            print(f"[crypto-primer-audit]   skipped {fast}/{slow}: insufficient data")
            continue
        rows.append(flatten_summary(summary))

    metrics_path = out_root / "metrics_primer.csv"
    write_metrics_csv(rows, metrics_path)
    print(f"[crypto-primer-audit] wrote {metrics_path}")

    readme_path = out_root / "README.md"
    write_readme_crypto(rows, readme_path)
    print(f"[crypto-primer-audit] wrote {readme_path}")

    summary_index = out_root / "summary_index.json"
    summary_index.write_text(json.dumps(rows, indent=2, default=str))
    print(f"[crypto-primer-audit] wrote {summary_index}")


if __name__ == "__main__":
    main()
