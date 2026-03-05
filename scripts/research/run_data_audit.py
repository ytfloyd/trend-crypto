#!/usr/bin/env python
"""Phase 3a — Backtest data integrity audit.

Checks:
  1. Survivorship bias — identifies symbols that stopped trading (delisted /
     effectively dead) and reports their last-seen dates.
  2. Look-ahead bias trace — scans alpha pipelines for potential future data
     leakage patterns.
  3. Return distribution — per-symbol and aggregate statistics including skew,
     kurtosis, and worst-1% return.

Outputs a markdown report to artifacts/audit/data_audit_<date>.md

Usage
-----
    python scripts/research/run_data_audit.py
    python scripts/research/run_data_audit.py --db /path/to/market.duckdb
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

_RESEARCH_DIR = str(Path(__file__).resolve().parent)
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)
DEFAULT_DB = str(REPO_ROOT / ".." / "data" / "market.duckdb")
OUTPUT_DIR = REPO_ROOT / "artifacts" / "audit"


# ---------------------------------------------------------------------------
# 1. Survivorship bias check
# ---------------------------------------------------------------------------

def check_survivorship(con: duckdb.DuckDBPyConnection, stale_days: int = 30) -> pd.DataFrame:
    """Identify symbols that stopped trading before the last data date.

    A symbol is flagged as 'stale' if its most recent data point is more
    than `stale_days` before the global last date in the database.
    Uses candles_1m for speed; avoids COUNT DISTINCT which is slow.
    """
    result = con.execute("""
        WITH sym_bounds AS (
            SELECT
                symbol,
                MIN(ts) AS first_seen,
                MAX(ts) AS last_seen,
                COUNT(*) AS n_candles
            FROM candles_1m
            GROUP BY symbol
        ),
        global_max AS (
            SELECT MAX(ts) AS global_last FROM candles_1m
        )
        SELECT
            s.symbol,
            s.first_seen,
            s.last_seen,
            s.n_candles,
            g.global_last,
            DATE_DIFF('day', s.last_seen, g.global_last) AS days_since_last
        FROM sym_bounds s
        CROSS JOIN global_max g
        ORDER BY s.last_seen ASC
    """).fetch_df()

    result["is_stale"] = result["days_since_last"] > stale_days
    return result


# ---------------------------------------------------------------------------
# 2. Look-ahead bias trace
# ---------------------------------------------------------------------------

def trace_lookahead_patterns(src_root: Path) -> list[dict]:
    """Scan alpha and strategy source files for common look-ahead patterns.

    Flags:
    - shift(-N) without corresponding shift(+N) guard
    - .iloc[-1] or .iloc[-N:] in signal computation
    - future-peeking keywords in signal paths
    """
    patterns = [
        ("shift(-", "Potential future data access via negative shift"),
        (".iloc[-1]", "Accessing last element — verify not using future bar"),
        ("pct_change().shift(-", "Forward return used in signal computation"),
        (".rolling(", "Rolling window — verify min_periods set correctly"),
    ]

    findings = []
    search_dirs = [src_root / "src" / "alphas", src_root / "src" / "strategy"]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for py_file in sorted(search_dir.rglob("*.py")):
            content = py_file.read_text(encoding="utf-8")
            for pattern, description in patterns:
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    if pattern in line and not line.strip().startswith("#"):
                        findings.append({
                            "file": str(py_file.relative_to(src_root)),
                            "line": line_num,
                            "pattern": pattern,
                            "description": description,
                            "code": line.strip()[:120],
                        })

    return findings


# ---------------------------------------------------------------------------
# 3. Return distribution analysis
# ---------------------------------------------------------------------------

def analyze_return_distributions(db_path: str) -> pd.DataFrame:
    """Compute per-symbol return distribution statistics.

    Uses the research common data loader which caches daily bars to parquet
    for fast subsequent runs.
    """
    from common.data import load_daily_bars

    df = load_daily_bars(db_path=db_path, start="2015-01-01", end="2026-12-31")

    results = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("ts")
        rets = group["close"].pct_change().dropna().values
        if len(rets) < 10:
            continue
        results.append({
            "symbol": symbol,
            "n_days": len(rets),
            "mean_daily_ret": float(np.mean(rets)),
            "std_daily_ret": float(np.std(rets, ddof=1)),
            "skewness": float(pd.Series(rets).skew()),
            "kurtosis": float(pd.Series(rets).kurtosis()),
            "worst_1pct": float(np.percentile(rets, 1)),
            "worst_5pct": float(np.percentile(rets, 5)),
            "best_1pct": float(np.percentile(rets, 99)),
            "min_ret": float(np.min(rets)),
            "max_ret": float(np.max(rets)),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    survivorship: pd.DataFrame,
    lookahead: list[dict],
    distributions: pd.DataFrame,
) -> str:
    """Generate markdown audit report."""
    lines = [
        "# Data Integrity Audit Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "---",
        "",
        "## 1. Survivorship Bias Check",
        "",
    ]

    stale = survivorship[survivorship["is_stale"]]
    active = survivorship[~survivorship["is_stale"]]
    lines.append(f"- **Total symbols**: {len(survivorship)}")
    lines.append(f"- **Active symbols**: {len(active)}")
    lines.append(f"- **Stale/delisted symbols**: {len(stale)}")
    lines.append("")

    if len(stale) > 0:
        lines.append("### Stale Symbols (potential survivorship bias)")
        lines.append("")
        lines.append("| Symbol | First Seen | Last Seen | Days Since | Candles |")
        lines.append("|--------|-----------|-----------|------------|---------|")
        for _, row in stale.iterrows():
            lines.append(
                f"| {row['symbol']} | {str(row['first_seen'])[:10]} | "
                f"{str(row['last_seen'])[:10]} | {int(row['days_since_last'])} | "
                f"{int(row['n_candles']):,} |"
            )
        lines.append("")
        lines.append(
            "> **ACTION REQUIRED**: Verify that backtest universe construction "
            "accounts for these delisted symbols. If excluded from the start, "
            "results may overstate performance."
        )
    else:
        lines.append("No stale symbols detected.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 2. Look-Ahead Bias Trace")
    lines.append("")

    if lookahead:
        lines.append(f"Found **{len(lookahead)}** potential look-ahead patterns:")
        lines.append("")
        lines.append("| File | Line | Pattern | Code |")
        lines.append("|------|------|---------|------|")
        for f in lookahead:
            code_escaped = f["code"].replace("|", "\\|")
            lines.append(
                f"| `{f['file']}` | {f['line']} | {f['description']} | "
                f"`{code_escaped}` |"
            )
        lines.append("")
        lines.append(
            "> **NOTE**: Not all flags are bugs. Review each finding manually. "
            "Negative shifts in forward return computation (for IC analysis) "
            "are expected. Negative shifts in *signal* computation are bugs."
        )
    else:
        lines.append("No look-ahead patterns detected.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Return Distribution Analysis")
    lines.append("")

    agg_skew = distributions["skewness"].mean()
    agg_kurt = distributions["kurtosis"].mean()
    worst_1_agg = distributions["worst_1pct"].mean()

    lines.append(f"- **Symbols analyzed**: {len(distributions)}")
    lines.append(f"- **Average skewness**: {agg_skew:.3f}")
    lines.append(f"- **Average excess kurtosis**: {agg_kurt:.3f}")
    lines.append(f"- **Average worst-1% daily return**: {worst_1_agg:.2%}")
    lines.append("")

    extreme = distributions.nlargest(10, "kurtosis")
    lines.append("### Top 10 Symbols by Excess Kurtosis (fattest tails)")
    lines.append("")
    lines.append(
        "| Symbol | Days | Skew | Kurtosis | Worst 1% | Min Ret |"
    )
    lines.append("|--------|------|------|----------|----------|---------|")
    for _, row in extreme.iterrows():
        lines.append(
            f"| {row['symbol']} | {int(row['n_days'])} | "
            f"{row['skewness']:.2f} | {row['kurtosis']:.1f} | "
            f"{row['worst_1pct']:.2%} | {row['min_ret']:.2%} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Sign-Off")
    lines.append("")
    lines.append("- [ ] CTO reviewed survivorship findings")
    lines.append("- [ ] CTO reviewed look-ahead trace results")
    lines.append("- [ ] CTO reviewed return distribution characteristics")
    lines.append("- [ ] Kelly criterion (Phase 3b) cleared for shadow mode")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data integrity audit.")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to DuckDB file")
    parser.add_argument("--stale-days", type=int, default=30)
    args = parser.parse_args()

    print("[audit] Connecting to database...", flush=True)
    con = duckdb.connect(args.db, read_only=True)

    try:
        print("[audit] 1/3 Checking survivorship bias...", flush=True)
        survivorship = check_survivorship(con, stale_days=args.stale_days)
        print(f"[audit]     -> {len(survivorship)} symbols checked", flush=True)
    finally:
        con.close()

    print("[audit] 2/3 Tracing look-ahead patterns...", flush=True)
    lookahead = trace_lookahead_patterns(REPO_ROOT)
    print(f"[audit]     -> {len(lookahead)} patterns found", flush=True)

    print("[audit] 3/3 Analyzing return distributions...", flush=True)
    distributions = analyze_return_distributions(args.db)
    print(f"[audit]     -> {len(distributions)} symbols analyzed", flush=True)

    report = generate_report(survivorship, lookahead, distributions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"data_audit_{ts}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[audit] Report written to {report_path}")

    json_path = OUTPUT_DIR / f"data_audit_{ts}.json"
    audit_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "survivorship": {
            "total_symbols": len(survivorship),
            "stale_symbols": int(survivorship["is_stale"].sum()),
            "active_symbols": int((~survivorship["is_stale"]).sum()),
        },
        "lookahead_findings": len(lookahead),
        "distributions": {
            "symbols_analyzed": len(distributions),
            "avg_skewness": float(distributions["skewness"].mean()),
            "avg_kurtosis": float(distributions["kurtosis"].mean()),
            "avg_worst_1pct": float(distributions["worst_1pct"].mean()),
        },
    }
    with json_path.open("w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"[audit] Summary JSON written to {json_path}")

    survivorship_csv = OUTPUT_DIR / f"survivorship_{ts}.csv"
    survivorship.to_csv(survivorship_csv, index=False)
    distributions_csv = OUTPUT_DIR / f"return_distributions_{ts}.csv"
    distributions.to_csv(distributions_csv, index=False)
    print(f"[audit] Data files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
