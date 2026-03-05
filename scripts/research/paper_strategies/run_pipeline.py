#!/usr/bin/env python
"""Master runner: Phase A -> B -> C -> Master Report.

Chains the three research phases derived from the paper discovery catalogue.
Each phase reads from and writes to disk; no in-memory state crosses phases.

Usage:
    python -m scripts.research.paper_strategies.run_pipeline [OPTIONS]

Options:
    --db PATH       DuckDB path (auto-detects if omitted)
    --start DATE    Start date (default: 2017-01-01)
    --end DATE      End date (default: 2026-12-31)
    --phase PHASE   Run only a specific phase: a, b, c, or report
    --skip-c        Skip Phase C (useful if universe is small)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path("artifacts/research")
LOG_PATH = ARTIFACTS / "run_log.txt"


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_phase(phase_module: str, db: str | None, start: str, end: str, table: str | None = None) -> int:
    """Run a phase script as a subprocess."""
    cmd = [sys.executable, "-m", phase_module]
    if db:
        cmd.extend(["--db", db])
    if table:
        cmd.extend(["--table", table])
    cmd.extend(["--start", start, "--end", end])

    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def generate_master_report(skip_c: bool = False) -> Path:
    """Generate the final master report from all phase outputs."""
    date_str = datetime.utcnow().strftime("%Y%m%d")
    report_path = ARTIFACTS / f"master_report_{date_str}.md"

    lines = [
        f"# Master Research Report",
        f"Generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC",
        "",
        "---",
        "",
    ]

    # --- Phase A summary ---
    lines.append("## Phase A: Alpha Decay Findings")
    lines.append("")

    queue_path = ARTIFACTS / "alpha_decay" / "research_queue.json"
    decay_path = ARTIFACTS / "alpha_decay" / "decay_report.md"
    failed_a = ARTIFACTS / "alpha_decay" / "FAILED.md"

    if failed_a.exists():
        lines.append(f"**FAILED**: {failed_a.read_text().strip()}")
    elif queue_path.exists():
        queue = json.loads(queue_path.read_text())
        high = [f for f in queue if f["priority"] == "HIGH"]
        med = [f for f in queue if f["priority"] == "MEDIUM"]
        low = [f for f in queue if f["priority"] == "LOW"]
        avoid = [f for f in queue if f["priority"] == "AVOID"]

        lines.extend([
            f"Factors analyzed: {len(queue)}",
            f"- HIGH priority: {len(high)} ({', '.join(f['factor'] for f in high)})" if high else "- HIGH priority: 0",
            f"- MEDIUM priority: {len(med)} ({', '.join(f['factor'] for f in med)})" if med else "- MEDIUM priority: 0",
            f"- LOW priority: {len(low)}" if low else "- LOW priority: 0",
            f"- AVOID: {len(avoid)}" if avoid else "- AVOID: 0",
            "",
        ])

        lines.append("| Factor | Sharpe | Decay | Half-life | Crowding | Priority |")
        lines.append("|---|---|---|---|---|---|")
        for f in queue:
            hl = f"{f['half_life_days']:.0f}d" if f.get("half_life_days") else "STABLE"
            lines.append(
                f"| {f['factor']} | {f['sharpe_full']:.3f} | "
                f"{f['decay_label']} | {hl} | {f['crowding_trend']} | "
                f"**{f['priority']}** |"
            )
        lines.append("")
    else:
        lines.append("Phase A outputs not found.")
    lines.append("")

    # --- Phase B summary ---
    lines.append("## Phase B: SPO Findings")
    lines.append("")

    spo_path = ARTIFACTS / "spo" / "spo_comparison_report.md"
    failed_b = ARTIFACTS / "spo" / "FAILED.md"

    if failed_b.exists():
        lines.append(f"**FAILED**: {failed_b.read_text().strip()}")
    elif spo_path.exists():
        lines.append(spo_path.read_text())
    else:
        lines.append("Phase B outputs not found.")
    lines.append("")

    # --- Phase C summary ---
    lines.append("## Phase C: Correlation Forecasting Findings")
    lines.append("")

    corr_path = ARTIFACTS / "correlation" / "correlation_forecast_report.md"
    failed_c = ARTIFACTS / "correlation" / "FAILED.md"

    if skip_c:
        lines.append("Phase C was skipped (--skip-c flag).")
    elif failed_c.exists():
        lines.append(failed_c.read_text())
    elif corr_path.exists():
        lines.append(corr_path.read_text())
    else:
        lines.append("Phase C outputs not found.")
    lines.append("")

    # --- Recommendations ---
    lines.extend([
        "---",
        "",
        "## Recommended Next Actions",
        "",
    ])

    if queue_path.exists():
        queue = json.loads(queue_path.read_text())
        high = [f for f in queue if f["priority"] == "HIGH"]
        if high:
            lines.append("### 1. Forward-test HIGH-priority factors")
            for f in high:
                lines.append(f"- **{f['factor']}**: Sharpe {f['sharpe_full']:.3f}, {f['decay_label']}")
            lines.extend([
                "  - Estimated effort: 2-3 days per factor",
                "  - Output: 30-day paper trading equity curve",
                "",
            ])

        lines.extend([
            "### 2. Integrate best SPO model into live signal pipeline",
            "  - If SPO improved Sharpe, wire the model into `src/strategy/`",
            "  - Estimated effort: 3-5 days",
            "",
            "### 3. Monitor factor decay quarterly",
            "  - Re-run Phase A every 90 days to track decay trajectories",
            "  - Estimated effort: 0.5 days (re-run script)",
            "",
        ])

    # --- Forward testing spec ---
    lines.extend([
        "---",
        "",
        "## Forward Testing Specification",
        "",
        "For any model/strategy that passed all three phases:",
        "",
        "- **Rebalance frequency**: Daily",
        "- **Position sizing**: Equal-volatility (or Phase C forecast if it improved)",
        "- **Target volatility**: 15% annualized",
        "- **Max position size**: 20%",
        "- **Paper trading period**: 30 days minimum before any live allocation",
        "",
        "**Kill switch conditions**:",
        "- Drawdown exceeds 10% in first 30 days -> halt and review",
        "- Sharpe (rolling 30d) drops below 0 for 14 consecutive days -> halt",
        "- Any single position moves > 3 standard deviations in one day -> reduce 50%, review",
    ])

    report_path.write_text("\n".join(lines))
    log(f"Master report written to {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Strategies Research Pipeline")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--table", type=str, default=None,
                        help="Pre-materialized table (e.g. bars_1d_usd_universe_clean)")
    parser.add_argument("--start", type=str, default="2017-01-01")
    parser.add_argument("--end", type=str, default="2026-12-31")
    parser.add_argument("--phase", type=str, default=None, choices=["a", "b", "c", "report"])
    parser.add_argument("--skip-c", action="store_true", help="Skip Phase C")
    args = parser.parse_args()

    log("=" * 70)
    log("PAPER STRATEGIES RESEARCH PIPELINE")
    log("=" * 70)

    phases = {
        "a": "scripts.research.paper_strategies.phase_a_decay",
        "b": "scripts.research.paper_strategies.phase_b_spo",
        "c": "scripts.research.paper_strategies.phase_c_correlation",
    }

    if args.phase == "report":
        generate_master_report(skip_c=args.skip_c)
        return

    if args.phase:
        run_targets = [args.phase]
    else:
        run_targets = ["a", "b"]
        if not args.skip_c:
            run_targets.append("c")

    for phase_key in run_targets:
        module = phases[phase_key]
        log(f"\n{'='*50}")
        log(f"LAUNCHING PHASE {phase_key.upper()}")
        log(f"{'='*50}")

        rc = run_phase(module, args.db, args.start, args.end, table=args.table)
        if rc != 0:
            log(f"Phase {phase_key.upper()} exited with code {rc}")
            log("Continuing to next phase (failure report should exist on disk)")

    # Generate master report
    generate_master_report(skip_c=args.skip_c)
    log("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()
