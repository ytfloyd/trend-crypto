"""Strategy catalogue: persistence, reporting, and comparison utilities.

Stores results as JSON + CSV in artifacts/research/paper_pipeline/.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .models import CatalogueEntry, FilterResult, PaperMeta, StrategySpec

DEFAULT_OUT_DIR = Path("artifacts/research/paper_pipeline")


def build_catalogue(
    passed: list[tuple[PaperMeta, FilterResult, StrategySpec]],
    rejected: list[tuple[PaperMeta, FilterResult]],
) -> list[CatalogueEntry]:
    """Build CatalogueEntry objects from pipeline results."""
    entries: list[CatalogueEntry] = []

    for paper, filt, spec in passed:
        entries.append(CatalogueEntry(paper=paper, filter_result=filt, strategy=spec))

    for paper, filt in rejected:
        entries.append(CatalogueEntry(paper=paper, filter_result=filt, strategy=None))

    return entries


def save_catalogue(
    entries: list[CatalogueEntry],
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Path]:
    """Persist catalogue to disk as JSON + CSV files.

    Returns dict of output file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Full JSON catalogue
    catalogue_json = out_dir / f"catalogue_{timestamp}.json"
    data = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "total_papers": len(entries),
        "passed": sum(1 for e in entries if e.filter_result.passed),
        "rejected": sum(1 for e in entries if not e.filter_result.passed),
        "entries": [e.to_dict() for e in entries],
    }
    catalogue_json.write_text(json.dumps(data, indent=2, default=str))

    # Also write a "latest" symlink-style copy
    latest_json = out_dir / "catalogue_latest.json"
    latest_json.write_text(json.dumps(data, indent=2, default=str))

    # Passed strategies CSV (for quick scanning)
    passed_rows = []
    for e in entries:
        if e.filter_result.passed and e.strategy:
            passed_rows.append({
                "paper_id": e.paper.paper_id,
                "title": e.paper.title,
                "authors": "; ".join(e.paper.authors[:3]),
                "date": e.paper.publication_date,
                "source": e.paper.source,
                "strategy_type": e.strategy.strategy_type.value,
                "asset_classes": ", ".join(e.strategy.asset_classes),
                "rebalance": e.strategy.rebalance_frequency,
                "claimed_sharpe": e.strategy.claimed_sharpe,
                "claimed_cagr": e.strategy.claimed_cagr,
                "oos": e.strategy.out_of_sample,
                "multi_market": e.strategy.multi_market,
                "staleness_flag": e.filter_result.staleness_flag,
                "url": e.paper.url,
                "signal": e.strategy.signal_description[:80],
            })

    passed_csv = out_dir / f"strategies_passed_{timestamp}.csv"
    if passed_rows:
        pd.DataFrame(passed_rows).to_csv(passed_csv, index=False)
    passed_latest = out_dir / "strategies_passed_latest.csv"
    if passed_rows:
        pd.DataFrame(passed_rows).to_csv(passed_latest, index=False)

    # Rejected papers CSV (audit trail)
    rejected_rows = []
    for e in entries:
        if not e.filter_result.passed:
            rejected_rows.append({
                "paper_id": e.paper.paper_id,
                "title": e.paper.title,
                "date": e.paper.publication_date,
                "source": e.paper.source,
                "verdict": e.filter_result.verdict.value,
                "rejection_reason": e.filter_result.rejection_reason,
                "url": e.paper.url,
            })

    rejected_csv = out_dir / f"papers_rejected_{timestamp}.csv"
    if rejected_rows:
        pd.DataFrame(rejected_rows).to_csv(rejected_csv, index=False)
    rejected_latest = out_dir / "papers_rejected_latest.csv"
    if rejected_rows:
        pd.DataFrame(rejected_rows).to_csv(rejected_latest, index=False)

    outputs = {
        "catalogue_json": catalogue_json,
        "catalogue_latest": latest_json,
        "strategies_csv": passed_csv,
        "rejected_csv": rejected_csv,
    }

    print(f"\nCatalogue saved to {out_dir}/")
    for label, path in outputs.items():
        print(f"  {label}: {path.name}")

    return outputs


def print_summary(entries: list[CatalogueEntry]) -> str:
    """Generate a human-readable summary report."""
    passed = [e for e in entries if e.filter_result.passed]
    rejected = [e for e in entries if not e.filter_result.passed]

    lines = [
        "",
        "=" * 80,
        "PAPER PIPELINE SUMMARY",
        "=" * 80,
        f"Total papers discovered: {len(entries)}",
        f"Passed all filters:     {len(passed)}",
        f"Rejected:               {len(rejected)}",
        "",
    ]

    # Rejection breakdown
    from collections import Counter
    verdicts = Counter(e.filter_result.verdict.value for e in rejected)
    lines.append("Rejection breakdown:")
    for verdict, count in verdicts.most_common():
        lines.append(f"  {verdict}: {count}")

    # Passed strategies table
    if passed:
        lines.append("")
        lines.append("-" * 80)
        lines.append("PASSED STRATEGIES")
        lines.append("-" * 80)
        lines.append(
            f"{'#':<4s} {'Type':<16s} {'Assets':<14s} {'Title':<44s}"
        )
        lines.append("-" * 80)
        for i, e in enumerate(passed, 1):
            stype = e.strategy.strategy_type.value if e.strategy else "?"
            assets = ", ".join(e.paper.asset_classes)[:13]
            title = e.paper.title[:43]
            lines.append(f"{i:<4d} {stype:<16s} {assets:<14s} {title}")

        # Flag staleness
        stale = [e for e in passed if e.filter_result.staleness_flag]
        if stale:
            lines.append("")
            lines.append("STALENESS WARNINGS:")
            for e in stale:
                reason = e.filter_result.filter_scores.get("staleness", {}).get("reason", "")
                lines.append(f"  - {e.paper.title[:60]}: {reason[:60]}")

    report = "\n".join(lines)
    print(report)
    return report


def load_latest_catalogue(out_dir: Path = DEFAULT_OUT_DIR) -> list[CatalogueEntry] | None:
    """Load the most recent catalogue from disk for incremental updates."""
    latest = out_dir / "catalogue_latest.json"
    if not latest.exists():
        return None

    data = json.loads(latest.read_text())
    # Return raw dicts for dedup - full deserialization not needed for ID checks
    return data.get("entries", [])


def get_known_paper_ids(out_dir: Path = DEFAULT_OUT_DIR) -> set[str]:
    """Get set of paper IDs already in the catalogue (for dedup)."""
    latest = out_dir / "catalogue_latest.json"
    if not latest.exists():
        return set()

    data = json.loads(latest.read_text())
    return {e["paper"]["paper_id"] for e in data.get("entries", [])}
