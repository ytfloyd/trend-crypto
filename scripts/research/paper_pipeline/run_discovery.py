#!/usr/bin/env python
"""Run the paper discovery -> filter -> extract -> catalogue pipeline.

Usage:
    python -m scripts.research.paper_pipeline.run_discovery [OPTIONS]

Options:
    --arxiv-max INT     Max results per arXiv query (default: 25)
    --ssrn-pages INT    Max SSRN search pages per query (default: 1)
    --out-dir PATH      Output directory (default: artifacts/research/paper_pipeline)
    --incremental       Skip papers already in catalogue
    --arxiv-only        Skip SSRN (faster for testing)
    --summary-only      Just print summary of latest catalogue
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.paper_pipeline.catalogue import (
    build_catalogue,
    get_known_paper_ids,
    print_summary,
    save_catalogue,
)
from scripts.research.paper_pipeline.discovery import (
    classify_paper,
    discover_arxiv,
    discover_ssrn,
)
from scripts.research.paper_pipeline.extractor import run_extraction
from scripts.research.paper_pipeline.filters import run_filters
from scripts.research.paper_pipeline.methodology_audit import run_methodology_audit
from scripts.research.paper_pipeline.models import PaperMeta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Academic paper discovery and strategy cataloguing pipeline"
    )
    parser.add_argument("--arxiv-max", type=int, default=25, help="Max results per arXiv query")
    parser.add_argument("--ssrn-pages", type=int, default=1, help="Max SSRN pages per query")
    parser.add_argument(
        "--out-dir", type=str, default="artifacts/research/paper_pipeline",
        help="Output directory",
    )
    parser.add_argument("--incremental", action="store_true", help="Skip already-catalogued papers")
    parser.add_argument("--arxiv-only", action="store_true", help="Skip SSRN discovery")
    parser.add_argument("--summary-only", action="store_true", help="Print latest catalogue summary")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.summary_only:
        from scripts.research.paper_pipeline.catalogue import load_latest_catalogue
        raw = load_latest_catalogue(out_dir)
        if raw is None:
            print("No catalogue found. Run discovery first.")
            return
        print(f"Loaded {len(raw)} entries from latest catalogue.")
        return

    # --- Phase 1: Discovery ---
    print("=" * 70)
    print("PAPER DISCOVERY PIPELINE")
    print("=" * 70)

    papers: list[PaperMeta] = []

    print("\n--- Phase 1a: arXiv Discovery ---")
    arxiv_papers = discover_arxiv(max_results_per_query=args.arxiv_max)
    papers.extend(arxiv_papers)

    if not args.arxiv_only:
        print("\n--- Phase 1b: SSRN Discovery ---")
        ssrn_papers = discover_ssrn(max_pages=args.ssrn_pages)
        papers.extend(ssrn_papers)

    # Classify all papers
    print(f"\nClassifying {len(papers)} papers ...")
    papers = [classify_paper(p) for p in papers]

    # Incremental dedup
    if args.incremental:
        known = get_known_paper_ids(out_dir)
        before = len(papers)
        papers = [p for p in papers if p.paper_id not in known]
        print(f"Incremental mode: {before} -> {len(papers)} new papers (skipped {before - len(papers)} known)")

    if not papers:
        print("No new papers to process.")
        return

    # --- Phase 2a: Filter stack ---
    passed, rejected = run_filters(papers)

    # --- Phase 2b: Methodology audit (plausibility gate) ---
    if passed:
        audit_passed, audit_rejected = run_methodology_audit(passed)
        rejected.extend(audit_rejected)
        passed = audit_passed

    # --- Phase 3: Extract ---
    if passed:
        extracted = run_extraction(passed)
    else:
        extracted = []
        print("\nNo papers survived filters + audit. Nothing to extract.")

    # --- Phase 4: Catalogue ---
    entries = build_catalogue(extracted, rejected)

    outputs = save_catalogue(entries, out_dir)
    report = print_summary(entries)

    # Save report
    report_path = out_dir / "latest_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
