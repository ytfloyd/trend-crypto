"""
CLI entry point for the TA-Lib Edge Scanner.

Usage:
    python -m talib_scanner --phase 1          # IC scan only
    python -m talib_scanner --phase 2 --top 10 # deep dive on top 10
    python -m talib_scanner --phase all        # full pipeline
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_THIS = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS.parents[2]
_RESEARCH = str(_PROJECT_ROOT / "scripts" / "research")
_SRC = str(_PROJECT_ROOT / "src")
_ROOT = str(_PROJECT_ROOT)

sys.path = [_RESEARCH, _SRC, _ROOT] + [p for p in sys.path if p not in (_RESEARCH, _SRC, _ROOT)]

from common.data import load_daily_bars, filter_universe  # noqa: E402
from talib_scanner.features import compute_all_features  # noqa: E402
from talib_scanner.scanner import run_ic_scan, rank_features, compute_feature_correlation  # noqa: E402
from talib_scanner.edge_analysis import run_edge_analysis  # noqa: E402
from talib_scanner.report import (  # noqa: E402
    print_ic_ranking,
    print_ic_by_horizon,
    print_correlation_clusters,
    print_edge_report,
    save_markdown_report,
)

DEFAULT_CONFIG = {
    "start": "2017-01-01",
    "end": "2026-12-31",
    "min_adv_usd": 500_000,
    "min_history_days": 365,
    "forward_horizons": [1, 3, 5, 7, 10, 14, 21],
    "min_assets_for_ic": 10,
    "phase2_top_n": 10,
    "phase2_forward_days": [1, 3, 5, 7, 10, 14, 21],
    "phase2_quantiles": [0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90],
    "output_dir": "artifacts/research/talib_scanner",
}


def load_config(config_path: str | None) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg:
            cfg.update(user_cfg)
    return cfg


def run_phase1(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Run Phase 1: feature computation + IC scan."""
    print("=" * 80)
    print("PHASE 1: IC SCAN")
    print("=" * 80)

    t0 = time.time()

    print("\n[1/3] Loading data...")
    panel = load_daily_bars(start=cfg["start"], end=cfg["end"])
    panel = filter_universe(
        panel,
        min_adv_usd=cfg["min_adv_usd"],
        min_history_days=cfg["min_history_days"],
    )
    n_sym = panel["symbol"].nunique()
    print(f"  {n_sym} assets after universe filter")

    print("\n[2/3] Computing features...")
    panel, feat_cols = compute_all_features(panel)
    print(f"  {len(feat_cols)} features computed")

    print("\n[3/3] Running IC scan...")
    scan = run_ic_scan(
        panel,
        feat_cols,
        forward_horizons=cfg["forward_horizons"],
        min_assets=cfg["min_assets_for_ic"],
        min_history=cfg["min_history_days"],
    )

    elapsed = time.time() - t0
    print(f"\nPhase 1 complete in {elapsed:.0f}s")

    return panel, scan, feat_cols


def run_phase2(
    panel: pd.DataFrame,
    scan: pd.DataFrame,
    feat_cols: list[str],
    cfg: dict,
) -> list[dict]:
    """Run Phase 2: edge deep-dive on top features."""
    print()
    print("=" * 80)
    print("PHASE 2: EDGE DEEP-DIVE")
    print("=" * 80)

    ranked = rank_features(scan, horizon=7, min_periods=50)
    top_n = cfg["phase2_top_n"]
    top_feats = ranked.head(top_n)

    analyses = []
    for _, row in top_feats.iterrows():
        feat = row["feature"]
        sign = row["sign"]
        print(f"\n  Analyzing: {feat} (IC sign: {sign})...")

        analysis = run_edge_analysis(
            panel,
            feat,
            forward_days=cfg["phase2_forward_days"],
            quantile_thresholds=cfg["phase2_quantiles"],
            ic_sign=sign,
        )
        analyses.append(analysis)

    return analyses


def main():
    parser = argparse.ArgumentParser(description="TA-Lib Edge Scanner")
    parser.add_argument("--phase", default="all", choices=["1", "2", "all"])
    parser.add_argument("--top", type=int, default=None,
                        help="Number of top features for Phase 2")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--save", action="store_true",
                        help="Save markdown report to artifacts/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.top:
        cfg["phase2_top_n"] = args.top

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    report_sections = []

    # ── Phase 1 ───────────────────────────────────────────────────────
    if args.phase in ("1", "all"):
        panel, scan, feat_cols = run_phase1(cfg)

        # rank and report
        ranked_7d = rank_features(scan, horizon=7, min_periods=50)
        ranked_best = rank_features(scan, horizon=None, min_periods=50)

        s1 = print_ic_ranking(ranked_7d, top_n=40, title="IC RANKING (7-day horizon)")
        report_sections.append(s1)

        s2 = print_ic_ranking(ranked_best, top_n=40, title="IC RANKING (best horizon per feature)")
        report_sections.append(s2)

        s3 = print_ic_by_horizon(scan, top_n=20)
        report_sections.append(s3)

        # correlation clusters among top features
        top_feat_names = ranked_7d.head(30)["feature"].tolist()
        valid_feats = [f for f in top_feat_names if f in panel.columns]
        if len(valid_feats) >= 5:
            corr = compute_feature_correlation(panel, valid_feats)
            s4 = print_correlation_clusters(corr, threshold=0.80, features=valid_feats)
            report_sections.append(s4)

        # save scan results
        scan.to_parquet(out_dir / "ic_scan_results.parquet", index=False)
        ranked_7d.to_csv(out_dir / "ic_ranking_7d.csv", index=False)
        print(f"\n[output] Scan results saved to {out_dir}")

    # ── Phase 2 ───────────────────────────────────────────────────────
    if args.phase in ("2", "all"):
        if args.phase == "2":
            # load saved scan results
            scan_path = out_dir / "ic_scan_results.parquet"
            if not scan_path.exists():
                print("ERROR: Run Phase 1 first (no saved scan results found).")
                sys.exit(1)
            scan = pd.read_parquet(scan_path)
            print("Loading data for Phase 2...")
            panel = load_daily_bars(start=cfg["start"], end=cfg["end"])
            panel = filter_universe(
                panel,
                min_adv_usd=cfg["min_adv_usd"],
                min_history_days=cfg["min_history_days"],
            )
            panel, feat_cols = compute_all_features(panel)

        analyses = run_phase2(panel, scan, feat_cols, cfg)

        for a in analyses:
            s = print_edge_report(a)
            report_sections.append(s)

    # ── Save report ───────────────────────────────────────────────────
    if args.save and report_sections:
        save_markdown_report(
            report_sections,
            out_dir / "scanner_report.md",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
