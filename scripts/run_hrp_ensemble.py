#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import polars as pl

from portfolio.hrp import HierarchicalRiskParity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HRP ensemble runner for Survivor Protocol outputs")
    p.add_argument("--survivors_csv", required=True, help="Path to gatekeeper_survivors.csv")
    p.add_argument("--survivor_dir", required=True, help="Path to survivors/ directory")
    p.add_argument("--output_dir", default="artifacts/ensemble/hrp_v0", help="Output directory for HRP artifacts")
    p.add_argument("--min_survivors", type=int, default=2, help="Minimum survivors required")
    p.add_argument("--cost_bps", type=float, default=0.0, help="Cost bps (reserved)")
    p.add_argument("--method", default="hrp", choices=["hrp"], help="Allocation method")
    p.add_argument("--title", default="HRP Ensemble Report", help="Report title")
    return p.parse_args()


def _load_survivors(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "alpha" in df.columns:
        return df["alpha"].dropna().astype(str).tolist()
    if "alpha_name" in df.columns:
        return df["alpha_name"].dropna().astype(str).tolist()
    raise ValueError("No alpha or alpha_name column found in survivors CSV.")


def _load_metadata(tearsheet_dir: Path, alpha: str) -> dict:
    meta_path = tearsheet_dir / alpha / f"{alpha}.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _load_spread_returns(survivor_dir: Path, alpha: str) -> pd.Series:
    spread_path = survivor_dir / alpha / f"{alpha}.spread_returns.parquet"
    if not spread_path.exists():
        raise FileNotFoundError(
            f"Missing spread return series for {alpha}. "
            "Re-run survivor tearsheets with --emit-returns."
        )
    df = pl.read_parquet(spread_path)
    if not {"ts", "spread_ret"}.issubset(set(df.columns)):
        raise ValueError(f"spread_returns parquet missing required columns: {spread_path}")
    pdf = df.to_pandas()
    pdf["ts"] = pd.to_datetime(pdf["ts"])
    return pdf.set_index("ts")["spread_ret"].rename(alpha)


def build_returns_df(survivor_dir: str, survivors: list[str]) -> pd.DataFrame:
    series = []
    base = Path(survivor_dir)
    for alpha in survivors:
        series.append(_load_spread_returns(base, alpha))
    returns = pd.concat(series, axis=1, join="inner")
    if returns.empty:
        raise ValueError("No overlapping timestamps across survivors.")
    return returns


def _corr_summary(corr: pd.DataFrame) -> dict:
    vals = corr.values
    mask = ~np.eye(vals.shape[0], dtype=bool)
    if vals.shape[0] <= 1:
        return {"max_corr": None, "median_corr": None, "top_pairs": []}
    off_diag = vals[mask]
    max_corr = float(np.nanmax(off_diag)) if off_diag.size else None
    median_corr = float(np.nanmedian(off_diag)) if off_diag.size else None
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:3]
    return {
        "max_corr": max_corr,
        "median_corr": median_corr,
        "top_pairs": pairs,
    }


def _write_report(
    *,
    output_dir: Path,
    title: str,
    survivors: list[str],
    meta_map: dict[str, dict],
    weights: pd.Series,
    corr_summary: dict,
    date_range: list[str],
    linkage_order: list[str],
) -> None:
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Date range: {date_range[0]} → {date_range[1]}")
    lines.append(f"- N survivors: {len(survivors)}")
    lines.append("- Method: HRP")
    lines.append(
        "- Returns used for HRP are canonical Q5−Q1 spread returns from tearsheet artifacts."
    )
    lines.append(
        f"- Correlation summary: max={corr_summary['max_corr']}, median={corr_summary['median_corr']}"
    )
    lines.append("")
    lines.append("## Survivor Table")
    lines.append("| alpha | mean_ic | spread_sharpe | mean_daily_turnover | verdict | cluster_order |")
    lines.append("|---|---|---|---|---|---|")
    for alpha in survivors:
        meta = meta_map.get(alpha, {})
        mean_ic = meta.get("mean_ic", "n/a")
        spread_sharpe = meta.get("spread_sharpe", "n/a")
        turnover = meta.get("mean_daily_turnover", "n/a")
        verdict = "PASS"
        if isinstance(turnover, (int, float)) and turnover > 0.40:
            verdict = "PROBATION"
        cluster_index = linkage_order.index(alpha) if alpha in linkage_order else "n/a"
        lines.append(
            f"| {alpha} | {mean_ic} | {spread_sharpe} | {turnover} | {verdict} | {cluster_index} |"
        )
    lines.append("")
    lines.append("## Correlation Check")
    lines.append(f"- max pairwise corr: {corr_summary['max_corr']}")
    lines.append(f"- median corr: {corr_summary['median_corr']}")
    if corr_summary["top_pairs"]:
        lines.append("- top correlated pairs:")
        for a, b, c in corr_summary["top_pairs"]:
            lines.append(f"  - {a} vs {b}: {c:.4f}")
    lines.append("")
    lines.append("## HRP Weights")
    lines.append("| alpha | weight |")
    lines.append("|---|---|")
    for alpha, w in weights.sort_values(ascending=False).items():
        lines.append(f"| {alpha} | {w:.6f} |")
    lines.append("")
    lines.append(f"Sum(weights)={weights.sum():.6f}")
    lines.append("")
    lines.append("Dendrogram: hrp_dendrogram.png")

    report_path = output_dir / "hrp_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_hrp_ensemble(
    *,
    survivors_csv: str,
    survivor_dir: str,
    output_dir: str,
    min_survivors: int,
    method: str,
    title: str,
) -> dict:
    survivors = _load_survivors(survivors_csv)
    if len(survivors) < min_survivors:
        raise SystemExit("Cannot ensemble <2 alphas.")

    meta_map = {}
    base = Path(survivor_dir)
    for alpha in survivors:
        meta_map[alpha] = _load_metadata(base, alpha)

    returns = build_returns_df(survivor_dir, survivors)
    weights = HierarchicalRiskParity.allocate(returns)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_df = weights.reset_index()
    weights_df.columns = ["alpha", "weight"]
    weights_df.to_csv(out_dir / "hrp_weights.csv", index=False)

    corr = returns.corr()
    cov = returns.cov()
    corr.to_csv(out_dir / "hrp_corr.csv")
    cov.to_csv(out_dir / "hrp_cov.csv")

    link = HierarchicalRiskParity.get_linkage(corr)
    sort_ix = HierarchicalRiskParity.get_quasi_diag(link)
    linkage_order = [corr.columns[i] for i in sort_ix]

    fig, ax = plt.subplots(figsize=(8, 4))
    from scipy.cluster.hierarchy import dendrogram

    dendrogram(link, labels=linkage_order, ax=ax)
    ax.set_title("HRP Dendrogram")
    fig.tight_layout()
    fig.savefig(out_dir / "hrp_dendrogram.png", dpi=150)
    plt.close(fig)

    corr_summary = _corr_summary(corr)
    meta = {
        "n_survivors": int(len(survivors)),
        "date_range": [str(returns.index.min()), str(returns.index.max())],
        "method": method,
        "survivors_csv": str(survivors_csv),
        "survivor_dir": str(survivor_dir),
        "corr_summary": {
            "max_corr": corr_summary["max_corr"],
            "median_corr": corr_summary["median_corr"],
        },
    }
    with open(out_dir / "hrp_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    _write_report(
        output_dir=out_dir,
        title=title,
        survivors=survivors,
        meta_map=meta_map,
        weights=weights,
        corr_summary=corr_summary,
        date_range=meta["date_range"],
        linkage_order=linkage_order,
    )

    return {
        "weights": weights,
        "returns": returns,
        "output_dir": out_dir,
    }


def main() -> None:
    args = parse_args()
    result = run_hrp_ensemble(
        survivors_csv=args.survivors_csv,
        survivor_dir=args.survivor_dir,
        output_dir=args.output_dir,
        min_survivors=args.min_survivors,
        method=args.method,
        title=args.title,
    )

    weights = result["weights"]
    returns = result["returns"]
    corr = returns.corr()
    corr_summary = _corr_summary(corr)
    print("=" * 72)
    print("HRP Ensemble")
    print("=" * 72)
    print(
        f"n_survivors={weights.shape[0]}, date_range={returns.index.min()} → {returns.index.max()}"
    )
    print(
        f"corr max={corr_summary['max_corr']}, median={corr_summary['median_corr']}"
    )
    for alpha, w in weights.items():
        print(f"{alpha}: {w:.6f}")
    print(f"weights_sum={weights.sum():.6f}")


if __name__ == "__main__":
    main()
