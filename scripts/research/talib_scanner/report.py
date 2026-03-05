"""
Report generation for TA-Lib Edge Scanner.

Produces console output and markdown reports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def print_ic_ranking(
    ranked: pd.DataFrame,
    top_n: int = 40,
    title: str = "IC RANKING",
) -> str:
    """Print and return formatted IC ranking table."""
    lines = []
    lines.append("=" * 100)
    lines.append(title)
    lines.append("=" * 100)

    horizon_col = "best_horizon" if "best_horizon" in ranked.columns else "horizon"

    header = (
        f"{'Rank':>4}  {'Feature':<25s} {'Horizon':>7} {'IC':>8} {'|IC|':>8} "
        f"{'t-stat':>7} {'p-val':>8} {'Hit%':>6} {'Sign':>4} {'Sig?':>4} {'N':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, r in ranked.head(top_n).iterrows():
        sig = "***" if r.get("significant_bh", False) else ""
        rnk = r.get("rank", 0)
        lines.append(
            f"  {rnk:>3}  {r['feature']:<25s} {r[horizon_col]:>5}d "
            f"{r['ic_mean']:>+7.4f} {r['abs_ic_mean']:>7.4f} "
            f"{r['ic_tstat']:>7.2f} {r['ic_pval']:>8.4f} "
            f"{r['hit_rate']:>5.1%} {r['sign']:>4s} {sig:>4s} {r['n_periods']:>6}"
        )

    text = "\n".join(lines)
    print(text)
    return text


def print_ic_by_horizon(
    scan_results: pd.DataFrame,
    features: list[str] | None = None,
    top_n: int = 15,
) -> str:
    """Print IC heatmap: features x horizons."""
    if features is None:
        best = (scan_results.groupby("feature")["abs_ic_mean"].max()
                .sort_values(ascending=False).head(top_n).index.tolist())
        features = best

    sub = scan_results[scan_results["feature"].isin(features)]
    pivot = sub.pivot_table(
        values="ic_mean", index="feature", columns="horizon", aggfunc="first"
    )
    pivot = pivot.reindex(features)

    lines = []
    lines.append("=" * 100)
    lines.append("IC BY HORIZON (top features)")
    lines.append("=" * 100)
    lines.append(pivot.to_string(float_format="{:+.4f}".format))

    text = "\n".join(lines)
    print(text)
    return text


def print_correlation_clusters(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.80,
    features: list[str] | None = None,
) -> str:
    """Identify and print highly correlated feature clusters."""
    if features is not None:
        cols = [c for c in features if c in corr_matrix.columns]
        corr_matrix = corr_matrix.loc[cols, cols]

    lines = []
    lines.append("=" * 100)
    lines.append(f"CORRELATED FEATURE CLUSTERS (|r| > {threshold:.0%})")
    lines.append("=" * 100)

    seen = set()
    clusters = []
    for i, col_i in enumerate(corr_matrix.columns):
        if col_i in seen:
            continue
        cluster = [col_i]
        for j, col_j in enumerate(corr_matrix.columns):
            if j <= i or col_j in seen:
                continue
            if abs(corr_matrix.iloc[i, j]) > threshold:
                cluster.append(col_j)
                seen.add(col_j)
        if len(cluster) > 1:
            clusters.append(cluster)
            seen.add(col_i)

    if clusters:
        for k, cl in enumerate(clusters):
            lines.append(f"  Cluster {k+1}: {', '.join(cl)}")
    else:
        lines.append("  No clusters found above threshold.")

    lines.append(f"\n  Total clusters: {len(clusters)}")
    lines.append(f"  Features in clusters: {sum(len(c) for c in clusters)}")

    text = "\n".join(lines)
    print(text)
    return text


def print_edge_report(analysis: dict) -> str:
    """Print Phase 2 edge analysis report for a single indicator."""
    lines = []
    feat = analysis["feature"]
    sign = analysis["ic_sign"]

    lines.append("=" * 100)
    lines.append(f"EDGE ANALYSIS: {feat} (IC sign: {sign})")
    lines.append("=" * 100)

    # pooled sweep
    pooled = analysis.get("pooled_sweep")
    if pooled is not None and not pooled.empty:
        lines.append(f"\nBest quantile: {analysis.get('best_quantile', 'N/A')}")
        lines.append("\nPooled sweep (avg across assets):")
        p7 = pooled[pooled["fwd_days"] == 7].sort_values("avg_edge", ascending=False)
        lines.append(f"  {'Quantile':>8}  {'#Assets':>7}  {'#Events':>7}  "
                      f"{'AvgMean':>8}  {'AvgHit':>7}  {'AvgEdge':>8}  {'%+Edge':>7}")
        lines.append("  " + "-" * 65)
        for _, r in p7.iterrows():
            lines.append(
                f"  {r['quantile']:>8.2f}  {r['n_assets']:>7}  {r['total_events']:>7}  "
                f"{r['avg_mean']:>+7.2%}  {r['avg_hit']:>6.1%}  "
                f"{r['avg_edge']:>+7.2%}  {r['pct_positive_edge']:>6.0%}"
            )

    # per-asset top/bottom
    per_asset = analysis.get("per_asset")
    if per_asset is not None and not per_asset.empty:
        lines.append(f"\nPer-asset results (7d, best quantile, top 15):")
        lines.append(f"  {'Symbol':<12} {'N':>4} {'Mean':>8} {'Hit%':>6} {'Edge':>8}")
        lines.append("  " + "-" * 45)
        for _, r in per_asset.head(15).iterrows():
            lines.append(f"  {r['symbol']:<12} {r['n']:>4} {r['mean']:>+7.2%} "
                          f"{r['hit']:>5.1%} {r['edge']:>+7.2%}")

    # temporal
    halves = analysis.get("temporal_halves")
    if halves:
        lines.append("\nTemporal stability:")
        h1, h2 = halves["first"], halves["second"]
        lines.append(f"  First half  ({h1['start']} to {h1['end']}, n={h1['n']}): "
                      f"mean={h1['mean']:+.2%}, hit%={h1['hit']:.1%}")
        lines.append(f"  Second half ({h2['start']} to {h2['end']}, n={h2['n']}): "
                      f"mean={h2['mean']:+.2%}, hit%={h2['hit']:.1%}")

    yearly = analysis.get("temporal_yearly", {})
    if yearly:
        lines.append("\n  Year-by-year:")
        for year, vals in sorted(yearly.items()):
            lines.append(f"    {year}: {vals['n']:>4} events, "
                          f"mean={vals['mean']:+.2%}, hit%={vals['hit']:.1%}")

    # bootstrap
    boot = analysis.get("bootstrap")
    if boot:
        lines.append(f"\nBootstrap significance ({boot['n_events']} events):")
        lines.append(f"  Observed mean: {boot['obs_mean']:+.2%}")
        lines.append(f"  Bootstrap null: {boot['boot_mean']:+.2%} +/- {boot['boot_std']:.2%}")
        lines.append(f"  P-value: {boot['p_value']:.4f}")
        lines.append(f"  Z-score: {boot['z_score']:.2f}")

    # tier analysis
    tiers = analysis.get("tier_analysis", {})
    if tiers:
        lines.append("\nMarket-cap tier analysis:")
        for tier, vals in tiers.items():
            lines.append(
                f"  {tier:<6s}: {vals['n_assets']} assets, "
                f"{vals['total_events']} events, "
                f"avg_edge={vals['avg_edge']:+.2%}, "
                f"avg_hit={vals['avg_hit']:.1%}, "
                f"+edge%={vals['pct_positive']:.0%}"
            )

    text = "\n".join(lines)
    print(text)
    return text


def save_markdown_report(
    sections: list[str],
    path: str | Path,
    title: str = "TA-Lib Edge Scanner Report",
) -> None:
    """Save combined report sections to markdown file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"```\n")
        for section in sections:
            f.write(section)
            f.write("\n\n")
        f.write("```\n")

    print(f"[report] Saved to {path}")
