#!/usr/bin/env python
"""Phase 5 — Regression calibration research.

Tests whether a rolling OLS regression layer between raw alpha signal and
forward returns improves Information Coefficient (IC).

Protocol
--------
1. For each alpha × rolling window combination:
   a. At each timestep t, fit OLS on the trailing `window` cross-sectional
      observations: forward_ret ~ alpha_signal
   b. Use the fitted beta to produce a calibrated signal: signal_cal = beta * signal_raw
   c. Compute IC of the calibrated signal vs forward return
2. Compare raw IC vs calibrated IC across all windows.
3. Report: does calibration *consistently* improve IC? Is beta stable?

Decision rule (CTO spec):
  - Only recommend building a production module if IC consistently improves
    AND beta is stable (low CV of beta across time).

Alphas tested:
  - alpha_008 (high-IC, ~0.052)
  - alpha_025 (medium-IC, ~0.028)

Usage
-----
    python scripts/research/run_regression_calibration_v0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "research" / "regression_calibration"

ALPHA_PANEL_PATH = REPO_ROOT / "artifacts" / "research" / "101_alphas" / "alphas_101_v1_adv10m.parquet"

ALPHAS_TO_TEST = ["alpha_008", "alpha_025"]
ROLLING_WINDOWS = [30, 60, 120, 252]


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load alpha panel and compute forward returns.

    Returns (alpha_panel, forward_returns) as long-format DataFrames.
    """
    if RESEARCH_DIR.as_posix() not in sys.path:
        sys.path.insert(0, str(RESEARCH_DIR))

    from common.data import load_daily_bars

    print("[regcal] Loading alpha panel...", flush=True)
    alphas = pd.read_parquet(ALPHA_PANEL_PATH)
    alphas["ts"] = pd.to_datetime(alphas["ts"]).dt.normalize()

    print("[regcal] Loading daily bars for returns...", flush=True)
    bars = load_daily_bars(start="2023-01-01", end="2025-12-31")
    bars["ts"] = pd.to_datetime(bars["ts"]).dt.normalize()

    close_wide = bars.pivot(index="ts", columns="symbol", values="close").sort_index()
    returns_wide = close_wide.pct_change(fill_method=None).shift(-1)

    return alphas, returns_wide


def rolling_regression_ic(
    alpha_long: pd.DataFrame,
    alpha_col: str,
    returns_wide: pd.DataFrame,
    window: int,
) -> dict:
    """Run rolling regression calibration for a single alpha × window.

    Returns dict with: raw IC series, calibrated IC series, beta series.
    """
    sig_wide = alpha_long.pivot(index="ts", columns="symbol", values=alpha_col).sort_index()

    common_ts = sig_wide.index.intersection(returns_wide.index).sort_values()
    common_sym = sig_wide.columns.intersection(returns_wide.columns)

    sig = sig_wide.reindex(index=common_ts, columns=common_sym)
    ret = returns_wide.reindex(index=common_ts, columns=common_sym)

    raw_ic_list = []
    cal_ic_list = []
    beta_list = []
    timestamps = []

    for i in range(window, len(common_ts)):
        t = common_ts[i]
        train_slice = slice(i - window, i)

        sig_train = sig.iloc[train_slice]
        ret_train = ret.iloc[train_slice]

        sig_flat = sig_train.values.flatten()
        ret_flat = ret_train.values.flatten()

        mask = np.isfinite(sig_flat) & np.isfinite(ret_flat)
        if mask.sum() < 20:
            continue

        slope, intercept, _, _, _ = stats.linregress(sig_flat[mask], ret_flat[mask])

        sig_today = sig.iloc[i].dropna()
        ret_today = ret.iloc[i].dropna()
        common_today = sig_today.index.intersection(ret_today.index)

        if len(common_today) < 3:
            continue

        s_vals = sig_today.loc[common_today].values
        r_vals = ret_today.loc[common_today].values

        if np.std(s_vals) == 0 or np.std(r_vals) == 0:
            continue

        raw_corr, _ = stats.spearmanr(s_vals, r_vals)

        cal_signal = slope * s_vals + intercept
        if np.std(cal_signal) > 0:
            cal_corr, _ = stats.spearmanr(cal_signal, r_vals)
        else:
            cal_corr = raw_corr

        raw_ic_list.append(raw_corr)
        cal_ic_list.append(cal_corr)
        beta_list.append(slope)
        timestamps.append(t)

    return {
        "timestamps": timestamps,
        "raw_ic": np.array(raw_ic_list),
        "cal_ic": np.array(cal_ic_list),
        "beta": np.array(beta_list),
    }


def summarize_results(
    alpha_col: str,
    window: int,
    result: dict,
) -> dict:
    """Compute summary statistics for a single alpha × window result.

    Separates genuine calibration benefit from trivial sign correction by
    comparing |calibrated IC| vs |raw IC|.
    """
    n = len(result["raw_ic"])
    nan_row = {
        "alpha": alpha_col, "window": window, "n_periods": n,
        "raw_ic_mean": np.nan, "abs_raw_ic_mean": np.nan,
        "cal_ic_mean": np.nan, "abs_cal_ic_mean": np.nan,
        "signed_improvement": np.nan, "abs_improvement": np.nan,
        "abs_improvement_pct": np.nan,
        "paired_ttest_pval": np.nan, "abs_paired_ttest_pval": np.nan,
        "beta_mean": np.nan, "beta_std": np.nan, "beta_cv": np.nan,
        "recommendation": "INSUFFICIENT_DATA",
    }
    if n < 10:
        return nan_row

    raw_mean = float(np.mean(result["raw_ic"]))
    cal_mean = float(np.mean(result["cal_ic"]))

    abs_raw = np.abs(result["raw_ic"])
    abs_cal = np.abs(result["cal_ic"])
    abs_raw_mean = float(np.mean(abs_raw))
    abs_cal_mean = float(np.mean(abs_cal))

    signed_improvement = cal_mean - raw_mean
    abs_improvement = abs_cal_mean - abs_raw_mean
    abs_improvement_pct = (abs_improvement / abs_raw_mean * 100) if abs_raw_mean > 0 else np.nan

    _, signed_pval = stats.ttest_rel(result["cal_ic"], result["raw_ic"])
    _, abs_pval = stats.ttest_rel(abs_cal, abs_raw)

    beta_mean = float(np.mean(result["beta"]))
    beta_std = float(np.std(result["beta"], ddof=1))
    beta_cv = abs(beta_std / beta_mean) if abs(beta_mean) > 1e-10 else np.inf

    abs_ic_improves = abs_improvement > 0 and abs_pval < 0.05
    beta_stable = beta_cv < 1.0

    if abs_ic_improves and beta_stable:
        recommendation = "BUILD"
    elif abs_ic_improves:
        recommendation = "MARGINAL (beta unstable)"
    elif abs_improvement > 0:
        recommendation = "MARGINAL (not significant)"
    else:
        recommendation = "DO_NOT_BUILD"

    return {
        "alpha": alpha_col,
        "window": window,
        "n_periods": n,
        "raw_ic_mean": raw_mean,
        "abs_raw_ic_mean": abs_raw_mean,
        "cal_ic_mean": cal_mean,
        "abs_cal_ic_mean": abs_cal_mean,
        "signed_improvement": signed_improvement,
        "abs_improvement": abs_improvement,
        "abs_improvement_pct": abs_improvement_pct,
        "paired_ttest_pval": float(signed_pval),
        "abs_paired_ttest_pval": float(abs_pval),
        "beta_mean": beta_mean,
        "beta_std": beta_std,
        "beta_cv": beta_cv,
        "recommendation": recommendation,
    }


def generate_report(all_summaries: list[dict]) -> str:
    """Generate markdown research report."""
    lines = [
        "# Regression Calibration Research Report",
        "",
        "## Protocol",
        "",
        "Rolling OLS: `forward_ret ~ beta * signal + intercept` over trailing window.",
        "Calibrated signal: `beta * signal_raw + intercept`.",
        "Compare Spearman IC of raw vs calibrated signal.",
        "",
        "**Key distinction**: signed IC improvement may be a trivial sign correction",
        "(negative beta flips the signal). The true test is whether **|calibrated IC|**",
        "exceeds **|raw IC|** — i.e. the regression adds information beyond direction.",
        "",
        "**Decision rule**: recommend production build only if |IC| consistently",
        "improves (paired t-test on absolute ICs, p < 0.05) AND beta is stable (CV < 1.0).",
        "",
        "## Results",
        "",
        "| Alpha | Window | N | Raw IC | |Raw IC| | Cal IC | |Cal IC| | |IC| Δ | |IC| Δ% | p-val | Beta Mean | Beta CV | Rec |",
        "|-------|--------|---|--------|---------|---------|---------|---------|---------|-----------|---------|----|",
    ]

    for s in all_summaries:
        lines.append(
            f"| {s['alpha']} | {s['window']} | {s['n_periods']} | "
            f"{s['raw_ic_mean']:.4f} | {s['abs_raw_ic_mean']:.4f} | "
            f"{s['cal_ic_mean']:.4f} | {s['abs_cal_ic_mean']:.4f} | "
            f"{s['abs_improvement']:+.4f} | {s['abs_improvement_pct']:+.1f}% | "
            f"{s['abs_paired_ttest_pval']:.4f} | {s['beta_mean']:.4f} | "
            f"{s['beta_cv']:.2f} | {s['recommendation']} |"
        )

    build_count = sum(1 for s in all_summaries if s["recommendation"] == "BUILD")
    total = len(all_summaries)

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Negative raw IC indicates the raw signal is anti-correlated with forward",
        "returns. The regression captures this via a negative beta (sign flip).",
        "The true question is: **does the regression add value beyond correcting**",
        "**the sign?**",
        "",
        "## Conclusion",
        "",
    ])

    if build_count == total:
        lines.append(
            "**RECOMMENDATION: BUILD.** |IC| consistently improves across all "
            "alpha/window combinations with stable betas. The regression adds "
            "genuine calibration value beyond sign correction."
        )
    elif build_count > 0:
        lines.append(
            f"**RECOMMENDATION: CONDITIONAL.** |IC| improves in {build_count}/{total} "
            "combinations. Review individual results before deciding."
        )
    else:
        lines.append(
            "**RECOMMENDATION: DO NOT BUILD.** Regression calibration does not "
            "consistently improve |IC| beyond sign correction. If signal direction "
            "is wrong, negate it directly. A full regression layer adds complexity "
            "without sufficient payoff."
        )

    return "\n".join(lines)


def main() -> None:
    alphas, returns_wide = load_panel()

    all_summaries = []

    for alpha_col in ALPHAS_TO_TEST:
        for window in ROLLING_WINDOWS:
            print(f"[regcal] {alpha_col} / window={window}...", flush=True)
            result = rolling_regression_ic(alphas, alpha_col, returns_wide, window)
            summary = summarize_results(alpha_col, window, result)
            all_summaries.append(summary)
            print(
                f"         raw={summary['raw_ic_mean']:.4f}  "
                f"|raw|={summary['abs_raw_ic_mean']:.4f}  "
                f"|cal|={summary['abs_cal_ic_mean']:.4f}  "
                f"|Δ|={summary['abs_improvement']:+.4f}  "
                f"beta_cv={summary['beta_cv']:.2f}  "
                f"-> {summary['recommendation']}",
                flush=True,
            )

    report = generate_report(all_summaries)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "regression_calibration_v0.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[regcal] Report -> {report_path}", flush=True)

    summary_df = pd.DataFrame(all_summaries)
    csv_path = ARTIFACTS_DIR / "regression_calibration_v0.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[regcal] CSV -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
