"""
Multi-window bubble confidence indicator.

For each (symbol, date), fits LPPL on several trailing windows and
aggregates fit quality into a single *bubble confidence score*.

Methodology
-----------
1.  For each evaluation date, extract trailing windows of different
    lengths (e.g. 60, 90, 120, 180, 252 days).
2.  Fit LPPL to each window (both positive and anti-bubble).
3.  For each valid fit, compute a quality score:
        q = R² × damping_bonus × stage_bonus
    where
        damping_bonus = min(damping_ratio, 2) / 2      ∈ [0,1]
        stage_bonus = sigmoid(tc_remaining, 30, 0.1)    ∈ [0,1]
            → peaks when tc is ~30 days out (early stage)
4.  The aggregate bubble confidence = mean of top-2 window scores
    (requires at least 2 valid windows for robustness).

Output columns
--------------
    bubble_conf     : float [0,1] — positive-bubble confidence
    antibubble_conf : float [0,1] — anti-bubble confidence (recovery imminent)
    tc_days         : float — median tc estimate (days from eval date)
    lppl_signal     : float [-1, +1] — net signal (positive = ride bubble,
                      negative = anti-bubble reversal expected)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lppl import LPPLFit, fit_lppl

WINDOWS = [60, 120, 252]
EVAL_EVERY = 20  # LPPL is expensive; evaluate infrequently, forward-fill


@dataclass
class BubbleScore:
    """Per-symbol, per-date bubble assessment."""

    bubble_conf: float
    antibubble_conf: float
    tc_days_median: float
    n_valid_windows: int


def _quality_score(fit: LPPLFit, window_len: int) -> float:
    """Scalar quality score for a single LPPL fit."""
    if not fit.converged or fit.r_squared < 0.3:
        return 0.0
    if not fit.is_valid(min_damping=0.3, max_osc_ratio=1.5):
        return 0.0

    r2 = max(fit.r_squared, 0.0)
    damping_bonus = min(fit.damping, 2.0) / 2.0

    # Stage bonus: peaks when tc is ~20-60 days out.
    # tc is in index units.  tc - (window_len - 1) = days remaining after last obs.
    tc_remaining = fit.tc - (window_len - 1)
    stage_bonus = 1.0 / (1.0 + np.exp(-0.1 * (tc_remaining - 30)))
    # Also penalise tc very far out (> 180 days) — probably not a real bubble
    if tc_remaining > 180:
        stage_bonus *= 0.3
    elif tc_remaining > 120:
        stage_bonus *= 0.6

    return r2 * damping_bonus * stage_bonus


def score_single_date(
    log_prices: np.ndarray,
    windows: list[int] | None = None,
) -> BubbleScore:
    """Compute bubble confidence scores at the last date of *log_prices*.

    Parameters
    ----------
    log_prices : np.ndarray
        Full log-price history up to and including the evaluation date.
    windows : list[int]
        Trailing window lengths to use.

    Returns
    -------
    BubbleScore
    """
    if windows is None:
        windows = WINDOWS

    pos_scores: list[float] = []
    neg_scores: list[float] = []
    tc_days_list: list[float] = []

    for w in windows:
        if len(log_prices) < w:
            continue
        seg = log_prices[-w:]
        t = np.arange(len(seg), dtype=float)

        # Positive bubble
        pfit = fit_lppl(seg, t, bubble_type="positive", refine=True)
        ps = _quality_score(pfit, w)
        if ps > 0:
            pos_scores.append(ps)
            tc_remaining = pfit.tc - (w - 1)
            tc_days_list.append(tc_remaining)

        # Anti-bubble (accelerating decline → reversal)
        nfit = fit_lppl(seg, t, bubble_type="negative", refine=True)
        ns = _quality_score(nfit, w)
        if ns > 0:
            neg_scores.append(ns)
            tc_remaining_neg = nfit.tc - (w - 1)
            tc_days_list.append(tc_remaining_neg)

    # Aggregate: mean of top-2 scores (need ≥ 1 valid)
    def _agg(scores: list[float]) -> float:
        if len(scores) < 1:
            return 0.0
        top = sorted(scores, reverse=True)[:2]
        return float(np.mean(top))

    bubble_conf = _agg(pos_scores)
    antibubble_conf = _agg(neg_scores)
    tc_median = float(np.median(tc_days_list)) if tc_days_list else float("nan")

    return BubbleScore(
        bubble_conf=bubble_conf,
        antibubble_conf=antibubble_conf,
        tc_days_median=tc_median,
        n_valid_windows=len(pos_scores) + len(neg_scores),
    )


def compute_bubble_panel(
    panel: pd.DataFrame,
    symbols: list[str] | None = None,
    eval_every: int = EVAL_EVERY,
    windows: list[int] | None = None,
    min_history: int = 252,
) -> pd.DataFrame:
    """Compute bubble indicators for all symbols across time.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format with columns [symbol, ts, close].
        Must be sorted by (symbol, ts).
    symbols : list[str] | None
        Restrict to these symbols (default: all).
    eval_every : int
        Evaluate every N-th trading day to save compute.
    windows : list[int]
        LPPL window lengths.
    min_history : int
        Minimum observations before first evaluation.

    Returns
    -------
    pd.DataFrame
        Columns: [symbol, ts, bubble_conf, antibubble_conf, tc_days, n_valid]
    """
    if windows is None:
        windows = WINDOWS

    df = panel.sort_values(["symbol", "ts"]).copy()
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]

    results = []
    unique_symbols = df["symbol"].unique()
    n_symbols = len(unique_symbols)

    for i_sym, sym in enumerate(unique_symbols):
        sdf = df[df["symbol"] == sym].reset_index(drop=True)
        if len(sdf) < min_history:
            continue

        close = sdf["close"].values
        log_p = np.log(close)
        dates = sdf["ts"].values

        eval_indices = list(range(min_history, len(sdf), eval_every))
        if eval_indices and eval_indices[-1] != len(sdf) - 1:
            eval_indices.append(len(sdf) - 1)

        for idx in eval_indices:
            bs = score_single_date(log_p[: idx + 1], windows=windows)
            results.append({
                "symbol": sym,
                "ts": dates[idx],
                "bubble_conf": bs.bubble_conf,
                "antibubble_conf": bs.antibubble_conf,
                "tc_days": bs.tc_days_median,
                "n_valid": bs.n_valid_windows,
            })

        if (i_sym + 1) % 10 == 0:
            print(f"  [{i_sym+1}/{n_symbols}] {sym}: {len(eval_indices)} eval dates")

    out = pd.DataFrame(results)
    if not out.empty:
        out["ts"] = pd.to_datetime(out["ts"])
    return out
