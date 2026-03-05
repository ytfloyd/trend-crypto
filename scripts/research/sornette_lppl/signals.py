"""
Trading signals: blend LPPL bubble indicator with super-exponential
growth detector for the "Jumpers" portfolio.

Signal architecture
-------------------
The signal has two layers:

**Layer 1 — Fast (super-exponential + burst)**:
    Fires early when price starts accelerating.  Based on convexity of
    log-price and return z-score.  Cheap to compute, high recall.

**Layer 2 — Confirmation (LPPL)**:
    Fires once the bubble pattern is well-formed.  Higher precision,
    sizes the position.  Also provides tc estimate for exit timing.

Combined signal:
    signal = w_fast × fast_score  +  w_lppl × lppl_score

When both layers agree → strongest signal (confirmed jumper).
When only fast fires → early-stage entry (scout position).
When only LPPL fires → mature bubble (ride cautiously).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def blend_signals(
    lppl_panel: pd.DataFrame,
    superexp_panel: pd.DataFrame,
    *,
    w_fast: float = 0.55,
    w_lppl: float = 0.45,
    min_signal: float = 0.02,
    tc_danger_zone: float = 10.0,
) -> pd.DataFrame:
    """Blend LPPL and super-exponential signals into a unified score.

    Parameters
    ----------
    lppl_panel : pd.DataFrame
        Output of ``compute_bubble_panel`` with columns:
        [symbol, ts, bubble_conf, antibubble_conf, tc_days, n_valid].
    superexp_panel : pd.DataFrame
        Output of ``compute_superexp_panel`` with columns:
        [symbol, ts, se_score, burst_score, ...].
    w_fast, w_lppl : float
        Blending weights.
    min_signal : float
        Minimum combined score to emit a signal.
    tc_danger_zone : float
        Dampen LPPL when tc < this many days.

    Returns
    -------
    pd.DataFrame
        Columns: [symbol, ts, signal, signal_type, lppl_score,
                  fast_score, bubble_conf, se_score, tc_days].
    """
    # Merge on (symbol, ts) — allow approximate date matching
    lppl = lppl_panel.copy()
    se = superexp_panel.copy()

    # Normalise dates to date only (drop time component)
    lppl["date"] = pd.to_datetime(lppl["ts"]).dt.normalize()
    se["date"] = pd.to_datetime(se["ts"]).dt.normalize()

    # Merge; use outer to keep all observations
    merged = pd.merge(
        lppl[["symbol", "date", "bubble_conf", "antibubble_conf", "tc_days", "n_valid"]],
        se[["symbol", "date", "se_score", "burst_score", "accel", "n_convex"]],
        on=["symbol", "date"],
        how="outer",
    ).fillna(0.0)

    # --- LPPL score ---
    rider = merged["bubble_conf"].copy()
    tc = merged["tc_days"].copy()

    # Dampen near tc (exit zone)
    danger_mask = (tc > 0) & (tc < tc_danger_zone)
    rider = rider.where(~danger_mask, rider * 0.2)

    # Anti-bubble reversal (boost when tc is imminent → snap-back)
    ab = merged["antibubble_conf"].copy()
    reversal_boost = np.clip(1.0 - tc / 60.0, 0.0, 1.0)
    ab_score = ab * reversal_boost

    lppl_score = np.maximum(rider, ab_score)

    # --- Fast score ---
    se_score = merged["se_score"].clip(lower=0.0)
    burst = merged["burst_score"].clip(lower=0.0)
    fast_score = 0.7 * se_score + 0.3 * burst

    # Normalise each to [0, 1]
    def _norm(s: pd.Series) -> pd.Series:
        mx = s.quantile(0.99) if len(s) > 10 else s.max()
        if mx > 1e-10:
            return (s / mx).clip(0, 1)
        return s

    lppl_norm = _norm(lppl_score)
    fast_norm = _norm(fast_score)

    # --- Combined signal ---
    combined = w_fast * fast_norm + w_lppl * lppl_norm
    combined = combined.where(combined >= min_signal, 0.0)

    # Classify
    conditions = [
        (fast_norm > lppl_norm) & (combined > min_signal) & (merged["accel"] > 0),
        (lppl_norm >= fast_norm) & (combined > min_signal) & (rider > ab_score),
        (lppl_norm >= fast_norm) & (combined > min_signal) & (ab_score >= rider),
    ]
    choices = ["super_exponential", "bubble_rider", "antibubble_reversal"]
    signal_type = np.select(conditions, choices, default="none")

    merged["signal"] = combined
    merged["signal_type"] = signal_type
    merged["lppl_score"] = lppl_norm
    merged["fast_score"] = fast_norm
    merged["ts"] = merged["date"]

    out_cols = [
        "symbol", "ts", "signal", "signal_type",
        "lppl_score", "fast_score",
        "bubble_conf", "antibubble_conf", "se_score", "burst_score",
        "tc_days", "n_valid", "n_convex",
    ]
    return merged[[c for c in out_cols if c in merged.columns]].copy()


# ---------------------------------------------------------------------------
# Legacy single-source signal (LPPL only) — kept for backward compat
# ---------------------------------------------------------------------------
def compute_signals(
    bubble_panel: pd.DataFrame,
    *,
    min_confidence: float = 0.02,
    tc_danger_zone: float = 10.0,
) -> pd.DataFrame:
    """Turn LPPL-only bubble indicators into trading signals.

    For backward compatibility with run_bubble_scan.
    """
    df = bubble_panel.copy()

    rider = df["bubble_conf"].fillna(0.0).copy()
    tc = df["tc_days"].fillna(100.0)
    danger_mask = tc < tc_danger_zone
    rider = rider.where(~danger_mask, rider * 0.2)

    ab = df["antibubble_conf"].fillna(0.0).copy()
    reversal_boost = np.clip(1.0 - tc / 60.0, 0.0, 1.0)
    ab_signal = ab * reversal_boost

    raw = np.maximum(rider, ab_signal)
    raw = raw.where(raw >= min_confidence, 0.0)
    signal = raw / max(raw.max(), 1e-10)

    df["signal_raw"] = raw
    df["signal"] = signal

    conditions = [
        (rider > ab_signal) & (raw > min_confidence),
        (ab_signal >= rider) & (raw > min_confidence),
    ]
    choices = ["bubble_rider", "antibubble_reversal"]
    df["signal_type"] = np.select(conditions, choices, default="none")

    return df
