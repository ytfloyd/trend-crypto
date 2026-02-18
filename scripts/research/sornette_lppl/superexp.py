"""
Super-exponential growth detection — the *fast* signal layer.

LPPL is precise but slow to converge.  This module detects the
hallmark of a Sornette bubble — **faster-than-exponential growth** —
using a simple quadratic-in-log-price test that fires much earlier
than a full LPPL fit.

Method
------
Fit  ln(p(t)) = a + b·t + c·t²  over a trailing window.

- c > 0  → log-price is *convex*  → super-exponential (bubble-like)
- c < 0  → log-price is *concave* → decelerating (anti-bubble recovery)

Score = c_normalised × R² of quadratic fit.

We also compute a "burst" detector: rolling z-score of returns
identifies sudden explosive moves in their early days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _quad_fit(log_p: np.ndarray) -> tuple[float, float, float]:
    """Fit a + b*t + c*t² to log_p.  Returns (c, r2, b)."""
    n = len(log_p)
    if n < 10:
        return 0.0, 0.0, 0.0
    t = np.arange(n, dtype=float)
    coeffs = np.polyfit(t, log_p, 2)  # [c, b, a]
    c, b, a = coeffs

    fitted = np.polyval(coeffs, t)
    ss_res = np.sum((log_p - fitted) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    return float(c), float(r2), float(b)


def superexp_score(
    log_prices: np.ndarray,
    windows: list[int] | None = None,
) -> dict:
    """Compute super-exponential score at the last observation.

    Parameters
    ----------
    log_prices : np.ndarray
        Full log-price history up to eval date.
    windows : list[int]
        Trailing windows to check.

    Returns
    -------
    dict with keys:
        se_score : float — aggregate super-exponential score [0, 1+]
        accel    : float — mean quadratic coefficient (positive = bubble)
        r2_mean  : float — mean R² of quadratic fits
        n_convex : int   — number of windows showing convexity
    """
    if windows is None:
        windows = [20, 40, 60, 90]

    scores = []
    r2s = []
    accels = []
    n_convex = 0

    for w in windows:
        if len(log_prices) < w:
            continue
        seg = log_prices[-w:]
        c, r2, b = _quad_fit(seg)

        if c > 0:
            n_convex += 1
        # Normalise c by window length to make comparable
        c_norm = c * w * w
        score = max(c_norm, 0.0) * max(r2, 0.0)
        scores.append(score)
        r2s.append(r2)
        accels.append(c_norm)

    if not scores:
        return {"se_score": 0.0, "accel": 0.0, "r2_mean": 0.0, "n_convex": 0}

    return {
        "se_score": float(np.mean(scores)),
        "accel": float(np.mean(accels)),
        "r2_mean": float(np.mean(r2s)),
        "n_convex": n_convex,
    }


def return_burst_score(
    returns: np.ndarray,
    lookback: int = 20,
    z_threshold: float = 1.5,
) -> float:
    """Detect a sudden burst in returns via z-score.

    Returns a score in [0, 1+]: how far recent returns exceed
    the historical norm.  This fires on the *first days* of a jump.
    """
    if len(returns) < lookback + 5:
        return 0.0

    hist = returns[:-5]
    recent = returns[-5:]

    mu = np.mean(hist[-lookback:])
    sigma = np.std(hist[-lookback:])
    if sigma < 1e-8:
        return 0.0

    z = (np.mean(recent) - mu) / sigma
    return max(z - z_threshold, 0.0) / 3.0  # scale to ~[0, 1]


def compute_superexp_panel(
    panel: pd.DataFrame,
    eval_every: int = 5,
    min_history: int = 90,
) -> pd.DataFrame:
    """Compute super-exponential indicators for all symbols.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format [symbol, ts, close]. Must be sorted by (symbol, ts).
    eval_every : int
        Evaluate every N-th day.
    min_history : int
        Minimum days before first evaluation.

    Returns
    -------
    pd.DataFrame
        Columns: [symbol, ts, se_score, accel, r2_mean, n_convex, burst_score].
    """
    df = panel.sort_values(["symbol", "ts"])
    results = []
    symbols = df["symbol"].unique()
    n_sym = len(symbols)

    for i_sym, sym in enumerate(symbols):
        sdf = df[df["symbol"] == sym].reset_index(drop=True)
        if len(sdf) < min_history:
            continue

        close = sdf["close"].values
        log_p = np.log(close)
        rets = np.diff(log_p)  # log returns
        dates = sdf["ts"].values

        eval_indices = list(range(min_history, len(sdf), eval_every))
        if eval_indices and eval_indices[-1] != len(sdf) - 1:
            eval_indices.append(len(sdf) - 1)

        for idx in eval_indices:
            se = superexp_score(log_p[: idx + 1])
            burst = return_burst_score(rets[:idx])

            results.append({
                "symbol": sym,
                "ts": dates[idx],
                "se_score": se["se_score"],
                "accel": se["accel"],
                "r2_mean": se["r2_mean"],
                "n_convex": se["n_convex"],
                "burst_score": burst,
            })

        if (i_sym + 1) % 20 == 0:
            print(f"  [superexp {i_sym+1}/{n_sym}] {sym}")

    out = pd.DataFrame(results)
    if not out.empty:
        out["ts"] = pd.to_datetime(out["ts"])
    return out
