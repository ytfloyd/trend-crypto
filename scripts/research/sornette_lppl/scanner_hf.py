"""
High-frequency (hourly) bubble scanner.

Two-stage detection:
  Stage 1 (fast, every hour):  Super-exponential convexity on 12/24/48/72h windows.
  Stage 2 (on trigger only):   Full LPPLS fit for tc estimation + confirmation.

The tc estimate from Stage 2 drives exit timing — the core edge of LPPLS
over a simple convexity screen.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .lppl import fit_lppl, LPPLFit


# -----------------------------------------------------------------------
# Stage 1: Fast convexity + burst (runs every hour, ~0.1ms per symbol)
# -----------------------------------------------------------------------
FAST_WINDOWS_HOURS = [12, 24, 48, 72]
MIN_MOVE_PCT = 5.0  # only flag moves > 5% over the window


def _quad_fit(log_p: np.ndarray) -> tuple[float, float, float]:
    """Fit a + b*t + c*t^2 to log_p.  Returns (c, r2, b)."""
    n = len(log_p)
    if n < 6:
        return 0.0, 0.0, 0.0
    t = np.arange(n, dtype=float)
    coeffs = np.polyfit(t, log_p, 2)
    c, b, a = coeffs
    fitted = np.polyval(coeffs, t)
    ss_res = np.sum((log_p - fitted) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    return float(c), float(r2), float(b)


def fast_scan_single(
    log_prices: np.ndarray,
    windows: list[int] | None = None,
    min_move_pct: float = MIN_MOVE_PCT,
) -> dict:
    """Fast super-exponential detection at the last hourly bar.

    Returns dict with keys:
        triggered : bool — whether the fast layer fires
        se_score  : float — aggregate super-exponential score
        accel     : float — mean quadratic coefficient
        burst     : float — return burst z-score
        best_window : int — window with strongest signal
        move_pct  : float — total move over best window (%)
    """
    if windows is None:
        windows = FAST_WINDOWS_HOURS

    best_score = 0.0
    best_window = 0
    best_move = 0.0
    scores = []

    for w in windows:
        if len(log_prices) < w:
            continue
        seg = log_prices[-w:]

        # Amplitude filter: only consider if the move is big enough
        move_pct = (np.exp(seg[-1] - seg[0]) - 1) * 100
        if move_pct < min_move_pct * (w / 24):  # scale threshold by window
            continue

        c, r2, b = _quad_fit(seg)
        if c <= 0:
            continue

        c_norm = c * w * w
        score = c_norm * max(r2, 0.0)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_window = w
            best_move = move_pct

    # Burst detector on last 6 hours vs prior 24
    burst = 0.0
    if len(log_prices) >= 30:
        rets = np.diff(log_prices)
        recent = rets[-6:]
        hist = rets[-30:-6]
        mu = np.mean(hist)
        sigma = np.std(hist)
        if sigma > 1e-8:
            z = (np.mean(recent) - mu) / sigma
            burst = max(z - 1.5, 0.0) / 3.0

    combined = 0.7 * (np.mean(scores) if scores else 0.0) + 0.3 * burst
    triggered = combined > 0.05 and len(scores) >= 1

    return {
        "triggered": triggered,
        "se_score": combined,
        "accel": np.mean([s for s in scores]) if scores else 0.0,
        "burst": burst,
        "best_window": best_window,
        "move_pct": best_move,
        "n_convex": len(scores),
    }


# -----------------------------------------------------------------------
# Stage 2: LPPLS confirmation + tc estimate (runs only when Stage 1 fires)
# -----------------------------------------------------------------------
LPPL_WINDOWS_HOURS = [48, 96, 168]  # 2d, 4d, 7d


def lppl_confirm(
    log_prices: np.ndarray,
    windows: list[int] | None = None,
) -> dict:
    """Run LPPLS on triggered asset. Returns tc estimate + confidence.

    Parameters
    ----------
    log_prices : np.ndarray
        Hourly log-price history up to current bar.
    windows : list[int]
        Trailing window lengths in hours.

    Returns
    -------
    dict with keys:
        confirmed : bool — LPPLS fit is valid
        tc_hours  : float — estimated hours until critical time
        bubble_conf : float — fit quality score
        best_fit  : LPPLFit | None
    """
    if windows is None:
        windows = LPPL_WINDOWS_HOURS

    best_fit: LPPLFit | None = None
    best_score = 0.0
    tc_estimates = []

    for w in windows:
        if len(log_prices) < w:
            continue
        seg = log_prices[-w:]
        t = np.arange(len(seg), dtype=float)

        fit = fit_lppl(
            seg, t,
            tc_range=(1, w),  # tc within the window horizon
            n_tc=12,
            bubble_type="positive",
            refine=True,
        )

        if not fit.converged or fit.r_squared < 0.3:
            continue

        if not fit.is_valid(min_damping=0.3, max_osc_ratio=1.5):
            continue

        tc_remaining = fit.tc - (w - 1)
        if tc_remaining <= 0:
            continue

        # Quality score
        r2 = max(fit.r_squared, 0)
        damping_bonus = min(fit.damping, 2.0) / 2.0
        score = r2 * damping_bonus

        if score > best_score:
            best_score = score
            best_fit = fit

        tc_estimates.append(tc_remaining)

    if best_fit is None:
        return {
            "confirmed": False,
            "tc_hours": float("nan"),
            "bubble_conf": 0.0,
            "best_fit": None,
        }

    tc_median = float(np.median(tc_estimates)) if tc_estimates else float("nan")

    return {
        "confirmed": True,
        "tc_hours": tc_median,
        "bubble_conf": best_score,
        "best_fit": best_fit,
    }


# -----------------------------------------------------------------------
# Full scanner: sweep all symbols at a single hourly timestamp
# -----------------------------------------------------------------------
def scan_universe_at_hour(
    panel: pd.DataFrame,
    eval_ts: pd.Timestamp,
    min_history_hours: int = 72,
) -> pd.DataFrame:
    """Scan all symbols at a single hourly timestamp.

    Parameters
    ----------
    panel : pd.DataFrame
        Hourly data (symbol, ts, close), sorted by (symbol, ts).
    eval_ts : pd.Timestamp
        The timestamp to evaluate.
    min_history_hours : int
        Minimum bars before evaluation.

    Returns
    -------
    pd.DataFrame with one row per triggered symbol.
    """
    results = []
    symbols = panel["symbol"].unique()

    for sym in symbols:
        sdf = panel[(panel["symbol"] == sym) & (panel["ts"] <= eval_ts)]
        if len(sdf) < min_history_hours:
            continue

        log_p = np.log(sdf["close"].values)

        # Stage 1: Fast scan
        fast = fast_scan_single(log_p)
        if not fast["triggered"]:
            continue

        # Stage 2: LPPLS confirmation
        lppl = lppl_confirm(log_p)

        results.append({
            "symbol": sym,
            "ts": eval_ts,
            "se_score": fast["se_score"],
            "burst": fast["burst"],
            "move_pct": fast["move_pct"],
            "best_window_h": fast["best_window"],
            "n_convex": fast["n_convex"],
            "lppl_confirmed": lppl["confirmed"],
            "tc_hours": lppl["tc_hours"],
            "bubble_conf": lppl["bubble_conf"],
        })

    return pd.DataFrame(results)


def scan_rolling(
    panel: pd.DataFrame,
    eval_every_hours: int = 1,
    start_date: str | None = None,
    min_history_hours: int = 72,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rolling hourly scan across all dates.

    Parameters
    ----------
    panel : pd.DataFrame
        Full hourly panel (symbol, ts, close).
    eval_every_hours : int
        Evaluate every N hours.
    start_date : str
        Start scanning from this date.
    min_history_hours : int
        Minimum hourly history.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame — all triggered signals across time.
    """
    all_hours = sorted(panel["ts"].unique())
    if start_date:
        cutoff = pd.Timestamp(start_date)
        all_hours = [h for h in all_hours if h >= cutoff]

    eval_hours = all_hours[::eval_every_hours]
    if verbose:
        print(f"[scanner_hf] Scanning {len(eval_hours)} timestamps across "
              f"{panel['symbol'].nunique()} symbols ...")

    all_results = []
    t0 = time.time()

    for i, ts in enumerate(eval_hours):
        snap = panel[panel["ts"] <= ts].copy()
        result = scan_universe_at_hour(snap, ts, min_history_hours)
        if not result.empty:
            all_results.append(result)

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            n_sig = sum(len(r) for r in all_results)
            print(f"  [{i+1}/{len(eval_hours)}] {ts} | "
                  f"{n_sig} signals | {elapsed:.0f}s")

    if all_results:
        out = pd.concat(all_results, ignore_index=True)
    else:
        out = pd.DataFrame()

    elapsed = time.time() - t0
    if verbose:
        print(f"[scanner_hf] Done in {elapsed:.0f}s | {len(out)} total signals")
    return out
