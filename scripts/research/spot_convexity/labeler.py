"""Stop-aware R-multiple labeler — the core primitive of the spot-convexity sleeve.

Given a single asset's daily OHLCV and a signal bar, simulate a long trade with a fixed
ATR-based initial stop and an upward-only ATR trailing stop, and return the realized
R-multiple plus the full set of target variables (Deliverable #3 of the research brief).

Execution semantics (NON-NEGOTIABLE — see docs/research/spot_convexity/01_research_design.md §4):
  * Entry at the NEXT bar's open after the signal bar (no same-bar look-ahead).
  * Initial risk per unit  R = entry - initial_stop = stop_mult * atr_ref, where atr_ref is the
    ATR known AT THE SIGNAL DATE (passed in by the caller; the labeler never recomputes it from
    forward data).
  * Per bar, in order:
      1. GAP-THROUGH: if the bar OPENS at/below the current stop -> exit at the OPEN (realized loss
         may exceed -1R). This is the spot framework's left-tail leak; it must be modeled.
      2. INTRADAY BREACH: else if the bar's LOW <= current stop -> exit AT the stop price.
      3. otherwise update the high-water mark and ratchet the trailing stop UP only.
  * Time stop: at max_horizon bars held, exit at that bar's close.
  * Trailing stop never moves down.
  * Insufficient forward history (trade neither exits nor completes max_horizon) -> incomplete=True
    (right-censored), reported but excludable.

The labeler is PURE (numpy in, dict out) and has no look-ahead by construction: it only ever reads
bars at and after entry to compute the realized outcome (the label), which is allowed.
"""
from __future__ import annotations

import numpy as np

GAP_STOP, INTRADAY_STOP, TRAIL_STOP, TIME_STOP = "gap_stop", "intraday_stop", "trail_stop", "time_stop"


def label_trade(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    signal_idx: int,
    atr_ref: float,
    *,
    stop_mult: float = 2.0,
    trail_mult: float = 3.0,
    max_horizon: int = 60,
    pos_threshold_R: float = 2.0,
) -> dict:
    """Label one long trade. Arrays are 1-D, chronological, same length.

    Parameters
    ----------
    signal_idx : index of the SIGNAL bar; entry is at open[signal_idx + 1].
    atr_ref    : ATR known at the signal bar (sets the risk unit). Must be > 0.
    stop_mult  : initial stop = entry - stop_mult*atr_ref  (risk unit R = stop_mult*atr_ref).
    trail_mult : trailing stop = high_watermark - trail_mult*atr_ref (only ratchets up).
    max_horizon: max bars held after entry.
    pos_threshold_R : R level defining the positive-convexity label.

    Returns a dict of target variables (R-multiples, labels, excursions, timings).
    """
    n = len(close)
    entry_idx = signal_idx + 1
    incomplete_blank = {
        "incomplete": True, "valid": False, "r_multiple": np.nan, "exit_reason": None,
        "entry_idx": entry_idx, "exit_idx": None, "entry_price": np.nan,
        "initial_stop": np.nan, "risk_per_unit": np.nan,
        "mfe_R": np.nan, "mae_R": np.nan, "mfe_minus_mae_R": np.nan,
        "time_to_stop": np.nan, "time_to_peak": np.nan,
        "reached_1R": False, "reached_2R": False, "stopped_before_1R": False,
        "stop_out_loss": False, "positive_convexity": False, "bars_held": 0,
    }
    if not np.isfinite(atr_ref) or atr_ref <= 0:
        return incomplete_blank
    if entry_idx >= n:  # no bar to enter on
        return incomplete_blank

    entry = float(open_[entry_idx])
    if not np.isfinite(entry) or entry <= 0:
        return incomplete_blank
    risk = stop_mult * atr_ref            # R unit (per share/coin)
    stop = entry - risk                    # initial hard stop
    hwm = entry                            # high-water mark (for trailing + MFE)
    lwm = entry                            # low-water mark (for MAE)

    last_idx = min(entry_idx + max_horizon, n - 1)
    exit_idx, exit_price, exit_reason = None, None, None

    for t in range(entry_idx, last_idx + 1):
        o, h, l = float(open_[t]), float(high[t]), float(low[t])
        # excursions are tracked on bars actually experienced (through exit bar)
        hwm = max(hwm, h)
        lwm = min(lwm, l)

        # 1. gap-through at the open (skip the entry bar's own open == entry)
        if t > entry_idx and o <= stop:
            exit_idx, exit_price, exit_reason = t, o, GAP_STOP
            break
        # 2. intraday breach of the (possibly trailed) stop
        if l <= stop:
            exit_idx, exit_price = t, stop
            exit_reason = TRAIL_STOP if stop > entry - risk + 1e-12 else INTRADAY_STOP
            break
        # 3. ratchet trailing stop UP only
        stop = max(stop, hwm - trail_mult * atr_ref)
        # 4. time stop
        if t == entry_idx + max_horizon:
            exit_idx, exit_price, exit_reason = t, float(close[t]), TIME_STOP
            break

    if exit_idx is None:
        # ran out of data before exiting or completing the horizon -> right-censored
        out = dict(incomplete_blank)
        out["entry_price"] = entry
        out["initial_stop"] = entry - risk
        out["risk_per_unit"] = risk
        return out

    # round to 6 dp: kills floating-point cancellation noise (e.g. an at-stop exit computing
    # to -1.0000000001) so downstream threshold comparisons like "< -1R" are exact.
    r_multiple = round((exit_price - entry) / risk, 6)
    mfe_R = round((hwm - entry) / risk, 6)
    mae_R = round((lwm - entry) / risk, 6)
    bars_held = exit_idx - entry_idx
    return {
        "incomplete": False, "valid": True,
        "r_multiple": float(r_multiple),
        "exit_reason": exit_reason,
        "entry_idx": entry_idx, "exit_idx": exit_idx,
        "entry_price": entry, "initial_stop": entry - risk, "risk_per_unit": risk,
        "mfe_R": float(mfe_R), "mae_R": float(mae_R),
        "mfe_minus_mae_R": float(mfe_R - mae_R),
        "time_to_stop": int(bars_held) if exit_reason in (GAP_STOP, INTRADAY_STOP, TRAIL_STOP) else np.nan,
        "time_to_peak": np.nan,  # filled below
        "reached_1R": bool(mfe_R >= 1.0),
        "reached_2R": bool(mfe_R >= 2.0),
        "stopped_before_1R": bool(exit_reason in (GAP_STOP, INTRADAY_STOP, TRAIL_STOP) and mfe_R < 1.0),
        "stop_out_loss": bool(r_multiple < 0.0),
        "positive_convexity": bool(r_multiple >= pos_threshold_R),
        "bars_held": int(bars_held),
    }


def label_trade_with_peak(
    open_, high, low, close, signal_idx, atr_ref, **kw
) -> dict:
    """As label_trade, but also computes time_to_peak (bar of max favorable high before exit)."""
    res = label_trade(open_, high, low, close, signal_idx, atr_ref, **kw)
    if not res.get("valid"):
        return res
    e, x = res["entry_idx"], res["exit_idx"]
    seg_high = np.asarray(high[e:x + 1], dtype="float64")
    res["time_to_peak"] = int(np.argmax(seg_high))  # bars after entry to the peak high
    return res
