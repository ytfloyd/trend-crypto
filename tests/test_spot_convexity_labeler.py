"""Unit tests for the spot-convexity stop-aware R-multiple labeler.

These pin the execution semantics the whole sleeve depends on: entry at next open (no
look-ahead), gap-through-stop exits at the OPEN (loss can exceed -1R), intraday breach
exits at the stop, the trailing stop only ratchets up, the time stop, and right-censoring
of trades with insufficient forward history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research" / "spot_convexity"))

from labeler import (  # noqa: E402
    GAP_STOP, INTRADAY_STOP, TIME_STOP, TRAIL_STOP, label_trade, label_trade_with_peak,
)


def A(xs):
    return np.array(xs, dtype="float64")


KW = dict(atr_ref=1.0, stop_mult=2.0, trail_mult=3.0, max_horizon=60)  # risk = 2.0


def test_clean_runner_trails_out_for_large_R():
    o = A([99, 100, 103, 108, 113, 120])
    h = A([100, 104, 110, 115, 122, 121])
    lo = A([98, 99, 102, 108, 112, 117])
    c = A([100, 103, 109, 114, 121, 118])
    r = label_trade(o, h, lo, c, signal_idx=0, **KW)
    assert r["valid"] and not r["incomplete"]
    assert r["entry_price"] == 100.0          # entered at next open, not signal bar
    assert r["exit_reason"] == TRAIL_STOP
    assert r["r_multiple"] == pytest.approx(6.0)   # exit at trailed stop 112 -> (112-100)/2
    assert r["mfe_R"] == pytest.approx(11.0)       # hwm 122 -> (122-100)/2
    assert r["reached_1R"] and r["reached_2R"]
    assert not r["stop_out_loss"] and not r["stopped_before_1R"]


def test_immediate_intraday_stop_is_minus_1R():
    o = A([99, 100, 99])
    h = A([100, 101, 100])
    lo = A([98, 97, 99])
    c = A([100, 98, 99])
    r = label_trade(o, h, lo, c, signal_idx=0, **KW)
    assert r["exit_reason"] == INTRADAY_STOP
    assert r["r_multiple"] == pytest.approx(-1.0)   # exit AT stop 98
    assert r["stop_out_loss"] and r["stopped_before_1R"] and not r["reached_1R"]
    assert r["time_to_stop"] == 0


def test_gap_through_stop_exits_at_open_below_minus_1R():
    # the critical realism test: a gap opens below the stop -> loss exceeds -1R
    o = A([99, 100, 95])
    h = A([100, 101, 96])
    lo = A([98, 99, 94])
    c = A([100, 100, 95])
    r = label_trade(o, h, lo, c, signal_idx=0, **KW)
    assert r["exit_reason"] == GAP_STOP
    assert r["r_multiple"] == pytest.approx(-2.5)   # exit at open 95, worse than -1R
    assert r["stop_out_loss"]


def test_trailing_stop_never_moves_down():
    # rise to hwm=110 (trail=107), then a LOWER high (108) must not lower the stop;
    # a later low of 107 exits at 107, proving the stop held at the high-water level.
    o = A([99, 100, 104, 109, 107, 107])
    h = A([100, 105, 110, 108, 108, 108])
    lo = A([98, 99, 103, 107, 106, 106])
    c = A([100, 104, 109, 107, 107, 107])
    r = label_trade(o, h, lo, c, signal_idx=0, **KW)
    assert r["exit_reason"] == TRAIL_STOP
    assert r["r_multiple"] == pytest.approx(3.5)    # exit at 107, not the initial 98
    assert r["exit_idx"] == 3


def test_time_stop_at_horizon():
    o = A([99, 100, 100, 100, 100, 100])
    h = A([100, 101, 101, 101, 101, 101])
    lo = A([98, 99, 99, 99, 99, 99])
    c = A([100, 100, 100, 100, 100, 100])
    r = label_trade(o, h, lo, c, signal_idx=0, **{**KW, "max_horizon": 4})
    assert r["exit_reason"] == TIME_STOP
    assert r["r_multiple"] == pytest.approx(0.0)
    assert r["bars_held"] == 4


def test_incomplete_when_insufficient_forward_history():
    o = A([99, 100, 100, 100])
    h = A([100, 101, 101, 101])
    lo = A([98, 99, 99, 99])
    c = A([100, 100, 100, 100])
    r = label_trade(o, h, lo, c, signal_idx=0, **{**KW, "max_horizon": 10})
    assert r["incomplete"] and not r["valid"]
    assert r["entry_price"] == 100.0   # censored branch still records entry/stop


def test_no_bar_to_enter_is_incomplete():
    o = A([99, 100])
    h = A([100, 101])
    lo = A([98, 99])
    c = A([100, 100])
    r = label_trade(o, h, lo, c, signal_idx=1, **KW)   # entry_idx=2 >= n
    assert r["incomplete"] and not r["valid"]


def test_nonpositive_atr_is_invalid():
    o = A([99, 100, 101])
    h = A([100, 101, 102])
    lo = A([98, 99, 100])
    c = A([100, 100, 101])
    assert not label_trade(o, h, lo, c, 0, atr_ref=0.0)["valid"]
    assert not label_trade(o, h, lo, c, 0, atr_ref=-1.0)["valid"]


def test_positive_convexity_label_threshold():
    o = A([99, 100, 103, 108, 113, 120])
    h = A([100, 104, 110, 115, 122, 121])
    lo = A([98, 99, 102, 108, 112, 117])
    c = A([100, 103, 109, 114, 121, 118])
    assert label_trade(o, h, lo, c, 0, **KW)["positive_convexity"]            # R=6 >= 2
    assert not label_trade(o, h, lo, c, 0, **{**KW, "pos_threshold_R": 8.0})["positive_convexity"]


def test_time_to_peak():
    o = A([99, 100, 103, 108, 113, 120])
    h = A([100, 104, 110, 115, 122, 121])
    lo = A([98, 99, 102, 108, 112, 117])
    c = A([100, 103, 109, 114, 121, 118])
    r = label_trade_with_peak(o, h, lo, c, 0, **KW)
    assert r["time_to_peak"] == 3   # peak high (122) is 3 bars after entry
