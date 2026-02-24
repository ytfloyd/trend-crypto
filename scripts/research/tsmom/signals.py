"""
TSMOM signal definitions — wide-format, per-asset trend signals.

Each function takes wide-format DataFrames (index=ts, columns=symbols)
and returns a wide-format signal DataFrame.  Positive signal => uptrend,
negative/zero => no trend (cash).

RET, VOL_SCALED, BINARY delegate to alpha_lab.signals (single source of
truth for the underlying math).  MAC, EMAC, LREG are wide-format
implementations — the multifreq module has per-symbol-group versions of the
same math, but no wide-format equivalents exist elsewhere in the codebase.

Signal types:
  RET        Raw trailing return
  VOL_SCALED Trailing return / realized vol (canonical Moskowitz et al.)
  BINARY     sign(trailing return)
  MAC        SMA crossover (fast/slow)
  EMAC       EMA crossover (fast/slow)
  LREG       Linear regression t-stat of log-price
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_RESEARCH_DIR = str(Path(__file__).resolve().parents[1])
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from alpha_lab.signals import (
    _tsmom as _alpha_lab_tsmom,
    _tsmom_binary as _alpha_lab_tsmom_binary,
)


# -----------------------------------------------------------------------
# Thin wrappers around alpha_lab signal functions
# -----------------------------------------------------------------------

def signal_ret(
    close: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Raw trailing return over *lookback* days.

    Uses the same return calculation as alpha_lab._tsmom but without
    vol-normalisation.
    """
    return close / close.shift(lookback) - 1.0


def signal_vol_scaled(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int,
    vol_lookback: int = 63,
) -> pd.DataFrame:
    """Trailing return / realized vol (canonical Moskowitz et al.).

    Delegates to alpha_lab.signals._tsmom.
    """
    dummy_volume = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    return _alpha_lab_tsmom(
        close, dummy_volume, returns,
        {"lookback": lookback, "vol_lookback": vol_lookback},
    )


def signal_binary(
    close: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Binary trend signal: +1 if lookback return > 0, else -1.

    Delegates to alpha_lab.signals._tsmom_binary.
    """
    dummy_volume = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    dummy_returns = close.pct_change()
    return _alpha_lab_tsmom_binary(
        close, dummy_volume, dummy_returns,
        {"lookback": lookback},
    )


# -----------------------------------------------------------------------
# Wide-format signals with no existing wide-format equivalent elsewhere.
# Math mirrors multifreq/run_momentum_multifreq.py signal_mac / signal_emac
# / signal_lreg, but those operate on per-symbol groups — not wide-format.
# -----------------------------------------------------------------------

def signal_mac(
    close: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """SMA crossover: (fast_MA - slow_MA) / slow_MA.

    fast window = lookback // 4, slow window = lookback.
    """
    fast = max(2, lookback // 4)
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(lookback, min_periods=lookback).mean()
    return (fast_ma - slow_ma) / slow_ma.replace(0, np.nan)


def signal_emac(
    close: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """EMA crossover: (fast_EMA - slow_EMA) / slow_EMA.

    fast span = lookback // 4, slow span = lookback.
    """
    fast = max(2, lookback // 4)
    fast_ema = close.ewm(span=fast, min_periods=fast).mean()
    slow_ema = close.ewm(span=lookback, min_periods=lookback).mean()
    return (fast_ema - slow_ema) / slow_ema.replace(0, np.nan)


def signal_lreg(
    close: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Linear regression t-statistic of log-price over lookback window."""
    log_close = np.log(close.clip(lower=1e-10))

    def _tstat(window):
        if window.isna().any() or len(window) < lookback:
            return np.nan
        y = window.values
        x = np.arange(len(y))
        slope, _, _, _, std_err = sp_stats.linregress(x, y)
        return slope / std_err if std_err > 1e-10 else 0.0

    return log_close.rolling(lookback, min_periods=lookback).apply(
        _tstat, raw=False
    )


SIGNAL_FUNCTIONS: dict[str, callable] = {
    "RET": signal_ret,
    "VOL_SCALED": signal_vol_scaled,
    "BINARY": signal_binary,
    "MAC": signal_mac,
    "EMAC": signal_emac,
    "LREG": signal_lreg,
}


def compute_signal(
    name: str,
    close: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int,
    vol_lookback: int = 63,
) -> pd.DataFrame:
    """Dispatch to the appropriate signal function by name."""
    if name not in SIGNAL_FUNCTIONS:
        raise ValueError(f"Unknown signal {name!r}. Choose from {list(SIGNAL_FUNCTIONS)}")

    fn = SIGNAL_FUNCTIONS[name]
    if name == "VOL_SCALED":
        return fn(close, returns, lookback, vol_lookback=vol_lookback)
    elif name in ("RET", "BINARY", "MAC", "EMAC", "LREG"):
        return fn(close, lookback)
    else:
        raise ValueError(f"No dispatch rule for signal {name!r}")
