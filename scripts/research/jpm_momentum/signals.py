"""
Momentum signal implementations for JPM Momentum research.

All signal types from Kolanovic & Wei (2015), adapted for crypto:

- **RET**  : Price Return — ``close[t-1] / close[t-1-L] - 1``
- **MAC**  : Moving Average Crossover — ``SMA(fast) - SMA(slow)``, normalised
- **EMAC** : Exponential MA Crossover — ``EMA(fast) - EMA(slow)``, normalised
- **BRK**  : Breakout Channel — Donchian channel position ``(close - low_L) / (high_L - low_L)``
- **LREG** : Linear Regression Slope — OLS slope on log-prices, t-stat normalised
- **RADJ** : Risk-Adjusted Momentum — ``return_L / vol_L`` (Sharpe-like, paper Ch.3 p.66)

Each function signature follows:
    ``compute_<name>(panel, lookback, **kwargs) -> pd.DataFrame``
adding a ``signal`` column to the long-format panel.

Lookback grid (adapted from paper's 1-12 months):
    [5, 10, 21, 42, 63, 126, 252] days
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from scripts.research.groupby_utils import apply_by_symbol


# ---------------------------------------------------------------------------
# Price Return (RET) — paper baseline
# ---------------------------------------------------------------------------
def compute_ret(panel: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Price return momentum: close[t-1] / close[t-1-L] - 1.

    Signal is computed with a 1-bar lag (signal at close t uses info up to t-1).
    """
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        close = g["close"]
        sig = close.shift(1) / close.shift(1 + lookback) - 1.0
        g = g.copy()
        g["signal"] = sig
        return g

    return apply_by_symbol(df, _per_symbol)


# ---------------------------------------------------------------------------
# Moving Average Crossover (MAC)
# ---------------------------------------------------------------------------
def compute_mac(
    panel: pd.DataFrame,
    lookback: int = 21,
    *,
    fast_span: int | None = None,
    slow_span: int | None = None,
) -> pd.DataFrame:
    """SMA crossover signal: SMA(fast) - SMA(slow), normalised by slow SMA.

    Defaults: fast = lookback // 3 (min 2), slow = lookback.
    """
    fast = fast_span if fast_span is not None else max(2, lookback // 3)
    slow = slow_span if slow_span is not None else lookback
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        close = g["close"].shift(1)
        sma_fast = close.rolling(fast, min_periods=fast).mean()
        sma_slow = close.rolling(slow, min_periods=slow).mean()
        sig = (sma_fast - sma_slow) / sma_slow
        g = g.copy()
        g["signal"] = sig
        return g

    return apply_by_symbol(df, _per_symbol)


# ---------------------------------------------------------------------------
# Exponential MA Crossover (EMAC)
# ---------------------------------------------------------------------------
def compute_emac(
    panel: pd.DataFrame,
    lookback: int = 21,
    *,
    fast_span: int | None = None,
    slow_span: int | None = None,
) -> pd.DataFrame:
    """EMA crossover signal: EMA(fast) - EMA(slow), normalised by slow EMA.

    Defaults: fast = lookback // 3 (min 2), slow = lookback.
    """
    fast = fast_span if fast_span is not None else max(2, lookback // 3)
    slow = slow_span if slow_span is not None else lookback
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        close = g["close"].shift(1)
        ema_fast = close.ewm(span=fast, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow).mean()
        sig = (ema_fast - ema_slow) / ema_slow
        g = g.copy()
        g["signal"] = sig
        return g

    return apply_by_symbol(df, _per_symbol)


# ---------------------------------------------------------------------------
# Breakout Channel (BRK) — Donchian channel position
# ---------------------------------------------------------------------------
def compute_brk(panel: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Breakout signal: (close - low_L) / (high_L - low_L).

    Maps close price to [0, 1] within the L-day Donchian channel.
    """
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        close = g["close"].shift(1)
        high_L = g["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        low_L = g["low"].shift(1).rolling(lookback, min_periods=lookback).min()
        channel_range = high_L - low_L
        sig = (close - low_L) / channel_range.replace(0, np.nan)
        g = g.copy()
        g["signal"] = sig
        return g

    return apply_by_symbol(df, _per_symbol)


# ---------------------------------------------------------------------------
# Linear Regression Slope (LREG) — t-stat normalised
# ---------------------------------------------------------------------------
def compute_lreg(panel: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Linear regression slope on log-prices over window L, t-stat normalised."""
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        log_close = np.log(g["close"].shift(1).clip(lower=1e-10))
        sig = log_close.rolling(lookback, min_periods=lookback).apply(
            _linreg_tstat, raw=True
        )
        g = g.copy()
        g["signal"] = sig
        return g

    return apply_by_symbol(df, _per_symbol)


def _linreg_tstat(y: np.ndarray) -> float:
    """Compute t-statistic of OLS slope on an array."""
    n = len(y)
    if n < 3 or np.any(~np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=float)
    slope, _intercept, _r, p_value, std_err = sp_stats.linregress(x, y)
    if std_err > 0:
        return slope / std_err
    return np.nan


# ---------------------------------------------------------------------------
# Risk-Adjusted Momentum (RADJ) — Sharpe-like signal
# ---------------------------------------------------------------------------
def compute_radj(panel: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Risk-adjusted momentum: return_L / vol_L (paper Ch.3 p.66)."""
    df = panel.copy().sort_values(["symbol", "ts"])

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        close = g["close"]
        ret_L = close.shift(1) / close.shift(1 + lookback) - 1.0
        daily_ret = close.pct_change().shift(1)
        vol_L = daily_ret.rolling(lookback, min_periods=lookback).std() * np.sqrt(365.0)
        sig = ret_L / vol_L.replace(0, np.nan)
        g = g.copy()
        g["signal"] = sig
        return g

    return apply_by_symbol(df, _per_symbol)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
SIGNAL_DISPATCH: dict[str, callable] = {
    "RET": compute_ret,
    "MAC": compute_mac,
    "EMAC": compute_emac,
    "BRK": compute_brk,
    "LREG": compute_lreg,
    "RADJ": compute_radj,
}


def compute_signal(
    panel: pd.DataFrame,
    signal_type: str,
    lookback: int = 21,
    **kwargs,
) -> pd.DataFrame:
    """Dispatch to the appropriate signal function by name.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel (symbol, ts, open, high, low, close, volume).
    signal_type : str
        One of: RET, MAC, EMAC, BRK, LREG, RADJ.
    lookback : int
        Lookback window in trading days.

    Returns
    -------
    pd.DataFrame
        Input panel with added ``signal`` column.
    """
    fn = SIGNAL_DISPATCH.get(signal_type)
    if fn is None:
        raise ValueError(
            f"Unknown signal_type {signal_type!r}. "
            f"Choose from {list(SIGNAL_DISPATCH)}"
        )
    return fn(panel, lookback=lookback, **kwargs)
