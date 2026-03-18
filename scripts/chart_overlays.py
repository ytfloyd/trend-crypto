"""TA-Lib overlay registry for chart.py.

Maps CLI overlay names to TA-Lib computation functions so that any of the
~40 indicators can be rendered with ``--overlay name:params``.

TA-Lib (C library + Python wrapper) must be installed separately::

    brew install ta-lib          # macOS
    pip install TA-Lib>=0.4.28   # Python wrapper
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import talib

    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False


def _require_talib() -> None:
    if not _HAS_TALIB:
        raise ImportError(
            "TA-Lib is required for this overlay.  Install it with:\n"
            "  brew install ta-lib && pip install TA-Lib"
        )


# ---------------------------------------------------------------------------
# Registry data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OverlayDef:
    """Metadata + compute function for a single TA-Lib overlay."""

    name: str
    category: str
    description: str
    axis: str  # "secondary", "price", "multi_secondary"
    defaults: list[Any] = field(default_factory=list)
    param_help: str = ""
    compute: Callable[..., dict] = field(repr=False, default=lambda df, params: {})


REGISTRY: dict[str, OverlayDef] = {}


def _register(defn: OverlayDef) -> None:
    REGISTRY[defn.name] = defn


# ---------------------------------------------------------------------------
# Helper – extract float64 OHLCV arrays
# ---------------------------------------------------------------------------
def _ohlcv(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return (
        df["open"].values.astype(np.float64),
        df["high"].values.astype(np.float64),
        df["low"].values.astype(np.float64),
        df["close"].values.astype(np.float64),
        df["volume"].values.astype(np.float64),
    )


def _secondary(series: pd.Series, label: str, **style_kw: Any) -> dict:
    base = {"linewidth": 1.5, "alpha": 0.85}
    base.update(style_kw)
    return {"series": series, "axis": "secondary", "label": label, "style": base}


def _multi_secondary(series_dict: dict[str, pd.Series], label: str) -> dict:
    """Multiple lines on the same secondary axis (e.g. Stoch K+D, MACD)."""
    return {"series": series_dict, "axis": "multi_secondary", "label": label, "style": {}}


def _price(series_dict: dict[str, pd.Series], label: str | None = None) -> dict:
    return {"series": series_dict, "axis": "price", "label": label, "style": {"linewidth": 1.2}}


# ═══════════════════════════════════════════════════════════════════════════
# TREND
# ═══════════════════════════════════════════════════════════════════════════

def _adx(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(pd.Series(talib.ADX(h, lo, c, timeperiod=p), index=df.index), f"ADX({p})")


def _dx(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(pd.Series(talib.DX(h, lo, c, timeperiod=p), index=df.index), f"DX({p})")


def _plus_di(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(
        pd.Series(talib.PLUS_DI(h, lo, c, timeperiod=p), index=df.index), f"+DI({p})"
    )


def _minus_di(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(
        pd.Series(talib.MINUS_DI(h, lo, c, timeperiod=p), index=df.index), f"-DI({p})"
    )


def _macd(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    fast = int(params[0]) if len(params) > 0 else 12
    slow = int(params[1]) if len(params) > 1 else 26
    sig = int(params[2]) if len(params) > 2 else 9
    macd_line, macd_signal, macd_hist = talib.MACD(
        c, fastperiod=fast, slowperiod=slow, signalperiod=sig
    )
    return _multi_secondary(
        {
            f"MACD({fast},{slow})": pd.Series(macd_line, index=df.index),
            f"Signal({sig})": pd.Series(macd_signal, index=df.index),
            f"Hist": pd.Series(macd_hist, index=df.index),
        },
        f"MACD({fast},{slow},{sig})",
    )


def _aroon(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, _, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    down, up = talib.AROON(h, lo, timeperiod=p)
    osc = up - down
    return _secondary(pd.Series(osc, index=df.index), f"Aroon Osc({p})")


def _linearreg_slope(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(
        pd.Series(talib.LINEARREG_SLOPE(c, timeperiod=p), index=df.index),
        f"LinReg Slope({p})",
    )


def _sar(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, _, _ = _ohlcv(df)
    accel = float(params[0]) if len(params) > 0 else 0.02
    maximum = float(params[1]) if len(params) > 1 else 0.2
    vals = talib.SAR(h, lo, acceleration=accel, maximum=maximum)
    return _price(
        {f"SAR({accel},{maximum})": pd.Series(vals, index=df.index)}, "SAR"
    )


def _ht_trendline(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    return _price(
        {"HT Trendline": pd.Series(talib.HT_TRENDLINE(c), index=df.index)},
        "HT Trendline",
    )


def _kama(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 10
    return _price({f"KAMA({p})": pd.Series(talib.KAMA(c, timeperiod=p), index=df.index)})


# ═══════════════════════════════════════════════════════════════════════════
# MOMENTUM / OSCILLATORS
# ═══════════════════════════════════════════════════════════════════════════

def _rsi(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(pd.Series(talib.RSI(c, timeperiod=p), index=df.index), f"RSI({p})")


def _stoch(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    fastk = int(params[0]) if len(params) > 0 else 14
    slowk = int(params[1]) if len(params) > 1 else 3
    slowd = int(params[2]) if len(params) > 2 else 3
    k, d = talib.STOCH(h, lo, c, fastk_period=fastk, slowk_period=slowk, slowd_period=slowd)
    return _multi_secondary(
        {f"Stoch %K": pd.Series(k, index=df.index), f"Stoch %D": pd.Series(d, index=df.index)},
        f"Stoch({fastk},{slowk},{slowd})",
    )


def _stochf(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    fastk = int(params[0]) if len(params) > 0 else 14
    fastd = int(params[1]) if len(params) > 1 else 3
    k, d = talib.STOCHF(h, lo, c, fastk_period=fastk, fastd_period=fastd)
    return _multi_secondary(
        {f"FastK": pd.Series(k, index=df.index), f"FastD": pd.Series(d, index=df.index)},
        f"StochF({fastk},{fastd})",
    )


def _stochrsi(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if len(params) > 0 else 14
    fastk = int(params[1]) if len(params) > 1 else 5
    fastd = int(params[2]) if len(params) > 2 else 3
    k, d = talib.STOCHRSI(c, timeperiod=p, fastk_period=fastk, fastd_period=fastd)
    return _multi_secondary(
        {
            f"StochRSI %K": pd.Series(k, index=df.index),
            f"StochRSI %D": pd.Series(d, index=df.index),
        },
        f"StochRSI({p},{fastk},{fastd})",
    )


def _cci(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(pd.Series(talib.CCI(h, lo, c, timeperiod=p), index=df.index), f"CCI({p})")


def _willr(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(
        pd.Series(talib.WILLR(h, lo, c, timeperiod=p), index=df.index), f"Williams %R({p})"
    )


def _mfi(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, v = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(pd.Series(talib.MFI(h, lo, c, v, timeperiod=p), index=df.index), f"MFI({p})")


def _roc(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 10
    return _secondary(pd.Series(talib.ROC(c, timeperiod=p), index=df.index), f"ROC({p})")


def _mom(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 10
    return _secondary(pd.Series(talib.MOM(c, timeperiod=p), index=df.index), f"MOM({p})")


def _ultosc(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    p1 = int(params[0]) if len(params) > 0 else 7
    p2 = int(params[1]) if len(params) > 1 else 14
    p3 = int(params[2]) if len(params) > 2 else 28
    return _secondary(
        pd.Series(
            talib.ULTOSC(h, lo, c, timeperiod1=p1, timeperiod2=p2, timeperiod3=p3),
            index=df.index,
        ),
        f"UltOsc({p1},{p2},{p3})",
    )


def _apo(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    fast = int(params[0]) if len(params) > 0 else 12
    slow = int(params[1]) if len(params) > 1 else 26
    return _secondary(
        pd.Series(talib.APO(c, fastperiod=fast, slowperiod=slow), index=df.index),
        f"APO({fast},{slow})",
    )


def _ppo(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    fast = int(params[0]) if len(params) > 0 else 12
    slow = int(params[1]) if len(params) > 1 else 26
    return _secondary(
        pd.Series(talib.PPO(c, fastperiod=fast, slowperiod=slow), index=df.index),
        f"PPO({fast},{slow})",
    )


def _cmo(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(pd.Series(talib.CMO(c, timeperiod=p), index=df.index), f"CMO({p})")


def _bop(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    o, h, lo, c, _ = _ohlcv(df)
    return _secondary(pd.Series(talib.BOP(o, h, lo, c), index=df.index), "BOP")


def _trix(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(pd.Series(talib.TRIX(c, timeperiod=p), index=df.index), f"TRIX({p})")


# ═══════════════════════════════════════════════════════════════════════════
# VOLATILITY
# ═══════════════════════════════════════════════════════════════════════════

def _natr(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    return _secondary(pd.Series(talib.NATR(h, lo, c, timeperiod=p), index=df.index), f"NATR({p})")


def _bb_width(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 20
    upper, mid, lower = talib.BBANDS(c, timeperiod=p, nbdevup=2, nbdevdn=2)
    width = (upper - lower) / np.where(mid > 0, mid, np.nan)
    return _secondary(pd.Series(width, index=df.index), f"BB Width({p})")


def _bb_pctb(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 20
    upper, _mid, lower = talib.BBANDS(c, timeperiod=p, nbdevup=2, nbdevdn=2)
    rng = upper - lower
    pctb = (c - lower) / np.where(rng > 0, rng, np.nan)
    return _secondary(pd.Series(pctb, index=df.index), f"BB %B({p})")


def _trange(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    return _secondary(pd.Series(talib.TRANGE(h, lo, c), index=df.index), "True Range")


# ═══════════════════════════════════════════════════════════════════════════
# VOLUME
# ═══════════════════════════════════════════════════════════════════════════

def _obv(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, _, _, c, v = _ohlcv(df)
    p = int(params[0]) if params else 14
    raw = talib.OBV(c, v)
    slope = talib.LINEARREG_SLOPE(raw, timeperiod=p)
    return _secondary(pd.Series(slope, index=df.index), f"OBV Slope({p})")


def _ad(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, v = _ohlcv(df)
    p = int(params[0]) if params else 14
    raw = talib.AD(h, lo, c, v)
    slope = talib.LINEARREG_SLOPE(raw, timeperiod=p)
    return _secondary(pd.Series(slope, index=df.index), f"A/D Slope({p})")


# ═══════════════════════════════════════════════════════════════════════════
# PRICE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

def _donchian(df: pd.DataFrame, params: list[str]) -> dict:
    _require_talib()
    _, h, lo, c, _ = _ohlcv(df)
    p = int(params[0]) if params else 14
    chan_hi = talib.MAX(h, p)
    chan_lo = talib.MIN(lo, p)
    rng = chan_hi - chan_lo
    pos = (c - chan_lo) / np.where(rng > 0, rng, np.nan)
    return _secondary(pd.Series(pos, index=df.index), f"Donchian Pos({p})")


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

_DEFS: list[OverlayDef] = [
    # -- Trend --
    OverlayDef("adx", "Trend", "Average Directional Index", "secondary", [14], "N", _adx),
    OverlayDef("dx", "Trend", "Directional Movement Index", "secondary", [14], "N", _dx),
    OverlayDef("plus_di", "Trend", "+DI (bullish directional)", "secondary", [14], "N", _plus_di),
    OverlayDef("minus_di", "Trend", "-DI (bearish directional)", "secondary", [14], "N", _minus_di),
    OverlayDef(
        "macd", "Trend", "MACD line + signal + histogram", "multi_secondary",
        [12, 26, 9], "fast,slow,sig", _macd,
    ),
    OverlayDef("aroon", "Trend", "Aroon Oscillator (up-down)", "secondary", [14], "N", _aroon),
    OverlayDef(
        "linearreg_slope", "Trend", "Linear regression slope of close",
        "secondary", [14], "N", _linearreg_slope,
    ),
    OverlayDef("sar", "Trend", "Parabolic SAR", "price", [0.02, 0.2], "accel,max", _sar),
    OverlayDef(
        "ht_trendline", "Trend", "Hilbert Transform trendline", "price", [], "", _ht_trendline,
    ),
    OverlayDef("kama", "Trend", "Kaufman Adaptive MA", "price", [10], "N", _kama),
    # -- Momentum / Oscillators --
    OverlayDef("rsi", "Momentum", "Relative Strength Index", "secondary", [14], "N", _rsi),
    OverlayDef(
        "stoch", "Momentum", "Slow Stochastic %K/%D", "multi_secondary",
        [14, 3, 3], "fastk,slowk,slowd", _stoch,
    ),
    OverlayDef(
        "stochf", "Momentum", "Fast Stochastic %K/%D", "multi_secondary",
        [14, 3], "fastk,fastd", _stochf,
    ),
    OverlayDef(
        "stochrsi", "Momentum", "Stochastic RSI %K/%D", "multi_secondary",
        [14, 5, 3], "N,fastk,fastd", _stochrsi,
    ),
    OverlayDef("cci", "Momentum", "Commodity Channel Index", "secondary", [14], "N", _cci),
    OverlayDef("willr", "Momentum", "Williams %R", "secondary", [14], "N", _willr),
    OverlayDef("mfi", "Momentum", "Money Flow Index", "secondary", [14], "N", _mfi),
    OverlayDef("roc", "Momentum", "Rate of Change", "secondary", [10], "N", _roc),
    OverlayDef("mom", "Momentum", "Momentum", "secondary", [10], "N", _mom),
    OverlayDef(
        "ultosc", "Momentum", "Ultimate Oscillator", "secondary",
        [7, 14, 28], "p1,p2,p3", _ultosc,
    ),
    OverlayDef(
        "apo", "Momentum", "Absolute Price Oscillator", "secondary",
        [12, 26], "fast,slow", _apo,
    ),
    OverlayDef(
        "ppo", "Momentum", "Percentage Price Oscillator", "secondary",
        [12, 26], "fast,slow", _ppo,
    ),
    OverlayDef("cmo", "Momentum", "Chande Momentum Oscillator", "secondary", [14], "N", _cmo),
    OverlayDef("bop", "Momentum", "Balance of Power", "secondary", [], "", _bop),
    OverlayDef("trix", "Momentum", "Triple EMA rate-of-change", "secondary", [14], "N", _trix),
    # -- Volatility --
    OverlayDef("natr", "Volatility", "Normalized ATR (%)", "secondary", [14], "N", _natr),
    OverlayDef("bb_width", "Volatility", "Bollinger Band width", "secondary", [20], "N", _bb_width),
    OverlayDef("bb_pctb", "Volatility", "Bollinger %B (0-1)", "secondary", [20], "N", _bb_pctb),
    OverlayDef("trange", "Volatility", "True Range", "secondary", [], "", _trange),
    # -- Volume --
    OverlayDef("obv", "Volume", "OBV linear-regression slope", "secondary", [14], "N", _obv),
    OverlayDef("ad", "Volume", "Accum/Dist linear-regression slope", "secondary", [14], "N", _ad),
    # -- Price Structure --
    OverlayDef(
        "donchian", "Structure", "Donchian channel position (0-1)", "secondary", [14], "N",
        _donchian,
    ),
]

for _d in _DEFS:
    _register(_d)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

CATEGORIES = ["Trend", "Momentum", "Volatility", "Volume", "Structure"]


def list_overlays() -> str:
    """Return a formatted string listing every registered TA-Lib overlay."""
    lines: list[str] = []
    for cat in CATEGORIES:
        members = [d for d in _DEFS if d.category == cat]
        if not members:
            continue
        lines.append(f"\n  {cat}")
        lines.append("  " + "─" * 50)
        for d in members:
            syntax = f"{d.name}:{d.param_help}" if d.param_help else d.name
            defaults = ",".join(str(v) for v in d.defaults)
            default_note = f"  (default: {defaults})" if defaults else ""
            lines.append(f"    {syntax:<28s} {d.description}{default_note}")
    return "\n".join(lines)


def compute_talib_overlay(df: pd.DataFrame, name: str, params: list[str]) -> dict | None:
    """Look up *name* in the registry and compute it, or return None."""
    defn = REGISTRY.get(name)
    if defn is None:
        return None
    return defn.compute(df, params)
