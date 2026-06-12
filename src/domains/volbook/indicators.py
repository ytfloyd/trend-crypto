"""Compute every TA-Lib indicator for a bar series.

The goal is to feed the canvas a structured, grouped view of every
indicator the library exposes, with per-output history trimmed to a
handful of most-recent bars so the rendered ``.canvas.tsx`` stays a
reasonable size.

Design notes
------------
* We iterate ``talib.get_function_groups()`` so future TA-Lib releases
  automatically surface new functions without code changes.
* Multi-input operators (``ADD``, ``DIV``, ``SUB``, ``MULT`` — two
  independent series) are skipped: they aren't single-series technical
  indicators and would need a second series we don't have.
* Indicators with required-but-unusual inputs (``MAVP`` needs a per-bar
  ``periods`` array; ``SAREXT`` has many obscure params) are caught via
  try/except and recorded as ``error`` rows so the UI shows why they're
  missing rather than silently dropping them.
* ``NaN``s are replaced with ``None`` before JSON serialization so the
  output is strict JSON and the canvas can test ``value === null``.
"""
from __future__ import annotations

import math
from typing import Any, TypedDict

import numpy as np

try:
    import talib
    from talib import abstract as _ta_abstract
    TALIB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without extra
    TALIB_AVAILABLE = False


# Indicators we deliberately skip because the stock TA-Lib inputs don't
# map cleanly onto a single OHLCV series.
_SKIP_FUNCTIONS = frozenset(
    {
        "MAVP",   # requires a per-bar `periods` input array
    }
)


class IndicatorRecord(TypedDict):
    """One row of indicator output data for the canvas tables."""

    function: str
    display_name: str
    params: dict[str, Any]
    outputs: dict[str, list[float | None]]
    error: str | None


IndicatorsByCategory = dict[str, list[IndicatorRecord]]


def _ohlcv_arrays(bars: list[Any]) -> dict[str, np.ndarray]:
    """Build the float64 input arrays TA-Lib expects from Bar records."""
    return {
        "open": np.array([b.o for b in bars], dtype=np.float64),
        "high": np.array([b.h for b in bars], dtype=np.float64),
        "low": np.array([b.l for b in bars], dtype=np.float64),
        "close": np.array([b.c for b in bars], dtype=np.float64),
        "volume": np.array([b.v for b in bars], dtype=np.float64),
    }


def _tail_clean(arr: np.ndarray, tail: int) -> list[float | None]:
    """Return the last ``tail`` entries, converting NaN/inf to ``None``."""
    if arr is None:
        return []
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.ravel()
    if tail > 0:
        arr = arr[-tail:]
    out: list[float | None] = []
    for v in arr.tolist():
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out.append(None)
        else:
            out.append(float(v))
    return out


def _is_multi_input(input_names: Any) -> bool:
    """Skip indicators that take two independent price series."""
    # ``input_names`` is an OrderedDict whose values are either strings
    # (pointing at a single OHLCV column) or lists (a bundle of columns
    # from one series). Two or more keys typically means two separate
    # price inputs — outside the scope of a single-series snapshot.
    return len(input_names) > 1


def compute_all_indicators(
    bars: list[Any],
    *,
    tail: int = 20,
) -> IndicatorsByCategory:
    """Run every TA-Lib indicator on ``bars`` and return per-category results.

    ``tail`` caps each output's history so the canvas stays lean — set to
    0 to keep the full series (discouraged for large bundles).
    """
    if not TALIB_AVAILABLE:
        return {}
    if not bars:
        return {}

    inputs = _ohlcv_arrays(bars)
    groups = talib.get_function_groups()
    out: IndicatorsByCategory = {}

    for category, names in groups.items():
        rows: list[IndicatorRecord] = []
        for name in names:
            if name in _SKIP_FUNCTIONS:
                continue
            try:
                fn = _ta_abstract.Function(name)
                if _is_multi_input(fn.input_names):
                    continue
                params = {k: v for k, v in fn.parameters.items()}
                result = fn(inputs)
                output_names = list(fn.output_names)
                # TA-Lib returns a single ndarray for single-output
                # functions, and either a tuple or a list of ndarrays
                # for multi-output functions depending on version.
                if isinstance(result, (tuple, list)):
                    arrays = list(result)
                elif isinstance(result, np.ndarray) and result.ndim == 2:
                    # Some builds stack outputs along axis 0.
                    arrays = [result[i] for i in range(result.shape[0])]
                else:
                    arrays = [result]
                outputs: dict[str, list[float | None]] = {}
                for out_name, arr in zip(output_names, arrays):
                    outputs[out_name] = _tail_clean(arr, tail)
                display_name = _display_name(fn)
                rows.append(
                    IndicatorRecord(
                        function=name,
                        display_name=display_name,
                        params=params,
                        outputs=outputs,
                        error=None,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # Record the error row so the UI can show why a function
                # is missing — easier than silently dropping it.
                rows.append(
                    IndicatorRecord(
                        function=name,
                        display_name=name,
                        params={},
                        outputs={},
                        error=str(exc),
                    )
                )
        if rows:
            out[category] = rows
    return out


def _display_name(fn: Any) -> str:
    """Best-effort human-readable name from the TA-Lib info dict."""
    try:
        info = fn.info
        disp = info.get("display_name") if isinstance(info, dict) else None
        if disp:
            return str(disp)
    except Exception:  # noqa: BLE001
        pass
    return str(getattr(fn, "name", "") or fn.__class__.__name__)
