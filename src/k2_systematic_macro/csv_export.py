"""CSV export hygiene for K2 research artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pdt

EXCEL_TEXT_PREFIX = "'"
FLOAT_FORMAT = "%.12g"

_NUMERIC_EXACT_COLUMNS = {
    "o",
    "h",
    "l",
    "c",
    "v",
    "bar_count",
    "close",
    "regime",
    "regime_probability",
    "fold",
    "fold_start",
    "fold_end",
    "train_rows",
    "test_rows",
    "y_true",
    "y_prob",
    "prob_bin",
    "probability_decile",
    "expansion_decile",
    "prior_cell_n",
    "prior_expansion_n",
    "prior_top_cell_rank",
    "directional_prob_up",
    "directional_prob_down",
    "expected_move_atr",
    "expected_upside_move_atr",
    "expected_downside_move_atr",
    "upside_downside_asymmetry",
    "directional_asymmetry",
    "convexity_research_score",
    "confidence_overlay_multiplier",
    "research_confidence",
    "expost_target_vol_expansion_event_1d",
    "expost_forward_return_1d",
    "expost_forward_log_return_1d",
    "expost_forward_move_atr_1d",
    "n",
    "event_n",
    "event_rate",
    "mean_probability",
    "auc",
    "roc_auc",
    "brier_score",
    "precision",
    "recall",
    "f1",
    "share",
    "volatility",
    "skew",
    "max_drawdown",
    "tail_participation",
    "mean",
    "median",
    "std",
    "min",
    "max",
}
_NUMERIC_PREFIXES = (
    "target_",
    "return_",
    "log_return_",
    "realized_vol_",
    "vol_of_vol_",
    "expansion_",
    "breakout_",
    "trend_",
)
_NUMERIC_SUBSTRINGS = (
    "_return_",
    "_log_return_",
    "_move_norm_",
    "_vol_expansion_",
    "_event_",
)


def write_research_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    """Write a research CSV with numeric fields exported as numeric values.

    The generated artifacts are often opened in spreadsheet tools. This helper
    removes Excel-style text prefixes from values before writing and applies a
    stable float format so OHLC fields do not leak binary float noise.
    """
    export = frame.reset_index() if index else frame.copy()
    clean_csv_frame(export).to_csv(path, index=False, float_format=FLOAT_FORMAT)
    return path


def clean_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a CSV-ready copy with Excel text prefixes removed."""
    out = frame.copy()
    for col in out.columns:
        series = out[col]
        if pdt.is_bool_dtype(series):
            continue
        if pdt.is_datetime64_any_dtype(series):
            out[col] = series.map(_iso_or_empty)
            continue
        if pdt.is_numeric_dtype(series) or _is_numeric_column(str(col)):
            stripped = _strip_excel_text_prefix(series)
            out[col] = pd.to_numeric(stripped, errors="coerce")
            continue
        if pdt.is_object_dtype(series) or pdt.is_string_dtype(series):
            out[col] = series.map(_strip_prefix_if_safe)
    return out


def _is_numeric_column(col: str) -> bool:
    return (
        col in _NUMERIC_EXACT_COLUMNS
        or col.startswith(_NUMERIC_PREFIXES)
        or any(token in col for token in _NUMERIC_SUBSTRINGS)
        or col.startswith(("atr_", "range_compression", "true_range"))
    )


def _strip_excel_text_prefix(series: pd.Series) -> pd.Series:
    if pdt.is_object_dtype(series) or pdt.is_string_dtype(series):
        return series.map(_strip_prefix_if_present)
    return series


def _strip_prefix_if_present(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(EXCEL_TEXT_PREFIX):
        return value[1:]
    return value


def _strip_prefix_if_safe(value: Any) -> Any:
    if not isinstance(value, str) or not value.startswith(EXCEL_TEXT_PREFIX):
        return value
    candidate = value[1:]
    if _looks_numeric(candidate) or _looks_datetime(candidate):
        return candidate
    return value


def _looks_numeric(value: str) -> bool:
    return pd.to_numeric(pd.Series([value]), errors="coerce").notna().iloc[0]


def _looks_datetime(value: str) -> bool:
    if not value:
        return False
    parsed = pd.to_datetime(pd.Series([value]), errors="coerce", utc=True)
    return parsed.notna().iloc[0]


def _iso_or_empty(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()
