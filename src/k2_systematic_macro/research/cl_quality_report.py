"""Exploratory CL data-quality and regime diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from k2_systematic_macro.data.dataset import ResearchDataset


@dataclass(frozen=True)
class FrameDiagnostics:
    timeframe: str
    rows: int
    start_ts: str | None
    end_ts: str | None
    duplicate_timestamps: int
    missing_close: int
    zero_volume_rows: int
    largest_gap_minutes: float | None
    median_gap_minutes: float | None
    sessions: int
    median_bars_per_session: float | None
    median_session_return: float | None
    median_session_range: float | None
    median_realized_vol: float | None
    median_atr_compression: float | None
    low_compression_share: float | None
    high_vol_of_vol_share: float | None
    trend_persistence_median: float | None
    positive_trend_share: float | None


def build_cl_quality_report(dataset: ResearchDataset) -> str:
    """Render a markdown report for CL bars/features before trading rules."""
    diagnostics = [
        _diagnose_frame(timeframe, dataset.bars[timeframe], dataset.features.get(timeframe))
        for timeframe in dataset.bars
    ]
    lines = [
        "# K2 Systematic Macro: CL First-Slice Diagnostics",
        "",
        "This report is deliberately limited to data quality and exploratory regime "
        "state diagnostics. It does not define entries, exits, position sizing, or "
        "trading rules.",
        "",
        "## Dataset Lineage",
        "",
        f"- Symbol: `{dataset.symbol}`",
        f"- Primary timeframe: `{dataset.primary_timeframe}`",
        f"- Source system: `{dataset.metadata.get('source_system', 'volbook')}`",
        f"- Source: `{dataset.metadata.get('source')}`",
        f"- Contract source: `{dataset.metadata.get('contract_source')}`",
        f"- Roll rule: `{dataset.metadata.get('roll_rule')}`",
        f"- Front-month coverage: `{dataset.metadata.get('front_month_coverage')}`",
        "",
        "## Coverage And Quality",
        "",
    ]
    for diag in diagnostics:
        lines.extend(
            [
                f"### {diag.timeframe}",
                "",
                f"- Rows: `{diag.rows}`",
                f"- Range: `{diag.start_ts}` to `{diag.end_ts}`",
                f"- Duplicate timestamps: `{diag.duplicate_timestamps}`",
                f"- Missing closes: `{diag.missing_close}`",
                f"- Zero-volume rows: `{diag.zero_volume_rows}`",
                f"- Median timestamp gap, minutes: `{diag.median_gap_minutes}`",
                f"- Largest timestamp gap, minutes: `{diag.largest_gap_minutes}`",
                f"- Sessions: `{diag.sessions}`",
                f"- Median bars/session: `{diag.median_bars_per_session}`",
                f"- Median session return: `{diag.median_session_return}`",
                f"- Median session range: `{diag.median_session_range}`",
                f"- Median realized volatility: `{diag.median_realized_vol}`",
                f"- Median ATR compression: `{diag.median_atr_compression}`",
                f"- Share of compressed bars: `{diag.low_compression_share}`",
                f"- Share of elevated vol-of-vol bars: `{diag.high_vol_of_vol_share}`",
                f"- Median trend persistence: `{diag.trend_persistence_median}`",
                f"- Share of positive trend-return bars: `{diag.positive_trend_share}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- Compression is measured as ATR relative to its rolling median; values below "
            "`0.75` are candidates for further volatility-expansion research.",
            "- Session behavior is summarized from completed bars grouped by exchange-local "
            "`session_date`; this is still exploratory and is not a trading session model.",
            "- Vol-of-vol is summarized cross-sectionally within this single CL panel by "
            "comparing each bar with the panel's own 75th percentile.",
            _roll_interpretation_note(dataset),
            "",
            "## Next Research Gates",
            "",
            "- Validate CL holiday calendars and roll assumptions before portfolio tests.",
            "- Compare GMM regimes with HMM/change-point baselines once optional dependencies "
            "and enough out-of-sample data are available.",
            "- Keep predictive model evaluation separate from entries, exits, and sizing.",
        ]
    )
    return "\n".join(lines) + "\n"


def _roll_interpretation_note(dataset: ResearchDataset) -> str:
    if dataset.metadata.get("contract_source") == "institutional_continuous":
        coverage = dataset.metadata.get("front_month_coverage", {})
        status = coverage.get("status") if isinstance(coverage, dict) else None
        if status and status != "valid":
            return (
                "- WARNING: institutional continuous front-month coverage is not valid; "
                "near/front dated contracts must be backfilled before using this dataset."
            )
        return (
            "- Institutional continuous output uses CL calendar rolls, volume crossover where available, "
            "and explicit back-adjustment metadata; the current NYMEX observed-calendar support covers "
            "the 2025-2026 research window."
        )
    coverage = dataset.metadata.get("front_month_coverage", {})
    status = coverage.get("status") if isinstance(coverage, dict) else None
    if status == "opaque":
        return "- Vendor continuous output is opaque; dated front-month coverage cannot be independently validated."
    if status and status != "valid":
        return "- WARNING: dated-front coverage is not valid; missing near/front contracts must be backfilled."
    return (
        "- Dated-front output is unadjusted. Roll windows and price discontinuities "
        "must be upgraded before portfolio tests or production signals."
    )


def _diagnose_frame(
    timeframe: str,
    bars: pd.DataFrame,
    features: pd.DataFrame | None,
) -> FrameDiagnostics:
    if bars.empty:
        return FrameDiagnostics(
            timeframe,
            0,
            None,
            None,
            0,
            0,
            0,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    ts = pd.to_datetime(bars["ts"], utc=True)
    gaps = ts.sort_values().diff().dt.total_seconds().div(60.0)
    feature_frame = features if features is not None else pd.DataFrame(index=bars.index)
    compression = _series(feature_frame, "atr_compression")
    vol_of_vol = _first_matching(feature_frame, "vol_of_vol_")
    realized_vol = _first_matching(feature_frame, "realized_vol_")
    trend_persistence = _series(feature_frame, "trend_persistence")
    trend_return = _series(feature_frame, "trend_return")
    session_stats = _session_stats(bars)
    return FrameDiagnostics(
        timeframe=timeframe,
        rows=int(bars.shape[0]),
        start_ts=_iso(ts.min()),
        end_ts=_iso(ts.max()),
        duplicate_timestamps=int(ts.duplicated().sum()),
        missing_close=int(bars["c"].isna().sum()),
        zero_volume_rows=int((bars["v"].fillna(0.0) <= 0.0).sum()),
        largest_gap_minutes=_rounded(gaps.max()),
        median_gap_minutes=_rounded(gaps.median()),
        sessions=int(session_stats.shape[0]),
        median_bars_per_session=_rounded(session_stats["bars"].median()) if not session_stats.empty else None,
        median_session_return=_rounded(session_stats["session_return"].median())
        if not session_stats.empty
        else None,
        median_session_range=_rounded(session_stats["session_range"].median())
        if not session_stats.empty
        else None,
        median_realized_vol=_rounded(realized_vol.median()) if realized_vol is not None else None,
        median_atr_compression=_rounded(compression.median()) if compression is not None else None,
        low_compression_share=_share(compression < 0.75) if compression is not None else None,
        high_vol_of_vol_share=_high_share(vol_of_vol) if vol_of_vol is not None else None,
        trend_persistence_median=_rounded(trend_persistence.median())
        if trend_persistence is not None
        else None,
        positive_trend_share=_share(trend_return > 0.0) if trend_return is not None else None,
    )


def _session_stats(bars: pd.DataFrame) -> pd.DataFrame:
    if "session_date" not in bars or bars.empty:
        return pd.DataFrame(columns=["bars", "session_return", "session_range"])
    grouped = bars.sort_values("ts").groupby("session_date", dropna=True)
    stats = grouped.agg(
        bars=("c", "size"),
        session_return=("c", lambda s: float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) else None),
    )
    stats["session_range"] = grouped["h"].max() / grouped["l"].min() - 1.0
    return stats


def _first_matching(frame: pd.DataFrame, prefix: str) -> pd.Series | None:
    for col in frame.columns:
        if str(col).startswith(prefix):
            return _series(frame, str(col))
    return None


def _series(frame: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in frame:
        return None
    return pd.to_numeric(frame[col], errors="coerce").dropna()


def _share(mask: pd.Series) -> float | None:
    if mask.empty:
        return None
    return _rounded(mask.mean())


def _high_share(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return _share(values > values.quantile(0.75))


def _rounded(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 6)


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
