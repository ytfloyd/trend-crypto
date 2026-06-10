"""Canonical CL research dataset construction."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from k2_systematic_macro.configs.cl import CLResearchConfig
from k2_systematic_macro.data.volbook_adapter import VolbookLoadSpec, VolbookResearchDataAdapter
from k2_systematic_macro.features.core import build_feature_frame


@dataclass
class ResearchDataset:
    """Canonical futures research bars, features, and lineage metadata."""

    symbol: str
    primary_timeframe: str
    bars: dict[str, pd.DataFrame]
    features: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class CanonicalCLDatasetBuilder:
    """Build the first CL research dataset from the volbook minute store."""

    def __init__(self, config: CLResearchConfig | None = None) -> None:
        self.config = config or CLResearchConfig()
        load_spec = VolbookLoadSpec(
            symbol=self.config.symbol,
            source=self.config.source,
            contract_source=self.config.contract_source,
            expiry=self.config.expiry,
            roll_days_before_expiry=self.config.roll_days_before_expiry,
            continuous_adjustment=self.config.continuous_adjustment,
            continuous_roll_policy=self.config.continuous_roll_policy,
            continuous_business_calendar=self.config.continuous_business_calendar,
            continuous_roll_window_business_days=self.config.continuous_roll_window_business_days,
            continuous_forced_roll_business_days_before_last_trade=(
                self.config.continuous_forced_roll_business_days_before_last_trade
            ),
            continuous_volume_crossover_sessions=self.config.continuous_volume_crossover_sessions,
            front_month_guard=self.config.front_month_guard,
            front_month_guard_max_curve_position=self.config.front_month_guard_max_curve_position,
            front_month_guard_on_missing=self.config.front_month_guard_on_missing,
            start_ts=self.config.start_ts,
            end_ts=self.config.end_ts,
            parquet_root=self.config.parquet_root,
            lake_path=self.config.lake_path,
        )
        self.adapter = VolbookResearchDataAdapter(load_spec)

    def build(self, *, include_features: bool = True) -> ResearchDataset:
        """Load CL minutes, resample to canonical bars, and compute features."""
        minute_bars = self.adapter.load_minute_bars()
        bars = {
            timeframe: self.adapter.resample(
                minute_bars,
                timeframe,
                session_tz=self.config.session_tz,
                session_start_hour=self.config.session_start_hour,
            )
            for timeframe in self.config.timeframes
        }
        features = {
            timeframe: build_feature_frame(frame)
            for timeframe, frame in bars.items()
            if include_features and timeframe in self.config.feature_timeframes
        }
        metadata = self._metadata(minute_bars, bars)
        return ResearchDataset(
            symbol=self.config.symbol,
            primary_timeframe=self.config.primary_timeframe,
            bars=bars,
            features=features,
            metadata=metadata,
        )

    def write(self, dataset: ResearchDataset, output_root: Path | None = None) -> Path:
        """Persist bars, features, and metadata as a versioned research dataset."""
        root = output_root or self.config.output_root
        dataset_id = dataset.metadata.get("dataset_id", _dataset_id(dataset.symbol))
        out = root / dataset.symbol / dataset_id
        out.mkdir(parents=True, exist_ok=True)
        for timeframe, frame in dataset.bars.items():
            frame.to_parquet(out / f"bars_{timeframe}.parquet", index=False)
        for timeframe, frame in dataset.features.items():
            frame.to_parquet(out / f"features_{timeframe}.parquet", index=False)
        (out / "metadata.json").write_text(json.dumps(dataset.metadata, indent=2, default=str))
        return out

    def _metadata(
        self,
        minute_bars: pd.DataFrame,
        bars: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        row_counts = {timeframe: int(frame.shape[0]) for timeframe, frame in bars.items()}
        ranges = {
            timeframe: {
                "start_ts": _iso(frame["ts"].min()) if not frame.empty else None,
                "end_ts": _iso(frame["ts"].max()) if not frame.empty else None,
            }
            for timeframe, frame in bars.items()
        }
        now = datetime.now(timezone.utc)
        institutional_meta = self.adapter.last_load_metadata
        if self.config.contract_source == "institutional_continuous" and institutional_meta:
            expiries = [str(x) for x in institutional_meta.get("source_expiries", [])]
        else:
            expiries = sorted(str(x) for x in minute_bars["expiry"].dropna().unique())
        roll_rule = _roll_rule_metadata(self.config, institutional_meta)
        notes = [
            "First slice is data, features, and diagnostics only.",
        ]
        if self.config.contract_source == "institutional_continuous":
            notes.append(
                "Institutional continuous output uses dated contracts, CL roll-calendar logic, "
                "and explicit adjustment/roll lineage."
            )
        else:
            notes.append(
                "Dated-front output is unadjusted and should not be treated as a production continuous contract."
            )
        return {
            "dataset_id": _dataset_id(self.config.symbol, now),
            "created_at": now.isoformat(),
            "framework": "k2_systematic_macro",
            "symbol": self.config.symbol,
            "source_system": "volbook",
            "source": self.config.source,
            "contract_source": self.config.contract_source,
            "expiry": self.config.expiry,
            "start_ts": self.config.start_ts,
            "end_ts": self.config.end_ts,
            "source_expiries": expiries,
            "roll_rule": roll_rule,
            "institutional_continuous": institutional_meta,
            "front_month_coverage": institutional_meta.get("front_month_coverage", {}),
            "roll_schedule": _records(self.adapter.last_roll_schedule),
            "timeframes": list(self.config.timeframes),
            "primary_timeframe": self.config.primary_timeframe,
            "session": {
                "timezone": self.config.session_tz,
                "session_start_hour": self.config.session_start_hour,
                "timestamp_semantics": "bar end timestamp in UTC; session_date is exchange-local trade date",
            },
            "minute_rows": int(minute_bars.shape[0]),
            "bar_rows": row_counts,
            "bar_ranges": ranges,
            "notes": notes,
        }


def _dataset_id(symbol: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{symbol.upper()}_volbook_first_slice_{stamp}"


def _roll_rule_metadata(
    config: CLResearchConfig,
    institutional_meta: dict[str, Any],
) -> dict[str, Any]:
    if config.contract_source == "institutional_continuous":
        return {
            "method": institutional_meta.get("policy_name", config.continuous_roll_policy),
            "policy_version": institutional_meta.get("policy_version"),
            "adjustment": institutional_meta.get("adjustment_method", config.continuous_adjustment),
            "roll_window_business_days": config.continuous_roll_window_business_days,
            "forced_roll_business_days_before_last_trade": (
                config.continuous_forced_roll_business_days_before_last_trade
            ),
            "volume_crossover_sessions": config.continuous_volume_crossover_sessions,
            "calendar": institutional_meta.get("calendar"),
            "calendar_limitation": institutional_meta.get("calendar_limitation"),
            "open_interest": institutional_meta.get("open_interest"),
        }
    return {
        "method": "first-calendar-day-minus-N-days",
        "roll_days_before_expiry": config.roll_days_before_expiry,
        "adjustment": "none",
    }


def _iso(value: Any) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.to_dict(orient="records")
