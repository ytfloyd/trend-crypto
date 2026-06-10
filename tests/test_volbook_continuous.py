"""Tests for institutional continuous futures construction."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from volbook.continuous import (
    FrontMonthCoverageError,
    FrontMonthGuard,
    NymexObservedHolidayCalendar,
    RollPolicy,
    WeekendHolidayCalendar,
    build_roll_schedule,
    cl_last_trade_date,
    construct_continuous_series,
)


def _row(day: str, expiry: str, close: float, volume: float = 100.0) -> dict[str, object]:
    return {
        "ts": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=14),
        "symbol": "CL",
        "expiry": expiry,
        "o": close - 0.2,
        "h": close + 0.5,
        "l": close - 0.5,
        "c": close,
        "v": volume,
    }


def test_cl_last_trade_date_uses_third_business_day_prior_to_25th() -> None:
    assert cl_last_trade_date("202606") == date(2026, 5, 20)


def test_cl_last_trade_date_adjusts_weekend_25th_anchor() -> None:
    # April 25, 2026 is Saturday, so the anchor moves to Friday April 24
    # before subtracting three business days.
    assert cl_last_trade_date("202605") == date(2026, 4, 21)


def test_cl_last_trade_date_uses_injected_nymex_holidays() -> None:
    assert cl_last_trade_date("202601", calendar=WeekendHolidayCalendar()) == date(2025, 12, 22)
    assert cl_last_trade_date("202601", calendar=NymexObservedHolidayCalendar()) == date(2025, 12, 19)


def test_forced_roll_schedule_uses_last_trade_minus_business_days() -> None:
    bars = pd.DataFrame(
        [
            _row("2026-05-13", "202606", 70.0),
            _row("2026-05-13", "202607", 71.0),
        ]
    )
    schedule = build_roll_schedule(
        bars,
        policy=RollPolicy(name="last_trade_minus_n_business_days"),
    )

    event = schedule.iloc[0]
    assert event["front_expiry"] == "202606"
    assert event["next_expiry"] == "202607"
    assert event["last_trade_date"] == date(2026, 5, 20)
    assert event["roll_date"] == date(2026, 5, 15)
    assert event["trigger"] == "calendar_forced"
    assert not event["fallback"]


def test_forced_roll_schedule_can_use_nymex_holiday_calendar() -> None:
    bars = pd.DataFrame(
        [
            _row("2025-12-16", "202601", 70.0),
            _row("2025-12-16", "202602", 71.0),
        ]
    )
    schedule = build_roll_schedule(
        bars,
        policy=RollPolicy(name="last_trade_minus_n_business_days"),
        calendar=NymexObservedHolidayCalendar(),
    )

    event = schedule.iloc[0]
    assert event["last_trade_date"] == date(2025, 12, 19)
    assert event["roll_date"] == date(2025, 12, 16)


def test_volume_crossover_requires_consecutive_sessions() -> None:
    bars = pd.DataFrame(
        [
            _row("2026-05-12", "202606", 70.0, 200.0),
            _row("2026-05-12", "202607", 71.0, 150.0),
            _row("2026-05-13", "202606", 70.2, 100.0),
            _row("2026-05-13", "202607", 71.2, 200.0),
            _row("2026-05-14", "202606", 70.4, 100.0),
            _row("2026-05-14", "202607", 71.4, 250.0),
        ]
    )

    schedule = build_roll_schedule(
        bars,
        policy=RollPolicy(
            name="volume_crossover_with_calendar_guard",
            volume_crossover_sessions=2,
        ),
    )

    event = schedule.iloc[0]
    assert event["roll_date"] == date(2026, 5, 14)
    assert event["trigger"] == "volume_crossover"
    assert not event["fallback"]


def test_volume_crossover_falls_back_to_forced_roll() -> None:
    bars = pd.DataFrame(
        [
            _row("2026-05-13", "202606", 70.0, 300.0),
            _row("2026-05-13", "202607", 71.0, 100.0),
            _row("2026-05-14", "202606", 70.2, 250.0),
            _row("2026-05-14", "202607", 71.2, 150.0),
        ]
    )

    schedule = build_roll_schedule(
        bars,
        policy=RollPolicy(
            name="volume_crossover_with_calendar_guard",
            volume_crossover_sessions=2,
        ),
    )

    event = schedule.iloc[0]
    assert event["roll_date"] == date(2026, 5, 15)
    assert event["trigger"] == "calendar_guard_forced"
    assert event["fallback"]


def test_additive_adjustment_math_removes_roll_gap_from_history() -> None:
    bars = pd.DataFrame(
        [
            _row("2026-05-13", "202606", 69.0, 100.0),
            _row("2026-05-13", "202607", 71.0, 200.0),
            _row("2026-05-14", "202606", 70.0, 100.0),
            _row("2026-05-14", "202607", 72.0, 200.0),
            _row("2026-05-15", "202607", 73.0, 250.0),
        ]
    )

    result = construct_continuous_series(
        bars,
        policy=RollPolicy(
            name="volume_crossover_with_calendar_guard",
            volume_crossover_sessions=2,
        ),
        adjustment="additive",
    )

    out = result.bars
    assert out["expiry"].to_list() == ["202606", "202607", "202607"]
    assert out["c"].to_list() == pytest.approx([71.0, 72.0, 73.0])
    assert out.loc[0, "adjustment_offset"] == pytest.approx(2.0)
    assert out.loc[1, "adjustment_offset"] == pytest.approx(0.0)
    assert result.schedule.loc[0, "roll_gap"] == pytest.approx(2.0)
    assert result.schedule.loc[0, "roll_return"] == pytest.approx(72.0 / 70.0 - 1.0)
    assert result.metadata["open_interest"] == "not_available_in_minute_lake; no OI crossover is inferred"


def test_front_month_guard_rejects_far_dated_only_cl_history() -> None:
    bars = pd.DataFrame(
        [
            _row("2025-09-15", "202702", 70.0),
            _row("2025-09-16", "202702", 70.5),
        ]
    )

    with pytest.raises(FrontMonthCoverageError, match="eligible front range.*202510"):
        construct_continuous_series(bars)


def test_front_month_guard_allows_m1_m2_roll_window() -> None:
    bars = pd.DataFrame(
        [
            _row("2026-05-13", "202606", 69.0, 100.0),
            _row("2026-05-13", "202607", 71.0, 200.0),
            _row("2026-05-14", "202606", 70.0, 100.0),
            _row("2026-05-14", "202607", 72.0, 200.0),
            _row("2026-05-15", "202607", 73.0, 250.0),
        ]
    )

    result = construct_continuous_series(
        bars,
        policy=RollPolicy(volume_crossover_sessions=2),
    )

    assert result.metadata["front_month_coverage"]["status"] == "valid"
    assert result.bars["front_month_valid"].all()


def test_front_month_guard_mark_mode_flags_invalid_rows() -> None:
    bars = pd.DataFrame(
        [
            _row("2025-09-15", "202702", 70.0),
            _row("2025-09-16", "202702", 70.5),
        ]
    )

    result = construct_continuous_series(
        bars,
        front_month_guard=FrontMonthGuard(on_missing="mark"),
    )

    assert result.metadata["front_month_coverage"]["status"] == "invalid"
    assert result.bars["expiry"].to_list() == ["202702", "202702"]
    assert result.bars["front_month_valid"].to_list() == [False, False]
