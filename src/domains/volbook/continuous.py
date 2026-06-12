"""Institutional continuous futures construction from dated volbook bars.

The initial implementation is deliberately CL-focused.  It uses the exchange
last-trade-date convention for NYMEX WTI crude oil futures with a simple
business-day calendar: weekends are excluded and an optional holiday set can be
injected by callers.  Until a full exchange holiday calendar is supplied, dates
around exchange holidays should be treated as a documented approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Literal, Protocol

import pandas as pd


OHLC_COLUMNS = ("o", "h", "l", "c")
POLICY_VERSION = "volbook.cl_continuous.v1"

AdjustmentMethod = Literal["raw", "additive", "ratio"]
GuardAction = Literal["raise", "mark"]
RollPolicyName = Literal[
    "last_trade_minus_n_business_days",
    "volume_crossover",
    "volume_crossover_with_calendar_guard",
]


class BusinessDayCalendar(Protocol):
    """Minimal exchange-calendar interface used by roll construction."""

    def is_business_day(self, day: date) -> bool:
        """Return whether ``day`` is an exchange business day."""


@dataclass(frozen=True)
class WeekendHolidayCalendar:
    """Simple business-day calendar with injectable holidays.

    This fallback excludes Saturdays, Sundays, and any explicit dates supplied
    in ``holidays``.  It is not a substitute for a full CME/NYMEX holiday
    calendar, but keeps the roll engine deterministic and easy to replace.
    """

    holidays: frozenset[date] = frozenset()

    def __init__(self, holidays: Iterable[date] | None = None) -> None:
        object.__setattr__(self, "holidays", frozenset(holidays or ()))

    def is_business_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.holidays


NYMEX_OBSERVED_HOLIDAYS_2025_2026 = frozenset(
    {
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    }
)


class NymexObservedHolidayCalendar(WeekendHolidayCalendar):
    """Observed NYMEX full-close holiday calendar for the 2025-2026 CL window."""

    name = "nymex_observed_2025_2026"

    def __init__(self, holidays: Iterable[date] | None = None) -> None:
        extra = set(holidays or ())
        super().__init__(NYMEX_OBSERVED_HOLIDAYS_2025_2026 | extra)


def calendar_from_name(name: str | None) -> BusinessDayCalendar:
    """Resolve a documented roll-calendar name to a business-day calendar."""

    normalized = (name or "weekend").lower()
    if normalized in {"weekend", "weekends", "weekend_only"}:
        return WeekendHolidayCalendar()
    if normalized in {"nymex", "nymex_observed", "nymex_observed_2025_2026"}:
        return NymexObservedHolidayCalendar()
    raise ValueError("business calendar must be 'weekend' or 'nymex_observed'")


@dataclass(frozen=True)
class RollPolicy:
    """Roll trigger configuration for dated-contract continuous series."""

    name: RollPolicyName = "volume_crossover_with_calendar_guard"
    roll_window_business_days: int = 10
    forced_roll_business_days_before_last_trade: int = 3
    volume_crossover_sessions: int = 2
    version: str = POLICY_VERSION


@dataclass(frozen=True)
class FrontMonthGuard:
    """Eligibility guard for internally constructed front-month series.

    The default applies to CL only and permits the active front contract plus
    the next listed month. That is enough for normal CL roll windows but rejects
    stale lakes that only contain far-dated contracts such as M9+.
    """

    enabled: bool = True
    symbols: tuple[str, ...] = ("CL",)
    max_curve_position: int = 2
    on_missing: GuardAction = "raise"


class FrontMonthCoverageError(ValueError):
    """Raised when dated bars cannot support a front-month construction."""


@dataclass(frozen=True)
class RollEvent:
    """One scheduled transition from a front contract to its successor."""

    front_expiry: str
    next_expiry: str
    last_trade_date: date
    roll_window_start: date
    forced_roll_date: date
    roll_date: date
    trigger: str
    fallback: bool
    policy_name: str
    policy_version: str


@dataclass(frozen=True)
class ContinuousResult:
    """Constructed continuous bars plus schedule and metadata lineage."""

    bars: pd.DataFrame
    schedule: pd.DataFrame
    metadata: dict[str, object]


def cl_last_trade_date(
    delivery_month: str | date | datetime,
    *,
    calendar: BusinessDayCalendar | None = None,
) -> date:
    """Return the CL last-trade date for a delivery month.

    CL normally stops trading on the third business day before the 25th calendar
    day of the month preceding delivery.  If that 25th day is not a business
    day, the anchor is moved to the preceding business day first.  The default
    calendar excludes weekends only; pass an exchange holiday calendar to handle
    NYMEX holidays exactly.
    """

    cal = calendar or WeekendHolidayCalendar()
    delivery = _delivery_month_start(delivery_month)
    prior_month = _add_months(delivery, -1)
    anchor = date(prior_month.year, prior_month.month, 25)
    anchor = previous_business_day(anchor, cal, include_start=True)
    return subtract_business_days(anchor, 3, cal)


def previous_business_day(
    day: date,
    calendar: BusinessDayCalendar,
    *,
    include_start: bool = False,
) -> date:
    cursor = day if include_start else day - timedelta(days=1)
    while not calendar.is_business_day(cursor):
        cursor -= timedelta(days=1)
    return cursor


def subtract_business_days(
    day: date,
    count: int,
    calendar: BusinessDayCalendar,
) -> date:
    cursor = day
    remaining = max(0, int(count))
    while remaining:
        cursor -= timedelta(days=1)
        if calendar.is_business_day(cursor):
            remaining -= 1
    return cursor


def build_roll_schedule(
    bars: pd.DataFrame,
    *,
    symbol: str = "CL",
    policy: RollPolicy | None = None,
    calendar: BusinessDayCalendar | None = None,
) -> pd.DataFrame:
    """Build a CL roll schedule from dated-contract bars and volume.

    ``volume_crossover`` and ``volume_crossover_with_calendar_guard`` use daily
    summed volume only.  Open interest is intentionally not inferred from volume
    because the current minute lake does not store OI.
    """

    if symbol.upper() != "CL":
        raise NotImplementedError("institutional continuous construction is currently implemented for CL only")
    pol = policy or RollPolicy()
    cal = calendar or WeekendHolidayCalendar()
    dated = _standardize_bars(bars)
    expiries = sorted(dated["expiry"].dropna().astype(str).unique())
    expiries = [expiry for expiry in expiries if _is_yyyymm(expiry)]
    events: list[RollEvent] = []
    for front, nxt in zip(expiries, expiries[1:]):
        ltd = cl_last_trade_date(front, calendar=cal)
        forced = subtract_business_days(ltd, pol.forced_roll_business_days_before_last_trade, cal)
        window_start = subtract_business_days(ltd, pol.roll_window_business_days, cal)
        roll_date, trigger, fallback = _select_roll_date(
            dated,
            front,
            nxt,
            policy=pol,
            window_start=window_start,
            forced_roll_date=forced,
        )
        events.append(
            RollEvent(
                front_expiry=front,
                next_expiry=nxt,
                last_trade_date=ltd,
                roll_window_start=window_start,
                forced_roll_date=forced,
                roll_date=roll_date,
                trigger=trigger,
                fallback=fallback,
                policy_name=pol.name,
                policy_version=pol.version,
            )
        )
    return pd.DataFrame([event.__dict__ for event in events])


def construct_continuous_series(
    bars: pd.DataFrame,
    *,
    symbol: str = "CL",
    policy: RollPolicy | None = None,
    adjustment: AdjustmentMethod = "additive",
    calendar: BusinessDayCalendar | None = None,
    front_month_guard: FrontMonthGuard | None = None,
) -> ContinuousResult:
    """Construct raw or adjusted continuous bars from dated contracts."""

    pol = policy or RollPolicy()
    guard = front_month_guard or FrontMonthGuard()
    dated = _standardize_bars(bars)
    dated = dated[dated["expiry"].str.fullmatch(r"\d{6}", na=False)].copy()
    if dated.empty:
        empty = _empty_continuous_frame(adjustment, pol)
        return ContinuousResult(
            bars=empty,
            schedule=pd.DataFrame(),
            metadata=_metadata(symbol, pol, adjustment, [], pd.DataFrame(), pd.DataFrame(), guard, calendar),
        )

    coverage = evaluate_front_month_coverage(
        dated,
        symbol=symbol,
        guard=guard,
        calendar=calendar,
    )
    _enforce_front_month_coverage(coverage, symbol=symbol, guard=guard)
    schedule = build_roll_schedule(dated, symbol=symbol, policy=pol, calendar=calendar)
    active = _select_active_rows(dated, schedule)
    coverage = evaluate_front_month_coverage(
        dated,
        symbol=symbol,
        guard=guard,
        calendar=calendar,
        selected=active[["ts", "active_expiry"]],
    )
    _enforce_front_month_coverage(coverage, symbol=symbol, guard=guard)
    gaps = _roll_gaps(dated, schedule)
    schedule = _enrich_schedule_with_gaps(schedule, gaps)
    active = _apply_adjustment(active, schedule, gaps, adjustment)
    active = _attach_lineage(active, schedule, gaps, pol, adjustment)
    active = _attach_front_month_coverage(active, coverage, guard)
    metadata = _metadata(symbol, pol, adjustment, sorted(dated["expiry"].unique()), schedule, coverage, guard, calendar)
    return ContinuousResult(bars=active, schedule=schedule, metadata=metadata)


def evaluate_front_month_coverage(
    bars: pd.DataFrame,
    *,
    symbol: str = "CL",
    guard: FrontMonthGuard | None = None,
    calendar: BusinessDayCalendar | None = None,
    selected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return per-session front-month coverage diagnostics."""

    active_guard = guard or FrontMonthGuard()
    if not _guard_applies(symbol, active_guard):
        return pd.DataFrame(
            [
                {
                    "symbol": symbol.upper(),
                    "front_month_guard_status": "skipped",
                    "reason": "guard disabled or symbol not configured",
                }
            ]
        )

    dated = _standardize_bars(bars)
    dated = dated[dated["expiry"].str.fullmatch(r"\d{6}", na=False)].copy()
    if dated.empty:
        return pd.DataFrame(
            [
                {
                    "symbol": symbol.upper(),
                    "front_month_guard_status": "invalid",
                    "reason": "no dated YYYYMM expiries available",
                    "available_expiries": [],
                    "eligible_expiries": [],
                }
            ]
        )

    selected_by_ts: dict[pd.Timestamp, str] = {}
    if selected is not None and not selected.empty:
        sel = selected.copy()
        sel["ts"] = pd.to_datetime(sel["ts"], utc=True)
        expiry_col = "active_expiry" if "active_expiry" in sel else "expiry"
        selected_by_ts = dict(zip(sel["ts"], sel[expiry_col].astype(str), strict=False))

    rows: list[dict[str, object]] = []
    for session_date, session in dated.groupby("session_date", sort=True):
        available = sorted(session["expiry"].dropna().astype(str).unique())
        eligible = front_month_eligible_expiries(
            symbol,
            session_date,
            guard=active_guard,
            calendar=calendar,
        )
        eligible_set = set(eligible)
        eligible_available = [expiry for expiry in available if expiry in eligible_set]
        selected_expiries = sorted(
            {
                selected_by_ts.get(ts)
                for ts in session["ts"]
                if selected_by_ts.get(ts) is not None
            }
        )
        invalid_selected = [expiry for expiry in selected_expiries if expiry not in eligible_set]
        status = "valid"
        reason = ""
        if not eligible_available:
            status = "invalid"
            reason = "missing eligible near/front contract"
        elif invalid_selected:
            status = "invalid"
            reason = "selected expiry is outside eligible front-month range"
        rows.append(
            {
                "symbol": symbol.upper(),
                "session_date": session_date,
                "start_ts": session["ts"].min(),
                "end_ts": session["ts"].max(),
                "available_expiries": available,
                "eligible_expiries": eligible,
                "eligible_available_expiries": eligible_available,
                "selected_expiries": selected_expiries,
                "invalid_selected_expiries": invalid_selected,
                "front_month_guard_status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def front_month_eligible_expiries(
    symbol: str,
    session_date: date,
    *,
    guard: FrontMonthGuard | None = None,
    calendar: BusinessDayCalendar | None = None,
) -> list[str]:
    """Return the eligible front curve range for a session."""

    active_guard = guard or FrontMonthGuard()
    front = front_delivery_month(symbol, session_date, calendar=calendar)
    count = max(1, int(active_guard.max_curve_position))
    return [_format_yyyymm(_add_months(front, offset)) for offset in range(count)]


def front_delivery_month(
    symbol: str,
    session_date: date,
    *,
    calendar: BusinessDayCalendar | None = None,
) -> date:
    """Return the expected front delivery month for a session.

    CL uses the last-trade-date convention. Other monthly products fall back to
    a simple delivery-month calendar that can be replaced by product-specific
    logic as coverage expands.
    """

    month = date(session_date.year, session_date.month, 1)
    if symbol.upper() == "CL":
        for offset in range(36):
            candidate = _add_months(month, offset)
            if cl_last_trade_date(candidate, calendar=calendar) >= session_date:
                return candidate
        raise ValueError(f"could not determine CL front month for session {session_date}")

    if session_date.day == 1:
        return month
    return _add_months(month, 1)


def _select_roll_date(
    bars: pd.DataFrame,
    front: str,
    nxt: str,
    *,
    policy: RollPolicy,
    window_start: date,
    forced_roll_date: date,
) -> tuple[date, str, bool]:
    if policy.name == "last_trade_minus_n_business_days":
        return forced_roll_date, "calendar_forced", False

    if policy.name not in {"volume_crossover", "volume_crossover_with_calendar_guard"}:
        raise ValueError(f"unsupported roll policy {policy.name!r}")

    trigger = _volume_crossover_date(
        bars,
        front,
        nxt,
        window_start=window_start,
        forced_roll_date=forced_roll_date,
        consecutive_sessions=policy.volume_crossover_sessions,
    )
    if trigger is not None:
        return trigger, "volume_crossover", False
    if policy.name == "volume_crossover":
        return forced_roll_date, "volume_crossover_missing_forced", True
    return forced_roll_date, "calendar_guard_forced", True


def _volume_crossover_date(
    bars: pd.DataFrame,
    front: str,
    nxt: str,
    *,
    window_start: date,
    forced_roll_date: date,
    consecutive_sessions: int,
) -> date | None:
    window = bars[
        (bars["expiry"].isin([front, nxt]))
        & (bars["session_date"] >= window_start)
        & (bars["session_date"] <= forced_roll_date)
    ]
    if window.empty:
        return None
    daily = (
        window.groupby(["session_date", "expiry"], as_index=False)["v"]
        .sum()
        .pivot(index="session_date", columns="expiry", values="v")
        .fillna(0.0)
        .sort_index()
    )
    streak = 0
    needed = max(1, int(consecutive_sessions))
    for session, row in daily.iterrows():
        if float(row.get(nxt, 0.0)) > float(row.get(front, 0.0)):
            streak += 1
        else:
            streak = 0
        if streak >= needed:
            return session
    return None


def _select_active_rows(bars: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        active_expiry = sorted(bars["expiry"].unique())[0]
        active = bars[bars["expiry"] == active_expiry].copy()
        active["active_expiry"] = active_expiry
        return active

    roll_pairs = [
        (row.front_expiry, row.next_expiry, row.roll_date)
        for row in schedule.itertuples(index=False)
    ]
    first_expiry = roll_pairs[0][0]

    def active_for_session(session: date) -> str:
        active = first_expiry
        for _front, nxt, roll_date in roll_pairs:
            if session >= roll_date:
                active = nxt
            else:
                break
        return active

    bars = bars.copy()
    bars["active_expiry"] = bars["session_date"].map(active_for_session)
    active = bars[bars["expiry"] == bars["active_expiry"]].copy()
    return active.sort_values(["ts", "expiry"]).drop_duplicates("ts", keep="first")


def _roll_gaps(bars: pd.DataFrame, schedule: pd.DataFrame) -> dict[date, dict[str, float | str | bool | None]]:
    gaps: dict[date, dict[str, float | str | bool | None]] = {}
    for row in schedule.itertuples(index=False):
        front = bars[bars["expiry"] == row.front_expiry][["ts", "c"]].rename(columns={"c": "front_c"})
        nxt = bars[bars["expiry"] == row.next_expiry][["ts", "c"]].rename(columns={"c": "next_c"})
        common = front.merge(nxt, on="ts", how="inner")
        common = common[pd.to_datetime(common["ts"], utc=True).dt.date >= row.roll_date].sort_values("ts")
        if common.empty:
            gap = None
            factor = None
        else:
            first = common.iloc[0]
            front_c = float(first["front_c"])
            next_c = float(first["next_c"])
            gap = next_c - front_c
            factor = next_c / front_c if front_c > 0 and next_c > 0 else None
        gaps[row.roll_date] = {
            "front_expiry": row.front_expiry,
            "next_expiry": row.next_expiry,
            "gap": gap,
            "factor": factor,
            "fallback": bool(row.fallback),
        }
    return gaps


def _enrich_schedule_with_gaps(
    schedule: pd.DataFrame,
    gaps: dict[date, dict[str, float | str | bool | None]],
) -> pd.DataFrame:
    if schedule.empty:
        return schedule
    out = schedule.copy()
    out["roll_gap"] = out["roll_date"].map(lambda roll_date: gaps.get(roll_date, {}).get("gap"))
    out["roll_ratio"] = out["roll_date"].map(lambda roll_date: gaps.get(roll_date, {}).get("factor"))
    out["roll_return"] = out["roll_ratio"].map(
        lambda factor: float(factor) - 1.0 if factor is not None and not pd.isna(factor) else pd.NA
    )
    out["roll_carry"] = out["roll_gap"]
    return out


def _apply_adjustment(
    active: pd.DataFrame,
    schedule: pd.DataFrame,
    gaps: dict[date, dict[str, float | str | bool | None]],
    adjustment: AdjustmentMethod,
) -> pd.DataFrame:
    out = active.copy()
    out["adjustment_offset"] = 0.0
    out["adjustment_factor"] = 1.0
    if schedule.empty or adjustment == "raw":
        return out
    if adjustment == "additive":
        for row in schedule.itertuples(index=False):
            gap = gaps.get(row.roll_date, {}).get("gap")
            if gap is not None:
                out.loc[out["session_date"] < row.roll_date, "adjustment_offset"] += float(gap)
        for col in OHLC_COLUMNS:
            out[col] = out[col] + out["adjustment_offset"]
        return out
    if adjustment == "ratio":
        for row in schedule.itertuples(index=False):
            factor = gaps.get(row.roll_date, {}).get("factor")
            if factor is None or not (0.2 <= float(factor) <= 5.0):
                raise ValueError(
                    f"unstable or unavailable ratio adjustment at roll {row.front_expiry}->{row.next_expiry}"
                )
            out.loc[out["session_date"] < row.roll_date, "adjustment_factor"] *= float(factor)
        for col in OHLC_COLUMNS:
            out[col] = out[col] * out["adjustment_factor"]
        return out
    raise ValueError("adjustment must be 'raw', 'additive', or 'ratio'")


def _attach_lineage(
    active: pd.DataFrame,
    schedule: pd.DataFrame,
    gaps: dict[date, dict[str, float | str | bool | None]],
    policy: RollPolicy,
    adjustment: AdjustmentMethod,
) -> pd.DataFrame:
    out = active.copy()
    out["roll_date"] = pd.NaT
    out["next_expiry"] = pd.NA
    out["roll_gap"] = pd.NA
    out["roll_fallback"] = False
    if not schedule.empty:
        events = list(schedule.itertuples(index=False))

        def next_event(session: date):
            for event in events:
                if session < event.roll_date:
                    return event
            return None

        event_by_session = out["session_date"].map(next_event)
        out["roll_date"] = event_by_session.map(lambda event: event.roll_date if event is not None else pd.NaT)
        out["next_expiry"] = event_by_session.map(lambda event: event.next_expiry if event is not None else pd.NA)
        out["roll_gap"] = event_by_session.map(
            lambda event: gaps.get(event.roll_date, {}).get("gap") if event is not None else pd.NA
        )
        out["roll_fallback"] = event_by_session.map(lambda event: bool(event.fallback) if event is not None else False)
    out["adjustment_method"] = adjustment
    out["roll_policy"] = policy.name
    out["roll_policy_version"] = policy.version
    out["expiry"] = out["active_expiry"]
    ordered = [
        "ts",
        "symbol",
        "expiry",
        "active_expiry",
        "o",
        "h",
        "l",
        "c",
        "v",
        "roll_date",
        "next_expiry",
        "roll_gap",
        "adjustment_offset",
        "adjustment_factor",
        "adjustment_method",
        "roll_policy",
        "roll_policy_version",
        "roll_fallback",
    ]
    return out.sort_values("ts")[ordered].reset_index(drop=True)


def _attach_front_month_coverage(
    active: pd.DataFrame,
    coverage: pd.DataFrame,
    guard: FrontMonthGuard,
) -> pd.DataFrame:
    if coverage.empty or not _guard_applies(str(active["symbol"].dropna().iloc[0]) if not active.empty else "", guard):
        return active
    by_session = coverage.set_index("session_date")
    out = active.copy()
    out["front_month_valid"] = out["ts"].dt.date.map(
        lambda session: by_session.at[session, "front_month_guard_status"] == "valid"
        if session in by_session.index
        else False
    )
    out["front_month_eligible_expiries"] = out["ts"].dt.date.map(
        lambda session: by_session.at[session, "eligible_expiries"] if session in by_session.index else []
    )
    out["front_month_available_expiries"] = out["ts"].dt.date.map(
        lambda session: by_session.at[session, "available_expiries"] if session in by_session.index else []
    )
    out["front_month_guard_reason"] = out["ts"].dt.date.map(
        lambda session: by_session.at[session, "reason"] if session in by_session.index else "missing coverage diagnostic"
    )
    return out


def _metadata(
    symbol: str,
    policy: RollPolicy,
    adjustment: AdjustmentMethod,
    expiries: list[str],
    schedule: pd.DataFrame,
    coverage: pd.DataFrame,
    guard: FrontMonthGuard,
    calendar: BusinessDayCalendar | None,
) -> dict[str, object]:
    calendar_name = _calendar_name(calendar)
    return {
        "symbol": symbol.upper(),
        "construction": "institutional_continuous",
        "product": "CL",
        "adjustment_method": adjustment,
        "policy_name": policy.name,
        "policy_version": policy.version,
        "roll_window_business_days": policy.roll_window_business_days,
        "forced_roll_business_days_before_last_trade": policy.forced_roll_business_days_before_last_trade,
        "volume_crossover_sessions": policy.volume_crossover_sessions,
        "source_expiries": expiries,
        "roll_count": int(schedule.shape[0]),
        "front_month_guard": {
            "enabled": guard.enabled,
            "symbols": list(guard.symbols),
            "max_curve_position": guard.max_curve_position,
            "on_missing": guard.on_missing,
        },
        "front_month_coverage": _coverage_summary(coverage),
        "calendar": calendar_name,
        "calendar_limitation": (
            "NYMEX observed-calendar mode encodes full-close holidays for the 2025-2026 research window. "
            "Inject a fuller exchange calendar before extending holiday-sensitive production rolls outside "
            "that observed window."
        ),
        "open_interest": "not_available_in_minute_lake; no OI crossover is inferred",
    }


def _calendar_name(calendar: BusinessDayCalendar | None) -> str:
    if calendar is None:
        return "weekend_only"
    return str(getattr(calendar, "name", calendar.__class__.__name__))


def _coverage_summary(coverage: pd.DataFrame) -> dict[str, object]:
    if coverage.empty:
        return {
            "status": "not_checked",
            "invalid_session_count": 0,
            "sessions_checked": 0,
        }
    if "front_month_guard_status" not in coverage:
        return {
            "status": "not_checked",
            "invalid_session_count": 0,
            "sessions_checked": 0,
        }
    invalid = coverage[coverage["front_month_guard_status"] == "invalid"]
    status = "valid" if invalid.empty else "invalid"
    first_invalid = invalid.iloc[0].to_dict() if not invalid.empty else {}
    return {
        "status": status,
        "invalid_session_count": int(invalid.shape[0]),
        "sessions_checked": int(coverage.shape[0]),
        "first_invalid": _jsonable_coverage_row(first_invalid),
    }


def _jsonable_coverage_row(row: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if hasattr(value, "isoformat"):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


def _enforce_front_month_coverage(
    coverage: pd.DataFrame,
    *,
    symbol: str,
    guard: FrontMonthGuard,
) -> None:
    if not _guard_applies(symbol, guard) or guard.on_missing == "mark" or coverage.empty:
        return
    invalid = coverage[coverage["front_month_guard_status"] == "invalid"]
    if invalid.empty:
        return
    first = invalid.iloc[0]
    raise FrontMonthCoverageError(
        "front-month coverage guard failed for "
        f"{symbol.upper()} on {first.get('session_date')}: {first.get('reason')}. "
        f"Available expiries: {first.get('available_expiries')}; "
        f"eligible front range: {first.get('eligible_expiries')}; "
        f"selected invalid expiries: {first.get('invalid_selected_expiries')}. "
        "Backfill the missing near/front dated contracts or use a fixed expiry/vendor continuous source explicitly."
    )


def _guard_applies(symbol: str, guard: FrontMonthGuard) -> bool:
    return guard.enabled and symbol.upper() in {item.upper() for item in guard.symbols}


def _standardize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"ts", "expiry", "o", "h", "l", "c", "v"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"continuous construction missing required columns: {sorted(missing)}")
    out = bars.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    if "symbol" in out:
        out["symbol"] = out["symbol"].astype(str).str.upper()
    else:
        out["symbol"] = ""
    out["expiry"] = out["expiry"].astype(str)
    for col in (*OHLC_COLUMNS, "v"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["ts", "expiry", "o", "h", "l", "c"]).sort_values(["ts", "expiry"])
    out["session_date"] = out["ts"].dt.date
    return out.reset_index(drop=True)


def _empty_continuous_frame(adjustment: AdjustmentMethod, policy: RollPolicy) -> pd.DataFrame:
    columns = [
        "ts",
        "symbol",
        "expiry",
        "active_expiry",
        "o",
        "h",
        "l",
        "c",
        "v",
        "roll_date",
        "next_expiry",
        "roll_gap",
        "adjustment_offset",
        "adjustment_factor",
        "adjustment_method",
        "roll_policy",
        "roll_policy_version",
        "roll_fallback",
    ]
    frame = pd.DataFrame(columns=columns)
    frame["adjustment_method"] = adjustment
    frame["roll_policy"] = policy.name
    frame["roll_policy_version"] = policy.version
    return frame


def _delivery_month_start(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return date(value.year, value.month, 1)
    if isinstance(value, date):
        return date(value.year, value.month, 1)
    text = str(value)
    if not _is_yyyymm(text):
        raise ValueError(f"delivery month must be YYYYMM, got {value!r}")
    return date(int(text[:4]), int(text[4:6]), 1)


def _format_yyyymm(day: date) -> str:
    return f"{day.year:04d}{day.month:02d}"


def _add_months(day: date, months: int) -> date:
    idx = day.year * 12 + day.month - 1 + months
    return date(idx // 12, idx % 12 + 1, 1)


def _is_yyyymm(value: str) -> bool:
    return len(value) == 6 and value.isdigit() and 1 <= int(value[4:6]) <= 12
