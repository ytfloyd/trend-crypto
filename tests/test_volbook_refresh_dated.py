"""Tests for the dated-refresh CLI's pure helpers and chunk loop.

The refresh CLI is the forward-walking counterpart to the dated walker;
this module covers the parts that are easy to unit-test without an IB
connection: the active-expiry predicate, the forward end-cursor
sequence, and the chunk loop's behavior on success / 0-bar / timeout
paths.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.volbook import refresh_dated_futures_minute as refresh
from volbook.bundle import Bar
from volbook.contracts import resolve_futures_spec
from volbook.datalake import MinuteLake
from volbook.ibkr_client import HistoricalDataTimeout, IBHistoricalClient


# ---------------------------------------------------------------------------
# is_expiry_active
# ---------------------------------------------------------------------------


def _today(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def test_active_predicate_keeps_current_expiry_month() -> None:
    """A contract whose expiry month equals today's month is always active."""
    assert refresh.is_expiry_active(
        "202506", keep_active_days=14, today=_today(2025, 6, 17)
    )


def test_active_predicate_keeps_future_expiries() -> None:
    assert refresh.is_expiry_active(
        "202509", keep_active_days=14, today=_today(2025, 6, 17)
    )
    assert refresh.is_expiry_active(
        "202612", keep_active_days=14, today=_today(2025, 6, 17)
    )


def test_active_predicate_within_grace_window() -> None:
    """Just-expired contracts stay refreshable for keep_active_days days."""
    # Jun 2025 contract: end-of-month is 2025-07-01. With grace=14 days,
    # cutoff is 2025-07-15 — should still be active on that date.
    assert refresh.is_expiry_active(
        "202506", keep_active_days=14, today=_today(2025, 7, 15)
    )
    # On 2025-07-16, cutoff (2025-07-15) is in the past — inactive.
    assert not refresh.is_expiry_active(
        "202506", keep_active_days=14, today=_today(2025, 7, 16)
    )


def test_active_predicate_filters_dead_expiries() -> None:
    """Contracts that expired well before grace window are skipped."""
    assert not refresh.is_expiry_active(
        "202403", keep_active_days=14, today=_today(2025, 6, 17)
    )
    assert not refresh.is_expiry_active(
        "202206", keep_active_days=14, today=_today(2025, 6, 17)
    )


def test_active_predicate_handles_zero_grace() -> None:
    """With keep_active_days=0, a contract is active only through end of expiry month."""
    assert refresh.is_expiry_active(
        "202506", keep_active_days=0, today=_today(2025, 7, 1)
    )
    assert not refresh.is_expiry_active(
        "202506", keep_active_days=0, today=_today(2025, 7, 2)
    )


def test_active_predicate_rejects_invalid_expiry() -> None:
    assert not refresh.is_expiry_active(
        "continuous", keep_active_days=14, today=_today(2025, 6, 17)
    )
    assert not refresh.is_expiry_active(
        "abc", keep_active_days=14, today=_today(2025, 6, 17)
    )


# ---------------------------------------------------------------------------
# _forward_cursors
# ---------------------------------------------------------------------------


def test_forward_cursors_seed_when_no_state() -> None:
    """No existing data => single trailing chunk anchored at now."""
    now = _today(2025, 7, 15)
    cursors = refresh._forward_cursors(latest_ts=None, now=now, chunk_days=30)
    assert cursors == [now]


def test_forward_cursors_skip_when_already_current() -> None:
    """If latest_ts >= now, nothing to do."""
    now = _today(2025, 7, 15)
    cursors = refresh._forward_cursors(latest_ts=now, now=now, chunk_days=30)
    assert cursors == []
    cursors = refresh._forward_cursors(
        latest_ts=now + timedelta(hours=1), now=now, chunk_days=30
    )
    assert cursors == []


def test_forward_cursors_single_chunk_for_small_gap() -> None:
    """A gap shorter than chunk_days is closed by one trailing chunk at now."""
    now = _today(2025, 7, 15)
    latest = now - timedelta(days=5)
    cursors = refresh._forward_cursors(
        latest_ts=latest, now=now, chunk_days=30
    )
    assert cursors == [now]


def test_forward_cursors_marches_through_long_gap() -> None:
    """Long gaps decompose into chunk_days steps + a trailing now anchor."""
    now = _today(2026, 5, 3)
    latest = _today(2025, 6, 30)
    cursors = refresh._forward_cursors(
        latest_ts=latest, now=now, chunk_days=30
    )
    assert cursors[0] == latest + timedelta(days=30)
    assert cursors[-1] == now
    # Every intermediate cursor must strictly precede now.
    assert all(c < now for c in cursors[:-1])
    # Differences between adjacent cursors must respect the chunk window.
    deltas = [(cursors[i] - cursors[i - 1]).days for i in range(1, len(cursors))]
    assert all(d <= 30 for d in deltas)


# ---------------------------------------------------------------------------
# _refresh_dated_contract end-to-end (against an in-memory lake + fake client)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip pacing/backoff sleeps so tests run instantly."""
    monkeypatch.setattr(refresh, "_TIMEOUT_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(refresh.time, "sleep", lambda *_a, **_k: None)


class _FakeClient:
    """Test double that reuses the real retry method on IBHistoricalClient.

    Behaviors are scripted as a list — each call to
    ``fetch_dated_minute_bars_by_contract`` consumes the next entry. An
    entry that is a list-of-Bar is returned; an entry that is an
    exception is raised.
    """

    fetch_dated_minute_bars_with_retry = (
        IBHistoricalClient.fetch_dated_minute_bars_with_retry
    )

    def __init__(self, behaviors: list) -> None:
        self._behaviors = list(behaviors)
        self.calls: list[datetime] = []
        self.reconnects = 0

    def fetch_dated_minute_bars_by_contract(self, contract, **kwargs):
        self.calls.append(kwargs["end_datetime"])
        if not self._behaviors:
            raise AssertionError("ran out of scripted behaviors")
        nxt = self._behaviors.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    def reconnect_if_needed(self) -> bool:
        self.reconnects += 1
        return False


def _make_bar(ts: datetime) -> Bar:
    return Bar(t=ts.isoformat(), o=1.0, h=1.0, l=1.0, c=1.0, v=1.0)


def _refresh_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        chunk_days=30,
        max_chunks_per_contract=0,
        what_to_show="TRADES",
        use_rth=False,
        pace_seconds=0.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_refresh_skips_when_already_current(tmp_path) -> None:
    """latest_ts >= now (e.g. clock-skewed) means no chunks issued."""
    lake = MinuteLake(tmp_path / "lake.duckdb")
    lake.connect()
    spec = resolve_futures_spec(alias="ES_JUN26")
    # Use a clearly-future timestamp so the predicate definitively skips
    # regardless of how much wall-clock advances between setup and call.
    future_ts = datetime.now(timezone.utc) + timedelta(hours=1)
    lake.upsert_bars("ES", [_make_bar(future_ts)], expiry=spec.expiry)

    client = _FakeClient([])
    failures = refresh._refresh_dated_contract(
        spec=spec,
        contract=SimpleNamespace(localSymbol="ESM6"),
        client=client,
        lake=lake,
        args=_refresh_args(),
    )
    lake.close()
    assert failures == []
    assert client.calls == []


def test_refresh_writes_bars_for_small_gap(tmp_path) -> None:
    """A normal day's gap closes in one chunk."""
    lake = MinuteLake(tmp_path / "lake.duckdb")
    lake.connect()
    spec = resolve_futures_spec(alias="ES_JUN26")
    base = datetime.now(timezone.utc) - timedelta(hours=2)
    lake.upsert_bars("ES", [_make_bar(base)], expiry=spec.expiry)

    new_bar = _make_bar(base + timedelta(minutes=30))
    client = _FakeClient([[new_bar]])

    failures = refresh._refresh_dated_contract(
        spec=spec,
        contract=SimpleNamespace(localSymbol="ESM6"),
        client=client,
        lake=lake,
        args=_refresh_args(),
    )
    state = lake.get_state("ES", expiry=spec.expiry)
    lake.close()

    assert failures == []
    assert len(client.calls) == 1
    assert state is not None and state.latest_ts >= base + timedelta(minutes=30)


def test_refresh_chunk_cap_truncates_walk(tmp_path) -> None:
    """--max-chunks-per-contract trims the cursor list."""
    lake = MinuteLake(tmp_path / "lake.duckdb")
    lake.connect()
    spec = resolve_futures_spec(alias="ES_JUN26")
    # Big gap forces multi-chunk plan
    base = datetime.now(timezone.utc) - timedelta(days=120)
    lake.upsert_bars("ES", [_make_bar(base)], expiry=spec.expiry)

    bars = [_make_bar(base + timedelta(days=i * 10)) for i in range(1, 6)]
    client = _FakeClient([[b] for b in bars])

    failures = refresh._refresh_dated_contract(
        spec=spec,
        contract=SimpleNamespace(localSymbol="ESM6"),
        client=client,
        lake=lake,
        args=_refresh_args(max_chunks_per_contract=2),
    )
    lake.close()
    assert failures == []
    assert len(client.calls) == 2  # capped


def test_refresh_records_timeout_failure_and_bails(tmp_path) -> None:
    """Persistent timeouts surface as RefreshFailure, not silent stop."""
    lake = MinuteLake(tmp_path / "lake.duckdb")
    lake.connect()
    spec = resolve_futures_spec(alias="ES_JUN26")
    base = datetime.now(timezone.utc) - timedelta(days=5)
    lake.upsert_bars("ES", [_make_bar(base)], expiry=spec.expiry)

    client = _FakeClient(
        [HistoricalDataTimeout(f"sim {i}") for i in range(refresh._TIMEOUT_RETRY_ATTEMPTS)]
    )
    failures = refresh._refresh_dated_contract(
        spec=spec,
        contract=SimpleNamespace(localSymbol="ESM6"),
        client=client,
        lake=lake,
        args=_refresh_args(),
    )
    lake.close()
    assert len(failures) == 1
    assert failures[0].stage == "timeout"
    assert failures[0].expiry == spec.expiry
    # All retries exhausted on the same end-cursor.
    assert len(client.calls) == refresh._TIMEOUT_RETRY_ATTEMPTS


def test_refresh_handles_zero_bars_in_quiet_window(tmp_path) -> None:
    """A 0-bar middle chunk just means a quiet window — keep walking."""
    lake = MinuteLake(tmp_path / "lake.duckdb")
    lake.connect()
    spec = resolve_futures_spec(alias="ES_JUN26")
    base = datetime.now(timezone.utc) - timedelta(days=70)
    lake.upsert_bars("ES", [_make_bar(base)], expiry=spec.expiry)

    new_bar = _make_bar(datetime.now(timezone.utc) - timedelta(minutes=30))
    # Three planned chunks: empty, empty, [new_bar]
    client = _FakeClient([[], [], [new_bar]])

    failures = refresh._refresh_dated_contract(
        spec=spec,
        contract=SimpleNamespace(localSymbol="ESM6"),
        client=client,
        lake=lake,
        args=_refresh_args(),
    )
    lake.close()
    assert failures == []
    assert len(client.calls) == 3


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


def test_build_parser_defaults_make_sense() -> None:
    args = refresh.build_parser().parse_args([])
    assert args.universe == "options-underlyings"
    assert args.keep_active_days == 14
    assert args.max_new_expiries == 0
    assert args.chunk_days == 30
    assert args.client_id == 31  # distinct from walker (30) and continuous refresh (29)
    assert args.pace_seconds > 0


def test_build_parser_accepts_aliases() -> None:
    args = refresh.build_parser().parse_args(["--aliases", "ES_JUN26", "CL_JUN26"])
    assert args.aliases == ["ES_JUN26", "CL_JUN26"]
