"""Tests for the dated-walker's chunk-level retry behavior.

The walker must distinguish "IB returned 0 bars" (= reached data head,
stop walking) from "IB request timed out" (= transient, retry). A bug
in the original implementation conflated both as ``[]`` and prematurely
ended the walk for ESM4 (Jun'24) when TWS hiccuped during the overnight
ES backfill.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts.volbook import walk_dated_futures_minute as walker
from volbook.ibkr_client import HistoricalDataTimeout, IBHistoricalClient


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the timeout backoff in tests so they run instantly."""
    monkeypatch.setattr(walker, "_TIMEOUT_BACKOFF_SECONDS", 0.0)


class _FakeClient:
    """Stand-in for ``IBHistoricalClient`` that reuses the real retry method.

    The walker now delegates retries to
    :meth:`IBHistoricalClient.fetch_dated_minute_bars_with_retry`, so we
    rebind that method here with a scripted ``fetch_dated_minute_bars_by_contract``
    and ``reconnect_if_needed``. This way the test exercises the real
    retry/backoff logic instead of duplicating it in the fixture.
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
        next_behavior = self._behaviors.pop(0)
        if isinstance(next_behavior, BaseException):
            raise next_behavior
        return next_behavior

    def reconnect_if_needed(self) -> bool:
        self.reconnects += 1
        return False


def _now_utc() -> datetime:
    return datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


def test_contract_history_upper_bound_uses_qualified_last_trade_date() -> None:
    contract = SimpleNamespace(lastTradeDateOrContractMonth="20250922")

    assert walker._contract_history_upper_bound(contract, "202510") == datetime(
        2025, 9, 23, tzinfo=timezone.utc
    )


def test_contract_history_upper_bound_falls_back_to_month_end() -> None:
    contract = SimpleNamespace(lastTradeDateOrContractMonth="202510")

    assert walker._contract_history_upper_bound(contract, "202510") == datetime(
        2025, 11, 1, tzinfo=timezone.utc
    )


def test_fetch_chunk_returns_bars_first_try() -> None:
    fake_bars = ["bar0", "bar1", "bar2"]
    client = _FakeClient([fake_bars])

    result = walker._fetch_chunk_with_retry(
        client=client,
        contract=SimpleNamespace(localSymbol="ESM4"),
        end_cursor=_now_utc(),
        duration="30 D",
        what_to_show="TRADES",
        use_rth=False,
        label="ES Jun'24",
    )
    assert result == fake_bars
    assert len(client.calls) == 1
    assert client.reconnects == 0


def test_fetch_chunk_returns_empty_without_retry() -> None:
    """Empty list (= reached head) is a normal terminal — no retry."""
    client = _FakeClient([[]])

    result = walker._fetch_chunk_with_retry(
        client=client,
        contract=SimpleNamespace(localSymbol="ESM4"),
        end_cursor=_now_utc(),
        duration="30 D",
        what_to_show="TRADES",
        use_rth=False,
        label="ES Jun'24",
    )
    assert result == []
    assert len(client.calls) == 1
    assert client.reconnects == 0


def test_fetch_chunk_retries_on_timeout_then_succeeds() -> None:
    fake_bars = ["bar0"]
    client = _FakeClient(
        [
            HistoricalDataTimeout("simulated timeout 1"),
            HistoricalDataTimeout("simulated timeout 2"),
            fake_bars,
        ]
    )

    result = walker._fetch_chunk_with_retry(
        client=client,
        contract=SimpleNamespace(localSymbol="ESM4"),
        end_cursor=_now_utc(),
        duration="30 D",
        what_to_show="TRADES",
        use_rth=False,
        label="ES Jun'24",
    )
    assert result == fake_bars
    assert len(client.calls) == 3
    assert client.reconnects == 2


def test_fetch_chunk_raises_after_max_timeouts() -> None:
    """Persistent timeouts must surface as ``HistoricalDataTimeout``."""
    client = _FakeClient(
        [HistoricalDataTimeout(f"sim {i}") for i in range(walker._TIMEOUT_RETRY_ATTEMPTS)]
    )

    with pytest.raises(HistoricalDataTimeout):
        walker._fetch_chunk_with_retry(
            client=client,
            contract=SimpleNamespace(localSymbol="ESM4"),
            end_cursor=_now_utc(),
            duration="30 D",
            what_to_show="TRADES",
            use_rth=False,
            label="ES Jun'24",
        )
    assert len(client.calls) == walker._TIMEOUT_RETRY_ATTEMPTS


def test_fetch_chunk_passes_through_non_timeout_errors() -> None:
    boom = ValueError("not a timeout")
    client = _FakeClient([boom])

    with pytest.raises(ValueError):
        walker._fetch_chunk_with_retry(
            client=client,
            contract=SimpleNamespace(localSymbol="ESM4"),
            end_cursor=_now_utc(),
            duration="30 D",
            what_to_show="TRADES",
            use_rth=False,
            label="ES Jun'24",
        )
    assert len(client.calls) == 1
    assert client.reconnects == 0
