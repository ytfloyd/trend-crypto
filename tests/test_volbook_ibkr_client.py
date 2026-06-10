"""Unit tests for the volbook IBKR client helpers.

Focuses on the timeout-detection wrapper that distinguishes a genuinely
empty IB response (Error 162: HMDS query returned no data) from a stuck
``reqHistoricalData`` request that ib_insync silently exits with ``[]``.
The walker depends on this distinction to avoid mistaking a timeout for
the head of a contract's data.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from volbook.ibkr_client import HistoricalDataTimeout, IBHistoricalClient


def _build_client_with_mock_ib(
    *,
    return_value=None,
    sleep_seconds: float = 0.0,
    raise_exc: BaseException | None = None,
) -> tuple[IBHistoricalClient, list[dict]]:
    """Construct an IBHistoricalClient with a mock ``_ib.reqHistoricalData``.

    Returns the client plus a list that captures each call's kwargs so
    tests can assert on the request shape.
    """
    calls: list[dict] = []

    def fake_req(*args, **kwargs):
        calls.append(kwargs)
        if sleep_seconds:
            time.sleep(sleep_seconds)
        if raise_exc is not None:
            raise raise_exc
        return [] if return_value is None else return_value

    client = IBHistoricalClient()
    client._ib = SimpleNamespace(reqHistoricalData=fake_req)
    return client, calls


def test_req_minute_bars_returns_bars_unchanged() -> None:
    fake_bars = ["bar0", "bar1"]
    client, calls = _build_client_with_mock_ib(return_value=fake_bars)
    contract = SimpleNamespace(localSymbol="ESM4")

    out = client._req_minute_bars(
        contract,
        end_str="20240601 00:00:00 UTC",
        duration="30 D",
        what_to_show="TRADES",
        use_rth=False,
        timeout=2.0,
    )
    assert out == fake_bars
    assert len(calls) == 1
    assert calls[0]["barSizeSetting"] == "1 min"
    assert calls[0]["durationStr"] == "30 D"
    assert calls[0]["endDateTime"] == "20240601 00:00:00 UTC"


def test_req_minute_bars_empty_fast_response_treated_as_no_data() -> None:
    """An empty response that returns instantly is the head-of-data signal."""
    client, _calls = _build_client_with_mock_ib(return_value=[])
    contract = SimpleNamespace(localSymbol="ESH3")

    out = client._req_minute_bars(
        contract,
        end_str="20230301 00:00:00 UTC",
        duration="30 D",
        what_to_show="TRADES",
        use_rth=False,
        timeout=2.0,
    )
    assert out == []


def test_req_minute_bars_empty_after_full_timeout_raises() -> None:
    """If ib_insync exits empty after the full timeout, raise to caller."""
    client, _calls = _build_client_with_mock_ib(return_value=[], sleep_seconds=0.1)
    contract = SimpleNamespace(localSymbol="ESM4")

    with pytest.raises(HistoricalDataTimeout) as excinfo:
        client._req_minute_bars(
            contract,
            end_str="20240701 00:00:00 UTC",
            duration="30 D",
            what_to_show="TRADES",
            use_rth=False,
            timeout=0.1,
        )
    msg = str(excinfo.value)
    assert "ESM4" in msg
    assert "20240701" in msg


def test_req_minute_bars_uses_explicit_label_in_error() -> None:
    client, _calls = _build_client_with_mock_ib(return_value=[], sleep_seconds=0.05)
    contract = SimpleNamespace(localSymbol="ESM4")

    with pytest.raises(HistoricalDataTimeout) as excinfo:
        client._req_minute_bars(
            contract,
            end_str="",
            duration="30 D",
            what_to_show="TRADES",
            use_rth=False,
            timeout=0.05,
            label="ES Jun'24",
        )
    assert "ES Jun'24" in str(excinfo.value)


def test_req_minute_bars_propagates_underlying_exception() -> None:
    """Connection errors etc. bubble up — only timeouts are wrapped."""
    boom = ConnectionError("socket reset")
    client, _calls = _build_client_with_mock_ib(raise_exc=boom)
    contract = SimpleNamespace(localSymbol="ESM4")

    with pytest.raises(ConnectionError):
        client._req_minute_bars(
            contract,
            end_str="",
            duration="30 D",
            what_to_show="TRADES",
            use_rth=False,
            timeout=2.0,
        )
