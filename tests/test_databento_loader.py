"""Hermetic tests for the Databento CL loader (no network)."""
from __future__ import annotations

import argparse
import importlib.util
from datetime import timezone
from pathlib import Path

import pandas as pd
import pytest

from volbook.databento_loader import (
    DatabentoLoader,
    cl_raw_symbol,
    contract_window,
    enumerate_cl_expiries,
    populated_expiries,
    _df_to_bars,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cli_module():
    """Import the backfill CLI script by path (it lives under scripts/)."""
    path = _REPO_ROOT / "scripts" / "volbook" / "backfill_databento_cl.py"
    spec = importlib.util.spec_from_file_location("backfill_databento_cl", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_cl_raw_symbol_month_codes():
    assert cl_raw_symbol("202001") == "CLF0"   # Jan 2020
    assert cl_raw_symbol("201912") == "CLZ9"   # Dec 2019
    assert cl_raw_symbol("202012") == "CLZ0"   # Dec 2020
    assert cl_raw_symbol("202406") == "CLM4"   # Jun 2024
    assert cl_raw_symbol("201007") == "CLN0"   # Jul 2010


def test_cl_raw_symbol_validates():
    with pytest.raises(ValueError):
        cl_raw_symbol("2020")
    with pytest.raises(ValueError):
        cl_raw_symbol("202013")


def test_enumerate_cl_expiries_every_month_with_year_wrap():
    out = enumerate_cl_expiries("201911", "202002")
    assert out == ["201911", "201912", "202001", "202002"]
    assert enumerate_cl_expiries("202002", "202001") == []  # reversed -> empty


def test_contract_window_precedes_last_trade_and_is_utc():
    # CL Jan-2020 last-trade is in Dec-2019; window opens 45d earlier, closes day after.
    start, end = contract_window("202001", window_days=45)
    assert start.tzinfo is not None and start.utcoffset() == timezone.utc.utcoffset(None)
    assert start < end
    assert (end - start).days == 46
    # last-trade for CLF0 is ~2019-12-19; window should sit in late-2019.
    assert start.year == 2019 and start.month in (11, 12)
    assert end.year == 2019 and end.month == 12


def test_df_to_bars_maps_fields_and_utc_iso():
    idx = pd.to_datetime(["2020-01-02 00:00:00", "2020-01-02 00:01:00"], utc=True)
    df = pd.DataFrame(
        {"open": [50.1, 50.3], "high": [50.4, 50.5], "low": [50.0, 50.2],
         "close": [50.3, 50.4], "volume": [120.0, 88.0]},
        index=idx,
    )
    bars = _df_to_bars(df)
    assert len(bars) == 2
    assert bars[0].o == 50.1 and bars[0].c == 50.3 and bars[0].v == 120.0
    assert bars[0].t.startswith("2020-01-02T00:00:00")
    assert "+00:00" in bars[0].t  # tz-aware UTC


def test_df_to_bars_localizes_naive_index_to_utc():
    idx = pd.to_datetime(["2020-01-02 00:00:00"])  # tz-naive
    df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0]}, index=idx)
    bars = _df_to_bars(df)
    assert "+00:00" in bars[0].t


class _FakeMetadata:
    def __init__(self, costs=None):
        self.calls = []
        # costs: mapping raw_symbol -> usd; default flat 2.5
        self._costs = costs or {}

    def get_dataset_range(self, *, dataset):
        return {"start": "2010-06-06T00:00:00+00:00", "end": "2026-05-30T00:00:00+00:00"}

    def get_cost(self, *, dataset, symbols, schema, stype_in, start, end, mode=None):
        self.calls.append((symbols[0], start, end, mode))
        return float(self._costs.get(symbols[0], 2.5))

    def list_unit_prices(self, *, dataset):
        return [
            {"mode": "historical-streaming", "unit_prices": {"ohlcv-1m": 19.0, "trades": 99.0}},
            {"mode": "batch", "unit_prices": {"ohlcv-1m": 5.0}},
        ]


class _FakeClient:
    def __init__(self, costs=None):
        self.metadata = _FakeMetadata(costs=costs)


class _FakeLake:
    """Minimal stand-in for MinuteLake.populated_expiries introspection."""

    def __init__(self, expiries):
        self._expiries = list(expiries)
        self.bars_table = "bars_1m"

    def connect(self):
        return self

    def execute(self, sql, params):
        symbol, min_rows = params
        rows = [(e,) for e in self._expiries]
        return _FakeResult(rows)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


def test_estimate_cost_sums_and_clamps_before_dataset_start():
    client = _FakeClient()
    loader = DatabentoLoader(client=client)
    # 201001..201008 straddles the 2010-06-06 dataset start.
    quote = loader.estimate_cost("201001", "201008", window_days=45, clamp_to_dataset=True)
    priced = [r for r in quote["breakdown"] if not r.get("skipped")]
    skipped = [r for r in quote["breakdown"] if r.get("skipped")]
    # Early contracts whose window ends before the dataset start are skipped (cost 0).
    assert skipped, "expected some pre-dataset-start contracts to be skipped"
    assert quote["total_cost_usd"] == pytest.approx(2.5 * len(priced))
    # Every priced request's start is clamped to >= dataset start.
    for sym, start, end, _mode in client.metadata.calls:
        assert start >= "2010-06-06"


def test_loader_requires_key_without_client(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        DatabentoLoader()


def test_get_cost_passes_streaming_mode():
    client = _FakeClient()
    loader = DatabentoLoader(client=client)
    loader.estimate_cost("202401", "202402", window_days=45, clamp_to_dataset=True)
    assert client.metadata.calls, "expected get_cost calls"
    assert all(mode == "historical-streaming" for *_x, mode in client.metadata.calls)


def test_unit_price_reads_streaming_ohlcv_rate():
    loader = DatabentoLoader(client=_FakeClient())
    assert loader.unit_price_usd_per_gb() == 19.0


def test_unit_price_falls_back_when_missing():
    class _NoPrices(_FakeClient):
        def __init__(self):
            super().__init__()
            self.metadata.list_unit_prices = lambda *, dataset: []

    loader = DatabentoLoader(client=_NoPrices())
    assert loader.unit_price_usd_per_gb() == 280.0


def test_plan_within_budget_selects_newest_prefix_under_cap():
    # 4 contracts, $2.50 each; cap fits exactly 3 -> the 3 newest selected.
    client = _FakeClient()
    loader = DatabentoLoader(client=client)
    plan = loader.plan_within_budget(
        "202401", "202404", cap_usd=7.5, window_days=45, order="newest",
    )
    selected = [r["expiry"] for r in plan["selected"]]
    deferred = [r["expiry"] for r in plan["deferred"]]
    assert selected == ["202404", "202403", "202402"]
    assert deferred == ["202401"]
    assert plan["planned_cost_usd"] <= plan["cap_usd"]
    assert plan["full_cost_usd"] == pytest.approx(10.0)


def test_plan_within_budget_oldest_order():
    loader = DatabentoLoader(client=_FakeClient())
    plan = loader.plan_within_budget("202401", "202404", cap_usd=5.0, window_days=45, order="oldest")
    assert [r["expiry"] for r in plan["selected"]] == ["202401", "202402"]


def test_plan_within_budget_never_exceeds_cap_with_many_contracts():
    loader = DatabentoLoader(client=_FakeClient())
    plan = loader.plan_within_budget("202001", "202412", cap_usd=11.0, window_days=45, order="newest")
    assert plan["planned_cost_usd"] <= plan["cap_usd"]
    # $2.5 each, cap 11 -> at most 4 selected.
    assert len(plan["selected"]) == 4


def test_plan_excludes_given_expiries():
    loader = DatabentoLoader(client=_FakeClient())
    plan = loader.plan_within_budget(
        "202401", "202404", cap_usd=100.0, window_days=45,
        exclude_expiries=["202403", "202401"],
    )
    chosen = {r["expiry"] for r in plan["selected"]}
    assert chosen == {"202404", "202402"}
    assert plan["excluded_count"] == 2


def test_populated_expiries_reads_lake():
    lake = _FakeLake(["202407", "202408", "202409"])
    got = populated_expiries(lake)
    assert got == {"202407", "202408", "202409"}


def test_cli_resolve_credit_flag_and_env(monkeypatch):
    cli = _load_cli_module()
    monkeypatch.delenv("DATABENTO_CREDIT_USD", raising=False)
    assert cli._resolve_credit(argparse.Namespace(credit_usd=30.0)) == 30.0
    monkeypatch.setenv("DATABENTO_CREDIT_USD", "25")
    assert cli._resolve_credit(argparse.Namespace(credit_usd=None)) == 25.0
    monkeypatch.delenv("DATABENTO_CREDIT_USD", raising=False)
    assert cli._resolve_credit(argparse.Namespace(credit_usd=None)) is None


def test_cli_download_refuses_without_credit(monkeypatch, capsys):
    cli = _load_cli_module()
    monkeypatch.delenv("DATABENTO_CREDIT_USD", raising=False)
    args = argparse.Namespace(
        credit_usd=None, safety_frac=0.10, safety_usd=1.0,
        start="202401", end="202402", window_days=45,
        lake_path="/tmp/should-not-be-touched.duckdb", order="newest",
        skip_existing=True, no_clamp=False, manifest=None,
        confirm_usd=None, confirm_tol=0.01,
    )
    rc = cli._do_download(loader=None, args=args)
    assert rc == 2
    assert "REFUSING" in capsys.readouterr().out
