"""Tests for the volbook futures universe registry."""
from __future__ import annotations

from datetime import datetime, timezone

from scripts.volbook import refresh_core_futures_daily, refresh_front_futures
from scripts.volbook.refresh_core_futures import (
    DAILY_ONLY_TIMEFRAMES,
    DEFAULT_TIMEFRAMES,
    _attempted_request_count,
    _curve_specs,
    _selected_timeframes,
    build_parser,
)
from volbook.contracts import (
    CORE_MACRO_ALIASES,
    DEFAULT_MONTH_PATTERN,
    KNOWN_FUTURES,
    OPTIONS_UNDERLYING_ALIASES,
    enumerate_dated_specs,
    month_pattern_for_root,
    resolve_futures_spec,
)
from volbook.ibkr_client import _contract_month_from_local_symbol


def test_core_macro_aliases_resolve_to_june_2026_contracts() -> None:
    expected_symbols = {
        "CL",
        "NG",
        "RB",
        "HO",
        "GC",
        "SI",
        "HG",
        "ES",
        "NQ",
        "RTY",
        "ZN",
        "ZB",
        "6E",
        "6J",
    }

    resolved = [resolve_futures_spec(alias=a) for a in CORE_MACRO_ALIASES]

    assert {s.label_symbol for s in resolved} == expected_symbols
    assert all(s.expiry == "202606" for s in resolved)
    assert all(a in KNOWN_FUTURES for a in CORE_MACRO_ALIASES)


def test_core_macro_exchange_mapping() -> None:
    expected = {
        "CL": "NYMEX",
        "NG": "NYMEX",
        "RB": "NYMEX",
        "HO": "NYMEX",
        "GC": "COMEX",
        "SI": "COMEX",
        "HG": "COMEX",
        "ES": "CME",
        "NQ": "CME",
        "RTY": "CME",
        "ZN": "CBOT",
        "ZB": "CBOT",
        "6E": "CME",
        "6J": "CME",
    }

    for alias in CORE_MACRO_ALIASES:
        spec = resolve_futures_spec(alias=alias)
        assert spec.exchange == expected[spec.label_symbol]


def test_options_underlying_aliases_cover_42_markets() -> None:
    expected_symbols = {
        "SR3",
        "ES",
        "ZN",
        "CL",
        "NG",
        "ZF",
        "ZC",
        "ZB",
        "ZT",
        "ZS",
        "GC",
        "ZL",
        "SDA",
        "6E",
        "LE",
        "ZW",
        "HE",
        "ZM",
        "NQ",
        "KE",
        "6J",
        "GF",
        "SME",
        "SI",
        "DC",
        "HG",
        "6A",
        "6B",
        "RTY",
        "HO",
        "6C",
        "GDK",
        "CSC",
        "MET",
        "GNF",
        "CB",
        "6S",
        "PL",
        "MBT",
        "RB",
        "DY",
        "PA",
    }

    resolved = [resolve_futures_spec(alias=a) for a in OPTIONS_UNDERLYING_ALIASES]

    assert len(OPTIONS_UNDERLYING_ALIASES) == 42
    assert {s.label_symbol for s in resolved} == expected_symbols
    assert all(s.expiry == "202606" for s in resolved)
    assert all(a in KNOWN_FUTURES for a in OPTIONS_UNDERLYING_ALIASES)


def test_refresh_defaults_to_options_underlying_universe() -> None:
    args = build_parser().parse_args([])

    assert args.universe == "options-underlyings"
    assert args.aliases is None
    assert args.continuous_only is False
    assert args.daily_only is False
    assert args.hourly_curve_points == 5


def test_ib_symbol_overrides_for_fx_futures() -> None:
    six_e = resolve_futures_spec(alias="6E_JUN26")
    six_j = resolve_futures_spec(alias="6J_JUN26")
    six_a = resolve_futures_spec(alias="6A_JUN26")
    six_b = resolve_futures_spec(alias="6B_JUN26")
    six_c = resolve_futures_spec(alias="6C_JUN26")
    six_s = resolve_futures_spec(alias="6S_JUN26")

    assert six_e.symbol == "EUR"
    assert six_e.label_symbol == "6E"
    assert six_e.trading_class == "6E"
    assert six_j.symbol == "JPY"
    assert six_j.label_symbol == "6J"
    assert six_j.trading_class == "6J"
    assert six_a.symbol == "AUD"
    assert six_a.label_symbol == "6A"
    assert six_a.multiplier == "100000"
    assert six_b.symbol == "GBP"
    assert six_b.label_symbol == "6B"
    assert six_b.multiplier == "62500"
    assert six_c.symbol == "CAD"
    assert six_c.label_symbol == "6C"
    assert six_c.multiplier == "100000"
    assert six_s.symbol == "CHF"
    assert six_s.label_symbol == "6S"
    assert six_s.multiplier == "125000"


def test_silver_alias_uses_full_size_multiplier() -> None:
    si = resolve_futures_spec(alias="SI_JUN26")
    assert si.symbol == "SI"
    assert si.multiplier == "5000"
    assert si.trading_class == "SI"


def test_with_expiry_preserves_display_and_ib_contract_details() -> None:
    six_e = resolve_futures_spec(alias="6E_JUN26")
    next_point = six_e.with_expiry("202609")

    assert next_point.symbol == "EUR"
    assert next_point.label_symbol == "6E"
    assert next_point.trading_class == "6E"
    assert next_point.multiplier == "125000"
    assert next_point.expiry == "202609"
    assert next_point.label == "6E Sep'26"
    assert next_point.key == "6E-202609"


def test_batch_refresh_timeframes_are_daily_and_hourly() -> None:
    assert DEFAULT_TIMEFRAMES == (("1 day", "1 Y"), ("1 hour", "30 D"))


def test_batch_refresh_daily_only_selects_daily_timeframes() -> None:
    args = build_parser().parse_args(["--daily-only"])

    assert args.daily_only is True
    assert DAILY_ONLY_TIMEFRAMES == (("1 day", "1 Y"),)
    assert _selected_timeframes(args) == DAILY_ONLY_TIMEFRAMES


def test_batch_refresh_default_selects_all_timeframes() -> None:
    args = build_parser().parse_args([])

    assert _selected_timeframes(args) == DEFAULT_TIMEFRAMES


def test_attempted_request_count_respects_daily_only_and_hourly_cap() -> None:
    args = build_parser().parse_args(
        [
            "--aliases",
            "CL_JUN26",
            "ES_JUN26",
            "--curve-points",
            "7",
            "--hourly-curve-points",
            "2",
        ]
    )
    daily_args = build_parser().parse_args(
        [
            "--daily-only",
            "--aliases",
            "CL_JUN26",
            "ES_JUN26",
            "--curve-points",
            "7",
            "--hourly-curve-points",
            "2",
        ]
    )

    assert _attempted_request_count(args, _selected_timeframes(args)) == 18
    assert _attempted_request_count(daily_args, _selected_timeframes(daily_args)) == 14


def test_attempted_request_count_respects_continuous_only() -> None:
    args = build_parser().parse_args(
        [
            "--continuous-only",
            "--aliases",
            "CL_JUN26",
            "ES_JUN26",
            "--curve-points",
            "7",
            "--hourly-curve-points",
            "2",
        ]
    )
    daily_args = build_parser().parse_args(
        [
            "--continuous-only",
            "--daily-only",
            "--aliases",
            "CL_JUN26",
            "ES_JUN26",
            "--curve-points",
            "7",
            "--hourly-curve-points",
            "2",
        ]
    )

    assert _attempted_request_count(args, _selected_timeframes(args)) == 4
    assert _attempted_request_count(daily_args, _selected_timeframes(daily_args)) == 2


def test_daily_wrapper_forwards_daily_only(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_refresh_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(refresh_core_futures_daily, "refresh_main", fake_refresh_main)

    assert refresh_core_futures_daily.main(["--replace"]) == 0
    assert refresh_core_futures_daily.main(["--daily-only", "--replace"]) == 0
    assert calls == [["--daily-only", "--replace"], ["--daily-only", "--replace"]]


def test_front_wrapper_forwards_continuous_only(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_refresh_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(refresh_front_futures, "refresh_main", fake_refresh_main)

    assert refresh_front_futures.main(["--no-canvas"]) == 0
    assert refresh_front_futures.main(["--continuous-only", "--no-canvas"]) == 0
    assert calls == [
        ["--continuous-only", "--no-canvas"],
        ["--continuous-only", "--no-canvas"],
    ]


def test_curve_specs_uses_discovery_by_default() -> None:
    base = resolve_futures_spec(alias="CL_JUN26")

    class FakeClient:
        def discover_futures_curve(self, spec, *, limit):
            assert spec == base
            assert limit == 5
            return [base.with_expiry("202605"), base.with_expiry("202606")]

    specs = _curve_specs(
        FakeClient(),
        base,
        curve_points=5,
        fixed_alias_expiry=False,
    )

    assert [s.expiry for s in specs] == ["202605", "202606"]


def test_curve_specs_can_keep_fixed_alias_expiry() -> None:
    base = resolve_futures_spec(alias="CL_JUN26")

    class FakeClient:
        def discover_futures_curve(self, spec, *, limit):  # pragma: no cover
            raise AssertionError("should not discover when fixed_alias_expiry=True")

    specs = _curve_specs(
        FakeClient(),
        base,
        curve_points=5,
        fixed_alias_expiry=True,
    )

    assert specs == [base]


def test_contract_month_from_local_symbol() -> None:
    today = datetime(2026, 4, 24, tzinfo=timezone.utc)

    assert _contract_month_from_local_symbol("CLM6", today=today) == "202606"
    assert _contract_month_from_local_symbol("6EZ6", today=today) == "202612"
    assert _contract_month_from_local_symbol("RTYH7", today=today) == "202703"
    assert _contract_month_from_local_symbol("", today=today) is None


def test_month_pattern_for_root_known_quarterly() -> None:
    assert month_pattern_for_root("ES") == (3, 6, 9, 12)
    assert month_pattern_for_root("ZN") == (3, 6, 9, 12)
    assert month_pattern_for_root("6E") == (3, 6, 9, 12)


def test_month_pattern_for_root_5cycle_and_bimonthly() -> None:
    assert month_pattern_for_root("ZC") == (3, 5, 7, 9, 12)
    assert month_pattern_for_root("GC") == (2, 4, 6, 8, 10, 12)
    assert month_pattern_for_root("LE") == (2, 4, 6, 8, 10, 12)


def test_month_pattern_for_root_unknown_falls_back_to_default() -> None:
    assert month_pattern_for_root("XYZ123") == DEFAULT_MONTH_PATTERN


def test_enumerate_dated_specs_es_quarterly_window() -> None:
    spec = resolve_futures_spec(alias="ES_JUN26")
    out = enumerate_dated_specs(spec, min_expiry="202503", max_expiry="202612")
    assert [s.expiry for s in out] == [
        "202503", "202506", "202509", "202512",
        "202603", "202606", "202609", "202612",
    ]
    assert all(s.symbol == "ES" for s in out)
    assert all(s.exchange == "CME" for s in out)


def test_enumerate_dated_specs_respects_inclusive_bounds() -> None:
    spec = resolve_futures_spec(alias="CL_JUN26")
    out = enumerate_dated_specs(spec, min_expiry="202506", max_expiry="202508")
    assert [s.expiry for s in out] == ["202506", "202507", "202508"]


def test_enumerate_dated_specs_inverted_range_returns_empty() -> None:
    spec = resolve_futures_spec(alias="ES_JUN26")
    assert enumerate_dated_specs(spec, min_expiry="202607", max_expiry="202506") == []
