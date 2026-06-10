"""Unit tests for the gamma screener pure-function layer.

Covers:
    - FeatureRow filter logic (min spot, IV bounds, bid-ask, ADV)
    - Cross-sectional scoring: direction of signs (cheap IV → higher rank),
      earnings penalty, robustness to NaN features, rank density.

IB snapshotting and DB I/O are not covered here; they require a live TWS
connection. Run them via the runner CLI against a small ticker list.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from screeners.gamma.config import GammaScreenerConfig  # noqa: E402
from screeners.gamma.score import rank_universe  # noqa: E402
from screeners.gamma.signals import FeatureRow  # noqa: E402


def _mk_row(
    symbol: str,
    *,
    iv30: float = 0.25,
    iv7: float = 0.28,
    iv60: float = 0.24,
    iv90: float = 0.24,
    rv_yz20: float = 0.30,
    rv_cc10: float = 0.30,
    rv_cc20: float = 0.30,
    rv_yz60: float = 0.30,
    spot: float = 100.0,
    bid_ask_pct: float = 0.05,
    stock_adv_usd: float = 50_000_000.0,
    earnings: bool = False,
) -> FeatureRow:
    return FeatureRow(
        as_of_date=date(2026, 1, 15),
        symbol=symbol,
        spot=spot,
        iv7=iv7, iv30=iv30, iv60=iv60, iv90=iv90,
        rv_cc10=rv_cc10, rv_cc20=rv_cc20,
        rv_yz20=rv_yz20, rv_yz60=rv_yz60,
        iv30_rv20_ratio=iv30 / rv_yz20,
        iv7_rv10_ratio=iv7 / rv_yz20,  # use rv_yz20 as rv10 proxy in tests
        term_30_90=iv30 - iv90,
        term_7_30=iv7 - iv30,
        skew_25d_30=0.01,
        butterfly_25d_30=0.005,
        iv_rank_252=None,
        bid_ask_pct=bid_ask_pct,
        stock_adv_usd=stock_adv_usd,
        options_adv_usd=5_000_000.0,
        earnings_in_window=earnings,
    )


def test_cheap_iv_ranks_higher() -> None:
    """A ticker with low IV/RV ratio should rank above one with high IV/RV."""
    cfg = GammaScreenerConfig()
    cheap = _mk_row("CHEAP", iv30=0.20, rv_yz20=0.40, iv60=0.20, iv90=0.25)
    mid = _mk_row("MID", iv30=0.30, rv_yz20=0.30, iv60=0.30, iv90=0.30)
    rich = _mk_row("RICH", iv30=0.50, rv_yz20=0.20, iv60=0.45, iv90=0.40)

    scored = rank_universe([cheap, mid, rich], cfg)
    by_sym = {r.symbol: r for r in scored}

    assert by_sym["CHEAP"].rank_combined is not None
    assert by_sym["RICH"].rank_combined is not None
    assert by_sym["CHEAP"].rank_combined < by_sym["RICH"].rank_combined


def test_earnings_penalty_demotes() -> None:
    cfg = GammaScreenerConfig()
    a = _mk_row("A", iv30=0.20, rv_yz20=0.40, iv60=0.22, iv90=0.25, earnings=False)
    b = _mk_row("B", iv30=0.20, rv_yz20=0.40, iv60=0.22, iv90=0.25, earnings=True)
    c = _mk_row("C", iv30=0.35, rv_yz20=0.30, iv60=0.32, iv90=0.30, earnings=False)

    scored = rank_universe([a, b, c], cfg)
    by_sym = {r.symbol: r for r in scored}
    # A should outrank B because of the earnings penalty, all else equal.
    assert by_sym["A"].rank_combined < by_sym["B"].rank_combined


def test_filter_excludes_illiquid_or_bad_quotes() -> None:
    cfg = GammaScreenerConfig(min_stock_adv_usd=20_000_000.0, max_bid_ask_pct=0.10)
    good = _mk_row("GOOD", stock_adv_usd=100_000_000.0, bid_ask_pct=0.03)
    illiquid = _mk_row("ILL", stock_adv_usd=1_000_000.0)
    wide = _mk_row("WIDE", bid_ask_pct=0.50)
    penny = _mk_row("PENNY", spot=1.0)

    scored = rank_universe([good, illiquid, wide, penny], cfg)
    by_sym = {r.symbol: r for r in scored}
    assert by_sym["GOOD"].rank_combined == 1
    assert by_sym["ILL"].rank_combined is None
    assert by_sym["WIDE"].rank_combined is None
    assert by_sym["PENNY"].rank_combined is None


def test_term_structure_sign() -> None:
    """Backwardation (front cheap vs back) should score HIGHER."""
    cfg = GammaScreenerConfig(
        weight_short=0.0, weight_thirty=0.0, weight_term=1.0,
        weight_earnings_penalty=0.0,
    )
    contango = _mk_row("CONT", iv30=0.35, iv90=0.25)   # front rich
    back = _mk_row("BACK", iv30=0.22, iv90=0.30)       # front cheap
    flat = _mk_row("FLAT", iv30=0.28, iv90=0.28)

    scored = rank_universe([contango, back, flat], cfg)
    by_sym = {r.symbol: r for r in scored}
    # BACK has front-cheap, so lowest (iv30 - iv90), so should rank #1
    assert by_sym["BACK"].rank_combined == 1
    assert by_sym["CONT"].rank_combined == 3


def test_ranks_are_dense_and_unique_for_eligible() -> None:
    cfg = GammaScreenerConfig()
    rows = [
        _mk_row(f"T{i}", iv30=0.2 + 0.01 * i, rv_yz20=0.3, iv60=0.22, iv90=0.22)
        for i in range(10)
    ]
    scored = rank_universe(rows, cfg)
    ranks = sorted(r.rank_combined for r in scored if r.rank_combined is not None)
    assert ranks == list(range(1, 11))


def test_empty_input() -> None:
    assert rank_universe([], GammaScreenerConfig()) == []


def test_all_filtered_returns_none_ranks() -> None:
    cfg = GammaScreenerConfig()
    rows = [_mk_row("A", spot=1.0), _mk_row("B", spot=2.0)]  # both below min_spot
    scored = rank_universe(rows, cfg)
    assert all(r.rank_combined is None for r in scored)
    assert all(r.score_combined is None for r in scored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
