"""Tests for the convexity-pipeline backtest adapter.

Hermetic: a `FakeProvider` supplies deterministic synthetic OHLCV so the tests
do not depend on the DuckDB lakes. Covers trade extraction, variant dispatch,
cost monotonicity, fold partitioning, universe-drop, regime masking, parameter
perturbation, and `BacktestResult` schema completeness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from convexity_pipeline.adapters import ExistingEngineAdapter
from convexity_pipeline.adapters.existing_engine_adapter import (
    _extract_trades,
    _perturb_params,
    _split_blocks,
)
from convexity_pipeline.types import BacktestResult, Candidate, Hypothesis, PayoffShape, Track


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
class FakeProvider:
    """Returns deterministic synthetic daily bars per symbol."""

    def __init__(self, n: int = 1200, seed: int = 7):
        self.n = n
        self.seed = seed
        self._cache: dict = {}

    def get_bars(self, symbol: str, bar_frequency: str) -> pd.DataFrame:
        if symbol in self._cache:
            return self._cache[symbol].copy()
        rng = np.random.default_rng(self.seed + abs(hash(symbol)) % 1000)
        idx = pd.date_range("2015-01-01", periods=self.n, freq="D")
        drift = 0.0004 + 0.0001 * (abs(hash(symbol)) % 5)
        rets = rng.normal(drift, 0.02, self.n)
        close = 100.0 * np.cumprod(1.0 + rets)
        high = close * (1.0 + np.abs(rng.normal(0, 0.005, self.n)))
        low = close * (1.0 - np.abs(rng.normal(0, 0.005, self.n)))
        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close,
             "volume": rng.uniform(1e6, 5e6, self.n)},
            index=idx,
        )
        df.index.name = "ts"
        self._cache[symbol] = df
        return df.copy()


def _sma_signal(bars: pd.DataFrame, fast: int = 20, slow: int = 100, **_: object) -> pd.Series:
    c = bars["close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)


def _make_candidate(universe=("AAA", "BBB", "CCC")) -> Candidate:
    hyp = Hypothesis(
        name="test_trend",
        statement="trend breakout with trailing stop",
        rationale="momentum trend following",
        expected_payoff_shape=PayoffShape.CONVEX,
        convexity_track=Track.TREND,
        horizon_bars=40,
        universe=list(universe),
        bar_frequency="1d",
        params={"fast": 20, "slow": 100},
        researcher="test",
        registration_date="2024-12-01",
    )
    return Candidate(registry_id="TEST-1", hypothesis=hyp, signal_fn=_sma_signal)


@pytest.fixture()
def adapter() -> ExistingEngineAdapter:
    return ExistingEngineAdapter(provider=FakeProvider(), oos_fold_count=8)


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------
def test_extract_trades_basic():
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    held = pd.Series([0, 1, 1, 1, 0, 1, 1, 0], index=idx, dtype=float)
    net = pd.Series([0.0, 0.01, 0.02, -0.01, 0.0, 0.03, 0.01, 0.0], index=idx)
    pnls, durs = _extract_trades(net, held)
    # Two long runs: bars 1-3 (sum 0.02, dur 3) and bars 5-6 (sum 0.04, dur 2).
    assert len(pnls) == 2
    assert durs.tolist() == [3.0, 2.0]
    assert pnls.iloc[0] == pytest.approx(0.02)
    assert pnls.iloc[1] == pytest.approx(0.04)


def test_extract_trades_handles_short_and_flat():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    held = pd.Series([1, 1, -1, -1, 0, 1], index=idx, dtype=float)
    net = pd.Series([0.01, 0.02, -0.02, 0.01, 0.0, 0.05], index=idx)
    pnls, durs = _extract_trades(net, held)
    # long(2), short(2), long(1)
    assert durs.tolist() == [2.0, 2.0, 1.0]


def test_extract_trades_empty():
    pnls, durs = _extract_trades(pd.Series(dtype=float), pd.Series(dtype=float))
    assert pnls.empty and durs.empty


def test_split_blocks_partition():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    blocks = _split_blocks(idx, 9)
    assert len(blocks) == 9
    # Blocks are contiguous, non-overlapping, and cover the whole index.
    recombined = pd.DatetimeIndex(np.concatenate([b.values for b in blocks]))
    assert recombined.equals(idx)
    assert sum(len(b) for b in blocks) == 100


def test_perturb_params_band():
    base = {"fast": 20, "slow": 100, "mult": 3.0, "flag": True, "name": "x"}
    out = _perturb_params(base, 1.20)
    assert out["fast"] == 24 and out["slow"] == 120
    assert out["mult"] == pytest.approx(3.6)
    assert out["flag"] is True and out["name"] == "x"
    # ints never collapse below 1
    assert _perturb_params({"k": 1}, 0.80)["k"] == 1


# ----------------------------------------------------------------------
# Variant dispatch + schema
# ----------------------------------------------------------------------
def _assert_schema(r: BacktestResult):
    assert isinstance(r.alpha_returns, pd.Series)
    assert isinstance(r.underlying_returns, pd.Series)
    assert isinstance(r.equity, pd.Series)
    assert isinstance(r.trade_pnls, pd.Series)
    assert isinstance(r.trade_durations, pd.Series)
    assert isinstance(r.per_instrument, dict)
    assert isinstance(r.meta, dict)
    # alpha and underlying aligned
    assert r.alpha_returns.index.equals(r.underlying_returns.index)
    assert len(r.equity) == len(r.alpha_returns)


@pytest.mark.parametrize("variant", [
    "stage1_simple_cost", "stage1_pre_cost", "stage2_pre_cost", "stage2_realistic",
    "is", "oos_fold_0", "oos_fold_7", "perturb_0", "perturb_3",
    "regime_bull", "regime_bear", "regime_sideways", "regime_high_vol", "regime_low_vol",
    "universe_drop", "cost_2x",
])
def test_variant_dispatch_schema(adapter, variant):
    r = adapter.run(_make_candidate(), variant)
    _assert_schema(r)
    assert r.meta.get("variant") == variant
    assert len(r.alpha_returns) > 0


def test_per_instrument_populated(adapter):
    r = adapter.run(_make_candidate(), "stage2_realistic")
    assert set(r.per_instrument) == {"AAA", "BBB", "CCC"}
    for sub in r.per_instrument.values():
        _assert_schema(sub)


def test_cost_monotonicity(adapter):
    cand = _make_candidate()
    pre = adapter.run(cand, "stage2_pre_cost").alpha_returns.sum()
    realistic = adapter.run(cand, "stage2_realistic").alpha_returns.sum()
    cost2x = adapter.run(cand, "cost_2x").alpha_returns.sum()
    # More cost -> lower total return (same positions).
    assert pre >= realistic >= cost2x


def test_universe_drop_removes_one(adapter):
    r = adapter.run(_make_candidate(), "universe_drop")
    assert len(r.per_instrument) == 2


def test_oos_folds_disjoint_and_cover(adapter):
    cand = _make_candidate()
    is_idx = adapter.run(cand, "is").alpha_returns.index
    fold_idxs = [adapter.run(cand, f"oos_fold_{i}").alpha_returns.index for i in range(8)]
    all_idx = is_idx
    for f in fold_idxs:
        assert all_idx.intersection(f).empty  # disjoint from earlier blocks
        all_idx = all_idx.union(f)
    # Folds collectively add material out-of-sample coverage beyond IS.
    assert len(all_idx) > len(is_idx) * 5


def test_regime_masks_are_subsets(adapter):
    cand = _make_candidate()
    full = adapter.run(cand, "stage2_realistic").alpha_returns.index
    for name in ("bull", "bear", "high_vol", "low_vol"):
        sub = adapter.run(cand, f"regime_{name}").alpha_returns.index
        assert sub.isin(full).all()
        assert len(sub) <= len(full)


def test_callable_interface_matches_run(adapter):
    cand = _make_candidate()
    a = adapter(cand, "stage2_realistic").alpha_returns.sum()
    b = adapter.run(cand, "stage2_realistic").alpha_returns.sum()
    assert a == pytest.approx(b)
