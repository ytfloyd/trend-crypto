"""Unit tests for the BofA primer-style robustness anchors added to
``src/analysis/tearsheet.py``.

Each helper is exercised directly with deterministic synthetic data so the
tests are tight and don't depend on the full quintile pipeline.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from analysis.tearsheet import (
    _annual_breakdown,
    _attach_fed_phase,
    _crowding_series,
    _drawdown_stats,
    _hit_rate,
    _load_fed_cycles,
    _nw_tstat,
    _regime_breakdown,
    _rolling_percentile,
    generate_tearsheet,
)


def test_hit_rate_basic():
    arr = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02])
    assert _hit_rate(arr) == pytest.approx(3 / 6)


def test_hit_rate_skips_nans():
    arr = np.array([0.01, np.nan, -0.02, 0.03])
    assert _hit_rate(arr) == pytest.approx(2 / 3)


def test_hit_rate_empty_returns_zero():
    assert _hit_rate(np.array([])) == 0.0
    assert _hit_rate(np.array([np.nan, np.nan])) == 0.0


def test_nw_tstat_iid_matches_simple_tstat():
    rng = np.random.default_rng(seed=42)
    arr = rng.normal(loc=0.001, scale=0.01, size=2000)
    nw_t, nw_lags = _nw_tstat(arr)
    simple_t = arr.mean() / (arr.std(ddof=1) / math.sqrt(arr.size))
    assert nw_lags >= 1
    assert abs(nw_t - simple_t) < 0.5


def test_nw_tstat_persistent_series_dampens_tstat():
    rng = np.random.default_rng(seed=7)
    n = 2000
    eps = rng.normal(0, 0.01, size=n)
    arr = np.zeros(n)
    arr[0] = eps[0]
    rho = 0.7
    for i in range(1, n):
        arr[i] = rho * arr[i - 1] + eps[i]
    arr += 0.005

    simple_t = arr.mean() / (arr.std(ddof=1) / math.sqrt(arr.size))
    nw_t, nw_lags = _nw_tstat(arr)
    assert nw_lags >= 1
    assert abs(nw_t) < abs(simple_t)


def test_nw_tstat_short_series_returns_zero():
    assert _nw_tstat(np.array([0.01, 0.02])) == (0.0, 0)


def test_drawdown_stats_simple_curve():
    eq = np.array([1.0, 1.10, 1.20, 0.90, 0.85, 0.95, 1.30])
    ts = [datetime(2024, 1, i + 1) for i in range(len(eq))]
    dd = _drawdown_stats(eq, ts)
    assert dd["max_dd"] == pytest.approx(0.85 / 1.20 - 1.0)
    assert dd["peak_ts"] == str(ts[2])
    assert dd["trough_ts"] == str(ts[4])
    assert dd["recovery_ts"] == str(ts[6])
    assert dd["drawdown_periods"] == 2
    assert dd["recovery_periods"] == 2


def test_drawdown_stats_no_recovery_yet():
    eq = np.array([1.0, 1.30, 0.80])
    ts = [datetime(2024, 1, i + 1) for i in range(len(eq))]
    dd = _drawdown_stats(eq, ts)
    assert dd["recovery_ts"] is None
    assert dd["recovery_periods"] is None
    assert dd["max_dd"] == pytest.approx(0.80 / 1.30 - 1.0)


def test_annual_breakdown_partitions_by_year():
    ts = [datetime(2023, 6, 1), datetime(2023, 7, 1), datetime(2024, 1, 1), datetime(2024, 2, 1)]
    spread_df = pl.DataFrame({"ts": ts, "Spread": [0.01, 0.02, -0.01, -0.02]})
    rows = _annual_breakdown(spread_df, periods_per_year=12.0)
    by_year = {r["year"]: r for r in rows}
    assert set(by_year) == {2023, 2024}
    assert by_year[2023]["mean"] == pytest.approx(0.015)
    assert by_year[2024]["mean"] == pytest.approx(-0.015)
    assert by_year[2023]["hit_rate"] == 1.0
    assert by_year[2024]["hit_rate"] == 0.0
    assert by_year[2023]["n"] == 2


def test_load_fed_cycles_and_phase_attach(tmp_path):
    csv = tmp_path / "fed.csv"
    csv.write_text(
        "start_ts,end_ts,phase,notes\n"
        "2022-03-01,2023-02-28,early_hike,a\n"
        "2023-03-01,2023-07-31,late_hike,b\n"
        "2023-08-01,2024-08-31,neutral,c\n"
        "2024-09-01,2099-12-31,easing,d\n"
    )
    cycles = _load_fed_cycles(csv)
    assert cycles is not None
    assert cycles.height == 4

    spread = pl.DataFrame(
        {
            "ts": [
                datetime(2022, 6, 1),
                datetime(2023, 5, 1),
                datetime(2024, 1, 1),
                datetime(2025, 5, 1),
            ],
            "Spread": [0.01, -0.01, 0.02, -0.02],
        }
    )
    joined = _attach_fed_phase(spread, cycles)
    phases = joined["phase"].to_list()
    assert phases == ["early_hike", "late_hike", "neutral", "easing"]


def test_load_fed_cycles_missing_path_returns_none(tmp_path):
    assert _load_fed_cycles(None) is None
    assert _load_fed_cycles(tmp_path / "nope.csv") is None


def test_regime_breakdown_segments_returns(tmp_path):
    csv = tmp_path / "fed.csv"
    csv.write_text(
        "start_ts,end_ts,phase,notes\n"
        "2022-03-01,2023-02-28,early_hike,a\n"
        "2023-03-01,2099-12-31,easing,b\n"
    )
    cycles = _load_fed_cycles(csv)
    spread = pl.DataFrame(
        {
            "ts": [
                datetime(2022, 6, 1),
                datetime(2022, 12, 1),
                datetime(2023, 6, 1),
                datetime(2023, 12, 1),
            ],
            "Spread": [0.02, 0.04, -0.01, -0.03],
        }
    )
    rows = _regime_breakdown(spread, cycles, periods_per_year=12.0)
    by_phase = {r["phase"]: r for r in rows}
    assert by_phase["early_hike"]["mean"] == pytest.approx(0.03)
    assert by_phase["easing"]["mean"] == pytest.approx(-0.02)
    assert by_phase["early_hike"]["hit_rate"] == 1.0
    assert by_phase["easing"]["hit_rate"] == 0.0
    assert by_phase["early_hike"]["n"] == 2


def test_regime_breakdown_returns_empty_without_cycles():
    spread = pl.DataFrame({"ts": [datetime(2023, 1, 1)], "Spread": [0.01]})
    assert _regime_breakdown(spread, None, periods_per_year=12.0) == []


def test_rolling_percentile_finite_and_in_range():
    rng = np.random.default_rng(seed=99)
    arr = rng.normal(0, 1, size=200)
    pct = _rolling_percentile(arr, window=30)
    assert pct.size == arr.size
    finite = pct[~np.isnan(pct)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0
    assert finite.size >= arr.size - 1


def test_rolling_percentile_monotone_input_ranks_to_one():
    arr = np.arange(100, dtype=float)
    pct = _rolling_percentile(arr, window=20)
    assert pct[-1] == pytest.approx(1.0)


def test_crowding_series_emits_long_basket_pct():
    ts = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
    rows = []
    for t in ts:
        for s in ["A", "B", "C", "D"]:
            sig = {"A": 0.1, "B": 0.4, "C": 0.7, "D": 1.0}[s]
            rows.append({"ts": t, "symbol": s, "signal": sig, "quantile": {"A": 0, "B": 1, "C": 2, "D": 3}[s]})
    df = pl.DataFrame(rows)
    crowding = _crowding_series(df, n_quantiles=4, window=10)
    assert crowding.height == 40
    assert "long_signal" in crowding.columns
    assert "pct_history" in crowding.columns
    long_vals = crowding["long_signal"].to_list()
    assert all(v == pytest.approx(1.0) for v in long_vals)


def test_generate_tearsheet_emits_primer_keys(tmp_path):
    """End-to-end: every new key shows up in summary.json."""
    ts = pl.date_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 4, 1),
        interval="1d",
        eager=True,
    )
    symbols = [f"S{i}" for i in range(8)]
    df = (
        pl.DataFrame({"ts": ts})
        .join(pl.DataFrame({"symbol": symbols}), how="cross")
        .sort(["symbol", "ts"])
    )
    df = df.with_columns(
        [
            (pl.col("ts").rank().over("symbol") % 5).cast(pl.Float64).alias("signal"),
            (pl.col("ts").rank().over("symbol") * 0.001).alias("forward_ret"),
        ]
    )

    fed_csv = tmp_path / "fed.csv"
    fed_csv.write_text(
        "start_ts,end_ts,phase,notes\n"
        "2023-01-01,2024-12-31,easing,test\n"
    )

    out = tmp_path / "alpha_primer"
    summary = generate_tearsheet(
        df,
        str(out),
        alpha_name="alpha_primer",
        fed_cycles_path=fed_csv,
    )

    for key in [
        "hit_rate",
        "spread_nw_tstat",
        "spread_nw_lags",
        "annual_breakdown",
        "regime_breakdown",
        "drawdown",
        "crowding",
    ]:
        assert key in summary, f"missing {key}"

    assert isinstance(summary["annual_breakdown"], list)
    assert isinstance(summary["regime_breakdown"], list)
    assert isinstance(summary["drawdown"], dict)
    assert isinstance(summary["crowding"], dict)
    assert (out.with_suffix(".pdf")).exists()
    assert (out.with_suffix(".png")).exists()
    assert (out.with_name(out.stem + "_p2").with_suffix(".png")).exists()
