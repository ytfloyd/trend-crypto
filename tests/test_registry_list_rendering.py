import os

import pytest

from scripts.research.strategy_registry_v0 import build_list_row, load_registry


def _get_strategy(strategies, strategy_id):
    for strat in strategies:
        if strat.get("id") == strategy_id:
            return strat
    return None


@pytest.mark.parametrize(
    "strategy_id",
    [
        "ma_5_40_btc_usd_baseline_v0",
        "ma_5_40_eth_usd_baseline_v0",
    ],
)
def test_registry_list_row_has_dates_and_metrics(strategy_id):
    strategies = load_registry()
    strat = _get_strategy(strategies, strategy_id)
    assert strat is not None, f"Missing registry entry: {strategy_id}"

    metrics_csv = strat.get("metrics_csv")
    if not metrics_csv or not os.path.exists(metrics_csv):
        pytest.skip(f"metrics CSV missing for {strategy_id}")

    row = build_list_row(strat)

    assert row["start"], f"start missing for {strategy_id}"
    assert row["end"], f"end missing for {strategy_id}"
    assert row["CAGR"] != "n/a", f"CAGR missing for {strategy_id}"
    assert row["MaxDD"] != "n/a", f"MaxDD missing for {strategy_id}"
