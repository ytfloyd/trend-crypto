import pandas as pd

from scripts.research.kuma_01_lib_v0 import (
    apply_dynamic_atr_trailing_stop,
    build_inverse_vol_weights,
    compute_atr30,
    compute_breakout_and_ma_filter,
    compute_vol31,
    simulate_portfolio,
)


def _panel() -> pd.DataFrame:
    ts = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"]
    )
    data = []
    # AAA trends up then drops to trigger stop
    aaa_closes = [10, 11, 12, 13, 14, 6]
    # BBB flat to be non-competitive
    bbb_closes = [10, 10, 10, 10, 10, 10]
    for symbol, closes in [("AAA-USD", aaa_closes), ("BBB-USD", bbb_closes)]:
        for t, close in zip(ts, closes):
            data.append(
                {
                    "symbol": symbol,
                    "ts": t,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1.0,
                }
            )
    return pd.DataFrame(data)


def test_kuma_01_stop_and_weights():
    panel = _panel()
    panel = compute_breakout_and_ma_filter(panel, breakout_lookback=2, fast=2, slow=3)
    panel = compute_atr30(panel, window=2)
    panel = compute_vol31(panel, window=2)
    panel = apply_dynamic_atr_trailing_stop(panel, atr_mult=1.0)

    weights = build_inverse_vol_weights(panel)
    subset = weights[weights["ts"] == pd.Timestamp("2020-01-05")]
    assert subset["w_signal"].sum() in (0.0, 1.0)

    equity_df, weights_held = simulate_portfolio(panel, weights, cost_bps=0.0, execution_lag_bars=1)
    assert (equity_df["gross_exposure"] <= 1.0 + 1e-9).all()

    first_ts = equity_df["ts"].iloc[0]
    first_weights = weights_held[weights_held["ts"] == first_ts]["w_held"]
    assert (first_weights.abs() < 1e-12).all()

    # Stop hit should block eligibility at the hit day (decision for next bar)
    stop_hits = panel[(panel["symbol"] == "AAA-USD") & (panel["stop_hit"])]
    if not stop_hits.empty:
        hit_ts = stop_hits["ts"].iloc[0]
        block_same = panel[(panel["symbol"] == "AAA-USD") & (panel["ts"] == hit_ts)]
        if not block_same.empty:
            assert bool(block_same["stop_block"].iloc[0]) is True
