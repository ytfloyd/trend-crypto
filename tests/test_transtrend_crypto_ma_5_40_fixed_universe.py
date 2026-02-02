import pandas as pd

from scripts.research.transtrend_crypto_ma_5_40_fixed_universe_lib import (
    MA540FixedUniverseConfig,
    build_equal_weights,
    compute_signals,
    simulate_portfolio,
)


def _panel() -> pd.DataFrame:
    ts = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"])
    data = []
    for symbol, closes in [("AAA-USD", [10, 11, 12, 13, 14]), ("BBB-USD", [10, 9, 8, 7, 6])]:
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


def test_weights_sum_and_gross_exposure():
    panel = _panel()
    panel = compute_signals(panel, fast=2, slow=3)
    weights = build_equal_weights(panel)

    for _, group in weights.groupby("ts"):
        total = group["w_signal"].sum()
        if total > 0:
            assert abs(total - 1.0) < 1e-9
        else:
            assert abs(total) < 1e-9

    cfg = MA540FixedUniverseConfig(fast_ma=2, slow_ma=3, cost_bps=0.0, execution_lag_bars=1)
    equity_df, _ = simulate_portfolio(panel, weights, cfg)
    assert (equity_df["gross_exposure"] <= 1.0 + 1e-9).all()


def test_model_b_shift():
    panel = _panel()
    panel = compute_signals(panel, fast=2, slow=3)
    weights = build_equal_weights(panel)

    cfg = MA540FixedUniverseConfig(fast_ma=2, slow_ma=3, cost_bps=0.0, execution_lag_bars=1)
    equity_df, weights_held = simulate_portfolio(panel, weights, cfg)

    first_ts = equity_df["ts"].iloc[0]
    first_weights = weights_held[weights_held["ts"] == first_ts]["w_held"]
    assert (first_weights.abs() < 1e-12).all()
