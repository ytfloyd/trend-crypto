import pandas as pd

from scripts.research.transtrend_crypto_simple_baseline_lib import (
    SimpleBaselineConfig,
    build_equal_weights,
    compute_signals,
    simulate_portfolio,
)


def _panel() -> pd.DataFrame:
    ts = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    data = []
    for symbol, closes in [("AAA-USD", [10, 11, 12, 13]), ("BBB-USD", [10, 9, 8, 7])]:
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


def test_equal_weight_signals_and_equity():
    panel = _panel()
    panel = compute_signals(panel, fast=2, slow=3)
    weights = build_equal_weights(panel)

    # By 2020-01-04, AAA-USD signal should be on, BBB-USD off.
    latest = weights[weights["ts"] == pd.Timestamp("2020-01-04")]
    w_map = dict(zip(latest["symbol"], latest["w_signal"]))
    assert w_map.get("AAA-USD", 0.0) == 1.0
    assert w_map.get("BBB-USD", 0.0) == 0.0

    cfg = SimpleBaselineConfig(fast_ma=2, slow_ma=3, cost_bps=0.0, execution_lag_bars=1)
    equity_df, weights_held = simulate_portfolio(panel, weights, cfg)

    # Equity should be cumulative product of (1 + portfolio_ret)
    expected_equity = (1 + equity_df["portfolio_ret"]).cumprod()
    pd.testing.assert_series_equal(
        equity_df["portfolio_equity"].reset_index(drop=True),
        expected_equity.reset_index(drop=True),
        check_names=False,
    )

    # Weights held should be lagged by one bar
    assert weights_held["ts"].min() == equity_df["ts"].min()
