import pandas as pd

from scripts.research.kuma_trend_lib_v0 import KumaConfig, run_kuma_trend_backtest


def _panel() -> pd.DataFrame:
    ts = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"]
    )
    data = []
    for symbol, closes in [("AAA-USD", [10, 11, 12, 13, 14, 15]), ("BBB-USD", [10, 9, 8, 7, 6, 5])]:
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
    df = pd.DataFrame(data).set_index(["symbol", "ts"])
    return df


def test_kuma_trend_lib_smoke():
    panel = _panel()
    cfg = KumaConfig(
        breakout_lookback=2,
        fast_ma=2,
        slow_ma=3,
        atr_window=2,
        vol_window=2,
        cash_yield_annual=0.0,
        cash_buffer=0.0,
        atr_k=2.0,
    )
    weights_df, equity_df, positions = run_kuma_trend_backtest(panel, cfg)

    assert not equity_df.empty
    assert {"ts", "portfolio_ret", "portfolio_equity", "turnover"}.issubset(equity_df.columns)
    assert not weights_df.empty
    assert not positions.empty
