import pandas as pd

from scripts.research.transtrend_crypto_ma_5_40_topk_lib import (
    build_topk_weights,
    compute_rank_score,
    compute_signals,
    simulate_portfolio,
)


def _panel(symbols: list[str]) -> pd.DataFrame:
    ts = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
    )
    data = []
    for idx, symbol in enumerate(symbols):
        base = 10 + idx
        closes = [base + i for i in range(len(ts))]
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


def test_topk_weights_and_shift():
    symbols = [f"S{i:02d}-USD" for i in range(25)]
    panel = _panel(symbols)
    panel = compute_signals(panel, fast=2, slow=3)
    panel = compute_rank_score(panel, lookback=2, method="ret")

    # Case 1: >k qualifying (expect exactly k at 1/k each)
    weights = build_topk_weights(panel, k=5)
    subset = weights[weights["ts"] == pd.Timestamp("2020-01-05")]
    assert (subset["w_signal"] > 0).sum() == 5
    assert abs(subset["w_signal"].sum() - 1.0) < 1e-9
    assert (subset["w_signal"][subset["w_signal"] > 0] == 1.0 / 5.0).all()

    # Case 2: 3 qualifying (expect gross 1.0, 1/3 each)
    panel_case2 = panel.copy()
    keep = set(symbols[:3])
    panel_case2.loc[~panel_case2["symbol"].isin(keep), "signal"] = 0.0
    weights_case2 = build_topk_weights(panel_case2, k=5)
    subset2 = weights_case2[weights_case2["ts"] == pd.Timestamp("2020-01-05")]
    assert (subset2["w_signal"] > 0).sum() == 3
    assert abs(subset2["w_signal"].sum() - 1.0) < 1e-9
    assert (subset2["w_signal"][subset2["w_signal"] > 0] == 1.0 / 3.0).all()

    # Case 3: none qualifying (gross 0)
    panel_case3 = panel.copy()
    panel_case3["signal"] = 0.0
    weights_case3 = build_topk_weights(panel_case3, k=5)
    subset3 = weights_case3[weights_case3["ts"] == pd.Timestamp("2020-01-05")]
    assert abs(subset3["w_signal"].sum()) < 1e-12

    equity_df, weights_held = simulate_portfolio(panel, weights, cost_bps=0.0, execution_lag_bars=1)
    assert (equity_df["gross_exposure"] <= 1.0 + 1e-9).all()

    first_ts = equity_df["ts"].iloc[0]
    first_weights = weights_held[weights_held["ts"] == first_ts]["w_held"]
    assert (first_weights.abs() < 1e-12).all()
