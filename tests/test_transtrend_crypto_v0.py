import pandas as pd

from scripts.research.transtrend_crypto_lib_v0 import (
    HorizonSpec,
    TranstrendConfig,
    build_target_weights,
    compute_trend_scores,
    simulate_portfolio,
)


def test_weights_shift_and_danger_gating():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    panel = pd.DataFrame(
        {
            "symbol": ["BTC-USD"] * 6 + ["ETH-USD"] * 6,
            "ts": list(dates) * 2,
            "open": [100, 101, 102, 103, 104, 105] * 2,
            "high": [101, 102, 103, 104, 105, 106] * 2,
            "low": [99, 100, 101, 102, 103, 104] * 2,
            "close": [101, 102, 103, 104, 105, 106] * 2,
            "volume": [1000] * 12,
        }
    )

    cfg = TranstrendConfig(
        horizons=[
            HorizonSpec("fast", 2, 2, 3),
            HorizonSpec("mid", 2, 2, 3),
            HorizonSpec("slow", 2, 2, 3),
        ],
        target_vol_annual=0.20,
        danger_gross=0.25,
        cost_bps=20.0,
        vol_floor=0.10,
        vol_window=2,
        danger_btc_ret5_threshold=1.0,
        execution_lag_bars=1,
    )

    panel = compute_trend_scores(panel, cfg)
    weights_signal, danger = build_target_weights(panel, cfg)
    equity_df, weights_held = simulate_portfolio(panel, weights_signal, cfg, danger)

    # w_held should be lagged by 1 bar
    wide_signal = weights_signal.pivot(index="ts", columns="symbol", values="w_signal")
    wide_held = weights_held.pivot(index="ts", columns="symbol", values="w_held")
    assert (wide_held.iloc[1].fillna(0.0) == wide_signal.iloc[0].fillna(0.0)).all()

    # Ensure danger gating caps gross exposure when danger=True
    weights_wide = weights_signal.pivot(index="ts", columns="symbol", values="w_signal").fillna(0.0)
    gross_by_ts = weights_wide.sum(axis=1)
    danger_ts = danger.reindex(gross_by_ts.index).fillna(False)
    assert danger_ts.any()
    assert (gross_by_ts[danger_ts] <= cfg.danger_gross + 1e-8).all()
