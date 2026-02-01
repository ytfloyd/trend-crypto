import pandas as pd

from scripts.research.transtrend_crypto_lib_v1 import (
    HorizonSpec,
    TranstrendConfigV1,
    build_target_weights,
    compute_trend_scores,
    simulate_portfolio,
)


def test_atr_stop_triggers_and_cooldown():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    panel = pd.DataFrame(
        {
            "symbol": ["BTC-USD"] * 10,
            "ts": list(dates),
            "open": [100, 101, 102, 103, 104, 90, 90, 90, 90, 90],
            "high": [101, 102, 103, 104, 105, 91, 91, 91, 91, 91],
            "low": [99, 100, 101, 102, 103, 80, 80, 80, 80, 80],
            "close": [101, 102, 103, 104, 105, 80, 80, 80, 80, 80],
            "volume": [1000] * 10,
        }
    )

    cfg = TranstrendConfigV1(
        horizons=[HorizonSpec("fast", 2, 2, 3)],
        target_vol_annual=0.20,
        danger_gross=0.25,
        cost_bps=20.0,
        vol_floor=0.10,
        vol_window=2,
        atr_window=2,
        atr_k=1.0,
        stop_cooldown_days=2,
        execution_lag_bars=1,
    )

    panel = compute_trend_scores(panel, cfg)
    weights_signal, danger, stop_levels, stop_events = build_target_weights(panel, cfg)

    # stop should trigger after the sharp drop
    assert not stop_events.empty

    weights_wide = weights_signal.pivot(index="ts", columns="symbol", values="w_signal").fillna(0.0)
    stop_dates = stop_events["ts"].sort_values().unique()
    stop_date = pd.to_datetime(stop_dates[0])

    # w_signal should be zero after stop for cooldown period
    post_stop = weights_wide.loc[weights_wide.index > stop_date].head(cfg.stop_cooldown_days)
    assert (post_stop["BTC-USD"] == 0.0).all()

    # Model-B timing: w_held is shifted by 1
    equity_df, weights_held = simulate_portfolio(panel, weights_signal, cfg, danger)
    wide_held = weights_held.pivot(index="ts", columns="symbol", values="w_held")
    assert (wide_held.iloc[1].fillna(0.0) == weights_wide.iloc[0].fillna(0.0)).all()
