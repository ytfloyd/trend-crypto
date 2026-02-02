#!/usr/bin/env python
from __future__ import annotations

import pandas as pd

from scripts.research.transtrend_crypto_lib_v0 import (
    HorizonSpec,
    TranstrendConfig,
    build_target_weights,
    compute_trend_scores,
    simulate_portfolio,
)


def run_sleeve(panel: pd.DataFrame, cfg: TranstrendConfig, horizons: list[HorizonSpec], sleeve_name: str) -> dict:
    cfg = TranstrendConfig(
        horizons=horizons,
        target_vol_annual=cfg.target_vol_annual,
        danger_gross=cfg.danger_gross,
        cost_bps=cfg.cost_bps,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        cash_yield_annual=cfg.cash_yield_annual,
        cash_buffer=cfg.cash_buffer,
        max_gross=cfg.max_gross,
        vol_floor=cfg.vol_floor,
        vol_window=cfg.vol_window,
        danger_btc_vol_threshold=cfg.danger_btc_vol_threshold,
        danger_btc_dd20_threshold=cfg.danger_btc_dd20_threshold,
        danger_btc_ret5_threshold=cfg.danger_btc_ret5_threshold,
        execution_lag_bars=cfg.execution_lag_bars,
    )

    panel_scored = compute_trend_scores(panel, cfg)
    weights_signal, danger = build_target_weights(panel_scored, cfg)
    equity_df, weights_held = simulate_portfolio(panel_scored, weights_signal, cfg, danger)

    return {
        "sleeve": sleeve_name,
        "equity_df": equity_df,
        "weights_signal": weights_signal,
        "weights_held": weights_held,
        "danger": danger,
    }


def combine_sleeves(
    equity_fast: pd.DataFrame,
    equity_slow: pd.DataFrame,
    w_fast: float,
    w_slow: float,
) -> pd.DataFrame:
    fast = equity_fast.copy().rename(
        columns={
            "portfolio_ret": "ret_fast",
            "portfolio_equity": "eq_fast",
            "turnover_one_sided": "turn_fast",
            "turnover_two_sided": "turn2_fast",
            "gross_exposure": "gross_fast",
            "danger": "danger_fast",
        }
    )
    slow = equity_slow.copy().rename(
        columns={
            "portfolio_ret": "ret_slow",
            "portfolio_equity": "eq_slow",
            "turnover_one_sided": "turn_slow",
            "turnover_two_sided": "turn2_slow",
            "gross_exposure": "gross_slow",
            "danger": "danger_slow",
        }
    )

    merged = fast.merge(slow, on="ts", how="inner")
    merged = merged.sort_values("ts")

    combined_ret = w_fast * merged["ret_fast"] + w_slow * merged["ret_slow"]
    combined_equity = (1 + combined_ret).cumprod()
    combined_turn = w_fast * merged["turn_fast"] + w_slow * merged["turn_slow"]
    combined_turn2 = w_fast * merged["turn2_fast"] + w_slow * merged["turn2_slow"]
    combined_gross = w_fast * merged["gross_fast"] + w_slow * merged["gross_slow"]
    combined_danger = merged["danger_fast"] | merged["danger_slow"]

    out = pd.DataFrame(
        {
            "ts": merged["ts"],
            "portfolio_ret": combined_ret.values,
            "portfolio_equity": combined_equity.values,
            "turnover_one_sided": combined_turn.values,
            "turnover_two_sided": combined_turn2.values,
            "gross_exposure": combined_gross.values,
            "danger": combined_danger.values,
        }
    )
    return out
