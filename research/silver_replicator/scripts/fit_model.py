"""
Grid-search the SignalGrammar across all four timeframes against the actual
silver book and save the top performers.

Usage:
    python scripts/fit_model.py
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

PKG = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from src.backtest import (
    ART,
    composite,
    generate_signal_path,
    load_bars,
    load_features,
    load_trades,
    realised_pnl_curve,
    reconstruct_position_path,
    score,
)
from src.signal_grammar import SignalParams, default_grid


TFS = ["1H", "4H", "8H", "1D"]


def _grid_combos(grid):
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, vals))
        if d["fast"] >= d["slow"]:
            continue
        combos.append(d)
    return combos


def _slim_combos(combos, cap=4000):
    if len(combos) <= cap:
        return combos
    # deterministic stride
    step = max(1, len(combos) // cap)
    return combos[::step][:cap]


def fit_tf(tf: str, combos: list[dict]) -> pd.DataFrame:
    bars = load_bars(tf)
    feats = load_features(tf)
    trades = load_trades()
    pos = reconstruct_position_path(trades, bars.index)
    pnl = realised_pnl_curve(trades, bars.index)

    rows = []
    for combo in combos:
        params = SignalParams(**combo)
        try:
            sig = generate_signal_path(feats, params)
            sc = score(sig, pos, pnl, bars)
        except Exception as e:  # noqa: BLE001
            sc = dict(direction_accuracy=0.0, cohens_kappa=0.0,
                      pearson_pos=0.0, pnl_curve_corr=0.0,
                      sim_pnl_total=0.0, actual_pnl_total=0.0,
                      n_bars=0, sim_scale=1.0)
        row = dict(tf=tf, **combo, **sc)
        row["composite"] = composite(sc)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main() -> int:
    grid = default_grid()
    combos = _grid_combos(grid)
    print(f"raw combo count: {len(combos)}")
    combos = _slim_combos(combos, cap=4000)
    print(f"using {len(combos)} combos per TF (x{len(TFS)} TFs)")

    all_results = []
    for tf in TFS:
        t0 = time.perf_counter()
        df = fit_tf(tf, combos)
        df.to_parquet(ART / f"fit_grid_{tf}.parquet", index=False)
        all_results.append(df)
        top = df.sort_values("composite", ascending=False).head(5)
        elapsed = time.perf_counter() - t0
        print(f"\n=== {tf} top 5 (composite) | {elapsed:.1f}s ===")
        print(top[
            ["fast", "slow", "rsi_long_thr", "use_macd", "adx_min",
             "direction_accuracy", "cohens_kappa", "pnl_curve_corr",
             "sim_pnl_total", "composite"]
        ].round(4).to_string(index=False))

    big = pd.concat(all_results, ignore_index=True)
    top10 = big.sort_values("composite", ascending=False).head(10)
    top10.to_csv(ART / "fit_top10.csv", index=False)

    best = top10.iloc[0].to_dict()
    best_params_keys = list(SignalParams().as_dict().keys())
    best_params = {k: best[k] for k in best_params_keys}
    payload = dict(
        tf=str(best["tf"]),
        params=best_params,
        scores={
            "direction_accuracy": float(best["direction_accuracy"]),
            "cohens_kappa": float(best["cohens_kappa"]),
            "pearson_pos": float(best["pearson_pos"]),
            "pnl_curve_corr": float(best["pnl_curve_corr"]),
            "sim_pnl_total": float(best["sim_pnl_total"]),
            "actual_pnl_total": float(best["actual_pnl_total"]),
            "composite": float(best["composite"]),
            "n_bars": int(best["n_bars"]),
            "sim_scale": float(best["sim_scale"]),
        },
    )
    with open(ART / "best_params.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nBest overall: tf={payload['tf']} composite={payload['scores']['composite']:.4f}")
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
