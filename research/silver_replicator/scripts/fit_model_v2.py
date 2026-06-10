"""
Grid-search the v2 BB-regime + vol-of-vol extensions across all four
timeframes, holding the prior winning trend-layer params fixed.

Writes:
    artifacts/fit_grid_v2_{tf}.parquet
    artifacts/fit_top10_v2.csv
    artifacts/best_params_v2.json

Includes a hard sanity-check that the new code reproduces the prior v1
baseline (composite ~0.7451 on 8H) when both new switches are False.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys
import time

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
from src.signal_grammar import SignalParams, v2_extension_grid


TFS = ["1H", "4H", "8H", "1D"]

# Trend layer is locked to the prior v1 winner.
FIXED_TREND = dict(
    fast=50,
    slow=200,
    rsi_long_thr=50.0,
    rsi_short_thr=45.0,
    use_macd=True,
    use_adx=True,
    adx_min=25.0,
    atr_max=0.04,
    pattern_lookback=3,
    use_pattern_boost=False,
)


def _v2_combos():
    grid = v2_extension_grid()
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, vals))
        combos.append({**FIXED_TREND, **d})
    return combos


def _baseline_check(tf: str = "8H") -> float:
    """Both new switches False -> must reproduce v1 composite (~0.7451 on 8H)."""
    bars = load_bars(tf)
    feats = load_features(tf)
    trades = load_trades()
    pos = reconstruct_position_path(trades, bars.index)
    pnl = realised_pnl_curve(trades, bars.index)
    p = SignalParams(**FIXED_TREND, use_bb_regime=False, use_vov_trigger=False)
    sig = generate_signal_path(feats, p)
    sc = score(sig, pos, pnl, bars)
    c = composite(sc)
    expected = 0.7451
    assert abs(c - expected) < 1e-3, (
        f"baseline sanity-check FAILED: {c:.4f} != {expected:.4f} (delta="
        f"{c - expected:+.4f}). The new code path is no longer identical to v1."
    )
    return c


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
        except Exception:  # noqa: BLE001
            sc = dict(direction_accuracy=0.0, cohens_kappa=0.0,
                      pearson_pos=0.0, pnl_curve_corr=0.0,
                      sim_pnl_total=0.0, actual_pnl_total=0.0,
                      n_bars=0, sim_scale=1.0)
        row = dict(tf=tf, **combo, **sc)
        row["composite"] = composite(sc)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    print("Sanity check: reproducing v1 baseline on 8H...")
    c0 = _baseline_check("8H")
    print(f"  v1 baseline composite reproduced: {c0:.4f}\n")

    combos = _v2_combos()
    print(f"v2 combos per TF: {len(combos)} (x{len(TFS)} TFs)\n")

    all_results = []
    for tf in TFS:
        t0 = time.perf_counter()
        df = fit_tf(tf, combos)
        df.to_parquet(ART / f"fit_grid_v2_{tf}.parquet", index=False)
        elapsed = time.perf_counter() - t0

        # Baseline row(s) for this TF = both switches off
        base = df[(~df["use_bb_regime"]) & (~df["use_vov_trigger"])]
        base_c = float(base["composite"].max()) if len(base) else float("nan")
        ext = df[(df["use_bb_regime"]) | (df["use_vov_trigger"])]
        ext_c = float(ext["composite"].max()) if len(ext) else float("nan")

        print(f"=== {tf} | {elapsed:.1f}s | combos={len(df)} ===")
        print(f"  BASELINE (both off) best composite: {base_c:.4f}")
        print(f"  EXTENDED (any switch on) best composite: {ext_c:.4f}  "
              f"delta={ext_c - base_c:+.4f}")
        top5 = df.sort_values("composite", ascending=False).head(5)
        cols = ["use_bb_regime", "use_vov_trigger", "bb_period",
                "bb_regime_thr_pct", "bb_width_pctl_lookback",
                "vov_window", "vov_zscore_thr", "vov_action",
                "direction_accuracy", "cohens_kappa", "pnl_curve_corr",
                "sim_pnl_total", "composite"]
        print(top5[cols].round(4).to_string(index=False))
        print()
        all_results.append(df)

    big = pd.concat(all_results, ignore_index=True)
    top10 = big.sort_values("composite", ascending=False).head(10)
    top10.to_csv(ART / "fit_top10_v2.csv", index=False)

    best = top10.iloc[0].to_dict()
    defaults = SignalParams().as_dict()
    best_params_keys = list(defaults.keys())
    best_params = {k: best.get(k, defaults[k]) for k in best_params_keys}

    # cast numpy scalars to plain python
    for k, v in list(best_params.items()):
        if isinstance(v, (np.integer,)):
            best_params[k] = int(v)
        elif isinstance(v, (np.floating,)):
            best_params[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            best_params[k] = bool(v)

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
    with open(ART / "best_params_v2.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nBest v2 overall: tf={payload['tf']} composite={payload['scores']['composite']:.4f}")
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
