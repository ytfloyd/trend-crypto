"""Resume helper: runs the 1D TF if missing and writes aggregate top10 + best_params_v2.json."""
from __future__ import annotations
import itertools, json, pathlib, sys, time
import numpy as np
import pandas as pd

PKG = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from src.backtest import (
    ART, composite, generate_signal_path, load_bars, load_features, load_trades,
    realised_pnl_curve, reconstruct_position_path, score,
)
from src.signal_grammar import SignalParams, v2_extension_grid

FIXED_TREND = dict(
    fast=50, slow=200, rsi_long_thr=50.0, rsi_short_thr=45.0,
    use_macd=True, use_adx=True, adx_min=25.0, atr_max=0.04,
    pattern_lookback=3, use_pattern_boost=False,
)

def _combos():
    grid = v2_extension_grid()
    keys = list(grid.keys())
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, vals))
        out.append({**FIXED_TREND, **d})
    return out


def fit_tf(tf, combos):
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
        except Exception:
            sc = dict(direction_accuracy=0.0, cohens_kappa=0.0, pearson_pos=0.0,
                      pnl_curve_corr=0.0, sim_pnl_total=0.0, actual_pnl_total=0.0,
                      n_bars=0, sim_scale=1.0)
        row = dict(tf=tf, **combo, **sc)
        row["composite"] = composite(sc)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    tfs = ["1H", "4H", "8H", "1D"]
    combos = _combos()
    print(f"combos per TF: {len(combos)}")

    dfs = []
    for tf in tfs:
        p = ART / f"fit_grid_v2_{tf}.parquet"
        if p.exists():
            print(f"[{tf}] cached -> {p}")
            dfs.append(pd.read_parquet(p))
        else:
            print(f"[{tf}] computing...")
            t0 = time.perf_counter()
            df = fit_tf(tf, combos)
            df.to_parquet(p, index=False)
            print(f"  done in {time.perf_counter()-t0:.1f}s")
            dfs.append(df)

    big = pd.concat(dfs, ignore_index=True)
    top10 = big.sort_values("composite", ascending=False).head(10)
    top10.to_csv(ART / "fit_top10_v2.csv", index=False)

    best = top10.iloc[0].to_dict()
    defaults = SignalParams().as_dict()
    best_params_keys = list(defaults.keys())
    best_params = {k: best.get(k, defaults[k]) for k in best_params_keys}
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

    # Per-TF baseline-vs-extended print
    for tf in tfs:
        sub = big[big["tf"] == tf]
        base = sub[(~sub["use_bb_regime"]) & (~sub["use_vov_trigger"])]
        ext = sub[(sub["use_bb_regime"]) | (sub["use_vov_trigger"])]
        bc = float(base["composite"].max()) if len(base) else float("nan")
        ec = float(ext["composite"].max()) if len(ext) else float("nan")
        print(f"[{tf}] baseline={bc:.4f}  extended_best={ec:.4f}  delta={ec-bc:+.4f}")
        print(sub.sort_values("composite", ascending=False).head(5)[
            ["use_bb_regime", "use_vov_trigger", "bb_period", "bb_regime_thr_pct",
             "vov_window", "vov_zscore_thr", "vov_action",
             "direction_accuracy", "cohens_kappa", "pnl_curve_corr", "composite"]
        ].round(4).to_string(index=False))
        print()

    print(f"\nBest overall: tf={payload['tf']} composite={payload['scores']['composite']:.4f}")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
