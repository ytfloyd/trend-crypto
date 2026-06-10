"""
Walk-forward CV of the v2 signal grammar at the 8H timeframe (chosen winner TF).

Setup
-----
- bars: 8H, total ~488 in current history.
- TRAIN window = 180 bars (~60 calendar days, 3 bars/day).
- TEST  window =  60 bars (~20 days).
- STEP         =  30 bars (~10 days).
- Warm-up:   start once we have 200+ bars available before the first TRAIN.

For each (train, test) window we grid-search a SMALL subset of the v2
extension dims while holding the trend layer locked to the v1 winner, then
re-score the train-best params on the held-out TEST window.

Outputs
-------
- artifacts/walkforward_folds.parquet
- artifacts/walkforward_summary.json
- figures/07_walkforward.png
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from src.signal_grammar import SignalParams


TF = "8H"

# v1 winner trend-layer (held fixed)
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

# Walk-forward window geometry (in 8H bars)
TRAIN_BARS = 180  # ~60d
TEST_BARS = 60    # ~20d
STEP_BARS = 30    # ~10d
WARMUP_BARS = 200

# Small per-fold grid over the v2 extension dims.
SMALL_GRID = dict(
    bb_regime_thr_pct=[0.15, 0.20, 0.30],
    bb_width_pctl_lookback=[100, 200],
    use_bb_regime=[True, False],
    use_vov_trigger=[True, False],
    vov_zscore_thr=[1.0, 1.5, 2.0],
)

# v2 in-sample composite (from artifacts/best_params_v2.json) for the inset.
V2_IN_SAMPLE_COMPOSITE = 0.7546


def _build_combos():
    keys = list(SMALL_GRID.keys())
    out = []
    for vals in itertools.product(*[SMALL_GRID[k] for k in keys]):
        d = dict(zip(keys, vals))
        out.append({**FIXED_TREND, **d})
    return out


def _score_window(
    feats: pd.DataFrame,
    bars: pd.DataFrame,
    pos: pd.Series,
    pnl: pd.Series,
    combo: dict,
    sl: slice,
) -> dict:
    """Score one parameter combo on one window slice."""
    params = SignalParams(**combo)
    sig = generate_signal_path(feats.iloc[sl], params)
    sc = score(sig, pos.iloc[sl], pnl.iloc[sl], bars.iloc[sl])
    sc["composite"] = composite(sc)
    return sc


def _v2_keys() -> list[str]:
    """Subset of params we record as the chosen combo per fold."""
    return list(SMALL_GRID.keys())


def main() -> int:
    bars = load_bars(TF)
    feats = load_features(TF)
    trades = load_trades()
    pos = reconstruct_position_path(trades, bars.index)
    pnl = realised_pnl_curve(trades, bars.index)
    n = len(bars)
    print(f"loaded {TF}: {n} bars, range {bars.index[0]} -> {bars.index[-1]}")

    combos = _build_combos()
    print(f"per-fold grid size: {len(combos)} combos")

    fold_rows: list[dict] = []
    fold_id = 0
    # walk start indices of TRAIN windows
    # we need: i + TRAIN + TEST <= n, and i >= max(WARMUP_BARS - TRAIN, 0)
    # The "warm-up" semantic: at least WARMUP_BARS of history available *before*
    # the first TEST window. Simplest is: ensure TRAIN starts no earlier than 0
    # AND test starts no earlier than WARMUP_BARS.
    start = max(0, WARMUP_BARS - TRAIN_BARS)
    last_train_start = n - TRAIN_BARS - TEST_BARS
    if last_train_start < start:
        raise SystemExit(
            f"not enough bars: n={n}, need >= {start + TRAIN_BARS + TEST_BARS}"
        )

    t0 = time.perf_counter()
    train_starts = list(range(start, last_train_start + 1, STEP_BARS))
    print(f"running {len(train_starts)} folds...")

    for ts_start in train_starts:
        train_sl = slice(ts_start, ts_start + TRAIN_BARS)
        test_sl = slice(ts_start + TRAIN_BARS, ts_start + TRAIN_BARS + TEST_BARS)

        # In-fold search: best combo on TRAIN by composite
        best_combo = None
        best_train_sc = None
        best_train_c = -np.inf
        for combo in combos:
            try:
                sc = _score_window(feats, bars, pos, pnl, combo, train_sl)
            except Exception:  # noqa: BLE001
                continue
            if sc["composite"] > best_train_c:
                best_train_c = sc["composite"]
                best_combo = combo
                best_train_sc = sc

        if best_combo is None:
            print(f"  fold {fold_id}: no valid combo, skipping")
            fold_id += 1
            continue

        # Score that combo on TEST
        try:
            test_sc = _score_window(feats, bars, pos, pnl, best_combo, test_sl)
        except Exception as e:  # noqa: BLE001
            print(f"  fold {fold_id}: TEST scoring error {e!r}")
            test_sc = dict(direction_accuracy=0.0, cohens_kappa=0.0,
                           pearson_pos=0.0, pnl_curve_corr=0.0,
                           sim_pnl_total=0.0, actual_pnl_total=0.0,
                           n_bars=0, sim_scale=1.0, composite=0.0)

        chosen = {k: best_combo[k] for k in _v2_keys()}
        row = dict(
            fold_id=fold_id,
            train_start=bars.index[train_sl.start],
            train_end=bars.index[train_sl.stop - 1],
            test_start=bars.index[test_sl.start],
            test_end=bars.index[test_sl.stop - 1],
            chosen_params=json.dumps(chosen, default=str),
            train_composite=float(best_train_sc["composite"]),
            test_composite=float(test_sc["composite"]),
            test_dir_acc=float(test_sc["direction_accuracy"]),
            test_kappa=float(test_sc["cohens_kappa"]),
            test_pnl_corr=float(test_sc["pnl_curve_corr"]),
            test_sim_pnl=float(test_sc["sim_pnl_total"]),
            test_actual_pnl=float(test_sc["actual_pnl_total"]),
            test_n_bars=int(test_sc["n_bars"]),
            use_bb_regime=bool(chosen["use_bb_regime"]),
            use_vov_trigger=bool(chosen["use_vov_trigger"]),
        )
        fold_rows.append(row)
        print(f"  fold {fold_id:2d}: train {row['train_start'].date()}..{row['train_end'].date()}"
              f" | test {row['test_start'].date()}..{row['test_end'].date()}"
              f" | train_c={row['train_composite']:.3f} test_c={row['test_composite']:.3f}"
              f" | bb={int(row['use_bb_regime'])} vov={int(row['use_vov_trigger'])}")
        fold_id += 1

    elapsed = time.perf_counter() - t0
    print(f"\nfinished {len(fold_rows)} folds in {elapsed:.1f}s")

    folds_df = pd.DataFrame(fold_rows)
    out_path = ART / "walkforward_folds.parquet"
    folds_df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")

    # ---------------- summary ----------------
    tc = folds_df["test_composite"]
    summary = dict(
        tf=TF,
        n_folds=int(len(folds_df)),
        train_bars=TRAIN_BARS,
        test_bars=TEST_BARS,
        step_bars=STEP_BARS,
        warmup_bars=WARMUP_BARS,
        per_fold_grid_size=len(combos),
        test_composite_mean=float(tc.mean()),
        test_composite_median=float(tc.median()),
        test_composite_std=float(tc.std(ddof=0)),
        test_composite_min=float(tc.min()),
        test_composite_max=float(tc.max()),
        test_dir_acc_mean=float(folds_df["test_dir_acc"].mean()),
        test_kappa_mean=float(folds_df["test_kappa"].mean()),
        test_pnl_corr_mean=float(folds_df["test_pnl_corr"].mean()),
        frac_folds_bb_regime_on=float(folds_df["use_bb_regime"].mean()),
        frac_folds_vov_on=float(folds_df["use_vov_trigger"].mean()),
        v2_in_sample_composite=V2_IN_SAMPLE_COMPOSITE,
    )
    sum_path = ART / "walkforward_summary.json"
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"wrote {sum_path}")
    print(json.dumps(summary, indent=2, default=str))

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(folds_df))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in tc]
    ax.bar(x, tc.values, color=colors, alpha=0.85, label="test composite")
    mean_val = float(tc.mean())
    ax.axhline(mean_val, color="black", linestyle="--", linewidth=1.2,
               label=f"mean = {mean_val:.3f}")
    ax.axhline(V2_IN_SAMPLE_COMPOSITE, color="green", linestyle=":", linewidth=1.2,
               label=f"v2 in-sample = {V2_IN_SAMPLE_COMPOSITE:.3f}")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("fold id")
    ax.set_ylabel("composite (test window)")
    ax.set_title(f"Walk-forward {TF}: TRAIN={TRAIN_BARS} TEST={TEST_BARS} STEP={STEP_BARS} bars")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in folds_df["fold_id"]], rotation=0, fontsize=8)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # inset: chosen-test composite distribution vs v2 in-sample point
    ax2 = ax.inset_axes([0.62, 0.62, 0.34, 0.32])
    ax2.hist(tc.values, bins=12, color="#1f77b4", alpha=0.6, edgecolor="black")
    ax2.axvline(mean_val, color="black", linestyle="--", linewidth=1.0)
    ax2.axvline(V2_IN_SAMPLE_COMPOSITE, color="green", linestyle=":", linewidth=1.4)
    ax2.set_title("test composite dist.\nvs v2 in-sample", fontsize=8)
    ax2.tick_params(labelsize=7)

    fig.tight_layout()
    fig_path = pathlib.Path(PKG) / "figures" / "07_walkforward.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
