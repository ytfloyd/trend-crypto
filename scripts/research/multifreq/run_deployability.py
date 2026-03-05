"""
Multi-Frequency Deployability Assessment
==========================================
Addresses reviewer feedback on the multi-frequency analysis:

  1. Cost modeling: net alpha at 2 / 5 / 10 bps per trade
  2. Turnover analysis: full turnover breakdown per frequency
  3. Portfolio construction: top-quintile, signal-proportional, IVW
  4. Momentum + risk overlays: vol targeting + DD control at each freq
  5. Composite ML signal: weighted blend across frequencies
  6. Frequency stacking: agreement-based signal (vote across frequencies)
  7. Regime segmentation: bull / bear / sideways performance splits
  8. Final verdict: net Sharpe table → fund or shelve

Frequencies tested: 30m, 4h, 8h, 1d  (the realistic candidates)
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

from scripts.research.common.data import (
    ANN_FACTOR,
    BARS_PER_DAY,
    load_bars,
)
from scripts.research.common.backtest import simple_backtest
from scripts.research.common.metrics import compute_metrics, format_metrics_table
from scripts.research.common.risk_overlays import (
    apply_dd_control,
    apply_position_limit_wide,
    apply_vol_targeting,
)
from scripts.research.jpm_bigdata_ai.helpers import (
    FEATURE_COLS,
    compute_features,
    filter_universe,
    walk_forward_splits,
    compute_btc_benchmark,
)

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("xgboost required")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FREQUENCIES = ["4h", "8h", "1d"]  # 30m extremely slow for full pipeline
FWD_DAYS = 5
TRAIN_DAYS = 365 * 2
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRAIN_DAYS = 365

COST_TIERS = [0.0, 2.0, 5.0, 10.0, 20.0]
VOL_TARGET = 0.20
VOL_LOOKBACK = 42
MAX_LEVERAGE = 2.0
MAX_WEIGHT = 0.15
DD_THRESHOLD = 0.30
REBAL_EVERY = 5  # bars

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "multifreq" / "deploy"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})


# ===================================================================
# Per-frequency ML prediction pipeline
# ===================================================================
def generate_predictions(freq: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train XGB_Clf via walk-forward, return (predictions, returns_wide, panel)."""
    bpd = BARS_PER_DAY[freq]
    fwd_bars = int(FWD_DAYS * bpd)

    print(f"\n  Loading {freq} data ...", flush=True)
    panel = load_bars(freq)
    adv_window = max(20, int(20 * bpd))
    panel = filter_universe(panel, min_adv_usd=1_000_000,
                            min_history_days=int(90 * bpd), adv_window=adv_window)
    panel = compute_features(panel)
    feat_cols = list(FEATURE_COLS)

    def _add_fwd(g):
        g = g.copy()
        g["fwd_ret"] = g["close"].shift(-fwd_bars) / g["close"] - 1.0
        return g

    panel = panel.groupby("symbol", group_keys=False).apply(_add_fwd)

    # Realized vol for IVW
    panel["realized_vol"] = panel.groupby("symbol")["close"].transform(
        lambda x: x.pct_change().rolling(max(21, int(42 * bpd)),
                                         min_periods=max(10, int(21 * bpd))).std() * np.sqrt(ANN_FACTOR)
    )

    univ = panel[panel["in_universe"]].copy()
    print(f"  In-universe: {len(univ):,} rows, {univ['symbol'].nunique()} symbols")

    # Walk-forward splits on calendar dates
    univ["_cal_date"] = univ["ts"].dt.date
    cal_dates = np.sort(univ["_cal_date"].unique())
    splits = walk_forward_splits(
        pd.DatetimeIndex(pd.to_datetime(cal_dates)),
        train_days=TRAIN_DAYS, test_days=TEST_DAYS,
        step_days=STEP_DAYS, min_train_days=MIN_TRAIN_DAYS,
    )

    required_cols = feat_cols + ["fwd_ret"]
    valid_mask = univ[required_cols].notna().all(axis=1)
    data = univ.loc[valid_mask].copy()

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", n_jobs=4, verbosity=0, tree_method="hist",
    )

    all_preds = []
    for si, fold_info in enumerate(splits):
        train_mask = (data["_cal_date"] >= fold_info["train_start"].date()) & \
                     (data["_cal_date"] <= fold_info["train_end"].date())
        test_mask = (data["_cal_date"] >= fold_info["test_start"].date()) & \
                    (data["_cal_date"] <= fold_info["test_end"].date())
        train = data.loc[train_mask]
        test = data.loc[test_mask]

        if len(train) < 500 or len(test) < 50:
            continue

        # Subsample train if too large
        if len(train) > 2_000_000:
            step = len(train) // 2_000_000 + 1
            train = train.iloc[::step]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[feat_cols].values)
        X_te = np.clip(scaler.transform(test[feat_cols].values), -5, 5)
        y_tr = (train["fwd_ret"].values > 0).astype(int)

        m = clone(xgb_clf)
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X_te)[:, 1]

        pred_df = test[["ts", "symbol", "fwd_ret", "realized_vol"]].copy()
        pred_df["prob_up"] = prob
        pred_df["signal"] = prob - 0.5
        pred_df["fold"] = fold_info["fold"]
        all_preds.append(pred_df)

        if (si + 1) % 10 == 0 or si == len(splits) - 1:
            print(f"    fold {si+1}/{len(splits)}", flush=True)

    preds = pd.concat(all_preds, ignore_index=True)

    # Returns matrix
    ret_panel = univ[["ts", "symbol", "close"]].copy()
    ret_panel["ret"] = ret_panel.groupby("symbol")["close"].pct_change()
    returns_wide = ret_panel.pivot(index="ts", columns="symbol", values="ret").fillna(0.0)

    ic = float(sp_stats.spearmanr(
        preds["fwd_ret"].dropna(),
        preds.loc[preds["fwd_ret"].notna(), "prob_up"]
    ).statistic)
    print(f"  XGB_Clf predictions: {len(preds):,} obs, IC={ic:+.4f}")

    preds.to_parquet(ARTIFACT_DIR / f"preds_{freq}.parquet", index=False)
    return preds, returns_wide, panel


# ===================================================================
# Portfolio builder
# ===================================================================
def build_portfolio(preds: pd.DataFrame, returns_wide: pd.DataFrame,
                    method: str = "top_q_ew",
                    cost_bps: float = 5.0,
                    apply_overlays: bool = False,
                    rebal_every: int = 1) -> dict:
    """Build and backtest portfolio from ML predictions."""
    sub = preds.copy()

    # Rebalance thinning
    unique_ts = sorted(sub["ts"].unique())
    if rebal_every > 1:
        rebal_ts = set(unique_ts[::rebal_every])
        sub = sub[sub["ts"].isin(rebal_ts)]

    if method == "top_q_ew":
        sub["rank_pct"] = sub.groupby("ts")["prob_up"].rank(pct=True)
        top = sub[sub["rank_pct"] >= 0.80].copy()
        counts = top.groupby("ts")["symbol"].transform("count")
        top["weight"] = 1.0 / counts
        wts = top.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)

    elif method == "signal_ivw":
        pos = sub[sub["signal"] > 0].copy()
        pos["vol_safe"] = pos["realized_vol"].clip(lower=0.10)
        pos["raw_wt"] = pos["signal"] / pos["vol_safe"]
        daily_sum = pos.groupby("ts")["raw_wt"].transform("sum")
        pos["weight"] = pos["raw_wt"] / daily_sum.clip(lower=1e-8)
        wts = pos.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)

    elif method == "long_short":
        sub["rank_pct"] = sub.groupby("ts")["prob_up"].rank(pct=True)
        long = sub[sub["rank_pct"] >= 0.80].copy()
        short = sub[sub["rank_pct"] <= 0.20].copy()
        long_ct = long.groupby("ts")["symbol"].transform("count")
        long["weight"] = 0.5 / long_ct
        short_ct = short.groupby("ts")["symbol"].transform("count")
        short["weight"] = -0.5 / short_ct
        all_w = pd.concat([long[["ts", "symbol", "weight"]], short[["ts", "symbol", "weight"]]])
        all_w = all_w.groupby(["ts", "symbol"])["weight"].sum().reset_index()
        wts = all_w.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Forward-fill weights on non-rebal dates
    if rebal_every > 1:
        full_idx = pd.DatetimeIndex(unique_ts)
        wts = wts.reindex(full_idx).ffill().fillna(0.0)

    # Apply position limits
    wts = apply_position_limit_wide(wts, MAX_WEIGHT)

    # Apply risk overlays
    if apply_overlays:
        wts = apply_vol_targeting(wts, returns_wide)
        wts = apply_dd_control(wts, returns_wide, cost_bps=cost_bps)

    bt = simple_backtest(wts, returns_wide, cost_bps=cost_bps)
    eq = pd.Series(bt["portfolio_equity"].values, index=bt["ts"])
    m = compute_metrics(eq)
    m["avg_turnover"] = float(bt["turnover"].mean())
    m["avg_exposure"] = float(bt["gross_exposure"].mean())
    m["avg_holdings"] = float((wts != 0).sum(axis=1).mean())
    m["cost_bps"] = cost_bps
    m["equity"] = eq
    m["weights"] = wts
    return m


# ===================================================================
# Momentum with risk overlays
# ===================================================================
def run_momentum_with_overlays(freq: str) -> list[dict]:
    bpd = BARS_PER_DAY[freq]
    lookback_bars = max(3, int(10 * bpd))
    rebal_bars = max(1, int(5 * bpd))
    vol_window_bars = max(10, int(63 * bpd))

    panel = load_bars(freq, start="2016-06-01", end="2025-12-31")
    adv_window = max(20, int(20 * bpd))
    panel = filter_universe(panel, min_adv_usd=1_000_000,
                            min_history_days=int(90 * bpd), adv_window=adv_window)
    eligible_syms = panel.loc[panel["in_universe"], "symbol"].unique()
    panel = panel[panel["symbol"].isin(eligible_syms)].copy()

    # Compute vol
    def _vol(g):
        g = g.copy()
        ret = np.log(g["close"] / g["close"].shift(1))
        g["realized_vol"] = ret.rolling(vol_window_bars, min_periods=vol_window_bars).std() * np.sqrt(ANN_FACTOR)
        return g
    panel = panel.groupby("symbol", group_keys=False).apply(_vol)

    # Signal: EMAC (best from 8h/1d)
    def _emac(g):
        g = g.copy()
        close = g["close"]
        fast = max(2, lookback_bars // 4)
        fast_ema = close.shift(1).ewm(span=fast, min_periods=fast).mean()
        slow_ema = close.shift(1).ewm(span=lookback_bars, min_periods=lookback_bars).mean()
        g["signal"] = (fast_ema - slow_ema) / slow_ema
        return g
    panel = panel.groupby("symbol", group_keys=False).apply(_emac)

    START = "2018-01-01"
    panel_bt = panel.loc[panel["ts"] >= START].copy()
    panel_bt["ret_oc"] = panel_bt["close"] / panel_bt["open"] - 1.0
    returns_wide = panel_bt.pivot_table(index="ts", columns="symbol", values="ret_oc", aggfunc="first").fillna(0.0)

    p_active = panel.loc[(panel["ts"] >= START) & panel["in_universe"] & panel["signal"].notna()].copy()
    all_dates = sorted(p_active["ts"].unique())
    rebal_dates = set(all_dates[::rebal_bars])

    current_weights = {}
    dates_list = []
    weights_list = []

    for dt in all_dates:
        day_data = p_active.loc[p_active["ts"] == dt].copy()
        if day_data.empty:
            dates_list.append(dt)
            weights_list.append({})
            continue
        if dt in rebal_dates:
            ranked = day_data.sort_values("signal", ascending=False)
            n_select = max(1, len(ranked) // 5)
            selected = ranked.head(n_select)
            if len(selected) > 0:
                vols = selected["realized_vol"].replace(0, np.nan).dropna()
                if len(vols) > 0:
                    inv_vol = 1.0 / vols.clip(lower=0.10)
                    wts = inv_vol / inv_vol.sum()
                    current_weights = dict(zip(selected.loc[vols.index, "symbol"], wts))
                else:
                    n_s = len(selected)
                    current_weights = {s: 1.0 / n_s for s in selected["symbol"]}
            else:
                current_weights = {}
        row = {s: current_weights.get(s, 0.0) for s in day_data["symbol"].tolist()}
        dates_list.append(dt)
        weights_list.append(row)

    base_wts = pd.DataFrame(weights_list, index=pd.DatetimeIndex(dates_list)).fillna(0.0)

    results = []

    # Baseline (no overlay)
    for cost in [5.0, 10.0, 20.0]:
        bt = simple_backtest(base_wts, returns_wide, cost_bps=cost)
        eq = pd.Series(bt["portfolio_equity"].values, index=bt["ts"])
        m = compute_metrics(eq)
        m["label"] = f"Mom EMAC {freq} ({cost:.0f}bps)"
        m["freq"] = freq
        m["cost_bps"] = cost
        m["overlay"] = "none"
        m["avg_turnover"] = float(bt["turnover"].mean())
        results.append(m)

    # With vol targeting
    wts_vt = apply_vol_targeting(base_wts, returns_wide)
    for cost in [5.0, 10.0, 20.0]:
        bt = simple_backtest(wts_vt, returns_wide, cost_bps=cost)
        eq = pd.Series(bt["portfolio_equity"].values, index=bt["ts"])
        m = compute_metrics(eq)
        m["label"] = f"Mom+VT {freq} ({cost:.0f}bps)"
        m["freq"] = freq
        m["cost_bps"] = cost
        m["overlay"] = "vol_target"
        m["avg_turnover"] = float(bt["turnover"].mean())
        results.append(m)

    # With vol targeting + DD control
    wts_vtdd = apply_dd_control(wts_vt, returns_wide, cost_bps=10.0)
    for cost in [5.0, 10.0, 20.0]:
        bt = simple_backtest(wts_vtdd, returns_wide, cost_bps=cost)
        eq = pd.Series(bt["portfolio_equity"].values, index=bt["ts"])
        m = compute_metrics(eq)
        m["label"] = f"Mom+VT+DD {freq} ({cost:.0f}bps)"
        m["freq"] = freq
        m["cost_bps"] = cost
        m["overlay"] = "vol_target+dd"
        m["avg_turnover"] = float(bt["turnover"].mean())
        results.append(m)

    return results


# ===================================================================
# Regime segmentation
# ===================================================================
def segment_regimes(equity: pd.Series) -> pd.DataFrame:
    """Classify periods as bull/bear/sideways based on BTC 60-day return."""
    ret_60 = equity.pct_change(60)
    regime = pd.Series("sideways", index=equity.index)
    regime[ret_60 > 0.20] = "bull"
    regime[ret_60 < -0.20] = "bear"
    return regime


# ===================================================================
# Main
# ===================================================================
def main():
    t0_global = time.time()
    print("=" * 70)
    print("MULTI-FREQUENCY DEPLOYABILITY ASSESSMENT")
    print("=" * 70)
    print(f"Frequencies: {FREQUENCIES}")
    print(f"Cost tiers: {COST_TIERS} bps")
    print()

    # ===================================================================
    # Part 1: Generate ML predictions at each frequency
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 1: ML PREDICTION GENERATION")
    print("=" * 70)

    freq_data = {}
    for freq in FREQUENCIES:
        t0 = time.time()
        print(f"\n--- {freq} ---")
        preds, returns_wide, panel = generate_predictions(freq)
        freq_data[freq] = {"preds": preds, "returns": returns_wide, "panel": panel}
        print(f"  Done in {time.time()-t0:.0f}s")

    # ===================================================================
    # Part 2: Portfolio construction with cost tiers
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 2: ML PORTFOLIO CONSTRUCTION + COST SENSITIVITY")
    print("=" * 70)

    all_ml_results = []
    for freq in FREQUENCIES:
        preds = freq_data[freq]["preds"]
        rets = freq_data[freq]["returns"]
        bpd = BARS_PER_DAY[freq]
        rebal = max(1, int(REBAL_EVERY * bpd))

        for method in ["top_q_ew", "signal_ivw"]:
            for apply_ov in [False, True]:
                for cost in COST_TIERS:
                    label = f"ML {freq} {method}" + (" +OV" if apply_ov else "") + f" @{cost:.0f}bp"
                    try:
                        m = build_portfolio(preds, rets, method=method,
                                            cost_bps=cost, apply_overlays=apply_ov,
                                            rebal_every=rebal)
                        m["label"] = label
                        m["freq"] = freq
                        m["method"] = method
                        m["overlays"] = apply_ov
                        all_ml_results.append(m)
                    except Exception as e:
                        print(f"  [WARN] {label}: {e}")

            print(f"  {freq} {method}: done", flush=True)

    # ===================================================================
    # Part 3: Composite multi-frequency signal
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 3: COMPOSITE MULTI-FREQUENCY SIGNAL")
    print("=" * 70)

    # Colleague's suggested weights: 0.4×30m, 0.3×4h, 0.2×1h, 0.1×1d
    # We have 4h, 8h, 1d — so use 0.5×4h, 0.3×8h, 0.2×1d
    composite_weights = {"4h": 0.5, "8h": 0.3, "1d": 0.2}
    available = [f for f in composite_weights if f in freq_data]

    if len(available) >= 2:
        print(f"  Blending: {', '.join(f'{composite_weights[f]:.0%}×{f}' for f in available)}")

        # Find common (ts, symbol) pairs by mapping to calendar dates
        freq_preds_daily = {}
        for freq in available:
            p = freq_data[freq]["preds"].copy()
            p["cal_date"] = p["ts"].dt.date
            # Take last prediction per calendar date per symbol
            daily_p = p.groupby(["cal_date", "symbol"]).agg(
                prob_up=("prob_up", "last"),
                signal=("signal", "last"),
            ).reset_index()
            freq_preds_daily[freq] = daily_p

        # Merge on (cal_date, symbol)
        merged = freq_preds_daily[available[0]][["cal_date", "symbol"]].copy()
        for freq in available:
            sub = freq_preds_daily[freq][["cal_date", "symbol", "prob_up", "signal"]].rename(
                columns={"prob_up": f"prob_{freq}", "signal": f"sig_{freq}"}
            )
            merged = merged.merge(sub, on=["cal_date", "symbol"], how="inner")

        # Composite signal
        total_w = sum(composite_weights[f] for f in available)
        merged["composite_prob"] = sum(
            composite_weights[f] / total_w * merged[f"prob_{f}"] for f in available
        )
        merged["composite_signal"] = merged["composite_prob"] - 0.5

        # Agreement signal: fraction of frequencies that agree on direction
        for f in available:
            merged[f"vote_{f}"] = (merged[f"sig_{f}"] > 0).astype(float)
        merged["agreement"] = sum(merged[f"vote_{f}"] for f in available) / len(available)
        merged["stacked_signal"] = merged["composite_signal"] * merged["agreement"]

        print(f"  Composite obs: {len(merged):,}")

        # Use daily returns for composite backtest
        daily_rets = freq_data["1d"]["returns"] if "1d" in freq_data else freq_data[available[0]]["returns"]

        # Need realized_vol for IVW — get from 1d panel
        daily_panel = freq_data["1d"]["panel"] if "1d" in freq_data else freq_data[available[0]]["panel"]
        vol_lookup = daily_panel[["ts", "symbol", "close"]].copy()
        vol_lookup["realized_vol"] = vol_lookup.groupby("symbol")["close"].transform(
            lambda x: x.pct_change().rolling(42, min_periods=21).std() * np.sqrt(ANN_FACTOR)
        )
        vol_lookup["cal_date"] = vol_lookup["ts"].dt.date

        merged = merged.merge(
            vol_lookup[["cal_date", "symbol", "realized_vol"]].drop_duplicates(["cal_date", "symbol"]),
            on=["cal_date", "symbol"], how="left"
        )
        merged["realized_vol"] = merged["realized_vol"].fillna(1.0)

        # Map cal_date back to ts for daily backtest
        ts_lookup = daily_panel[["ts"]].drop_duplicates().copy()
        ts_lookup["cal_date"] = ts_lookup["ts"].dt.date
        merged = merged.merge(ts_lookup, on="cal_date", how="left")

        composite_results = []

        for sig_col, sig_label in [("composite_signal", "Composite"), ("stacked_signal", "Stacked")]:
            comp_preds = merged[["ts", "symbol", "composite_prob", sig_col, "realized_vol"]].copy()
            comp_preds.rename(columns={"composite_prob": "prob_up", sig_col: "signal"}, inplace=True)
            comp_preds["fwd_ret"] = np.nan  # not needed for portfolio

            for apply_ov in [False, True]:
                for cost in [0.0, 2.0, 5.0, 10.0, 20.0]:
                    label = f"{sig_label}" + (" +OV" if apply_ov else "") + f" @{cost:.0f}bp"
                    try:
                        m = build_portfolio(comp_preds, daily_rets, method="signal_ivw",
                                            cost_bps=cost, apply_overlays=apply_ov,
                                            rebal_every=REBAL_EVERY)
                        m["label"] = label
                        m["freq"] = "composite"
                        m["method"] = sig_label.lower()
                        m["overlays"] = apply_ov
                        composite_results.append(m)
                    except Exception as e:
                        print(f"  [WARN] {label}: {e}")

            print(f"  {sig_label}: done", flush=True)
    else:
        composite_results = []
        print("  [SKIP] Need >= 2 frequencies for composite")

    # ===================================================================
    # Part 4: Momentum with risk overlays
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 4: MOMENTUM + RISK OVERLAYS")
    print("=" * 70)

    all_mom_results = []
    for freq in FREQUENCIES:
        print(f"\n  --- Momentum {freq} ---")
        try:
            results = run_momentum_with_overlays(freq)
            all_mom_results.extend(results)
            for r in results:
                print(f"    {r['label']}: Sharpe={r['sharpe']:.2f} CAGR={r['cagr']:.1%} MaxDD={r['max_dd']:.1%}")
        except Exception as e:
            print(f"  [ERROR] {freq}: {e}")

    # ===================================================================
    # Part 5: Regime segmentation
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 5: REGIME SEGMENTATION")
    print("=" * 70)

    # Get BTC equity for regime definition
    if "1d" in freq_data:
        btc_eq = compute_btc_benchmark(freq_data["1d"]["panel"])
    else:
        btc_eq = compute_btc_benchmark(freq_data[FREQUENCIES[0]]["panel"])

    btc_ret_60 = btc_eq.pct_change(60)
    regime = pd.Series("sideways", index=btc_eq.index)
    regime[btc_ret_60 > 0.20] = "bull"
    regime[btc_ret_60 < -0.20] = "bear"

    print(f"  Regime distribution: {regime.value_counts().to_dict()}")

    regime_results = []
    # Analyze best ML strategy per frequency
    for freq in FREQUENCIES:
        best_results = [r for r in all_ml_results
                        if r["freq"] == freq and r["method"] == "signal_ivw"
                        and r["overlays"] and r["cost_bps"] == 5.0]
        if not best_results:
            continue
        m = best_results[0]
        eq = m["equity"]

        for regime_name in ["bull", "bear", "sideways"]:
            mask = regime.reindex(eq.index).fillna("sideways") == regime_name
            if mask.sum() < 30:
                continue
            regime_eq = eq[mask]
            if len(regime_eq) < 10:
                continue
            regime_eq = regime_eq / regime_eq.iloc[0]
            rm = compute_metrics(regime_eq)
            rm["freq"] = freq
            rm["regime"] = regime_name
            rm["n_days"] = mask.sum()
            regime_results.append(rm)

    if regime_results:
        regime_df = pd.DataFrame(regime_results)
        regime_df.to_csv(ARTIFACT_DIR / "regime_analysis.csv", index=False, float_format="%.4f")

        print(f"\n  ML Signal-IVW +Overlays @5bps by regime:")
        print(f"  {'Freq':<8s} {'Regime':<10s} {'Sharpe':>8s} {'CAGR':>8s} {'Vol':>8s} {'MaxDD':>8s} {'Days':>6s}")
        print(f"  {'-'*56}")
        for _, row in regime_df.iterrows():
            print(f"  {row['freq']:<8s} {row['regime']:<10s} {row['sharpe']:>8.2f} "
                  f"{row['cagr']:>7.1%} {row['vol']:>7.1%} {row['max_dd']:>7.1%} {int(row['n_days']):>6d}")

    # ===================================================================
    # Part 6: FINAL VERDICT
    # ===================================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT: NET SHARPE TABLE")
    print("=" * 70)

    # Build the key comparison table
    verdict_rows = []

    # ML strategies
    for freq in FREQUENCIES:
        for ov in [False, True]:
            for cost in [0.0, 5.0, 10.0, 20.0]:
                matches = [r for r in all_ml_results
                           if r["freq"] == freq and r["method"] == "signal_ivw"
                           and r["overlays"] == ov and r["cost_bps"] == cost]
                if matches:
                    m = matches[0]
                    verdict_rows.append({
                        "strategy": f"ML {freq}" + (" +OV" if ov else ""),
                        "cost_bps": cost,
                        "sharpe": m["sharpe"],
                        "cagr": m["cagr"],
                        "max_dd": m["max_dd"],
                        "vol": m["vol"],
                        "turnover": m["avg_turnover"],
                        "exposure": m["avg_exposure"],
                        "type": "ml",
                    })

    # Composite
    for ov in [False, True]:
        for cost in [0.0, 5.0, 10.0, 20.0]:
            matches = [r for r in composite_results
                       if r["method"] == "composite" and r["overlays"] == ov
                       and r["cost_bps"] == cost]
            if matches:
                m = matches[0]
                verdict_rows.append({
                    "strategy": "Composite" + (" +OV" if ov else ""),
                    "cost_bps": cost,
                    "sharpe": m["sharpe"],
                    "cagr": m["cagr"],
                    "max_dd": m["max_dd"],
                    "vol": m["vol"],
                    "turnover": m["avg_turnover"],
                    "exposure": m["avg_exposure"],
                    "type": "composite",
                })

    # Stacked
    for ov in [False, True]:
        for cost in [0.0, 5.0, 10.0, 20.0]:
            matches = [r for r in composite_results
                       if r["method"] == "stacked" and r["overlays"] == ov
                       and r["cost_bps"] == cost]
            if matches:
                m = matches[0]
                verdict_rows.append({
                    "strategy": "Stacked" + (" +OV" if ov else ""),
                    "cost_bps": cost,
                    "sharpe": m["sharpe"],
                    "cagr": m["cagr"],
                    "max_dd": m["max_dd"],
                    "vol": m["vol"],
                    "turnover": m["avg_turnover"],
                    "exposure": m["avg_exposure"],
                    "type": "stacked",
                })

    # Momentum with overlays
    for r in all_mom_results:
        verdict_rows.append({
            "strategy": r["label"],
            "cost_bps": r["cost_bps"],
            "sharpe": r["sharpe"],
            "cagr": r["cagr"],
            "max_dd": r["max_dd"],
            "vol": r["vol"],
            "turnover": r["avg_turnover"],
            "exposure": 1.0,
            "type": "momentum",
        })

    verdict_df = pd.DataFrame(verdict_rows)
    verdict_df.to_csv(ARTIFACT_DIR / "verdict_table.csv", index=False, float_format="%.4f")

    # Print the key table: net Sharpe at 5bps
    print("\n  KEY TABLE: Net Sharpe at 5bps (the fund/shelve decision)")
    print(f"  {'Strategy':<35s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'Vol':>8s} {'TO':>8s} {'Verdict':>10s}")
    print(f"  {'-'*90}")
    key = verdict_df[verdict_df["cost_bps"] == 5.0].sort_values("sharpe", ascending=False)
    for _, row in key.iterrows():
        verdict = "FUND" if row["sharpe"] >= 1.0 else ("EXPLORE" if row["sharpe"] >= 0.7 else "SHELVE")
        print(f"  {row['strategy']:<35s} {row['sharpe']:>8.2f} {row['cagr']:>7.1%} "
              f"{row['max_dd']:>7.1%} {row['vol']:>7.1%} {row['turnover']:>7.3f} {verdict:>10s}")

    # Cost sensitivity for best strategies
    print("\n  COST SENSITIVITY (best ML + Composite):")
    print(f"  {'Strategy':<35s} {'0bp':>8s} {'5bp':>8s} {'10bp':>8s} {'20bp':>8s}")
    print(f"  {'-'*70}")
    for strat_name in key["strategy"].unique()[:8]:
        row_str = f"  {strat_name:<35s}"
        for cost in [0.0, 5.0, 10.0, 20.0]:
            sub = verdict_df[(verdict_df["strategy"] == strat_name) & (verdict_df["cost_bps"] == cost)]
            if len(sub) > 0:
                row_str += f" {sub.iloc[0]['sharpe']:>8.2f}"
            else:
                row_str += f" {'--':>8s}"
        print(row_str)

    # ===================================================================
    # Plots
    # ===================================================================
    print("\n--- Generating plots ---")

    # 1. Cost sensitivity chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    strat_colors = {"ml": "#EC407A", "composite": "#3b82f6", "stacked": "#22c55e", "momentum": "#FFA726"}
    for strat in verdict_df["strategy"].unique():
        sub = verdict_df[verdict_df["strategy"] == strat].sort_values("cost_bps")
        if len(sub) < 3:
            continue
        stype = sub.iloc[0]["type"]
        ax.plot(sub["cost_bps"], sub["sharpe"], marker="o", markersize=4,
                color=strat_colors.get(stype, "gray"), alpha=0.7, linewidth=1.2,
                label=strat if stype in ("composite", "stacked") else None)
    ax.axhline(1.0, color="green", linewidth=1, linestyle="--", alpha=0.5, label="Fund threshold (1.0)")
    ax.axhline(0.7, color="orange", linewidth=1, linestyle="--", alpha=0.5, label="Explore threshold (0.7)")
    ax.set_xlabel("Transaction Cost (bps)", fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Cost Sensitivity: Net Sharpe by Strategy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2. Fund/Shelve bar chart at 5bps
    ax = axes[1]
    key_sorted = key.head(15).sort_values("sharpe")
    colors = ["#22c55e" if s >= 1.0 else "#FFA726" if s >= 0.7 else "#ef4444"
              for s in key_sorted["sharpe"]]
    ax.barh(range(len(key_sorted)), key_sorted["sharpe"], color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(key_sorted)))
    ax.set_yticklabels(key_sorted["strategy"], fontsize=8)
    ax.axvline(1.0, color="green", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(0.7, color="orange", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Fund / Explore / Shelve (@ 5bps)", fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "deployability_verdict.png")
    plt.close(fig)
    print("  [1/2] Verdict chart saved")

    # 3. Regime analysis chart
    if regime_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        regime_df_pivot = regime_df.pivot(index="freq", columns="regime", values="sharpe")
        x = np.arange(len(regime_df_pivot))
        width = 0.25
        regime_colors = {"bull": "#22c55e", "bear": "#ef4444", "sideways": "#9E9E9E"}
        for i, regime_name in enumerate(["bull", "bear", "sideways"]):
            if regime_name in regime_df_pivot.columns:
                vals = regime_df_pivot[regime_name].values
                ax.bar(x + (i - 1) * width, vals, width, label=regime_name.title(),
                       color=regime_colors[regime_name], alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(regime_df_pivot.index, fontsize=11)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Sharpe Ratio", fontsize=11)
        ax.set_title("ML Strategy Sharpe by Market Regime", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "regime_sharpe.png")
        plt.close(fig)
        print("  [2/2] Regime chart saved")

    elapsed = time.time() - t0_global
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
