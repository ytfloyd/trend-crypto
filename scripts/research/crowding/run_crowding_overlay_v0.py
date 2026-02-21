"""
El Farol Crowding Overlay — v0 Research Script
================================================
Tests whether scaling down exposure when trend signals are unanimous
improves risk-adjusted returns.

Core hypothesis (Arthur / El Farol):
    When everyone agrees, the consensus self-negates.  In trend-following,
    unanimous long signals across the universe precede crowded-trade
    drawdowns.  Fading unanimity should reduce drawdown severity with
    modest return drag.

Crowding indicators tested
--------------------------
1. **Signal breadth** — fraction of universe with positive trend signal.
   When breadth > threshold (e.g. 90%), scale weights down.

2. **Conviction concentration** — cross-sectional std of raw signals.
   Low dispersion = everyone agrees = crowded.

3. **Return unanimity** — fraction of universe with positive trailing
   20-day return (from Sornette breadth indicator).

Each overlay is applied to a baseline EMAC trend-following strategy
across the crypto universe.

Methodology
-----------
- Universe: dynamic filter (min $1M ADV, 90 days history)
- Baseline signal: EMA crossover (fast=5, slow=40) — simple, transparent
- Weighting: long-only, inverse-volatility weighted, top-quintile
- Execution: Model-B (signal at close t, earn return at t+1)
- Costs: 20 bps round-trip (conservative for crypto)
- Crowding overlay: multiplicative scalar on portfolio weights

Output: metrics comparison table, equity curves, crowding indicator plots.
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

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

from scripts.research.common.data import (
    ANN_FACTOR,
    load_daily_bars,
    filter_universe,
)
from scripts.research.common.backtest import simple_backtest
from scripts.research.common.metrics import (
    compute_metrics,
    format_metrics_table,
)
from scripts.research.common.data import compute_btc_benchmark

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "research"
    / "crowding"
)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAST_EMA = 5
SLOW_EMA = 40
VOL_WINDOW = 42
TOP_FRAC = 0.20
MAX_WEIGHT = 0.15
COST_BPS = 20.0
START = "2018-01-01"

BREADTH_LOOKBACK = 20
BREADTH_THRESHOLDS = [0.70, 0.80, 0.90]
CONVICTION_THRESHOLDS = [0.10, 0.15, 0.20]  # low std = crowded

# BTC regime filter — only allocate in bull/risk-on (SMA(50) on BTC)
USE_REGIME_FILTER = True
REGIME_FAST_SMA = 50
REGIME_SLOW_SMA = 200
MIN_ADV_USD = 5_000_000  # tighter than default to exclude micro-cap zombies


# ===================================================================
# BTC regime filter
# ===================================================================
def compute_btc_regime(panel: pd.DataFrame) -> pd.Series:
    """Boolean mask: True when BTC is in a bull/risk-on regime.

    BULL:   BTC close > SMA(50) AND SMA(50) > SMA(200)
    RISK-ON: BTC close > SMA(50)
    BEAR:   otherwise — go to cash.
    """
    btc = (
        panel.loc[panel["symbol"] == "BTC-USD", ["ts", "close"]]
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .set_index("ts")["close"]
    )
    sma_fast = btc.rolling(REGIME_FAST_SMA, min_periods=REGIME_FAST_SMA).mean()
    above_fast = btc > sma_fast
    return above_fast  # True = risk-on or bull


# ===================================================================
# Signal generation
# ===================================================================
def compute_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute EMAC trend signal and volatility per symbol."""

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        close = g["close"]
        fast = close.shift(1).ewm(span=FAST_EMA, min_periods=FAST_EMA).mean()
        slow = close.shift(1).ewm(span=SLOW_EMA, min_periods=SLOW_EMA).mean()
        g["signal"] = (fast - slow) / slow
        ret = np.log(close / close.shift(1))
        g["realized_vol"] = (
            ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
            * np.sqrt(ANN_FACTOR)
        )
        g["ret_cc"] = close / close.shift(1) - 1.0
        g["ret_20d"] = close / close.shift(BREADTH_LOOKBACK) - 1.0
        return g

    return panel.groupby("symbol", group_keys=False).apply(_per_symbol)


# ===================================================================
# Crowding indicators (cross-sectional, computed per timestamp)
# ===================================================================
def compute_crowding_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute per-date crowding metrics from the active universe.

    Returns DataFrame indexed by ts with columns:
        signal_breadth  — frac of universe with signal > 0
        return_breadth  — frac of universe with positive 20d return
        signal_dispersion — cross-sectional std of signals
        n_active         — number of symbols in universe
    """
    active = panel.loc[panel["in_universe"] & panel["signal"].notna()].copy()

    def _per_date(g: pd.DataFrame) -> pd.Series:
        sigs = g["signal"]
        return pd.Series({
            "signal_breadth": float((sigs > 0).mean()),
            "return_breadth": float((g["ret_20d"] > 0).mean()) if "ret_20d" in g else np.nan,
            "signal_dispersion": float(sigs.std()) if len(sigs) > 1 else 0.0,
            "n_active": len(g),
        })

    crowd = active.groupby("ts").apply(_per_date)
    return crowd


# ===================================================================
# Baseline portfolio construction
# ===================================================================
def build_baseline_weights(
    panel: pd.DataFrame,
    regime_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-only top-quintile IVW weights from EMAC signal.

    If regime_mask is provided, weights are zeroed on bear-market days.
    Returns (weights_wide, returns_wide).
    """
    active = panel.loc[
        (panel["ts"] >= START)
        & panel["in_universe"]
        & panel["signal"].notna()
        & panel["realized_vol"].notna()
    ].copy()

    # Only go long when signal is positive (true trend filter)
    active = active.loc[active["signal"] > 0].copy()

    returns_wide = (
        panel.loc[panel["ts"] >= START]
        .pivot_table(index="ts", columns="symbol", values="ret_cc", aggfunc="first")
        .fillna(0.0)
    )

    dates = sorted(active["ts"].unique())
    rows = []
    for dt in dates:
        day = active.loc[active["ts"] == dt].copy()
        if len(day) < 3:
            continue
        ranked = day.sort_values("signal", ascending=False)
        n_select = max(1, int(len(ranked) * TOP_FRAC))
        top = ranked.head(n_select)

        vols = top["realized_vol"].clip(lower=0.10)
        inv_vol = 1.0 / vols
        wts = inv_vol / inv_vol.sum()

        for sym, w in zip(top["symbol"], wts):
            rows.append({"ts": dt, "symbol": sym, "weight": w})

    wdf = pd.DataFrame(rows)
    weights_wide = wdf.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)

    # Position limits
    for _ in range(5):
        row_sum = weights_wide.sum(axis=1).replace(0, np.nan)
        pct = weights_wide.div(row_sum, axis=0).fillna(0)
        over = pct > MAX_WEIGHT
        if not over.any().any():
            break
        weights_wide = weights_wide.where(
            ~over, pct.clip(upper=MAX_WEIGHT).mul(row_sum, axis=0)
        )

    # Apply regime filter: zero weights on bear-market days
    if regime_mask is not None:
        rm = regime_mask.reindex(weights_wide.index).ffill().fillna(False)
        weights_wide = weights_wide.mul(rm.astype(float), axis=0)

    return weights_wide, returns_wide


# ===================================================================
# Crowding overlay — multiplicative weight scalar
# ===================================================================
def apply_crowding_overlay(
    weights: pd.DataFrame,
    crowd: pd.DataFrame,
    indicator: str,
    threshold: float,
    fade_strength: float = 0.5,
    invert: bool = False,
) -> pd.DataFrame:
    """Scale weights down when crowding indicator exceeds threshold.

    For breadth indicators (signal_breadth, return_breadth):
        scalar = 1.0 when indicator <= threshold
        scalar = 1.0 - fade_strength * (indicator - threshold) / (1.0 - threshold)
        Clamped to [1 - fade_strength, 1.0].

    For dispersion (invert=True): scale down when dispersion is LOW
        scalar = 1.0 when dispersion >= threshold
        scalar = 1.0 - fade_strength * (threshold - dispersion) / threshold
    """
    indicator_series = crowd[indicator].reindex(weights.index).ffill().fillna(0.5)

    if invert:
        excess = (threshold - indicator_series).clip(lower=0.0)
        denom = max(threshold, 1e-8)
    else:
        excess = (indicator_series - threshold).clip(lower=0.0)
        denom = max(1.0 - threshold, 1e-8)

    scalar = (1.0 - fade_strength * excess / denom).clip(lower=1.0 - fade_strength)

    return weights.mul(scalar, axis=0)


# ===================================================================
# Run one variant and return metrics
# ===================================================================
def run_variant(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    label: str,
    cost_bps: float = COST_BPS,
) -> dict:
    """Backtest a weight matrix and return labeled metrics."""
    bt = simple_backtest(weights, returns_wide, cost_bps=cost_bps)
    eq = pd.Series(bt["portfolio_equity"].values, index=bt["ts"])
    m = compute_metrics(eq)
    m["label"] = label
    m["avg_turnover"] = float(bt["turnover"].mean())
    m["avg_exposure"] = float(bt["gross_exposure"].mean())
    m["equity"] = eq
    return m


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("EL FAROL CROWDING OVERLAY — v0")
    print("=" * 70)
    print(f"Baseline: EMAC({FAST_EMA}/{SLOW_EMA}), IVW top-{TOP_FRAC:.0%}, {COST_BPS:.0f} bps")
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/6] Loading daily bars ...")
    panel = load_daily_bars()
    panel = filter_universe(panel, min_adv_usd=MIN_ADV_USD, min_history_days=90)
    n_sym = panel.loc[panel["in_universe"], "symbol"].nunique()
    print(f"  {len(panel):,} rows, {n_sym} symbols in universe")

    # ------------------------------------------------------------------
    # 2. Compute signals
    # ------------------------------------------------------------------
    print("[2/6] Computing signals ...")
    panel = compute_signals(panel)

    # ------------------------------------------------------------------
    # 3. Compute crowding indicators
    # ------------------------------------------------------------------
    print("[3/6] Computing crowding indicators ...")
    crowd = compute_crowding_indicators(panel)
    print(f"  Signal breadth:    mean={crowd['signal_breadth'].mean():.2f}  "
          f"std={crowd['signal_breadth'].std():.2f}")
    print(f"  Return breadth:    mean={crowd['return_breadth'].mean():.2f}  "
          f"std={crowd['return_breadth'].std():.2f}")
    print(f"  Signal dispersion: mean={crowd['signal_dispersion'].mean():.4f}  "
          f"std={crowd['signal_dispersion'].std():.4f}")

    # ------------------------------------------------------------------
    # 3b. BTC regime filter
    # ------------------------------------------------------------------
    regime_mask_series = None
    if USE_REGIME_FILTER:
        print("  Computing BTC regime filter ...")
        regime_mask_series = compute_btc_regime(panel)
        bull_pct = regime_mask_series.mean()
        print(f"  BTC risk-on: {bull_pct:.0%} of days")

    # ------------------------------------------------------------------
    # 4. Build baseline portfolio
    # ------------------------------------------------------------------
    print("[4/6] Building baseline portfolio ...")
    base_weights, returns_wide = build_baseline_weights(panel, regime_mask=regime_mask_series)
    print(f"  Weights: {base_weights.shape[0]} dates × {base_weights.shape[1]} symbols")

    # ------------------------------------------------------------------
    # 5. Run all variants
    # ------------------------------------------------------------------
    print("[5/6] Running crowding overlay variants ...")
    all_results = []

    baseline = run_variant(base_weights, returns_wide, "Baseline (no overlay)")
    all_results.append(baseline)
    print(f"  Baseline: Sharpe={baseline['sharpe']:.2f}  "
          f"CAGR={baseline['cagr']:.1%}  MaxDD={baseline['max_dd']:.1%}")

    # --- Signal breadth overlays ---
    for thresh in BREADTH_THRESHOLDS:
        for fade in [0.30, 0.50, 0.70]:
            label = f"SigBreadth>{thresh:.0%} fade={fade:.0%}"
            w = apply_crowding_overlay(
                base_weights, crowd, "signal_breadth", thresh, fade
            )
            m = run_variant(w, returns_wide, label)
            all_results.append(m)

    # --- Return breadth overlays ---
    for thresh in BREADTH_THRESHOLDS:
        for fade in [0.30, 0.50, 0.70]:
            label = f"RetBreadth>{thresh:.0%} fade={fade:.0%}"
            w = apply_crowding_overlay(
                base_weights, crowd, "return_breadth", thresh, fade
            )
            m = run_variant(w, returns_wide, label)
            all_results.append(m)

    # --- Signal dispersion overlays (inverted: low disp = crowded) ---
    for thresh in CONVICTION_THRESHOLDS:
        for fade in [0.30, 0.50, 0.70]:
            label = f"SigDisp<{thresh:.2f} fade={fade:.0%}"
            w = apply_crowding_overlay(
                base_weights, crowd, "signal_dispersion", thresh, fade,
                invert=True,
            )
            m = run_variant(w, returns_wide, label)
            all_results.append(m)

    # --- Combined: breadth + dispersion ---
    for b_thresh in [0.80, 0.90]:
        for d_thresh in [0.10, 0.15]:
            label = f"Combined B>{b_thresh:.0%}+D<{d_thresh:.2f}"
            w = apply_crowding_overlay(
                base_weights, crowd, "signal_breadth", b_thresh, 0.50
            )
            w = apply_crowding_overlay(
                w, crowd, "signal_dispersion", d_thresh, 0.30, invert=True
            )
            m = run_variant(w, returns_wide, label)
            all_results.append(m)

    # ------------------------------------------------------------------
    # 6. Results & plots
    # ------------------------------------------------------------------
    print("\n[6/6] Generating results ...")

    # Sort by Sharpe improvement over baseline
    base_sharpe = baseline["sharpe"]
    for r in all_results:
        r["sharpe_delta"] = r["sharpe"] - base_sharpe
        r["dd_improvement"] = r["max_dd"] - baseline["max_dd"]

    results_table = sorted(all_results, key=lambda x: x["sharpe"], reverse=True)

    print("\n" + "=" * 70)
    print("RESULTS: ALL VARIANTS (sorted by Sharpe)")
    print("=" * 70)
    header = (
        f"{'Strategy':<40s} {'Sharpe':>7s} {'dSh':>6s} {'CAGR':>8s} "
        f"{'MaxDD':>8s} {'dDD':>6s} {'Vol':>7s} {'TO':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in results_table:
        dd_str = f"{r['dd_improvement']:+.1%}" if r['label'] != "Baseline (no overlay)" else "  --"
        dsh_str = f"{r['sharpe_delta']:+.2f}" if r['label'] != "Baseline (no overlay)" else "  --"
        print(
            f"{r['label']:<40s} {r['sharpe']:>7.2f} {dsh_str:>6s} "
            f"{r['cagr']:>7.1%} {r['max_dd']:>7.1%} {dd_str:>6s} "
            f"{r['vol']:>6.1%} {r['avg_turnover']:>7.3f}"
        )

    # Save metrics CSV
    metrics_rows = []
    for r in results_table:
        row = {k: v for k, v in r.items() if k not in ("equity",)}
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(
        ARTIFACT_DIR / "crowding_overlay_results_v0.csv", index=False, float_format="%.6f"
    )

    # === PLOT 1: Equity curves (baseline vs top 3 overlays) ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    base_eq = baseline["equity"]
    ax.plot(base_eq.index, base_eq.values, color="#666", linewidth=1.5,
            label=f"Baseline (Sh={baseline['sharpe']:.2f})")
    colors = ["#3b82f6", "#22c55e", "#EC407A", "#FFA726"]
    overlay_results = [r for r in results_table if r["label"] != "Baseline (no overlay)"][:4]
    for i, r in enumerate(overlay_results):
        eq = r["equity"]
        ax.plot(eq.index, eq.values, color=colors[i % len(colors)], linewidth=1.2,
                alpha=0.85, label=f"{r['label']} (Sh={r['sharpe']:.2f})")
    ax.set_ylabel("Portfolio Equity", fontsize=11)
    ax.set_title("Equity Curves: Baseline vs Top Crowding Overlays", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # === PLOT 2: Crowding indicators time series ===
    ax = axes[0, 1]
    ts = crowd.index
    ax.plot(ts, crowd["signal_breadth"], color="#3b82f6", linewidth=0.8, alpha=0.7,
            label="Signal breadth")
    ax.plot(ts, crowd["return_breadth"], color="#22c55e", linewidth=0.8, alpha=0.7,
            label="Return breadth (20d)")
    ax.axhline(0.80, color="#EC407A", linewidth=1, linestyle="--", alpha=0.5, label="80% threshold")
    ax.axhline(0.90, color="#FFA726", linewidth=1, linestyle="--", alpha=0.5, label="90% threshold")
    ax.set_ylabel("Breadth (fraction)", fontsize=11)
    ax.set_title("Crowding Indicators Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # === PLOT 3: Sharpe improvement vs max DD improvement ===
    ax = axes[1, 0]
    overlay_only = [r for r in all_results if r["label"] != "Baseline (no overlay)"]
    sh_deltas = [r["sharpe_delta"] for r in overlay_only]
    dd_imps = [r["dd_improvement"] for r in overlay_only]
    scatter_colors = []
    for r in overlay_only:
        if "SigBreadth" in r["label"]:
            scatter_colors.append("#3b82f6")
        elif "RetBreadth" in r["label"]:
            scatter_colors.append("#22c55e")
        elif "SigDisp" in r["label"]:
            scatter_colors.append("#FFA726")
        else:
            scatter_colors.append("#EC407A")
    ax.scatter(dd_imps, sh_deltas, c=scatter_colors, s=40, alpha=0.7, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Max DD improvement (positive = shallower)", fontsize=10)
    ax.set_ylabel("Sharpe improvement", fontsize=10)
    ax.set_title("Overlay Impact: Sharpe vs Drawdown", fontsize=13, fontweight="bold")

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#3b82f6", markersize=8, label="Signal Breadth"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#22c55e", markersize=8, label="Return Breadth"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFA726", markersize=8, label="Signal Dispersion"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#EC407A", markersize=8, label="Combined"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.grid(True, alpha=0.3)

    # === PLOT 4: Drawdown comparison ===
    ax = axes[1, 1]
    base_eq = baseline["equity"]
    base_dd = base_eq / base_eq.cummax() - 1.0
    ax.fill_between(base_dd.index, base_dd.values, 0, color="#ef4444", alpha=0.3,
                    label=f"Baseline (MaxDD={baseline['max_dd']:.1%})")
    if overlay_results:
        best = overlay_results[0]
        best_eq = best["equity"]
        best_dd = best_eq / best_eq.cummax() - 1.0
        ax.fill_between(best_dd.index, best_dd.values, 0, color="#3b82f6", alpha=0.3,
                        label=f"Best overlay (MaxDD={best['max_dd']:.1%})")
    ax.set_ylabel("Drawdown", fontsize=11)
    ax.set_title("Drawdown Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "crowding_overlay_v0.png")
    plt.close(fig)
    print(f"\n  Chart saved: {ARTIFACT_DIR / 'crowding_overlay_v0.png'}")

    # === PLOT 5: Conditional performance — returns when crowded vs not ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    base_bt = simple_backtest(base_weights, returns_wide, cost_bps=COST_BPS)
    base_rets = pd.Series(base_bt["portfolio_ret"].values, index=base_bt["ts"])
    sig_breadth = crowd["signal_breadth"].reindex(base_rets.index).ffill()

    for ax, thresh_val, title in [
        (axes[0], 0.80, "Signal Breadth > 80%"),
        (axes[1], 0.90, "Signal Breadth > 90%"),
    ]:
        crowded = base_rets[sig_breadth > thresh_val]
        uncrowded = base_rets[sig_breadth <= thresh_val]

        crowded_ann = float(crowded.mean()) * ANN_FACTOR if len(crowded) > 0 else 0
        uncrowded_ann = float(uncrowded.mean()) * ANN_FACTOR if len(uncrowded) > 0 else 0
        crowded_vol = float(crowded.std()) * np.sqrt(ANN_FACTOR) if len(crowded) > 1 else 0
        uncrowded_vol = float(uncrowded.std()) * np.sqrt(ANN_FACTOR) if len(uncrowded) > 1 else 0
        crowded_sh = crowded_ann / crowded_vol if crowded_vol > 0 else 0
        uncrowded_sh = uncrowded_ann / uncrowded_vol if uncrowded_vol > 0 else 0

        bars = ax.bar(
            ["Crowded", "Not Crowded"],
            [crowded_sh, uncrowded_sh],
            color=["#ef4444", "#22c55e"],
            alpha=0.8,
            edgecolor="white",
        )
        ax.set_ylabel("Sharpe Ratio", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5)

        for bar, sh in zip(bars, [crowded_sh, uncrowded_sh]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{sh:.2f}", ha="center", fontsize=11, fontweight="bold")

        n_crowded = len(crowded)
        n_total = len(base_rets)
        ax.text(0.02, 0.98, f"Crowded: {n_crowded}/{n_total} days ({n_crowded/n_total:.0%})",
                transform=ax.transAxes, fontsize=8, va="top")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "crowding_conditional_v0.png")
    plt.close(fig)
    print(f"  Chart saved: {ARTIFACT_DIR / 'crowding_conditional_v0.png'}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.0f}s")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
