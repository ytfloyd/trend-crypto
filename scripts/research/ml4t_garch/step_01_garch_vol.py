"""
Study 1 — GARCH Volatility Models for Crypto
==============================================
Replicates & extends Chapter 9 of Jansen (2020)
"ML for Algorithmic Trading, 2nd Ed."

Objective: Replace rolling-window realized vol estimates with
conditional volatility from GARCH-family models.

Steps:
  1. Load daily crypto data, fit GARCH(1,1), EGARCH, GJR-GARCH
     to the top-liquidity assets
  2. Compare 1-day-ahead conditional vol forecasts vs rolling vol
  3. Out-of-sample evaluation: vol forecast accuracy (QLIKE, MSE)
  4. Portfolio application: GARCH-based vol targeting vs rolling-vol
     targeting on the EMAC momentum strategy from the JPM study
  5. GARCH-based IVW: use conditional vol for inverse-vol weighting

Reference: Jansen (2020) Ch. 9 §"ARCH Models" + §"Volatility Forecasts"
Also: Engle (1982), Bollerslev (1986), Nelson (1991), Glosten et al. (1993)
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

from arch import arch_model
from scipy import stats as sp_stats

from scripts.research.common.data import (
    ANN_FACTOR,
    compute_btc_benchmark,
    filter_universe,
    load_daily_bars,
)
from scripts.research.common.backtest import simple_backtest
from scripts.research.common.metrics import compute_metrics, format_metrics_table
from scripts.research.common.risk_overlays import apply_vol_targeting

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START = "2018-01-01"
END = "2025-12-31"
DATA_START = "2016-06-01"
MIN_ADV = 1_000_000
MIN_HISTORY = 90

# Vol estimation params
ROLLING_VOL_WINDOW = 42
GARCH_FIT_WINDOW = 365 * 2  # 2-year expanding/rolling fit
GARCH_REFIT_EVERY = 63      # re-fit GARCH every ~quarter

# Momentum strategy params (from JPM study best config)
LOOKBACK = 21
REBAL_FREQ = 10
VOL_TARGET = 0.20
MAX_LEVERAGE = 2.0
COST_BPS = 20.0

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "ml4t_garch"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})


# ===================================================================
# 1. Load data and select top-liquidity assets
# ===================================================================
def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load panel, compute returns, select top assets."""
    panel = load_daily_bars(start=DATA_START, end=END)
    panel = filter_universe(panel, min_adv_usd=MIN_ADV, min_history_days=MIN_HISTORY)

    # Wide-format returns and close
    panel["ret"] = panel.groupby("symbol")["close"].pct_change()
    returns_wide = panel.pivot_table(
        index="ts", columns="symbol", values="ret", aggfunc="first"
    ).fillna(0.0)
    close_wide = panel.pivot_table(
        index="ts", columns="symbol", values="close", aggfunc="first"
    )

    return panel, returns_wide, close_wide


# ===================================================================
# 2. GARCH model fitting and forecasting
# ===================================================================
GARCH_SPECS = {
    "GARCH(1,1)":  {"vol": "Garch", "p": 1, "q": 1, "o": 0, "dist": "studentst"},
    "EGARCH(1,1)": {"vol": "EGARCH", "p": 1, "q": 1, "o": 1, "dist": "studentst"},
    "GJR-GARCH":   {"vol": "Garch", "p": 1, "q": 1, "o": 1, "dist": "studentst"},
}


def fit_garch_forecasts(
    returns: pd.Series,
    spec_name: str = "GARCH(1,1)",
    fit_window: int = GARCH_FIT_WINDOW,
    refit_every: int = GARCH_REFIT_EVERY,
) -> pd.Series:
    """Walk-forward GARCH conditional volatility forecasts.

    Fits the model on an expanding window, re-fits every `refit_every` days,
    and produces 1-day-ahead annualized conditional vol.
    """
    spec = GARCH_SPECS[spec_name]
    ret_pct = returns.dropna() * 100  # arch expects pct returns

    dates = ret_pct.index
    forecasts = pd.Series(np.nan, index=dates)

    last_fit_idx = -refit_every  # force first fit
    model_result = None

    for i in range(fit_window, len(dates)):
        if i - last_fit_idx >= refit_every or model_result is None:
            # Fit on expanding window
            train = ret_pct.iloc[max(0, i - fit_window * 2):i]
            if len(train) < 252:
                continue
            try:
                am = arch_model(
                    train,
                    mean="Constant",
                    vol=spec["vol"],
                    p=spec["p"],
                    q=spec["q"],
                    o=spec["o"],
                    dist=spec["dist"],
                )
                model_result = am.fit(disp="off", show_warning=False)
                last_fit_idx = i
            except Exception:
                continue

        if model_result is not None:
            try:
                # 1-step forecast from the fitted model using data up to i
                train_to_i = ret_pct.iloc[max(0, i - fit_window * 2):i + 1]
                am_fcast = arch_model(
                    train_to_i,
                    mean="Constant",
                    vol=spec["vol"],
                    p=spec["p"],
                    q=spec["q"],
                    o=spec["o"],
                    dist=spec["dist"],
                )
                res = am_fcast.fit(
                    disp="off", show_warning=False,
                    starting_values=model_result.params.values,
                )
                fcast = res.forecast(horizon=1, reindex=False)
                cond_var = fcast.variance.iloc[-1, 0]
                # Convert from pct^2 daily to annualized vol
                forecasts.iloc[i] = np.sqrt(cond_var) / 100.0 * np.sqrt(ANN_FACTOR)
            except Exception:
                pass

    return forecasts


def compute_rolling_vol(returns: pd.Series, window: int = ROLLING_VOL_WINDOW) -> pd.Series:
    """Standard rolling realized vol (annualized)."""
    return returns.rolling(window, min_periods=max(10, window // 2)).std() * np.sqrt(ANN_FACTOR)


# ===================================================================
# 3. Forecast evaluation
# ===================================================================
def evaluate_vol_forecasts(
    realized_vol: pd.Series,
    forecast_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    """Evaluate vol forecast accuracy with QLIKE and MSE."""
    results = []
    for name, fcast in forecast_dict.items():
        common = realized_vol.dropna().index.intersection(fcast.dropna().index)
        if len(common) < 30:
            continue
        rv = realized_vol.loc[common]
        fc = fcast.loc[common]

        # QLIKE: E[rv^2 / fc^2 - log(rv^2 / fc^2) - 1]
        ratio = (rv ** 2) / (fc ** 2).clip(lower=1e-10)
        qlike = float((ratio - np.log(ratio) - 1.0).mean())

        # MSE on annualized vol
        mse = float(((rv - fc) ** 2).mean())

        # MAE
        mae = float((rv - fc).abs().mean())

        # Correlation
        corr = float(rv.corr(fc))

        results.append({
            "model": name,
            "qlike": qlike,
            "mse": mse,
            "mae": mae,
            "correlation": corr,
            "n_obs": len(common),
        })

    return pd.DataFrame(results)


# ===================================================================
# 4. Momentum strategy builder (from JPM study)
# ===================================================================
def build_emac_weights(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build EMAC 21d top-quintile IVW weights (JPM study baseline)."""
    def _per_sym(g):
        g = g.copy()
        close = g["close"]
        fast = max(2, LOOKBACK // 4)
        fast_ema = close.shift(1).ewm(span=fast, min_periods=fast).mean()
        slow_ema = close.shift(1).ewm(span=LOOKBACK, min_periods=LOOKBACK).mean()
        g["signal"] = (fast_ema - slow_ema) / slow_ema

        ret = np.log(close / close.shift(1))
        g["realized_vol"] = ret.rolling(63, min_periods=63).std() * np.sqrt(ANN_FACTOR)
        return g

    p = panel.groupby("symbol", group_keys=False).apply(_per_sym)
    p_active = p.loc[
        (p["ts"] >= START) & p["in_universe"] & p["signal"].notna()
    ].copy()

    all_dates = sorted(p_active["ts"].unique())
    rebal_dates = set(all_dates[::REBAL_FREQ])

    current_weights: dict[str, float] = {}
    dates_list = []
    weights_list = []

    for dt in all_dates:
        day = p_active.loc[p_active["ts"] == dt].copy()
        if day.empty:
            dates_list.append(dt)
            weights_list.append({})
            continue

        if dt in rebal_dates:
            ranked = day.sort_values("signal", ascending=False)
            n_select = max(1, len(ranked) // 5)
            selected = ranked.head(n_select)

            vols = selected["realized_vol"].replace(0, np.nan).dropna()
            if len(vols) > 0:
                inv_vol = 1.0 / vols.clip(lower=0.10)
                wts = inv_vol / inv_vol.sum()
                current_weights = dict(zip(selected.loc[vols.index, "symbol"], wts))
            else:
                n = len(selected)
                current_weights = {s: 1.0 / n for s in selected["symbol"]}

        row = {s: current_weights.get(s, 0.0) for s in day["symbol"].tolist()}
        dates_list.append(dt)
        weights_list.append(row)

    return pd.DataFrame(weights_list, index=pd.DatetimeIndex(dates_list)).fillna(0.0)


def build_emac_weights_garch_ivw(
    panel: pd.DataFrame,
    garch_vol_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Build EMAC weights using GARCH conditional vol for IVW instead of rolling vol."""
    def _per_sym(g):
        g = g.copy()
        close = g["close"]
        fast = max(2, LOOKBACK // 4)
        fast_ema = close.shift(1).ewm(span=fast, min_periods=fast).mean()
        slow_ema = close.shift(1).ewm(span=LOOKBACK, min_periods=LOOKBACK).mean()
        g["signal"] = (fast_ema - slow_ema) / slow_ema
        return g

    p = panel.groupby("symbol", group_keys=False).apply(_per_sym)
    p_active = p.loc[
        (p["ts"] >= START) & p["in_universe"] & p["signal"].notna()
    ].copy()

    all_dates = sorted(p_active["ts"].unique())
    rebal_dates = set(all_dates[::REBAL_FREQ])

    current_weights: dict[str, float] = {}
    dates_list = []
    weights_list = []

    for dt in all_dates:
        day = p_active.loc[p_active["ts"] == dt].copy()
        if day.empty:
            dates_list.append(dt)
            weights_list.append({})
            continue

        if dt in rebal_dates:
            ranked = day.sort_values("signal", ascending=False)
            n_select = max(1, len(ranked) // 5)
            selected = ranked.head(n_select)

            # Use GARCH vol if available, else fall back to rolling
            vols_dict = {}
            for _, row in selected.iterrows():
                sym = row["symbol"]
                if sym in garch_vol_wide.columns and dt in garch_vol_wide.index:
                    gv = garch_vol_wide.at[dt, sym]
                    if not np.isnan(gv) and gv > 0:
                        vols_dict[sym] = gv
                        continue
                # Fallback: use a naive estimate
                vols_dict[sym] = 0.5  # neutral

            if vols_dict:
                inv_vol = {s: 1.0 / max(v, 0.10) for s, v in vols_dict.items()}
                total = sum(inv_vol.values())
                current_weights = {s: v / total for s, v in inv_vol.items()}
            else:
                n = len(selected)
                current_weights = {s: 1.0 / n for s in selected["symbol"]}

        row = {s: current_weights.get(s, 0.0) for s in day["symbol"].tolist()}
        dates_list.append(dt)
        weights_list.append(row)

    return pd.DataFrame(weights_list, index=pd.DatetimeIndex(dates_list)).fillna(0.0)


# ===================================================================
# GARCH-based vol targeting
# ===================================================================
def apply_garch_vol_targeting(
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    garch_port_vol: pd.Series,
    vol_target: float = VOL_TARGET,
    max_leverage: float = MAX_LEVERAGE,
) -> pd.DataFrame:
    """Vol targeting using GARCH portfolio vol forecast instead of rolling."""
    common = weights.columns.intersection(returns_wide.columns)
    w = weights[common].copy()
    gv = garch_port_vol.reindex(w.index).ffill()
    scalar = (vol_target / gv).clip(lower=0.0, upper=max_leverage).fillna(1.0)
    return w.mul(scalar, axis=0)


# ===================================================================
# Main
# ===================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("STUDY 1: GARCH VOLATILITY MODELS FOR CRYPTO")
    print("Replicating Jansen (2020) Ch. 9 — ARCH Models")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Part 1: Load data
    # ------------------------------------------------------------------
    print("\n--- Part 1: Loading data ---")
    panel, returns_wide, close_wide = load_and_prepare()
    print(f"  Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols")
    print(f"  Date range: {panel['ts'].min().date()} to {panel['ts'].max().date()}")

    # Select top 20 most liquid assets for detailed GARCH analysis
    adv = panel.groupby("symbol").apply(
        lambda g: (g["close"] * g["volume"]).mean()
    ).sort_values(ascending=False)
    top_syms = adv.head(20).index.tolist()
    print(f"  Top 20 by ADV: {', '.join(s.replace('-USD','') for s in top_syms[:10])} ...")

    # ------------------------------------------------------------------
    # Part 2: Fit GARCH models to top assets
    # ------------------------------------------------------------------
    print("\n--- Part 2: GARCH Model Fitting ---")
    print(f"  Models: {list(GARCH_SPECS.keys())}")
    print(f"  Fit window: {GARCH_FIT_WINDOW}d, refit every {GARCH_REFIT_EVERY}d")

    garch_forecasts: dict[str, dict[str, pd.Series]] = {}
    rolling_vols: dict[str, pd.Series] = {}
    realized_vols: dict[str, pd.Series] = {}

    for sym in top_syms:
        sym_ret = returns_wide[sym].dropna()
        if len(sym_ret) < GARCH_FIT_WINDOW + 100:
            print(f"  {sym}: skipped (insufficient data)")
            continue

        # Realized vol (proxy for "truth") — use 5-day forward realized vol
        rv = sym_ret.rolling(5, min_periods=5).std().shift(-5) * np.sqrt(ANN_FACTOR)
        realized_vols[sym] = rv

        # Rolling vol baseline
        roll = compute_rolling_vol(sym_ret)
        rolling_vols[sym] = roll

        garch_forecasts[sym] = {}
        for spec_name in GARCH_SPECS:
            print(f"  {sym} — {spec_name} ...", end="", flush=True)
            fcast = fit_garch_forecasts(sym_ret, spec_name)
            n_valid = fcast.notna().sum()
            garch_forecasts[sym][spec_name] = fcast
            print(f" {n_valid} forecasts")

    # ------------------------------------------------------------------
    # Part 3: Evaluate forecast accuracy
    # ------------------------------------------------------------------
    print("\n--- Part 3: Forecast Accuracy ---")

    all_evals = []
    for sym in garch_forecasts:
        rv = realized_vols[sym]
        forecasts = {"Rolling 42d": rolling_vols[sym]}
        forecasts.update(garch_forecasts[sym])
        ev = evaluate_vol_forecasts(rv, forecasts)
        ev["symbol"] = sym
        all_evals.append(ev)

    if all_evals:
        eval_df = pd.concat(all_evals, ignore_index=True)
        eval_df.to_csv(ARTIFACT_DIR / "vol_forecast_eval.csv", index=False, float_format="%.6f")

        # Aggregate across assets
        agg = eval_df.groupby("model")[["qlike", "mse", "mae", "correlation"]].mean()
        agg = agg.sort_values("qlike")

        print("\n  Averaged vol forecast accuracy (lower QLIKE/MSE = better):")
        print(f"  {'Model':<18s} {'QLIKE':>10s} {'MSE':>10s} {'MAE':>10s} {'Corr':>8s}")
        print(f"  {'-'*58}")
        for idx, row in agg.iterrows():
            print(f"  {idx:<18s} {row['qlike']:>10.4f} {row['mse']:>10.4f} "
                  f"{row['mae']:>10.4f} {row['correlation']:>8.3f}")

        best_model = agg.index[0]
        print(f"\n  Best model by QLIKE: {best_model}")
    else:
        best_model = "GARCH(1,1)"
        eval_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Part 4: Portfolio application — GARCH vs rolling vol targeting
    # ------------------------------------------------------------------
    print("\n--- Part 4: Portfolio Application ---")
    print("  Strategy: EMAC 21d, top-quintile, IVW, rf=10d")

    # Build base weights (rolling vol IVW)
    base_weights = build_emac_weights(panel)

    ret_panel = panel.loc[panel["ts"] >= START].copy()
    ret_panel["ret_oc"] = ret_panel["close"] / ret_panel["open"] - 1.0
    returns_bt = ret_panel.pivot_table(
        index="ts", columns="symbol", values="ret_oc", aggfunc="first"
    ).fillna(0.0)

    strategies = {}

    # A. No vol targeting
    print("  [A] Baseline (no vol targeting)")
    bt_a = simple_backtest(base_weights, returns_bt, cost_bps=COST_BPS)
    eq_a = pd.Series(bt_a["portfolio_equity"].values, index=bt_a["ts"])
    m_a = compute_metrics(eq_a)
    m_a["label"] = "A: No VolTarget"
    strategies["A: No VolTarget"] = {"bt": bt_a, "eq": eq_a, "m": m_a}

    # B. Rolling vol targeting (current approach)
    print("  [B] Rolling vol targeting (42d)")
    wts_rolling_vt = apply_vol_targeting(base_weights, returns_bt, vol_target=VOL_TARGET)
    bt_b = simple_backtest(wts_rolling_vt, returns_bt, cost_bps=COST_BPS)
    eq_b = pd.Series(bt_b["portfolio_equity"].values, index=bt_b["ts"])
    m_b = compute_metrics(eq_b)
    m_b["label"] = "B: Rolling VolTarget"
    strategies["B: Rolling VolTarget"] = {"bt": bt_b, "eq": eq_b, "m": m_b}

    # C. GARCH portfolio vol targeting
    print(f"  [C] GARCH portfolio vol targeting ({best_model})")

    # Compute portfolio returns using base weights, then fit GARCH on that
    w_held = base_weights.shift(1).fillna(0.0)
    common_cols = w_held.columns.intersection(returns_bt.columns)
    port_ret = (w_held[common_cols] * returns_bt[common_cols].reindex(w_held.index).fillna(0.0)).sum(axis=1)
    port_ret = port_ret.loc[port_ret.index >= START]

    print("  Fitting GARCH to portfolio returns ...")
    garch_port_vol = fit_garch_forecasts(
        port_ret, best_model,
        fit_window=GARCH_FIT_WINDOW,
        refit_every=GARCH_REFIT_EVERY,
    )
    n_valid = garch_port_vol.notna().sum()
    print(f"  {n_valid} portfolio GARCH forecasts")

    wts_garch_vt = apply_garch_vol_targeting(
        base_weights, returns_bt, garch_port_vol
    )
    bt_c = simple_backtest(wts_garch_vt, returns_bt, cost_bps=COST_BPS)
    eq_c = pd.Series(bt_c["portfolio_equity"].values, index=bt_c["ts"])
    m_c = compute_metrics(eq_c)
    m_c["label"] = "C: GARCH VolTarget"
    strategies["C: GARCH VolTarget"] = {"bt": bt_c, "eq": eq_c, "m": m_c}

    # D. GARCH-based IVW weighting
    print(f"  [D] GARCH IVW weighting ({best_model})")

    # Build wide GARCH vol for top universe assets
    all_eligible = panel.loc[panel["in_universe"], "symbol"].unique()
    # Fit GARCH for the most liquid 30 assets
    garch_ivw_syms = adv.head(30).index.tolist()
    garch_vol_dict = {}
    for sym in garch_ivw_syms:
        if sym not in returns_wide.columns:
            continue
        sr = returns_wide[sym].dropna()
        if len(sr) < GARCH_FIT_WINDOW + 50:
            continue
        gv = fit_garch_forecasts(sr, best_model, refit_every=GARCH_REFIT_EVERY * 2)
        if gv.notna().sum() > 100:
            garch_vol_dict[sym] = gv
            print(f"    {sym}: {gv.notna().sum()} forecasts")

    garch_vol_wide = pd.DataFrame(garch_vol_dict)
    garch_vol_wide.to_parquet(ARTIFACT_DIR / "garch_vol_wide.parquet")

    garch_ivw_wts = build_emac_weights_garch_ivw(panel, garch_vol_wide)
    wts_d = apply_vol_targeting(garch_ivw_wts, returns_bt, vol_target=VOL_TARGET)

    bt_d = simple_backtest(wts_d, returns_bt, cost_bps=COST_BPS)
    eq_d = pd.Series(bt_d["portfolio_equity"].values, index=bt_d["ts"])
    m_d = compute_metrics(eq_d)
    m_d["label"] = "D: GARCH IVW + VolTarget"
    strategies["D: GARCH IVW + VolTarget"] = {"bt": bt_d, "eq": eq_d, "m": m_d}

    # BTC benchmark
    btc_eq = compute_btc_benchmark(panel)
    btc_c = btc_eq.reindex(eq_a.index).ffill().bfill()
    if len(btc_c) > 0:
        btc_c = btc_c / btc_c.iloc[0]
        m_btc = compute_metrics(btc_c)
        m_btc["label"] = "BTC Buy & Hold"
    else:
        m_btc = {"label": "BTC Buy & Hold"}

    # ------------------------------------------------------------------
    # Part 5: Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: GARCH VOL TARGETING vs ROLLING VOL TARGETING")
    print("=" * 70)

    all_metrics = [s["m"] for s in strategies.values()] + [m_btc]
    print("\n" + format_metrics_table(all_metrics))

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(ARTIFACT_DIR / "strategy_metrics.csv", index=False, float_format="%.4f")

    # Turnover comparison
    print("\n  Turnover comparison:")
    print(f"  {'Strategy':<30s} {'AvgTurnover':>12s} {'AvgExposure':>12s}")
    print(f"  {'-'*56}")
    for label, s in strategies.items():
        bt = s["bt"]
        print(f"  {label:<30s} {bt['turnover'].mean():>12.4f} {bt['gross_exposure'].mean():>12.2f}")

    # Realized vol of strategies
    print("\n  Realized portfolio vol (target: 20%):")
    for label, s in strategies.items():
        eq = s["eq"]
        rv = eq.pct_change().dropna().std() * np.sqrt(ANN_FACTOR)
        print(f"  {label:<30s} {rv:.1%}")

    # ------------------------------------------------------------------
    # Part 6: Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")

    COLORS = ["#9E9E9E", "#3b82f6", "#22c55e", "#ef4444", "#FFA726"]

    # Plot 1: Equity curves
    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (label, s) in enumerate(strategies.items()):
        eq = s["eq"]
        lw = 2.0 if "GARCH" in label else 1.2
        ax.plot(eq.index, eq.values, label=label, color=COLORS[i], linewidth=lw)
    if len(btc_c) > 0:
        ax.plot(btc_c.index, btc_c.values, label="BTC B&H",
                color="#FFA726", linewidth=0.8, alpha=0.5, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log)", fontsize=11)
    ax.set_title("GARCH vs Rolling Vol Targeting — EMAC 21d Momentum", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "equity_curves.png")
    plt.close(fig)
    print("  [1/4] Equity curves")

    # Plot 2: Vol forecast comparison for BTC
    if "BTC-USD" in garch_forecasts:
        fig, ax = plt.subplots(figsize=(16, 6))
        rv_btc = realized_vols["BTC-USD"].dropna()
        ax.plot(rv_btc.index, rv_btc.values, label="Realized Vol (5d fwd)",
                color="gray", alpha=0.4, linewidth=0.8)
        ax.plot(rolling_vols["BTC-USD"].dropna().index,
                rolling_vols["BTC-USD"].dropna().values,
                label="Rolling 42d", color="#3b82f6", linewidth=1.0)
        for spec_name, fcast in garch_forecasts["BTC-USD"].items():
            fc = fcast.dropna()
            if len(fc) > 0:
                ax.plot(fc.index, fc.values, label=spec_name, linewidth=1.0)
        ax.set_ylabel("Annualized Volatility", fontsize=11)
        ax.set_title("BTC Volatility Forecasts: GARCH vs Rolling", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 3.0)
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "btc_vol_forecasts.png")
        plt.close(fig)
        print("  [2/4] BTC vol forecasts")

    # Plot 3: Vol forecast accuracy bar chart
    if len(eval_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        agg = eval_df.groupby("model")[["qlike", "mse", "correlation"]].mean()
        for ax, (col, title) in zip(axes, [
            ("qlike", "QLIKE (lower=better)"),
            ("mse", "MSE (lower=better)"),
            ("correlation", "Correlation (higher=better)")
        ]):
            vals = agg[col].sort_values(ascending=(col != "correlation"))
            ax.barh(range(len(vals)), vals.values,
                    color=["#22c55e" if i == 0 else "#3b82f6" for i in range(len(vals))],
                    alpha=0.85, edgecolor="white")
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(vals.index, fontsize=9)
            ax.set_title(title, fontsize=11)
        fig.suptitle("Volatility Forecast Accuracy (averaged across assets)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "vol_forecast_accuracy.png")
        plt.close(fig)
        print("  [3/4] Forecast accuracy")

    # Plot 4: Realized vol of portfolios over time
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, (label, s) in enumerate(strategies.items()):
        eq = s["eq"]
        rv = eq.pct_change().dropna().rolling(63, min_periods=30).std() * np.sqrt(ANN_FACTOR)
        ax.plot(rv.index, rv.values, label=label, color=COLORS[i], linewidth=1.0)
    ax.axhline(VOL_TARGET, color="red", linewidth=1.5, linestyle="--", alpha=0.7, label=f"Target ({VOL_TARGET:.0%})")
    ax.set_ylabel("Rolling 63d Annualized Vol", fontsize=11)
    ax.set_title("Portfolio Realized Vol: GARCH vs Rolling Targeting", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "portfolio_realized_vol.png")
    plt.close(fig)
    print("  [4/4] Portfolio vol tracking")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("STUDY 1 SUMMARY — GARCH VOLATILITY MODELS")
    print(f"{'='*70}")
    print(f"\nKey findings:")
    m_dict = {s["m"]["label"]: s["m"] for s in strategies.values()}
    for label, m in m_dict.items():
        print(f"  {label:<30s} Sharpe={m.get('sharpe',0):.2f}  "
              f"CAGR={m.get('cagr',0):.1%}  MaxDD={m.get('max_dd',0):.1%}  "
              f"Vol={m.get('vol',0):.1%}")

    if "B: Rolling VolTarget" in m_dict and "C: GARCH VolTarget" in m_dict:
        sr_diff = m_dict["C: GARCH VolTarget"]["sharpe"] - m_dict["B: Rolling VolTarget"]["sharpe"]
        direction = "improves" if sr_diff > 0 else "hurts"
        print(f"\n  GARCH vol targeting {direction} Sharpe by {abs(sr_diff):.2f} vs rolling")

    print(f"\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
