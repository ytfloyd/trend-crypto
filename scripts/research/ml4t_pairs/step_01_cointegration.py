"""
Study 2 — Cointegration & Pairs Trading for Crypto
====================================================
Replicates & extends Chapter 9 of Jansen (2020)
"ML for Algorithmic Trading, 2nd Ed."

Objective: Identify cointegrated crypto pairs and build a mean-reversion
strategy — a completely different alpha source from momentum / ML.

Steps:
  1. Load daily data for top-liquidity universe
  2. Screen all pairs for cointegration (Engle-Granger ADF test)
  3. Estimate hedge ratios via OLS for cointegrated pairs
  4. Compute spread z-scores and build trading signals
  5. Backtest pairs trading strategy with entry/exit thresholds
  6. Compare to momentum baselines

Reference: Jansen (2020) Ch. 9 §"Cointegration" + §"Pairs Trading"
Also: Engle & Granger (1987), Vidyamurthy (2004)
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import itertools
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

from scripts.research.common.data import (
    ANN_FACTOR,
    compute_btc_benchmark,
    filter_universe,
    load_daily_bars,
)
from scripts.research.common.backtest import simple_backtest
from scripts.research.common.metrics import compute_metrics, format_metrics_table

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START = "2018-01-01"
END = "2025-12-31"
DATA_START = "2016-06-01"
MIN_ADV = 1_000_000
MIN_HISTORY = 180
MAX_PAIRS_TO_TEST = 500   # cap on number of pairs to test
COINT_PVALUE = 0.05       # significance threshold for cointegration

# Spread trading params
ZSCORE_LOOKBACK = 63      # rolling z-score window
ENTRY_Z = 2.0             # enter when |z| > 2
EXIT_Z = 0.5              # exit when |z| < 0.5
STOP_Z = 4.0              # stop-loss when |z| > 4
HEDGE_LOOKBACK = 126      # rolling hedge ratio window
REFIT_EVERY = 21          # re-estimate hedge ratio every 21 days

# Portfolio params
MAX_PAIR_ALLOCATION = 0.10  # max 10% per pair
COST_BPS = 20.0

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "research" / "ml4t_pairs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})


# ===================================================================
# 1. Cointegration screening
# ===================================================================
def screen_cointegration(
    close_wide: pd.DataFrame,
    symbols: list[str],
    min_obs: int = 252,
    max_pairs: int = MAX_PAIRS_TO_TEST,
) -> pd.DataFrame:
    """Test all pairs for cointegration using Engle-Granger method.

    Returns DataFrame with columns: sym1, sym2, coint_stat, pvalue,
    adf_stat, hedge_ratio, half_life
    """
    pairs = list(itertools.combinations(symbols, 2))
    if len(pairs) > max_pairs:
        np.random.seed(42)
        pairs = [pairs[i] for i in np.random.choice(len(pairs), max_pairs, replace=False)]

    results = []
    for i, (s1, s2) in enumerate(pairs):
        if s1 not in close_wide.columns or s2 not in close_wide.columns:
            continue

        p1 = close_wide[s1].dropna()
        p2 = close_wide[s2].dropna()
        common = p1.index.intersection(p2.index)
        if len(common) < min_obs:
            continue

        y = np.log(p1.loc[common].values)
        x = np.log(p2.loc[common].values)

        try:
            # Engle-Granger cointegration test
            coint_stat, pvalue, _ = coint(y, x)

            # OLS hedge ratio
            X = add_constant(x)
            model = OLS(y, X).fit()
            hedge_ratio = model.params[1]

            # Spread and half-life
            spread = y - hedge_ratio * x
            spread_diff = np.diff(spread)
            spread_lag = spread[:-1]
            if len(spread_lag) > 10:
                ols_hl = OLS(spread_diff, add_constant(spread_lag)).fit()
                phi = ols_hl.params[1]
                half_life = -np.log(2) / phi if phi < 0 else np.inf
            else:
                half_life = np.inf

            # ADF on the spread itself
            adf_stat, adf_p, _, _, _, _ = adfuller(spread, maxlag=21)

            results.append({
                "sym1": s1,
                "sym2": s2,
                "coint_stat": coint_stat,
                "pvalue": pvalue,
                "adf_stat": adf_stat,
                "adf_pvalue": adf_p,
                "hedge_ratio": hedge_ratio,
                "half_life": half_life,
                "n_obs": len(common),
            })

        except Exception:
            continue

        if (i + 1) % 100 == 0:
            print(f"    tested {i+1}/{len(pairs)} pairs ...", flush=True)

    return pd.DataFrame(results)


# ===================================================================
# 2. Spread computation and z-score signals
# ===================================================================
def compute_spread_signals(
    close_wide: pd.DataFrame,
    pair: dict,
    hedge_lookback: int = HEDGE_LOOKBACK,
    zscore_lookback: int = ZSCORE_LOOKBACK,
    refit_every: int = REFIT_EVERY,
) -> pd.DataFrame:
    """Compute rolling spread, z-score, and trading signals for a pair."""
    s1, s2 = pair["sym1"], pair["sym2"]
    p1 = np.log(close_wide[s1])
    p2 = np.log(close_wide[s2])

    common = p1.dropna().index.intersection(p2.dropna().index)
    p1 = p1.loc[common]
    p2 = p2.loc[common]

    # Rolling hedge ratio
    hedge_ratios = pd.Series(np.nan, index=common)
    for i in range(hedge_lookback, len(common)):
        if (i - hedge_lookback) % refit_every != 0 and not np.isnan(hedge_ratios.iloc[i - 1]):
            hedge_ratios.iloc[i] = hedge_ratios.iloc[i - 1]
            continue
        y = p1.iloc[i - hedge_lookback:i].values
        x = p2.iloc[i - hedge_lookback:i].values
        X = add_constant(x)
        try:
            model = OLS(y, X).fit()
            hedge_ratios.iloc[i] = model.params[1]
        except Exception:
            pass
    hedge_ratios = hedge_ratios.ffill()

    # Spread
    spread = p1 - hedge_ratios * p2
    spread = spread.dropna()

    # Z-score
    spread_ma = spread.rolling(zscore_lookback, min_periods=max(10, zscore_lookback // 2)).mean()
    spread_std = spread.rolling(zscore_lookback, min_periods=max(10, zscore_lookback // 2)).std()
    zscore = (spread - spread_ma) / spread_std.clip(lower=1e-8)

    return pd.DataFrame({
        "ts": spread.index,
        "spread": spread.values,
        "zscore": zscore.values,
        "hedge_ratio": hedge_ratios.reindex(spread.index).values,
    }).set_index("ts")


# ===================================================================
# 3. Pairs trading backtest
# ===================================================================
def backtest_pair(
    spread_df: pd.DataFrame,
    close_wide: pd.DataFrame,
    pair: dict,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
    stop_z: float = STOP_Z,
    cost_bps: float = COST_BPS,
) -> dict:
    """Backtest a single pair with z-score entry/exit rules."""
    s1, s2 = pair["sym1"], pair["sym2"]
    df = spread_df.copy()
    df = df.loc[df.index >= START]

    if len(df) < 60:
        return {"label": f"{s1}/{s2}", "sharpe": np.nan}

    position = 0  # +1 = long spread (long s1, short s2), -1 = short spread
    positions = []
    pnl = []

    p1 = close_wide[s1].reindex(df.index).ffill()
    p2 = close_wide[s2].reindex(df.index).ffill()

    for i in range(1, len(df)):
        z = df["zscore"].iloc[i]
        hr = df["hedge_ratio"].iloc[i]

        # Entry signals
        if position == 0:
            if z < -entry_z:
                position = 1   # spread is too low → long spread
            elif z > entry_z:
                position = -1  # spread is too high → short spread

        # Exit signals
        elif position == 1:
            if z > -exit_z or z < -stop_z:
                position = 0
        elif position == -1:
            if z < exit_z or z > stop_z:
                position = 0

        positions.append(position)

        # PnL: daily return on the spread
        if i > 0 and not np.isnan(hr):
            ret_s1 = (p1.iloc[i] / p1.iloc[i - 1] - 1.0) if p1.iloc[i - 1] > 0 else 0.0
            ret_s2 = (p2.iloc[i] / p2.iloc[i - 1] - 1.0) if p2.iloc[i - 1] > 0 else 0.0
            spread_ret = ret_s1 - abs(hr) * ret_s2
            pnl.append(positions[-1] * spread_ret)
        else:
            pnl.append(0.0)

    pnl_series = pd.Series(pnl, index=df.index[1:])

    # Deduct costs on position changes
    pos_series = pd.Series(positions, index=df.index[1:])
    turnover = pos_series.diff().abs().fillna(0)
    cost = turnover * (cost_bps / 10_000) * 2  # 2 legs
    pnl_net = pnl_series - cost

    equity = (1 + pnl_net).cumprod()
    m = compute_metrics(equity)
    m["label"] = f"{s1.replace('-USD','')}/{s2.replace('-USD','')}"
    m["sym1"] = s1
    m["sym2"] = s2
    m["avg_position"] = abs(pos_series).mean()
    m["n_trades"] = int(turnover.sum() / 2)
    m["half_life"] = pair.get("half_life", np.nan)
    m["pvalue"] = pair.get("pvalue", np.nan)
    m["equity"] = equity

    return m


# ===================================================================
# 4. Multi-pair portfolio
# ===================================================================
def build_pairs_portfolio(
    pair_equities: dict[str, pd.Series],
    max_alloc: float = MAX_PAIR_ALLOCATION,
) -> pd.Series:
    """Equal-weight portfolio of pair strategies, capped per pair."""
    rets = {}
    for name, eq in pair_equities.items():
        rets[name] = eq.pct_change().fillna(0.0)

    rets_df = pd.DataFrame(rets).dropna(how="all").fillna(0.0)
    n_pairs = len(rets_df.columns)
    if n_pairs == 0:
        return pd.Series(dtype=float)

    # Equal weight, capped
    w = min(1.0 / n_pairs, max_alloc)
    port_ret = rets_df.sum(axis=1) * w
    port_equity = (1 + port_ret).cumprod()
    return port_equity


# ===================================================================
# Main
# ===================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("STUDY 2: COINTEGRATION & PAIRS TRADING FOR CRYPTO")
    print("Replicating Jansen (2020) Ch. 9 — Cointegration & Stat Arb")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Part 1: Load data
    # ------------------------------------------------------------------
    print("\n--- Part 1: Loading data ---")
    panel = load_daily_bars(start=DATA_START, end=END)
    panel = filter_universe(panel, min_adv_usd=MIN_ADV, min_history_days=MIN_HISTORY)

    close_wide = panel.pivot_table(
        index="ts", columns="symbol", values="close", aggfunc="first"
    )

    # Select symbols with sufficient in-universe history during the test period
    in_univ = panel.loc[panel["in_universe"] & (panel["ts"] >= START)].copy()
    sym_counts = in_univ.groupby("symbol")["ts"].count()
    # Require at least 2 years of in-universe days during test period
    min_days_in_test = 365 * 2
    eligible = sorted(sym_counts[sym_counts >= min_days_in_test].index.tolist())

    print(f"  Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols")
    print(f"  Eligible for pairs testing: {len(eligible)} symbols")
    print(f"  Possible pairs: {len(eligible) * (len(eligible) - 1) // 2:,}")

    # ------------------------------------------------------------------
    # Part 2: Cointegration screening
    # ------------------------------------------------------------------
    print("\n--- Part 2: Cointegration Screening ---")
    print(f"  Testing up to {MAX_PAIRS_TO_TEST} pairs (p < {COINT_PVALUE})")

    coint_df = screen_cointegration(close_wide, eligible)

    if len(coint_df) == 0:
        print("  No pairs found. Exiting.")
        return

    coint_df.to_csv(ARTIFACT_DIR / "cointegration_screen.csv", index=False, float_format="%.6f")

    # Filter significant pairs
    sig_pairs = coint_df[coint_df["pvalue"] < COINT_PVALUE].copy()
    sig_pairs = sig_pairs[sig_pairs["half_life"].between(5, 180)]  # reasonable half-lives
    sig_pairs = sig_pairs.sort_values("pvalue")

    print(f"\n  Total pairs tested: {len(coint_df)}")
    print(f"  Cointegrated (p < {COINT_PVALUE}): {len(sig_pairs)}")
    if len(sig_pairs) > 0:
        print(f"  Half-life range: {sig_pairs['half_life'].min():.0f} - {sig_pairs['half_life'].max():.0f} days")

        print(f"\n  Top 20 cointegrated pairs:")
        print(f"  {'Pair':<25s} {'p-value':>10s} {'ADF':>8s} {'HedgeR':>8s} {'HalfLife':>10s} {'Obs':>6s}")
        print(f"  {'-'*70}")
        for _, row in sig_pairs.head(20).iterrows():
            s1 = row['sym1'].replace('-USD', '')
            s2 = row['sym2'].replace('-USD', '')
            print(f"  {s1+'/'+s2:<25s} {row['pvalue']:>10.4f} {row['adf_stat']:>8.2f} "
                  f"{row['hedge_ratio']:>8.3f} {row['half_life']:>9.1f}d {int(row['n_obs']):>6d}")

    # ------------------------------------------------------------------
    # Part 3: Backtest top pairs
    # ------------------------------------------------------------------
    print("\n--- Part 3: Pairs Trading Backtest ---")
    print(f"  Entry: |z| > {ENTRY_Z}, Exit: |z| < {EXIT_Z}, Stop: |z| > {STOP_Z}")

    n_to_trade = min(30, len(sig_pairs))
    trade_pairs = sig_pairs.head(n_to_trade)

    pair_results = []
    pair_equities = {}

    for _, pair_row in trade_pairs.iterrows():
        pair = pair_row.to_dict()
        try:
            spread_df = compute_spread_signals(close_wide, pair)
            m = backtest_pair(spread_df, close_wide, pair)
            if not np.isnan(m.get("sharpe", np.nan)):
                pair_results.append(m)
                if "equity" in m:
                    pair_equities[m["label"]] = m["equity"]
                print(f"  {m['label']:<20s} Sharpe={m['sharpe']:>6.2f}  "
                      f"CAGR={m['cagr']:>7.1%}  MaxDD={m['max_dd']:>7.1%}  "
                      f"Trades={m['n_trades']:>4d}  HL={m['half_life']:>5.0f}d")
        except Exception as e:
            print(f"  {pair['sym1']}/{pair['sym2']}: error — {e}")

    if not pair_results:
        print("  No tradeable pairs found.")
        return

    # Save individual pair results
    pair_metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k != "equity"} for r in pair_results])
    pair_metrics_df.to_csv(ARTIFACT_DIR / "pair_metrics.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------------------
    # Part 4: Multi-pair portfolio
    # ------------------------------------------------------------------
    print("\n--- Part 4: Multi-Pair Portfolio ---")

    # Use pairs with positive Sharpe
    good_pairs = {k: v for k, v in pair_equities.items()
                  if any(r["label"] == k and r["sharpe"] > 0 for r in pair_results)}
    print(f"  Pairs with Sharpe > 0: {len(good_pairs)} / {len(pair_equities)}")

    portfolio_eq = build_pairs_portfolio(good_pairs) if good_pairs else pd.Series(dtype=float)

    # BTC benchmark
    btc_eq = compute_btc_benchmark(panel)

    all_metrics = []

    if len(portfolio_eq) > 10:
        m_port = compute_metrics(portfolio_eq)
        m_port["label"] = "Pairs Portfolio"
        all_metrics.append(m_port)

    # Best single pair
    if pair_results:
        best_pair = max(pair_results, key=lambda x: x.get("sharpe", -99))
        all_metrics.append({k: v for k, v in best_pair.items() if k != "equity"})

    # BTC
    btc_c = btc_eq.reindex(
        portfolio_eq.index if len(portfolio_eq) > 0 else close_wide.loc[START:].index
    ).ffill().bfill()
    if len(btc_c) > 0:
        btc_c = btc_c / btc_c.iloc[0]
        m_btc = compute_metrics(btc_c)
        m_btc["label"] = "BTC Buy & Hold"
        all_metrics.append(m_btc)

    # ------------------------------------------------------------------
    # Part 5: Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: PAIRS TRADING")
    print("=" * 70)

    if all_metrics:
        print("\n" + format_metrics_table(all_metrics))

    # Summary stats
    sharpes = [r["sharpe"] for r in pair_results if not np.isnan(r["sharpe"])]
    if sharpes:
        print(f"\n  Individual pair Sharpe distribution:")
        print(f"    Mean:   {np.mean(sharpes):.2f}")
        print(f"    Median: {np.median(sharpes):.2f}")
        print(f"    Std:    {np.std(sharpes):.2f}")
        print(f"    >0:     {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")
        print(f"    >0.5:   {sum(1 for s in sharpes if s > 0.5)}/{len(sharpes)}")
        print(f"    >1.0:   {sum(1 for s in sharpes if s > 1.0)}/{len(sharpes)}")

    # ------------------------------------------------------------------
    # Part 6: Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")

    # Plot 1: Cointegration p-value distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(coint_df["pvalue"], bins=50, color="#3b82f6", alpha=0.7, edgecolor="white")
    ax.axvline(COINT_PVALUE, color="red", linewidth=1.5, linestyle="--", label=f"p={COINT_PVALUE}")
    ax.set_xlabel("Cointegration p-value")
    ax.set_ylabel("Count")
    ax.set_title("Cointegration Test Distribution")
    ax.legend()

    ax = axes[1]
    if len(sig_pairs) > 0:
        ax.hist(sig_pairs["half_life"].clip(upper=200), bins=30,
                color="#22c55e", alpha=0.7, edgecolor="white")
        ax.set_xlabel("Half-Life (days)")
        ax.set_title("Half-Life of Cointegrated Pairs")

    ax = axes[2]
    if sharpes:
        ax.hist(sharpes, bins=30, color="#ef4444", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_title("Individual Pair Sharpe Distribution")

    fig.suptitle("Cointegration & Pairs Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "cointegration_analysis.png")
    plt.close(fig)
    print("  [1/3] Cointegration analysis")

    # Plot 2: Portfolio equity
    if len(portfolio_eq) > 10:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        ax = axes[0]
        ax.plot(portfolio_eq.index, portfolio_eq.values, label="Pairs Portfolio",
                color="#22c55e", linewidth=2)
        if len(btc_c) > 0:
            ax.plot(btc_c.index, btc_c.values, label="BTC B&H",
                    color="#FFA726", linewidth=1, alpha=0.5, linestyle="--")
        # Plot top 5 individual pairs
        sorted_pairs = sorted(pair_results, key=lambda x: x.get("sharpe", -99), reverse=True)
        for r in sorted_pairs[:5]:
            if r["label"] in pair_equities:
                eq = pair_equities[r["label"]]
                ax.plot(eq.index, eq.values, linewidth=0.7, alpha=0.5,
                        label=f"{r['label']} (SR={r['sharpe']:.2f})")
        ax.set_yscale("log")
        ax.set_ylabel("Equity (log)")
        ax.set_title("Pairs Trading — Equity Curves", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)

        # Drawdown
        ax = axes[1]
        dd = portfolio_eq / portfolio_eq.cummax() - 1.0
        ax.fill_between(dd.index, 0, dd.values, color="#ef4444", alpha=0.4)
        ax.set_ylabel("Drawdown")
        ax.set_title("Pairs Portfolio Drawdown")

        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "pairs_portfolio.png")
        plt.close(fig)
        print("  [2/3] Portfolio equity")

    # Plot 3: Best pair spread analysis
    if pair_results:
        best = sorted_pairs[0]
        pair_info = sig_pairs[
            (sig_pairs["sym1"] == best["sym1"]) & (sig_pairs["sym2"] == best["sym2"])
        ].iloc[0].to_dict() if len(sig_pairs) > 0 else best

        try:
            spread_df = compute_spread_signals(close_wide, pair_info)
            spread_bt = spread_df.loc[spread_df.index >= START]

            fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

            axes[0].plot(spread_bt.index, spread_bt["spread"], color="#3b82f6", linewidth=0.8)
            axes[0].set_ylabel("Log Spread")
            axes[0].set_title(f"Best Pair: {best['label']} (Sharpe={best['sharpe']:.2f})", fontsize=13)

            axes[1].plot(spread_bt.index, spread_bt["zscore"], color="#8b5cf6", linewidth=0.8)
            axes[1].axhline(ENTRY_Z, color="green", linestyle="--", alpha=0.5, label=f"Entry ±{ENTRY_Z}")
            axes[1].axhline(-ENTRY_Z, color="green", linestyle="--", alpha=0.5)
            axes[1].axhline(EXIT_Z, color="orange", linestyle=":", alpha=0.5, label=f"Exit ±{EXIT_Z}")
            axes[1].axhline(-EXIT_Z, color="orange", linestyle=":", alpha=0.5)
            axes[1].axhline(0, color="black", linewidth=0.5)
            axes[1].set_ylabel("Z-Score")
            axes[1].legend(fontsize=8)

            if best["label"] in pair_equities:
                eq = pair_equities[best["label"]]
                axes[2].plot(eq.index, eq.values, color="#22c55e", linewidth=1.2)
                axes[2].set_ylabel("Equity")
                axes[2].set_title("Pair P&L")

            fig.tight_layout()
            fig.savefig(ARTIFACT_DIR / "best_pair_analysis.png")
            plt.close(fig)
            print("  [3/3] Best pair analysis")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("STUDY 2 SUMMARY — COINTEGRATION & PAIRS TRADING")
    print(f"{'='*70}")
    print(f"  Pairs tested: {len(coint_df)}")
    print(f"  Cointegrated (p<{COINT_PVALUE}): {len(sig_pairs)}")
    print(f"  Traded: {len(pair_results)}")
    if sharpes:
        print(f"  Profitable: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")
        print(f"  Best pair: {sorted_pairs[0]['label']} (Sharpe={sorted_pairs[0]['sharpe']:.2f})")
    if len(portfolio_eq) > 10 and all_metrics:
        pm = [m for m in all_metrics if m.get("label") == "Pairs Portfolio"]
        if pm:
            print(f"  Portfolio: Sharpe={pm[0]['sharpe']:.2f} CAGR={pm[0]['cagr']:.1%} MaxDD={pm[0]['max_dd']:.1%}")

    print(f"\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
