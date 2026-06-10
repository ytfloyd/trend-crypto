"""
Futures-only grid search: maximize $ replication of the actual book ($77,329)
using QI mini-silver contracts only, no options overlay, no vol-scaling.

Search dimensions:
  - Timeframe                  : 1H, 4H, 8H, 1D
  - min_hold_bars              : 1, 3, 5, 10, 20
  - hysteresis_bars            : 0, 1, 3
  - sizing_mode + sizing_params:
      * fixed(n)               n in {2,4,6,10}
      * vol_target(target_$/d) target in {5k,10k,15k,25k}
      * confidence_scaled(b,m) (b,m) in {(1,4),(2,8),(4,16)}

The trend layer (fast/slow/rsi/macd/adx/atr/pattern) and the BB-regime + vov
flags are LOCKED to the v3 winner (artifacts/best_params_v3.json).

Selection objective:
  Primary  : minimize |sim_pnl - 77329|
  Subject to:
      max_drawdown_$  <= 40,000
      sharpe_annualized >= 0.5
  Tie-break: higher Sharpe.

Outputs:
  artifacts/futures_only_grid.parquet   — full grid results
  artifacts/futures_only_top20.csv      — top 20 by objective (constrained)
  artifacts/futures_only_best.json      — best constrained config
  artifacts/futures_only_oracle.json    — best unconstrained ($-replication
                                          oracle for context)
  artifacts/futures_only_summary.md     — human-readable summary
  figures/09_futures_only_equity_curves.png
  figures/10_futures_only_efficient_frontier.png
  figures/11_futures_only_pnl_distribution.png
"""

from __future__ import annotations

import json
import pathlib
import sys
from itertools import product
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.signal_grammar import SignalParams, SignalGrammar
from src import sizing as siz
from src.backtest_unscaled import simulate_futures_pnl
from src.backtest import (
    load_bars,
    load_features,
    load_trades,
    reconstruct_position_path,
    realised_pnl_curve,
)

ART = ROOT / "artifacts"
FIG = ROOT / "figures"
ART.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

ACTUAL_TOTAL = 77_329.12          # all instruments
ACTUAL_QI_ONLY = -28_824.80       # futures-only realized pnl from the book
TARGET = ACTUAL_TOTAL

# ----------------------------------------------------------- constants --

TIMEFRAMES = ["1H", "4H", "8H", "1D"]
MIN_HOLDS = [1, 3, 5, 10, 20]
HYSTERESES = [0, 1, 3]

FIXED_NS = [2, 4, 6, 10]
TARGET_DOLLAR_VOL_PER_DAYS = [5_000, 10_000, 15_000, 25_000]
CONF_BASE_MAX = [(1, 4), (2, 8), (4, 16)]

CONSTRAINT_MAX_DD = 40_000.0
CONSTRAINT_MIN_SHARPE = 0.5


# ----------------------------------------------------------- load v3 --


def _load_v3() -> Dict[str, Any]:
    return json.loads((ART / "best_params_v3.json").read_text())["params"]


V3 = _load_v3()


# --------------------------------------------- per-timeframe pre-load --


def _prep_timeframe(tf: str):
    bars = load_bars(tf)
    feats = load_features(tf)
    # Restrict feature window so backtest excludes very leading NaNs in indicators
    feats = feats.dropna(subset=[f"sma_{V3['fast']}", f"sma_{V3['slow']}"])
    bars = bars.loc[feats.index]
    return bars, feats


# ------------------------------------------------------------- runner --


def _build_contracts(state: pd.Series,
                     features: pd.DataFrame,
                     params: SignalParams,
                     sizing_mode: str,
                     sizing_args: Dict[str, Any],
                     tf: str) -> pd.Series:
    if sizing_mode == "fixed":
        return siz.fixed(state, n_contracts=sizing_args["n_contracts"])
    if sizing_mode == "vol_target":
        atr = features["atr_14"].astype(float).ffill().fillna(0.0)
        return siz.vol_target(
            state,
            atr_per_oz=atr,
            target_dollar_vol_per_day=sizing_args["target_$_per_day"],
            bars_per_day=siz.bars_per_day(tf),
        )
    if sizing_mode == "confidence_scaled":
        return siz.confidence_scaled(
            state,
            features=features,
            signal_params=params,
            base_contracts=sizing_args["base"],
            max_contracts=sizing_args["max"],
        )
    raise ValueError(f"unknown sizing_mode {sizing_mode!r}")


def _evaluate_one(tf: str,
                  bars: pd.DataFrame,
                  feats: pd.DataFrame,
                  min_hold: int,
                  hysteresis: int,
                  sizing_mode: str,
                  sizing_args: Dict[str, Any]) -> Dict[str, Any]:
    params = SignalParams(**{**V3, "min_hold_bars": min_hold,
                             "hysteresis_bars": hysteresis})
    state = SignalGrammar(params).generate(feats)
    contracts = _build_contracts(state, feats, params, sizing_mode, sizing_args, tf)
    stats = simulate_futures_pnl(
        state, contracts, bars,
        bars_per_year=siz.bars_per_year(tf),
    )
    row = dict(
        tf=tf,
        min_hold_bars=int(min_hold),
        hysteresis_bars=int(hysteresis),
        sizing_mode=sizing_mode,
        sizing_args=json.dumps(sizing_args, sort_keys=True),
    )
    row.update(stats)
    row["abs_err_to_target"] = abs(stats["total_pnl"] - TARGET)
    row["pct_replication"] = stats["total_pnl"] / TARGET * 100.0
    return row


def _iter_combos() -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    for tf, mh, hb in product(TIMEFRAMES, MIN_HOLDS, HYSTERESES):
        for n in FIXED_NS:
            combos.append(dict(tf=tf, min_hold=mh, hysteresis=hb,
                               sizing_mode="fixed",
                               sizing_args={"n_contracts": n}))
        for t in TARGET_DOLLAR_VOL_PER_DAYS:
            combos.append(dict(tf=tf, min_hold=mh, hysteresis=hb,
                               sizing_mode="vol_target",
                               sizing_args={"target_$_per_day": t}))
        for b, m in CONF_BASE_MAX:
            combos.append(dict(tf=tf, min_hold=mh, hysteresis=hb,
                               sizing_mode="confidence_scaled",
                               sizing_args={"base": b, "max": m}))
    return combos


# --------------------------------------------------------- equity curve --


def _equity_curve(tf: str, row: Dict[str, Any]) -> pd.Series:
    bars, feats = _prep_timeframe(tf)
    sizing_args = json.loads(row["sizing_args"])
    params = SignalParams(**{**V3, "min_hold_bars": int(row["min_hold_bars"]),
                             "hysteresis_bars": int(row["hysteresis_bars"])})
    state = SignalGrammar(params).generate(feats)
    contracts = _build_contracts(state, feats, params, row["sizing_mode"],
                                 sizing_args, tf)
    stats, curve = simulate_futures_pnl(
        state, contracts, bars,
        bars_per_year=siz.bars_per_year(tf),
        return_curve=True,
    )
    return curve["equity"].rename(f"{tf}/{row['sizing_mode']}/mh{row['min_hold_bars']}/hb{row['hysteresis_bars']}")


# --------------------------------------------------------------- main --


def main():
    print("=== futures-only grid search ===")
    print(f"target: ${TARGET:,.2f}; constraints: maxDD<=${CONSTRAINT_MAX_DD:,.0f}, "
          f"sharpe>={CONSTRAINT_MIN_SHARPE}")
    combos = _iter_combos()
    print(f"total combinations: {len(combos)}")
    assert len(combos) <= 8_000, "exceeded combo cap"

    # Preload bars/features per timeframe
    cache = {tf: _prep_timeframe(tf) for tf in TIMEFRAMES}

    rows = []
    for i, c in enumerate(combos):
        tf = c["tf"]
        bars, feats = cache[tf]
        try:
            r = _evaluate_one(tf, bars, feats,
                              c["min_hold"], c["hysteresis"],
                              c["sizing_mode"], c["sizing_args"])
        except Exception as exc:
            print(f"  combo {i} failed: {exc!r}")
            continue
        rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(combos)} done. last={r['total_pnl']:.0f} "
                  f"err={r['abs_err_to_target']:.0f}")

    grid = pd.DataFrame(rows)
    grid.to_parquet(ART / "futures_only_grid.parquet")
    print(f"saved {len(grid)} rows -> futures_only_grid.parquet")

    # -------- constrained selection --------
    feasible = grid[
        (grid["max_drawdown_$"] <= CONSTRAINT_MAX_DD) &
        (grid["sharpe_annualized"] >= CONSTRAINT_MIN_SHARPE)
    ].copy()
    if len(feasible) == 0:
        print("WARNING: no feasible configs under constraints. Relaxing for picks.")
        feasible = grid.copy()
    feasible = feasible.sort_values(
        ["abs_err_to_target", "sharpe_annualized"],
        ascending=[True, False],
    )
    feasible.head(20).to_csv(ART / "futures_only_top20.csv", index=False)
    best = feasible.iloc[0].to_dict()
    (ART / "futures_only_best.json").write_text(json.dumps(best, indent=2, default=str))
    print(f"BEST CONSTRAINED: pnl=${best['total_pnl']:,.0f} "
          f"(err=${best['abs_err_to_target']:,.0f}), "
          f"sharpe={best['sharpe_annualized']:.2f}, "
          f"maxDD=${best['max_drawdown_$']:,.0f}, "
          f"tf={best['tf']}, mh={best['min_hold_bars']}, "
          f"hb={best['hysteresis_bars']}, mode={best['sizing_mode']}, "
          f"args={best['sizing_args']}")

    # -------- unconstrained oracle --------
    oracle = grid.sort_values("abs_err_to_target").iloc[0].to_dict()
    (ART / "futures_only_oracle.json").write_text(json.dumps(oracle, indent=2, default=str))
    print(f"ORACLE (no constraint): pnl=${oracle['total_pnl']:,.0f} "
          f"(err=${oracle['abs_err_to_target']:,.0f}), "
          f"sharpe={oracle['sharpe_annualized']:.2f}, "
          f"maxDD=${oracle['max_drawdown_$']:,.0f}, "
          f"tf={oracle['tf']}, mh={oracle['min_hold_bars']}, "
          f"hb={oracle['hysteresis_bars']}, mode={oracle['sizing_mode']}, "
          f"args={oracle['sizing_args']}")

    # =================== figures =====================
    print("rendering figures...")
    _make_figures(grid, feasible, best, oracle)

    # =================== summary =====================
    _write_summary(grid, feasible, best, oracle)
    print("done.")


def _make_figures(grid, feasible, best, oracle):
    # ---- actual equity curve (full book + QI-only) ----
    bars_8h = load_bars("8H")
    trades = load_trades()
    pnl_actual = realised_pnl_curve(trades, bars_8h.index)

    qi_trades = trades[trades["Symbol"].str.startswith("QI")]
    pnl_qi = realised_pnl_curve(qi_trades, bars_8h.index)

    # ---- top 5 equity curves ----
    top5 = feasible.head(5)
    curves = []
    for _, row in top5.iterrows():
        c = _equity_curve(row["tf"], row.to_dict())
        curves.append(c)

    # ----- figure 09: equity curves -----
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in curves:
        ax.plot(c.index, c.values, alpha=0.8, label=c.name)
    ax.plot(pnl_actual.index, pnl_actual.values, color="black", linewidth=2.5,
            label="actual book (all)")
    ax.plot(pnl_qi.index, pnl_qi.values, color="red", linewidth=2,
            linestyle="--", label="actual QI futures only")
    ax.axhline(TARGET, color="black", linestyle=":", alpha=0.5)
    ax.axhline(ACTUAL_QI_ONLY, color="red", linestyle=":", alpha=0.5)
    ax.set_title("Futures-only replication — top-5 equity curves vs actual")
    ax.set_ylabel("Cumulative $ P&L")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "09_futures_only_equity_curves.png", dpi=150)
    plt.close(fig)

    # ----- figure 10: efficient frontier scatter -----
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(grid["max_drawdown_$"], grid["total_pnl"],
                    c=grid["sharpe_annualized"], cmap="viridis",
                    alpha=0.6, s=20)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Sharpe (annualized)")
    ax.axhline(TARGET, color="black", linestyle="--", linewidth=1,
               label=f"actual total ${TARGET:,.0f}")
    ax.axhline(ACTUAL_QI_ONLY, color="red", linestyle="--", linewidth=1,
               label=f"actual QI-only ${ACTUAL_QI_ONLY:,.0f}")
    ax.axvline(CONSTRAINT_MAX_DD, color="grey", linestyle=":",
               label=f"DD limit ${CONSTRAINT_MAX_DD:,.0f}")
    ax.scatter([best["max_drawdown_$"]], [best["total_pnl"]],
               s=240, edgecolor="red", facecolor="none", linewidth=2,
               label="best (constrained)")
    ax.scatter([oracle["max_drawdown_$"]], [oracle["total_pnl"]],
               s=240, edgecolor="blue", facecolor="none", linewidth=2,
               label="best (oracle)")
    ax.set_xlabel("Max Drawdown ($)")
    ax.set_ylabel("Total P&L ($)")
    ax.set_title("Efficient frontier: total P&L vs max drawdown (color = Sharpe)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "10_futures_only_efficient_frontier.png", dpi=150)
    plt.close(fig)

    # ----- figure 11: P&L distribution -----
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(grid["total_pnl"], bins=80, color="steelblue", alpha=0.75,
            edgecolor="white")
    ax.axvline(ACTUAL_QI_ONLY, color="red", linestyle="--",
               label=f"actual QI-only ${ACTUAL_QI_ONLY:,.0f}")
    ax.axvline(TARGET, color="black", linestyle="--",
               label=f"actual total ${TARGET:,.0f}")
    ax.axvline(best["total_pnl"], color="green", linestyle="-", linewidth=2,
               label=f"best constrained ${best['total_pnl']:,.0f}")
    ax.axvline(oracle["total_pnl"], color="purple", linestyle="-",
               linewidth=1.5,
               label=f"oracle ${oracle['total_pnl']:,.0f}")
    ax.set_xlabel("Total P&L ($)")
    ax.set_ylabel("# configurations")
    ax.set_title("Futures-only grid: total P&L distribution")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "11_futures_only_pnl_distribution.png", dpi=150)
    plt.close(fig)


def _write_summary(grid, feasible, best, oracle):
    n_feasible = int(((grid["max_drawdown_$"] <= CONSTRAINT_MAX_DD) &
                      (grid["sharpe_annualized"] >= CONSTRAINT_MIN_SHARPE)).sum())
    best_args = json.loads(best["sizing_args"])
    oracle_args = json.loads(oracle["sizing_args"])

    # Bars/day -> hold in calendar days
    bpd = siz.bars_per_day(best["tf"])
    longest_days = best["longest_hold_bars"] / bpd

    lines = [
        "# Futures-only replication — summary",
        "",
        f"- Target (actual book total): **${TARGET:,.2f}**",
        f"- Actual QI futures-only realised: **${ACTUAL_QI_ONLY:,.2f}**",
        f"- Grid size: {len(grid)} configs ({n_feasible} feasible under "
        f"maxDD <= ${CONSTRAINT_MAX_DD:,.0f}, Sharpe >= {CONSTRAINT_MIN_SHARPE})",
        "",
        "## Best constrained config",
        "",
        f"| field | value |",
        f"|---|---|",
        f"| timeframe | {best['tf']} |",
        f"| min_hold_bars | {best['min_hold_bars']} |",
        f"| hysteresis_bars | {best['hysteresis_bars']} |",
        f"| sizing_mode | {best['sizing_mode']} |",
        f"| sizing_args | `{best['sizing_args']}` |",
        f"| total $ P&L | **${best['total_pnl']:,.0f}** |",
        f"| % of $77,329 target | **{best['pct_replication']:.1f}%** |",
        f"| Sharpe (annualized) | {best['sharpe_annualized']:.2f} |",
        f"| Max drawdown ($) | ${best['max_drawdown_$']:,.0f} |",
        f"| # trades | {int(best['num_trades'])} |",
        f"| longest hold | {int(best['longest_hold_bars'])} bars (~{longest_days:.1f} cal days) |",
        f"| gross P&L | ${best['gross_pnl']:,.0f} |",
        f"| commission paid | ${best['total_commission']:,.0f} |",
        "",
        "## Unconstrained oracle (no DD/Sharpe guardrails)",
        "",
        f"| field | value |",
        f"|---|---|",
        f"| timeframe | {oracle['tf']} |",
        f"| min_hold_bars | {oracle['min_hold_bars']} |",
        f"| hysteresis_bars | {oracle['hysteresis_bars']} |",
        f"| sizing_mode | {oracle['sizing_mode']} |",
        f"| sizing_args | `{oracle['sizing_args']}` |",
        f"| total $ P&L | **${oracle['total_pnl']:,.0f}** |",
        f"| % of $77,329 target | **{oracle['pct_replication']:.1f}%** |",
        f"| Sharpe (annualized) | {oracle['sharpe_annualized']:.2f} |",
        f"| Max drawdown ($) | ${oracle['max_drawdown_$']:,.0f} |",
        "",
        "## What changed vs the prior $63k v3 'futures-only' sim",
        "",
        "Prior v3 sim vol-scaled the simulated futures P&L *down* to match the "
        "actual book's daily $ vol — that compressed it to ~$63k. This grid "
        "drops that scaling entirely and lets sizing emerge from the search.",
        "",
        f"- Sizing mode chosen: **{best['sizing_mode']}** with `{best['sizing_args']}`",
        f"- Hold layer: **min_hold_bars={best['min_hold_bars']}**, "
        f"**hysteresis_bars={best['hysteresis_bars']}** "
        f"(prior v3 implicit: 1/0)",
        f"- Timeframe: **{best['tf']}** (v3 was 8H)",
        f"- Longest single position run: {int(best['longest_hold_bars'])} bars "
        f"({longest_days:.1f} calendar days), which captured the trend regimes "
        f"the v3 sim flat-flipped through.",
        "",
        "## Files",
        "",
        "- `artifacts/futures_only_grid.parquet`",
        "- `artifacts/futures_only_top20.csv`",
        "- `artifacts/futures_only_best.json`",
        "- `artifacts/futures_only_oracle.json`",
        "- `figures/09_futures_only_equity_curves.png`",
        "- `figures/10_futures_only_efficient_frontier.png`",
        "- `figures/11_futures_only_pnl_distribution.png`",
    ]
    (ART / "futures_only_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
