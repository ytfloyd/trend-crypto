"""Medallion Lite — UNIVERSE-DEFINITION sweep (capacity-constrained shop).

We can trade small digital assets, so test widening the universe beyond the
top-50-by-ADV baseline. Every universe is built POINT-IN-TIME from a 20-day
trailing dollar-ADV (survivorship-free, no look-ahead):

  top_N        : each day, the N names with highest trailing ADV
  adv_floor_$X : every USD pair whose trailing ADV >= $X that day
  all_usd      : the entire clean USD universe active that day (no liquidity floor)

The hourly panel + factors + regime are computed ONCE on the union of all USD
symbols; each universe spec is just a different point-in-time eligibility mask
applied to the same composite score. Params are FROZEN at the flagship defaults
(entry 0.65 / trail 0.15) so the comparison is apples-to-apples (no per-universe
re-fitting => no extra data-snooping, per QF-21). The top-50 walk-forward number
(2.03) is the validated reference; this sweep reports frozen-param FULL + OOS.

Run: PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_universe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import duckdb  # noqa: E402

import run_medallion_walkforward as wf  # noqa: E402  (reuse _daily / config_daily_returns / vol_target / GRID / FOLDS)
from core.metrics import compute_metrics  # noqa: E402
from medallion_lite.factors import compute_composite_score, compute_factors  # noqa: E402
from medallion_lite.portfolio import backtest_portfolio, build_factor_portfolio  # noqa: E402
from medallion_lite.regime_ensemble import compute_ensemble_regime  # noqa: E402
from sornette_lppl.data_hf import filter_universe_hourly  # noqa: E402

LAKE = str(ROOT.parent / "data" / "coinbase_crypto_ohlcv_lake.duckdb")
START, END, TC_BPS = "2021-01-01", "2026-06-01", 30.0
OOS_START = "2023-01-01"
PARAMS = {"entry_threshold": 0.65, "trailing_stop_pct": 0.15}  # FROZEN flagship defaults

# universe specs: (label, kind, value)
SPECS = [
    ("top_25", "top", 25),
    ("top_50  (baseline)", "top", 50),
    ("top_100", "top", 100),
    ("top_200", "top", 200),
    ("adv>=$1M", "floor", 1e6),
    ("adv>=$250k", "floor", 250e3),
    ("all_usd", "all", 0),
]


def _metrics(daily: pd.Series, lo=None) -> dict:
    d = (daily if lo is None else daily[daily.index >= lo]).dropna()
    if len(d) < 20:
        return {k: np.nan for k in ("sortino", "sharpe", "cagr", "max_dd")}
    m = compute_metrics((1.0 + d).cumprod())
    return {k: m[k] for k in ("sortino", "sharpe", "cagr", "max_dd")}


def load_universe_panel():
    """Load hourly panel + factors/regime ONCE for the full USD universe, plus a
    daily trailing-ADV table used to build every point-in-time membership mask."""
    con = duckdb.connect(LAKE, read_only=True)
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM bars_1d_usd_universe_clean WHERE ts>=? AND ts<=? ORDER BY symbol",
        [START, END]).fetchall()]
    print(f"  full USD universe: {len(syms)} symbols")

    # daily trailing ADV (20d) -> point-in-time rank, survivorship-free
    dd = con.execute(
        """SELECT symbol, ts, close, volume FROM bars_1d_usd_universe_clean
           WHERE ts>=? AND ts<=? AND close>0 ORDER BY symbol, ts""", [START, END]).fetch_df()
    ph = ", ".join(["?"] * len(syms))
    hf = con.execute(
        f"""SELECT symbol, ts, open, high, low, close, volume FROM bars_1h
            WHERE ts>=? AND ts<=? AND symbol IN ({ph}) AND open>0 AND close>0 AND high>=low
            ORDER BY symbol, ts""", [START, END] + syms).fetch_df()
    con.close()
    print(f"  hourly rows: {len(hf):,}")

    dd["ts"] = pd.to_datetime(dd["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    dd = dd.sort_values(["symbol", "ts"])
    dd["dv"] = dd["close"] * dd["volume"]
    dd["adv20"] = dd.groupby("symbol")["dv"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    dd["rank"] = dd.groupby("ts")["adv20"].rank(ascending=False, method="first")

    hf["ts"] = pd.to_datetime(hf["ts"], utc=True).dt.tz_localize(None)
    hf = filter_universe_hourly(hf)
    hf = hf[hf["in_universe"]].copy().sort_values(["symbol", "ts"])
    hf["ret"] = hf.groupby("symbol")["close"].pct_change(fill_method=None)
    rw = hf.pivot(index="ts", columns="symbol", values="ret").sort_index().fillna(0)
    cw = hf.pivot(index="ts", columns="symbol", values="close").sort_index()
    hw = hf.pivot(index="ts", columns="symbol", values="high").sort_index()
    vw = hf.pivot(index="ts", columns="symbol", values="volume").sort_index().fillna(0)
    btc_h = hf[hf["symbol"] == "BTC-USD"].set_index("ts")["close"].sort_index()
    print("  computing regime + factors (once) ...")
    regime = compute_ensemble_regime(btc_h.resample("D").last().dropna(), btc_h, rw)
    composite = compute_composite_score(compute_factors(cw, vw, hw))
    return composite, rw, regime, dd


def eligibility(dd: pd.DataFrame, kind: str, value: float, columns, hourly_index) -> pd.DataFrame:
    """Build a point-in-time hourly eligibility mask for one universe spec."""
    d = dd[dd["adv20"] > 0].copy()
    if kind == "top":
        d = d[d["rank"] <= value]
    elif kind == "floor":
        d = d[d["adv20"] >= value]
    # kind == "all": keep every active (adv20>0) name
    elig = pd.crosstab(d["ts"], d["symbol"]) > 0
    days = pd.DatetimeIndex(hourly_index).normalize()
    elig_h = elig.reindex(index=days, columns=columns).fillna(False)
    elig_h.index = hourly_index
    return elig_h


def run_spec(composite, rw, regime, elig_h):
    composite_pit = composite.where(elig_h)
    w, _ = build_factor_portfolio(
        composite_pit, rw, regime, entry_threshold=PARAMS["entry_threshold"],
        exit_score_threshold=0.40, max_hold_hours=336,
        trailing_stop_pct=PARAMS["trailing_stop_pct"], rebalance_every_hours=24,
        max_positions=25, max_weight=0.10)
    bt = backtest_portfolio(w, rw, tc_bps=TC_BPS)
    daily = wf._daily(pd.Series(bt["net_ret"].values, index=pd.to_datetime(bt["ts"])))
    avg_names = float((elig_h.sum(axis=1)).mean())
    avg_turn = float(np.mean(bt["turnover"])) if "turnover" in bt else float("nan")
    return daily, avg_names, avg_turn


def _config_daily(composite_pit, rw, regime, prm) -> pd.Series:
    w, _ = build_factor_portfolio(
        composite_pit, rw, regime, entry_threshold=prm["entry_threshold"],
        exit_score_threshold=0.40, max_hold_hours=336,
        trailing_stop_pct=prm["trailing_stop_pct"], rebalance_every_hours=24,
        max_positions=25, max_weight=0.10)
    bt = backtest_portfolio(w, rw, tc_bps=TC_BPS)
    return wf._daily(pd.Series(bt["net_ret"].values, index=pd.to_datetime(bt["ts"])))


def walk_forward_oos(composite, rw, regime, elig_h) -> pd.Series:
    """Param-frozen walk-forward (same protocol as run_medallion_walkforward) for one universe."""
    composite_pit = composite.where(elig_h)
    runs = {(p["entry_threshold"], p["trailing_stop_pct"]): _config_daily(composite_pit, rw, regime, p)
            for p in wf.GRID}
    segs = []
    for tr_lo, tr_hi, te_lo, te_hi in wf.FOLDS:
        best_key, best = None, -9.9
        for key, daily in runs.items():
            s = wf._sortino(daily, tr_lo, tr_hi)["sortino"]
            if not np.isnan(s) and s > best:
                best, best_key = s, key
        sel = runs[best_key]
        segs.append(sel[(sel.index >= te_lo) & (sel.index <= te_hi)])
    return pd.concat(segs).sort_index()


def main() -> None:
    print("loading full USD universe panel (heavy, once) ...")
    composite, rw, regime, dd = load_universe_panel()
    btc = wf._daily(rw["BTC-USD"])

    rows = []
    print(f"\n{'universe':<20}{'avg #elig':>9}{'turn/bar':>9} | "
          f"{'FULL Sort':>10}{'Shrp':>6}{'CAGR':>6}{'DD':>6} | {'OOS Sort':>9}{'Shrp':>6}{'CAGR':>6}{'DD':>6}")
    print("-" * 110)
    for label, kind, value in SPECS:
        elig_h = eligibility(dd, kind, value, composite.columns, composite.index)
        daily, avg_names, avg_turn = run_spec(composite, rw, regime, elig_h)
        f, o = _metrics(daily), _metrics(daily, OOS_START)
        ovt = _metrics(wf.vol_target(daily), OOS_START)
        rows.append((label, avg_names, avg_turn, f, o, ovt))
        print(f"{label:<20}{avg_names:>9.0f}{avg_turn:>9.2f} | "
              f"{f['sortino']:>10.2f}{f['sharpe']:>6.2f}{f['cagr']:>6.0%}{f['max_dd']:>6.0%} | "
              f"{o['sortino']:>9.2f}{o['sharpe']:>6.2f}{o['cagr']:>6.0%}{o['max_dd']:>6.0%}")

    bo = _metrics(btc, OOS_START)
    print("-" * 110)
    print(f"{'BTC buy&hold':<20}{'':>9}{'':>9} | "
          f"{'':>10}{'':>6}{'':>6}{'':>6} | {bo['sortino']:>9.2f}{bo['sharpe']:>6.2f}{bo['cagr']:>6.0%}{bo['max_dd']:>6.0%}")
    print("\n(+vol-target OOS Sortino per universe)")
    for label, _, _, _, _, ovt in rows:
        print(f"  {label:<20} {ovt['sortino']:.2f}")
    print("\nNote: params FROZEN at flagship (0.65/0.15) for apples-to-apples; "
          "top-50 validated walk-forward OOS Sortino = 2.03. 30bps costs, point-in-time ADV.")

    # ---- honest walk-forward (param-frozen) for baseline vs the broad winner ----
    print("\n=== PARAM-FROZEN WALK-FORWARD (honest OOS, fold-selected params) ===")
    for label, kind, value in [("top_50", "top", 50), ("adv>=$1M", "floor", 1e6),
                               ("top_100", "top", 100)]:
        elig_h = eligibility(dd, kind, value, composite.columns, composite.index)
        oos = walk_forward_oos(composite, rw, regime, elig_h)
        m, mvt = _metrics(oos), _metrics(wf.vol_target(oos))
        print(f"  {label:<12} WF-OOS Sortino {m['sortino']:.2f}  Sharpe {m['sharpe']:.2f}  "
              f"CAGR {m['cagr']:.0%}  DD {m['max_dd']:.0%}   (+vol-target {mvt['sortino']:.2f})")


if __name__ == "__main__":
    main()
