"""Medallion Lite — param-frozen WALK-FORWARD (point-in-time universe) + overlays.

(a) Honest OOS: small param grid (entry x trailing) run over the full period on
    the PIT universe; per fold, SELECT the config with the best TRAIN Sortino,
    FREEZE it, score on the fold's TEST window; concatenate test segments into
    one OOS series. No peeking — params are chosen only on past data per fold.

(b) On that walk-forward OOS series, layer the rulebook's Sortino levers:
    vol-targeting (QF-07, neutral-leverage dynamic de-risking) and a regime tilt
    (scale exposure by the ensemble regime score). See if the clean number clears 2.0.

Run: PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_walkforward.py
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

from core.metrics import compute_metrics  # noqa: E402
from medallion_lite.factors import compute_composite_score, compute_factors  # noqa: E402
from medallion_lite.portfolio import backtest_portfolio, build_factor_portfolio  # noqa: E402
from medallion_lite.regime_ensemble import compute_ensemble_regime  # noqa: E402
from sornette_lppl.data_hf import filter_universe_hourly  # noqa: E402

LAKE = str(ROOT.parent / "data" / "coinbase_crypto_ohlcv_lake.duckdb")
MEMBERSHIP = "bars_1d_usd_universe_clean_top50_adv10m_membership"
START, END, TC_BPS = "2021-01-01", "2026-06-01", 30.0

# small grid (kept tiny to limit search-snooping); entry x trailing-stop
GRID = [{"entry_threshold": e, "trailing_stop_pct": t}
        for e in (0.60, 0.65, 0.70) for t in (0.12, 0.15, 0.20)]
FOLDS = [("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
         ("2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
         ("2021-01-01", "2024-12-31", "2025-01-01", "2026-06-01")]


def _daily(hourly_ret: pd.Series) -> pd.Series:
    r = hourly_ret.copy()
    r.index = pd.to_datetime(r.index)
    return (1.0 + r.fillna(0.0)).resample("D").prod() - 1.0


def _sortino(daily: pd.Series, lo=None, hi=None) -> dict:
    d = daily
    if lo:
        d = d[d.index >= lo]
    if hi:
        d = d[d.index <= hi]
    d = d.dropna()
    if len(d) < 20:
        return {"sortino": np.nan, "sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan}
    m = compute_metrics((1.0 + d).cumprod())
    return {k: m[k] for k in ("sortino", "sharpe", "cagr", "max_dd")}


def load_pit():
    con = duckdb.connect(LAKE, read_only=True)
    memb = con.execute(f"SELECT ts, symbol FROM {MEMBERSHIP} WHERE ts>=? AND ts<=? ORDER BY ts",
                       [START, END]).fetch_df()
    memb["ts"] = pd.to_datetime(memb["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    union = sorted(memb["symbol"].unique())
    ph = ", ".join(["?"] * len(union))
    df = con.execute(
        f"""SELECT symbol, ts, open, high, low, close, volume FROM bars_1h
            WHERE ts>=? AND ts<=? AND symbol IN ({ph}) AND open>0 AND close>0 AND high>=low
            ORDER BY symbol, ts""", [START, END] + union).fetch_df()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    df = filter_universe_hourly(df)
    df = df[df["in_universe"]].copy().sort_values(["symbol", "ts"])
    df["ret"] = df.groupby("symbol")["close"].pct_change(fill_method=None)
    rw = df.pivot(index="ts", columns="symbol", values="ret").sort_index().fillna(0)
    cw = df.pivot(index="ts", columns="symbol", values="close").sort_index()
    hw = df.pivot(index="ts", columns="symbol", values="high").sort_index()
    vw = df.pivot(index="ts", columns="symbol", values="volume").sort_index().fillna(0)
    btc_h = df[df["symbol"] == "BTC-USD"].set_index("ts")["close"].sort_index()
    regime = compute_ensemble_regime(btc_h.resample("D").last().dropna(), btc_h, rw)
    composite = compute_composite_score(compute_factors(cw, vw, hw))
    elig = (pd.crosstab(memb["ts"], memb["symbol"]) > 0)
    days = pd.DatetimeIndex(composite.index).normalize()
    elig_h = elig.reindex(index=days, columns=composite.columns).fillna(False)
    elig_h.index = composite.index
    return composite.where(elig_h), rw, regime


def config_daily_returns(composite_pit, rw, regime, params) -> pd.Series:
    w, _ = build_factor_portfolio(
        composite_pit, rw, regime, entry_threshold=params["entry_threshold"],
        exit_score_threshold=0.40, max_hold_hours=336,
        trailing_stop_pct=params["trailing_stop_pct"], rebalance_every_hours=24,
        max_positions=25, max_weight=0.10)
    bt = backtest_portfolio(w, rw, tc_bps=TC_BPS)
    return _daily(pd.Series(bt["net_ret"].values, index=pd.to_datetime(bt["ts"])))


def vol_target(daily: pd.Series, lookback=30) -> pd.Series:
    rv = daily.rolling(lookback, min_periods=10).std() * np.sqrt(365.0)
    target = daily.std() * np.sqrt(365.0)  # neutral leverage (avg ~1x): isolate dynamic de-risk
    lev = (target / rv).clip(upper=3.0).shift(1).fillna(1.0)
    return daily * lev


def regime_tilt(daily: pd.Series, regime_hourly: pd.Series) -> pd.Series:
    rd = regime_hourly.copy()
    rd.index = pd.to_datetime(rd.index)
    rd = rd.resample("D").last().reindex(daily.index).ffill()
    tilt = (rd / rd.mean()).clip(0.0, 2.0).shift(1).fillna(1.0)  # avg exposure ~1
    return daily * tilt


def main() -> None:
    print("loading point-in-time panel + factors/regime (once) ...")
    composite_pit, rw, regime = load_pit()

    # run each grid config over the FULL period once
    runs = {}
    for prm in GRID:
        key = (prm["entry_threshold"], prm["trailing_stop_pct"])
        runs[key] = config_daily_returns(composite_pit, rw, regime, prm)

    # ---- (a) walk-forward: select on train, freeze, score on test ----
    oos_segments = []
    print("\n=== (a) PARAM-FROZEN WALK-FORWARD ===")
    print(f"{'fold test window':<24} {'selected (entry,trail)':<24} {'train Sort':>10} {'TEST Sort':>10}")
    for tr_lo, tr_hi, te_lo, te_hi in FOLDS:
        best_key, best_train = None, -9.9
        for key, daily in runs.items():
            s = _sortino(daily, tr_lo, tr_hi)["sortino"]
            if not np.isnan(s) and s > best_train:
                best_train, best_key = s, key
        sel = runs[best_key]
        te_sort = _sortino(sel, te_lo, te_hi)["sortino"]
        seg = sel[(sel.index >= te_lo) & (sel.index <= te_hi)]
        oos_segments.append(seg)
        print(f"{te_lo[:7]+'..'+te_hi[:7]:<24} {str(best_key):<24} {best_train:>10.2f} {te_sort:>10.2f}")

    oos = pd.concat(oos_segments).sort_index()
    btc = _daily(rw["BTC-USD"])
    btc_oos = btc[btc.index >= "2023-01-01"]

    m = _sortino(oos)
    mb = _sortino(btc_oos)
    print(f"\nconcatenated WALK-FORWARD OOS (2023-2026, frozen params):")
    print(f"  Medallion WF      Sortino {m['sortino']:.2f}  Sharpe {m['sharpe']:.2f}  CAGR {m['cagr']:.0%}  DD {m['max_dd']:.0%}")
    print(f"  BTC buy&hold      Sortino {mb['sortino']:.2f}  Sharpe {mb['sharpe']:.2f}  CAGR {mb['cagr']:.0%}  DD {mb['max_dd']:.0%}")

    # ---- (b) overlays on the walk-forward OOS series ----
    print("\n=== (b) OVERLAYS on the walk-forward OOS series ===")
    vt = vol_target(oos)
    rt = regime_tilt(oos, regime)
    both = regime_tilt(vol_target(oos), regime)
    for name, series in (("WF baseline", oos), ("+ vol-target (QF-07)", vt),
                         ("+ regime tilt", rt), ("+ both", both)):
        mm = _sortino(series)
        print(f"  {name:<24} Sortino {mm['sortino']:.2f}  Sharpe {mm['sharpe']:.2f}  "
              f"CAGR {mm['cagr']:.0%}  DD {mm['max_dd']:.0%}")


if __name__ == "__main__":
    main()
