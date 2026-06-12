"""Medallion Lite on a POINT-IN-TIME universe — the survivorship-free test.

The flagship run picks its 50 names by FULL-PERIOD median dollar volume (look-
ahead: only survivors). This re-runs the SAME strategy but restricts entries to
names that were actually top-50-by-ADV members AS OF each date, using the lake's
purpose-built point-in-time membership table
(bars_1d_usd_universe_clean_top50_adv10m_membership). If Sortino stays > 2 here,
the edge is real; if it collapses, the headline was survivorship.

Run:  PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_pit.py
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
START, END = "2021-01-01", "2026-06-01"
TC_BPS = 30.0


def daily_metrics(hourly_ret: pd.Series, lo: str | None) -> dict:
    r = hourly_ret.copy()
    r.index = pd.to_datetime(r.index)
    daily = (1.0 + r.fillna(0.0)).resample("D").prod() - 1.0
    if lo:
        daily = daily[daily.index >= lo]
    daily = daily.dropna()
    if len(daily) < 30:
        return {"sortino": np.nan, "sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan}
    m = compute_metrics((1.0 + daily).cumprod())
    return {k: m[k] for k in ("sortino", "sharpe", "cagr", "max_dd")}


def main() -> None:
    con = duckdb.connect(LAKE, read_only=True)
    memb = con.execute(
        f"SELECT ts, symbol FROM {MEMBERSHIP} WHERE ts >= ? AND ts <= ? ORDER BY ts",
        [START, END],
    ).fetch_df()
    memb["ts"] = pd.to_datetime(memb["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    union = sorted(memb["symbol"].unique())
    print(f"point-in-time universe: {len(union)} distinct member names over {START}..{END}")

    ph = ", ".join(["?"] * len(union))
    df = con.execute(
        f"""SELECT symbol, ts, open, high, low, close, volume FROM bars_1h
            WHERE ts >= ? AND ts <= ? AND symbol IN ({ph})
              AND open > 0 AND close > 0 AND high >= low
            ORDER BY symbol, ts""",
        [START, END] + union,
    ).fetch_df()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    df = filter_universe_hourly(df)
    df = df[df["in_universe"]].copy()

    df = df.sort_values(["symbol", "ts"])
    df["ret"] = df.groupby("symbol")["close"].pct_change(fill_method=None)
    returns_wide = df.pivot(index="ts", columns="symbol", values="ret").sort_index().fillna(0)
    close_wide = df.pivot(index="ts", columns="symbol", values="close").sort_index()
    high_wide = df.pivot(index="ts", columns="symbol", values="high").sort_index()
    volume_wide = df.pivot(index="ts", columns="symbol", values="volume").sort_index().fillna(0)

    btc_h = df[df["symbol"] == "BTC-USD"].set_index("ts")["close"].sort_index()
    regime = compute_ensemble_regime(btc_h.resample("D").last().dropna(), btc_h, returns_wide)

    factors = compute_factors(close_wide, volume_wide, high_wide)
    composite = compute_composite_score(factors)

    # --- point-in-time eligibility mask: member as-of the bar's date ---
    elig = (pd.crosstab(memb["ts"], memb["symbol"]) > 0)
    days = pd.DatetimeIndex(composite.index).normalize()
    elig_h = elig.reindex(index=days, columns=composite.columns).fillna(False)
    elig_h.index = composite.index
    avg_elig = elig_h.sum(axis=1).mean()
    composite_pit = composite.where(elig_h)
    print(f"avg eligible names per bar (point-in-time): {avg_elig:.1f}")

    weights, _ = build_factor_portfolio(
        composite_pit, returns_wide, regime,
        entry_threshold=0.65, exit_score_threshold=0.40, max_hold_hours=336,
        trailing_stop_pct=0.15, rebalance_every_hours=24, max_positions=25, max_weight=0.10,
    )
    bt = backtest_portfolio(weights, returns_wide, tc_bps=TC_BPS)
    net = pd.Series(bt["net_ret"].values, index=pd.to_datetime(bt["ts"]))
    btc = returns_wide.get("BTC-USD", pd.Series(0.0, index=returns_wide.index))
    btc.index = pd.to_datetime(btc.index)

    print()
    hdr = f"{'series':<26} {'Sortino':>8} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}"
    for lo, name in ((None, "FULL 2021-2026"), ("2023-01-01", "OOS 2023-2026")):
        print(f"=== {name} (daily, POINT-IN-TIME universe) ===")
        print(hdr)
        for label, s in (("Medallion Lite (PIT)", net), ("BTC buy&hold", btc)):
            m = daily_metrics(s, lo)
            print(f"{label:<26} {m['sortino']:>8.2f} {m['sharpe']:>8.2f} {m['cagr']:>8.0%} {m['max_dd']:>8.0%}")
        print()


if __name__ == "__main__":
    main()
