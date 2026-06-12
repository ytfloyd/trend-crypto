"""K2 TRADE ATLAS — Sortino > 2.0 hunt on crypto OHLCV.

Implements the OHLCV-feasible, mandate-aligned (long-convexity) rulebook
strategies and ranks them by OUT-OF-SAMPLE Sortino:

  XS-MOM   (QF-01)  cross-sectional momentum: rank by past-return (skip last
                    month), long top / short bottom (or long-only), vol-targeted.
  TSMOM    (QF-10/CV-17) per-asset time-series momentum, inverse-vol sized
                    (equal risk), portfolio vol-targeted — the synthetic-straddle
                    long-convexity core.
  MA-XOVER (TR/ma_5_40) long-only fast/slow MA crossover basket, vol-targeted.

Discipline (QF-21): split IS 2020-2022 / OOS 2023-2026, report OOS, costs in,
treat the best-of-sweep skeptically. Mandate (long convexity): trend/momentum
have a synthetic-long-straddle payoff (CV-17); vol-targeting (QF-07) is the
Sortino lever.

Run:  PYTHONPATH=src python scripts/research/k2_atlas/sortino_hunt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.backtest import simple_backtest  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402

DB = str(Path(__file__).resolve().parents[3].parent / "data" / "coinbase_crypto_ohlcv_lake.duckdb")
TABLE = "bars_1d_usd_universe_clean_adv10m"
COST_BPS = 20.0
IS_END = "2022-12-31"   # IS: 2020-2022 ; OOS: 2023-2026
PANEL_START = "2020-01-01"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_close_volume() -> tuple[pd.DataFrame, pd.DataFrame]:
    import duckdb
    con = duckdb.connect(DB, read_only=True)
    try:
        df = con.execute(
            f"SELECT symbol, ts, close, volume FROM {TABLE} WHERE ts >= ? ORDER BY ts, symbol",
            [PANEL_START],
        ).fetch_df()
    finally:
        con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    close = df.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last").sort_index()
    vol = df.pivot_table(index="ts", columns="symbol", values="volume", aggfunc="last").sort_index()
    return close, vol


# ----------------------------------------------------------------------
# Signals -> wide weights
# ----------------------------------------------------------------------
def _trailing_vol(rets: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return rets.rolling(lookback, min_periods=max(lookback // 2, 5)).std()


def _portfolio_vol_target(weights: pd.DataFrame, rets: pd.DataFrame,
                          target_ann: float, lookback: int = 30) -> pd.DataFrame:
    """Dynamic leverage so trailing realized portfolio vol ~ target_ann (QF-07)."""
    gross_ret = (weights.shift(1) * rets).sum(axis=1)
    realized = gross_ret.rolling(lookback, min_periods=10).std() * np.sqrt(365.0)
    scalar = (target_ann / realized).clip(upper=5.0).fillna(1.0)
    return weights.mul(scalar, axis=0)


def xs_momentum(close: pd.DataFrame, lookback: int, skip: int, top_frac: float,
                long_short: bool, reb: int, vol_target: float) -> pd.DataFrame:
    rets = close.pct_change(fill_method=None)
    signal = close.shift(skip) / close.shift(skip + lookback) - 1.0
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i, ts in enumerate(close.index):
        if i % reb != 0:
            w.iloc[i] = w.iloc[i - 1] if i > 0 else 0.0
            continue
        row = signal.loc[ts].dropna()
        n = len(row)
        if n < 6:
            continue
        k = max(1, int(n * top_frac))
        ranked = row.sort_values()
        longs = ranked.index[-k:]
        w.loc[ts, longs] = 1.0 / k
        if long_short:
            shorts = ranked.index[:k]
            w.loc[ts, shorts] = -1.0 / k
    w = _portfolio_vol_target(w, rets, vol_target)
    return w


def tsmom(close: pd.DataFrame, lookback: int, vol_lookback: int, vol_target: float,
          long_only: bool) -> pd.DataFrame:
    rets = close.pct_change(fill_method=None)
    sig = np.sign(close / close.shift(lookback) - 1.0)
    if long_only:
        sig = sig.clip(lower=0.0)
    vol = _trailing_vol(rets, vol_lookback) * np.sqrt(365.0)
    inv_vol = (vol_target / vol).replace([np.inf, -np.inf], np.nan)
    raw = sig * inv_vol
    n_active = sig.abs().sum(axis=1).replace(0, np.nan)
    w = raw.div(n_active, axis=0).fillna(0.0)  # equal risk budget across active names
    w = _portfolio_vol_target(w, rets, vol_target)
    return w


def ewmac_trend(close: pd.DataFrame, speeds: tuple[tuple[int, int], ...], vol_lookback: int,
                vol_target: float, long_only: bool) -> pd.DataFrame:
    """Multi-speed EWMAC trend (QF-11/QF-20): blend normalized crossover speeds,
    inverse-vol size (equal risk), portfolio vol-target. Smoother than sign(TSMOM)."""
    rets = close.pct_change(fill_method=None)
    vol_d = _trailing_vol(rets, vol_lookback)
    price_vol = (close * vol_d)  # price units of daily vol, for normalization
    fc = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for nf, ns in speeds:
        raw = close.ewm(span=nf, adjust=False).mean() - close.ewm(span=ns, adjust=False).mean()
        norm = (raw / price_vol).clip(-4, 4)  # vol-normalized crossover ~ z-units
        fc = fc.add(norm, fill_value=0.0)
    fc = fc / len(speeds)
    if long_only:
        fc = fc.clip(lower=0.0)
    vol_ann = vol_d * np.sqrt(365.0)
    inv_vol = (vol_target / vol_ann).replace([np.inf, -np.inf], np.nan)
    raw_w = (fc / 2.0) * inv_vol  # /2: forecast ~[-2,2] -> ~unit risk
    n_active = (fc.abs() > 0).sum(axis=1).replace(0, np.nan)
    w = raw_w.div(n_active, axis=0).fillna(0.0)
    w = _portfolio_vol_target(w, rets, vol_target)
    return w


def dd_control(weights: pd.DataFrame, close: pd.DataFrame, max_dd: float = 0.20,
               floor: float = 0.30) -> pd.DataFrame:
    """Scale the whole book down as the strategy's OWN drawdown deepens (QF-06/QF-04).
    At drawdown >= max_dd, exposure is cut to `floor`; linear in between."""
    rets = close.pct_change(fill_method=None)
    port = (weights.shift(1) * rets).sum(axis=1).fillna(0.0)
    eq = (1.0 + port).cumprod()
    dd = eq / eq.cummax() - 1.0
    scale = (1.0 + (1.0 - floor) / max_dd * dd).clip(lower=floor, upper=1.0)  # dd<=0
    return weights.mul(scale.shift(1).fillna(1.0), axis=0)


def ma_xover(close: pd.DataFrame, fast: int, slow: int, vol_target: float) -> pd.DataFrame:
    rets = close.pct_change(fill_method=None)
    sig = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)
    sig = sig.where(close.rolling(slow).mean().notna())
    n_active = sig.sum(axis=1).replace(0, np.nan)
    w = sig.div(n_active, axis=0).fillna(0.0)  # equal-weight long-only basket
    w = _portfolio_vol_target(w, rets, vol_target)
    return w


# ----------------------------------------------------------------------
# Backtest + IS/OOS metrics
# ----------------------------------------------------------------------
def _metrics_window(port_ret: pd.Series, lo: str | None, hi: str | None) -> dict:
    r = port_ret
    if lo:
        r = r[r.index >= lo]
    if hi:
        r = r[r.index <= hi]
    r = r.dropna()
    if len(r) < 30:
        return {"sortino": np.nan, "sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan, "n": len(r)}
    eq = (1.0 + r).cumprod()
    m = compute_metrics(eq)
    return {"sortino": m["sortino"], "sharpe": m["sharpe"], "cagr": m["cagr"],
            "max_dd": m["max_dd"], "n": len(r)}


def run(label: str, weights: pd.DataFrame, close: pd.DataFrame) -> dict:
    rets = close.pct_change(fill_method=None)
    bt = simple_backtest(weights.fillna(0.0), rets.fillna(0.0), cost_bps=COST_BPS)
    port = pd.Series(bt["portfolio_ret"].values, index=pd.to_datetime(bt["ts"]))
    is_m = _metrics_window(port, None, IS_END)
    oos_m = _metrics_window(port, "2023-01-01", None)
    full = _metrics_window(port, None, None)
    return {"label": label, "is": is_m, "oos": oos_m, "full": full}


def majors_universe(close: pd.DataFrame, vol: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep only the top-N names by trailing 60d dollar volume each day (mask others NaN)."""
    dollar = (close * vol).rolling(60, min_periods=20).mean()
    rank = dollar.rank(axis=1, ascending=False)
    keep = rank <= top_n
    return close.where(keep)


def buy_and_hold(close: pd.DataFrame, symbols: list[str] | None) -> pd.DataFrame:
    cols = [s for s in (symbols or list(close.columns)) if s in close.columns]
    sub = close[cols]
    rets = sub.pct_change(fill_method=None)
    w = pd.DataFrame(1.0 / len(cols), index=close.index, columns=cols)
    w = w.where(sub.notna(), 0.0)
    return w.reindex(columns=close.columns, fill_value=0.0)


def main() -> None:
    close, vol = load_close_volume()
    majors = majors_universe(close, vol, top_n=15)
    print(f"universe: {close.shape[1]} symbols, {close.index.min().date()} -> {close.index.max().date()}, "
          f"{len(close)} days; costs={COST_BPS}bps; IS<= {IS_END} < OOS\n")

    results = []
    # TSMOM (QF-10 / CV-17) — long-short and long-only, a few speeds
    for lb in (30, 60, 90, 120):
        for lo in (True, False):
            tag = f"TSMOM lb={lb} {'LO' if lo else 'LS'} vt=30%"
            results.append(run(tag, tsmom(close, lb, 30, 0.30, lo), close))
    # XS-MOM (QF-01) — classic 12-1 variants
    for lb in (90, 180, 270):
        for ls in (True, False):
            tag = f"XSMOM lb={lb} skip=30 top=0.3 {'LS' if ls else 'LO'} reb=7 vt=30%"
            results.append(run(tag, xs_momentum(close, lb, 30, 0.30, ls, 7, 0.30), close))
    # MA crossover long-only basket (TR / ma_5_40 family)
    for fast, slow in ((5, 40), (10, 50), (20, 100)):
        tag = f"MAxover {fast}/{slow} LO vt=30%"
        results.append(run(tag, ma_xover(close, fast, slow, 0.30), close))
    # Multi-speed EWMAC trend (QF-11/QF-20) — the smoother trend core
    SPEEDS = ((8, 24), (16, 48), (32, 96))
    for lo in (True, False):
        tag = f"EWMAC[8/16/32] {'LO' if lo else 'LS'} vt=30%"
        results.append(run(tag, ewmac_trend(close, SPEEDS, 30, 0.30, lo), close))
    # Same EWMAC + drawdown-control overlay (QF-06/QF-04) — cut the left tail (Sortino lever)
    for lo in (True, False):
        w = dd_control(ewmac_trend(close, SPEEDS, 30, 0.30, lo), close, max_dd=0.20, floor=0.25)
        results.append(run(f"EWMAC[8/16/32] {'LO' if lo else 'LS'} +DDctl vt=30%", w, close))
    # Best naive winner (MA 20/100 LO) + drawdown control
    results.append(run("MAxover 20/100 LO +DDctl vt=30%",
                       dd_control(ma_xover(close, 20, 100, 0.30), close, 0.20, 0.25), close))

    # --- Benchmarks: buy-and-hold (the bull-market baseline to beat) ---
    results.append(run("BTC buy&hold", buy_and_hold(close, ["BTC-USD"]), close))
    results.append(run("ETH buy&hold", buy_and_hold(close, ["ETH-USD"]), close))
    results.append(run("EW-majors(15) buy&hold", buy_and_hold(majors, None), close))

    # --- MAJORS-ONLY (top-15 by $vol): avoid alt blowups (Sortino lever) ---
    SP = ((8, 24), (16, 48), (32, 96))
    results.append(run("MAJORS EWMAC LO vt=30%", ewmac_trend(majors, SP, 30, 0.30, True), close))
    results.append(run("MAJORS EWMAC LO +DDctl", dd_control(ewmac_trend(majors, SP, 30, 0.30, True), close, 0.20, 0.25), close))
    results.append(run("MAJORS MAxover 20/100 LO", ma_xover(majors, 20, 100, 0.30), close))
    results.append(run("MAJORS MAxover 20/100 LO +DDctl", dd_control(ma_xover(majors, 20, 100, 0.30), close, 0.20, 0.25), close))
    results.append(run("MAJORS XSMOM lb=90 LO reb=7", xs_momentum(majors, 90, 30, 0.30, False, 7, 0.30), close))
    results.append(run("MAJORS TSMOM lb=60 LO", tsmom(majors, 60, 30, 0.30, True), close))

    results.sort(key=lambda r: (r["oos"]["sortino"] if not np.isnan(r["oos"]["sortino"]) else -9), reverse=True)
    hdr = (f"{'strategy':<42} {'OOS Sort':>9} {'OOS Shrp':>9} {'OOS DD':>7} | "
           f"{'IS Sort':>8} | {'FULL Sort':>9} {'FULL CAGR':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        o, i, f = r["oos"], r["is"], r["full"]
        print(f"{r['label']:<42} {o['sortino']:>9.2f} {o['sharpe']:>9.2f} "
              f"{o['max_dd']:>7.0%} | {i['sortino']:>8.2f} | {f['sortino']:>9.2f} {f['cagr']:>8.0%}")


if __name__ == "__main__":
    main()
