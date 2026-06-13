"""Medallion Lite — FACTOR-COUNT experiment: 5-factor baseline vs a 100-factor
TA-Lib factor zoo (and the two combined).

Question: does expanding the cross-sectional composite from the 5 hand-chosen
factors to ~100 TA-Lib indicators improve, or dilute, out-of-sample performance?

Method (kept honest to isolate the factor-count effect):
  * Same point-in-time top-100 universe, survivorship-free, within-universe ranking.
  * Same param-frozen walk-forward (entry x trailing grid; select-on-train / freeze /
    score-on-test) and 30 bps costs as the validated 2.95 baseline.
  * Composite = EQUAL-WEIGHT mean of cross-sectional percentile ranks. Factor weights
    are NOT fit on returns (that would be snooping); each factor is oriented by economic
    CONVENTION (a fixed +/- prior, not a fitted IC sign). This is the fair test of
    "shovel in more indicators" — the walk-forward reveals signal vs noise (QF-21).

Run: PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/run_medallion_factors.py
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
import talib as T  # noqa: E402

import run_medallion_universe as uni  # noqa: E402
import run_medallion_walkforward as wf  # noqa: E402
from medallion_lite.factors import compute_factors  # noqa: E402
from medallion_lite.portfolio import backtest_portfolio, build_factor_portfolio  # noqa: E402
from medallion_lite.regime_ensemble import compute_ensemble_regime  # noqa: E402
from sornette_lppl.data_hf import filter_universe_hourly  # noqa: E402

LAKE = str(ROOT.parent / "data" / "coinbase_crypto_ohlcv_lake.duckdb")
START, END = "2021-01-01", "2026-06-01"
OOS_START = "2023-01-01"


# ----------------------------------------------------------------------
# 100-factor TA-Lib zoo: (label, fn(o,h,l,c,v)->1d array, sign).  sign is an
# economic-convention prior (higher rank = more attractive for a trend/momentum
# cross-section), NOT fit on returns.
# ----------------------------------------------------------------------
def factor_specs():
    S = []

    def add(label, fn, sign=1):
        S.append((label, fn, sign))

    for p in (7, 14, 21, 28):
        add(f"RSI{p}", lambda o, h, l, c, v, p=p: T.RSI(c, p))
    for p in (10, 21, 63):
        add(f"MOM{p}", lambda o, h, l, c, v, p=p: T.MOM(c, p))
    for p in (5, 10, 21, 63):
        add(f"ROC{p}", lambda o, h, l, c, v, p=p: T.ROC(c, p))
    for p in (14, 28):
        add(f"CMO{p}", lambda o, h, l, c, v, p=p: T.CMO(c, p))
    for p in (14, 28):
        add(f"CCI{p}", lambda o, h, l, c, v, p=p: T.CCI(h, l, c, p))
    for p in (14, 28):
        add(f"WILLR{p}", lambda o, h, l, c, v, p=p: T.WILLR(h, l, c, p))
    for p in (14, 28):
        add(f"ADX{p}", lambda o, h, l, c, v, p=p: T.ADX(h, l, c, p))
    add("ADXR14", lambda o, h, l, c, v: T.ADXR(h, l, c, 14))
    add("DX14", lambda o, h, l, c, v: T.DX(h, l, c, 14))
    add("PLUS_DI14", lambda o, h, l, c, v: T.PLUS_DI(h, l, c, 14))
    add("MINUS_DI14", lambda o, h, l, c, v: T.MINUS_DI(h, l, c, 14), -1)
    add("PLUS_DM14", lambda o, h, l, c, v: T.PLUS_DM(h, l, 14))
    add("MINUS_DM14", lambda o, h, l, c, v: T.MINUS_DM(h, l, 14), -1)
    for p in (14, 25):
        add(f"AROONOSC{p}", lambda o, h, l, c, v, p=p: T.AROONOSC(h, l, p))
    add("APO", lambda o, h, l, c, v: T.APO(c, 12, 26))
    add("PPO", lambda o, h, l, c, v: T.PPO(c, 12, 26))
    add("MACD", lambda o, h, l, c, v: T.MACD(c, 12, 26, 9)[0])
    add("MACDhist", lambda o, h, l, c, v: T.MACD(c, 12, 26, 9)[2])
    for p in (15, 30):
        add(f"TRIX{p}", lambda o, h, l, c, v, p=p: T.TRIX(c, p))
    add("ULTOSC", lambda o, h, l, c, v: T.ULTOSC(h, l, c))
    add("MFI14", lambda o, h, l, c, v: T.MFI(h, l, c, v, 14))
    add("BOP", lambda o, h, l, c, v: T.BOP(o, h, l, c))
    add("STOCH_K", lambda o, h, l, c, v: T.STOCH(h, l, c)[0])
    add("STOCH_D", lambda o, h, l, c, v: T.STOCH(h, l, c)[1])
    add("STOCHF_K", lambda o, h, l, c, v: T.STOCHF(h, l, c)[0])
    add("STOCHRSI_K", lambda o, h, l, c, v: T.STOCHRSI(c, 14)[0])
    add("ROCR100_10", lambda o, h, l, c, v: T.ROCR100(c, 10))
    # overlap studies as close/MA - 1 (scale-free, cross-sectionally comparable)
    for p in (10, 20, 50, 100, 200):
        add(f"cSMA{p}", lambda o, h, l, c, v, p=p: c / T.SMA(c, p) - 1.0)
    for p in (10, 20, 50, 100):
        add(f"cEMA{p}", lambda o, h, l, c, v, p=p: c / T.EMA(c, p) - 1.0)
    add("cWMA20", lambda o, h, l, c, v: c / T.WMA(c, 20) - 1.0)
    add("cDEMA20", lambda o, h, l, c, v: c / T.DEMA(c, 20) - 1.0)
    add("cTEMA20", lambda o, h, l, c, v: c / T.TEMA(c, 20) - 1.0)
    add("cKAMA20", lambda o, h, l, c, v: c / T.KAMA(c, 20) - 1.0)
    add("cT3_20", lambda o, h, l, c, v: c / T.T3(c, 20) - 1.0)
    add("cTRIMA20", lambda o, h, l, c, v: c / T.TRIMA(c, 20) - 1.0)
    add("cMA30", lambda o, h, l, c, v: c / T.MA(c, 30) - 1.0)
    add("BB_pctB20", lambda o, h, l, c, v: (
        (c - T.BBANDS(c, 20)[2]) / (T.BBANDS(c, 20)[0] - T.BBANDS(c, 20)[2])))
    add("cMIDPOINT14", lambda o, h, l, c, v: c / T.MIDPOINT(c, 14) - 1.0)
    add("cMIDPRICE14", lambda o, h, l, c, v: c / T.MIDPRICE(h, l, 14) - 1.0)
    add("cSAR", lambda o, h, l, c, v: c / T.SAR(h, l) - 1.0)
    add("cHT_TREND", lambda o, h, l, c, v: c / T.HT_TRENDLINE(c) - 1.0)
    # volatility (medallion treats higher realised vol as trend-capture opportunity: +)
    for p in (14, 21):
        add(f"ATR{p}c", lambda o, h, l, c, v, p=p: T.ATR(h, l, c, p) / c)
    for p in (14, 21):
        add(f"NATR{p}", lambda o, h, l, c, v, p=p: T.NATR(h, l, c, p))
    add("TRANGEc", lambda o, h, l, c, v: T.TRANGE(h, l, c) / c)
    for p in (10, 21, 63):
        add(f"STDDEV{p}c", lambda o, h, l, c, v, p=p: T.STDDEV(c, p) / c)
    add("VAR20c", lambda o, h, l, c, v: T.VAR(c, 20) / (c * c))
    # volume
    add("OBV_roc21", lambda o, h, l, c, v: T.ROC(T.OBV(c, v), 21))
    add("AD_roc21", lambda o, h, l, c, v: T.ROC(T.AD(h, l, c, v), 21))
    add("ADOSC", lambda o, h, l, c, v: T.ADOSC(h, l, c, v))
    for p in (5, 21):
        add(f"VOLroc{p}", lambda o, h, l, c, v, p=p: T.ROC(v, p))
    # statistics
    for p in (14, 63):
        add(f"LRSLOPE{p}", lambda o, h, l, c, v, p=p: T.LINEARREG_SLOPE(c, p) / c)
    add("LRANGLE14", lambda o, h, l, c, v: T.LINEARREG_ANGLE(c, 14))
    add("TSF14c", lambda o, h, l, c, v: c / T.TSF(c, 14) - 1.0)
    add("BETA20", lambda o, h, l, c, v: T.BETA(h, l, 20))
    add("CORREL20", lambda o, h, l, c, v: T.CORREL(h, l, 20))
    add("LR14c", lambda o, h, l, c, v: c / T.LINEARREG(c, 14) - 1.0)
    # cycle (Hilbert transform)
    add("HT_DCPERIOD", lambda o, h, l, c, v: T.HT_DCPERIOD(c))
    add("HT_DCPHASE", lambda o, h, l, c, v: T.HT_DCPHASE(c))
    add("HT_TRENDMODE", lambda o, h, l, c, v: T.HT_TRENDMODE(c).astype(float))
    add("HT_SINE", lambda o, h, l, c, v: T.HT_SINE(c)[0])
    add("HT_PHASOR", lambda o, h, l, c, v: T.HT_PHASOR(c)[0])
    # extra parameterizations to reach 100 (deliberately collinear with the above —
    # part of the point: more indicators != more independent information)
    for p in (10, 50):
        add(f"RSI{p}b", lambda o, h, l, c, v, p=p: T.RSI(c, p))
    for p in (3, 126):
        add(f"ROC{p}b", lambda o, h, l, c, v, p=p: T.ROC(c, p))
    for p in (5, 126):
        add(f"MOM{p}b", lambda o, h, l, c, v, p=p: T.MOM(c, p))
    add("CCI50", lambda o, h, l, c, v: T.CCI(h, l, c, 50))
    add("WILLR50", lambda o, h, l, c, v: T.WILLR(h, l, c, 50))
    add("AROON_up14", lambda o, h, l, c, v: T.AROON(h, l, 14)[1])
    add("AROON_dn14", lambda o, h, l, c, v: T.AROON(h, l, 14)[0], -1)
    add("MACD_signal", lambda o, h, l, c, v: T.MACD(c, 12, 26, 9)[1])
    add("PPO_21_55", lambda o, h, l, c, v: T.PPO(c, 21, 55))
    add("TRIX9", lambda o, h, l, c, v: T.TRIX(c, 9))
    add("cEMA200", lambda o, h, l, c, v: c / T.EMA(c, 200) - 1.0)
    add("cSMA150", lambda o, h, l, c, v: c / T.SMA(c, 150) - 1.0)
    add("LRSLOPE126", lambda o, h, l, c, v: T.LINEARREG_SLOPE(c, 126) / c)
    add("STDDEV126c", lambda o, h, l, c, v: T.STDDEV(c, 126) / c)
    add("ROCP21", lambda o, h, l, c, v: T.ROCP(c, 21))
    add("NATR28", lambda o, h, l, c, v: T.NATR(h, l, c, 28))
    assert len(S) >= 100, f"only {len(S)} factors"
    return S[:100]


# ----------------------------------------------------------------------
def load_panel():
    """Load OHLCV hourly for symbols EVER in the point-in-time top-100; build wide
    frames, 5-factor composite inputs, regime, daily ADV table, and per-symbol arrays."""
    con = duckdb.connect(LAKE, read_only=True)
    dd = con.execute(
        """SELECT symbol, ts, close, volume FROM bars_1d_usd_universe_clean
           WHERE ts>=? AND ts<=? AND close>0 ORDER BY symbol, ts""", [START, END]).fetch_df()
    dd["ts"] = pd.to_datetime(dd["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    dd = dd.sort_values(["symbol", "ts"])
    dd["dv"] = dd["close"] * dd["volume"]
    dd["adv20"] = dd.groupby("symbol")["dv"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    dd["rank"] = dd.groupby("ts")["adv20"].rank(ascending=False, method="first")
    ever = sorted(set(dd.loc[(dd["adv20"] > 0) & (dd["rank"] <= 100), "symbol"]) | {"BTC-USD"})
    ph = ", ".join(["?"] * len(ever))
    hf = con.execute(
        f"""SELECT symbol, ts, open, high, low, close, volume FROM bars_1h
            WHERE ts>=? AND ts<=? AND symbol IN ({ph}) AND open>0 AND close>0 AND high>=low
            ORDER BY symbol, ts""", [START, END] + ever).fetch_df()
    con.close()
    print(f"  symbols ever in top-100: {len(ever)}   hourly rows: {len(hf):,}")
    hf["ts"] = pd.to_datetime(hf["ts"], utc=True).dt.tz_localize(None)
    hf = filter_universe_hourly(hf)
    hf = hf[hf["in_universe"]].copy().sort_values(["symbol", "ts"])
    hf["ret"] = hf.groupby("symbol")["close"].pct_change(fill_method=None)
    rw = hf.pivot(index="ts", columns="symbol", values="ret").sort_index().fillna(0)
    cw = hf.pivot(index="ts", columns="symbol", values="close").sort_index()
    hw = hf.pivot(index="ts", columns="symbol", values="high").sort_index()
    vw = hf.pivot(index="ts", columns="symbol", values="volume").sort_index().fillna(0)
    btc_h = hf[hf["symbol"] == "BTC-USD"].set_index("ts")["close"].sort_index()
    regime = compute_ensemble_regime(btc_h.resample("D").last().dropna(), btc_h, rw)
    factors5 = compute_factors(cw, vw, hw)
    # per-symbol contiguous OHLCV arrays for TA-Lib
    sym_arr = {}
    for sym, g in hf.groupby("symbol"):
        sym_arr[sym] = (g["open"].to_numpy("float64"), g["high"].to_numpy("float64"),
                        g["low"].to_numpy("float64"), g["close"].to_numpy("float64"),
                        g["volume"].to_numpy("float64"), pd.DatetimeIndex(g["ts"].values))
    return dd, rw, regime, factors5, cw, sym_arr


def value_frame(fn, sym_arr, full_index):
    cols = {}
    for sym, (o, h, l, c, v, idx) in sym_arr.items():
        try:
            out = fn(o, h, l, c, v)
        except Exception:
            continue
        if out is None or np.all(~np.isfinite(out)):
            continue
        cols[sym] = pd.Series(out, index=idx)
    df = pd.DataFrame(cols).reindex(index=full_index)
    return df.replace([np.inf, -np.inf], np.nan)


def composite_talib(specs, sym_arr, full_index, elig_h):
    """Equal-weight mean of within-universe, convention-oriented percentile ranks."""
    cols = elig_h.columns
    ssum = pd.DataFrame(0.0, index=full_index, columns=cols)
    cnt = pd.DataFrame(0.0, index=full_index, columns=cols)
    for i, (label, fn, sign) in enumerate(specs, 1):
        v = value_frame(fn, sym_arr, full_index).reindex(columns=cols)
        r = v.where(elig_h).rank(axis=1, pct=True)
        if sign < 0:
            r = 1.0 - r
        ssum = ssum.add(r.fillna(0.0))
        cnt = cnt.add(r.notna().astype(float))
        if i % 25 == 0:
            print(f"    ... {i}/{len(specs)} factors")
    comp = (ssum / cnt.replace(0.0, np.nan)).where(elig_h)
    return comp


def wf_from_composite(comp_pit, rw, regime) -> pd.Series:
    runs = {(p["entry_threshold"], p["trailing_stop_pct"]):
            uni._config_daily(comp_pit, rw, regime, p)[0] for p in wf.GRID}
    segs = []
    for tr_lo, tr_hi, te_lo, te_hi in wf.FOLDS:
        best, bk = -9.9, None
        for k, d in runs.items():
            s = wf._sortino(d, tr_lo, tr_hi)["sortino"]
            if not np.isnan(s) and s > best:
                best, bk = s, k
        sel = runs[bk]
        segs.append(sel[(sel.index >= te_lo) & (sel.index <= te_hi)])
    return pd.concat(segs).sort_index()


def _disp(comp) -> float:
    return float(comp.std(axis=1).mean())  # mean cross-sectional dispersion


def avg_holdings(comp_pit, rw, regime, prm=None) -> float:
    prm = prm or {"entry_threshold": 0.65, "trailing_stop_pct": 0.15}
    w, _ = build_factor_portfolio(
        comp_pit, rw, regime, entry_threshold=prm["entry_threshold"],
        exit_score_threshold=0.40, max_hold_hours=336, trailing_stop_pct=prm["trailing_stop_pct"],
        rebalance_every_hours=24, max_positions=25, max_weight=0.10)
    bt = backtest_portfolio(w, rw, tc_bps=30.0)
    if "n_holdings" in bt:
        nh = pd.Series(bt["n_holdings"]).replace(0, np.nan)
        return float(nh.mean())
    return float("nan")


def rerank(comp, elig_h):
    """Force uniform cross-sectional dispersion so the entry threshold has equal
    selectivity regardless of factor count (isolates concentration artifacts)."""
    return comp.rank(axis=1, pct=True).where(elig_h)


def report(name, comp_pit, rw, regime, n_factors):
    oos = wf_from_composite(comp_pit, rw, regime)
    m, mvt = uni._metrics(oos), uni._metrics(wf.vol_target(oos))
    nh = avg_holdings(comp_pit, rw, regime)
    print(f"  {name:<26} n={n_factors:<4} disp={_disp(comp_pit):.3f} hold~{nh:4.1f} | "
          f"Sortino {m['sortino']:>5.2f}  Sharpe {m['sharpe']:>4.2f}  "
          f"CAGR {m['cagr']:>6.0%}  DD {m['max_dd']:>5.0%}  (+vt {mvt['sortino']:.2f})")
    return m


def main() -> None:
    print("loading panel (symbols ever in top-100) ...")
    dd, rw, regime, factors5, cw, sym_arr = load_panel()
    full_index = cw.index
    elig_h = uni.eligibility(dd, None, "top", 100, cw.columns, full_index)
    specs = factor_specs()
    print(f"  TA-Lib factor zoo: {len(specs)} factors")

    print("building composites ...")
    comp5 = uni.composite_within(factors5, elig_h)
    comp100 = composite_talib(specs, sym_arr, full_index, elig_h)
    comp_both = ((comp5 + comp100) / 2.0).where(elig_h)

    print("\n=== AS-IS composites (equal-weight mean of ranks) ===")
    print("  [low disp => scores compress to the mean => few names clear the entry "
          "threshold => concentrated book => unstable, inflated OOS]")
    report("5-factor (baseline)", comp5, rw, regime, 5)
    report("100-factor TA-Lib zoo", comp100, rw, regime, len(specs))
    report("5 + 100 combined (blend)", comp_both, rw, regime, 5 + len(specs))

    print("\n=== RE-RANKED composites (uniform dispersion = EQUAL selectivity) ===")
    print("  [apples-to-apples: same #names selected at a given threshold for every "
          "composite, so any difference is real signal, not a concentration artifact]")
    report("5-factor [re-ranked]", rerank(comp5, elig_h), rw, regime, 5)
    report("100-factor [re-ranked]", rerank(comp100, elig_h), rw, regime, len(specs))
    report("5 + 100 combined [re-ranked]", rerank(comp_both, elig_h), rw, regime, 5 + len(specs))

    btc = wf._daily(rw["BTC-USD"])
    bo = uni._metrics(btc[btc.index >= OOS_START])
    print(f"\n  {'BTC buy&hold':<26} {'':<20} | "
          f"Sortino {bo['sortino']:>5.2f}  Sharpe {bo['sharpe']:>4.2f}  CAGR {bo['cagr']:>6.0%}  DD {bo['max_dd']:>5.0%}")
    print("\nNote: equal-weight, convention-oriented ranks (no return-fitted weights/signs). "
          "'disp'=mean cross-sectional dispersion; 'hold'=avg #positions (flagship params). 30bps, OOS 2023+.")


if __name__ == "__main__":
    main()
