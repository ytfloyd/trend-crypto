"""
Options overlay v2: replace the generic 50-DTE / 5-delta default with the
empirical distribution observed in the actual SO fills.

Step 1: parse the 54 SO fills, compute per-O-fill DTE / delta / call-put.
Step 2: write empirical medians to artifacts/overlay_empirical_fit.json.
Step 3: re-run the BS overlay using those empirical medians (separate
        params for calls vs puts) on the v3/v2 winning signal.
Step 4: save artifacts/overlay_trades_v2.parquet and figure
        figures/08_overlay_distribution.png.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from src.backtest import (
    ART,
    generate_signal_path,
    load_bars,
    load_features,
    load_trades,
)
from src.signal_grammar import SignalParams


# ---------------- contract / expiry helpers ----------------

# Futures month codes
MONTH_CODE = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# Pattern e.g. "SOH6 C7675" or "SOG7 P10175"
RE_SO = re.compile(r"^SO([FGHJKMNQUVXZ])(\d)\s+([CP])(\d+)$")


def parse_so_symbol(sym: str) -> dict | None:
    """Return dict with expiry_year, expiry_month, side ('C'/'P'), strike."""
    m = RE_SO.match(sym.strip())
    if not m:
        return None
    mc, yc, side, k = m.groups()
    month = MONTH_CODE[mc]
    # 2020s decade
    year = 2020 + int(yc)
    strike = float(k) / 100.0
    return dict(expiry_year=year, expiry_month=month, side=side, strike=strike)


def stub_expiry(year: int, month: int) -> pd.Timestamp:
    """Coarse expiry stub: first of month + 25 days (per task spec)."""
    return pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.Timedelta(days=25)


# ---------------- BS pricing helpers ----------------

def bs_d1(S: float, K: float, T: float, sigma: float, r: float = 0.05) -> float:
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.05,
             is_call: bool = True) -> float:
    if T <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, T, sigma, r)
    return float(norm.cdf(d1)) if is_call else float(norm.cdf(d1) - 1.0)


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.05,
             is_call: bool = True) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic
    d1 = bs_d1(S, K, T, sigma, r)
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_strike_for_delta(S: float, T: float, sigma: float, target_delta: float,
                        r: float = 0.05, is_call: bool = True) -> float:
    if sigma <= 0 or T <= 0:
        return S
    d_abs = min(max(abs(target_delta), 1e-3), 0.99)
    if is_call:
        d1 = norm.ppf(d_abs)
    else:
        d1 = norm.ppf(1.0 - d_abs)
    K = S / np.exp(d1 * sigma * np.sqrt(T) - (r + 0.5 * sigma ** 2) * T)
    return float(K)


# ---------------- realised-vol per timestamp ----------------

def _ann_factor_for_tf(tf: str) -> float:
    return {"1H": 24 * 252, "4H": 6 * 252, "8H": 3 * 252, "1D": 252}[tf]


def _bars_per_day(tf: str) -> float:
    return {"1H": 24, "4H": 6, "8H": 3, "1D": 1}[tf]


def underlying_at(bars: pd.DataFrame, ts: pd.Timestamp) -> Tuple[float, int]:
    """Nearest-prior bar close and its integer index for the given fill time."""
    idx = bars.index
    # searchsorted to find first idx >= ts -> step back 1
    pos = idx.searchsorted(ts, side="right") - 1
    pos = int(np.clip(pos, 0, len(idx) - 1))
    return float(bars["c"].iloc[pos]), pos


def realised_vol_series(bars: pd.DataFrame, tf: str, window: int = 30,
                        iv_mult: float = 1.2) -> pd.Series:
    close = bars["c"].astype(float)
    ann = _ann_factor_for_tf(tf)
    ret = np.log(close / close.shift(1))
    rv = ret.rolling(window, min_periods=10).std() * np.sqrt(ann)
    rv = rv.bfill().ffill().fillna(0.4)
    return rv * iv_mult


# ---------------- empirical fit ----------------

def empirical_fit(trades: pd.DataFrame, bars: pd.DataFrame, tf: str = "8H",
                  vol_window: int = 30, iv_mult: float = 1.2) -> dict:
    iv_series = realised_vol_series(bars, tf, vol_window, iv_mult)

    opt_O = trades[(trades["Symbol"].str.startswith("SO"))
                   & (trades["Open/CloseIndicator"] == "O")].copy()
    rows = []
    for _, r in opt_O.iterrows():
        parsed = parse_so_symbol(r["Symbol"])
        if parsed is None:
            continue
        ts = pd.Timestamp(r["DateTime"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        expiry = stub_expiry(parsed["expiry_year"], parsed["expiry_month"])
        dte_days = (expiry - ts).total_seconds() / 86400.0
        S, bar_idx = underlying_at(bars, ts)
        K = parsed["strike"]
        sigma = float(iv_series.iloc[bar_idx])
        T = max(dte_days / 365.25, 1.0 / 365.25)
        is_call = parsed["side"] == "C"
        delta = bs_delta(S, K, T, sigma, r=0.05, is_call=is_call)
        moneyness = float(np.log(K / S))
        rows.append(dict(
            DateTime=ts,
            Symbol=r["Symbol"],
            side=parsed["side"],
            strike=K,
            expiry_ts=expiry,
            dte_days=dte_days,
            S=S,
            sigma_iv=sigma,
            delta=delta,
            moneyness=moneyness,
            quantity=int(r["Quantity"]),
        ))
    fits = pd.DataFrame(rows)
    n = len(fits)
    n_call = int((fits["side"] == "C").sum())
    n_put = int((fits["side"] == "P").sum())

    calls = fits[fits["side"] == "C"]
    puts = fits[fits["side"] == "P"]

    summary = dict(
        n_open_fills=n,
        n_calls=n_call,
        n_puts=n_put,
        p_call=float(n_call / n) if n else 0.0,
        p_put=float(n_put / n) if n else 0.0,
        median_dte_calls=float(calls["dte_days"].median()) if len(calls) else float("nan"),
        median_dte_puts=float(puts["dte_days"].median()) if len(puts) else float("nan"),
        median_delta_calls=float(calls["delta"].median()) if len(calls) else float("nan"),
        median_delta_puts=float(puts["delta"].median()) if len(puts) else float("nan"),
        median_moneyness_calls=float(calls["moneyness"].median()) if len(calls) else float("nan"),
        median_moneyness_puts=float(puts["moneyness"].median()) if len(puts) else float("nan"),
        iv_assumption="30-bar realised vol on 8H bars * 1.2",
        rate_used=0.05,
        expiry_stub="first-of-contract-month + 25 days",
    )
    return dict(fits_df=fits, summary=summary)


# ---------------- overlay v2 ----------------

@dataclass
class OverlayConfig:
    dte_days_call: float
    dte_days_put: float
    target_delta_call: float   # signed: positive
    target_delta_put: float    # signed: negative magnitude (we use abs internally)
    iv_window: int = 30
    iv_mult: float = 1.2
    r: float = 0.05
    multiplier: int = 5_000


def run_overlay_v2(tf: str, params: SignalParams, cfg: OverlayConfig) -> pd.DataFrame:
    bars = load_bars(tf)
    feats = load_features(tf)
    sig = generate_signal_path(feats, params)
    state = sig.astype(int)
    close = bars["c"].astype(float)
    ann = _ann_factor_for_tf(tf)
    bpd = _bars_per_day(tf)
    rv = realised_vol_series(bars, tf, cfg.iv_window, cfg.iv_mult)
    rv_div_mult = rv / cfg.iv_mult   # restore raw; rv already x mult
    # Actually keep rv as the IV (already includes iv_mult).
    iv = rv

    prev = state.shift(1).fillna(0).astype(int)
    flip_long = (prev <= 0) & (state > 0)
    flip_short = (prev >= 0) & (state < 0)
    flip_flat = (prev != 0) & (state == 0)

    idx = bars.index
    trades: List[dict] = []
    open_pos = None

    def _open(i: int, side: str) -> dict:
        S = float(close.iloc[i])
        sigma = float(iv.iloc[i])
        if side == "C":
            dte_days = cfg.dte_days_call
            tgt = abs(cfg.target_delta_call)
            is_call = True
        else:
            dte_days = cfg.dte_days_put
            tgt = abs(cfg.target_delta_put)
            is_call = False
        dte_bars = int(round(dte_days * bpd))
        T = dte_bars / ann
        K = bs_strike_for_delta(S, T, sigma, tgt, cfg.r, is_call)
        K = round(K * 4) / 4.0
        prem = bs_price(S, K, T, sigma, cfg.r, is_call)
        exp_i = min(i + dte_bars, len(idx) - 1)
        return dict(
            entry_ts=idx[i], expiry_ts=idx[exp_i],
            entry_idx=i, expiry_idx=exp_i,
            S0=S, sigma=sigma, K=K, side=side,
            premium_in=prem, dte_days_in=dte_days,
            target_delta=tgt,
        )

    def _close(pos: dict, i: int, reason: str) -> dict:
        S = float(close.iloc[i])
        bars_left = max(pos["expiry_idx"] - i, 0)
        T = bars_left / ann
        sigma = float(iv.iloc[i])
        is_call = pos["side"] == "C"
        prem_out = bs_price(S, pos["K"], T, sigma, cfg.r, is_call)
        pnl = (prem_out - pos["premium_in"]) * cfg.multiplier
        return dict(
            entry_ts=pos["entry_ts"], exit_ts=idx[i], expiry_ts=pos["expiry_ts"],
            side=pos["side"], K=pos["K"], S0=pos["S0"], S_exit=S,
            sigma_in=pos["sigma"], sigma_out=sigma,
            premium_in=pos["premium_in"], premium_out=prem_out, pnl=pnl,
            dte_days_in=pos["dte_days_in"], target_delta=pos["target_delta"],
            exit_reason=reason,
        )

    for i in range(len(state)):
        if open_pos is not None and i >= open_pos["expiry_idx"]:
            trades.append(_close(open_pos, i, "expiry"))
            open_pos = None
        fl, fs, ff = bool(flip_long.iloc[i]), bool(flip_short.iloc[i]), bool(flip_flat.iloc[i])
        if open_pos is not None:
            if ff or (open_pos["side"] == "C" and fs) or (open_pos["side"] == "P" and fl):
                trades.append(_close(open_pos, i, "flip"))
                open_pos = None
        if open_pos is None:
            if fl:
                open_pos = _open(i, "C")
            elif fs:
                open_pos = _open(i, "P")

    if open_pos is not None:
        trades.append(_close(open_pos, len(state) - 1, "end_of_data"))

    return pd.DataFrame(trades)


# ---------------- main ----------------

def main() -> int:
    tf = "8H"
    bars = load_bars(tf)
    trades = load_trades()

    # === step 1+2: empirical fit ===
    fit = empirical_fit(trades, bars, tf=tf, vol_window=30, iv_mult=1.2)
    print("Empirical fit summary:")
    print(json.dumps(fit["summary"], indent=2, default=str))
    out_emp = ART / "overlay_empirical_fit.json"
    with open(out_emp, "w") as fh:
        json.dump(fit["summary"], fh, indent=2, default=str)
    print(f"wrote {out_emp}")

    # Save the per-fill table too (handy for the figure & for review)
    fits_df = fit["fits_df"]
    fits_df.to_parquet(ART / "overlay_empirical_fills.parquet", index=False)

    # === step 3: rebuild overlay using empirical medians ===
    # Use the v3/v2 winning signal params (they're identical at the top by tie).
    with open(ART / "best_params_v2.json") as fh:
        best = json.load(fh)
    sig_tf = best["tf"]
    sig_params = SignalParams(**best["params"])

    s = fit["summary"]
    cfg = OverlayConfig(
        dte_days_call=float(s["median_dte_calls"]),
        dte_days_put=float(s["median_dte_puts"]),
        target_delta_call=float(s["median_delta_calls"]),
        target_delta_put=float(s["median_delta_puts"]),
    )
    print(f"\noverlay config: dte_C={cfg.dte_days_call:.2f}d, dte_P={cfg.dte_days_put:.2f}d, "
          f"delta_C={cfg.target_delta_call:.3f}, delta_P={cfg.target_delta_put:.3f}")

    df = run_overlay_v2(sig_tf, sig_params, cfg)
    out_path = ART / "overlay_trades_v2.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} (n={len(df)})")

    pnl_total = float(df["pnl"].sum()) if len(df) else 0.0
    print(f"overlay v2 total pnl: ${pnl_total:,.2f}")

    # v1 reference
    v1 = pd.read_parquet(ART / "overlay_trades.parquet")
    v1_pnl = float(v1["pnl"].sum())
    print(f"overlay v1 total pnl: ${v1_pnl:,.2f}")
    actual_so_pnl = float(trades[trades["Symbol"].str.startswith("SO")]["FifoPnlRealized"].sum())
    print(f"actual SO realised pnl: ${actual_so_pnl:,.2f}")

    # === step 4: figure ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- panel 1: empirical fills on (DTE, delta) ----
    ax = axes[0]
    em_call = fits_df[fits_df["side"] == "C"]
    em_put = fits_df[fits_df["side"] == "P"]
    ax.scatter(em_call["dte_days"], em_call["delta"], s=80, alpha=0.7,
               color="#1f77b4", label=f"actual C (n={len(em_call)})", edgecolor="black")
    ax.scatter(em_put["dte_days"], em_put["delta"], s=80, alpha=0.7,
               color="#d62728", label=f"actual P (n={len(em_put)})", edgecolor="black")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("DTE (calendar days)")
    ax.set_ylabel("BS delta (signed)")
    ax.set_title("Actual SO O-fills: delta vs DTE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- panel 2: synthetic v2 overlay fills on same axes ----
    ax2 = axes[1]
    if len(df):
        # Recompute the delta at entry for each synthetic fill so we can plot it.
        ann = _ann_factor_for_tf(sig_tf)
        bpd = _bars_per_day(sig_tf)
        syn_rows = []
        for _, r in df.iterrows():
            S = r["S0"]; K = r["K"]; sigma = r["sigma_in"]
            dte_days = r["dte_days_in"]
            T = (dte_days * bpd) / ann
            d = bs_delta(S, K, T, sigma, r=0.05, is_call=(r["side"] == "C"))
            syn_rows.append(dict(side=r["side"], dte_days_in=dte_days, delta=d))
        syn = pd.DataFrame(syn_rows)
        syn_c = syn[syn["side"] == "C"]
        syn_p = syn[syn["side"] == "P"]
        ax2.scatter(syn_c["dte_days_in"], syn_c["delta"], s=60, alpha=0.6,
                    color="#1f77b4", marker="s",
                    label=f"v2 synth C (n={len(syn_c)})", edgecolor="black")
        ax2.scatter(syn_p["dte_days_in"], syn_p["delta"], s=60, alpha=0.6,
                    color="#d62728", marker="s",
                    label=f"v2 synth P (n={len(syn_p)})", edgecolor="black")
        # overlay empirical too as small dots
        ax2.scatter(em_call["dte_days"], em_call["delta"], s=25, alpha=0.4,
                    color="#1f77b4", label="actual C", edgecolor="none")
        ax2.scatter(em_put["dte_days"], em_put["delta"], s=25, alpha=0.4,
                    color="#d62728", label="actual P", edgecolor="none")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_xlabel("DTE (calendar days)")
    ax2.set_ylabel("BS delta (signed)")
    ax2.set_title("Synthetic v2 overlay fills vs actual")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Options overlay v2: empirical fit vs synthetic  "
                 f"(v1=${v1_pnl:,.0f}, v2=${pnl_total:,.0f}, actual=${actual_so_pnl:,.0f})")
    fig.tight_layout()
    fig_path = pathlib.Path(PKG) / "figures" / "08_overlay_distribution.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
