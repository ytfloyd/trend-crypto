"""
Synthetic options overlay on top of the best directional signal.

On a flip from <=0 to +1 we "buy" a 45-60 DTE 5-delta-OTM call valued
via Black-Scholes with rolling 30-bar realised vol (annualised, x1.2).
On a flip back to <=0 we close it.  Symmetric treatment for short flips
(buy a put).
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm

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


# ---------------------------------------------------- BS pricing helpers --


SQRT_252 = np.sqrt(252.0)


def _ann_factor_for_tf(tf: str) -> float:
    """Bars per year, used to annualise close-to-close vol."""
    return {
        "1H": 24 * 252,
        "4H": 6 * 252,
        "8H": 3 * 252,
        "1D": 252,
    }[tf]


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.04,
             is_call: bool = True) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-r * T) * norm.cdf(-d1)


def bs_strike_for_delta(S: float, T: float, sigma: float, target_delta: float,
                        r: float = 0.04, is_call: bool = True) -> float:
    """Closed-form strike that gives the target absolute delta (no q)."""
    if sigma <= 0 or T <= 0:
        return S
    delta_abs = min(max(abs(target_delta), 1e-3), 0.99)
    if is_call:
        # d1 = N^-1(delta) -> K
        d1 = norm.ppf(delta_abs)
    else:
        d1 = norm.ppf(1.0 - delta_abs)
    K = S / np.exp(d1 * sigma * np.sqrt(T) - (r + 0.5 * sigma ** 2) * T)
    return float(K)


# ------------------------------------------------------------- overlay --


@dataclass
class OverlayConfig:
    dte_bars: int            # bars per option contract life
    iv_window: int = 30      # bars of close-to-close vol
    iv_mult: float = 1.2     # IV proxy multiplier on realised vol
    target_delta: float = 0.05  # "5-delta OTM"
    r: float = 0.04
    multiplier: int = 5_000  # SI / SO point multiplier (oz)


def _bars_per_day(tf: str) -> float:
    return {"1H": 24, "4H": 6, "8H": 3, "1D": 1}[tf]


def run_overlay(tf: str, params: SignalParams, dte_days: int = 50) -> pd.DataFrame:
    bars = load_bars(tf)
    feats = load_features(tf)
    sig = generate_signal_path(feats, params)
    state = sig.astype(int)

    close = bars["c"].astype(float)
    ann = _ann_factor_for_tf(tf)
    ret = np.log(close / close.shift(1))
    rv = ret.rolling(30, min_periods=10).std() * np.sqrt(ann)
    rv = rv.bfill().ffill().fillna(0.4)

    bpd = _bars_per_day(tf)
    dte_bars = int(round(dte_days * bpd))
    cfg = OverlayConfig(dte_bars=dte_bars)

    # transitions: open on flip 0/-1 -> +1; or 0/+1 -> -1
    prev = state.shift(1).fillna(0).astype(int)
    flip_long = (prev <= 0) & (state > 0)
    flip_short = (prev >= 0) & (state < 0)
    flip_flat = (prev != 0) & (state == 0)

    trades: List[dict] = []
    open_pos = None  # dict tracking the live synthetic option

    idx = bars.index

    def _open_option(i: int, side: str) -> dict:
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i]) * cfg.iv_mult
        T = cfg.dte_bars / ann
        is_call = side == "C"
        K = bs_strike_for_delta(S, T, sigma, cfg.target_delta, cfg.r, is_call)
        # Round strike to nearest $0.25 like real silver options
        K = round(K * 4) / 4.0
        premium = bs_price(S, K, T, sigma, cfg.r, is_call)
        expiry_idx = min(i + cfg.dte_bars, len(idx) - 1)
        return dict(
            entry_ts=idx[i],
            expiry_ts=idx[expiry_idx],
            entry_idx=i,
            expiry_idx=expiry_idx,
            S0=S,
            sigma=sigma,
            K=K,
            side=side,
            premium_in=premium,
        )

    def _close_option(pos: dict, i: int, reason: str) -> dict:
        S = float(close.iloc[i])
        # rebuild T from remaining bars
        bars_left = max(pos["expiry_idx"] - i, 0)
        T = bars_left / ann
        sigma = float(rv.iloc[i]) * cfg.iv_mult
        is_call = pos["side"] == "C"
        premium_out = bs_price(S, pos["K"], T, sigma, cfg.r, is_call)
        pnl = (premium_out - pos["premium_in"]) * cfg.multiplier
        return dict(
            entry_ts=pos["entry_ts"],
            exit_ts=idx[i],
            expiry_ts=pos["expiry_ts"],
            side=pos["side"],
            K=pos["K"],
            S0=pos["S0"],
            S_exit=S,
            sigma_in=pos["sigma"],
            sigma_out=sigma,
            premium_in=pos["premium_in"],
            premium_out=premium_out,
            pnl=pnl,
            exit_reason=reason,
        )

    for i in range(len(state)):
        # auto-close if expiry hit
        if open_pos is not None and i >= open_pos["expiry_idx"]:
            trades.append(_close_option(open_pos, i, "expiry"))
            open_pos = None

        flip_l = bool(flip_long.iloc[i])
        flip_s = bool(flip_short.iloc[i])
        flat = bool(flip_flat.iloc[i])

        if open_pos is not None:
            # close on flat or opposite flip
            if flat or (open_pos["side"] == "C" and flip_s) or (open_pos["side"] == "P" and flip_l):
                trades.append(_close_option(open_pos, i, "flip"))
                open_pos = None

        if open_pos is None:
            if flip_l:
                open_pos = _open_option(i, "C")
            elif flip_s:
                open_pos = _open_option(i, "P")

    # close any dangling position at last bar
    if open_pos is not None:
        trades.append(_close_option(open_pos, len(state) - 1, "end_of_data"))

    df = pd.DataFrame(trades)
    return df


def main() -> int:
    with open(ART / "best_params.json") as fh:
        best = json.load(fh)
    tf = best["tf"]
    params = SignalParams(**best["params"])
    print(f"Building overlay for tf={tf}, params={params}")
    df = run_overlay(tf, params, dte_days=50)
    if df.empty:
        print("No overlay trades generated.")
    else:
        df.to_parquet(ART / "overlay_trades.parquet", index=False)
        print(f"wrote {len(df)} overlay trades -> {ART/'overlay_trades.parquet'}")
        print(df[["entry_ts", "exit_ts", "side", "K", "S0", "S_exit",
                  "premium_in", "premium_out", "pnl", "exit_reason"]]
              .head(15).to_string(index=False))
        print(f"\noverlay total pnl: ${df['pnl'].sum():,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
