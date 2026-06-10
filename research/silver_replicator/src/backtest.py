"""
Backtest / scoring engine for silver replicator.

Reconstructs the actual book's position path in QI-equivalent contracts,
generates the model's signal path, and scores both with a small bundle of
agreement metrics plus a P&L curve correlation.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from .signal_grammar import SignalGrammar, SignalParams


ART = pathlib.Path(__file__).resolve().parents[1] / "artifacts"

# Contract multipliers (oz)
MULT_QI = 2_500
MULT_SI = 5_000

# Delta proxies for options (rough)
CALL_DELTA = 0.5
PUT_DELTA = -0.5


# --------------------------------------------------------------------- data --


def load_trades(path: str | pathlib.Path | None = None) -> pd.DataFrame:
    p = pathlib.Path(path) if path else ART / "flex_trades.parquet"
    t = pd.read_parquet(p).copy()
    # already tz-naive UTC per the brief; make tz-aware
    if pd.api.types.is_datetime64_any_dtype(t["DateTime"]) and t["DateTime"].dt.tz is None:
        t["DateTime"] = t["DateTime"].dt.tz_localize("UTC")
    t = t.sort_values("DateTime").reset_index(drop=True)
    t["is_option"] = t["Symbol"].str.startswith("SO")
    t["is_call"] = t["Symbol"].str.contains(r"\sC\d", regex=True)
    t["is_put"] = t["Symbol"].str.contains(r"\sP\d", regex=True)
    return t


def load_bars(tf: str) -> pd.DataFrame:
    b = pd.read_parquet(ART / f"si_front_month_{tf}.parquet")
    b = b.set_index("ts").sort_index()
    return b


def load_features(tf: str) -> pd.DataFrame:
    f = pd.read_parquet(ART / f"features_{tf}.parquet").set_index("ts").sort_index()
    bars = load_bars(tf)
    f["c"] = bars["c"]
    return f


# ---------------------------------------------------------- position rebuild --


def _qi_eq_delta(row) -> float:
    """Signed QI-mini-equivalent contract delta for a single fill."""
    qty = row["Quantity"]  # already signed
    if not row["is_option"]:
        # QI vs SI/SIM6 futures: convert SI to QI-eq via mult ratio (2x)
        mult = MULT_QI if row["Symbol"].startswith("QI") else MULT_SI
        return qty * (mult / MULT_QI)
    # Options on SI -> 5000 oz underlying.  Use delta proxy.
    if row["is_call"]:
        d = CALL_DELTA
    elif row["is_put"]:
        d = PUT_DELTA
    else:
        d = 0.0
    return qty * d * (MULT_SI / MULT_QI)


def reconstruct_position_path(trades: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.Series:
    """Net QI-equivalent contract delta per bar (cumulative)."""
    t = trades.copy()
    t["delta_contracts"] = t.apply(_qi_eq_delta, axis=1)
    bar_idx = pd.DatetimeIndex(bar_index).sort_values()
    tz = bar_idx.tz
    ts_vals = pd.DatetimeIndex(t["DateTime"])
    if tz is not None and ts_vals.tz is None:
        ts_vals = ts_vals.tz_localize(tz)
    pos_idx = bar_idx.searchsorted(ts_vals, side="left")
    pos_idx = np.clip(pos_idx, 0, len(bar_idx) - 1)
    t["bar_ts"] = bar_idx[pos_idx]
    by_bar = t.groupby("bar_ts")["delta_contracts"].sum()
    s = by_bar.reindex(bar_idx, fill_value=0.0).cumsum()
    s.name = "actual_position_qi_eq"
    return s


def realised_pnl_curve(trades: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.Series:
    """Cumulative realised P&L from C-side fills, aligned to bar_index."""
    closes = trades[trades["Open/CloseIndicator"] == "C"].copy()
    bar_idx = pd.DatetimeIndex(bar_index).sort_values()
    tz = bar_idx.tz
    ts_vals = pd.DatetimeIndex(closes["DateTime"])
    if tz is not None and ts_vals.tz is None:
        ts_vals = ts_vals.tz_localize(tz)
    pos_idx = bar_idx.searchsorted(ts_vals, side="left")
    pos_idx = np.clip(pos_idx, 0, len(bar_idx) - 1)
    closes["bar_ts"] = bar_idx[pos_idx]
    by_bar = closes.groupby("bar_ts")["FifoPnlRealized"].sum()
    s = by_bar.reindex(bar_idx, fill_value=0.0).cumsum()
    s.name = "actual_cum_pnl"
    return s


# ------------------------------------------------------------- signal layer --


def generate_signal_path(features: pd.DataFrame, params: SignalParams) -> pd.Series:
    sg = SignalGrammar(params)
    s = sg.generate(features)
    s.name = "model_signal"
    return s


# ---------------------------------------------------------------- scoring --


def _restrict_window(
    model: pd.Series,
    actual_pos: pd.Series,
    pnl_curve: pd.Series,
    bars: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    # restrict to the period spanning any nonzero activity in either series
    pos_live = actual_pos.abs() > 1e-9
    pnl_live = pnl_curve.diff().fillna(0).abs() > 1e-9
    live = pos_live | pnl_live
    if live.any():
        first = live[live].index.min()
        last = live[live].index.max()
    else:
        first, last = model.index.min(), model.index.max()
    sel = (model.index >= first) & (model.index <= last)
    return model[sel], actual_pos[sel], pnl_curve[sel], bars.loc[sel]


def score(
    model_signal: pd.Series,
    actual_pos: pd.Series,
    actual_cum_pnl: pd.Series,
    bars: pd.DataFrame,
    contracts_per_unit: int = 4,
) -> Dict[str, float]:
    """
    Return a bundle of scoring metrics for a single model run.

    `bars` is needed for close-to-close returns in the P&L mimicry term.
    """
    model, actual_pos, actual_cum_pnl, bars = _restrict_window(
        model_signal, actual_pos, actual_cum_pnl, bars
    )

    # ---- direction accuracy / kappa ----
    sgn_model = np.sign(model.to_numpy()).astype(int)
    sgn_actual = np.sign(actual_pos.to_numpy()).astype(int)
    if len(sgn_model) == 0:
        return _empty_scores()

    dir_acc = float(np.mean(sgn_model == sgn_actual))
    try:
        kappa = float(cohen_kappa_score(sgn_actual, sgn_model, labels=[-1, 0, 1]))
    except ValueError:
        kappa = 0.0
    if not np.isfinite(kappa):
        kappa = 0.0

    # ---- pearson(scaled model qty, actual pos) ----
    model_qty = model.astype(float) * contracts_per_unit
    if model_qty.std() < 1e-12 or actual_pos.std() < 1e-12:
        pearson_pos = 0.0
    else:
        pearson_pos = float(stats.pearsonr(model_qty, actual_pos)[0])
    if not np.isfinite(pearson_pos):
        pearson_pos = 0.0

    # ---- P&L mimicry ----
    # Walk model signal on close-to-close returns; scale to match actual daily $ vol.
    close = bars["c"].astype(float)
    ret = close.diff().fillna(0.0)  # $/oz move per bar
    # P&L per QI contract per bar = ret * MULT_QI; signal lagged 1 bar to avoid look-ahead
    sig_lag = model.shift(1).fillna(0).astype(float)
    model_pnl_unit = sig_lag * ret * MULT_QI  # $ per 1 QI-eq contract
    actual_daily_pnl = actual_cum_pnl.diff().fillna(0.0)
    target_vol = actual_daily_pnl.std()
    cur_vol = model_pnl_unit.std()
    scale = (target_vol / cur_vol) if cur_vol > 1e-9 else 1.0
    model_pnl = model_pnl_unit * scale
    model_cum = model_pnl.cumsum()

    if model_cum.std() < 1e-9 or actual_cum_pnl.std() < 1e-9:
        pnl_corr = 0.0
    else:
        pnl_corr = float(stats.pearsonr(model_cum, actual_cum_pnl)[0])
    if not np.isfinite(pnl_corr):
        pnl_corr = 0.0

    total_model_pnl = float(model_cum.iloc[-1]) if len(model_cum) else 0.0
    total_actual_pnl = float(actual_cum_pnl.iloc[-1]) if len(actual_cum_pnl) else 0.0

    return dict(
        direction_accuracy=dir_acc,
        cohens_kappa=kappa,
        pearson_pos=pearson_pos,
        pnl_curve_corr=pnl_corr,
        sim_pnl_total=total_model_pnl,
        actual_pnl_total=total_actual_pnl,
        n_bars=int(len(model)),
        sim_scale=float(scale),
    )


def _empty_scores() -> Dict[str, float]:
    return dict(
        direction_accuracy=0.0,
        cohens_kappa=0.0,
        pearson_pos=0.0,
        pnl_curve_corr=0.0,
        sim_pnl_total=0.0,
        actual_pnl_total=0.0,
        n_bars=0,
        sim_scale=1.0,
    )


def composite(scores: Dict[str, float]) -> float:
    return (
        0.5 * scores["pnl_curve_corr"]
        + 0.3 * scores["direction_accuracy"]
        + 0.2 * scores["cohens_kappa"]
    )


if __name__ == "__main__":
    tf = "1D"
    bars = load_bars(tf)
    feats = load_features(tf)
    trades = load_trades()
    pos = reconstruct_position_path(trades, bars.index)
    pnl = realised_pnl_curve(trades, bars.index)
    sig = generate_signal_path(feats, SignalParams())
    sc = score(sig, pos, pnl, bars)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in sc.items()})
    print("composite:", round(composite(sc), 4))
