"""Phase 1 — generate stop-aware R-multiple labels across the universe and characterise the
UNCONDITIONAL trade-outcome distribution (the base rate).

This applies the default stop+trail to EVERY eligible (symbol, signal_date) — i.e. "enter long on
every name every day" — with no selection signal. It is the floor any signal must beat, and a
sanity check that the labeler behaves sensibly on real data. Costs are charged in R-terms at a
round-trip assumption. Trades overlap in time (serially correlated) — fine for a per-trade base
rate; the portfolio phase will space/dedupe.

Run: PYTHONPATH=scripts/research/spot_convexity python scripts/research/spot_convexity/make_labels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_daily_panel  # noqa: E402
from features import atr_wilder  # noqa: E402
from labeler import label_trade_with_peak  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts" / "spot_convexity"
STOP_MULT, TRAIL_MULT, MAX_HORIZON, ATR_N = 2.0, 3.0, 60, 14
MIN_HISTORY = 60            # bars before a signal is eligible (ATR + context)
COST_BPS_RT = 30.0         # round-trip cost assumption (R-terms), medallion-consistent
TOP_N = 100


def generate() -> pd.DataFrame:
    panel = load_daily_panel(top_n=TOP_N)
    rows = []
    n_incomplete = 0
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < MIN_HISTORY + 2:
            continue
        o = g["open"].to_numpy("float64")
        h = g["high"].to_numpy("float64")
        lo = g["low"].to_numpy("float64")
        c = g["close"].to_numpy("float64")
        atr = atr_wilder(g["high"], g["low"], g["close"], ATR_N).to_numpy("float64")
        in_u = g["in_universe"].to_numpy()
        ts = g["ts"].to_numpy()
        for i in range(MIN_HISTORY, len(g) - 1):
            if not in_u[i] or not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            r = label_trade_with_peak(o, h, lo, c, signal_idx=i, atr_ref=atr[i],
                                      stop_mult=STOP_MULT, trail_mult=TRAIL_MULT, max_horizon=MAX_HORIZON)
            if not r["valid"]:
                n_incomplete += 1
                continue
            cost_R = (COST_BPS_RT / 1e4) * r["entry_price"] / r["risk_per_unit"]
            rows.append({
                "symbol": sym, "signal_date": ts[i],
                "r_gross": r["r_multiple"], "r_net": r["r_multiple"] - cost_R,
                "exit_reason": r["exit_reason"], "mfe_R": r["mfe_R"], "mae_R": r["mae_R"],
                "reached_1R": r["reached_1R"], "reached_2R": r["reached_2R"],
                "stopped_before_1R": r["stopped_before_1R"], "stop_out_loss": r["stop_out_loss"],
                "bars_held": r["bars_held"], "time_to_stop": r["time_to_stop"],
                "time_to_peak": r["time_to_peak"], "atr_pct": atr[i] / c[i],
                "realized_below_minus1R": r["r_multiple"] < -1.0,
            })
    trades = pd.DataFrame(rows)
    trades.attrs["n_incomplete"] = n_incomplete
    return trades


def summarize(t: pd.DataFrame) -> dict:
    def pct_ge(col, x):
        return float((t[col] >= x).mean())
    n = len(t)
    exit_mix = (t["exit_reason"].value_counts(normalize=True).round(3)).to_dict()
    return {
        "n_trades": n, "n_incomplete": int(t.attrs.get("n_incomplete", 0)),
        "mean_R_gross": round(float(t["r_gross"].mean()), 3),
        "mean_R_net": round(float(t["r_net"].mean()), 3),
        "median_R_net": round(float(t["r_net"].median()), 3),
        "win_rate_net": round(float((t["r_net"] > 0).mean()), 3),
        "stop_out_loss_rate": round(float(t["stop_out_loss"].mean()), 3),
        "gap_stop_freq": round(float((t["exit_reason"] == "gap_stop").mean()), 4),
        "realized_below_-1R_freq": round(float(t["realized_below_minus1R"].mean()), 4),
        "pct_ge_+1R_net": round(pct_ge("r_net", 1.0), 3),
        "pct_ge_+2R_net": round(pct_ge("r_net", 2.0), 3),
        "pct_ge_+3R_net": round(pct_ge("r_net", 3.0), 3),
        "reached_1R_freq": round(float(t["reached_1R"].mean()), 3),
        "reached_2R_freq": round(float(t["reached_2R"].mean()), 3),
        "avg_winner_R_net": round(float(t.loc[t["r_net"] > 0, "r_net"].mean()), 3),
        "avg_loser_R_net": round(float(t.loc[t["r_net"] <= 0, "r_net"].mean()), 3),
        "mean_mfe_R": round(float(t["mfe_R"].mean()), 3),
        "mean_mae_R": round(float(t["mae_R"].mean()), 3),
        "median_bars_held": float(t["bars_held"].median()),
        "exit_reason_mix": exit_mix,
        "params": {"stop_mult": STOP_MULT, "trail_mult": TRAIL_MULT, "max_horizon": MAX_HORIZON,
                   "atr_n": ATR_N, "cost_bps_rt": COST_BPS_RT, "top_n": TOP_N, "min_history": MIN_HISTORY},
    }


def main() -> None:
    print("Phase 1 — generating unconditional R-multiple labels (base rate) ...")
    t = generate()
    s = summarize(t)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_DIR / "labels_unconditional.csv", index=False)
    (OUT_DIR / "labels_unconditional_summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    print(f"\nlabels -> {OUT_DIR / 'labels_unconditional.csv'}  ({len(t):,} trades)")


if __name__ == "__main__":
    main()
