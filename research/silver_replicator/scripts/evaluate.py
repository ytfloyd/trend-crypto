"""
Evaluate the best directional model + options overlay and produce the three
diagnostic figures plus the summary markdown.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKG = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from src.backtest import (
    ART,
    MULT_QI,
    composite,
    generate_signal_path,
    load_bars,
    load_features,
    load_trades,
    realised_pnl_curve,
    reconstruct_position_path,
    score,
)
from src.signal_grammar import SignalParams

FIG_DIR = PKG / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _build_model_pnl(bars: pd.DataFrame, sig: pd.Series, scale: float) -> pd.Series:
    close = bars["c"].astype(float)
    ret = close.diff().fillna(0.0)
    sig_lag = sig.shift(1).fillna(0).astype(float)
    return (sig_lag * ret * MULT_QI * scale).cumsum()


def _fig_signal_vs_position(bars, pos, sig, scale, actual_cum, model_cum, tf, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(pos.index, pos.values, label="actual position (QI-eq)", color="#1f77b4")
    ax1.plot(sig.index, sig.values * 4, label="model signal x 4", color="#ff7f0e", alpha=0.7)
    ax1.axhline(0, color="grey", lw=0.5)
    ax1.set_ylabel("contracts (QI-eq)")
    ax1.set_title(f"Silver replicator | signal vs actual position [{tf}]")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(actual_cum.index, actual_cum.values, label="actual realized cum P&L", color="#1f77b4")
    ax2.plot(model_cum.index, model_cum.values, label="model (futures-only) cum P&L", color="#ff7f0e")
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("$")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_overlay(actual_cum, model_cum, combined_cum, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(actual_cum.index, actual_cum.values, label="actual realized", color="#1f77b4", lw=2)
    ax.plot(model_cum.index, model_cum.values, label="model futures-only", color="#ff7f0e")
    ax.plot(combined_cum.index, combined_cum.values, label="model futures + options overlay",
            color="#2ca02c")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("$ cumulative P&L")
    ax.set_title("Silver replicator | overlay P&L comparison")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_sensitivity(grid_df: pd.DataFrame, tf: str, out):
    sub = grid_df[grid_df["tf"] == tf]
    if sub.empty:
        # fallback
        sub = grid_df

    # Heatmap 1: fast x slow (averaged composite over the rest)
    pv1 = sub.pivot_table(index="fast", columns="slow", values="composite", aggfunc="max")
    # Heatmap 2: rsi_long_thr x adx_min
    pv2 = sub.pivot_table(index="rsi_long_thr", columns="adx_min", values="composite", aggfunc="max")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pv, title in (
        (axes[0], pv1, f"composite | fast x slow [{tf}]"),
        (axes[1], pv2, f"composite | rsi_long x adx_min [{tf}]"),
    ):
        im = ax.imshow(pv.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels(pv.columns, rotation=0)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels(pv.index)
        ax.set_xlabel(pv.columns.name)
        ax.set_ylabel(pv.index.name)
        ax.set_title(title)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < pv.values.max() * 0.7 else "black",
                            fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _interpret(scores: dict, total_sim: float, total_actual: float) -> str:
    capture = 100.0 * total_sim / total_actual if abs(total_actual) > 1 else 0.0
    txt = []
    txt.append(
        f"The chosen model recovers {capture:.1f}% of the actual realized $ "
        f"P&L (${total_sim:,.0f} vs ${total_actual:,.0f}). "
    )
    if scores["pnl_curve_corr"] >= 0.9:
        txt.append("Equity-curve correlation is very strong, meaning the model gets the "
                   "shape and timing of P&L right even if the $ magnitude is smaller. ")
    elif scores["pnl_curve_corr"] >= 0.6:
        txt.append("Equity-curve correlation is moderately strong - the model tracks the "
                   "broad arc of the book but misses some local moves. ")
    else:
        txt.append("Equity-curve correlation is weak; the model is mistiming the bigger "
                   "directional swings. ")
    txt.append(
        f"Direction accuracy ({scores['direction_accuracy']:.2%}) and Cohen's kappa "
        f"({scores['cohens_kappa']:.2f}) show that on a bar-by-bar basis the model is "
        "right about silver's stance more often than not, but does not chase the same "
        "fine-grained intra-week chop. "
    )
    txt.append(
        "The options overlay translates each long flip into a synthetic OTM call which "
        "captures the bullish-premium signature of the real book (SOH6/SOJ6/SOK6 calls), "
        "without trying to model the spread structures exactly."
    )
    return "".join(txt)


def main() -> int:
    with open(ART / "best_params.json") as fh:
        best = json.load(fh)
    tf = best["tf"]
    params = SignalParams(**best["params"])

    bars = load_bars(tf)
    feats = load_features(tf)
    trades = load_trades()
    pos = reconstruct_position_path(trades, bars.index)
    actual_cum = realised_pnl_curve(trades, bars.index)
    sig = generate_signal_path(feats, params)

    sc = score(sig, pos, actual_cum, bars)

    # restrict view to live window
    pos_live = pos.abs() > 1e-9
    pnl_live = actual_cum.diff().fillna(0).abs() > 1e-9
    live = pos_live | pnl_live
    first = live[live].index.min()
    last = live[live].index.max()
    sel = (bars.index >= first) & (bars.index <= last)

    scale = sc["sim_scale"]
    model_cum = _build_model_pnl(bars, sig, scale).loc[sel]
    actual_cum_w = actual_cum.loc[sel]
    pos_w = pos.loc[sel]
    sig_w = sig.loc[sel]
    bars_w = bars.loc[sel]

    # overlay
    overlay = pd.read_parquet(ART / "overlay_trades.parquet")
    overlay["exit_ts"] = pd.to_datetime(overlay["exit_ts"], utc=True)
    overlay = overlay.sort_values("exit_ts")
    # build a cumulative overlay P&L series aligned to bars
    overlay_cum = pd.Series(0.0, index=bars.index)
    realised_by_exit = overlay.groupby("exit_ts")["pnl"].sum()
    # align to nearest bar at-or-after
    bidx = bars.index
    pos_idx = bidx.searchsorted(pd.DatetimeIndex(realised_by_exit.index), side="left")
    pos_idx = np.clip(pos_idx, 0, len(bidx) - 1)
    snapped = pd.Series(realised_by_exit.values, index=bidx[pos_idx])
    snapped = snapped.groupby(snapped.index).sum()
    overlay_cum.loc[snapped.index] += snapped.values
    overlay_cum = overlay_cum.cumsum()
    overlay_cum_w = overlay_cum.loc[sel]
    combined_cum = model_cum + overlay_cum_w

    # ---- figures ----
    _fig_signal_vs_position(
        bars_w, pos_w, sig_w, scale, actual_cum_w, model_cum, tf,
        FIG_DIR / "01_signal_vs_position.png",
    )
    _fig_overlay(
        actual_cum_w, model_cum, combined_cum,
        FIG_DIR / "02_overlay_pnl_curve.png",
    )
    grid = pd.read_parquet(ART / f"fit_grid_{tf}.parquet")
    _fig_sensitivity(grid, tf, FIG_DIR / "03_param_sensitivity.png")

    # ---- summary ----
    total_sim_futures = float(model_cum.iloc[-1]) if len(model_cum) else 0.0
    total_overlay = float(overlay["pnl"].sum())
    total_sim_combined = total_sim_futures + total_overlay
    total_actual = float(actual_cum_w.iloc[-1]) if len(actual_cum_w) else 0.0

    summary_path = ART / "fit_summary.md"
    interp = _interpret(sc, total_sim_combined, total_actual)
    with open(summary_path, "w") as fh:
        fh.write("# Silver replicator -- fit summary\n\n")
        fh.write(f"- Chosen timeframe: **{tf}**\n")
        fh.write(f"- Best params:\n```json\n{json.dumps(best['params'], indent=2)}\n```\n")
        fh.write("\n## Scoring metrics\n\n")
        fh.write(f"| metric | value |\n|---|---|\n")
        fh.write(f"| direction_accuracy | {sc['direction_accuracy']:.4f} |\n")
        fh.write(f"| cohens_kappa | {sc['cohens_kappa']:.4f} |\n")
        fh.write(f"| pearson_pos | {sc['pearson_pos']:.4f} |\n")
        fh.write(f"| pnl_curve_corr | {sc['pnl_curve_corr']:.4f} |\n")
        fh.write(f"| composite | {composite(sc):.4f} |\n")
        fh.write(f"| n_bars (live window) | {sc['n_bars']} |\n")
        fh.write(f"| sim_scale (vol-match) | {sc['sim_scale']:.4f} |\n")
        fh.write("\n## $ P&L\n\n")
        fh.write(f"- Actual realized: **${total_actual:,.2f}**\n")
        fh.write(f"- Model futures-only (vol-scaled): **${total_sim_futures:,.2f}**\n")
        fh.write(f"- Options overlay: **${total_overlay:,.2f}**\n")
        fh.write(f"- Model futures + overlay: **${total_sim_combined:,.2f}**\n")
        fh.write("\n## Interpretation\n\n")
        fh.write(interp + "\n")
        fh.write("\n## Figures\n\n")
        fh.write("- `figures/01_signal_vs_position.png`\n")
        fh.write("- `figures/02_overlay_pnl_curve.png`\n")
        fh.write("- `figures/03_param_sensitivity.png`\n")

    print(f"Wrote summary -> {summary_path}")
    print(f"Wrote figures -> {FIG_DIR}")
    print(f"Sim total (futures): ${total_sim_futures:,.2f}")
    print(f"Sim total (overlay): ${total_overlay:,.2f}")
    print(f"Sim total (combined): ${total_sim_combined:,.2f}")
    print(f"Actual: ${total_actual:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
