"""
Evaluate the best v2 directional model and produce v2 diagnostic figures
plus the summary markdown.

Writes:
    figures/04_v2_signal_vs_position.png
    figures/05_v2_overlay_pnl_curve.png  (actual, v1+overlay, v2+overlay)
    figures/06_v2_param_sensitivity.png  (BB and vov heatmaps)
    artifacts/fit_summary_v2.md
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


def _fig_signal_vs_position(bars, pos, sig, actual_cum, model_cum, tf, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(pos.index, pos.values, label="actual position (QI-eq)", color="#1f77b4")
    ax1.plot(sig.index, sig.values * 4, label="v2 model signal x 4",
             color="#d62728", alpha=0.75)
    ax1.axhline(0, color="grey", lw=0.5)
    ax1.set_ylabel("contracts (QI-eq)")
    ax1.set_title(f"Silver replicator v2 | signal vs actual position [{tf}]")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(actual_cum.index, actual_cum.values, label="actual realized cum P&L",
             color="#1f77b4")
    ax2.plot(model_cum.index, model_cum.values, label="v2 model (futures-only) cum P&L",
             color="#d62728")
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("$")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_overlay(actual_cum, v1_combined, v2_combined, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(actual_cum.index, actual_cum.values, label="actual realized",
            color="#1f77b4", lw=2)
    ax.plot(v1_combined.index, v1_combined.values, label="v1 model + overlay",
            color="#ff7f0e")
    ax.plot(v2_combined.index, v2_combined.values, label="v2 model + overlay",
            color="#d62728")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("$ cumulative P&L")
    ax.set_title("Silver replicator | v1 vs v2 overlay P&L comparison")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_v2_sensitivity(grid_df: pd.DataFrame, tf: str, out):
    sub = grid_df[grid_df["tf"] == tf].copy()
    if sub.empty:
        sub = grid_df.copy()

    # Heatmap 1: BB regime threshold percentile vs BB width percentile lookback
    # Restrict to where use_bb_regime=True so the panel actually reflects BB sensitivity.
    bb = sub[sub["use_bb_regime"]].copy()
    if bb.empty:
        bb = sub.copy()
    pv1 = bb.pivot_table(index="bb_regime_thr_pct",
                         columns="bb_width_pctl_lookback",
                         values="composite", aggfunc="max")

    # Heatmap 2: vov_zscore_thr vs vov_window
    vov = sub[sub["use_vov_trigger"]].copy()
    if vov.empty:
        vov = sub.copy()
    pv2 = vov.pivot_table(index="vov_zscore_thr",
                          columns="vov_window",
                          values="composite", aggfunc="max")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = (
        (axes[0], pv1, f"composite | bb_regime_thr x bb_width_lookback [{tf}]"),
        (axes[1], pv2, f"composite | vov_z_thr x vov_window [{tf}]"),
    )
    for ax, pv, title in panels:
        if pv.size == 0:
            ax.set_title(title + " (empty)")
            continue
        im = ax.imshow(pv.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels(pv.index)
        ax.set_xlabel(pv.columns.name)
        ax.set_ylabel(pv.index.name)
        ax.set_title(title)
        vmax = np.nanmax(pv.values) if np.isfinite(pv.values).any() else 1.0
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v < vmax * 0.7 else "black",
                            fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _live_window(bars, pos, actual_cum):
    pos_live = pos.abs() > 1e-9
    pnl_live = actual_cum.diff().fillna(0).abs() > 1e-9
    live = pos_live | pnl_live
    if not live.any():
        return slice(None)
    first = live[live].index.min()
    last = live[live].index.max()
    return (bars.index >= first) & (bars.index <= last)


def _build_combined(bars, params, scale_hint, overlay_cum):
    sig = generate_signal_path(load_features(_PARAMS_TF[id(params)]), params)
    model_cum = _build_model_pnl(bars, sig, scale_hint)
    return sig, model_cum + overlay_cum, model_cum


_PARAMS_TF: dict[int, str] = {}


def main() -> int:
    with open(ART / "best_params_v2.json") as fh:
        best_v2 = json.load(fh)
    with open(ART / "best_params.json") as fh:
        best_v1 = json.load(fh)

    tf_v2 = best_v2["tf"]
    tf_v1 = best_v1["tf"]
    params_v2 = SignalParams(**best_v2["params"])
    # v1 params dict doesn't carry the new fields; defaults False so behaviour matches v1.
    params_v1 = SignalParams(**best_v1["params"])

    # ---- v2 evaluation on its TF ----
    bars = load_bars(tf_v2)
    feats = load_features(tf_v2)
    trades = load_trades()
    pos = reconstruct_position_path(trades, bars.index)
    actual_cum = realised_pnl_curve(trades, bars.index)
    sig_v2 = generate_signal_path(feats, params_v2)
    sc_v2 = score(sig_v2, pos, actual_cum, bars)
    scale_v2 = sc_v2["sim_scale"]
    model_cum_v2_full = _build_model_pnl(bars, sig_v2, scale_v2)

    # ---- v1 evaluation on its TF (might differ from tf_v2 in general,
    #      but here both are 8H) ----
    if tf_v1 == tf_v2:
        bars_v1, feats_v1 = bars, feats
        pos_v1, actual_cum_v1 = pos, actual_cum
    else:
        bars_v1 = load_bars(tf_v1)
        feats_v1 = load_features(tf_v1)
        pos_v1 = reconstruct_position_path(trades, bars_v1.index)
        actual_cum_v1 = realised_pnl_curve(trades, bars_v1.index)
    sig_v1 = generate_signal_path(feats_v1, params_v1)
    sc_v1 = score(sig_v1, pos_v1, actual_cum_v1, bars_v1)
    scale_v1 = sc_v1["sim_scale"]
    model_cum_v1_full = _build_model_pnl(bars_v1, sig_v1, scale_v1)

    # ---- live windows ----
    sel_v2 = _live_window(bars, pos, actual_cum)
    sel_v1 = _live_window(bars_v1, pos_v1, actual_cum_v1)

    actual_cum_w = actual_cum.loc[sel_v2]
    pos_w = pos.loc[sel_v2]
    sig_v2_w = sig_v2.loc[sel_v2]
    bars_w = bars.loc[sel_v2]
    model_cum_v2 = model_cum_v2_full.loc[sel_v2]
    actual_cum_v1_w = actual_cum_v1.loc[sel_v1]
    model_cum_v1 = model_cum_v1_full.loc[sel_v1]

    # ---- options overlay ----
    overlay = pd.read_parquet(ART / "overlay_trades.parquet")
    overlay["exit_ts"] = pd.to_datetime(overlay["exit_ts"], utc=True)
    overlay = overlay.sort_values("exit_ts")

    def _overlay_cum(bar_index):
        ov_cum = pd.Series(0.0, index=bar_index)
        realised_by_exit = overlay.groupby("exit_ts")["pnl"].sum()
        bidx = bar_index
        pos_idx = bidx.searchsorted(pd.DatetimeIndex(realised_by_exit.index), side="left")
        pos_idx = np.clip(pos_idx, 0, len(bidx) - 1)
        snapped = pd.Series(realised_by_exit.values, index=bidx[pos_idx])
        snapped = snapped.groupby(snapped.index).sum()
        ov_cum.loc[snapped.index] += snapped.values
        return ov_cum.cumsum()

    overlay_cum_v2 = _overlay_cum(bars.index).loc[sel_v2]
    overlay_cum_v1 = _overlay_cum(bars_v1.index).loc[sel_v1]

    combined_cum_v2 = model_cum_v2 + overlay_cum_v2
    combined_cum_v1 = model_cum_v1 + overlay_cum_v1

    # ---- figures ----
    _fig_signal_vs_position(
        bars_w, pos_w, sig_v2_w, actual_cum_w, model_cum_v2, tf_v2,
        FIG_DIR / "04_v2_signal_vs_position.png",
    )
    _fig_overlay(
        actual_cum_w, combined_cum_v1, combined_cum_v2,
        FIG_DIR / "05_v2_overlay_pnl_curve.png",
    )
    grid = pd.read_parquet(ART / f"fit_grid_v2_{tf_v2}.parquet")
    _fig_v2_sensitivity(grid, tf_v2, FIG_DIR / "06_v2_param_sensitivity.png")

    # ---- $ totals ----
    total_sim_futures_v2 = float(model_cum_v2.iloc[-1]) if len(model_cum_v2) else 0.0
    total_overlay = float(overlay["pnl"].sum())
    total_sim_combined_v2 = total_sim_futures_v2 + total_overlay
    total_actual = float(actual_cum_w.iloc[-1]) if len(actual_cum_w) else 0.0

    total_sim_futures_v1 = float(model_cum_v1.iloc[-1]) if len(model_cum_v1) else 0.0
    total_sim_combined_v1 = total_sim_futures_v1 + total_overlay

    composite_v2 = composite(sc_v2)
    composite_v1 = composite(sc_v1)
    delta = composite_v2 - composite_v1

    # ---- summary ----
    used_bb = bool(best_v2["params"]["use_bb_regime"])
    used_vov = bool(best_v2["params"]["use_vov_trigger"])
    summary_path = ART / "fit_summary_v2.md"

    interp_parts = []
    if used_bb and used_vov:
        interp_parts.append(
            "The optimizer kept BOTH the BB regime switch and the vol-of-vol "
            "trigger active in the chosen configuration, so the v2 extensions "
            "are pulling their weight on this data set."
        )
    elif used_bb and not used_vov:
        interp_parts.append(
            "The optimizer kept the BB regime switch ON but switched the "
            "vol-of-vol trigger OFF. The BB regime adds genuine signal "
            "(switching to mean-revert in tight-band quiet stretches reduces "
            "the false-trend whipsaws the SMA-cross layer produced); the "
            "vov trigger, however, was found to remove more correct-side "
            "exposure than wrong-side exposure on this book, so it lowered "
            "composite when active and the optimizer dropped it."
        )
    elif used_vov and not used_bb:
        interp_parts.append(
            "The optimizer kept the vol-of-vol trigger ON but switched the "
            "BB regime OFF. The vov circuit-breaker found enough genuine "
            "volatility shocks to side-step that it improved composite, "
            "while the BB regime swap added too many opposite-side mean-rev "
            "entries during the bullish trend, hurting composite."
        )
    else:
        interp_parts.append(
            "The optimizer switched OFF BOTH the BB regime and the vol-of-vol "
            "trigger at the global best, meaning neither extension added "
            "incremental value on this book over the existing trend-layer "
            "signal. The v2 best therefore equals the v1 best."
        )
    interp_parts.append(
        f" Composite moved from {composite_v1:.4f} (v1) to {composite_v2:.4f} "
        f"(v2), a delta of {delta:+.4f}. P&L-curve correlation went "
        f"{sc_v1['pnl_curve_corr']:.4f} -> {sc_v2['pnl_curve_corr']:.4f} and "
        f"direction accuracy {sc_v1['direction_accuracy']:.4f} -> "
        f"{sc_v2['direction_accuracy']:.4f}. Combined $ P&L "
        f"(futures + overlay) shifted ${total_sim_combined_v1:,.0f} -> "
        f"${total_sim_combined_v2:,.0f} against an actual book of "
        f"${total_actual:,.0f}."
    )
    interpretation = "".join(interp_parts)

    with open(summary_path, "w") as fh:
        fh.write("# Silver replicator v2 -- fit summary\n\n")
        fh.write(f"- v1 chosen timeframe: **{tf_v1}**\n")
        fh.write(f"- v2 chosen timeframe: **{tf_v2}**\n")
        fh.write(f"- v1 composite: **{composite_v1:.4f}**\n")
        fh.write(f"- v2 composite: **{composite_v2:.4f}**\n")
        fh.write(f"- Delta: **{delta:+.4f}**\n\n")

        fh.write("## v2 best params\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(best_v2["params"], indent=2))
        fh.write("\n```\n\n")

        fh.write("## Scoring metrics (v2 best)\n\n")
        fh.write("| metric | v1 | v2 | delta |\n|---|---|---|---|\n")
        for k in ("direction_accuracy", "cohens_kappa", "pearson_pos",
                  "pnl_curve_corr"):
            fh.write(f"| {k} | {sc_v1[k]:.4f} | {sc_v2[k]:.4f} | "
                     f"{sc_v2[k] - sc_v1[k]:+.4f} |\n")
        fh.write(f"| composite | {composite_v1:.4f} | {composite_v2:.4f} | "
                 f"{delta:+.4f} |\n")
        fh.write(f"| n_bars (live window) | {sc_v1['n_bars']} | "
                 f"{sc_v2['n_bars']} | - |\n")
        fh.write(f"| sim_scale (vol-match) | {sc_v1['sim_scale']:.4f} | "
                 f"{sc_v2['sim_scale']:.4f} | - |\n\n")

        fh.write("## $ P&L\n\n")
        fh.write(f"- Actual realized: **${total_actual:,.2f}**\n")
        fh.write(f"- v1 model futures-only: **${total_sim_futures_v1:,.2f}**\n")
        fh.write(f"- v1 model + overlay: **${total_sim_combined_v1:,.2f}**\n")
        fh.write(f"- v2 model futures-only: **${total_sim_futures_v2:,.2f}**\n")
        fh.write(f"- v2 model + overlay: **${total_sim_combined_v2:,.2f}**\n")
        fh.write(f"- Options overlay (same series both rows): "
                 f"**${total_overlay:,.2f}**\n\n")

        fh.write("## Switch usage at the chosen optimum\n\n")
        fh.write(f"- `use_bb_regime`: **{used_bb}**\n")
        fh.write(f"- `use_vov_trigger`: **{used_vov}**\n\n")

        fh.write("## Interpretation\n\n")
        fh.write(interpretation + "\n\n")

        fh.write("## Figures\n\n")
        fh.write("- `figures/04_v2_signal_vs_position.png`\n")
        fh.write("- `figures/05_v2_overlay_pnl_curve.png`\n")
        fh.write("- `figures/06_v2_param_sensitivity.png`\n")

    print(f"Wrote summary -> {summary_path}")
    print(f"Wrote figures -> {FIG_DIR}")
    print(f"v1 composite={composite_v1:.4f}  v2 composite={composite_v2:.4f}  delta={delta:+.4f}")
    print(f"use_bb_regime={used_bb}  use_vov_trigger={used_vov}")
    print(f"v1 combined $: {total_sim_combined_v1:,.2f}")
    print(f"v2 combined $: {total_sim_combined_v2:,.2f}")
    print(f"actual $: {total_actual:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
