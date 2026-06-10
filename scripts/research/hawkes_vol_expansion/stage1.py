#!/usr/bin/env python3
"""Stage 1 calibration gate: state-dependent Hawkes vol-expansion hazard.

Targets: CL, ZL, ZW (independent runs).
Pre-registered PASS: top-decile realized event rate >= 1.5× bottom-decile for
>=2/3 contracts AND time-rescaling KS does not reject at 5% for those contracts.

Data gaps (documented, not proxied beyond spec):
  - POSEXT (COT/CFTC): no loader in repo; omitted from μ
  - GAMMA: no clean options gamma series; omitted
  - OIBUILD: no OI time series; volume/range proxy labeled OIBUILD_PROXY
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))

from hawkes_vol_expansion.data import (  # noqa: E402
    ROLL_CONVENTIONS,
    load_continuous_minute_bars,
    resample_bars,
    resolve_lake_path,
)
from hawkes_vol_expansion.features import (  # noqa: E402
    build_causal_features,
    extract_expansion_events,
)
from hawkes_vol_expansion.hawkes import (  # noqa: E402
    FitResult,
    HawkesParams,
    compensator,
    fit_hawkes_mle,
    intensity_at_events,
    ks_exp_test,
    predicted_event_probability,
    time_rescaling_gaps,
)

SYMBOLS = ("CL", "ZL", "ZW")
KAPPA_GRID = (2.5, 3.0, 3.5)
PRIMARY_KAPPA = 3.0
HORIZON_DAYS = 5
MIN_TRAIN_EVENTS = 20
MIN_TRAIN_DAYS = 120
TEST_STEP_DAYS = 45
BAR_SIZE_DEFAULT = "4h"
HORIZON_BY_BAR = {"1D": 5, "4h": 30}  # 30×4h ≈ 5 calendar days
DECILE_LIFT_THRESHOLD = 1.5
KS_ALPHA = 0.05
GATE_MIN_CONTRACTS = 2


@dataclass
class ContractStage1Result:
    symbol: str
    kappa: float
    n_events: int
    n_days: int
    fit: FitResult
    decile_bottom_rate: float
    decile_top_rate: float
    decile_lift: float
    ks_stat: float
    ks_pvalue: float
    gate_decile_pass: bool
    gate_ks_pass: bool
    gate_pass: bool
    diagnosis: str
    roll_convention: str
    data_start: str
    data_end: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    p.add_argument("--kappa", type=float, default=PRIMARY_KAPPA)
    p.add_argument("--lake-path", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/research/hawkes_vol_expansion/stage1",
    )
    p.add_argument("--sweep-kappa", action="store_true")
    p.add_argument("--bar-size", default=BAR_SIZE_DEFAULT, help="1D or 4h (default 4h)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lake = resolve_lake_path(args.lake_path)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    kappas = list(KAPPA_GRID) if args.sweep_kappa else [args.kappa]
    all_results: list[ContractStage1Result] = []

    for symbol in args.symbols:
        for kappa in kappas:
            print(f"\n=== {symbol} κ={kappa} ===")
            result, event_frame = run_contract_stage1(
                symbol, kappa=kappa, lake_path=lake, out_dir=out_dir, bar_size=args.bar_size
            )
            all_results.append(result)
            _write_contract_plots(result, event_frame, out_dir, bar_size=args.bar_size)

    primary = [r for r in all_results if abs(r.kappa - PRIMARY_KAPPA) < 1e-9]
    verdict = _aggregate_verdict(primary)
    card = _build_result_card(primary, verdict, lake, kappas, bar_size=args.bar_size)
    card_path = out_dir / "STAGE1_RESULT_CARD.md"
    card_path.write_text(card)
    json_path = out_dir / "stage1_results.json"
    json_path.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "results": [_result_to_dict(r) for r in all_results],
            },
            indent=2,
            default=str,
        )
    )
    print("\n" + card)
    print(f"\nWrote {card_path}")
    return 0 if verdict["pass"] else 1


def run_contract_stage1(
    symbol: str,
    *,
    kappa: float,
    lake_path: Path,
    out_dir: Path,
    bar_size: str = BAR_SIZE_DEFAULT,
) -> tuple[ContractStage1Result, pd.DataFrame]:
    minute = load_continuous_minute_bars(symbol, lake_path=lake_path)
    daily = resample_bars(minute, bar_size)
    horizon = HORIZON_BY_BAR.get(bar_size, HORIZON_DAYS)
    daily = build_causal_features(daily)
    daily = extract_expansion_events(daily, kappa=kappa)
    daily = daily.dropna(subset=["VOLCOMP_z", "OIBUILD_PROXY", "sigma_ewma_prior"]).reset_index(drop=True)

    t0 = pd.Timestamp(daily["ts"].iloc[0]).date().isoformat()
    t1 = pd.Timestamp(daily["ts"].iloc[-1]).date().isoformat()
    origin = pd.Timestamp(daily["ts"].iloc[0])
    daily["t_days"] = (pd.to_datetime(daily["ts"], utc=True) - origin).dt.total_seconds() / 86400.0

    oos_deciles: list[dict[str, float]] = []
    oos_gaps: list[float] = []
    last_fit: FitResult | None = None

    train_end = MIN_TRAIN_DAYS
    max_t = float(daily["t_days"].iloc[-1])
    while train_end + TEST_STEP_DAYS <= max_t:
        train = daily[daily["t_days"] < train_end]
        test = daily[
            (daily["t_days"] >= train_end) & (daily["t_days"] < train_end + TEST_STEP_DAYS)
        ]
        events = train[train["is_event"]]
        if len(events) < MIN_TRAIN_EVENTS:
            train_end += TEST_STEP_DAYS
            continue

        event_t = events["t_days"].to_numpy()
        z = np.column_stack([events["VOLCOMP_z"], events["OIBUILD_PROXY"]])
        fit = fit_hawkes_mle(event_t, z)
        last_fit = fit
        if not fit.success:
            train_end += TEST_STEP_DAYS
            continue

        p = fit.params
        for _, row in test.iterrows():
            mu_t = p.mu(float(row["VOLCOMP_z"]), float(row["OIBUILD_PROXY"]))
            past = train[train["t_days"] < row["t_days"]]
            past_events = past[past["is_event"]]["t_days"].to_numpy()
            exc = 0.0
            for pe in past_events:
                exc += np.exp(-p.beta * (row["t_days"] - pe))
            lam = mu_t + p.alpha * exc
            prob = predicted_event_probability(lam, horizon)
            realized = bool(
                daily[
                    (daily["t_days"] > row["t_days"])
                    & (daily["t_days"] <= row["t_days"] + horizon)
                ]["is_event"].any()
            )
            oos_deciles.append({"prob": prob, "realized": float(realized)})

        test_events = test[test["is_event"]]["t_days"].to_numpy()
        if len(test_events) >= 2:
            mu_grid = np.exp(
                np.clip(
                    p.w0
                    + p.w_volcomp * test["VOLCOMP_z"].to_numpy()
                    + p.w_oibuild * test["OIBUILD_PROXY"].to_numpy(),
                    -20,
                    20,
                )
            )
            tg = test["t_days"].to_numpy()
            gaps = time_rescaling_gaps(test_events, mu_grid, tg, alpha=p.alpha, beta=p.beta)
            oos_gaps.extend(gaps.tolist())

        train_end += TEST_STEP_DAYS

    if not oos_deciles or last_fit is None:
        fit = last_fit or FitResult(
            params=HawkesParams(0, 0, 0, 0, 1),
            log_likelihood=float("nan"),
            n_events=0,
            success=False,
            message="no_oos_folds",
            std_errors={},
        )
        return (
            ContractStage1Result(
                symbol=symbol,
                kappa=kappa,
                n_events=int(daily["is_event"].sum()),
                n_days=len(daily),
                fit=fit,
                decile_bottom_rate=0.0,
                decile_top_rate=0.0,
                decile_lift=0.0,
                ks_stat=float("nan"),
                ks_pvalue=float("nan"),
                gate_decile_pass=False,
                gate_ks_pass=False,
                gate_pass=False,
                diagnosis="insufficient_oos_folds",
                roll_convention=ROLL_CONVENTIONS.get(symbol, "unknown"),
                data_start=t0,
                data_end=t1,
            ),
            daily,
        )

    dec = pd.DataFrame(oos_deciles)
    dec["decile"] = pd.qcut(dec["prob"], 10, labels=False, duplicates="drop")
    by_dec = dec.groupby("decile")["realized"].mean()
    bottom = float(by_dec.min()) if len(by_dec) else 0.0
    top = float(by_dec.max()) if len(by_dec) else 0.0
    lift = top / bottom if bottom > 0 else (float("inf") if top > 0 else 0.0)

    ks_stat, ks_p = ks_exp_test(np.array(oos_gaps))
    gate_decile = lift >= DECILE_LIFT_THRESHOLD
    gate_ks = bool(np.isfinite(ks_p) and ks_p >= KS_ALPHA)
    gate_pass = gate_decile and gate_ks
    diagnosis = _diagnose(last_fit.params, gate_decile, gate_ks)

    return (
        ContractStage1Result(
            symbol=symbol,
            kappa=kappa,
            n_events=int(daily["is_event"].sum()),
            n_days=len(daily),
            fit=last_fit,
            decile_bottom_rate=bottom,
            decile_top_rate=top,
            decile_lift=lift,
            ks_stat=float(ks_stat) if np.isfinite(ks_stat) else float("nan"),
            ks_pvalue=float(ks_p) if np.isfinite(ks_p) else float("nan"),
            gate_decile_pass=gate_decile,
            gate_ks_pass=gate_ks,
            gate_pass=gate_pass,
            diagnosis=diagnosis,
            roll_convention=ROLL_CONVENTIONS.get(symbol, "unknown"),
            data_start=t0,
            data_end=t1,
        ),
        daily,
    )


def _diagnose(params: HawkesParams, decile_ok: bool, ks_ok: bool) -> str:
    br = params.branching_ratio
    if decile_ok and ks_ok:
        return "pass"
    if br < 0.15:
        return "(a) alpha_small_edge_in_mu: collapse to conditional-hazard logit"
    return "(b) clustering_present_subdaily_resolution: try finer bar resolution"


def _aggregate_verdict(results: list[ContractStage1Result]) -> dict[str, object]:
    passing = [r for r in results if r.gate_pass]
    decile_pass = [r for r in results if r.gate_decile_pass]
    return {
        "pass": len(passing) >= GATE_MIN_CONTRACTS,
        "contracts_passing": [r.symbol for r in passing],
        "contracts_decile_pass": [r.symbol for r in decile_pass],
        "n_contracts": len(results),
        "criterion": (
            f"top/bottom decile lift >= {DECILE_LIFT_THRESHOLD} for >={GATE_MIN_CONTRACTS}/3 "
            f"AND KS p >= {KS_ALPHA}"
        ),
    }


def _build_result_card(
    results: list[ContractStage1Result],
    verdict: dict[str, object],
    lake: Path,
    kappas: list[float],
    bar_size: str,
) -> str:
    lines = [
        "# Stage 1 — Hawkes Vol-Expansion Hazard (Result Card)",
        "",
        f"**Verdict: {'PASS' if verdict['pass'] else 'FAIL'}**",
        "",
        "## Pre-registered gate",
        f"- κ primary = {PRIMARY_KAPPA}; sweep = {kappas}",
        f"- Bar size = {bar_size}; horizon = {HORIZON_BY_BAR.get(bar_size, HORIZON_DAYS)} bars",
        f"- Walk-forward: expanding window, min train {MIN_TRAIN_DAYS}d / {MIN_TRAIN_EVENTS} events, "
        f"test step {TEST_STEP_DAYS}d",
        f"- Lift threshold: {DECILE_LIFT_THRESHOLD}× (top vs bottom decile, OOS only)",
        f"- KS test: do not reject Exp(1) rescaled gaps at α={KS_ALPHA}",
        "",
        "## Data & feature gaps",
        "- **POSEXT (COT/CFTC):** not in repo; omitted (no silent lookahead proxy).",
        "- **GAMMA:** omitted (no clean dealer-gamma series).",
        "- **OIBUILD:** no futures OI time series; `OIBUILD_PROXY` = z-scored volume growth − range tightness.",
        f"- Lake: `{lake}`",
        "",
        "## Per-contract (OOS, κ=3)",
        "",
        "| Symbol | Events | Bars | α/β | w_VOLCOMP | Lift (top/bot) | KS p | Gate |",
        "|--------|--------|------|-----|-----------|----------------|------|------|",
    ]
    for r in results:
        p = r.fit.params
        lines.append(
            f"| {r.symbol} | {r.n_events} | {r.n_days} | {p.branching_ratio:.3f} | "
            f"{p.w_volcomp:.3f} | {r.decile_lift:.2f}× ({r.decile_top_rate:.3f}/{r.decile_bottom_rate:.3f}) | "
            f"{r.ks_pvalue:.3f} | {'PASS' if r.gate_pass else 'FAIL'} |"
        )
        lines.append(f"| roll | {r.roll_convention} | {r.data_start} → {r.data_end} | | | | | |")

    lines.extend(["", "## Diagnosis (failures)", ""])
    for r in results:
        if not r.gate_pass:
            lines.append(f"- **{r.symbol}:** {r.diagnosis}")

    if not verdict["pass"]:
        lines.extend(
            [
                "",
                "## Recommendation",
                "Stage 1 did not pass. **Do not proceed to Stage 2.**",
                "See per-contract (a)/(b) diagnosis above.",
            ]
        )
    else:
        lines.extend(["", "## Recommendation", "Stage 1 passed. Stage 2 backtest may proceed."])

    lines.extend(["", f"Artifacts: `artifacts/research/hawkes_vol_expansion/stage1/`"])
    return "\n".join(lines)


def _write_contract_plots(
    result: ContractStage1Result,
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    bar_size: str,
) -> None:
    sym = result.symbol
    tag = f"{sym}_kappa{result.kappa:.1f}"

    _plot_inter_event(sym, frame, out_dir / f"{tag}_inter_event.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dec_path = out_dir / f"{tag}_decile_calibration.png"
    qq_path = out_dir / f"{tag}_rescale_qq.png"

    ax = axes[0]
    ax.bar(["bottom", "top"], [result.decile_bottom_rate, result.decile_top_rate], color=["#4c72b0", "#c44e52"])
    ax.set_ylabel("Realized event rate (OOS)")
    ax.set_title(f"{sym} decile calibration (κ={result.kappa})")
    ax.axhline(result.decile_bottom_rate * DECILE_LIFT_THRESHOLD, ls="--", c="gray", label="1.5× floor")
    ax.legend()

    ax2 = axes[1]
    ax2.text(
        0.1,
        0.5,
        f"KS stat={result.ks_stat:.3f}\nKS p={result.ks_pvalue:.3f}\n"
        f"α/β={result.fit.params.branching_ratio:.3f}\n"
        f"Gate: {'PASS' if result.gate_pass else 'FAIL'}",
        fontsize=11,
        family="monospace",
    )
    ax2.axis("off")
    ax2.set_title("Time-rescaling KS")
    fig.tight_layout()
    fig.savefig(dec_path, dpi=120)
    plt.close(fig)


def _plot_inter_event(sym: str, frame: pd.DataFrame, path: Path) -> None:
    ev = frame[frame["is_event"]]["t_days"].to_numpy()
    gaps = np.diff(ev) if len(ev) > 1 else np.array([])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if len(gaps):
        ax.hist(gaps, bins=min(30, max(5, len(gaps) // 3)), color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Inter-event time (days)")
    ax.set_ylabel("Count")
    ax.set_title(f"{sym} inter-event gaps ({len(ev)} events)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _result_to_dict(r: ContractStage1Result) -> dict[str, object]:
    d = asdict(r)
    d["fit"] = {
        "params": asdict(r.fit.params),
        "log_likelihood": r.fit.log_likelihood,
        "success": r.fit.success,
        "std_errors": r.fit.std_errors,
    }
    return d


if __name__ == "__main__":
    raise SystemExit(main())
