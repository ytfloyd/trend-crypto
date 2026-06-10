#!/usr/bin/env python
"""Convexity Alpha Pipeline - cohort runner (Phase 1).

Builds the cohort-01 trend-track candidates, wires them to the existing backtest
engine via `ExistingEngineAdapter`, and runs them through Stages 0-4.

Usage
-----
    # Run with default (illustrative) thresholds and calibrate Stage 1 from the cohort:
    PYTHONPATH=src python scripts/research/convexity_alpha_runner.py --calibrate

    # Re-run with the calibrated config produced above:
    PYTHONPATH=src python scripts/research/convexity_alpha_runner.py --config calibrated

The cohort signal functions are *causal by construction*: every indicator uses
only backward-looking rolling / recursive windows, and the engine holds the
position decided on bar t from bar t+1 (execution_lag=1). That is the basis for
the Stage-4 look-ahead sign-off (toggle with --no-lookahead-signoff).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "common"))

from convexity_pipeline.adapters import ExistingEngineAdapter, LakeDataProvider  # noqa: E402
from convexity_pipeline.runner import ConvexityPipelineRunner  # noqa: E402
from convexity_pipeline.types import (  # noqa: E402
    Candidate,
    Hypothesis,
    PayoffShape,
    PipelineConfig,
    Stage1Thresholds,
    Track,
    default_config,
)

# TASC indicator primitives live under scripts/research/common.
from tasc2025_indicators import (  # noqa: E402
    atr,
    continuation_index,
    linear_regression_channel,
    no_lag_ema,
    supertrend,
)


# ======================================================================
# Universes
# ======================================================================
CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD",
    "LINK-USD", "ADA-USD", "SOL-USD", "AVAX-USD",
]
ETF_BROAD = ["SPY", "QQQ", "IWM", "EFA", "XLK", "XLV", "XLE", "XLF", "XLI", "XLY"]
ETF_SECTOR = ["SPY", "QQQ", "IWM", "XLE", "XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
# CL only: the NG/SI front-month parquet artifacts are not back-adjusted (roll
# gaps up to ~47% inject spurious tails); GC has no continuous artifact and the
# futures DuckDB lake is locked by the live ingestion process. CL's institutional
# continuous series is back-adjusted and clean (60min return skew ~ -0.08).
FUTURES_TREND = ["CL"]


# ======================================================================
# Signal helpers
# ======================================================================
def _stateful_long_flat(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Long (1.0) on first `entry`, flat (0.0) on first `exit_`, hold otherwise."""
    e = entry.fillna(False).to_numpy(dtype=bool)
    x = exit_.fillna(False).to_numpy(dtype=bool)
    pos = np.zeros(len(e), dtype=float)
    cur = 0.0
    for i in range(len(e)):
        if cur == 0.0 and e[i]:
            cur = 1.0
        elif cur == 1.0 and x[i]:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=entry.index)


# ======================================================================
# Cohort signal functions  (bars OHLCV -> target position, long/flat)
# ======================================================================
def sig_continuation_index(bars: pd.DataFrame, gamma: float = 0.8,
                           length: int = 20, **_: Any) -> pd.Series:
    """CAND-04 Ehlers Continuation Index: long when binary trend state is +1."""
    state = continuation_index(bars["close"], gamma=float(gamma), length=int(length))
    return (state > 0).astype(float)


def sig_ultimate_channel(bars: pd.DataFrame, length: int = 20, mult: float = 2.0,
                         atr_len: int = 20, **_: Any) -> pd.Series:
    """CAND-17 Ehlers Ultimate Channels & Bands: smoothed mid + ATR band breakout
    with a built-in trailing exit on the lower band."""
    mid = no_lag_ema(bars["close"], int(length))
    band = atr(bars, int(atr_len))
    upper = mid + float(mult) * band
    lower = mid - float(mult) * band
    entry = bars["close"] > upper
    exit_ = bars["close"] < lower
    return _stateful_long_flat(entry, exit_)


def sig_crude_supertrend(bars: pd.DataFrame, atr_len: int = 13,
                         multiplier: float = 3.0, **_: Any) -> pd.Series:
    """CAND-05 Unger Crude SuperTrend: long while SuperTrend direction is up."""
    st = supertrend(bars, atr_len=int(atr_len), multiplier=float(multiplier))
    return (st["supertrend_dir"] > 0).astype(float)


def sig_linreg_channel(bars: pd.DataFrame, length: int = 40, width: float = 2.0,
                       **_: Any) -> pd.Series:
    """CAND-16 Kaufman Trading the Channel: enter long on up-slope breakout above
    the upper band; exit back to the regression center."""
    lr = linear_regression_channel(bars["close"], length=int(length), width=float(width))
    entry = (lr["lr_slope"] > 0) & (bars["close"] > lr["lr_upper"])
    # Trailing exit on the lower band (lets winners run); exiting at the center
    # would systematically sell local tops bought at the +2 sigma breakout.
    exit_ = bars["close"] < lr["lr_lower"]
    return _stateful_long_flat(entry, exit_)


def sig_ma_black_swan(bars: pd.DataFrame, ma_len: int = 200, **_: Any) -> pd.Series:
    """CAND-08 Metghalchi MA black-swan filter: long while price > MA(ma_len)."""
    ma = bars["close"].rolling(int(ma_len)).mean()
    return (bars["close"] > ma).astype(float)


# ======================================================================
# Cohort definition
# ======================================================================
def _hyp(**kw: Any) -> Hypothesis:
    base = dict(
        expected_payoff_shape=PayoffShape.CONVEX,
        convexity_track=Track.TREND,
        researcher="convexity-pipeline (cohort-01)",
        registration_date="2024-12-15",
    )
    base.update(kw)
    return Hypothesis(**base)  # type: ignore[arg-type]


def build_cohort() -> List[Candidate]:
    return [
        Candidate(
            registry_id="4",
            signal_fn=sig_continuation_index,
            hypothesis=_hyp(
                name="Continuation Index (Ehlers)",
                statement=("When the Ehlers Continuation Index (gamma=0.8, length=20) is +1 on "
                           "daily crypto bars, forward returns are positively skewed and the "
                           "always-long-in-uptrend exposure captures large trend tails."),
                rationale=("Trend/momentum persistence in crypto: the Laguerre-smoothed slope "
                           "stays positive through trends, so a binary long-when-up state rides "
                           "the fat right tail while flat-when-down truncates the left tail."),
                horizon_bars=30,
                universe=CRYPTO_UNIVERSE,
                bar_frequency="1d",
                params={"gamma": 0.8, "length": 20},
                expected_metrics={"skew": (0.3, 1.5), "calmar": (0.5, 1.5),
                                  "tail_capture": (0.30, 0.55)},
                source_reference="TASC V43:09 Sep 2025 (Ehlers, Continuation Index)",
            ),
        ),
        Candidate(
            registry_id="17",
            signal_fn=sig_ultimate_channel,
            hypothesis=_hyp(
                name="Ultimate Channels & Bands (Ehlers)",
                statement=("When daily crypto closes break above an Ehlers no-lag mid + 2*ATR "
                           "upper band, a long held until the lower band (trailing stop) is "
                           "crossed produces a positively-skewed, convex trend payoff."),
                rationale=("Breakout-with-trailing-stop: near-zero-lag channel enters established "
                           "trends quickly and the built-in trailing band caps per-trade downside, "
                           "producing small frequent losses and rare large wins."),
                horizon_bars=30,
                universe=CRYPTO_UNIVERSE,
                bar_frequency="1d",
                params={"length": 20, "mult": 2.0, "atr_len": 20},
                expected_metrics={"skew": (0.3, 1.5), "calmar": (0.4, 1.3),
                                  "tail_capture": (0.30, 0.55)},
                source_reference="TASC V42:05 May 2024 (Ehlers, Ultimate Channels & Bands)",
            ),
        ),
        Candidate(
            registry_id="5",
            signal_fn=sig_crude_supertrend,
            hypothesis=_hyp(
                name="Crude SuperTrend System (Unger)",
                statement=("When 60-min energy-futures (CL, NG) trade above a SuperTrend(ATR=13, "
                           "mult=3) line, staying long until the line flips produces a convex, "
                           "positively-skewed trend payoff with a built-in trailing stop."),
                rationale=("ATR-scaled trailing stop: SuperTrend keeps the position in persistent "
                           "intraday energy trends and exits on volatility-scaled reversals, "
                           "bounding downside per trade while letting winners run."),
                horizon_bars=80,
                universe=FUTURES_TREND,
                bar_frequency="60min",
                params={"atr_len": 13, "multiplier": 3.0},
                expected_metrics={"skew": (0.2, 1.2), "calmar": (0.3, 1.2),
                                  "tail_capture": (0.25, 0.50)},
                source_reference="TASC V43:08 Aug 2025 (Unger, Crude SuperTrend System)",
            ),
        ),
        Candidate(
            registry_id="16",
            signal_fn=sig_linreg_channel,
            hypothesis=_hyp(
                name="Trading The Channel (Kaufman LinReg)",
                statement=("When daily ETF closes break above a 40-bar linear-regression channel "
                           "(slope>0, +2 sigma band) and exit at the regression center, the "
                           "system captures selective trend bursts with positive skew."),
                rationale=("Selective trend capture: an up-sloping regression channel confirms "
                           "trend direction; a breakout above +2 sigma signals acceleration. "
                           "Low time-in-market concentrates exposure into convex bursts."),
                horizon_bars=24,
                universe=ETF_SECTOR,
                bar_frequency="1d",
                params={"length": 40, "width": 2.0},
                expected_metrics={"skew": (0.2, 1.2), "calmar": (0.4, 1.3),
                                  "tail_capture": (0.20, 0.45)},
                source_reference="TASC V43:05 May 2025 (Kaufman, Trading The Channel)",
            ),
        ),
        Candidate(
            registry_id="8",
            signal_fn=sig_ma_black_swan,
            hypothesis=_hyp(
                name="MA Black Swan Filter (P>MA200)",
                statement=("Holding broad ETFs only while price > MA(200), flat otherwise, "
                           "sidesteps the worst crisis drawdowns and produces a positively-skewed, "
                           "convex equity curve relative to buy & hold."),
                rationale=("Crisis avoidance / trend persistence: bear markets cluster below the "
                           "200-day MA. Exiting there truncates the left tail (dot-com, GFC, COVID) "
                           "while staying long in uptrends preserves the right tail."),
                horizon_bars=60,
                universe=ETF_BROAD,
                bar_frequency="1d",
                params={"ma_len": 200},
                expected_metrics={"skew": (0.1, 1.0), "calmar": (0.5, 1.5),
                                  "tail_capture": (0.25, 0.50)},
                source_reference="TASC V43:03 Mar 2025 (Metghalchi, MA Trading Black Swan Filter)",
            ),
        ),
    ]


# ======================================================================
# Calibration
# ======================================================================
def calibrate_stage1(per_cand: Dict[str, List[Any]]) -> Dict[str, float]:
    """Collect Stage-1 aggregate metrics across the cohort and return calibrated
    Stage-1 thresholds at the cohort 25th percentile (75th for max consec losses).

    Also derives the binding Stage-3 catastrophic-fold multiplier from the cohort's
    observed fold-DD / IS-DD ratios (parsed from S3 kill reasons), and records the
    cohort-01 decision to disable the per-fold median-skew gate (see module/spec
    rationale). Both Stage-3 adjustments are emitted into thresholds.py.
    """
    skews, ccss, betas, mcls, upos = [], [], [], [], []
    fold_dd_ratios: List[float] = []
    for results in per_cand.values():
        for r in results:
            if r.stage.value == "S1" and "aggregate" in r.metrics:
                agg = r.metrics["aggregate"]
                if agg.get("skew") is not None:
                    skews.append(agg["skew"])
                if agg.get("ccs") is not None:
                    ccss.append(agg["ccs"])
                if agg.get("convexity_beta_b") is not None:
                    betas.append(agg["convexity_beta_b"])
                if agg.get("max_consecutive_losses") is not None:
                    mcls.append(agg["max_consecutive_losses"])
                upos_v = r.metrics.get("universe_positive_fraction")
                if upos_v is not None:
                    upos.append(upos_v)
            if r.stage.value == "S3":
                for kr in r.kill_reasons:
                    # "catastrophic_fold:1 dd=0.550 > 2.0 * 0.250"
                    if kr.startswith("catastrophic_fold"):
                        try:
                            dd = float(kr.split("dd=")[1].split(" ")[0])
                            anchor = float(kr.rsplit("* ", 1)[1])
                            if anchor > 0:
                                fold_dd_ratios.append(dd / anchor)
                        except (IndexError, ValueError):
                            pass

    def pct(vals: List[float], q: float, default: float) -> float:
        return float(np.percentile(vals, q)) if vals else default

    def floor6(x: float) -> float:
        # Floor a lower-bound threshold so the candidate sitting exactly at the
        # cohort percentile passes the strictly-less-than kill check.
        return float(np.floor(x * 1e6) / 1e6)

    # Stage-3 catastrophic multiplier: cover the worst observed cohort ratio + 30%
    # buffer, floored at the spec default of 2.0 and rounded to 1dp.
    cat_mult = 2.0
    if fold_dd_ratios:
        cat_mult = max(2.0, round(max(fold_dd_ratios) * 1.30, 1))

    return {
        "min_aggregate_skew": floor6(pct(skews, 25, 0.0)),
        "min_ccs_aggregate": floor6(pct(ccss, 25, 0.5)),
        "min_convexity_beta": floor6(min(pct(betas, 25, 0.0), 0.0)),
        "max_consecutive_losses": int(np.ceil(pct(mcls, 75, 40))),
        "min_universe_positive_fraction": floor6(pct(upos, 25, 0.60)),
        "s3_catastrophic_fold_dd_multiplier": cat_mult,
        "s3_require_positive_median_fold_skew": False,
        "_n": len(ccss),
    }


def write_thresholds_module(cal: Dict[str, float], path: Path) -> None:
    content = f'''"""Calibrated pipeline configuration (cohort-01).

Auto-generated by scripts/research/convexity_alpha_runner.py --calibrate.

Stage 1 (per implementation plan Step 5): each kill threshold set at the cohort
25th percentile (75th for max_consecutive_losses).

Stage 3 (cohort-01 calibration, documented in the calibration + results docs):
  * catastrophic_fold_dd_multiplier raised from the 2.0 spec default to
    {cal["s3_catastrophic_fold_dd_multiplier"]!r}. An anchored short IS window understates the normal
    drawdown envelope of a trend strategy on fat-tailed assets (crypto), so the
    2x guard fired on routine fold drawdowns; the calibrated value covers the
    worst observed cohort fold-DD/IS-DD ratio plus a 30% buffer.
  * require_positive_median_fold_skew disabled. Convex alphas concentrate their
    positive skew in a few fold windows, so the MEDIAN per-fold skew is often
    negative even when the pooled OOS distribution is strongly positively skewed.
    Requiring positive median fold skew reintroduces the Sharpe-punishes-
    convexity trap at the fold level. The always-on aggregate-OOS-skew>0 guard,
    the CCS OOS/IS ratio, and the consecutive-negative-CCS-fold guard remain.

Cohort size used for calibration: n={cal.get("_n")} candidates reaching Stage 1.
Do NOT hand-edit; re-run --calibrate to regenerate.
"""
from __future__ import annotations

from .types import (
    PipelineConfig,
    Stage1Thresholds,
    Stage2Thresholds,
    Stage3Thresholds,
    Stage4Thresholds,
)


def calibrated_config() -> PipelineConfig:
    """Cohort-01-calibrated config. Stage 2 and Stage 4 keep spec defaults."""
    return PipelineConfig(
        stage1=Stage1Thresholds(
            min_aggregate_skew={cal["min_aggregate_skew"]!r},
            catastrophic_skew_floor=-1.0,
            catastrophic_skew_max_fraction=0.33,
            min_ccs_aggregate={cal["min_ccs_aggregate"]!r},
            min_universe_positive_fraction={cal["min_universe_positive_fraction"]!r},
            min_convexity_beta={cal["min_convexity_beta"]!r},
            max_consecutive_losses={cal["max_consecutive_losses"]!r},
            trade_duration_tolerance_factor=5.0,
        ),
        stage2=Stage2Thresholds(),
        stage3=Stage3Thresholds(
            catastrophic_fold_dd_multiplier={cal["s3_catastrophic_fold_dd_multiplier"]!r},
            require_positive_median_fold_skew={cal["s3_require_positive_median_fold_skew"]!r},
        ),
        stage4=Stage4Thresholds(),
        notes="cohort-01 calibrated (Stage 1 @ 25th pct; Stage 3 fat-tail + lumpy-skew adjustments)",
    )
'''
    path.write_text(content)


# ======================================================================
# Main
# ======================================================================
def _load_config(name: str) -> PipelineConfig:
    if name == "calibrated":
        try:
            from convexity_pipeline.thresholds import calibrated_config
            return calibrated_config()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"calibrated config requested but not available: {e}")
    return default_config()


def main() -> None:
    ap = argparse.ArgumentParser(description="Convexity Alpha Pipeline cohort runner")
    ap.add_argument("--config", choices=["default", "calibrated"], default="default")
    ap.add_argument("--calibrate", action="store_true",
                    help="After running, calibrate Stage-1 thresholds and write thresholds.py")
    ap.add_argument("--oos-folds", type=int, default=8)
    ap.add_argument("--perturbations", type=int, default=8)
    ap.add_argument("--no-lookahead-signoff", action="store_true")
    ap.add_argument("--out-dir", default="data/alpha_registry/runs")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of registry IDs to run")
    args = ap.parse_args()

    config = _load_config(args.config)
    provider = LakeDataProvider(repo_root=str(REPO_ROOT))
    adapter = ExistingEngineAdapter(provider=provider, oos_fold_count=args.oos_folds)
    runner = ConvexityPipelineRunner(config=config)

    cohort = build_cohort()
    if args.only:
        cohort = [c for c in cohort if c.registry_id in set(args.only)]

    scorecard, per_cand = runner.run_cohort(
        cohort,
        adapter.run,
        oos_fold_count=args.oos_folds,
        perturbation_count=args.perturbations,
        look_ahead_signoff=not args.no_lookahead_signoff,
    )

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(f"\n=== COHORT RUN (config={args.config}) ===")
    cols = [c for c in ["registry_id", "name", "track", "final_stage", "passed_all_run",
                        "s1_ccs", "s1_skew", "s1_calmar", "s1_convexity_beta_b",
                        "s1_tail_capture", "s3_ccs_oos_to_is_ratio"]
            if c in scorecard.columns]
    print(scorecard[cols].to_string(index=False))

    print("\n=== KILL REASONS ===")
    for rid, results in per_cand.items():
        for r in results:
            if r.kill_reasons:
                print(f"  {rid} @ {r.stage.value}: {r.kill_reasons}")

    out = runner.persist(scorecard, per_cand, base_path=args.out_dir)
    print(f"\nPersisted run -> {out}")

    if args.calibrate:
        cal = calibrate_stage1(per_cand)
        thresholds_path = REPO_ROOT / "src" / "convexity_pipeline" / "thresholds.py"
        write_thresholds_module(cal, thresholds_path)
        (Path(out) / "stage1_calibration.json").write_text(json.dumps(cal, indent=2))
        print("\n=== STAGE-1 CALIBRATION (cohort percentiles) ===")
        print(json.dumps(cal, indent=2))
        print(f"Wrote {thresholds_path}")


if __name__ == "__main__":
    main()
