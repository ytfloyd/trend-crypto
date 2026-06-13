"""Generate the medallion_lite tearsheet from the HONEST point-in-time run.

Uses the survivorship-free PIT pipeline (reused from run_medallion_walkforward)
with the flagship params, builds the strategy + BTC daily equity curves, and
renders the repo's standard HTML tearsheet (tearsheet_common_v0). Also prints
the headline stats and writes a daily-returns CSV.

Run: PYTHONPATH=scripts/research:src python scripts/research/k2_atlas/medallion_tearsheet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(HERE), str(ROOT / "scripts" / "research"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_medallion_walkforward as wf  # noqa: E402  (reuse load_pit / config_daily_returns / _daily / vol_target)
from core.metrics import compute_metrics  # noqa: E402
from tearsheet_common_v0 import build_standard_html_tearsheet  # noqa: E402

OUT_DIR = ROOT / "scripts" / "research" / "medallion_lite" / "output"
PARAMS = {"entry_threshold": 0.65, "trailing_stop_pct": 0.15}  # flagship defaults


def _m(daily: pd.Series, lo=None) -> dict:
    d = daily if lo is None else daily[daily.index >= lo]
    return compute_metrics((1.0 + d.dropna()).cumprod())


def main() -> None:
    print("building point-in-time (survivorship-free) medallion equity ...")
    composite_pit, rw, regime = wf.load_pit()
    daily = wf.config_daily_returns(composite_pit, rw, regime, PARAMS)
    daily_vt = wf.vol_target(daily)
    btc = wf._daily(rw["BTC-USD"])

    strat_eq = (1.0 + daily.fillna(0.0)).cumprod().rename("equity")
    btc_eq = (1.0 + btc.reindex(daily.index).fillna(0.0)).cumprod().rename("equity")

    full, oos = _m(daily), _m(daily, "2023-01-01")
    vt_oos = _m(daily_vt, "2023-01-01")
    print(f"  FULL 2021-26  Sortino {full['sortino']:.2f}  Sharpe {full['sharpe']:.2f}  CAGR {full['cagr']:.0%}  DD {full['max_dd']:.0%}")
    print(f"  OOS  2023-26  Sortino {oos['sortino']:.2f}  Sharpe {oos['sharpe']:.2f}  CAGR {oos['cagr']:.0%}  DD {oos['max_dd']:.0%}")
    print(f"  OOS +vol-tgt  Sortino {vt_oos['sortino']:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts": strat_eq.index, "portfolio_equity": strat_eq.values,
                  "daily_ret": daily.reindex(strat_eq.index).values}).to_csv(
        OUT_DIR / "medallion_lite_pit_equity.csv", index=False)

    out_html = OUT_DIR / "medallion_lite_pit_tearsheet.html"
    subtitle = (
        "Point-in-time (survivorship-free) universe · 30bps costs · daily. "
        f"Honest walk-forward OOS Sortino 2.03 (+vol-target 2.33); full-sample shown below. "
        "signal_fn in registry is a simplified daily proxy — these metrics are the flagship "
        "event-driven pipeline. Per-fold OOS Sortino decays 3.49->1.97->1.11 (recent edge weakening)."
    )
    path = build_standard_html_tearsheet(
        out_html=out_html,
        strategy_label="Medallion Lite (point-in-time)",
        strategy_equity=strat_eq,
        benchmark_equity=btc_eq,
        benchmark_label="BTC buy & hold",
        subtitle=subtitle,
        confidential_footer=True,
    )
    print(f"\ntearsheet -> {path}")


if __name__ == "__main__":
    main()
