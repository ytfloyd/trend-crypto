#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from volbook.continuous import FrontMonthGuard
from volbook.datalake import DEFAULT_LAKE_PATH, MinuteLake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dated-contract NG continuous research artifact and write "
            "front-month coverage diagnostics."
        )
    )
    parser.add_argument("--lake_path", default=str(DEFAULT_LAKE_PATH))
    parser.add_argument("--out_dir", default="artifacts/research/ng_institutional_continuous")
    parser.add_argument("--roll_days_before_expiry", type=int, default=0)
    return parser.parse_args()


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _coverage_summary(coverage: pd.DataFrame) -> dict[str, object]:
    if coverage.empty:
        return {"status": "not_checked", "sessions_checked": 0, "invalid_session_count": 0}
    invalid = coverage[coverage["front_month_guard_status"] == "invalid"]
    first_invalid = invalid.iloc[0].to_dict() if not invalid.empty else {}
    return {
        "status": "valid" if invalid.empty else "invalid",
        "sessions_checked": int(len(coverage)),
        "invalid_session_count": int(len(invalid)),
        "first_invalid": _jsonable(first_invalid),
    }


def write_figures(bars: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax_price = plt.subplots(figsize=(14, 6))
    ax_price.plot(bars["ts"], bars["c"], color="#1f77b4", linewidth=1.0, label="stitched close")
    ax_price.set_title("NG dated-contract stitch")
    ax_price.set_ylabel("close")
    ax_price.grid(True, alpha=0.25)

    ax_flag = ax_price.twinx()
    valid = bars["front_month_valid"].astype(float) if "front_month_valid" in bars else pd.Series(1.0, index=bars.index)
    ax_flag.fill_between(bars["ts"], 0.0, valid, color="#2ca02c", alpha=0.12, label="front-month valid")
    ax_flag.set_ylim(-0.05, 1.05)
    ax_flag.set_ylabel("front-month valid")

    lines, labels = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_flag.get_legend_handles_labels()
    ax_price.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_ng_stitched_timeseries.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lake = MinuteLake(args.lake_path)
    guard = FrontMonthGuard(symbols=("NG",), max_curve_position=2, on_missing="mark")
    try:
        stitched = lake.stitch_continuous_series(
            "NG",
            roll_days_before_expiry=args.roll_days_before_expiry,
            front_month_guard=guard,
        )
        coverage = lake.front_month_coverage("NG", front_month_guard=guard)
        source_expiries = [
            row[0]
            for row in lake.connect()
            .execute(
                """
                SELECT DISTINCT expiry
                FROM bars_1m
                WHERE symbol = 'NG' AND expiry != 'continuous'
                ORDER BY expiry
                """
            )
            .fetchall()
        ]
    finally:
        lake.close()

    bars = stitched.to_pandas()
    if bars.empty:
        raise RuntimeError("No NG dated rows were available to stitch.")

    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars["symbol"] = "NG"
    bars["active_expiry"] = bars["expiry"].astype(str)
    invalid_dates = set()
    if not coverage.empty and "front_month_guard_status" in coverage:
        invalid_dates = {
            str(value)
            for value in coverage.loc[
                coverage["front_month_guard_status"] == "invalid",
                "session_date",
            ].to_list()
        }
    bars["front_month_valid"] = ~bars["ts"].dt.date.astype(str).isin(invalid_dates)
    bars = bars[
        [
            "ts",
            "symbol",
            "expiry",
            "active_expiry",
            "o",
            "h",
            "l",
            "c",
            "v",
            "front_month_valid",
        ]
    ].sort_values("ts")

    bars.to_parquet(out_dir / "bars_1m.parquet", index=False)
    coverage.to_csv(out_dir / "front_month_coverage.csv", index=False)

    valid_rows = int(bars["front_month_valid"].sum())
    metadata = {
        "symbol": "NG",
        "construction": "dated_contract_stitch_with_front_month_guard",
        "lake_path": str(args.lake_path),
        "roll_days_before_expiry": int(args.roll_days_before_expiry),
        "source_expiries": source_expiries,
        "row_count": int(len(bars)),
        "valid_front_month_row_count": valid_rows,
        "start_ts": bars["ts"].min().isoformat(),
        "end_ts": bars["ts"].max().isoformat(),
        "selected_expiries": sorted(bars["active_expiry"].dropna().astype(str).unique()),
        "front_month_guard": {
            "enabled": guard.enabled,
            "symbols": list(guard.symbols),
            "max_curve_position": guard.max_curve_position,
            "on_missing": guard.on_missing,
        },
        "front_month_coverage": _coverage_summary(coverage),
        "warning": (
            "Use only rows where front_month_valid is true. If this count is zero, "
            "the lake does not contain the near/front NG dated contracts needed "
            "for a tradable institutional continuous contract."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_figures(bars, out_dir)

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
