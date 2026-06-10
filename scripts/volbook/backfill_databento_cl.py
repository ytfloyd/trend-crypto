#!/usr/bin/env python
"""Deep CL 1-minute backfill from Databento (GLBX.MDP3, ohlcv-1m).

Default mode is a ZERO-COST quote: it sums ``metadata.get_cost`` over each
contract's front+roll window and prints the exact USD figure -- no data is
downloaded.

Hard credit ceiling
-------------------
A billable download only runs with ``--download`` AND a credit ceiling
(``--credit-usd`` or the ``DATABENTO_CREDIT_USD`` env var).  The script:
  * reserves a safety buffer (``max(credit * --safety-frac, --safety-usd)``),
  * skips contracts already present in the lake (``--skip-existing``) so a
    resumed run never re-pays for stored data,
  * selects the largest set of contracts whose summed cost fits the cap,
  * re-checks each contract's cost immediately before its (only) billable call
    and stops the moment the next contract would breach the cap,
  * defers the remainder and writes a manifest so a later run resumes.

Because ``get_cost`` is essentially exact for historical requests and each
contract is checked before download, cumulative spend can never exceed the cap.

Examples
--------
    # Quote only (no spend):
    DATABENTO_API_KEY=... python scripts/volbook/backfill_databento_cl.py --cost-only

    # Download, never spending beyond a $30 credit balance:
    DATABENTO_API_KEY=... python scripts/volbook/backfill_databento_cl.py \
        --download --credit-usd 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from volbook.databento_loader import DatabentoLoader, populated_expiries  # noqa: E402
from volbook.datalake import DEFAULT_LAKE_PATH, MinuteLake  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="201007", help="First CL delivery month YYYYMM (default 201007).")
    p.add_argument("--end", default="202406", help="Last CL delivery month YYYYMM (default 202406, IBKR floor handoff).")
    p.add_argument("--window-days", type=int, default=45, help="Days before last-trade to fetch per contract.")
    p.add_argument("--lake-path", default=str(DEFAULT_LAKE_PATH))
    p.add_argument("--cost-only", action="store_true", help="Print the cost quote and exit (default behaviour).")
    p.add_argument("--download", action="store_true", help="Perform the BILLABLE download + lake upsert.")
    # Hard credit ceiling.
    p.add_argument("--credit-usd", type=float, default=None,
                   help="Available Databento credit (USD). Falls back to DATABENTO_CREDIT_USD env. Required for --download.")
    p.add_argument("--safety-frac", type=float, default=0.10,
                   help="Fraction of credit held back as buffer (default 0.10).")
    p.add_argument("--safety-usd", type=float, default=1.0,
                   help="Minimum absolute buffer in USD (default 1.0). Buffer = max(credit*frac, this).")
    p.add_argument("--order", choices=["newest", "oldest"], default="newest",
                   help="Which contracts to prioritise when credit is limited (default newest).")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                   help="Skip contracts already populated in the lake (default on; avoids paying twice).")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--no-clamp", action="store_true", help="Do not clamp windows to the dataset's available start.")
    p.add_argument("--manifest", default=None, help="Path to write the downloaded/deferred manifest (default alongside lake).")
    # Optional secondary acknowledgement of the planned spend.
    p.add_argument("--confirm-usd", type=float, default=None,
                   help="Optional: must match the planned spend within --confirm-tol.")
    p.add_argument("--confirm-tol", type=float, default=0.01, help="Relative tolerance for --confirm-usd (default 0.01 = 1 pct).")
    return p.parse_args()


def _resolve_credit(args: argparse.Namespace) -> float | None:
    if args.credit_usd is not None:
        return args.credit_usd
    env = os.environ.get("DATABENTO_CREDIT_USD")
    if env:
        try:
            return float(env)
        except ValueError:
            return None
    return None


def _print_quote(quote: dict) -> None:
    by_year: dict[str, float] = defaultdict(float)
    skipped = 0
    for row in quote["breakdown"]:
        if row.get("skipped"):
            skipped += 1
            continue
        by_year[row["expiry"][:4]] += row["cost_usd"]
    print(json.dumps({
        "dataset": quote["dataset"],
        "schema": quote["schema"],
        "rate_usd_per_gb": quote["rate_usd_per_gb"],
        "contracts": quote["contracts"],
        "contracts_skipped_before_dataset_start": skipped,
        "total_cost_usd": round(quote["total_cost_usd"], 4),
        "cost_by_year_usd": {y: round(v, 4) for y, v in sorted(by_year.items())},
    }, indent=2))
    print(f"\nTOTAL QUOTED COST: ${quote['total_cost_usd']:.2f} "
          f"({quote['contracts']} contracts, {quote['schema']} @ ${quote['rate_usd_per_gb']}/GB)")
    print("Compare against your Databento credit balance in the portal before approving --download.")


def _do_quote(loader: DatabentoLoader, args: argparse.Namespace) -> int:
    quote = loader.estimate_cost(
        args.start, args.end,
        window_days=args.window_days,
        clamp_to_dataset=not args.no_clamp,
    )
    _print_quote(quote)
    return 0


def _do_download(loader: DatabentoLoader, args: argparse.Namespace) -> int:
    credit = _resolve_credit(args)
    if credit is None:
        print("REFUSING: --download requires --credit-usd (or DATABENTO_CREDIT_USD env) so a hard "
              "spend ceiling can be enforced. No credit value supplied.")
        return 2
    buffer = max(credit * args.safety_frac, args.safety_usd)
    cap = credit - buffer
    if cap <= 0:
        print(f"REFUSING: credit ${credit:.2f} minus buffer ${buffer:.2f} leaves no spendable budget.")
        return 2

    lake = MinuteLake(args.lake_path)
    lake.connect()
    try:
        exclude: list[str] = []
        if args.skip_existing:
            exclude = sorted(populated_expiries(lake))
        plan = loader.plan_within_budget(
            args.start, args.end,
            cap_usd=cap,
            window_days=args.window_days,
            order=args.order,
            exclude_expiries=exclude,
            clamp_to_dataset=not args.no_clamp,
        )
        rate = plan["rate_usd_per_gb"]
        print(json.dumps({
            "credit_usd": credit,
            "safety_buffer_usd": round(buffer, 4),
            "spend_cap_usd": round(cap, 4),
            "order": args.order,
            "skip_existing_excluded": len(exclude),
            "selected_contracts": len(plan["selected"]),
            "deferred_contracts": len(plan["deferred"]),
            "planned_spend_usd": round(plan["planned_cost_usd"], 4),
            "deferred_cost_usd": round(plan["deferred_cost_usd"], 4),
            "full_job_cost_usd": round(plan["full_cost_usd"], 4),
            "rate_usd_per_gb": rate,
        }, indent=2))

        if not plan["selected"]:
            print("\nNothing to download: no contract fits under the cap (or all already present).")
            return 0

        if args.confirm_usd is not None:
            tol = abs(plan["planned_cost_usd"]) * args.confirm_tol + 1e-6
            if abs(args.confirm_usd - plan["planned_cost_usd"]) > tol:
                print(f"\nREFUSING: --confirm-usd {args.confirm_usd:.2f} != planned "
                      f"${plan['planned_cost_usd']:.2f} (tol {args.confirm_tol}).")
                return 2

        print(f"\nStarting BILLABLE download (cap ${cap:.2f}) -> {args.lake_path}")
        spent = 0.0
        spent_actual = 0.0
        downloaded: list[str] = []
        deferred: list[str] = [r["expiry"] for r in plan["deferred"]]
        clamp_start = loader._clamp_start(not args.no_clamp)
        for row in plan["selected"]:
            expiry = row["expiry"]
            # Runtime hard stop: re-check this contract's exact cost first.
            windows = loader.windows(expiry, expiry, window_days=args.window_days)
            cost, eff_start = loader.contract_cost(windows[0], clamp_start=clamp_start)
            if eff_start is None:
                continue
            if spent + cost > cap:
                print(f"  STOP before {expiry}: would reach ${spent + cost:.2f} > cap ${cap:.2f}")
                deferred.append(expiry)
                continue
            result = loader.backfill_contract(
                lake, expiry, window_days=args.window_days, unit_price_usd_per_gb=rate,
            )
            spent += cost
            spent_actual += result["actual_cost_usd"]
            downloaded.append(expiry)
            print(f"  {expiry} ({row['raw_symbol']}): {result['rows']} bars | "
                  f"est ${cost:.3f} act ${result['actual_cost_usd']:.3f} | cum est ${spent:.2f}")
            # Defensive: if actual bytes ever overshoot the cap, stop now.
            if spent_actual > cap:
                print(f"  STOP: actual spend ${spent_actual:.2f} reached cap ${cap:.2f}")
                break
    finally:
        lake.close()

    manifest_path = Path(args.manifest) if args.manifest else (
        Path(args.lake_path).with_name(f"databento_cl_backfill_manifest_"
                                       f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    )
    manifest = {
        "dataset": plan["dataset"],
        "schema": plan["schema"],
        "credit_usd": credit,
        "spend_cap_usd": cap,
        "downloaded_expiries": downloaded,
        "deferred_expiries": sorted(set(deferred)),
        "spent_estimate_usd": round(spent, 4),
        "spent_actual_usd": round(spent_actual, 4),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone: {len(downloaded)} contracts downloaded, {len(set(deferred))} deferred. "
          f"Est spend ${spent:.2f} (actual ${spent_actual:.2f}); cap was ${cap:.2f}.")
    print(f"Manifest: {manifest_path}")
    if deferred:
        print("Deferred contracts remain; raise --credit-usd and re-run (--skip-existing resumes).")
    return 0


def main() -> int:
    args = parse_args()
    loader = DatabentoLoader()  # uses DATABENTO_API_KEY from env
    if not args.download:
        return _do_quote(loader, args)
    return _do_download(loader, args)


if __name__ == "__main__":
    raise SystemExit(main())
