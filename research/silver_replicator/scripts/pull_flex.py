"""CLI: pull a silver-filtered IBKR Flex Trades report.

Requires two environment variables:
    IBKR_FLEX_QUERY_ID  -- numeric query id of a Trades Flex query
    IBKR_FLEX_TOKEN     -- Flex Web Service token

To obtain a token, open TWS or IBKR Client Portal and navigate:
    Settings -> Reporting -> Flex Web Service -> Configure -> Generate Token

Outputs:
    artifacts/flex_raw.xml         -- raw XML returned by IBKR
    artifacts/flex_trades.parquet  -- silver-filtered Trades DataFrame
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_ROOT))

from src.flex_puller import _read_env, fetch_flex_trades  # noqa: E402


def main() -> int:
    query_id, token = _read_env()  # exits cleanly if env vars are missing
    artifacts = _ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    out_path = artifacts / "flex_trades.parquet"
    raw_path = artifacts / "flex_raw.xml"

    df = fetch_flex_trades(query_id, token, out_path, raw_xml_path=raw_path)
    if df.empty:
        print("No silver trades returned by the Flex query.")
        return 0

    dt_col = next((c for c in ("dateTime", "tradeDate") if c in df.columns), None)
    sym_col = "symbol" if "symbol" in df.columns else None

    print(f"Wrote {len(df):,} silver fills -> {out_path}")
    if dt_col:
        print(f"First fill : {df[dt_col].min()}")
        print(f"Last fill  : {df[dt_col].max()}")
    if sym_col and dt_col:
        contract_days = (
            df.assign(_d=df[dt_col].dt.tz_convert("UTC").dt.date)
              .groupby(sym_col)["_d"].nunique()
              .sort_values(ascending=False)
        )
        print("\nContract-day counts (top 15):")
        print(contract_days.head(15).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
