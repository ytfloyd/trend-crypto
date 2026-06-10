"""Fetch IBKR Flex Web Service Trades reports via ib_insync.

The IBKR Flex Web Service exposes a two-call flow:
    1. Send a request with a (query_id, token) -> returns a ReferenceCode.
    2. Poll with the ReferenceCode until the report is ready, then download.

ib_insync.FlexReport wraps this. We persist the raw XML to disk and parse the
TradeConfirm / Trade rows into a tidy parquet, filtered to the silver complex.

To get a token in TWS / IBKR Client Portal:
    Settings -> Reporting -> Flex Web Service -> Configure -> Generate Token
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


_TRADE_NODES = ("TradeConfirm", "Trade")


def _import_flex_report():
    try:
        from ib_insync import FlexReport  # type: ignore
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "ib_insync is not installed in this venv. "
            "Run: pip install ib_insync"
        ) from exc
    return FlexReport


def _silver_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for silver-complex rows in a Flex Trades dataframe."""
    asset = df.get("assetCategory", pd.Series([""] * len(df))).astype(str).str.upper()
    in_class = asset.isin({"FUT", "FOP"})

    sym = df.get("symbol", pd.Series([""] * len(df))).astype(str).str.upper()
    underlying = df.get("underlyingSymbol", pd.Series([""] * len(df))).astype(str).str.upper()

    has_si_underlying = underlying.str.contains("SI", na=False)
    sym_root_match = sym.str.match(r"^(QI|SO)[FGHJKMNQUVXZ]\d")

    return in_class & (has_si_underlying | sym_root_match)


def _coerce_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce common Flex datetime cols to tz-aware UTC."""
    for col in ("dateTime", "tradeDate", "settleDateTarget", "orderTime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def fetch_flex_trades(
    query_id: int,
    token: str,
    out_path: Path,
    raw_xml_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch a Flex Trades report and persist raw XML + parsed parquet.

    Parameters
    ----------
    query_id : int
        Flex Query ID (silver-trades or all-trades query).
    token : str
        Flex Web Service token from TWS Settings -> Reporting -> Flex.
    out_path : Path
        Destination parquet path for the filtered silver Trades frame.
    raw_xml_path : Path | None
        Where to dump the raw XML. Defaults to alongside out_path.

    Returns
    -------
    pd.DataFrame
        The filtered silver Trades frame (also written to out_path).
    """
    FlexReport = _import_flex_report()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_xml_path = Path(raw_xml_path) if raw_xml_path else out_path.parent / "flex_raw.xml"

    report = FlexReport(token=token, queryId=query_id)
    # ib_insync's FlexReport.download() performs the send + poll loop.
    report.download(token, query_id)
    report.save(str(raw_xml_path))

    # Collect Trades from whichever node the query produced.
    frames = []
    for node_name in _TRADE_NODES:
        try:
            frame = report.df(node_name)
        except Exception:
            frame = None
        if frame is not None and len(frame):
            frames.append(frame)
    if not frames:
        raise RuntimeError(
            "Flex report contained no Trade or TradeConfirm nodes; "
            f"check that query {query_id} is a Trades query."
        )
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = _coerce_datetime(df)

    df = df[_silver_mask(df)].copy()

    if "openCloseIndicator" in df.columns:
        df["open_close"] = df["openCloseIndicator"].astype(str)
    else:
        df["open_close"] = pd.NA

    df.to_parquet(out_path, index=False)
    return df


def _read_env() -> tuple[int, str]:
    """Read IBKR_FLEX_QUERY_ID and IBKR_FLEX_TOKEN; raise a clear error if missing."""
    qid = os.environ.get("IBKR_FLEX_QUERY_ID")
    tok = os.environ.get("IBKR_FLEX_TOKEN")
    if not qid or not tok:
        raise SystemExit(
            "set IBKR_FLEX_QUERY_ID and IBKR_FLEX_TOKEN environment variables "
            "before running the Flex puller. See the docstring for how to "
            "generate a token (TWS -> Settings -> Reporting -> Flex Web Service)."
        )
    try:
        return int(qid), tok
    except ValueError:
        raise SystemExit(f"IBKR_FLEX_QUERY_ID must be an integer, got: {qid!r}")
