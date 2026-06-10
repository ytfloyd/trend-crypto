"""Build a silver-focused ledger from an IBKR activity-report CSV.

The IBKR activity report is a wide, section-tagged CSV: the first column is
the section name (e.g. "Trade Summary", "Performance by Underlying",
"Historical Performance"), the second column is the row kind ("Header",
"Data", "MetaInfo"). Within each section, "Header" rows define columns for
the following "Data" rows.

We extract the silver complex only:
  - COMEX MINY SILVER   (QI* futures)
  - NYMEX SILVER INDEX  (SO* futures-options on full-size SI)

The Trade Summary rows are AGGREGATES over the whole reporting window, not
fills. Use the Flex puller for per-fill granularity.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# IBKR monthly codes for futures and options.
_MONTH_CODE = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
_MONTH_NAME = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_SILVER_UNDERLYINGS = {"COMEX MINY SILVER", "NYMEX SILVER INDEX"}


def _read_sections(csv_path: Path) -> Dict[str, List[List[str]]]:
    """Read the activity-report CSV and bucket rows by section name."""
    sections: Dict[str, List[List[str]]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            section = row[0]
            sections.setdefault(section, []).append(row)
    return sections


def _section_frames(section_rows: List[List[str]]) -> List[pd.DataFrame]:
    """Within one section, pair each Header row with the following Data rows.

    A section may carry multiple Header blocks (e.g. Historical Performance
    has account-summary header then a Month/AccountReturn header).
    """
    frames: List[pd.DataFrame] = []
    current_cols: Optional[List[str]] = None
    bucket: List[List[str]] = []

    def _flush() -> None:
        if current_cols is None or not bucket:
            return
        width = len(current_cols)
        rows = []
        for r in bucket:
            payload = r[2:]
            if len(payload) < width:
                payload = payload + [""] * (width - len(payload))
            elif len(payload) > width:
                payload = payload[:width]
            rows.append(payload)
        frames.append(pd.DataFrame(rows, columns=current_cols))

    for row in section_rows:
        kind = row[1] if len(row) > 1 else ""
        if kind == "Header":
            _flush()
            current_cols = row[2:]
            bucket = []
        elif kind == "Data":
            bucket.append(row)
        # ignore MetaInfo
    _flush()
    return frames


def _parse_expiry_from_symbol(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (expiry_code, expiry_yyyymm) parsed from an IBKR future symbol.

    Examples
    --------
    QIH6   -> ("H6",  "202603")
    SOK6   -> ("K6",  "202605")
    QIZ5   -> ("Z5",  "202512")
    """
    if not symbol:
        return None, None
    # Remove any option strike suffix after a space.
    core = symbol.split(" ", 1)[0]
    m = re.search(r"([FGHJKMNQUVXZ])(\d)$", core)
    if not m:
        return None, None
    code = m.group(0)
    month = _MONTH_CODE[m.group(1)]
    year_digit = int(m.group(2))
    # IBKR single-digit year: map to the closest decade around the report.
    # Reports here span 2025-2026; treat 5 -> 2025, 6 -> 2026, etc.
    year = 2020 + year_digit
    return code, f"{year:04d}{month:02d}"


def _parse_expiry_from_description(desc: str) -> Optional[str]:
    """Parse 'SI MAR26 ...' or 'QI MAY26' description into YYYYMM."""
    if not desc:
        return None
    m = re.search(r"\b([A-Z]{3})(\d{2})\b", desc)
    if not m:
        return None
    mon = _MONTH_NAME.get(m.group(1))
    if mon is None:
        return None
    year = 2000 + int(m.group(2))
    return f"{year:04d}{mon:02d}"


def _parse_option_fields(symbol: str, description: str) -> Tuple[Optional[float], Optional[str]]:
    """Extract (strike, opt_type) for option symbols like 'SOH6 C10225'.

    The strike encoding is decimal-shifted (e.g. C10225 -> 102.25). Prefer the
    Description strike when available (it's already decimalised).
    """
    if not symbol:
        return None, None
    parts = symbol.split(" ", 1)
    if len(parts) != 2:
        return None, None
    suffix = parts[1].strip()
    m = re.match(r"^([CP])(\d+(?:\.\d+)?)$", suffix)
    if not m:
        return None, None
    opt_type = m.group(1)
    # Try Description first.
    if description:
        dm = re.search(r"([\d.]+)\s+[CP]\b", description)
        if dm:
            try:
                return float(dm.group(1)), opt_type
            except ValueError:
                pass
    raw = m.group(2)
    if "." in raw:
        return float(raw), opt_type
    # Heuristic: IBKR encodes strike * 100 for SI options (e.g. 10225 -> 102.25).
    val = float(raw)
    if val > 1000:
        val = val / 100.0
    return val, opt_type


def _classify_instrument(financial_instrument: str) -> str:
    fi = (financial_instrument or "").lower()
    if "option" in fi:
        return "future_option"
    return "future"


def _to_float(x: object) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in ("", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _build_trades_aggregate(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine Trade Summary frames and filter to silver rows."""
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Numeric coercions on the standard Trade Summary numeric columns.
    numeric_cols = [
        "Quantity Bought", "Average Price Bought", "Proceeds Bought",
        "Proceeds Bought in Base", "Quantity Sold", "Average Price Sold",
        "Proceeds Sold", "Proceeds Sold in Base",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].map(_to_float)

    # Filter to silver complex by symbol root or description prefix.
    sym = df.get("Symbol", pd.Series([""] * len(df))).astype(str)
    desc = df.get("Description", pd.Series([""] * len(df))).astype(str)
    is_qi = sym.str.match(r"^QI[FGHJKMNQUVXZ]\d") | desc.str.startswith("QI ")
    is_so = sym.str.match(r"^SO[FGHJKMNQUVXZ]\d") | desc.str.startswith("SI ")
    df = df[is_qi | is_so].copy()
    if df.empty:
        return df

    qb = df["Quantity Bought"].fillna(0.0)
    qs = df["Quantity Sold"].fillna(0.0).abs()
    df["gross_qty"] = qb + qs
    df["net_qty"] = qb - qs
    df["side"] = np.where(qb > qs, "long", np.where(qs > qb, "short", "flat"))

    df["instrument_class"] = df["Financial Instrument"].map(_classify_instrument)
    df["underlying_root"] = df["Symbol"].str[:2]

    expiries = df["Symbol"].map(_parse_expiry_from_symbol)
    df["expiry_code"] = expiries.map(lambda t: t[0] if t else None)
    df["expiry_yyyymm"] = expiries.map(lambda t: t[1] if t else None)
    # Prefer explicit description month when present.
    desc_yyyymm = df["Description"].map(_parse_expiry_from_description)
    df["expiry_yyyymm"] = desc_yyyymm.where(desc_yyyymm.notna(), df["expiry_yyyymm"])

    opt = df.apply(
        lambda r: _parse_option_fields(str(r.get("Symbol", "")), str(r.get("Description", ""))),
        axis=1,
    )
    df["strike"] = [t[0] for t in opt]
    df["opt_type"] = [t[1] for t in opt]

    # Realised P&L in base from proceeds (sells positive, buys negative).
    pb = df.get("Proceeds Bought in Base", pd.Series(np.nan, index=df.index)).fillna(0.0)
    ps = df.get("Proceeds Sold in Base", pd.Series(np.nan, index=df.index)).fillna(0.0)
    df["proceeds_net_base"] = pb + ps

    return df.reset_index(drop=True)


def _build_perf_by_underlying(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "Underlying" not in df.columns:
        return pd.DataFrame()
    df = df[df["Underlying"].isin(_SILVER_UNDERLYINGS)].copy()
    if df.empty:
        return df
    for c in ("AvgWeight", "Return", "Contribution", "Unrealized_P&L", "Realized_P&L"):
        if c in df.columns:
            df[c] = df[c].map(_to_float)
    df["is_total_row"] = df["Symbol"].astype(str).str.startswith("Total ")
    return df.reset_index(drop=True)


def _build_monthly_returns(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Return the Historical Performance monthly account-return rows.

    NOTE: this is ACCOUNT-LEVEL return %, not silver-only.
    """
    monthly_frames = [
        f for f in frames
        if "Month" in f.columns and "AccountReturn" in f.columns
    ]
    if not monthly_frames:
        return pd.DataFrame(columns=["yyyymm", "account_return_pct"])
    df = pd.concat(monthly_frames, ignore_index=True)
    out = pd.DataFrame({
        "yyyymm": df["Month"].astype(str),
        "account_return_pct": df["AccountReturn"].map(_to_float),
    })
    out = out.dropna(subset=["account_return_pct"]).reset_index(drop=True)
    return out


def build_silver_ledger(csv_path: str | Path) -> Dict[str, pd.DataFrame]:
    """Parse an IBKR activity-report CSV and return silver-only frames.

    Returns
    -------
    dict with keys:
        - "trades_aggregate" : Trade Summary aggregates for silver symbols.
        - "perf_by_underlying" : Performance by Underlying for the silver
          complex (with per-symbol and Total-* rows).
        - "monthly_returns" : Account-level monthly return % (caveat: NOT
          silver-specific).
    """
    csv_path = Path(csv_path)
    sections = _read_sections(csv_path)

    trade_frames = _section_frames(sections.get("Trade Summary", []))
    perf_frames = _section_frames(sections.get("Performance by Underlying", []))
    hist_frames = _section_frames(sections.get("Historical Performance", []))

    return {
        "trades_aggregate": _build_trades_aggregate(trade_frames),
        "perf_by_underlying": _build_perf_by_underlying(perf_frames),
        "monthly_returns": _build_monthly_returns(hist_frames),
    }
