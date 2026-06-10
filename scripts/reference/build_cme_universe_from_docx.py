#!/usr/bin/env python3
"""Build a CME tradable-products reference universe from a CME slate export.

The preferred source is CME's XLSX Product Slate export because it preserves
clean cell values. DOCX parsing remains available as a fallback for older
exports. Both paths use only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZipFile

DEFAULT_XLSX = Path("/Users/russellfloyd/Dropbox_OLD/Mac (4)/Downloads/Product Slate Export.xlsx")
DEFAULT_DOCX = Path("/Users/russellfloyd/Dropbox/cme_contracts.docx")
DEFAULT_SOURCE = DEFAULT_XLSX
DEFAULT_JSON = Path("data/reference/cme_products_20260425.json")
DEFAULT_MARKDOWN = Path("docs/cme_tradable_products_universe.md")
ACCOUNT_SIZE_USD = 20_000_000
ACCOUNT_20M_MIN_SCORE = 55.0
ACCOUNT_20M_MIN_VOLUME = 1_000
ACCOUNT_20M_MIN_OPEN_INTEREST = 10_000
ACCOUNT_20M_MIN_CLIP_CONTRACTS = 10

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
EXPECTED_HEADER = (
    "Product Name",
    "Clearing",
    "Globex",
    "Floor",
    "Clearport",
    "Exchange",
    "Asset Class",
    "Product Group",
    "Category",
    "Sub Category",
    "Cleared As",
    "Volume",
    "Open Interest",
)
HEADER_ALIASES = {
    "Exch": "Exchange",
    "Exchange": "Exchange",
    "Sub-Category": "Sub Category",
    "Sub Category": "Sub Category",
}


@dataclass(frozen=True)
class CmeProduct:
    product_name: str
    clearing: str
    globex: str
    floor: str
    clearport: str
    exchange: str
    asset_class: str
    product_group: str
    category: str
    sub_category: str
    cleared_as: str
    volume: int
    open_interest: int
    tradability_score: float
    volume_component: float
    open_interest_component: float
    futures_bonus: float
    root_symbol: str
    tier: str
    account_20m_tradable: bool
    account_20m_capacity_contracts: int


def _parse_int(s: str) -> int:
    s = (s or "").strip()
    if s in {"", "-"}:
        return 0
    return int(s.replace(",", ""))


def _normalize_header(name: str) -> str:
    return HEADER_ALIASES.get(name, name)


def _row_from_fields(product_name: str, fields: dict[str, str]) -> dict:
    cleared_as = fields.get("Cleared As", "").strip()
    if not product_name or not cleared_as:
        raise ValueError("missing product name or cleared-as field")
    return {
        "product_name": product_name,
        "clearing": fields.get("Clearing", ""),
        "globex": fields.get("Globex", ""),
        "floor": fields.get("Floor", ""),
        "clearport": fields.get("Clearport", ""),
        "exchange": fields.get("Exchange", ""),
        "asset_class": fields.get("Asset Class", ""),
        "product_group": fields.get("Product Group", ""),
        "category": "" if fields.get("Category", "") == "-" else fields.get("Category", ""),
        "sub_category": "" if fields.get("Sub Category", "") == "-" else fields.get("Sub Category", ""),
        "cleared_as": cleared_as,
        "volume": _parse_int(fields.get("Volume", "0")),
        "open_interest": _parse_int(fields.get("Open Interest", "0")),
    }


def _root_symbol(clearing: str, globex: str, product_name: str) -> str:
    """Choose the best root symbol from the slate code columns."""
    preferred = (globex or clearing or "").strip()
    if preferred and preferred != "-":
        # Globex often arrives as "6E-EC" or "SR3"; keep the tradable
        # electronic root before a dash.
        return preferred.split("-")[0]
    fallback = re.sub(r"[^A-Z0-9]+", "", product_name.upper())
    return fallback[:12] or "UNKNOWN"


def _tier(score: float) -> str:
    if score >= 90:
        return "S"
    if score >= 75:
        return "A"
    if score >= 55:
        return "B"
    if score >= 35:
        return "C"
    return "D"


def _account_20m_capacity_contracts(volume: int, open_interest: int) -> int:
    """Conservative DOCX-only clip capacity for a $20M account.

    A product passes only if the account can work at least a 10-contract clip
    while staying under 1% of reported volume and 0.10% of reported OI.
    """
    return int(min(volume * 0.01, open_interest * 0.001))


def _is_account_20m_tradable(score: float, volume: int, open_interest: int) -> bool:
    return (
        score >= ACCOUNT_20M_MIN_SCORE
        and volume >= ACCOUNT_20M_MIN_VOLUME
        and open_interest >= ACCOUNT_20M_MIN_OPEN_INTEREST
        and _account_20m_capacity_contracts(volume, open_interest) >= ACCOUNT_20M_MIN_CLIP_CONTRACTS
    )


def _score_products(raw_rows: list[dict]) -> list[CmeProduct]:
    max_vol = max((r["volume"] for r in raw_rows), default=1) or 1
    max_oi = max((r["open_interest"] for r in raw_rows), default=1) or 1
    products: list[CmeProduct] = []
    for r in raw_rows:
        vol_component = 100.0 * math.log1p(r["volume"]) / math.log1p(max_vol)
        oi_component = 100.0 * math.log1p(r["open_interest"]) / math.log1p(max_oi)
        futures_bonus = 5.0 if r["cleared_as"] == "Futures" else 0.0
        score = min(100.0, 0.48 * vol_component + 0.47 * oi_component + futures_bonus)
        root = _root_symbol(r["clearing"], r["globex"], r["product_name"])
        capacity_contracts = _account_20m_capacity_contracts(r["volume"], r["open_interest"])
        products.append(
            CmeProduct(
                **r,
                tradability_score=round(score, 2),
                volume_component=round(vol_component, 2),
                open_interest_component=round(oi_component, 2),
                futures_bonus=futures_bonus,
                root_symbol=root,
                tier=_tier(score),
                account_20m_tradable=_is_account_20m_tradable(score, r["volume"], r["open_interest"]),
                account_20m_capacity_contracts=capacity_contracts,
            )
        )
    return sorted(
        products,
        key=lambda p: (p.tradability_score, p.volume, p.open_interest),
        reverse=True,
    )


def _text_runs_by_paragraph(docx_path: Path) -> list[list[str]]:
    with ZipFile(docx_path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    rows: list[list[str]] = []
    for para in root.findall(".//w:p", NS):
        runs = [
            (t.text or "").strip()
            for t in para.findall(".//w:t", NS)
            if (t.text or "").strip()
        ]
        if runs:
            rows.append(runs)
    return rows


def _xlsx_column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - 64
    return index - 1


def _xlsx_shared_strings(z: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return [
        "".join(t.text or "" for t in item.findall(".//m:t", {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}))
        for item in root.findall(".//m:si", {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"})
    ]


def _xlsx_sheet_name(z: ZipFile) -> str:
    root = ET.fromstring(z.read("xl/workbook.xml"))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    sheet = root.find(".//m:sheet", ns)
    return sheet.attrib.get("name", "") if sheet is not None else ""


def _xlsx_rows(xlsx_path: Path) -> tuple[str, list[list[str]]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(xlsx_path) as z:
        shared_strings = _xlsx_shared_strings(z)
        sheet_name = _xlsx_sheet_name(z)
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", ns):
        max_col = -1
        cells: list[tuple[int, str]] = []
        for cell in row.findall("m:c", ns):
            col = _xlsx_column_index(cell.attrib.get("r", "A1"))
            max_col = max(max_col, col)
            value_node = cell.find("m:v", ns)
            text = "" if value_node is None or value_node.text is None else value_node.text
            if cell.attrib.get("t") == "s" and text:
                text = shared_strings[int(text)]
            cells.append((col, text))
        values = [] if max_col < 0 else [""] * (max_col + 1)
        for col, text in cells:
            values[col] = text.strip()
        rows.append(values)
    return sheet_name, rows


def parse_xlsx(xlsx_path: str | Path = DEFAULT_XLSX) -> tuple[dict, list[CmeProduct]]:
    xlsx_path = Path(xlsx_path)
    sheet_name, rows = _xlsx_rows(xlsx_path)
    header_index = next(i for i, row in enumerate(rows) if row and row[0] == "Product Name")
    header = [_normalize_header(col) for col in rows[header_index]]

    raw_rows: list[dict] = []
    for row in rows[header_index + 1 :]:
        if len(row) < len(header) or not row[0]:
            continue
        padded = row + [""] * (len(header) - len(row))
        fields = dict(zip(header, padded[: len(header)]))
        try:
            raw_rows.append(_row_from_fields(fields.get("Product Name", ""), fields))
        except ValueError:
            continue

    trade_date = ""
    match = re.search(r"([A-Z][a-z]{2} \d{1,2} \d{4})", sheet_name)
    if match:
        trade_date = match.group(1)
    meta = {
        "source_file": str(xlsx_path),
        "source_format": "xlsx",
        "source_docx": str(DEFAULT_DOCX),
        "sheet_name": sheet_name,
        "trade_date": trade_date,
        "row_count": len(raw_rows),
        "columns": list(EXPECTED_HEADER),
    }
    return meta, _score_products(raw_rows)


def parse_docx(docx_path: str | Path = DEFAULT_DOCX) -> tuple[dict, list[CmeProduct]]:
    docx_path = Path(docx_path)
    paragraphs = _text_runs_by_paragraph(docx_path)
    trade_date = ""
    for runs in paragraphs:
        if runs[:2] == ["Trade", "date:Thursday"]:
            trade_date = "Thursday " + " ".join(runs[2:])
            break
        flat = "".join(runs)
        if flat.startswith("Trade date:"):
            trade_date = flat.removeprefix("Trade date:")
            break

    raw_rows: list[dict] = []
    for runs in paragraphs:
        if len(runs) < len(EXPECTED_HEADER):
            continue
        if tuple(runs) == EXPECTED_HEADER:
            continue
        # Product names may be split across multiple Word runs (for example
        # "E-", "mini S", "&P 500 Futures"). The 12 fields after product name
        # are stable, so parse from the right.
        product_name = "".join(runs[: -12])
        fields = runs[-12:]
        try:
            raw_rows.append(
                _row_from_fields(
                    product_name,
                    dict(
                        zip(
                            [
                                "Clearing",
                                "Globex",
                                "Floor",
                                "Clearport",
                                "Exchange",
                                "Asset Class",
                                "Product Group",
                                "Category",
                                "Sub Category",
                                "Cleared As",
                                "Volume",
                                "Open Interest",
                            ],
                            fields,
                        )
                    ),
                )
            )
        except (ValueError, IndexError):
            continue

    meta = {
        "source_file": str(docx_path),
        "source_format": "docx",
        "source_docx": str(docx_path),
        "trade_date": trade_date,
        "row_count": len(raw_rows),
        "columns": list(EXPECTED_HEADER),
    }
    return meta, _score_products(raw_rows)


def parse_source(source_path: str | Path = DEFAULT_SOURCE) -> tuple[dict, list[CmeProduct]]:
    source_path = Path(source_path)
    if source_path.suffix.lower() == ".xlsx":
        return parse_xlsx(source_path)
    if source_path.suffix.lower() == ".docx":
        return parse_docx(source_path)
    raise ValueError(f"unsupported source format: {source_path}")


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _top_by(products: list[CmeProduct], *, cleared_as: str | None = None, n: int = 50) -> list[CmeProduct]:
    rows = products
    if cleared_as:
        rows = [p for p in rows if p.cleared_as == cleared_as]
    return rows[:n]


def _asset_class_summary(products: list[CmeProduct]) -> list[list[object]]:
    grouped: dict[str, list[CmeProduct]] = {}
    for p in products:
        grouped.setdefault(p.asset_class, []).append(p)
    rows = []
    for asset, ps in sorted(grouped.items(), key=lambda kv: sum(p.volume for p in kv[1]), reverse=True):
        fut = [p for p in ps if p.cleared_as == "Futures"]
        rows.append(
            [
                asset,
                len(ps),
                len(fut),
                _fmt_int(sum(p.volume for p in ps)),
                _fmt_int(sum(p.open_interest for p in ps)),
                ", ".join(p.root_symbol for p in sorted(fut, key=lambda p: p.tradability_score, reverse=True)[:8]),
            ]
        )
    return rows


def _product_rows(products: list[CmeProduct]) -> list[list[object]]:
    return [
        [
            i,
            p.tier,
            f"{p.tradability_score:.2f}",
            p.root_symbol,
            p.product_name,
            p.cleared_as,
            p.exchange,
            p.asset_class,
            p.product_group,
            _fmt_int(p.volume),
            _fmt_int(p.open_interest),
        ]
        for i, p in enumerate(products, 1)
    ]


def _account_20m_rows(products: list[CmeProduct]) -> list[list[object]]:
    return [
        [
            i,
            p.tier,
            f"{p.tradability_score:.2f}",
            p.root_symbol,
            p.product_name,
            p.cleared_as,
            p.exchange,
            p.asset_class,
            p.product_group,
            _fmt_int(p.volume),
            _fmt_int(p.open_interest),
            _fmt_int(p.account_20m_capacity_contracts),
        ]
        for i, p in enumerate(products, 1)
    ]


def render_markdown(meta: dict, products: list[CmeProduct]) -> str:
    futures = _top_by(products, cleared_as="Futures", n=40)
    options = _top_by(products, cleared_as="Options", n=40)
    all_top = _top_by(products, n=50)
    account_20m = [p for p in products if p.account_20m_tradable]
    account_20m_futures = [p for p in account_20m if p.cleared_as == "Futures"]
    account_20m_options = [p for p in account_20m if p.cleared_as == "Options"]

    lines = [
        "# CME Tradable Products Universe",
        "",
        "Data-driven reference universe for CME Group products, built from the desk's",
        "CME Product Slate export.",
        "",
        "## Source",
        "",
        f"- Source file: `{meta['source_file']}`",
        f"- Source format: `{meta.get('source_format', 'unknown')}`",
        f"- Sheet name: `{meta.get('sheet_name') or 'n/a'}`",
        f"- Trade date: `{meta.get('trade_date') or 'unknown'}`",
        f"- Parsed product rows: `{meta['row_count']:,}`",
        "- Columns: Product Name, Clearing, Globex, Floor, ClearPort, Exchange, Asset Class, Product Group, Category, Sub-Category, Cleared As, Volume, Open Interest.",
        "",
        "The source table contains product-level volume and open interest. Official",
        "contract specifications such as multiplier, tick size, trading hours, and",
        "last-trade rules still need to be verified from CME contract-spec pages and",
        "rulebooks before execution use.",
        "",
        "## Tradability Score",
        "",
        "```text",
        "volume_component        = 100 * log1p(volume) / log1p(max_volume)",
        "open_interest_component = 100 * log1p(open_interest) / log1p(max_open_interest)",
        "futures_bonus           = 5 if Cleared As == Futures else 0",
        "tradability_score       = min(100, 0.48*volume_component + 0.47*open_interest_component + futures_bonus)",
        "```",
        "",
        "Tiers: `S >= 90`, `A >= 75`, `B >= 55`, `C >= 35`, `D < 35`.",
        "",
        "## $20M Account Tradability Screen",
        "",
        f"Products marked tradable for a `${ACCOUNT_SIZE_USD:,}` portfolio pass a source-only",
        "liquidity-capacity screen:",
        "",
        "```text",
        f"tradability_score >= {ACCOUNT_20M_MIN_SCORE:.0f}",
        f"volume >= {ACCOUNT_20M_MIN_VOLUME:,}",
        f"open_interest >= {ACCOUNT_20M_MIN_OPEN_INTEREST:,}",
        "capacity_contracts = min(1% of volume, 0.10% of open interest)",
        f"capacity_contracts >= {ACCOUNT_20M_MIN_CLIP_CONTRACTS}",
        "```",
        "",
        "This is a liquidity proxy, not a full execution model. The source file does not",
        "include contract notional, margin, tick value, bid/ask spread, live depth,",
        "session liquidity, or broker limits. Use the screen to decide what belongs",
        "in the research/trading universe, then validate execution details separately.",
        "",
        f"Pass count: `{len(account_20m):,}` products "
        f"(`{len(account_20m_futures):,}` futures, `{len(account_20m_options):,}` options).",
        "",
        "## $20M Account Tradable Products",
        "",
        _md_table(
            [
                "Rank",
                "Tier",
                "Score",
                "Root",
                "Product",
                "Type",
                "Exch",
                "Asset",
                "Group",
                "Volume",
                "Open Interest",
                "Capacity Contracts",
            ],
            _account_20m_rows(account_20m),
        ),
        "",
        "## Top 50 Products By Tradability",
        "",
        _md_table(
            ["Rank", "Tier", "Score", "Root", "Product", "Type", "Exch", "Asset", "Group", "Volume", "Open Interest"],
            _product_rows(all_top),
        ),
        "",
        "## Top Futures Products",
        "",
        _md_table(
            ["Rank", "Tier", "Score", "Root", "Product", "Type", "Exch", "Asset", "Group", "Volume", "Open Interest"],
            _product_rows(futures),
        ),
        "",
        "## Top Options Products",
        "",
        _md_table(
            ["Rank", "Tier", "Score", "Root", "Product", "Type", "Exch", "Asset", "Group", "Volume", "Open Interest"],
            _product_rows(options),
        ),
        "",
        "## Asset Class Summary",
        "",
        _md_table(
            ["Asset Class", "Products", "Futures Products", "Total Volume", "Total Open Interest", "Top Futures Roots"],
            _asset_class_summary(products),
        ),
        "",
        "## Recommended Futures Universe",
        "",
        "The default tradable futures universe should start from the highest-scoring",
        "futures roots in each asset class, then apply desk-specific exclusions for",
        "calendar spreads, financial swaps, or products without reliable screen depth.",
        "",
        _md_table(
            ["Asset Class", "Candidate Futures Roots"],
            [
                [
                    asset,
                    ", ".join(
                        dict.fromkeys(
                            p.root_symbol
                            for p in sorted(
                                [x for x in products if x.asset_class == asset and x.cleared_as == "Futures"],
                                key=lambda p: p.tradability_score,
                                reverse=True,
                            )[:12]
                        )
                    ),
                ]
                for asset in sorted({p.asset_class for p in products})
            ],
        ),
        "",
        "## Maintenance Notes",
        "",
        "- Rebuild this file whenever a fresh CME product slate export is available.",
        "- Keep the JSON artifact under `data/reference/` as the machine-readable source for downstream tooling.",
        "- Do not scrape CME's live product slate page. Use an exported file, licensed CME APIs, or official reports.",
        "- Treat options rows as evidence of product-family liquidity; model options as tenors/chains under the futures root rather than standalone directional roots.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    products: list[CmeProduct],
    meta: dict,
    *,
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> tuple[Path, Path]:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        **meta,
        "account_20m_screen": {
            "account_size_usd": ACCOUNT_SIZE_USD,
            "min_tradability_score": ACCOUNT_20M_MIN_SCORE,
            "min_volume": ACCOUNT_20M_MIN_VOLUME,
            "min_open_interest": ACCOUNT_20M_MIN_OPEN_INTEREST,
            "capacity_contracts_formula": "min(1% of volume, 0.10% of open interest)",
            "min_capacity_contracts": ACCOUNT_20M_MIN_CLIP_CONTRACTS,
            "passed_products": sum(1 for p in products if p.account_20m_tradable),
            "passed_futures": sum(1 for p in products if p.account_20m_tradable and p.cleared_as == "Futures"),
            "passed_options": sum(1 for p in products if p.account_20m_tradable and p.cleared_as == "Options"),
        },
    }
    payload = {"meta": meta, "products": [asdict(p) for p in products]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(meta, products), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="CME Product Slate .xlsx or .docx export")
    p.add_argument("--docx", help="Deprecated alias for --source")
    p.add_argument("--json-out", default=str(DEFAULT_JSON))
    p.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.docx or args.source
    meta, products = parse_source(source)
    json_path, markdown_path = write_outputs(
        products,
        meta,
        json_path=args.json_out,
        markdown_path=args.markdown_out,
    )
    print(f"Parsed {len(products):,} products")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print("Top 10:")
    for i, p in enumerate(products[:10], 1):
        print(
            f"{i:>2}. {p.tier} {p.tradability_score:>6.2f} "
            f"{p.root_symbol:<8} {p.product_name} "
            f"vol={p.volume:,} oi={p.open_interest:,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
