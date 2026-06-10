#!/usr/bin/env python3
"""Write a formatted Excel workbook for the CME tradable universe."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

DEFAULT_JSON = Path("data/reference/cme_products_20260425.json")
DEFAULT_XLSX = Path("artifacts/reference/cme_tradable_universe_20m_20260425.xlsx")
DEFAULT_POSITIVE_OI_XLSX = Path("artifacts/reference/cme_tradable_universe_positive_oi_20260425.xlsx")
DEFAULT_CAPACITY_1_XLSX = Path("artifacts/reference/cme_tradable_universe_capacity_1_20260425.xlsx")


def _col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _cell_ref(row: int, col: int) -> str:
    return f"{_col_name(col)}{row}"


def _cell(value: object, row: int, col: int, style: int = 0) -> str:
    ref = _cell_ref(row, col)
    style_attr = f' s="{style}"' if style else ""
    if value is None:
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{ref}"{style_attr} t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _row(values: Sequence[object], row_num: int, styles: Sequence[int] | None = None) -> str:
    styles = styles or []
    cells = [
        _cell(value, row_num, col, styles[col] if col < len(styles) else 0)
        for col, value in enumerate(values)
    ]
    return f'<row r="{row_num}">{"".join(cells)}</row>'


def _sheet_xml(
    rows: list[Sequence[object]],
    *,
    widths: Sequence[float],
    frozen_row: int = 1,
    filter_range: str | None = None,
    title_rows: int = 0,
    numeric_cols: set[int] | None = None,
    score_cols: set[int] | None = None,
) -> str:
    numeric_cols = numeric_cols or set()
    score_cols = score_cols or set()
    col_xml = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{width}" customWidth="1"/>'
        for i, width in enumerate(widths)
    )
    row_xml = []
    for i, values in enumerate(rows, 1):
        if i <= title_rows:
            styles = [1] * len(values)
        elif i == title_rows + 1:
            styles = [2] * len(values)
        else:
            styles = [
                4 if col in numeric_cols else 5 if col in score_cols else 0
                for col in range(len(values))
            ]
        row_xml.append(_row(values, i, styles))

    pane = ""
    if frozen_row:
        pane = (
            f'<sheetViews><sheetView workbookViewId="0"><pane ySplit="{frozen_row}" '
            f'topLeftCell="A{frozen_row + 1}" activePane="bottomLeft" state="frozen"/>'
            "</sheetView></sheetViews>"
        )
    auto_filter = f'<autoFilter ref="{filter_range}"/>' if filter_range else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{pane}<cols>{col_xml}</cols><sheetData>{''.join(row_xml)}</sheetData>{auto_filter}"
        "</worksheet>"
    )


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0.00"/></numFmts>
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="14"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF5B9BD5"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"/><right style="thin"/><top style="thin"/><bottom style="thin"/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="6">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"/>
    <xf numFmtId="3" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1"/>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _content_types(sheet_count: int) -> str:
    sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f"{sheets}</Types>"
    )


def _workbook_xml(sheet_names: Sequence[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(sheet_names, 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def _workbook_rels(sheet_count: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, sheet_count + 1)
    )
    rels += (
        f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )


def _root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def _doc_props() -> tuple[str, str]:
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>CME Tradable Universe</dc:title><dc:creator>trend_crypto</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )
    app = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>trend_crypto</Application></Properties>"
    )
    return core, app


def _product_rows(products: Iterable[dict]) -> list[list[object]]:
    return [
        [
            i,
            p["tier"],
            p["tradability_score"],
            p["root_symbol"],
            p["product_name"],
            p["cleared_as"],
            p["exchange"],
            p["asset_class"],
            p["product_group"],
            p["volume"],
            p["open_interest"],
            p["account_20m_capacity_contracts"],
        ]
        for i, p in enumerate(products, 1)
    ]


def _asset_summary_rows(products: list[dict]) -> list[list[object]]:
    by_asset: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        by_asset[product["asset_class"]].append(product)
    rows = []
    for asset, asset_products in sorted(by_asset.items(), key=lambda item: sum(p["volume"] for p in item[1]), reverse=True):
        futures = [p for p in asset_products if p["cleared_as"] == "Futures"]
        options = [p for p in asset_products if p["cleared_as"] == "Options"]
        rows.append(
            [
                asset,
                len(asset_products),
                len(futures),
                len(options),
                sum(p["volume"] for p in asset_products),
                sum(p["open_interest"] for p in asset_products),
                ", ".join(p["root_symbol"] for p in futures[:10]),
            ]
        )
    return rows


def _select_products(payload: dict, screen_name: str) -> tuple[str, list[dict], dict]:
    meta = payload["meta"]
    all_products = payload["products"]
    if screen_name == "positive-oi":
        return (
            "CME Positive OI Tradable Universe",
            [p for p in all_products if p["open_interest"] > 0],
            {
                "screen": "positive_open_interest",
                "minimum_open_interest": 1,
                "description": "Every product with positive open interest is considered tradable.",
            },
        )
    if screen_name == "capacity-1":
        return (
            "CME Capacity >= 1 Tradable Universe",
            [p for p in all_products if p["account_20m_capacity_contracts"] >= 1],
            {
                "screen": "capacity_contracts_ge_1",
                "minimum_capacity_contracts": 1,
                "capacity_contracts_formula": "min(1% of volume, 0.10% of open interest)",
                "description": "Every product with Capacity Contracts >= 1 is considered tradable.",
            },
        )
    if screen_name == "account-20m":
        return (
            "CME $20M Tradable Universe",
            [p for p in all_products if p["account_20m_tradable"]],
            meta["account_20m_screen"],
        )
    raise ValueError(f"unknown screen: {screen_name}")


def write_workbook(json_path: Path, output_path: Path, *, screen_name: str = "account-20m") -> Path:
    payload = json.loads(json_path.read_text())
    meta = payload["meta"]
    title, products, screen = _select_products(payload, screen_name)
    futures = [p for p in products if p["cleared_as"] == "Futures"]
    options = [p for p in products if p["cleared_as"] == "Options"]
    other = [p for p in products if p["cleared_as"] not in {"Futures", "Options"}]
    counts = Counter(p["asset_class"] for p in products)

    product_header = [
        "Rank",
        "Tier",
        "Score",
        "Root",
        "Product",
        "Type",
        "Exchange",
        "Asset Class",
        "Product Group",
        "Volume",
        "Open Interest",
        "Capacity Contracts",
    ]
    product_widths = [8, 8, 10, 12, 54, 14, 12, 18, 22, 14, 16, 20]
    product_numeric = {0, 9, 10, 11}
    product_score = {2}

    summary_rows: list[list[object]] = [
        [title],
        ["Source File", meta["source_file"]],
        ["Source Sheet", meta.get("sheet_name", "")],
        ["Trade Date", meta.get("trade_date", "")],
        ["Parsed Products", meta["row_count"]],
        ["Tradable Products", len(products)],
        ["Tradable Futures", len(futures)],
        ["Tradable Options", len(options)],
        ["Other Tradable Product Types", len(other)],
        ["Screen", screen.get("screen", "account_20m")],
        ["Minimum Score", screen.get("min_tradability_score", "")],
        ["Minimum Volume", screen.get("min_volume", "")],
        ["Minimum Open Interest", screen.get("min_open_interest", screen.get("minimum_open_interest", ""))],
        ["Minimum Capacity Contracts", screen.get("min_capacity_contracts", "")],
        [],
        ["Asset Class", "Tradable Products"],
    ]
    summary_rows += [[asset, count] for asset, count in counts.most_common()]

    detail_rows = [product_header] + _product_rows(products)
    futures_rows = [product_header] + _product_rows(futures)
    options_rows = [product_header] + _product_rows(options)
    asset_rows = [
        ["Asset Class", "Products", "Futures", "Options", "Total Volume", "Total Open Interest", "Top Futures Roots"]
    ] + _asset_summary_rows(products)
    assumptions_rows = [
        ["Assumptions"],
        ["Screen", screen.get("screen", "account_20m")],
        ["Description", screen.get("description", "Products pass the $20M account liquidity-capacity screen.")],
        ["Account Size USD", screen.get("account_size_usd", "")],
        ["Tradability Score", "0.48*volume_component + 0.47*open_interest_component + futures_bonus"],
        ["Volume Component", "100 * log1p(volume) / log1p(max_volume)"],
        ["Open Interest Component", "100 * log1p(open_interest) / log1p(max_open_interest)"],
        ["Futures Bonus", "5 if Cleared As == Futures else 0"],
        ["Capacity Contracts", screen.get("capacity_contracts_formula", "")],
        ["Use Caveat", "Liquidity proxy only. Validate notional, margin, spread, live depth, and session liquidity before execution."],
    ]

    sheets = {
        "Summary": _sheet_xml(summary_rows, widths=[28, 120], frozen_row=1, title_rows=1, numeric_cols={1}, score_cols=set()),
        "Tradable Universe": _sheet_xml(
            detail_rows,
            widths=product_widths,
            frozen_row=1,
            filter_range=f"A1:L{len(detail_rows)}",
            numeric_cols=product_numeric,
            score_cols=product_score,
        ),
        "Futures": _sheet_xml(
            futures_rows,
            widths=product_widths,
            frozen_row=1,
            filter_range=f"A1:L{len(futures_rows)}",
            numeric_cols=product_numeric,
            score_cols=product_score,
        ),
        "Options": _sheet_xml(
            options_rows,
            widths=product_widths,
            frozen_row=1,
            filter_range=f"A1:L{len(options_rows)}",
            numeric_cols=product_numeric,
            score_cols=product_score,
        ),
        "Other Types": _sheet_xml(
            [product_header] + _product_rows(other),
            widths=product_widths,
            frozen_row=1,
            filter_range=f"A1:L{len(other) + 1}",
            numeric_cols=product_numeric,
            score_cols=product_score,
        ),
        "Asset Summary": _sheet_xml(
            asset_rows,
            widths=[20, 12, 12, 12, 16, 18, 80],
            frozen_row=1,
            filter_range=f"A1:G{len(asset_rows)}",
            numeric_cols={1, 2, 3, 4, 5},
        ),
        "Assumptions": _sheet_xml(assumptions_rows, widths=[28, 120], frozen_row=1, title_rows=1, numeric_cols={1}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    core, app = _doc_props()
    with ZipFile(output_path, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _content_types(len(sheets)))
        z.writestr("_rels/.rels", _root_rels())
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("xl/workbook.xml", _workbook_xml(list(sheets)))
        z.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(len(sheets)))
        z.writestr("xl/styles.xml", _styles_xml())
        for i, xml in enumerate(sheets.values(), 1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", xml)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--screen", choices=["account-20m", "positive-oi", "capacity-1"], default="account-20m")
    parser.add_argument("--out")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_out = {
        "account-20m": DEFAULT_XLSX,
        "positive-oi": DEFAULT_POSITIVE_OI_XLSX,
        "capacity-1": DEFAULT_CAPACITY_1_XLSX,
    }[args.screen]
    path = write_workbook(Path(args.json), Path(args.out) if args.out else default_out, screen_name=args.screen)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
