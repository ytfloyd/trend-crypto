#!/usr/bin/env python3
"""Write a print-optimized Excel workbook for CME options underlyings."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.reference.write_cme_options_underlying_screen import DEFAULT_JSON, build_rows
from scripts.reference.write_cme_tradable_universe_excel import (
    _content_types,
    _doc_props,
    _root_rels,
    _workbook_rels,
    _workbook_xml,
)

DEFAULT_XLSX = Path("artifacts/reference/cme_options_underlying_screen_print_20260425.xlsx")


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
    style_attr = f' s="{style}"' if style else ""
    ref = _cell_ref(row, col)
    if value is None or value == "":
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _row(values: Sequence[object], row_num: int, styles: Sequence[int], *, height: float | None = None) -> str:
    height_attr = f' ht="{height}" customHeight="1"' if height else ""
    cells = [_cell(value, row_num, col, styles[col] if col < len(styles) else 0) for col, value in enumerate(values)]
    return f'<row r="{row_num}"{height_attr}>{"".join(cells)}</row>'


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0.00"/></numFmts>
  <fonts count="4">
    <font><sz val="9"/><name val="Calibri"/></font>
    <font><b/><sz val="16"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>
    <font><b/><sz val="9"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/><color rgb="FF1F4E78"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF2F2F2"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFBFBFBF"/></left><right style="thin"><color rgb="FFBFBFBF"/></right><top style="thin"><color rgb="FFBFBFBF"/></top><bottom style="thin"><color rgb="FFBFBFBF"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="2" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="3" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="3" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1"><alignment horizontal="right" vertical="top"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1"><alignment horizontal="right" vertical="top"/></xf>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyFill="1" applyBorder="1"><alignment vertical="top" wrapText="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _print_sheet_xml(rows: list[Sequence[object]], styles: list[Sequence[int]], heights: dict[int, float]) -> str:
    widths = [6, 9, 28, 24, 19, 34, 12, 14, 12, 22]
    cols = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths))
    sheet_rows = "".join(
        _row(row, i, styles[i - 1], height=heights.get(i))
        for i, row in enumerate(rows, 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetPr><pageSetUpPr fitToPage="1"/></sheetPr>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="2" topLeftCell="A3" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f"<cols>{cols}</cols><sheetData>{sheet_rows}</sheetData>"
        f'<autoFilter ref="A2:J{len(rows)}"/>'
        '<printOptions horizontalCentered="1"/>'
        '<pageMargins left="0.25" right="0.25" top="0.45" bottom="0.45" header="0.2" footer="0.2"/>'
        '<pageSetup paperSize="5" orientation="landscape" fitToWidth="1" fitToHeight="0"/>'
        '<headerFooter><oddHeader>&amp;C&amp;B CME Options Underlying Screen</oddHeader><oddFooter>&amp;LGenerated from CME Product Slate&amp;RPage &amp;P of &amp;N</oddFooter></headerFooter>'
        "</worksheet>"
    )


def _appendix_sheet_xml(rows: list[Sequence[object]], styles: list[Sequence[int]]) -> str:
    widths = [7, 10, 30, 72, 65, 80]
    cols = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths))
    sheet_rows = "".join(_row(row, i, styles[i - 1], height=42 if i > 1 else None) for i, row in enumerate(rows, 1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetPr><pageSetUpPr fitToPage="1"/></sheetPr>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f"<cols>{cols}</cols><sheetData>{sheet_rows}</sheetData>"
        f'<autoFilter ref="A1:F{len(rows)}"/>'
        '<printOptions horizontalCentered="1"/>'
        '<pageMargins left="0.25" right="0.25" top="0.45" bottom="0.45" header="0.2" footer="0.2"/>'
        '<pageSetup paperSize="5" orientation="landscape" fitToWidth="1" fitToHeight="0"/>'
        "</worksheet>"
    )


def _build_print_rows(source_rows: list[list[object]]) -> tuple[list[Sequence[object]], list[Sequence[int]], dict[int, float]]:
    rows: list[Sequence[object]] = [
        ["CME Options Underlying Screen - Print Layout"],
        ["Rank", "Root", "Underlying", "Contract Unit", "Minimum Tick", "Settlement", "Opt Vol", "Opt OI", "Capacity", "Strategy Fit"],
    ]
    styles: list[Sequence[int]] = [[1], [2] * 10]
    heights = {1: 24, 2: 30}

    by_asset: dict[str, list[list[object]]] = defaultdict(list)
    for row in source_rows:
        by_asset[row[15]].append(row)

    row_num = 3
    for asset in sorted(by_asset, key=lambda a: sum(row[25] for row in by_asset[a]), reverse=True):
        rows.append([asset, "", "", "", "", "", "", "", "", ""])
        styles.append([3] * 10)
        heights[row_num] = 20
        row_num += 1
        for row in by_asset[asset]:
            rows.append(
                [
                    row[0],
                    row[1],
                    row[2],
                    row[11],
                    row[12],
                    row[13],
                    row[24],
                    row[25],
                    row[26],
                    row[27],
                ]
            )
            styles.append([5, 4, 4, 4, 4, 4, 5, 5, 5, 4])
            heights[row_num] = 54
            row_num += 1
    return rows, styles, heights


def _build_appendix_rows(source_rows: list[list[object]]) -> tuple[list[Sequence[object]], list[Sequence[int]]]:
    rows: list[Sequence[object]] = [["Rank", "Root", "Underlying", "Contract Description", "Rulebook / Spec URL", "Top Tradable Option Markets"]]
    styles: list[Sequence[int]] = [[2] * 6]
    for row in source_rows:
        rows.append([row[0], row[1], row[2], row[3], row[10], row[28]])
        styles.append([5, 4, 4, 4, 4, 4])
    return rows, styles


def write_workbook(json_path: Path, output_path: Path) -> Path:
    source_rows = build_rows(json_path)
    print_rows, print_styles, heights = _build_print_rows(source_rows)
    appendix_rows, appendix_styles = _build_appendix_rows(source_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    core, app = _doc_props()
    with ZipFile(output_path, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _content_types(2))
        z.writestr("_rels/.rels", _root_rels())
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("xl/workbook.xml", _workbook_xml(["Print Layout", "Appendix"]))
        z.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(2))
        z.writestr("xl/styles.xml", _styles_xml())
        z.writestr("xl/worksheets/sheet1.xml", _print_sheet_xml(print_rows, print_styles, heights))
        z.writestr("xl/worksheets/sheet2.xml", _appendix_sheet_xml(appendix_rows, appendix_styles))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--out", default=str(DEFAULT_XLSX))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = write_workbook(Path(args.json), Path(args.out))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
