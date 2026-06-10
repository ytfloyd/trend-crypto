#!/usr/bin/env python3
"""Write a print-ready PDF for CME underlyings with tradable options."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from scripts.reference.write_cme_options_underlying_screen import DEFAULT_JSON, build_rows

DEFAULT_PDF = Path("artifacts/reference/cme_options_underlying_screen_capacity_1_20260425.pdf")


def _fmt_int(value: object) -> str:
    if value in {"", None}:
        return ""
    return f"{int(value):,}"


def _fmt_score(value: object) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.2f}"


def _para(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text).replace("&", "&amp;"), style)


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.55 * inch, 0.35 * inch, "CME options underlying screen - capacity contracts >= 1")
    canvas.drawRightString(7.95 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _summary_tables(rows: list[list[object]], styles) -> list:
    total = len(rows)
    by_asset = Counter(row[15] for row in rows)
    futures_roots = sum(1 for row in rows if row[17])
    weekly = sum(1 for row in rows if row[23])

    story: list = []
    story.append(Paragraph("CME Underlyings With Tradable Options", styles["TitleCenter"]))
    story.append(Paragraph("Print Screen For Trend / Convexity Research", styles["Subtitle"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Screen basis: CME Product Slate Apr 25 2026. Options are included when the option product has "
            "Capacity Contracts >= 1, where capacity is min(1% of reported volume, 0.10% of reported open interest). "
            "Rows are aggregated from tradable option products to their underlying futures market.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 0.16 * inch))

    overview = [
        ["Metric", "Value"],
        ["Underlying markets", _fmt_int(total)],
        ["Markets with matched futures", _fmt_int(futures_roots)],
        ["Markets with weekly/EOM options", _fmt_int(weekly)],
        ["Total option volume", _fmt_int(sum(row[24] for row in rows))],
        ["Total option open interest", _fmt_int(sum(row[25] for row in rows))],
        ["Total option capacity contracts", _fmt_int(sum(row[26] for row in rows))],
    ]
    table = Table(overview, colWidths=[2.4 * inch, 2.0 * inch], hAlign="LEFT")
    table.setStyle(_table_style(header=True))
    story.append(table)
    story.append(Spacer(1, 0.18 * inch))

    asset_table = [["Asset Class", "Markets"]] + [[asset, count] for asset, count in by_asset.most_common()]
    table = Table(asset_table, colWidths=[2.4 * inch, 1.0 * inch], hAlign="LEFT")
    table.setStyle(_table_style(header=True))
    story.append(table)
    story.append(Spacer(1, 0.22 * inch))

    top = [["Rank", "Root", "Underlying", "Option OI", "Option Vol", "Fit"]]
    for row in rows[:12]:
        top.append([row[0], row[1], _para(row[2], styles["Small"]), _fmt_int(row[25]), _fmt_int(row[24]), _para(row[27], styles["Small"])])
    table = Table(top, colWidths=[0.45 * inch, 0.55 * inch, 2.0 * inch, 1.0 * inch, 1.0 * inch, 1.75 * inch])
    table.setStyle(_table_style(header=True))
    story.append(Paragraph("Top Markets By Option Open Interest", styles["H2"]))
    story.append(table)
    return story


def _table_style(header: bool = False) -> TableStyle:
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D3DF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    return TableStyle(commands)


def _market_card(row: list[object], styles) -> KeepTogether:
    title = f"{row[0]}. {row[1]} - {row[2]}"
    subtitle = f"{row[15]} / {row[16]} | {row[27]}"
    liquidity = [
        ["Metric", "Value", "Metric", "Value"],
        ["Future score", _fmt_score(row[19]), "Matched future", row[18]],
        ["Future volume", _fmt_int(row[20]), "Future OI", _fmt_int(row[21])],
        ["Option products", _fmt_int(row[22]), "Weekly/EOM products", _fmt_int(row[23])],
        ["Option volume", _fmt_int(row[24]), "Option OI", _fmt_int(row[25])],
        ["Option capacity", _fmt_int(row[26]), "Rulebook exchange", row[9]],
    ]
    liquidity_table = Table(liquidity, colWidths=[1.15 * inch, 1.45 * inch, 1.35 * inch, 2.45 * inch])
    liquidity_table.setStyle(_table_style(header=True))

    specs = [
        ["Field", "Value"],
        ["Description", _para(row[3], styles["Small"])],
        ["Contract unit", _para(row[11], styles["Small"])],
        ["Price quotation", _para(row[5], styles["Small"])],
        ["Minimum tick", _para(row[12], styles["Small"])],
        ["Settlement", _para(row[13], styles["Small"])],
        ["Rulebook/spec URL", _para(row[10], styles["Tiny"])],
        ["Verification notes", _para(row[14], styles["Small"])],
        ["Top tradable option markets", _para(row[28], styles["Tiny"])],
    ]
    specs_table = Table(specs, colWidths=[1.45 * inch, 5.75 * inch])
    specs_table.setStyle(_table_style(header=True))

    return KeepTogether(
        [
            Paragraph(title, styles["H3"]),
            Paragraph(subtitle, styles["SmallMuted"]),
            Spacer(1, 0.06 * inch),
            liquidity_table,
            Spacer(1, 0.07 * inch),
            specs_table,
            Spacer(1, 0.15 * inch),
        ]
    )


def write_pdf(json_path: Path, output_path: Path) -> Path:
    rows = build_rows(json_path)
    by_asset: dict[str, list[list[object]]] = defaultdict(list)
    for row in rows:
        by_asset[row[15]].append(row)

    base = getSampleStyleSheet()
    styles = {
        "TitleCenter": ParagraphStyle(
            "TitleCenter",
            parent=base["Title"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1F4E78"),
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            alignment=TA_CENTER,
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#666666"),
        ),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=9, leading=12),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#1F4E78")),
        "H3": ParagraphStyle("H3", parent=base["Heading3"], fontSize=11, leading=14, textColor=colors.HexColor("#1F4E78"), spaceAfter=2),
        "Small": ParagraphStyle("Small", parent=base["BodyText"], fontSize=8, leading=10, alignment=TA_LEFT),
        "Tiny": ParagraphStyle("Tiny", parent=base["BodyText"], fontSize=6.8, leading=8.2, alignment=TA_LEFT),
        "SmallMuted": ParagraphStyle("SmallMuted", parent=base["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#666666")),
    }

    story = _summary_tables(rows, styles)
    story.append(PageBreak())

    asset_order = sorted(by_asset, key=lambda asset: sum(row[25] for row in by_asset[asset]), reverse=True)
    for asset_i, asset in enumerate(asset_order):
        story.append(Paragraph(asset, styles["H2"]))
        story.append(
            Paragraph(
                f"{len(by_asset[asset])} underlyings | total option OI {_fmt_int(sum(row[25] for row in by_asset[asset]))} | "
                f"total option volume {_fmt_int(sum(row[24] for row in by_asset[asset]))}",
                styles["SmallMuted"],
            )
        )
        story.append(Spacer(1, 0.1 * inch))
        for row in by_asset[asset]:
            story.append(_market_card(row, styles))
        if asset_i != len(asset_order) - 1:
            story.append(PageBreak())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.55 * inch,
        title="CME Options Underlying Screen",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--out", default=str(DEFAULT_PDF))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = write_pdf(Path(args.json), Path(args.out))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
