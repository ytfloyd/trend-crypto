#!/usr/bin/env python3
"""Generate a clean magazine-style PDF of the trailing-stops Substack article.

Reads docs/blog/trailing_stops_substack.md, substitutes the
IMAGE_TO_BE_UPLOADED_* placeholders with the real PNG paths under
artifacts/research/weekly_breakout_v2/blog_charts/, and produces a
publication-quality PDF at docs/blog/trailing_stops_substack.pdf.
"""
from __future__ import annotations
import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable, Preformatted,
)

ROOT = Path("/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto")
MD   = ROOT / "docs/blog/trailing_stops_substack.md"
PDF  = ROOT / "docs/blog/trailing_stops_substack.pdf"
IMG_DIR = ROOT / "artifacts/research/weekly_breakout_v2/blog_charts"

# ── Color palette matching the chart styling ─────────────────────────────
COLOR_TEXT    = HexColor("#1c1c1c")
COLOR_GREY    = HexColor("#5f5f5f")
COLOR_NAVY    = HexColor("#1f4e79")
COLOR_ORANGE  = HexColor("#e26d2e")
COLOR_RED     = HexColor("#a02c2c")
COLOR_BORDER  = HexColor("#d4d4d4")
COLOR_BLOCKBG = HexColor("#f4f1ea")  # subtle cream for blockquotes
COLOR_CODEBG  = HexColor("#f6f5f0")
COLOR_TABLEHDR = HexColor("#eae6dd")
COLOR_TABLEALT = HexColor("#f9f8f4")
COLOR_PAGE_BG = HexColor("#fbfbfb")

# ── Typography styles ─────────────────────────────────────────────────────
def make_styles():
    return dict(
        title=ParagraphStyle(
            "Title", fontName="Times-Bold", fontSize=28,
            textColor=COLOR_TEXT, alignment=TA_LEFT,
            leading=32, spaceAfter=6),
        subtitle=ParagraphStyle(
            "Subtitle", fontName="Times-Italic", fontSize=15,
            textColor=COLOR_GREY, alignment=TA_LEFT,
            leading=20, spaceAfter=18),
        h2=ParagraphStyle(
            "H2", fontName="Helvetica-Bold", fontSize=17,
            textColor=COLOR_NAVY, alignment=TA_LEFT,
            leading=22, spaceBefore=16, spaceAfter=8),
        h3=ParagraphStyle(
            "H3", fontName="Helvetica-Bold", fontSize=12,
            textColor=COLOR_TEXT, alignment=TA_LEFT,
            leading=15, spaceBefore=10, spaceAfter=4),
        body=ParagraphStyle(
            "Body", fontName="Times-Roman", fontSize=11,
            textColor=COLOR_TEXT, alignment=TA_JUSTIFY,
            leading=15.5, spaceAfter=8),
        bullet=ParagraphStyle(
            "Bullet", fontName="Times-Roman", fontSize=11,
            textColor=COLOR_TEXT, alignment=TA_LEFT,
            leading=15.5, spaceAfter=2),
        bullet_marker=ParagraphStyle(
            "BulletMarker", fontName="Times-Roman", fontSize=11,
            textColor=COLOR_TEXT, alignment=TA_LEFT, leading=15.5),
        blockquote=ParagraphStyle(
            "Blockquote", fontName="Times-Roman", fontSize=10.5,
            textColor=COLOR_TEXT, alignment=TA_JUSTIFY,
            leading=15, leftIndent=12, rightIndent=12,
            spaceBefore=4, spaceAfter=4, backColor=COLOR_BLOCKBG),
        caption=ParagraphStyle(
            "Caption", fontName="Helvetica-Oblique", fontSize=9.5,
            textColor=COLOR_GREY, alignment=TA_CENTER,
            leading=12, spaceBefore=2, spaceAfter=14),
        code=ParagraphStyle(
            "Code", fontName="Courier", fontSize=9,
            textColor=COLOR_TEXT, alignment=TA_LEFT,
            leading=11.5, leftIndent=8, rightIndent=8,
            spaceBefore=6, spaceAfter=10),
        byline=ParagraphStyle(
            "Byline", fontName="Helvetica-Oblique", fontSize=8.5,
            textColor=COLOR_GREY, alignment=TA_CENTER,
            leading=11),
    )


def inline(text: str) -> str:
    """Convert markdown inline syntax to reportlab-flavored HTML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # We already escaped <>; need to do this in the right order.
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="10">\1</font>', text)
    return text


def parse_markdown(md: str):
    """Return a list of ('kind', payload) tuples."""
    lines = md.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Fenced code block
        if line.strip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            out.append(("code", "\n".join(code_lines)))
            continue
        # Blockquote (collect contiguous > lines)
        if line.startswith("> "):
            bq_lines = []
            while i < len(lines) and (lines[i].startswith("> ") or lines[i].strip() == ">"):
                bq_lines.append(lines[i][2:] if lines[i].startswith("> ") else "")
                i += 1
            out.append(("blockquote", "\n".join(bq_lines)))
            continue
        # Table (markdown pipe table)
        if line.startswith("|") and i+1 < len(lines) and lines[i+1].startswith("|") and "---" in lines[i+1]:
            tbl_lines = [line]
            i += 1
            sep = lines[i]; tbl_lines.append(sep); i += 1
            while i < len(lines) and lines[i].startswith("|"):
                tbl_lines.append(lines[i]); i += 1
            out.append(("table", tbl_lines))
            continue
        # Image
        m = re.match(r"!\[(.*?)\]\((.+?)\)", line.strip())
        if m:
            out.append(("image", (m.group(1), m.group(2))))
            i += 1
            continue
        # Heading
        if line.startswith("### "):
            out.append(("h3sub", line[4:].strip()))
            i += 1; continue
        if line.startswith("## "):
            out.append(("h2", line[3:].strip()))
            i += 1; continue
        if line.startswith("# "):
            out.append(("title", line[2:].strip()))
            i += 1; continue
        if line.strip() == "---":
            out.append(("hr", None))
            i += 1; continue
        # Bullet (merge continuation lines indented with whitespace)
        if line.startswith("- "):
            bullet_lines = [line[2:].strip()]
            i += 1
            while i < len(lines) and lines[i].startswith(("  ", "\t")) and lines[i].strip():
                bullet_lines.append(lines[i].strip())
                i += 1
            out.append(("bullet", " ".join(bullet_lines)))
            continue
        # Blank
        if line.strip() == "":
            out.append(("blank", None))
            i += 1; continue
        # Paragraph: accumulate until blank/heading/code/etc.
        para_lines = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if (nxt.strip() == "" or
                nxt.startswith("#") or nxt.startswith("```") or
                nxt.startswith("|") or nxt.startswith("- ") or
                nxt.startswith("> ") or nxt.startswith("![") or
                nxt.strip() == "---"):
                break
            para_lines.append(nxt)
            i += 1
        out.append(("para", " ".join(para_lines)))
    return out


def build_table(tbl_lines, styles):
    rows = []
    for ln in tbl_lines:
        if "---" in ln and all(c in "|-: " for c in ln):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    # Wrap cell text in Paragraph so it supports inline formatting + wraps
    body_style = ParagraphStyle("TblBody", parent=styles["body"], fontSize=9.5,
                                  leading=12, alignment=TA_LEFT, spaceAfter=0,
                                  fontName="Helvetica")
    head_style = ParagraphStyle("TblHead", parent=body_style,
                                  fontName="Helvetica-Bold", textColor=COLOR_NAVY)
    formatted = []
    for r_idx, row in enumerate(rows):
        styled_row = []
        for c in row:
            style = head_style if r_idx == 0 else body_style
            styled_row.append(Paragraph(inline(c), style))
        formatted.append(styled_row)

    n_cols = len(formatted[0])
    # First column wider for the metric name
    col_widths = [2.4*inch] + [(4.7*inch - 2.4*inch)/(n_cols-1) + 0.4*inch] * (n_cols-1)
    if n_cols == 3:
        col_widths = [2.4*inch, 1.6*inch, 1.7*inch]

    table = Table(formatted, colWidths=col_widths, hAlign="LEFT")
    ts = [
        ("BACKGROUND", (0,0), (-1,0), COLOR_TABLEHDR),
        ("LINEBELOW", (0,0), (-1,0), 1.0, COLOR_NAVY),
        ("LINEABOVE", (0,0), (-1,0), 0.5, COLOR_NAVY),
        ("LINEBELOW", (0,-1), (-1,-1), 0.5, COLOR_NAVY),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]
    for r in range(1, len(formatted)):
        if r % 2 == 0:
            ts.append(("BACKGROUND", (0,r), (-1,r), COLOR_TABLEALT))
    table.setStyle(TableStyle(ts))
    return table


def resolve_image_path(path_str: str) -> Path:
    """Resolve IMAGE_TO_BE_UPLOADED_xx_name.png → actual chart file."""
    name = Path(path_str).name
    if name.startswith("IMAGE_TO_BE_UPLOADED_"):
        name = name[len("IMAGE_TO_BE_UPLOADED_"):]
    return IMG_DIR / name


def header_footer(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    if page > 1:  # skip on title page
        # Footer
        canvas.setStrokeColor(COLOR_BORDER)
        canvas.setLineWidth(0.4)
        canvas.line(doc.leftMargin, 0.55*inch,
                    LETTER[0] - doc.rightMargin, 0.55*inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(COLOR_GREY)
        canvas.drawString(doc.leftMargin, 0.4*inch,
                           "The Stop-Cost Hierarchy · trend_crypto research")
        canvas.drawRightString(LETTER[0] - doc.rightMargin, 0.4*inch,
                                f"Page {page}")
    canvas.restoreState()


def build_story(md_text, styles):
    parsed = parse_markdown(md_text)
    story = []
    in_appendix = False
    for kind, payload in parsed:
        # Skip everything from the Substack-only "Image upload guide" onward
        if kind == "h2" and "Image upload guide" in payload:
            in_appendix = True
            continue
        if kind == "h3sub" and "Image upload guide" in payload:
            in_appendix = True
            continue
        if in_appendix:
            continue

        if kind == "title":
            story.append(Paragraph(inline(payload), styles["title"]))
        elif kind == "h3sub":
            # First ### in the document (right after the H1 title) is the article
            # subtitle (italic). Every subsequent ### is a sub-section header (bold).
            seen_subtitle = any(isinstance(s, Paragraph) and s.style == styles["subtitle"]
                                 for s in story)
            if not seen_subtitle:
                story.append(Paragraph(inline(payload), styles["subtitle"]))
            else:
                story.append(Paragraph(inline(payload), styles["h3"]))
        elif kind == "h2":
            story.append(Spacer(1, 6))
            story.append(Paragraph(inline(payload), styles["h2"]))
        elif kind == "hr":
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                      color=COLOR_BORDER, spaceBefore=4,
                                      spaceAfter=8))
        elif kind == "para":
            story.append(Paragraph(inline(payload), styles["body"]))
        elif kind == "bullet":
            bullet_row = Table(
                [[Paragraph("•", styles["bullet_marker"]),
                  Paragraph(inline(payload), styles["bullet"])]],
                colWidths=[28, 6.5*inch - 28])
            bullet_row.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (0,-1), 18),
                ("RIGHTPADDING", (0,0), (0,-1), 4),
                ("LEFTPADDING", (1,0), (1,-1), 0),
                ("RIGHTPADDING", (1,0), (1,-1), 0),
                ("TOPPADDING", (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ]))
            story.append(bullet_row)
            story.append(Spacer(1, 3))
        elif kind == "blockquote":
            text = payload.strip()
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            # Join paragraphs with a small <br/> spacer rather than nesting flowables
            joined = "<br/><br/>".join(inline(p) for p in paras)
            quote = Paragraph(joined, styles["blockquote"])
            inner = Table([[quote]], colWidths=[6.5*inch])
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), COLOR_BLOCKBG),
                ("LINEBEFORE", (0,0), (0,-1), 3, COLOR_NAVY),
                ("LEFTPADDING", (0,0), (-1,-1), 14),
                ("RIGHTPADDING", (0,0), (-1,-1), 14),
                ("TOPPADDING", (0,0), (-1,-1), 10),
                ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ]))
            story.append(inner)
            story.append(Spacer(1, 10))
        elif kind == "code":
            # Pretty code block with cream background
            code = payload
            box = Preformatted(code, styles["code"])
            wrap = Table([[box]], colWidths=[6.5*inch])
            wrap.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), COLOR_CODEBG),
                ("LINEBEFORE", (0,0), (0,-1), 2, COLOR_ORANGE),
                ("LEFTPADDING", (0,0), (-1,-1), 10),
                ("RIGHTPADDING", (0,0), (-1,-1), 10),
                ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ]))
            story.append(wrap)
            story.append(Spacer(1, 8))
        elif kind == "table":
            tbl = build_table(payload, styles)
            story.append(Spacer(1, 4))
            story.append(tbl)
            story.append(Spacer(1, 12))
        elif kind == "image":
            alt, path = payload
            img_path = resolve_image_path(path)
            if img_path.exists():
                from PIL import Image as PILImage
                pim = PILImage.open(img_path)
                w_px, h_px = pim.size
                aspect = h_px / w_px
                img_w = 6.5 * inch
                img_h = img_w * aspect
                # Cap height so a chart doesn't overflow a page
                if img_h > 7.0 * inch:
                    img_h = 7.0 * inch
                    img_w = img_h / aspect
                story.append(KeepTogether([
                    Image(str(img_path), width=img_w, height=img_h),
                    Paragraph(f"<i>{inline(alt)}</i>", styles["caption"]),
                ]))
            else:
                story.append(Paragraph(f"[Missing image: {img_path}]", styles["body"]))
        elif kind == "blank":
            story.append(Spacer(1, 3))
    return story


def main():
    md_text = MD.read_text()
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(PDF), pagesize=LETTER,
        leftMargin=1.0*inch, rightMargin=1.0*inch,
        topMargin=0.9*inch, bottomMargin=0.8*inch,
        title="The Stop-Cost Hierarchy",
        author="trend_crypto research",
        subject="Why trailing ATR stops destroy crypto trend-following Sharpe",
    )
    story = build_story(md_text, styles)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
