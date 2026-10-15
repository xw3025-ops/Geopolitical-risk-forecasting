"""
Build the trade-module section as a Word document from its Markdown draft, in the same style as
paper/build_docx.py (transformer_section.docx). The Markdown file stays the single source of
truth: edit paper/trade_section_draft.md, then rebuild.

    python paper/compute_trade_stats.py && python paper/make_trade_figures.py
    python paper/build_trade_docx.py            # -> paper/trade_section.docx

Supported Markdown: # / ## / ### headings, paragraphs, **bold**, *italic*, `code`, "- " bullets
(two-space indent = second level), "1. " numbered items, "> " display lines, pipe tables (a
paragraph that is entirely bold right above a table becomes its numbered caption),
![caption](file.png) figures, ``` code blocks, and "---" rules (skipped).
"""

import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "trade_section_draft.md")
OUT = os.path.join(HERE, "trade_section.docx")
INK = RGBColor(0x0B, 0x0B, 0x0B)
GREY = RGBColor(0x52, 0x51, 0x4E)
FONT = "Times New Roman"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.4)
sec.top_margin = sec.bottom_margin = Cm(2.2)
st = doc.styles["Normal"]
st.font.name = FONT
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for lvl, size in [(1, 14), (2, 12), (3, 11)]:
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = FONT
    h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    h.font.size, h.font.bold, h.font.italic = Pt(size), True, lvl == 3
    h.font.color.rgb = INK
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True

TOKENS = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)")
counters = {"table": 0}


def runs(p, text, size=None, color=None, italic=False):
    for part in TOKENS.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`"):
            r = p.add_run(part[1:-1])
            r.font.name = "Courier New"
            r.font.size = Pt((size or 11) - 1.5)
        elif part.startswith("*"):
            r = p.add_run(part[1:-1])
            r.italic = True
        else:
            r = p.add_run(part)
        if italic:
            r.italic = True
        if size and not part.startswith("`"):
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color


def paragraph(text, style=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, **kw):
    p = doc.add_paragraph(style=style)
    runs(p, text, **kw)
    p.alignment = align
    return p


def shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def table(lines, caption):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
    header, body = rows[0], rows[1:]
    if caption:
        counters["table"] += 1
        c = doc.add_paragraph()
        runs(c, f"**Table T{counters['table']}.** {caption}", size=9.5, color=GREY)
        c.paragraph_format.keep_with_next = True
        c.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate([header] + body):
        for j in range(len(header)):
            cell = t.rows[i].cells[j]
            cell.text = ""
            runs(cell.paragraphs[0], row[j] if j < len(row) else "", size=9)
            if i == 0:
                for r in cell.paragraphs[0].runs:
                    r.bold = True
                shade(cell, "E8E7E3")
            elif j > 0:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        t.rows[i]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def figure(caption, path):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(HERE, path), width=Cm(16.0))
    m = re.match(r"(Figure T\d+\.)\s*(.*)", caption)
    c = doc.add_paragraph()
    runs(c, f"**{m.group(1)}** {m.group(2)}" if m else caption, size=9.5, color=GREY)
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after = Pt(10)


def build():
    lines = open(SRC, encoding="utf-8").read().splitlines()
    i, pending_bold = 0, None

    def flush_pending():
        nonlocal pending_bold
        if pending_bold is not None:
            paragraph(pending_bold)
            pending_bold = None

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s or s == "---":
            i += 1
            continue
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            caption = pending_bold[2:-2].rstrip(":") if pending_bold else None
            pending_bold = None
            table(block, caption)
            continue
        flush_pending()
        if s.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                p = doc.add_paragraph()
                r = p.add_run(lines[i])
                r.font.name, r.font.size = "Courier New", Pt(8.5)
                p.paragraph_format.space_after = Pt(0)
                i += 1
            i += 1
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue
        m = re.match(r"!\[(.*)\]\((.*)\)", s)
        if m:
            figure(m.group(1), m.group(2))
        elif s.startswith("### "):
            doc.add_heading(s[4:], level=2)
        elif s.startswith("## "):
            doc.add_heading(s[3:], level=1)
        elif s.startswith("# "):
            h = doc.add_heading(s[2:], level=0)
            for r in h.runs:
                r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(17), INK
        elif s.startswith("> "):
            p = paragraph(s[2:], align=WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.space_before = Pt(2)
        elif re.match(r"^- ", ln):
            p = paragraph(s[2:], style="List Bullet")
            p.paragraph_format.space_after = Pt(2)
        elif re.match(r"^\s{2,}- ", ln):
            p = paragraph(s[2:], style="List Bullet 2")
            p.paragraph_format.space_after = Pt(2)
        elif re.match(r"^\d+\.\s", s):
            p = paragraph(re.sub(r"^\d+\.\s", "", s), style="List Number")
            p.paragraph_format.space_after = Pt(2)
        elif re.fullmatch(r"\*\*[^*]+\*\*:?", s):
            pending_bold = s.rstrip(":")  # may become a table caption
        elif s.startswith("*") and s.endswith("*") and not s.startswith("**"):
            paragraph(s[1:-1], italic=True, size=9.5, color=GREY, align=WD_ALIGN_PARAGRAPH.LEFT)
        else:
            paragraph(s)
        i += 1
    flush_pending()
    doc.save(OUT)
    print(f"wrote {OUT} ({counters['table']} tables)")


if __name__ == "__main__":
    build()
