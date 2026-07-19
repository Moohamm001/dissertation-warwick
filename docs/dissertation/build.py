"""Compile the dissertation to the WMG MSc template: docs/dissertation/*.md -> dissertation.docx.

Windows-first usage:
    python docs/dissertation/build.py            # build docx (+ combined md)
    python docs/dissertation/build.py --md-only  # combined markdown only

Produces the WMG front matter (submission pro-forma, title page, declaration, abstract,
acknowledgements, auto table of contents + lists of tables/figures), then Chapters 1-7,
References and Appendices, with auto-numbered captions and footer page numbers. Uses pandoc if it
is on PATH; otherwise the built-in python-docx renderer IS the production path (pandoc is not
installed on the build machine). The renderer covers exactly what these chapters use: ATX
headings, paragraphs, GFM tables, bullet/number lists, fenced code, images, inline bold/italic/
code. TOC/list fields are inserted as Word fields — after opening the .docx, select all and press
F9 (or right-click -> Update Field) to populate page numbers.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE = "Explainable Machine-Learning Credit Scoring for Green Loans under Extreme Class Imbalance"
COURSE = "Applied Artificial Intelligence"
SUBMITTED = "July 2026"
TEMPLATE_PATH = HERE / "24-25_wmg_ft_msc_dissertation_template.docx"

# Front matter is built explicitly; the body loop renders chapters 01..99 (00_abstract is folded
# into the front matter, so it is excluded there).
BODY_GLOB = "[0-9][0-9]_*.md"
ABSTRACT_FILE = "00_abstract.md"


def body_files() -> list[Path]:
    return [p for p in sorted(HERE.glob(BODY_GLOB)) if p.name != ABSTRACT_FILE]


def combined_markdown() -> str:
    parts = [f"% {TITLE}\n% MSc {COURSE} — University of Warwick\n"]
    parts.append((HERE / ABSTRACT_FILE).read_text(encoding="utf-8").strip() + "\n")
    for p in body_files():
        parts.append(p.read_text(encoding="utf-8").strip() + "\n")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ pandoc path
def build_with_pandoc(md_path: Path, out_path: Path) -> None:
    subprocess.run(
        ["pandoc", str(md_path), "-o", str(out_path),
         "--from", "markdown+pipe_tables", "--toc", "--toc-depth", "3",
         "--resource-path", str(HERE)],
        check=True,
    )


# ------------------------------------------------------------------ OXML field helpers
def _qn(tag: str):
    from docx.oxml.ns import qn
    return qn(tag)


def _field(paragraph, instruction: str, cached: str = "") -> None:
    """Insert a Word complex field (begin / instrText / separate / cached / end) into a run."""
    from docx.oxml import OxmlElement

    def _mk(tag, **attrs):
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(_qn(k), v)
        return el

    run = paragraph.add_run()
    run._r.append(_mk("w:fldChar", **{"w:fldCharType": "begin"}))
    instr = _mk("w:instrText", **{"xml:space": "preserve"})
    instr.text = instruction
    run._r.append(instr)
    run._r.append(_mk("w:fldChar", **{"w:fldCharType": "separate"}))
    if cached:
        t = _mk("w:t"); t.text = cached
        run._r.append(t)
    run._r.append(_mk("w:fldChar", **{"w:fldCharType": "end"}))


def _footer_page_numbers(section) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _field(p, "PAGE", cached="1")


def _set_page_numbering(section, fmt: str, start: int | None = None) -> None:
    """Set a section's page-number format (e.g. 'lowerRoman' front matter, 'decimal' body)."""
    from docx.oxml import OxmlElement
    sectPr = section._sectPr
    for old in sectPr.findall(_qn("w:pgNumType")):
        sectPr.remove(old)
    el = OxmlElement("w:pgNumType")
    el.set(_qn("w:fmt"), fmt)
    if start is not None:
        el.set(_qn("w:start"), str(start))
    sectPr.append(el)


def _enable_update_fields(doc) -> None:
    """Ask Word to update all fields (TOC, lists, page numbers) when the document is opened."""
    from docx.oxml import OxmlElement
    settings = doc.settings.element
    if settings.find(_qn("w:updateFields")) is None:
        el = OxmlElement("w:updateFields")
        el.set(_qn("w:val"), "true")
        settings.append(el)


def _apply_professional_styles(doc) -> None:
    """Academic house style: A4, 2.54 cm margins, Times New Roman 12/1.5 justified body,
    a clear bold heading hierarchy, and italic centred captions."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Mm, Pt

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)          # A4 portrait
    for side in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, side, Cm(2.54))

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    pf = normal.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for level, size in ((1, 16), (2, 13.5), (3, 12), (4, 11)):
        st = doc.styles[f"Heading {level}"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = None                                # keep formal black
        hpf = st.paragraph_format
        hpf.space_before = Pt(14 if level == 1 else 10)
        hpf.space_after = Pt(4)
        hpf.keep_with_next = True
        hpf.alignment = WD_ALIGN_PARAGRAPH.LEFT

    try:
        cap = doc.styles["Caption"]
        cap.font.name = "Times New Roman"
        cap.font.size = Pt(10.5)
        cap.font.italic = True
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(8)
    except KeyError:
        pass


# ------------------------------------------------------------------ inline / block renderers
_INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`[^`]+`)")


def _add_inline(par, text: str) -> None:
    for tok in _INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            par.add_run(tok[1:-1]).italic = True
        elif tok.startswith("`") and tok.endswith("`"):
            par.add_run(tok[1:-1]).font.name = "Consolas"
        else:
            par.add_run(tok)


def _list_para(doc, raw: str, ordered: bool = False, idx: int = 1):
    """A list item that prefers real list styles (present in from-scratch docs) and falls back
    to an indented paragraph with a literal marker when the template lacks those styles."""
    style = "List Number" if ordered else "List Bullet"
    try:
        p = doc.add_paragraph(style=style)
        _add_inline(p, raw)
        return p
    except KeyError:
        pass
    try:
        p = doc.add_paragraph(style="List Paragraph")
    except KeyError:
        p = doc.add_paragraph()
    _add_inline(p, (f"{idx}.\t" if ordered else "•\t") + raw)
    return p


def _add_table(doc, rows: list[str]) -> None:
    parsed = []
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        parsed.append(cells)
    if not parsed:
        return
    ncols = max(len(r) for r in parsed)
    table = doc.add_table(rows=len(parsed), cols=ncols)
    for style_name in ("Light Grid Accent 1", "Table Grid"):
        try:
            table.style = style_name
            break
        except KeyError:
            continue
    for i, r in enumerate(parsed):
        for j in range(ncols):
            cell_par = table.rows[i].cells[j].paragraphs[0]
            _add_inline(cell_par, r[j] if j < len(r) else "")
            if i == 0:
                for run in cell_par.runs:
                    run.bold = True


def _caption(doc, kind: str, title: str) -> None:
    """Caption-style paragraph with a SEQ field so the Lists of Tables/Figures populate."""
    try:
        p = doc.add_paragraph(style="Caption")
    except KeyError:
        p = doc.add_paragraph()
    p.add_run(f"{kind} ")
    _field(p, f"SEQ {kind} \\* ARABIC", cached="1")
    _add_inline(p, f": {title}")


# ------------------------------------------------------------------ front matter
# When building on the WMG template, front-matter titles use the template's own named styles
# ("Alt Heading 1", "Title") so they inherit its house look; otherwise a bold paragraph is used.
FRONT_HEADING_STYLE = None   # set to "Alt Heading 1" in template mode
FRONT_TITLE_STYLE = None     # set to "Title" in template mode


def _bold_title(doc, text: str, size: int = 16, center: bool = False, style: str | None = None):
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt
    if style:
        try:
            p = doc.add_paragraph(text, style=style)
            if center:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            return p
        except KeyError:
            pass
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.bold = True; r.font.size = Pt(size)
    return p


def _front_matter(doc) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    def para(text="", center=False, italic=False, bold=False, size=None):
        p = doc.add_paragraph()
        if center:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if text:
            r = p.add_run(text); r.italic = italic; r.bold = bold
            if size:
                r.font.size = Pt(size)
        return p

    # 1. Project Submission Pro-Forma
    _bold_title(doc, "Project Submission Pro-Forma", 15, style=FRONT_HEADING_STYLE)
    para("Student name:  ………………………………………………………………………")
    para("Student ID:  …………………………………………………………………………")
    para()
    para("I wish the dissertation to be considered for the following course:")
    para("MSc in Applied Artificial Intelligence")
    para()
    para("I confirm that I have included in my dissertation:", bold=True)
    for item in ("An abstract of the work completed",
                 "A declaration of my contribution to the work and its suitability for the degree",
                 "A table of contents",
                 "A list of figures and tables"):
        _list_para(doc, item, ordered=False)
    para()
    para("If receiving ethical approval, the ethical approval number for this research is: "
         "[to be inserted]. If ethical approval was not required, this is confirmed in Appendix B.")
    para()
    para("I consent to ongoing storage of this dissertation and potential access by third parties "
         "(e.g. for staff/student training purposes).")
    para()
    para("Signed: …………………………………………….          Date: ………………..")
    doc.add_page_break()

    # 2. Title page
    for _ in range(3):
        para()
    _bold_title(doc, TITLE, 20, center=True, style=FRONT_TITLE_STYLE)
    para()
    para("by", center=True)
    para()
    para("Name Surname, qualifications obtained", center=True)
    para()
    para(f"Dissertation submitted in partial fulfilment for the Degree of Master of Science in "
         f"{COURSE}", center=True)
    para()
    para(f"Submitted {SUBMITTED}", center=True)
    doc.add_page_break()

    # 3. Declaration (verbatim from the WMG template)
    _bold_title(doc, "Declaration", 15, style=FRONT_HEADING_STYLE)
    para("I have read and understood the rules on cheating, plagiarism and appropriate referencing "
         "as outlined in my handbook and I declare that the work contained in this assignment is my "
         "own, unless otherwise acknowledged.")
    para("No substantial part of the work submitted here has also been submitted by me in other "
         "assessments for this or previous degree courses, and I acknowledge that if this has been "
         "done an appropriate reduction in the mark I might otherwise have received will be made.")
    para()
    para("Signed: …………………………………………….          Date: ………………..")
    doc.add_page_break()

    # 4. Abstract (body of 00_abstract.md, minus its own H1)
    _bold_title(doc, "Abstract", 15, style=FRONT_HEADING_STYLE)
    for line in (HERE / ABSTRACT_FILE).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("# "):
            continue
        _add_inline(doc.add_paragraph(), s)
    doc.add_page_break()

    # 5. Acknowledgements
    _bold_title(doc, "Acknowledgements", 15, style=FRONT_HEADING_STYLE)
    para("[Optional: acknowledgements of supervisor, organisation, and personal support.]")
    doc.add_page_break()

    # 6. Table of contents
    _bold_title(doc, "Table of Contents", 15, style=FRONT_HEADING_STYLE)
    _field(doc.add_paragraph(), 'TOC \\o "1-3" \\h \\z \\u',
           cached="Right-click and Update Field to populate the table of contents.")
    doc.add_page_break()

    # 7. Lists of tables / figures
    _bold_title(doc, "List of Tables", 15, style=FRONT_HEADING_STYLE)
    _field(doc.add_paragraph(), 'TOC \\h \\z \\c "Table"',
           cached="Right-click and Update Field to populate the list of tables.")
    para()
    _bold_title(doc, "List of Figures", 15, style=FRONT_HEADING_STYLE)
    _field(doc.add_paragraph(), 'TOC \\h \\z \\c "Figure"',
           cached="Right-click and Update Field to populate the list of figures.")
    # No trailing page break: the body section break (build_with_docx) provides the transition
    # and switches page numbering from roman (front matter) to arabic (body).


# ------------------------------------------------------------------ body
_IMG_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")


def _render_chapter(doc, chapter: Path) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    lines = chapter.read_text(encoding="utf-8").splitlines()
    last_heading = chapter.stem
    i, in_code, code_buf, table_buf = 0, False, [], []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            _caption(doc, "Table", last_heading)   # caption ABOVE the table
            _add_table(doc, table_buf)
            table_buf = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            flush_table()
            if in_code:
                r = doc.add_paragraph().add_run("\n".join(code_buf))
                r.font.name = "Consolas"; r.font.size = Pt(9)
                code_buf = []
            in_code = not in_code
            i += 1; continue
        if in_code:
            code_buf.append(line); i += 1; continue

        if line.lstrip().startswith("|"):
            table_buf.append(line); i += 1; continue
        flush_table()

        m = _IMG_RE.match(line.strip())
        if m:
            alt, rel = m.group(1), m.group(2)
            img = (chapter.parent / rel).resolve()
            if img.exists():
                doc.add_picture(str(img), width=Inches(5.8))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                _caption(doc, "Figure", alt)       # caption BELOW the image
            else:
                print(f"  [warn] missing figure: {img}", file=sys.stderr)
            i += 1; continue

        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("# ").strip()
            last_heading = text
            doc.add_heading(text, level=min(level, 4))
            i += 1; continue

        stripped = line.strip()
        if re.match(r"^[-*]\s+", stripped):
            _list_para(doc, re.sub(r"^[-*]\s+", "", stripped), ordered=False)
            i += 1; continue
        mnum = re.match(r"^(\d+)\.\s+", stripped)
        if mnum:
            _list_para(doc, re.sub(r"^\d+\.\s+", "", stripped), ordered=True, idx=int(mnum.group(1)))
            i += 1; continue

        if stripped:
            buf = [stripped]
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not lines[i + 1].lstrip().startswith(("#", "|", "-", "*", "```", "!["))
                   and not re.match(r"^\d+\.\s+", lines[i + 1].strip())):
                i += 1
                buf.append(lines[i].strip())
            _add_inline(doc.add_paragraph(), " ".join(buf))
        i += 1

    flush_table()
    doc.add_page_break()


def _assemble(doc) -> None:
    """Front matter (roman page numbers) + a body section (arabic) + all chapters."""
    from docx.enum.section import WD_SECTION

    _footer_page_numbers(doc.sections[0])
    _set_page_numbering(doc.sections[0], "lowerRoman", start=1)
    _front_matter(doc)

    body = doc.add_section(WD_SECTION.NEW_PAGE)
    body.footer.is_linked_to_previous = False
    _footer_page_numbers(body)
    _set_page_numbering(body, "decimal", start=1)

    for chapter in body_files():
        _render_chapter(doc, chapter)

    _enable_update_fields(doc)


def build_with_docx(out_path: Path) -> None:
    """From-scratch build with our own academic house style (used if the template is absent)."""
    global FRONT_HEADING_STYLE, FRONT_TITLE_STYLE
    from docx import Document

    FRONT_HEADING_STYLE = FRONT_TITLE_STYLE = None
    doc = Document()
    _apply_professional_styles(doc)
    _assemble(doc)
    doc.save(out_path)


def _clear_body(doc) -> None:
    """Remove ALL of the template's placeholder content — paragraphs, tables, and structured
    document tags (Word wraps its TOC in a <w:sdt>, which is neither w:p nor w:tbl) — keeping
    only the trailing sectPr (page setup + header/footer references) so the house theme, styles
    and margins survive."""
    body = doc.element.body
    keep = _qn("w:sectPr")
    for child in list(body):
        if child.tag != keep:
            body.remove(child)


def build_from_template(out_path: Path) -> None:
    """Populate the actual WMG template .docx: inherit its styles, theme, fonts, margins and
    page-numbering, and pour our content in using the template's own named styles."""
    global FRONT_HEADING_STYLE, FRONT_TITLE_STYLE
    from docx import Document

    doc = Document(str(TEMPLATE_PATH))
    _clear_body(doc)
    FRONT_HEADING_STYLE, FRONT_TITLE_STYLE = "Alt Heading 1", "Title"
    _assemble(doc)
    doc.save(out_path)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python docs/dissertation/build.py")
    ap.add_argument("--md-only", action="store_true", help="write combined markdown only")
    ap.add_argument("--scratch", action="store_true",
                    help="ignore the WMG template and build with our own house style")
    args = ap.parse_args(argv)

    md_out = HERE / "dissertation.md"
    md_out.write_text(combined_markdown(), encoding="utf-8")
    print(f"[build] combined markdown -> {md_out}")
    if args.md_only:
        return 0

    docx_out = HERE / "dissertation.docx"
    if shutil.which("pandoc"):
        build_with_pandoc(md_out, docx_out)
        print(f"[build] pandoc -> {docx_out}")
    elif TEMPLATE_PATH.exists() and not args.scratch:
        build_from_template(docx_out)
        print(f"[build] populated the WMG template -> {docx_out}")
        print("[build] NOTE: Word will offer to update fields on open (updateFields=true); "
              "accept it to fill the contents/lists/page numbers, or select all and press F9.")
    else:
        build_with_docx(docx_out)
        print(f"[build] python-docx (house style, no template) -> {docx_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
