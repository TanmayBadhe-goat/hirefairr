"""Minimal markdown to PDF converter for the audit report.

Supports headings, paragraphs, bullet lists, pipe tables, fenced code, images,
horizontal rules and inline ``code`` spans. Uses the DejaVu fonts bundled with
matplotlib so no system font or network access is needed.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
from fpdf import FPDF

BODY_SIZE = 9.5
TABLE_SIZE = 6.5
INLINE_CODE = re.compile(r"`([^`]+)`")
IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")


def _font_paths() -> tuple[Path, Path]:
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return font_dir / "DejaVuSans.ttf", font_dir / "DejaVuSans-Bold.ttf"


class ReportPDF(FPDF):
    """FPDF subclass with report styling."""

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(120)
        self.cell(0, 6, "HireFair — Fairness Audit", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(120)
        self.cell(0, 6, f"page {self.page_no()}", align="C")
        self.set_text_color(0)


def _write_paragraph(pdf: ReportPDF, text: str) -> None:
    pdf.set_font("DejaVu", "", BODY_SIZE)
    parts = INLINE_CODE.split(text)
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 1:
            pdf.set_font("DejaVu", "", BODY_SIZE - 0.5)
            pdf.write(5, part)
            pdf.set_font("DejaVu", "", BODY_SIZE)
        else:
            pdf.write(5, part)
    pdf.ln(5)


def _write_bullets(pdf: ReportPDF, bullets: list[str]) -> None:
    pdf.set_font("DejaVu", "", BODY_SIZE)
    for bullet in bullets:
        pdf.set_x(pdf.l_margin + 3)
        pdf.multi_cell(0, 5, f"•  {bullet}")
    pdf.ln(1)


def _write_table(pdf: ReportPDF, rows: list[str]) -> None:
    parsed = []
    for line in rows:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
            continue
        parsed.append(cells)
    if not parsed:
        return
    n_cols = max(len(row) for row in parsed)
    parsed = [row + [""] * (n_cols - len(row)) for row in parsed]
    pdf.set_font("DejaVu", "", TABLE_SIZE)
    width = pdf.epw / n_cols
    with pdf.table(
        col_widths=[width] * n_cols,
        text_align="LEFT",
        line_height=3.6,
        first_row_as_headings=True,
    ) as table:
        for row in parsed:
            table_row = table.row()
            for cell in row:
                table_row.cell(INLINE_CODE.sub(r"\1", cell))
    pdf.ln(3)


def _write_code(pdf: ReportPDF, lines: list[str]) -> None:
    pdf.set_font("DejaVu", "", 7.5)
    pdf.set_fill_color(245)
    text = "\n".join(lines)
    pdf.multi_cell(0, 3.8, text, fill=True)
    pdf.ln(2)


def convert(markdown_path: str | Path, pdf_path: str | Path) -> Path:
    """Convert a markdown file to PDF."""
    markdown_path = Path(markdown_path)
    pdf_path = Path(pdf_path)
    text = markdown_path.read_text(encoding="utf-8")
    regular, bold = _font_paths()
    if not regular.exists():
        raise FileNotFoundError(f"missing DejaVu font at {regular}")

    pdf = ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(14, 14, 14)
    pdf.add_font("DejaVu", "", str(regular))
    pdf.add_font("DejaVu", "B", str(bold))
    pdf.add_page()

    lines = text.splitlines()
    buffer_paragraph: list[str] = []
    buffer_table: list[str] = []
    buffer_bullets: list[str] = []
    buffer_code: list[str] = []
    in_code = False

    def flush() -> None:
        nonlocal buffer_paragraph, buffer_table, buffer_bullets
        if buffer_paragraph:
            _write_paragraph(pdf, " ".join(buffer_paragraph))
            buffer_paragraph = []
        if buffer_table:
            _write_table(pdf, buffer_table)
            buffer_table = []
        if buffer_bullets:
            _write_bullets(pdf, buffer_bullets)
            buffer_bullets = []

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("```"):
            if in_code:
                _write_code(pdf, buffer_code)
                buffer_code = []
                in_code = False
            else:
                flush()
                in_code = True
            continue
        if in_code:
            buffer_code.append(line)
            continue
        if not stripped:
            flush()
            continue
        image = IMAGE.match(stripped)
        if image:
            flush()
            path = Path(image.group("path"))
            if not path.is_absolute():
                path = markdown_path.parent / path
            if path.exists():
                pdf.image(str(path), w=min(pdf.epw, 178))
                pdf.ln(2)
            else:
                pdf.set_font("DejaVu", "", BODY_SIZE)
                pdf.multi_cell(0, 5, f"[missing figure: {path}]")
            continue
        if stripped.startswith("|"):
            if buffer_paragraph:
                flush()
            buffer_table.append(stripped)
            continue
        if stripped.startswith("- "):
            if buffer_paragraph or buffer_table:
                flush()
            buffer_bullets.append(stripped[2:].strip())
            continue
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()
            sizes = {1: 17, 2: 13, 3: 11.5, 4: 10.5}
            if level > 1:
                pdf.ln(2)
            pdf.set_font("DejaVu", "B", sizes.get(level, 10))
            pdf.multi_cell(0, 6, title)
            pdf.ln(1)
            continue
        if stripped.strip() == "---":
            flush()
            pdf.ln(1)
            continue
        buffer_paragraph.append(stripped)

    flush()
    if in_code and buffer_code:
        _write_code(pdf, buffer_code)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))
    return pdf_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("usage: python report/md_to_pdf.py report/report.md out.pdf")
    print(convert(sys.argv[1], sys.argv[2]))
