"""
Génère Rapport_VisioFind.pdf à partir de rapport_visiofind.md + docs/screenshots/.
Usage (depuis n'importe quel répertoire) : python docs/build_rapport_pdf.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from fpdf import FPDF

DOCS = Path(__file__).resolve().parent
MD_FILE = DOCS / "rapport_visiofind.md"
OUT_PDF = DOCS / "Rapport_VisioFind.pdf"

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def unicode_font_paths() -> tuple[str, str]:
    import fpdf as _fpdf

    bundled = Path(_fpdf.__file__).parent / "font"
    for name_r, name_b in (
        ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
        ("DejaVuSansCondensed.ttf", "DejaVuSansCondensed-Bold.ttf"),
    ):
        pr, pb = bundled / name_r, bundled / name_b
        if pr.is_file() and pb.is_file():
            return str(pr), str(pb)
    windir = Path(os.environ.get("WINDIR", "C:\\Windows"))
    fonts = windir / "Fonts"
    ar = fonts / "arial.ttf"
    ab = fonts / "arialbd.ttf"
    if ar.is_file() and ab.is_file():
        return str(ar), str(ab)
    raise FileNotFoundError(
        "Aucune police TTF trouvée (DejaVu dans fpdf2 ou Arial Windows). "
        "Installez fpdf2 avec les polices ou utilisez Windows."
    )


class ReportPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("VisioFont", "", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def strip_md_inline(s: str) -> str:
    s = BOLD_RE.sub(r"\1", s)
    s = s.replace("`", "")
    return s


def write_paragraph(pdf: ReportPDF, text: str) -> None:
    text = strip_md_inline(text.strip())
    if not text:
        return
    pdf.set_x(pdf.l_margin)
    pdf.set_font("VisioFont", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(1)


def write_heading(pdf: ReportPDF, level: int, text: str) -> None:
    text = strip_md_inline(text.strip("# ").strip())
    if not text or text.lower().startswith("table des matières"):
        return
    sizes = {1: 16, 2: 13, 3: 11, 4: 10}
    size = sizes.get(min(level, 4), 10)
    pdf.ln(3 if level <= 2 else 2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("VisioFont", "B", size)
    pdf.set_text_color(20, 40, 100)
    pdf.multi_cell(0, 6 if level <= 2 else 5.5, text)
    pdf.ln(1)


def write_bullet(pdf: ReportPDF, line: str) -> None:
    body = strip_md_inline(line.lstrip("- ").strip())
    pdf.set_x(pdf.l_margin)
    pdf.set_font("VisioFont", "", 10)
    pdf.multi_cell(0, 5.5, f"{chr(0x2022)} {body}")


def add_image(pdf: ReportPDF, rel_path: str, caption: str) -> None:
    path = DOCS / rel_path.replace("\\", "/")
    if not path.is_file():
        pdf.set_font("VisioFont", "", 9)
        pdf.set_text_color(180, 0, 0)
        pdf.multi_cell(0, 5, f"[Image manquante : {path.name}]")
        pdf.set_text_color(0, 0, 0)
        return
    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("VisioFont", "B", 10)
    pdf.set_text_color(0, 0, 0)
    if caption:
        pdf.multi_cell(0, 5.5, strip_md_inline(caption))
    try:
        max_w = pdf.epw
        pdf.image(str(path), w=max_w)
    except Exception as exc:  # noqa: BLE001
        pdf.set_font("VisioFont", "", 9)
        pdf.multi_cell(0, 5, f"(Erreur image : {exc})")
    pdf.ln(2)
    pdf.set_font("VisioFont", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 4.5, f"Fichier : {path.name}")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


def parse_table_row(line: str) -> list[str] | None:
    if not line.strip().startswith("|"):
        return None
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 3:
        return None
    return parts[1:-1]


def is_table_sep(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and "---" in s


def build_pdf() -> None:
    if not MD_FILE.is_file():
        print(f"Fichier introuvable : {MD_FILE}", file=sys.stderr)
        sys.exit(1)

    regular, bold = unicode_font_paths()
    pdf = ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    pdf.add_font("VisioFont", "", regular)
    pdf.add_font("VisioFont", "B", bold)

    # Page de garde
    pdf.set_font("VisioFont", "B", 22)
    pdf.set_y(70)
    pdf.multi_cell(0, 10, "VisioFind", align="C")
    pdf.set_font("VisioFont", "", 14)
    pdf.ln(6)
    pdf.multi_cell(
        0,
        8,
        "Rapport de projet — Recherche visuelle\n"
        "(Deep Learning, traitement d'image, vidéo, cartes)",
        align="C",
    )
    pdf.ln(20)
    pdf.set_font("VisioFont", "", 11)
    pdf.multi_cell(0, 6, "Document généré à partir du Markdown du dépôt.\nLes captures d'écran sont intégrées dans la section 9.", align="C")
    pdf.add_page()

    def flush_table(rows: list[list[str]]) -> None:
        if not rows:
            return
        pdf.set_x(pdf.l_margin)
        pdf.set_font("VisioFont", "", 8)
        for row in rows:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 4.5, "  |  ".join(strip_md_inline(c) for c in row))
        pdf.ln(2)
        pdf.set_font("VisioFont", "", 10)

    lines = MD_FILE.read_text(encoding="utf-8").splitlines()
    i = 0
    table_rows: list[list[str]] = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        def is_table_line(s: str) -> bool:
            t = s.strip()
            return t.startswith("|") and len(t) > 1

        m_img = IMG_RE.search(line)
        if m_img and line.strip().startswith("!"):
            flush_table(table_rows)
            table_rows = []
            alt, src = m_img.group(1), m_img.group(2)
            add_image(pdf, src, alt)
            i += 1
            continue

        if line.strip() == "---":
            i += 1
            continue

        if line.startswith("#"):
            flush_table(table_rows)
            table_rows = []
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            write_heading(pdf, level, text)
            i += 1
            continue

        if is_table_line(line):
            if is_table_sep(line):
                i += 1
                continue
            row = parse_table_row(line)
            if row:
                table_rows.append(row)
            i += 1
            continue

        flush_table(table_rows)
        table_rows = []

        if line.strip().startswith("- "):
            write_bullet(pdf, line)
            i += 1
            continue

        if re.match(r"^\d+\.\s", line.strip()):
            write_paragraph(pdf, line)
            i += 1
            continue

        write_paragraph(pdf, line)
        i += 1

    flush_table(table_rows)

    pdf.output(OUT_PDF)
    print(f"PDF écrit : {OUT_PDF}")


if __name__ == "__main__":
    os.chdir(DOCS)
    build_pdf()
