"""
make_fixtures.py — generate synthetic Marathi Government Resolution PDFs for
smoke-testing the pipeline end to end without any real GR data.

Produces two LINKED GRs into data/fixtures/:
  * GR-2023-fees.pdf          — a fee-structure GR with a category→fee table
  * GR-2024-fees-revised.pdf  — a GR that SUPERSEDES the 2023 one with revised
                                fees and cites it by number

so the fixtures also seed the cross-reference / supersede / compare demos.

WHY reportlab (not an HTML→LibreOffice convert)
-----------------------------------------------
The fixture's whole job is that its TEXT LAYER extracts as correct Unicode and
its table has ruled lines pdfplumber can find. reportlab embeds the fonts with
a proper ToUnicode map, so BOTH PyMuPDF (the prose path) and pdfplumber (the
table path) read clean Marathi — and a GRID table gives real vector lines with
wide, fixed columns so fee numbers never wrap ("12000", not "120 00").

PER-SCRIPT FONT FALLBACK
------------------------
Noto Sans Devanagari has no Latin glyphs, and the base PDF fonts have no
Devanagari — so a real GR (which mixes both: "इतर मागासवर्ग / OBC") needs both.
`markup()` splits each string into Devanagari vs non-Devanagari runs and tags
them with the right font via Paragraph markup. NOTE: reportlab does not do
complex-script SHAPING, so Devanagari renders with decomposed matras — visually
imperfect, but the text extraction (what a smoke test checks) is correct. Real
GRs from orgpedia/mahGRs are properly shaped; these are test fixtures.

Usage:
    pip install reportlab            # plus a Devanagari TTF on the system
    python scripts/make_fixtures.py
"""

import html
import os
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR = os.path.join("data", "fixtures")
LATIN = "Helvetica"          # built-in; clean ASCII ToUnicode
LATIN_B = "Helvetica-Bold"

DEVA_CANDIDATES = [
    "/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/TTF/NotoSansDevanagari-Regular.ttf",
]
DEVA_B_CANDIDATES = [p.replace("Regular", "Bold") for p in DEVA_CANDIDATES]


def _register_fonts():
    reg = next((p for p in DEVA_CANDIDATES if os.path.exists(p)), None)
    if not reg:
        raise SystemExit(
            "No Noto Sans Devanagari TTF found. Install it, e.g.:\n"
            "  Arch/CachyOS: sudo pacman -S noto-fonts\n"
            "  Debian/Ubuntu: sudo apt install fonts-noto-devanagari")
    bold = next((p for p in DEVA_B_CANDIDATES if os.path.exists(p)), reg)
    pdfmetrics.registerFont(TTFont("Deva", reg))
    pdfmetrics.registerFont(TTFont("DevaB", bold))


_DEVA_RUN = re.compile(r"[ऀ-ॿ]+")


def markup(text, bold=False):
    """Wrap Devanagari runs in the Deva font and everything else (Latin,
    digits, punctuation) in the base font, as reportlab Paragraph markup."""
    deva, latin = ("DevaB", LATIN_B) if bold else ("Deva", LATIN)
    out, i = [], 0
    for m in _DEVA_RUN.finditer(text):
        if m.start() > i:
            out.append(f'<font name="{latin}">{html.escape(text[i:m.start()])}</font>')
        out.append(f'<font name="{deva}">{html.escape(m.group())}</font>')
        i = m.end()
    if i < len(text):
        out.append(f'<font name="{latin}">{html.escape(text[i:])}</font>')
    return "".join(out)


BODY = ParagraphStyle("body", fontName="Deva", fontSize=11, leading=17)
CENTER = ParagraphStyle("center", parent=BODY, alignment=1)
HEAD = ParagraphStyle("head", parent=CENTER, fontSize=13, leading=18)
SECTION = ParagraphStyle("section", parent=BODY, fontSize=11, leading=18, spaceBefore=10, spaceAfter=2)
RIGHT = ParagraphStyle("right", parent=BODY, alignment=2, spaceBefore=30)
RIGHT2 = ParagraphStyle("right2", parent=BODY, alignment=2)


def fee_table(header, rows):
    data = [[Paragraph(markup(header[0], bold=True), BODY),
             Paragraph(markup(header[1], bold=True), BODY)]]
    for cat, fee in rows:
        data.append([Paragraph(markup(cat), BODY), Paragraph(markup(fee), BODY)])
    t = Table(data, colWidths=[110 * mm, 60 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def para(text, style=BODY, bold=False):
    return Paragraph(markup(text, bold=bold), style)


def build(path, meta, subject, read, preamble, decision, table_header, table_rows, closing):
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=22 * mm, rightMargin=22 * mm,
                            topMargin=20 * mm, bottomMargin=20 * mm)
    flow = [
        para("महाराष्ट्र शासन", HEAD, bold=True),
        para("उच्च व तंत्र शिक्षण विभाग", CENTER),
        para("मंत्रालय, मुंबई ४०० ०३२", CENTER),
        Spacer(1, 8),
        para(f"शासन निर्णय क्रमांक: {meta['number']}"),
        para(f"दिनांक: {meta['date']}"),
    ]
    if read:
        flow += [para("वाचा:", SECTION, bold=True), para(read)]
    flow += [
        para("विषय:", SECTION, bold=True), para(subject),
        para("प्रस्तावना:", SECTION, bold=True), para(preamble),
        para("शासन निर्णय:", SECTION, bold=True), para(decision),
        Spacer(1, 6),
        fee_table(table_header, table_rows),
        Spacer(1, 6),
        para(closing),
        para("सही/-", RIGHT),
        para("(अवर सचिव)", RIGHT2),
        para("महाराष्ट्र शासन", RIGHT2),
    ]
    doc.build(flow)
    print("wrote", path)


def main():
    _register_fonts()
    os.makedirs(OUT_DIR, exist_ok=True)

    build(
        os.path.join(OUT_DIR, "GR-2023-fees.pdf"),
        meta={"number": "संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४", "date": "१५ जून, २०२३"},
        subject="शासकीय तंत्रनिकेतनांमधील (Government Polytechnics) प्रथम वर्ष पदविका "
                "(Diploma) अभ्यासक्रमांच्या शैक्षणिक शुल्क संरचनेबाबत — शैक्षणिक वर्ष २०२३-२४.",
        read="",
        preamble="राज्यातील शासकीय तंत्रनिकेतनांमध्ये प्रथम वर्ष पदविका अभ्यासक्रमांसाठी "
                 "शैक्षणिक वर्ष २०२३-२४ करिता एकसमान शुल्क संरचना निश्चित करण्याची बाब शासनाच्या "
                 "विचाराधीन होती. तंत्र शिक्षण संचालनालय (DTE) यांच्या शिफारशीच्या अनुषंगाने "
                 "शासनाने पुढीलप्रमाणे निर्णय घेतला आहे.",
        decision="उपरोक्त प्रस्तावनेच्या अनुषंगाने, शैक्षणिक वर्ष २०२३-२४ करिता प्रवर्गनिहाय "
                 "वार्षिक शैक्षणिक शुल्क खालीलप्रमाणे निश्चित करण्यात येत आहे:",
        table_header=("प्रवर्ग (Category)", "वार्षिक शुल्क (रु.)"),
        table_rows=[("खुला / Open", "12000"), ("इतर मागासवर्ग / OBC", "6000"),
                    ("अनुसूचित जाती / SC", "0"), ("अनुसूचित जमाती / ST", "0")],
        closing="अनुसूचित जाती व अनुसूचित जमाती प्रवर्गातील विद्यार्थ्यांना शासनाच्या शिष्यवृत्ती "
                "योजनेअंतर्गत संपूर्ण शुल्कमाफी अनुज्ञेय राहील.",
    )

    build(
        os.path.join(OUT_DIR, "GR-2024-fees-revised.pdf"),
        meta={"number": "संकीर्ण-२०२४/प्र.क्र.१२/तांशि-४", "date": "१० एप्रिल, २०२४"},
        subject="शासकीय तंत्रनिकेतनांमधील प्रथम वर्ष पदविका अभ्यासक्रमांच्या शैक्षणिक शुल्क "
                "संरचनेत सुधारणा करण्याबाबत — शैक्षणिक वर्ष २०२४-२५.",
        read="शासन निर्णय क्रमांक संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४, दिनांक १५ जून, २०२३.",
        preamble="संदर्भाधीन दिनांक १५ जून, २०२३ रोजीच्या शासन निर्णयाद्वारे शैक्षणिक वर्ष "
                 "२०२३-२४ करिता शुल्क संरचना निश्चित करण्यात आली होती. वाढत्या परिचालन खर्चाच्या "
                 "(operational cost) अनुषंगाने सदर शुल्कामध्ये सुधारणा करण्याची बाब शासनाच्या "
                 "विचाराधीन होती.",
        decision="संदर्भाधीन दिनांक १५ जून, २०२३ रोजीचा शासन निर्णय याद्वारे अधिक्रमित "
                 "(superseded) करण्यात येत असून, शैक्षणिक वर्ष २०२४-२५ पासून प्रवर्गनिहाय "
                 "सुधारित वार्षिक शुल्क खालीलप्रमाणे लागू राहील:",
        table_header=("प्रवर्ग (Category)", "सुधारित वार्षिक शुल्क (रु.)"),
        table_rows=[("खुला / Open", "15000"), ("इतर मागासवर्ग / OBC", "7500"),
                    ("अनुसूचित जाती / SC", "0"), ("अनुसूचित जमाती / ST", "0")],
        closing="सदर सुधारित शुल्क केवळ शैक्षणिक वर्ष २०२४-२५ व त्यापुढील प्रवेशांना लागू राहील.",
    )


if __name__ == "__main__":
    main()
