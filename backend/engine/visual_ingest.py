"""
visual_ingest.py — the layout-model half of the upload pipeline (v2).

One `parse_document` pass over a PDF yields THREE things reused together, so
the ~4s layout cost is paid once, not per-modality:
  * prose markdown per page  -> prose_pages()   (fed to the existing chunker)
  * figure crops + captions  -> visual_assets()  (content_type 'figure')
  * formula crops + context  -> visual_assets()  (content_type 'formula')

WHY CAPTION/CONTEXT TEXT, NOT A VISION DESCRIPTION
---------------------------------------------------
A figure/formula chunk is embedded by the human-written text AROUND it — its
caption plus a prefix/postfix window of body text — NOT by any vision-LLM
call. That is the whole reason ingestion stays fast: no LLM runs here. The
crop's PNG bytes ride along on the chunk; the vision model only ever sees an
image at QUERY time, and only if that chunk is actually retrieved (see the
generation layer). This keeps the ingestion pipeline deterministic, free, and
consistent with the corpus pipeline's "extract + embed, no generation" shape.

CONTRACT
--------
visual_assets() returns dicts shaped {content_type, page, image_bytes, text}.
The caller (ingest.process_pdf_v2) turns each into a normal chunk dict, so the
only thing new downstream is content_type in {'figure','formula'} and a
transient image_bytes key that api.py strips after uploading to Storage.

MANDATORY: use_ocr=OCRMode.NEVER — otherwise pymupdf4llm demands a system
Tesseract install and raises at import-of-the-model time. Non-scanned research
PDFs already have a text layer, so OCR buys nothing here anyway.
"""

# pyrefly: ignore [missing-import]
import fitz
from pymupdf4llm.helpers.document_layout import parse_document, OCRMode

CROP_DPI = 200            # sharp enough for a vision model, small enough to store
CONTEXT_WORDS = 40        # prefix/postfix words pulled around a figure/formula
CONTEXT_BAND = 60         # pts above/below a box to sweep for that context text
MIN_BOX_SIDE = 20         # ignore sub-20pt boxes (rules, icons, extraction noise)


def parse_pdf(pdf_path):
    """Open the PDF and run the layout model once. Returns (doc, parsed).
    The caller owns closing `doc` (it must stay open for cropping + text
    reads via get_pixmap / get_textbox on its pages)."""
    doc = fitz.open(pdf_path)
    parsed = parse_document(doc, force_text=True, use_ocr=OCRMode.NEVER)
    return doc, parsed


def prose_pages(parsed, strip_tables=False):
    """[(page_no, markdown)] straight from the layout model's per-page markdown.

    By default tables are LEFT INLINE (the layout model already rendered them
    as markdown, for free) — so uploads skip the separate ~4.4s pdfplumber
    pass. A table's data then lives in the text chunk right next to its caption
    and surrounding prose, which is good for retrieval; upload retrieval is
    flat (not per-modality), so a distinct 'table' content_type would buy
    nothing here anyway. `strip_tables=True` removes them instead (the old
    behavior, if a caller wants to pair this with the pdfplumber table path).
    """
    out = []
    for page in parsed.to_markdown(page_chunks=True):
        page_no = page["metadata"]["page_number"]
        text = _strip_table_blocks(page["text"]) if strip_tables else page["text"]
        out.append((page_no, text))
    return out


def _strip_table_blocks(md):
    """Drop contiguous markdown table blocks: a pipe row immediately followed
    by a `---|---` separator row, then all following pipe rows."""
    lines = md.splitlines()
    kept = []
    i = 0
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            is_separator = "|" in nxt and all(c in "-:| " for c in nxt if c)
            if is_separator:
                i += 2
                while i < len(lines) and "|" in lines[i]:
                    i += 1
                continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept)


def _text_in_rect(doc_page, rect):
    return " ".join(doc_page.get_textbox(rect).split())


def _caption_text(page_layout, doc_page, box):
    """The caption for a figure: the nearest `caption` box, preferring one
    that sits just below the figure (the usual placement). Returns '' if the
    layout model found no caption on the page."""
    caps = [b for b in page_layout.boxes if b.boxclass == "caption"]
    if not caps:
        return ""

    def rank(c):
        gap = c.y0 - box.y1                     # positive => caption below figure
        return gap if gap >= -5 else 1e6 + abs(gap)  # below preferred, else nearest

    best = min(caps, key=rank)
    return _text_in_rect(doc_page, fitz.Rect(best.x0, best.y0, best.x1, best.y1))


def _context_around(doc_page, box):
    """Prefix = up to CONTEXT_WORDS words immediately above the box; postfix =
    up to CONTEXT_WORDS immediately below. This is the 'prefix/postfix of the
    figure' the retrieval design embeds so a natural-language query matches
    the surrounding discussion."""
    width = doc_page.rect.width
    above = fitz.Rect(0, max(0, box.y0 - CONTEXT_BAND), width, box.y0)
    below = fitz.Rect(0, box.y1, width, box.y1 + CONTEXT_BAND)
    prefix = " ".join(_text_in_rect(doc_page, above).split()[-CONTEXT_WORDS:])
    postfix = " ".join(_text_in_rect(doc_page, below).split()[:CONTEXT_WORDS])
    return prefix, postfix


def visual_assets(doc, parsed):
    """Crop every figure/formula box and build its searchable text.

    Returns [{content_type, page, image_bytes, text}] where text = caption +
    prefix + postfix (whatever exists), never empty. image_bytes is the PNG
    crop; the caller persists it to Storage and drops it before the DB write.
    """
    assets = []
    for page_layout in parsed.pages:
        page_no = page_layout.page_number
        doc_page = doc[page_no - 1]
        for box in page_layout.boxes:
            if box.boxclass not in ("picture", "formula"):
                continue
            rect = fitz.Rect(box.x0, box.y0, box.x1, box.y1)
            if rect.width < MIN_BOX_SIDE or rect.height < MIN_BOX_SIDE:
                continue

            image_bytes = doc_page.get_pixmap(clip=rect, dpi=CROP_DPI).tobytes("png")
            content_type = "figure" if box.boxclass == "picture" else "formula"

            caption = _caption_text(page_layout, doc_page, box) if content_type == "figure" else ""
            prefix, postfix = _context_around(doc_page, box)
            text = " ".join(p for p in (caption, prefix, postfix) if p).strip()
            if not text:
                text = f"{content_type} on page {page_no}"  # keep it embeddable

            assets.append({
                "content_type": content_type,
                "page": page_no,
                "image_bytes": image_bytes,
                "text": text,
            })
    return assets
