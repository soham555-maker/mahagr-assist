"""
add_pdfs.py — append a folder of GR PDFs to an existing index.

Used to add real Government-Resolution PDFs (e.g. the operator's own downloads
from gr.maharashtra.gov.in) on top of an already-built index, via the PDF path
(process_pdf → text + OCR fallback + pdfplumber tables). Metadata is parsed with
gr_metadata; the order-id filename (first 8 digits = YYYYMMDD) is a date fallback.

Usage:  python scripts/add_pdfs.py "../Original Maha-GR" [index] [--ocr]

--ocr forces OCR on every page (ignore the text layer) — use it for government
PDFs whose embedded font is broken, so the text layer extracts as garbled
Devanagari. Needs Tesseract + the mar/hin/eng language data installed.
"""

import os
import re
import sys

from engine import gr_metadata
from engine.ingest import IngestionPipeline
from engine.vector_store import FaissStore


def _order_id(name):
    return os.path.splitext(name)[0]


def _date_from_id(order_id):
    m = re.match(r"(\d{4})(\d{2})(\d{2})", order_id)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def main(pdf_dir, index_dir="index", force_ocr=False):
    store = FaissStore.load(index_dir)
    print(f"Loaded index: {len(store)} vectors.  force_ocr={force_ocr}")
    pipeline = IngestionPipeline()  # bge-m3 (must match the index's model)

    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    added, skipped = 0, 0
    for i, name in enumerate(pdfs, 1):
        path = os.path.join(pdf_dir, name)
        order_id = _order_id(name)
        try:
            pages = pipeline.extract_pages_from_pdf(path, force_ocr=force_ocr)
            meta = gr_metadata.extract("\n".join(t for _, t in pages))
            meta["order_id"] = order_id
            meta.setdefault("date", _date_from_id(order_id))
            meta.setdefault("title", order_id)
            chunks = pipeline.process_pdf(path, source_type="gr", include_tables=True,
                                          extra_metadata=meta, pages=pages)
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] FAILED {name}: {e}")
            continue
        if not chunks:
            print(f"[{i}/{len(pdfs)}] EMPTY  {name} (no text even after OCR)")
            skipped += 1
            continue
        n_tab = sum(1 for c in chunks if c["metadata"]["content_type"] == "table")
        store.add(chunks)
        added += 1
        print(f"[{i}/{len(pdfs)}] ok  {order_id}  +{len(chunks)} chunks ({n_tab} table)  "
              f"gr={meta.get('gr_number')}  date={meta.get('date')}")

    print("\n" + "=" * 60)
    print(f"Added {added} PDFs, skipped {skipped}. Index now {len(store)} vectors.")
    store.save(index_dir)
    print(f"Saved index to {index_dir}/.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--ocr"]
    force = "--ocr" in sys.argv[1:]
    if not args:
        print('usage: python scripts/add_pdfs.py "<pdf_dir>" [index] [--ocr]'); sys.exit(1)
    main(*args, force_ocr=force)
