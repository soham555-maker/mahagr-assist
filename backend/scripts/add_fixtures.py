"""
add_fixtures.py — append the sample fee-table GRs (data/fixtures/*.pdf) to an
existing index, so a demo can show TABLE + exact-number retrieval alongside the
real Marathi GR corpus.

The real corpus (orgpedia .mr.txt) is OCR'd prose with no structured tables; the
fixtures go through the PDF path (process_pdf -> pdfplumber table extraction),
which is what demonstrates that fee tables and numbers survive ingestion.

Usage:  python scripts/add_fixtures.py            # index/ <- + data/fixtures/*.pdf
"""

import os
import sys

from engine import gr_metadata
from engine.ingest import IngestionPipeline
from engine.vector_store import FaissStore

FIX_DIR = "data/fixtures"


def main(index_dir="index"):
    store = FaissStore.load(index_dir)
    print(f"Loaded index: {len(store)} vectors.")
    pipeline = IngestionPipeline()  # bge-m3 (must match the index's model)

    pdfs = [f for f in os.listdir(FIX_DIR) if f.lower().endswith(".pdf")]
    for name in sorted(pdfs):
        path = os.path.join(FIX_DIR, name)
        pages = pipeline.extract_pages_from_pdf(path)
        meta = gr_metadata.extract("\n".join(t for _, t in pages))
        meta["order_id"] = os.path.splitext(name)[0]
        meta.setdefault("title", meta["order_id"])
        chunks = pipeline.process_pdf(path, source_type="gr", include_tables=True,
                                      extra_metadata=meta, pages=pages)
        n_tab = sum(1 for c in chunks if c["metadata"]["content_type"] == "table")
        store.add(chunks)
        print(f"  + {name}: {len(chunks)} chunks ({n_tab} table)  gr={meta.get('gr_number')}")

    store.save(index_dir)
    print(f"Saved. Index now {len(store)} vectors.")


if __name__ == "__main__":
    main(*sys.argv[1:]) if len(sys.argv) > 1 else main()
