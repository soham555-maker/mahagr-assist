"""
ingest_grs.py — build the MahaGR FAISS index from a folder of GR PDFs.

The dependency-light ingestion path (no Supabase / Storage, unlike
build_index.py which persists figure crops for the research-corpus pipeline).
It uses IngestionPipeline.process_pdf (v1): PyMuPDF text extraction with an OCR
fallback for scanned pages (see ingest._ocr_page) plus pdfplumber table
chunks — which is the right default for Government Resolutions, where scanned
Marathi documents and fee/seat tables matter more than figure crops.

Every chunk is embedded with bge-m3 (config.EMBED_MODEL) and tagged with the
source filename, so citations resolve to a real GR file the officer can open.

Usage:
    python scripts/ingest_grs.py                       # data/grs -> index/
    python scripts/ingest_grs.py data/grs index/       # explicit paths
"""

import os
import sys
import time

from engine import config
from engine.ingest import IngestionPipeline
from engine.vector_store import FaissStore


def main(pdf_dir="data/grs", index_dir="index"):
    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")) \
        if os.path.isdir(pdf_dir) else []
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}/. Drop some GR PDFs there and re-run.")
        return

    print(f"Loading embedding model {config.EMBED_MODEL} (downloads on first run)...")
    pipeline = IngestionPipeline()               # bge-m3
    store = FaissStore()                         # dim = config.EMBED_DIM (1024)

    processed, skipped, failed = 0, 0, []
    for i, name in enumerate(pdfs, start=1):
        path = os.path.join(pdf_dir, name)
        t0 = time.time()
        try:
            chunks = pipeline.process_pdf(
                path,
                source_type="gr",
                include_tables=True,
                # Minimal metadata for the baseline: the filename is the human
                # handle for citations. Richer GR metadata (number, department,
                # date, language) is a later domain step — see README roadmap.
                extra_metadata={"title": os.path.splitext(name)[0]},
            )
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] FAILED  {name}: {e}")
            failed.append((name, str(e)))
            continue

        if not chunks:
            print(f"[{i}/{len(pdfs)}] EMPTY   {name}  (no text even after OCR)")
            skipped += 1
            continue

        store.add(chunks)
        n_tab = sum(1 for c in chunks if c["metadata"]["content_type"] == "table")
        print(f"[{i}/{len(pdfs)}] ok      {name}  (+{len(chunks)} chunks, "
              f"{n_tab} table, {time.time() - t0:.1f}s, index now {len(store)})")
        processed += 1

    print("\n" + "=" * 60)
    print(f"Processed {processed}, skipped {skipped}, failed {len(failed)}.")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")

    if len(store) == 0:
        print("Index is empty — nothing to save.")
        return
    store.save(index_dir)
    print(f"Saved index to {index_dir}/ ({len(store)} vectors, dim {store.dim}).")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()
