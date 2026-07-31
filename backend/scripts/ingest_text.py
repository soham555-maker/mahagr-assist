"""
ingest_text.py — build the FAISS index from a folder of pre-extracted GR text
files (the orgpedia '*.mr.txt' corpus fetched by fetch_mahgrs.py).

Same index/metadata contract as ingest_grs.py, but the text is already OCR'd
upstream, so there's no PDF/OCR/table step — just chunk + embed (bge-m3) via
IngestionPipeline.process_text. Per-document metadata is parsed with
gr_metadata; the order id (first 8 digits = YYYYMMDD) is used as a reliable
date fallback when the header date doesn't parse cleanly.

Usage:
    python scripts/ingest_text.py                     # data/grs_text -> index/
    python scripts/ingest_text.py data/grs_text index
"""

import os
import re
import sys
import time

from engine import config, gr_metadata
from engine.ingest import IngestionPipeline
from engine.vector_store import FaissStore


def _order_id(name):
    """'201710121514029708.pdf.mr.txt' -> '201710121514029708'."""
    return name.split(".")[0]


def _date_from_id(order_id):
    """First 8 digits of the order id are YYYYMMDD."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})", order_id)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def main(text_dir="data/grs_text", index_dir="index"):
    files = sorted(f for f in os.listdir(text_dir) if f.endswith(".mr.txt")) \
        if os.path.isdir(text_dir) else []
    if not files:
        print(f"No .mr.txt files in {text_dir}/. Run scripts/fetch_mahgrs.py first.")
        return

    print(f"Loading embedding model {config.EMBED_MODEL} (downloads on first run)...")
    pipeline = IngestionPipeline()          # bge-m3
    store = FaissStore()                    # dim = config.EMBED_DIM

    processed, skipped = 0, 0
    for i, name in enumerate(files, 1):
        order_id = _order_id(name)
        text = open(os.path.join(text_dir, name), encoding="utf-8").read()

        meta = gr_metadata.extract(text)
        meta.setdefault("date", _date_from_id(order_id))     # reliable fallback
        meta.setdefault("title", order_id)
        meta["order_id"] = order_id

        try:
            chunks = pipeline.process_text(text, source_file=f"{order_id}.pdf",
                                           source_type="gr", extra_metadata=meta)
        except Exception as e:
            print(f"[{i}/{len(files)}] FAILED  {order_id}: {e}")
            continue
        if not chunks:
            skipped += 1
            continue

        store.add(chunks)
        processed += 1
        if i % 25 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {order_id}  gr={meta.get('gr_number')}  "
                  f"date={meta.get('date')}  (+{len(chunks)} chunks, index {len(store)})")

    print("\n" + "=" * 60)
    print(f"Processed {processed}, skipped {skipped} (empty).")
    if len(store) == 0:
        print("Index empty — nothing saved.")
        return
    store.save(index_dir)
    print(f"Saved index to {index_dir}/ ({len(store)} vectors, dim {store.dim}).")


if __name__ == "__main__":
    main(*sys.argv[1:]) if len(sys.argv) > 1 else main()
