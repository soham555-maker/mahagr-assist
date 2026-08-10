"""
ingest_corpus.py — build the multi-department corpus index (PLAN Phase 2).

Reads the per-department folders fetch_mahgrs.py downloaded, chunks every GR,
embeds the chunks on the GPU in large batches, and writes the two files that
together ARE the corpus:

    <index>/corpus.hnsw   FAISS IndexHNSWFlat   (the vectors, ~O(log n) search)
    <index>/corpus.db     SQLite                (documents, chunk text, FTS5 BM25)

They are joined by ONE integer: gr_chunks.faiss_id is the vector's position in
the FAISS index. FAISS gives back positions; SQLite turns positions into text.

WHY THIS SCRIPT IS RESUMABLE AND IDEMPOTENT
-------------------------------------------
This is a ~40-minute GPU job over 18,000 files. It WILL be interrupted — a
laptop sleeps, the OOM killer fires, someone needs the GPU back. So:

  * RESUMABLE — every document already in gr_documents is skipped, so a re-run
    costs a single SELECT and picks up exactly where it stopped.
  * IDEMPOTENT — running it twice produces the same corpus, not a doubled one
    (upsert_document is INSERT OR REPLACE, keyed by the order id).
  * CRASH-CONSISTENT — the two files can't be written in one atomic
    transaction, so the SAVED FAISS INDEX IS THE AUTHORITY. SQLite commits every
    batch; FAISS is checkpointed every --checkpoint documents. After a crash
    SQLite is therefore AHEAD of the index by up to one checkpoint, and those
    orphan rows are deleted on the next run (corpus_db.delete_chunks_from).
    Getting this backwards would be silent and disastrous: every chunk after the
    crash point would be off by a few positions, so every citation would name
    the wrong GR while looking perfectly plausible.

DEVICE
------
Bulk embedding wants the WHOLE GPU: run with EMBED_DEVICE unset (or =cuda) and
Ollama stopped. At serve time the opposite is true — EMBED_DEVICE=cpu, GPU to
the LLM (HANDOFF §5.20). The script prints which device it got, because getting
this wrong is a 10x slowdown that otherwise looks like normal slowness.

Usage:
    python scripts/ingest_corpus.py                       # all departments
    python scripts/ingest_corpus.py --limit 200           # smoke test
    python scripts/ingest_corpus.py --dept Tribal_Development_Department
    python scripts/ingest_corpus.py --reset               # start over
"""

import argparse
import os
import re
import sys
import time

from engine import config, corpus_db, gr_metadata
from engine.ingest import IngestionPipeline
from engine.vector_store import HnswStore

DEFAULT_CORPUS = os.environ.get("MAHAGR_CORPUS", "/mnt/win/mahagr/corpus")
DEFAULT_INDEX = os.environ.get("MAHAGR_INDEX_DIR", "/mnt/win/mahagr/index")


def _order_id(name):
    """'201710121514029708.pdf.mr.txt' -> '201710121514029708'."""
    return name.split(".")[0]


def _date_from_id(order_id):
    """First 8 digits of the order id are YYYYMMDD — a reliable date even when
    the GR's own header date is mangled by OCR."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})", order_id)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _department_label(folder):
    """'Higher_and_Technical_Education_Department' -> 'Higher and Technical
    Education Department'. The FOLDER is the authoritative department, not the
    line gr_metadata scrapes from the text: orgpedia filed the document, whereas
    the in-text department line is OCR output and is often missing or garbled.
    Filtering by department has to be exact, so it keys off the reliable one."""
    return folder.replace("_", " ").replace(" and ", " and ")


def discover(corpus_root, depts=None):
    """[(department_folder, filepath), ...] sorted by department then order id
    (which is chronological — the id starts YYYYMMDD)."""
    found = []
    if not os.path.isdir(corpus_root):
        return found
    for folder in sorted(os.listdir(corpus_root)):
        path = os.path.join(corpus_root, folder)
        if not os.path.isdir(path) or (depts and folder not in depts):
            continue
        for name in sorted(os.listdir(path)):
            if name.endswith(".mr.txt"):
                found.append((folder, os.path.join(path, name)))
    return found


def reconcile(conn, store):
    """Make SQLite agree with the saved FAISS index before adding anything.

    See the crash-consistency note at the top: the index is the authority, so
    any chunk row pointing past its last vector is dropped, and any document
    left with no chunks is dropped too so it gets re-ingested cleanly.
    """
    orphans = corpus_db.delete_chunks_from(conn, len(store))
    if orphans:
        conn.execute("""DELETE FROM gr_documents
                        WHERE id NOT IN (SELECT DISTINCT gr_id FROM gr_chunks)""")
        conn.commit()
        print(f"  recovered from an interrupted run: dropped {orphans} orphan chunk rows")
    return orphans


def flush(conn, store, pending, model, batch_size):
    """Embed one accumulated batch of chunks and write it to BOTH stores.

    pending: [(doc_id, chunk_dict), ...] collected across several documents so
    the GPU gets a batch worth filling. sentence-transformers sorts by length
    internally before batching, so a mixed-length batch does not waste compute
    on padding.
    """
    if not pending:
        return 0
    texts = [c["text"] for _, c in pending]
    embeddings = model.encode(texts, batch_size=batch_size,
                              normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)

    # FAISS first: it hands back the starting position that keys the SQL rows.
    start = store.add_vectors(embeddings)
    rows = [(start + i, doc_id, c.get("chunk_index", i),
             c["page_start"], c["page_end"], c["content_type"], c["text"])
            for i, (doc_id, c) in enumerate(pending)]
    corpus_db.insert_chunks(conn, rows)
    conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--dept", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0, help="max documents (0 = all)")
    ap.add_argument("--batch-chunks", type=int, default=512,
                    help="chunks accumulated before one encode() call")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="GPU batch size inside encode()")
    ap.add_argument("--checkpoint", type=int, default=500,
                    help="save the FAISS index every N documents")
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--reset", action="store_true", help="delete the index and start over")
    args = ap.parse_args()

    files = discover(args.corpus, set(args.dept) if args.dept else None)
    if not files:
        print(f"No .mr.txt files under {args.corpus}/. Run scripts/fetch_mahgrs.py first.")
        return 1

    db_path = os.path.join(args.index, "corpus.db")
    if args.reset:
        for p in (db_path, os.path.join(args.index, "corpus.hnsw"),
                  os.path.join(args.index, "hnsw_meta.json"), db_path + "-wal",
                  db_path + "-shm"):
            if os.path.exists(p):
                os.remove(p)
        print("reset: removed the existing index")

    os.makedirs(args.index, exist_ok=True)
    corpus_db.init(db_path)
    store = HnswStore.load_or_new(args.index)

    print(f"corpus     {args.corpus}  ({len(files)} GR files)")
    print(f"index      {args.index}")
    print(f"embedder   {config.EMBED_MODEL} on "
          f"{config.EMBED_DEVICE or 'auto (cuda if available)'}")
    print(f"hnsw       M={store.m} efConstruction={store.ef_construction} "
          f"efSearch={store.ef_search}  (existing vectors: {len(store)})")

    pipeline = IngestionPipeline()
    device = getattr(pipeline.model, "device", "?")
    print(f"           model loaded on device: {device}")
    if str(device).startswith("cpu"):
        print("           ! CPU embedding is ~10x slower — unset EMBED_DEVICE and "
              "stop Ollama to use the GPU.")

    with corpus_db.connect(db_path) as conn:
        reconcile(conn, store)
        done = corpus_db.ingested_ids(conn)
        todo = [(d, p) for d, p in files if _order_id(os.path.basename(p)) not in done]
        if args.limit:
            todo = todo[:args.limit]
        print(f"already ingested {len(done)}; {len(todo)} to do\n")
        if not todo:
            print("Nothing to do — the corpus is up to date.")
            return 0

        started = time.time()
        pending, n_docs, n_chunks, n_table_chunks, n_empty, since_ckpt = [], 0, 0, 0, 0, 0

        for i, (folder, path) in enumerate(todo, 1):
            order_id = _order_id(os.path.basename(path))
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError as e:
                print(f"  ! unreadable {order_id}: {e}")
                continue

            chunks = pipeline.chunk_text(text, chunk_size=args.chunk_size,
                                         overlap=args.overlap)
            n_table = sum(1 for c in chunks if c.get("content_type") == "table")
            n_table_chunks += n_table
            if not chunks:
                n_empty += 1
                continue

            meta = gr_metadata.extract(text)
            meta["date"] = meta.get("date") or _date_from_id(order_id)
            meta["title"] = meta.get("title") or order_id
            # Folder wins: see _department_label.
            meta["department"] = _department_label(folder)
            corpus_db.upsert_document(conn, order_id, meta, text, len(chunks))

            for idx, c in enumerate(chunks):
                c["chunk_index"] = idx
                pending.append((order_id, c))
            n_docs += 1
            since_ckpt += 1

            if len(pending) >= args.batch_chunks:
                n_chunks += flush(conn, store, pending, pipeline.model, args.batch_size)
                pending = []

            if since_ckpt >= args.checkpoint:
                n_chunks += flush(conn, store, pending, pipeline.model, args.batch_size)
                pending = []
                store.save(args.index)
                since_ckpt = 0

            if i % 250 == 0 or i == len(todo):
                rate = i / max(time.time() - started, 1e-9)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"  [{i}/{len(todo)}] docs={n_docs} chunks={n_chunks + len(pending)} "
                      f"(tables={n_table_chunks}) vectors={len(store)}  {rate:.1f} docs/s  eta {eta/60:.1f} min",
                      flush=True)

        n_chunks += flush(conn, store, pending, pipeline.model, args.batch_size)
        store.save(args.index)
        stats = corpus_db.stats(conn)

    elapsed = time.time() - started
    print("\n" + "=" * 66)
    print(f"Ingested {n_docs} documents / {n_chunks} chunks ({n_table_chunks} table chunks) in {elapsed/60:.1f} min "
          f"({n_empty} empty, skipped)")
    print(f"Corpus now: {stats['documents']} GRs, {stats['chunks']} chunks, "
          f"{len(stats['departments'])} departments, {stats['date_from']} .. {stats['date_to']}")
    for d in stats["departments"]:
        print(f"   {d['name']:55s} {d['documents']:6d}")
    print(f"FAISS vectors: {len(store)}  ->  {args.index}/")
    if len(store) != stats["chunks"]:
        print(f"! MISMATCH: {len(store)} vectors vs {stats['chunks']} chunk rows")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
