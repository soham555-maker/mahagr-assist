"""
verify_corpus.py — prove the FAISS index and the SQLite corpus still agree.

THE FAILURE THIS EXISTS TO CATCH
--------------------------------
`gr_chunks.faiss_id` IS the vector's position in `corpus.hnsw`. That single
integer is the entire join between the two files, and the two files cannot be
written in one transaction. If they drift by even a few rows, nothing crashes:
every retrieved vector hydrates to the WRONG chunk, so the system keeps
answering fluently and confidently while citing the wrong Government
Resolution. For a tool whose whole promise is "source-grounded", that is the
worst failure available — and it is invisible from the outside.

`ingest_corpus.py` already checks the cheap half (`len(store) == COUNT(*)`).
Equal counts do NOT prove equal ALIGNMENT: an off-by-500 shift in the middle of
the corpus leaves both numbers identical. So this script checks alignment
directly.

THE THREE CHECKS
----------------
1. COUNTS — `len(store)` vs `COUNT(*) FROM gr_chunks`, and that the faiss_id
   space is exactly 0..n-1 with no gaps or duplicates. Cheap, catches the
   crude failures.

2. ALIGNMENT (the real one) — take a random sample of chunks, re-embed their
   stored TEXT, and ask the index for the nearest vector. If SQLite row
   `faiss_id=k` really is the vector at position k, that vector is the
   embedding of this exact text, so the self-similarity must be ~1.0. A
   shifted index scores like an unrelated chunk (typically 0.3-0.8) and is
   caught immediately. This is done with `score_for_index`, which reads the
   vector at a KNOWN position — not with a search, because a search could
   return a near-duplicate neighbouring chunk and mask the shift.

3. REFERENTIAL — every `gr_chunks.gr_id` points at a real `gr_documents` row,
   and every document has at least one chunk.

Embedding a sample is the only slow part, and it runs fine on CPU: --sample 200
takes well under a minute and needs no GPU, so this is safe to run while the
API is up.

Usage:
    python scripts/verify_corpus.py                 # 200-chunk sample
    python scripts/verify_corpus.py --sample 1000   # more paranoid
"""

import argparse
import os
import random
import sys

DEFAULT_INDEX = os.environ.get("MAHAGR_INDEX_DIR", "/mnt/win/mahagr/index")

# A correctly aligned chunk re-embeds to itself. It is not EXACTLY 1.0: the
# corpus was embedded in fp16 on the GPU (config.EMBED_FP16) and this check
# re-embeds in fp32 on the CPU, and the measured cosine agreement between those
# two is >= 0.9997. A misaligned chunk is a different piece of text entirely and
# scores far below this, so the threshold sits in a very wide gap.
SELF_SIM_MIN = 0.98


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--sample", type=int, default=200,
                    help="chunks to re-embed for the alignment check")
    ap.add_argument("--seed", type=int, default=0, help="0 = random each run")
    args = ap.parse_args()

    from engine import corpus_db
    from engine.vector_store import HnswStore

    db_path = os.path.join(args.index, "corpus.db")
    if not os.path.exists(db_path):
        print(f"No corpus at {db_path}.")
        return 1

    store = HnswStore.load(args.index)
    failures = []

    with corpus_db.connect(db_path, readonly=True) as conn:
        n_chunks, lo, hi, n_distinct = conn.execute(
            "SELECT COUNT(*), MIN(faiss_id), MAX(faiss_id), COUNT(DISTINCT faiss_id) "
            "FROM gr_chunks").fetchone()
        n_docs = conn.execute("SELECT COUNT(*) FROM gr_documents").fetchone()[0]

        print("1. COUNTS")
        print(f"   FAISS vectors      {len(store)}")
        print(f"   gr_chunks rows     {n_chunks}")
        print(f"   gr_documents rows  {n_docs}")
        print(f"   faiss_id range     {lo}..{hi}  ({n_distinct} distinct)")
        if len(store) != n_chunks:
            failures.append(f"vector count {len(store)} != chunk rows {n_chunks}")
        if n_chunks and (lo != 0 or hi != n_chunks - 1 or n_distinct != n_chunks):
            failures.append(f"faiss_id space is not a dense 0..{n_chunks - 1}")
        print("   " + ("OK" if not failures else "FAIL"))

        print("\n2. ALIGNMENT — re-embedding a sample and scoring it against its OWN vector")
        rng = random.Random(args.seed or None)
        ids = sorted(rng.sample(range(min(len(store), n_chunks)),
                                min(args.sample, len(store), n_chunks)))
        by_id = corpus_db.chunks_by_faiss_ids(conn, ids)   # {faiss_id: chunk-dict}

        from engine.ingest import IngestionPipeline
        model = IngestionPipeline().model
        texts = [by_id[i]["text"] for i in ids if i in by_id]
        kept = [i for i in ids if i in by_id]
        if len(kept) != len(ids):
            failures.append(f"{len(ids) - len(kept)} sampled faiss_ids had no chunk row")

        embeddings = model.encode(texts, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
        worst, worst_id, bad = 1.0, None, 0
        for fid, emb in zip(kept, embeddings):
            score = float(store.score_for_index(fid, emb))
            if score < worst:
                worst, worst_id = score, fid
            if score < SELF_SIM_MIN:
                bad += 1
                if bad <= 5:
                    print(f"   ! faiss_id {fid} scored {score:.4f} against its own "
                          f"stored text (gr {by_id[fid]['metadata']['order_id']})")
        print(f"   sampled {len(kept)} chunks · worst self-similarity {worst:.4f} "
              f"(faiss_id {worst_id}) · {bad} below {SELF_SIM_MIN}")
        if bad:
            failures.append(f"{bad}/{len(kept)} sampled chunks do not match their own vector "
                            f"— the index and the DB are MISALIGNED")
        print("   " + ("OK" if not bad else "FAIL"))

        print("\n3. REFERENTIAL INTEGRITY")
        orphan_chunks = conn.execute(
            "SELECT COUNT(*) FROM gr_chunks WHERE gr_id NOT IN "
            "(SELECT id FROM gr_documents)").fetchone()[0]
        empty_docs = conn.execute(
            "SELECT COUNT(*) FROM gr_documents WHERE id NOT IN "
            "(SELECT DISTINCT gr_id FROM gr_chunks)").fetchone()[0]
        print(f"   chunks with no document  {orphan_chunks}")
        print(f"   documents with no chunk  {empty_docs}")
        if orphan_chunks:
            failures.append(f"{orphan_chunks} chunks reference a missing document")
        if empty_docs:
            failures.append(f"{empty_docs} documents have no chunks")
        print("   " + ("OK" if not (orphan_chunks or empty_docs) else "FAIL"))

    print("\n" + "=" * 66)
    if failures:
        print("CORPUS IS NOT CONSISTENT:")
        for f in failures:
            print(f"  - {f}")
        print("\nThe saved FAISS index is the authority. Re-run ingest_corpus.py:\n"
              "  it drops chunk rows past the index's ntotal and re-ingests those docs.")
        return 1
    print(f"CORPUS CONSISTENT — {len(store)} vectors aligned with {n_chunks} chunk "
          f"rows across {n_docs} documents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
