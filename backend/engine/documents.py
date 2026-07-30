"""
documents.py — user-uploaded papers, ingested through the SAME pipeline as
the shared corpus, stored per-user, and merged into retrieval at query time.

WHY A SEPARATE STORE, NOT THE SHARED FAISS INDEX
--------------------------------------------------
Same split PLAN.md always drew for chat memory, applied here to uploads: the
corpus is shared, read-only, identical for everyone -> in-memory FAISS. An
uploaded paper is private and mutable (a user can delete it) and must
survive restarts -> Postgres rows tagged user_id, exactly like
conversations/messages (see api.py). Per-user scale stays small (a handful
of papers, at most a few hundred chunks), so brute-force cosine in Python is
the right call here too — the same "no premature approximation" reasoning
vector_store.py already uses for the flat FAISS corpus index.

SAME PIPELINE, SAME SHAPE, ZERO TRANSLATION
--------------------------------------------
ingest_upload() below is a thin wrapper over ingest.py's IngestionPipeline —
the exact code that builds the corpus, not a parallel reimplementation. Its
output ({'text', 'embedding', 'metadata'}) is stored with `metadata` as-is
(jsonb), so a document_chunks row and a FaissStore hit are structurally
identical. hybrid_search() below returns hits in the same {'score', 'text',
'metadata'} shape FaissStore.search() does, so rag.py's format_block() and
resolve_citations() need no changes to handle either source — they were
already written generically over "a hit dict," not "a FAISS hit."

MERGE AT QUERY TIME, NOT A SEPARATE FEATURE
--------------------------------------------
retrieval.py's Retriever is untouched — it remains "the shared corpus
retrieval contract." A user's document hits are fetched separately (by
api.py, see /chat) and merged into rag.answer()'s result by score. Same
model + same metric -> scores are commensurable, so a plain score-sorted
merge is valid (no rank fusion needed) — the identical argument PLAN.md
always made for merging corpus + per-user memory.

UNCALIBRATED THRESHOLD, HONESTLY
----------------------------------
retrieval.py's 0.69/0.66 thresholds were measured against a 21-question gold
set on THIS corpus (docs/rag.md §3). No such gold set exists for an
arbitrary user's own upload, so DOCUMENT_SEARCH_THRESHOLD below is a
deliberately lenient, uncalibrated default: false negatives are worse here
(a user asking about their own paper who gets nothing back) than false
positives (the candidate pool is just their own few papers, not a whole
corpus of distractors). Revisit with real data if it misbehaves.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor

# pyrefly: ignore [missing-import]
import numpy as np

from engine import hybrid

ASSET_BUCKET = "document-assets"  # private Storage bucket for figure/formula crops.
# Shared by BOTH owners: uploads live at "{user_id}/{document_id}/{asset_id}.png",
# the shared corpus lives at "corpus/{paper_id}/{asset_id}.png" — the literal
# "corpus/" prefix can never collide with a user_id (always a UUID), so one
# bucket serves both without a second bucket to create/manage/clean up.

DOCUMENT_SEARCH_THRESHOLD = 0.55
DOCUMENT_SEARCH_K = 6
DOCUMENT_CANDIDATE_K = 20  # dense + BM25 candidate pool size before RRF fusion
# Rerank-logit keep-gate for uploads. Same value as the corpus's calibrated
# cutoff (RetrievalConfig.rerank_threshold, mid-gap of the measured in/out
# separation) — the same cross-encoder scores both paths, so the same scale
# applies. Like the cosine 0.55 above, it leans lenient for uploads: a user's
# own paper failing to surface is worse than a marginal chunk slipping in.
DOCUMENT_RERANK_THRESHOLD = -6.0
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB — generous for a single paper, cheap to reject early


def ingest_upload(pdf_path, pipeline, document_id, title):
    """
    Run the ingestion pipeline (v2 — layout model, adds figure/formula chunks)
    on one uploaded PDF. Returns ingest.py's usual
    [{'text', 'embedding', 'metadata'}, ...], tagged source_type='upload' and
    paper_id=document_id so citations still resolve to (title, pages) — plus,
    for figure/formula chunks, a transient 'image_bytes' the caller persists to
    Storage (see api.py). Falls back to v1 internally if the layout model fails
    on a given PDF, so an upload never regresses to an error.
    """
    return pipeline.process_pdf_v2(
        pdf_path,
        source_type="upload",
        extra_metadata={"paper_id": str(document_id), "title": title},
    )


def hybrid_search(query, query_vec, chunks, k=DOCUMENT_SEARCH_K,
                  threshold=DOCUMENT_SEARCH_THRESHOLD, candidate_k=DOCUMENT_CANDIDATE_K,
                  reranker=None, rerank_threshold=DOCUMENT_RERANK_THRESHOLD):
    """
    Hybrid retrieval over a user's uploaded chunks: dense cosine + BM25 keyword,
    RRF-fused, optionally cross-encoder reranked — the per-user analogue of the
    corpus Retriever's path. Same {'score', 'text', 'metadata'} output shape as
    the cosine-only predecessor this replaced (see legacy/cosine_search.py),
    so the generation merge (rag.merge_extra_chunks) needed no changes.

    `query` is the raw question text (for BM25 + reranking); `query_vec` its
    embedding (for cosine). BM25 is built per call — per-user chunk counts are
    small, so this is cheap, and it keeps uploads consistent with the corpus's
    flat-in-memory philosophy (no persistent per-user keyword index to maintain).

    reranker: the SAME Reranker instance the corpus path uses (api.py passes
    it), so upload and corpus scores stay commensurable and answer()'s merge by
    score remains valid. With a reranker, the keep-gate reads the rerank logit
    (rerank_threshold); without one, the cosine threshold gates (Phase-1).
    No floor here, deliberately: an empty result just means "nothing from your
    uploads was relevant" — the corpus is the primary source and has its own
    floor, so the upload side must be allowed to contribute nothing.
    """
    if not chunks:
        return []

    # pyrefly: ignore [missing-import]
    from rank_bm25 import BM25Okapi

    vectors = np.asarray([c["embedding"] for c in chunks], dtype="float32")
    cosines = vectors @ np.asarray(query_vec, dtype="float32")
    dense_order = [int(i) for i in np.argsort(-cosines)[:candidate_k]]

    bm25 = BM25Okapi([hybrid.tokenize(c["text"]) for c in chunks])
    bm25_scores = bm25.get_scores(hybrid.tokenize(query))
    sparse_order = [int(i) for i in np.argsort(bm25_scores)[::-1][:candidate_k]
                    if bm25_scores[i] > 0]

    fused_order, _ = hybrid.rrf_fuse([dense_order, sparse_order])

    if reranker is not None:
        fused_hits = [{
            "score": float(cosines[pos]),
            "text": chunks[pos]["text"],
            "metadata": chunks[pos]["metadata"],
        } for pos in fused_order[:candidate_k]]
        ranked = reranker.rerank(query, fused_hits)
        return [h for h in ranked if h["score"] >= rerank_threshold][:k]

    results = []
    for pos in fused_order:
        if cosines[pos] < threshold:
            continue  # not break: fused order isn't cosine-sorted
        results.append({
            "score": float(cosines[pos]),
            "text": chunks[pos]["text"],
            "metadata": chunks[pos]["metadata"],
        })
        if len(results) == k:
            break
    return results


def persist_visual_assets(supabase, chunks, path_prefix):
    """
    Upload every visual chunk's PNG crop (chunk['image_bytes']) to Storage
    under path_prefix, stamp the object path into chunk['metadata']['image_path'],
    and drop image_bytes so the chunk is a plain {text, embedding, metadata} row
    again. Shared by BOTH callers:
      - api.py's upload handler: path_prefix = "{user_id}/{document_id}"
      - build_index.py's corpus build: path_prefix = "corpus/{paper_id}"
    One implementation, so uploads and the corpus persist images identically —
    the "full parity" requirement. Uploads run concurrently (network-bound),
    with a bounded retry per file: transient read/connection errors are common
    on a small thread pool hammering the same host, and they must not lose an
    otherwise-successful ingest (all the expensive extraction+embedding work
    is already done by the time this runs) — a chunk that still fails after
    retries just keeps no image_path (degrades to text-only, doesn't crash).
    """
    visual = [c for c in chunks if "image_bytes" in c]
    if not visual:
        return

    def mark_uploaded(chunk, asset_id, path):
        chunk["metadata"]["image_path"] = path
        chunk["metadata"]["asset_id"] = asset_id
        del chunk["image_bytes"]

    def upload_one(chunk):
        asset_id = str(uuid.uuid4())
        path = f"{path_prefix}/{asset_id}.png"
        last_err = None
        for attempt in range(3):
            try:
                supabase.storage.from_(ASSET_BUCKET).upload(
                    path, chunk["image_bytes"], {"content-type": "image/png"}
                )
                mark_uploaded(chunk, asset_id, path)
                return
            except Exception as e:
                # A 409 "already exists" on THIS freshly-minted UUID can only
                # mean our own earlier attempt actually succeeded server-side
                # and we just never saw the response (the read timed out, but
                # the write landed) -- observed in practice on the corpus
                # rebuild. That's proof of success, not a failure to retry:
                # treat it as done rather than discarding a real, uploaded
                # image because its ack got lost.
                if getattr(e, "status", None) == 409 or "already exists" in str(e).lower():
                    mark_uploaded(chunk, asset_id, path)
                    return
                last_err = e
                time.sleep(0.5 * (attempt + 1))  # brief backoff, not a 429 protocol
        print(f"  [warn] asset upload failed after 3 attempts ({path}): {last_err}")
        del chunk["image_bytes"]  # degrade to text-only rather than leak bytes into a DB row

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(upload_one, visual))


def clean_storage_prefix(supabase, prefix):
    """Remove every Storage object under prefix. Used to wipe stale corpus
    assets before a rebuild (idempotent re-runs don't accumulate orphans) and
    is the same primitive api.py's per-document delete uses for uploads."""
    try:
        listed = supabase.storage.from_(ASSET_BUCKET).list(prefix)
    except Exception:
        return
    paths = [f"{prefix}/{obj['name']}" for obj in listed if obj.get("name")]
    if paths:
        supabase.storage.from_(ASSET_BUCKET).remove(paths)


def new_document_id():
    """A document's id is generated up front (not left to the DB default)
    because ingest_upload() needs to stamp it onto every chunk's metadata
    BEFORE any row exists to read the id back from."""
    return uuid.uuid4()
