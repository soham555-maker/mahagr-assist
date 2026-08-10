"""
Retriever — the Day 4 box: question in, ranked relevant chunks out.

This is the deterministic half of the RAG loop. No LLM, no network, no
randomness: the same question always retrieves the same chunks, which is what
makes this layer testable with real assertions (test_retrieval.py) and
measurable with a gold set (eval_retrieval.py). Day 5's rag.py sits on top and
only formats + sends what this returns.

THE RETRIEVAL CONTRACT: threshold-filtered, capped, floored, flagged.
----------------------------------------------------------------------
* THRESHOLD, not fixed top-k: keep chunks scoring above a per-modality cosine
  cutoff, so context size adapts to how much the corpus knows about the
  question. Fixed k pads narrow questions with junk and starves broad ones.
* CAPPED: max_text_k / max_table_k stop a broad query from passing dozens of
  chunks and overflowing the prompt budget.
* FLOORED: if nothing clears a threshold, keep the overall top floor_k anyway.
  Empty context is the most dangerous state for RAG — the LLM would answer
  from its own weights, fluently and ungrounded.
* FLAGGED: when the floor fires, low_confidence=True rides along, so the
  Day 5 prompt can explicitly invite "the corpus doesn't cover this" instead
  of pretending the floor chunks are good.

PER-MODALITY SEARCH
-------------------
Text and table chunks are searched SEPARATELY (via FaissStore's where= filter)
and merged by score. Table sentences are templated prose outnumbered ~8:1 by
natural prose; in a single top-k they get crowded out even when a table
literally holds the answer. Merging by raw score across the two lists is valid
because both come from the same model and metric — the scores are
commensurable. (Different scorers would need rank fusion instead.)

QUERY EMBEDDING — the one place it happens
------------------------------------------
bge-m3 (the multilingual model this repo uses) takes the raw query text on the
QUERY side — NO instruction prefix, unlike bge-*-en-v1.5, which was fine-tuned
to expect "Represent this sentence...". config.QUERY_PREFIX is therefore ""
here; it stays a config knob so a future model that DOES want a prefix is a
one-line change. normalize_embeddings=True still matters and, like the prefix,
fails SILENTLY if forgotten — so both live in exactly one method: embed_query.

THRESHOLD VALUES — ⚠ NEED RECALIBRATION FOR bge-m3
--------------------------------------------------
The cosine/logit cutoffs below are MODEL-SPECIFIC. The originals were
calibrated for bge-small-en + an English ms-marco reranker on an arXiv gold
set; bge-m3 and bge-reranker-v2-m3 produce different score distributions, and
the corpus is now Marathi/English GRs. The values here are RECALL-LEANING
PLACEHOLDERS chosen to under-filter (a marginal chunk in the prompt is cheap; a
missing answer is fatal — and the floor + the refusal prompt are the real
safety nets). Build a small Marathi/English GR gold set and re-run
eval_retrieval.py `scores` to set these honestly before trusting the numbers.
"""

import os
from dataclasses import dataclass

from engine import config, hybrid
# Alias: several functions below take a RetrievalConfig parameter named
# `config`, which shadows the module. The alias keeps the module reachable
# inside them without renaming a public parameter every caller passes.
from engine import config as engine_config
from engine.vector_store import FaissStore


@dataclass
class RetrievalConfig:
    query_prefix: str = config.QUERY_PREFIX   # "" for bge-m3
    # Calibrated 2026-08-01, re-verified 2026-08-02 on the 23-question gold set
    # (scripts/eval_retrieval.py --no-rerank). Measured bge-m3 cosine: OOC
    # questions topped ~0.54, lowest RELEVANT top hit ~0.563 — so 0.55 still
    # sits in that gap (abstains on OOC, keeps correct hits). Recall-leaning.
    # NOTE: cosine thresholds only apply on the no-reranker path; the deployed
    # path uses the reranker + rerank_threshold below.
    text_threshold: float = 0.55
    table_threshold: float = 0.50   # no table chunks in the text corpus; matters for PDF GRs
    # Dense CANDIDATE generation is widened (vs. the old 8/4 final caps) so RRF
    # has room to promote a chunk that dense ranked just outside the old cap but
    # BM25 ranked high — that recall gain is the point of Phase-1 hybrid.
    candidate_k_text: int = 20
    candidate_k_table: int = 10
    candidate_k_sparse: int = 20   # BM25 keyword pool
    rrf_k: int = 60                # RRF constant (see hybrid.rrf_fuse)
    max_final_k: int = 12          # total chunks kept after fusion+threshold (was 8+4)
    floor_k: int = 2
    # Phase-2 reranking (used only when a Reranker is wired in).
    # 15 -> 40 after measuring on the 18,078-GR corpus (2026-08-06): the deeper
    # pool is the ONLY knob in this dataclass that moved retrieval quality —
    # hit@1 12->13/20, hit@5 14->15/20, MRR 0.642->0.688 — for ~+0.5 s. Widening
    # the ANN/BM25 candidate lists instead changed nothing, which says the right
    # chunk was already in the pool and the cross-encoder simply never got to
    # read it. Peak VRAM is bounded by config.RERANK_BATCH, not by this number.
    rerank_pool: int = 40          # fused candidates handed to the cross-encoder
    # Recalibrated 2026-08-02 (scripts/eval_retrieval.py) on the 196-GR HTE
    # corpus + a 23-question gold set (data/gold/gold.json, EN + Marathi).
    # bge-reranker-v2-m3 gives a 0..1 relevance score: RELEVANT top hits p10
    # 0.966 / median 0.996; out-of-corpus scored 0.001-0.014; in-corpus-but-
    # -irrelevant reached ~0.85. 0.85 abstains cleanly on OOC and keeps every
    # correct hit (hit@1 19/20, hit@5 20/20, MRR 0.975), recall-leaning vs the
    # ~0.92 midpoint. One threshold for all modalities: the cross-encoder reads
    # content, not templated form.
    rerank_threshold: float = 0.85
    # --- scale-path only (CorpusRetriever, PLAN Phase 2) ---
    # At 713 vectors nearly every hit came from a different GR. At 65k, one long
    # GR can easily own the entire top-15 with near-duplicate chunks (the 50-word
    # overlap makes neighbouring chunks genuinely similar), which starves the
    # cross-encoder of alternatives and makes the answer cite one document when
    # three were relevant. Capping chunks per GR before reranking buys DIVERSITY
    # in the pool; 2 keeps a long GR's best passage plus a backup.
    max_chunks_per_gr: int = 2
    # ANN is stage one of two, so it is asked for a wide pool, not a final answer:
    # the cross-encoder does the precision work afterwards. Wider costs ~nothing
    # on HNSW (0.035 ms/query measured) and protects recall through the grouping
    # step above.
    candidate_k_ann: int = 60


class KeywordIndex:
    """BM25 over the corpus chunk texts, index-aligned with FaissStore (chunk i
    in BM25 == vector i in FAISS == texts[i]/metadata[i]). Built once at load;
    lets a query retrieve by literal token overlap, catching rare names /
    acronyms / exact numbers that dense embeddings under-weight."""

    def __init__(self, texts):
        # pyrefly: ignore [missing-import]
        from rank_bm25 import BM25Okapi
        self._tokenized = [hybrid.tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, question, k):
        """Top-k (chunk_index, bm25_score) by BM25, best first, dropping
        zero-score chunks (no query token present)."""
        # pyrefly: ignore [missing-import]
        import numpy as np
        scores = self._bm25.get_scores(hybrid.tokenize(question))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


class Retriever:
    def __init__(self, store, model, config=None, keyword_index=None, reranker=None):
        """
        store: a loaded FaissStore (the shared corpus index).
        model: the SentenceTransformer used at ingestion — queries MUST be
               embedded by the same model that embedded the documents, or the
               two live in different vector spaces and scores are meaningless.
        keyword_index: a KeywordIndex over the same chunks, for hybrid (BM25)
               retrieval. If None, one is built from the store's texts.
        reranker: optional reranker.Reranker (cross-encoder). When present,
               the fused candidates are reranked and the confidence gate reads
               rerank_threshold on the rerank-logit scale; when None, retrieval
               is Phase-1 hybrid (cosine gate) — same contract shape either way.
        """
        self.store = store
        self.model = model
        self.config = config or RetrievalConfig()
        self.keyword = keyword_index or KeywordIndex(store.texts)
        self.reranker = reranker

    def embed_query(self, question):
        """
        The ONE place a query becomes a vector: instruction prefix + unit
        normalization, both mandatory, both silent if wrong — so there is a
        single line in the codebase where they can be wrong. Public: also
        the entry point documents.py's per-user document search uses, so a
        second, possibly-inconsistent embedding path never gets written.
        """
        return self.model.encode(
            [self.config.query_prefix + question],
            normalize_embeddings=True,
        )[0]

    def _search_modality(self, query_vec, content_type, k):
        return self.store.search(
            query_vec, k=k,
            where=lambda m: m.get("content_type") == content_type,
        )

    def _threshold(self, hit):
        ct = hit["metadata"].get("content_type")
        return self.config.table_threshold if ct == "table" else self.config.text_threshold

    def retrieve(self, question):
        """
        Run the full HYBRID contract for one question.

        1. DENSE candidates, per modality (text/table searched separately so a
           table is never crowded out of the pool), widened to candidate_k_*.
        2. SPARSE candidates from BM25 (keyword).
        3. Fuse the dense and sparse rankings with RRF -> one ordering that
           surfaces chunks either method ranked high.
        4. Keep fused chunks that clear their modality's COSINE threshold (the
           OOC/refusal gate stays cosine-based in Phase 1), in fused order,
           capped at max_final_k.
        5. If none clear it -> floor (top fused) + low_confidence=True.

        Returns {'chunks': [...], 'low_confidence': bool}. Chunks are hit dicts
        ({'score'(cosine), 'text', 'metadata', 'index'}) in FUSED order (no
        longer plain cosine-sorted — that's the hybrid change), each carrying
        full provenance metadata for citable context blocks.
        """
        cfg = self.config
        query_vec = self.embed_query(question)

        # 1. Dense per-modality candidates (each hit carries its store 'index').
        dense_hits = (
            self._search_modality(query_vec, "text", cfg.candidate_k_text)
            + self._search_modality(query_vec, "table", cfg.candidate_k_table)
        )
        dense_by_index = {h["index"]: h for h in dense_hits}
        dense_order = [h["index"] for h in
                       sorted(dense_hits, key=lambda h: h["score"], reverse=True)]

        # 2. Sparse (BM25) candidates, same index space.
        sparse_order = [idx for idx, _ in self.keyword.search(question, cfg.candidate_k_sparse)]

        # 3. RRF-fuse the two rankings.
        fused_order, _ = hybrid.rrf_fuse([dense_order, sparse_order], k=cfg.rrf_k)

        # Materialize a hit per fused chunk: reuse the dense hit (has cosine) or,
        # for a BM25-only chunk, reconstruct its vector to get a comparable cosine.
        fused_hits = [
            dense_by_index[idx] if idx in dense_by_index
            else self.store.hit_for_index(idx, query_vec)
            for idx in fused_order
        ]

        # 4a. Phase-2 path: cross-encoder rerank of the fused pool, then the
        # confidence gate reads the RERANK score (calibrated -6.0 cutoff, ~±4
        # margin) instead of cosine (0.02 gap) — the measured refusal-boundary
        # win that justified the reranker. Chunk order = true relevance order.
        if self.reranker is not None:
            ranked = self.reranker.rerank(question, fused_hits[:cfg.rerank_pool])
            kept = [h for h in ranked
                    if h["score"] >= cfg.rerank_threshold][:cfg.max_final_k]
            if kept:
                return {"chunks": kept, "low_confidence": False}
            return {"chunks": ranked[:cfg.floor_k], "low_confidence": True}

        # 4b. Phase-1 path (no reranker): threshold on cosine (per modality),
        # keep fused order, cap the total.
        kept = [h for h in fused_hits if h["score"] >= self._threshold(h)][:cfg.max_final_k]
        if kept:
            return {"chunks": kept, "low_confidence": False}

        # 5. Floor: nothing cleared threshold (hard or out-of-corpus). Keep the
        # top fused chunks anyway and flag it — the prompt turns that into
        # explicit permission to refuse.
        return {"chunks": fused_hits[:cfg.floor_k], "low_confidence": True}


class CorpusRetriever:
    """Two-stage retrieval over the scaled corpus (PLAN Phase 2).

    Same public contract as Retriever — retrieve() returns
    {'chunks': [...], 'low_confidence': bool} with chunks shaped
    {'score', 'text', 'metadata', 'index'} — so rag.py, officer.py and api.py
    did not change when the corpus grew 90x. Only the plumbing underneath is
    different:

        Retriever        FaissStore (O(n), text in RAM) + rank_bm25 (RAM)
        CorpusRetriever  HnswStore  (O(log n), no text) + SQLite FTS5 (disk)

    THE PIPELINE, AND WHY EACH STAGE EXISTS
    ---------------------------------------
      1. EMBED the query once (bge-m3, no prefix, normalized).
      2. ANN over the HNSW graph -> a wide, cheap, approximate candidate pool.
         Optimises RECALL: get the right chunk *somewhere* in the pool.
      3. BM25 over FTS5 -> the same pool from the other direction. Dense
         embeddings blur exact tokens; a GR number, an acronym or a rupee figure
         is precisely what a keyword index is good at. The two fail differently,
         which is the point.
      4. RRF-fuse the two rankings into one order (rank-based, so the two
         incomparable score scales never have to be normalised against each
         other).
      5. HYDRATE the fused pool's text from SQLite in ONE query. Nothing before
         this point has touched chunk text — that is what keeps memory flat.
      6. GROUP BY GR and cap chunks per GR, so the rerank pool covers several
         documents instead of one long one (see max_chunks_per_gr).
      7. RERANK with the cross-encoder -> precision. This is the stage that
         actually decides the answer, and the only one that reads the query and
         the chunk together.
      8. GATE on the calibrated rerank threshold; if nothing clears it, return
         the floor and flag low_confidence so the prompt is allowed to refuse.

    Stages 2-4 are optimised for recall and are cheap; stage 7 is expensive
    (a 568M cross-encoder over ~15 pairs) and is why the pool is narrowed first.
    That asymmetry is the entire argument for two-stage retrieval.
    """

    def __init__(self, store, model, config=None, reranker=None, db_path=None):
        self.store = store
        self.model = model
        self.config = config or RetrievalConfig()
        self.reranker = reranker
        self.db_path = db_path

    def embed_query(self, question):
        """Identical contract to Retriever.embed_query — same model, same empty
        prefix, same normalization. Kept as its own method rather than shared by
        inheritance because the two retrievers have nothing else in common."""
        return self.model.encode(
            [self.config.query_prefix + question],
            normalize_embeddings=True,
        )[0]

    def _connect(self):
        """A fresh SQLite connection per call, on purpose.

        FastAPI runs sync endpoints on a threadpool, and a sqlite3 connection
        may not be shared across threads (it raises ProgrammingError). Opening
        one costs tens of microseconds against a query that takes seconds, so
        per-call connections are the simple correct answer here — no pool, no
        thread-local, nothing to leak.
        """
        from engine import corpus_db
        return corpus_db.connect(self.db_path, readonly=True)

    def retrieve(self, question, filters=None):
        """filters: optional {'departments': [...], 'date_from': 'YYYY-MM-DD',
        'date_to': ..., 'language': 'mr'|'en'} — resolved to a set of allowed
        faiss_ids in SQLite and pushed INTO both searches, so filtering never
        silently eats the results (a post-filter would)."""
        from engine import corpus_db

        cfg = self.config
        query_vec = self.embed_query(question)

        with self._connect() as conn:
            allowed = None
            if filters:
                allowed = corpus_db.filter_faiss_ids(
                    conn,
                    departments=filters.get("departments"),
                    date_from=filters.get("date_from"),
                    date_to=filters.get("date_to"),
                    language=filters.get("language"))
                if allowed is not None and not allowed:
                    return {"chunks": [], "low_confidence": True}

            # 2. dense ANN
            dense = self.store.search(query_vec, k=cfg.candidate_k_ann,
                                      allowed_ids=allowed)
            dense_by_id = {h["index"]: h["score"] for h in dense}
            dense_order = [h["index"] for h in dense]

            # 3. sparse BM25 (over-fetched when filtering, since FTS5 knows
            # nothing about departments and its hits are pruned afterwards).
            sparse = corpus_db.search_bm25(
                conn, question,
                cfg.candidate_k_sparse * (4 if allowed else 1))
            sparse_order = [i for i, _ in sparse
                            if allowed is None or i in allowed][:cfg.candidate_k_sparse]

            # 4. fuse
            fused_order, _ = hybrid.rrf_fuse([dense_order, sparse_order], k=cfg.rrf_k)
            if not fused_order:
                return {"chunks": [], "low_confidence": True}

            # 5. hydrate text+metadata for the pool in one query
            pool = fused_order[:max(cfg.rerank_pool * 3, cfg.max_final_k)]
            rows = corpus_db.chunks_by_faiss_ids(conn, pool)

        # 6. group by GR, capped, preserving fused order
        per_gr, hits = {}, []
        for idx in pool:
            row = rows.get(idx)
            if row is None:
                continue                      # index/DB drift; skip rather than crash
            gr_id = row["metadata"]["order_id"]
            if per_gr.get(gr_id, 0) >= cfg.max_chunks_per_gr:
                continue
            per_gr[gr_id] = per_gr.get(gr_id, 0) + 1
            # BM25-only chunks have no cosine yet; reconstruct gives the exact
            # one, so every hit's score is on the same scale. (Explicit `is
            # None` rather than `or`: a genuine cosine of 0.0 is falsy and would
            # be recomputed for nothing.)
            score = dense_by_id.get(idx)
            if score is None:
                score = self.store.score_for_index(idx, query_vec)
            hits.append({
                "score": score,
                "text": row["text"],
                "metadata": row["metadata"],
                "index": idx,
            })

        if not hits:
            return {"chunks": [], "low_confidence": True}

        # 7 + 8. rerank and gate — identical logic and identical calibrated
        # threshold to Retriever, because it is the same cross-encoder on the
        # same score scale.
        if self.reranker is not None:
            ranked = self.reranker.rerank(question, hits[:cfg.rerank_pool])
            kept = [h for h in ranked
                    if h["score"] >= cfg.rerank_threshold][:cfg.max_final_k]
            if kept:
                return {"chunks": kept, "low_confidence": False}
            return {"chunks": ranked[:cfg.floor_k], "low_confidence": True}

        kept = [h for h in hits if h["score"] >= self._threshold(h)][:cfg.max_final_k]
        if kept:
            return {"chunks": kept, "low_confidence": False}
        return {"chunks": hits[:cfg.floor_k], "low_confidence": True}

    def _threshold(self, hit):
        ct = hit["metadata"].get("content_type")
        return self.config.table_threshold if ct == "table" else self.config.text_threshold


def load_default_retriever(index_dir=None, config=None, reranker=None):
    """
    Convenience for scripts/CLI: load the corpus index and the SAME embedding
    model ingestion uses (via IngestionPipeline, the single owner of the model
    name), and wire up a Retriever. Imported lazily so importing retrieval.py
    never drags in torch unless a model is actually needed.

    reranker: pass a reranker.Reranker to get the Phase-2 rerank path (api.py
    does); None keeps Phase-1 hybrid — the CLI and deterministic tests stay
    reranker-free so they don't need the cross-encoder downloaded.

    WHICH BACKEND: config.VECTOR_BACKEND picks between the exact 713-vector
    demo index ('flat') and the multi-department HNSW+SQLite corpus ('hnsw').
    Both return an object with the same retrieve() contract, so every caller —
    api.py, the officer tools, the eval scripts — is backend-agnostic.
    """
    from engine.ingest import IngestionPipeline

    index_dir = index_dir or engine_config.INDEX_DIR
    pipeline = IngestionPipeline()

    if engine_config.VECTOR_BACKEND == "hnsw":
        from engine.vector_store import HnswStore
        store = HnswStore.load(index_dir)
        db_path = os.path.join(index_dir, "corpus.db")
        return CorpusRetriever(store, pipeline.model, config,
                               reranker=reranker, db_path=db_path)

    store = FaissStore.load(index_dir)
    return Retriever(store, pipeline.model, config, reranker=reranker)


if __name__ == "__main__":
    import sys

    # Explore mode, mirroring test_index.py:
    #   python retrieval.py "how does self-attention work"
    if len(sys.argv) < 2:
        print('usage: python retrieval.py "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    retriever = load_default_retriever()
    result = retriever.retrieve(question)

    flag = "  [LOW CONFIDENCE — floor fired, nothing cleared a threshold]" \
        if result["low_confidence"] else ""
    print(f'\nQUERY: "{question}"{flag}')
    print("-" * 72)
    for rank, hit in enumerate(result["chunks"], start=1):
        m = hit["metadata"]
        loc = f"p{m['page_start']}" if m["page_start"] == m["page_end"] \
            else f"p{m['page_start']}-{m['page_end']}"
        preview = " ".join(hit["text"].split())[:160]
        label = m.get("paper_id") or m.get("source_file", "?")
        print(f"#{rank}  cos={hit['score']:.3f}  [{m['content_type']:5s}]  "
              f"{label} ({loc})  {m.get('title', '')[:40]}")
        print(f"     {preview}...")
    if not result["chunks"]:
        print("(no chunks — index empty?)")
