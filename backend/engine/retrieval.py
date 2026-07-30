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

from dataclasses import dataclass

from engine import config, hybrid
from engine.vector_store import FaissStore


@dataclass
class RetrievalConfig:
    query_prefix: str = config.QUERY_PREFIX   # "" for bge-m3
    # ⚠ PLACEHOLDER cutoffs for bge-m3 — not yet calibrated on a GR gold set.
    # bge-m3 dense cosine for relevant pairs typically sits ~0.5-0.7 and for
    # clearly-irrelevant pairs ~0.2-0.4, i.e. lower and more spread out than
    # bge-small-en's upward-compressed scores — so the old 0.69/0.66 would
    # over-filter here. Start low (recall) and tighten with measured data.
    text_threshold: float = 0.50
    table_threshold: float = 0.45
    # Dense CANDIDATE generation is widened (vs. the old 8/4 final caps) so RRF
    # has room to promote a chunk that dense ranked just outside the old cap but
    # BM25 ranked high — that recall gain is the point of Phase-1 hybrid.
    candidate_k_text: int = 20
    candidate_k_table: int = 10
    candidate_k_sparse: int = 20   # BM25 keyword pool
    rrf_k: int = 60                # RRF constant (see hybrid.rrf_fuse)
    max_final_k: int = 12          # total chunks kept after fusion+threshold (was 8+4)
    floor_k: int = 2
    # Phase-2 reranking (used only when a Reranker is wired in):
    rerank_pool: int = 15          # fused candidates handed to the cross-encoder
    # ⚠ PLACEHOLDER for bge-reranker-v2-m3 — its logit scale is NOT the old
    # ms-marco model's, so the old -6.0 is meaningless here. bge-reranker-v2-m3
    # tends to put clearly-relevant pairs above ~0 and irrelevant ones below;
    # 0.0 is a reasonable recall-leaning start. Recalibrate with
    # rerank_analysis.py on a GR gold set. One threshold for all modalities: the
    # cross-encoder reads content, not templated form, so text and table chunks
    # score on the same relevance scale.
    rerank_threshold: float = 0.0


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


def load_default_retriever(index_dir="index", config=None, reranker=None):
    """
    Convenience for scripts/CLI: load the corpus index and the SAME embedding
    model ingestion uses (via IngestionPipeline, the single owner of the model
    name), and wire up a Retriever. Imported lazily so importing retrieval.py
    never drags in torch unless a model is actually needed.

    reranker: pass a reranker.Reranker to get the Phase-2 rerank path (api.py
    does); None keeps Phase-1 hybrid — the CLI and deterministic tests stay
    reranker-free so they don't need the cross-encoder downloaded.
    """
    from engine.ingest import IngestionPipeline
    store = FaissStore.load(index_dir)
    pipeline = IngestionPipeline()
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
